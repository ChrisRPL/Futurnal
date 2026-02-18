"""Persistent job queue for ingestion orchestrator."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional

from .models import IngestionJob, JobPriority, JobType
from .exceptions import InvalidStateTransitionError, StateTransitionRaceError, JobNotFoundError

if TYPE_CHECKING:
    from .state_machine import StateMachineValidator


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    QUARANTINED = "quarantined"


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    priority INTEGER NOT NULL,
    scheduled_for TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class JobQueue:
    """SQLite-backed persistent queue for ingestion jobs.

    The queue maintains durable state for the orchestrator workers. Each method
    acquires the same connection-level lock to guarantee thread safety, and all
    writes happen within SQLite transactions to preserve crash recovery. Caller
    code should rely on the typed helpers instead of issuing raw SQL so audit
    metadata and retry counters remain consistent.
    """

    def __init__(self, path: Path, *, validator: Optional["StateMachineValidator"] = None) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(SCHEMA)
        self._lock = threading.Lock()

        # State machine validator (lazy import to avoid circular dependencies)
        if validator is None:
            from .state_machine import StateMachineValidator
            validator = StateMachineValidator()
        self._validator = validator

    def enqueue(self, job: IngestionJob) -> None:
        payload_json = json.dumps(job.payload)
        scheduled_for = job.scheduled_for.isoformat() if job.scheduled_for else None
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO jobs(job_id, job_type, payload, priority, scheduled_for, status, attempts, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', 0, COALESCE((SELECT created_at FROM jobs WHERE job_id = ?), ?), ?)
                    """,
                    (
                        job.job_id,
                        job.job_type.value,
                        payload_json,
                        job.priority.value,
                        scheduled_for,
                        job.job_id,
                        now,
                        now,
                    ),
                )

    def fetch_pending(self, limit: int = 10) -> Iterator[IngestionJob]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT job_id, job_type, payload, priority, scheduled_for
                FROM jobs
                WHERE status = 'pending'
                  AND (scheduled_for IS NULL OR scheduled_for <= ?)
                ORDER BY priority DESC, COALESCE(scheduled_for, datetime('now'))
                LIMIT ?
                """,
                (datetime.utcnow().isoformat(), limit),
            )
            rows = cur.fetchall()
        for job_id, job_type, payload, priority, scheduled_for in rows:
            yield IngestionJob(
                job_id=job_id,
                job_type=JobType(job_type),
                payload=json.loads(payload),
                priority=JobPriority(priority),
                scheduled_for=datetime.fromisoformat(scheduled_for) if scheduled_for else None,
            )

    def _get_status(self, job_id: str) -> JobStatus:
        """Get current status of a job.

        Args:
            job_id: Job identifier

        Returns:
            Current JobStatus of the job

        Raises:
            JobNotFoundError: If job doesn't exist
        """
        cur = self._conn.cursor()
        cur.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
        row = cur.fetchone()
        if not row:
            raise JobNotFoundError(f"Job {job_id} not found")
        return JobStatus(row[0])

    def mark_running(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as running.

        Args:
            job_id: Job identifier
            idempotent: If True, allow same-state transitions (default: True)

        Raises:
            InvalidStateTransitionError: If transition is invalid and not idempotent
            StateTransitionRaceError: If job state changed during transition
            JobNotFoundError: If job doesn't exist
        """
        with self._lock:
            # Get current status
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.RUNNING,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping invalid transition (idempotent mode)",
                    extra={"job_id": job_id},
                )
                return

            # Already running - idempotent
            if current_status == JobStatus.RUNNING and idempotent:
                return

            # Execute state transition atomically
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'running',
                        attempts = attempts + 1,
                        updated_at = ?
                    WHERE job_id = ? AND status = ?
                    """,
                    (datetime.utcnow().isoformat(), job_id, current_status.value),
                )

                # Verify transition succeeded
                if self._conn.total_changes == 0:
                    raise StateTransitionRaceError(
                        f"Job {job_id} changed state during transition"
                    )

    def mark_completed(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as completed.

        Args:
            job_id: Job identifier
            idempotent: If True, allow same-state transitions (default: True)

        Raises:
            InvalidStateTransitionError: If transition is invalid and not idempotent
            StateTransitionRaceError: If job state changed during transition
            JobNotFoundError: If job doesn't exist
        """
        with self._lock:
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.SUCCEEDED,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping invalid transition (idempotent mode)",
                    extra={"job_id": job_id},
                )
                return

            # Already succeeded - idempotent
            if current_status == JobStatus.SUCCEEDED and idempotent:
                return

            # Only succeed from RUNNING state
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'succeeded',
                        updated_at = ?
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )

                if self._conn.total_changes == 0 and not idempotent:
                    raise StateTransitionRaceError(
                        f"Job {job_id} not in RUNNING state"
                    )

    def mark_failed(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as failed.

        Args:
            job_id: Job identifier
            idempotent: If True, allow same-state transitions (default: True)

        Raises:
            InvalidStateTransitionError: If transition is invalid and not idempotent
            JobNotFoundError: If job doesn't exist
        """
        with self._lock:
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.FAILED,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping invalid transition (idempotent mode)",
                    extra={"job_id": job_id},
                )
                return

            # Already failed - idempotent
            if current_status == JobStatus.FAILED and idempotent:
                return

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        updated_at = ?
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )

    def mark_quarantined(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as quarantined.

        Args:
            job_id: Job identifier
            idempotent: If True, allow same-state transitions (default: True)

        Raises:
            InvalidStateTransitionError: If transition is invalid and not idempotent
            JobNotFoundError: If job doesn't exist
        """
        with self._lock:
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.QUARANTINED,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping invalid transition (idempotent mode)",
                    extra={"job_id": job_id},
                )
                return

            # Already quarantined - idempotent
            if current_status == JobStatus.QUARANTINED and idempotent:
                return

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'quarantined',
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )

    def reschedule(self, job_id: str, retry_delay_seconds: int) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        scheduled_for = DATETIME(?, '+%d seconds'),
                        updated_at = ?
                    WHERE job_id = ?
                    """
                    % retry_delay_seconds,
                    (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), job_id),
                )

    def pending_count(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'pending'"
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def running_count(self) -> int:
        """Get count of currently running jobs."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'running'"
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def completed_count(self, since: Optional[datetime] = None) -> int:
        """Get count of completed jobs, optionally since a given time.

        Args:
            since: Optional datetime to filter jobs after this time

        Returns:
            Count of completed jobs
        """
        with self._lock:
            cur = self._conn.cursor()
            if since:
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'succeeded' AND updated_at >= ?",
                    (since.isoformat(),),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'succeeded'"
                )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def failed_count(self, since: Optional[datetime] = None) -> int:
        """Get count of failed jobs, optionally since a given time.

        Args:
            since: Optional datetime to filter jobs after this time

        Returns:
            Count of failed jobs
        """
        with self._lock:
            cur = self._conn.cursor()
            if since:
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'failed' AND updated_at >= ?",
                    (since.isoformat(),),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'failed'"
                )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get single job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job dict with all fields, or None if not found
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT job_id, job_type, payload, priority, scheduled_for, status, attempts, created_at, updated_at
                FROM jobs WHERE job_id = ?
                """,
                (job_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        (
            job_id,
            job_type,
            payload,
            priority,
            scheduled_for,
            job_status,
            attempts,
            created_at,
            updated_at,
        ) = row

        parsed_payload = json.loads(payload)
        try:
            priority_enum = JobPriority(priority)
            priority_value = priority_enum.name.lower()
        except ValueError:
            priority_value = str(priority)

        return {
            "job_id": job_id,
            "job_type": job_type,
            "payload": parsed_payload,
            "priority": priority_value,
            "scheduled_for": scheduled_for,
            "status": job_status,
            "attempts": attempts,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def cancel_job(self, job_id: str) -> None:
        """Cancel a pending or running job.

        Args:
            job_id: Job identifier

        Raises:
            ValueError: If job is not in pending or running status
        """
        with self._lock:
            # Check current status
            cur = self._conn.cursor()
            cur.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()

            if not row:
                raise ValueError(f"Job {job_id} not found")

            current_status = row[0]
            if current_status not in ("pending", "running"):
                raise ValueError(
                    f"Cannot cancel job with status '{current_status}'. "
                    "Only pending or running jobs can be cancelled."
                )

            # Update to cancelled status
            with self._conn:
                self._conn.execute(
                    "UPDATE jobs SET status = 'failed', updated_at = ? WHERE job_id = ?",
                    (datetime.utcnow().isoformat(), job_id),
                )

    def snapshot(
        self,
        *,
        status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        query = (
            "SELECT job_id, job_type, payload, priority, scheduled_for, status, attempts, created_at, updated_at"
            " FROM jobs"
        )
        criteria = []
        params: List[object] = []
        if status is not None:
            criteria.append("status = ?")
            params.append(status.value)
        if criteria:
            query += " WHERE " + " AND ".join(criteria)
        query += " ORDER BY datetime(updated_at) DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

        entries: List[dict] = []
        for (
            job_id,
            job_type,
            payload,
            priority,
            scheduled_for,
            job_status,
            attempts,
            created_at,
            updated_at,
        ) in rows:
            parsed_payload = json.loads(payload)
            try:
                priority_enum = JobPriority(priority)
                priority_value = priority_enum.name.lower()
            except ValueError:
                priority_value = str(priority)
            entries.append(
                {
                    "job_id": job_id,
                    "job_type": job_type,
                    "payload": parsed_payload,
                    "priority": priority_value,
                    "scheduled_for": scheduled_for,
                    "status": job_status,
                    "attempts": attempts,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        return entries


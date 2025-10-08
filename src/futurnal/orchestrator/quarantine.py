"""Quarantine system for persistently failing ingestion jobs.

Provides graceful degradation by isolating jobs that exceed retry limits,
classifying failure reasons, enabling manual recovery, and giving operators
visibility into failure patterns.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..privacy.redaction import RedactionPolicy, redact_path
from .models import IngestionJob


class QuarantineReason(str, Enum):
    """Classification of why a job was quarantined."""

    PARSE_ERROR = "parse_error"  # Unstructured.io parsing failed
    PERMISSION_DENIED = "permission_denied"  # File/folder access denied
    RESOURCE_EXHAUSTED = "resource_exhausted"  # Out of memory/disk
    CONNECTOR_ERROR = "connector_error"  # Connector-specific failure
    TIMEOUT = "timeout"  # Job exceeded time limit
    INVALID_STATE = "invalid_state"  # State store corruption
    DEPENDENCY_FAILURE = "dependency_failure"  # Neo4j/ChromaDB unavailable
    UNKNOWN = "unknown"  # Uncategorized failure


@dataclass
class QuarantinedJob:
    """Represents a job in quarantine with recovery metadata."""

    job_id: str
    job_type: str
    original_payload: Dict[str, Any]
    reason: QuarantineReason
    error_message: str
    error_traceback: Optional[str] = None
    quarantined_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    retry_success_count: int = 0
    retry_failure_count: int = 0
    last_retry_at: Optional[datetime] = None
    can_retry: bool = True
    operator_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def retry_success_rate(self) -> float:
        """Calculate success rate of manual retries."""
        total_retries = self.retry_success_count + self.retry_failure_count
        if total_retries == 0:
            return 0.0
        return self.retry_success_count / total_retries


QUARANTINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS quarantined_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    original_payload TEXT NOT NULL,
    reason TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_traceback TEXT,
    quarantined_at TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_success_count INTEGER NOT NULL DEFAULT 0,
    retry_failure_count INTEGER NOT NULL DEFAULT 0,
    last_retry_at TEXT,
    can_retry INTEGER NOT NULL DEFAULT 1,
    operator_notes TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quarantine_reason ON quarantined_jobs(reason);
CREATE INDEX IF NOT EXISTS idx_quarantine_timestamp ON quarantined_jobs(quarantined_at);
"""


class QuarantineStore:
    """SQLite-backed persistent storage for quarantined jobs.

    The store maintains durable state for failed jobs that exceed retry limits.
    Each method acquires the same connection-level lock to guarantee thread safety,
    and all writes happen within SQLite transactions to preserve crash recovery.
    """

    def __init__(self, path: Path) -> None:
        """Initialize quarantine store with SQLite database.

        Args:
            path: Path to SQLite database file
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(QUARANTINE_SCHEMA)
        self._lock = threading.Lock()

    def quarantine(
        self,
        *,
        job: IngestionJob,
        reason: QuarantineReason,
        error_message: str,
        error_traceback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuarantinedJob:
        """Move a failed job into quarantine.

        Args:
            job: The failed ingestion job
            reason: Classification of failure reason
            error_message: Exception message (will be redacted)
            error_traceback: Optional full stack trace
            metadata: Additional context for debugging

        Returns:
            QuarantinedJob with complete metadata
        """
        # Redact sensitive information from error message
        redacted_error = self._redact_error_message(error_message)

        payload_json = json.dumps(job.payload)
        metadata_json = json.dumps(metadata or {})
        now = datetime.utcnow().isoformat()

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO quarantined_jobs(
                        job_id, job_type, original_payload, reason,
                        error_message, error_traceback, quarantined_at,
                        retry_count, retry_success_count, retry_failure_count,
                        last_retry_at, can_retry, operator_notes, metadata,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, NULL, 1, NULL, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.job_type.value,
                        payload_json,
                        reason.value,
                        redacted_error,
                        error_traceback,
                        now,
                        metadata_json,
                        now,
                        now,
                    ),
                )

        return QuarantinedJob(
            job_id=job.job_id,
            job_type=job.job_type.value,
            original_payload=job.payload,
            reason=reason,
            error_message=redacted_error,
            error_traceback=error_traceback,
            quarantined_at=datetime.fromisoformat(now),
            retry_count=0,
            retry_success_count=0,
            retry_failure_count=0,
            can_retry=True,
            metadata=metadata or {},
        )

    def list(
        self,
        *,
        reason: Optional[QuarantineReason] = None,
        limit: Optional[int] = None,
    ) -> List[QuarantinedJob]:
        """List quarantined jobs with optional filtering.

        Args:
            reason: Filter by specific quarantine reason
            limit: Maximum number of results

        Returns:
            List of quarantined jobs, ordered by quarantine time (newest first)
        """
        query = """
            SELECT job_id, job_type, original_payload, reason,
                   error_message, error_traceback, quarantined_at,
                   retry_count, retry_success_count, retry_failure_count,
                   last_retry_at, can_retry, operator_notes, metadata
            FROM quarantined_jobs
        """
        params: List[Any] = []

        if reason is not None:
            query += " WHERE reason = ?"
            params.append(reason.value)

        query += " ORDER BY quarantined_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

        jobs = []
        for row in rows:
            jobs.append(self._row_to_job(row))
        return jobs

    def get(self, job_id: str) -> Optional[QuarantinedJob]:
        """Retrieve specific quarantined job by ID.

        Args:
            job_id: Job identifier

        Returns:
            QuarantinedJob if found, None otherwise
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT job_id, job_type, original_payload, reason,
                       error_message, error_traceback, quarantined_at,
                       retry_count, retry_success_count, retry_failure_count,
                       last_retry_at, can_retry, operator_notes, metadata
                FROM quarantined_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            )
            row = cur.fetchone()

        if row is None:
            return None
        return self._row_to_job(row)

    def mark_retry_attempted(
        self,
        job_id: str,
        *,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a manual retry attempt.

        Args:
            job_id: Job identifier
            success: Whether retry succeeded
            error_message: Error message if retry failed
        """
        if success:
            # Job succeeded - remove from quarantine entirely
            self.remove(job_id)
        else:
            # Job failed again - update counters
            now = datetime.utcnow().isoformat()
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE quarantined_jobs
                        SET retry_count = retry_count + 1,
                            retry_failure_count = retry_failure_count + 1,
                            last_retry_at = ?,
                            updated_at = ?
                        WHERE job_id = ?
                        """,
                        (now, now, job_id),
                    )

    def remove(self, job_id: str) -> None:
        """Remove job from quarantine (after successful recovery or purge).

        Args:
            job_id: Job identifier
        """
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM quarantined_jobs WHERE job_id = ?",
                    (job_id,),
                )

    def purge_old(self, days: int = 30) -> int:
        """Remove quarantined jobs older than specified days.

        Args:
            days: Retention period in days

        Returns:
            Number of jobs removed
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM quarantined_jobs WHERE quarantined_at < ?",
                    (cutoff,),
                )
                return cur.rowcount

    def statistics(self) -> Dict[str, Any]:
        """Compute quarantine statistics by reason.

        Returns:
            Dictionary with aggregate statistics including:
            - total_quarantined: Total number of jobs in quarantine
            - by_reason: Breakdown of counts by failure reason
            - oldest_job_age_days: Age of oldest quarantined job
            - recent_quarantines_24h: Jobs quarantined in last 24 hours
            - retry_success_rate: Overall success rate of manual retries
        """
        with self._lock:
            cur = self._conn.cursor()

            # Total count
            cur.execute("SELECT COUNT(*) FROM quarantined_jobs")
            total = cur.fetchone()[0]

            # Count by reason
            cur.execute(
                """
                SELECT reason, COUNT(*) as count
                FROM quarantined_jobs
                GROUP BY reason
                """
            )
            by_reason = {row[0]: row[1] for row in cur.fetchall()}

            # Oldest job age
            cur.execute(
                "SELECT MIN(quarantined_at) FROM quarantined_jobs"
            )
            oldest_timestamp = cur.fetchone()[0]
            if oldest_timestamp:
                oldest = datetime.fromisoformat(oldest_timestamp)
                oldest_age_days = (datetime.utcnow() - oldest).days
            else:
                oldest_age_days = 0

            # Recent quarantines (24h)
            cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            cur.execute(
                "SELECT COUNT(*) FROM quarantined_jobs WHERE quarantined_at > ?",
                (cutoff_24h,),
            )
            recent_24h = cur.fetchone()[0]

            # Retry success rate
            cur.execute(
                """
                SELECT
                    SUM(retry_success_count),
                    SUM(retry_failure_count)
                FROM quarantined_jobs
                """
            )
            success_count, failure_count = cur.fetchone()
            success_count = success_count or 0
            failure_count = failure_count or 0
            total_retries = success_count + failure_count
            retry_success_rate = (
                success_count / total_retries if total_retries > 0 else 0.0
            )

        return {
            "total_quarantined": total,
            "by_reason": by_reason,
            "oldest_job_age_days": oldest_age_days,
            "recent_quarantines_24h": recent_24h,
            "retry_success_rate": retry_success_rate,
        }

    def _row_to_job(self, row: tuple) -> QuarantinedJob:
        """Convert database row to QuarantinedJob instance."""
        (
            job_id,
            job_type,
            original_payload,
            reason,
            error_message,
            error_traceback,
            quarantined_at,
            retry_count,
            retry_success_count,
            retry_failure_count,
            last_retry_at,
            can_retry,
            operator_notes,
            metadata,
        ) = row

        return QuarantinedJob(
            job_id=job_id,
            job_type=job_type,
            original_payload=json.loads(original_payload),
            reason=QuarantineReason(reason),
            error_message=error_message,
            error_traceback=error_traceback,
            quarantined_at=datetime.fromisoformat(quarantined_at),
            retry_count=retry_count,
            retry_success_count=retry_success_count,
            retry_failure_count=retry_failure_count,
            last_retry_at=datetime.fromisoformat(last_retry_at) if last_retry_at else None,
            can_retry=bool(can_retry),
            operator_notes=operator_notes,
            metadata=json.loads(metadata) if metadata else {},
        )

    def _redact_error_message(self, error_message: str) -> str:
        """Redact sensitive paths from error messages.

        Args:
            error_message: Raw error message

        Returns:
            Redacted error message with paths anonymized
        """
        # Simple path detection and redaction
        # Look for common path patterns and redact them
        import re

        policy = RedactionPolicy()

        # Match absolute paths (Unix and Windows)
        path_patterns = [
            r'/[^\s]+',  # Unix absolute paths
            r'[A-Z]:\\[^\s]+',  # Windows absolute paths
        ]

        redacted = error_message
        for pattern in path_patterns:
            for match in re.finditer(pattern, error_message):
                path_str = match.group(0)
                try:
                    redacted_path = redact_path(path_str, policy=policy)
                    redacted = redacted.replace(path_str, redacted_path.redacted)
                except Exception:
                    # If redaction fails, keep original
                    pass

        return redacted


def classify_failure(
    error_message: str,
    exception_type: Optional[Type[Exception]] = None,
) -> QuarantineReason:
    """Classify failure reason from error message and exception type.

    Uses a two-tier approach:
    1. First checks exception type if available
    2. Falls back to pattern matching on error message

    Args:
        error_message: The exception message
        exception_type: Optional exception class

    Returns:
        QuarantineReason classification
    """
    # Primary classification: exception type
    if exception_type is not None:
        if issubclass(exception_type, PermissionError):
            return QuarantineReason.PERMISSION_DENIED
        elif issubclass(exception_type, MemoryError):
            return QuarantineReason.RESOURCE_EXHAUSTED
        elif issubclass(exception_type, TimeoutError):
            return QuarantineReason.TIMEOUT

    # Secondary classification: message pattern matching
    error_lower = error_message.lower()

    if "permission denied" in error_lower or "access denied" in error_lower:
        return QuarantineReason.PERMISSION_DENIED
    elif "parse" in error_lower or "parsing" in error_lower:
        return QuarantineReason.PARSE_ERROR
    elif "memory" in error_lower or "disk" in error_lower:
        return QuarantineReason.RESOURCE_EXHAUSTED
    elif "timeout" in error_lower or "timed out" in error_lower:
        return QuarantineReason.TIMEOUT
    elif "neo4j" in error_lower or "chroma" in error_lower:
        return QuarantineReason.DEPENDENCY_FAILURE
    elif "connector" in error_lower:
        return QuarantineReason.CONNECTOR_ERROR
    elif "state" in error_lower or "corrupt" in error_lower:
        return QuarantineReason.INVALID_STATE
    else:
        return QuarantineReason.UNKNOWN

"""Crash recovery and durability management for job queue.

This module implements SQLite WAL-based crash recovery with automatic
rehydration of interrupted jobs, integrity verification, and performance
monitoring for the orchestrator's persistent job queue.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..privacy.audit import AuditLogger
    from .queue import JobQueue

logger = logging.getLogger(__name__)


@dataclass
class CrashRecoveryReport:
    """Report of crash recovery process.

    Attributes:
        recovered_at: Timestamp when recovery completed
        jobs_recovered: Total number of jobs found in queue
        jobs_pending: Number of jobs in pending state after recovery
        jobs_running_before_crash: Number of jobs that were running before crash
        jobs_reset_to_pending: Number of jobs reset from running to pending
        wal_size_before_recovery_bytes: Size of WAL file before recovery
        recovery_duration_seconds: Time taken to complete recovery
        errors: List of error messages encountered during recovery
    """

    recovered_at: datetime
    jobs_recovered: int
    jobs_pending: int
    jobs_running_before_crash: int
    jobs_reset_to_pending: int
    wal_size_before_recovery_bytes: int
    recovery_duration_seconds: float
    errors: List[str] = field(default_factory=list)

    def was_successful(self) -> bool:
        """Check if recovery completed without errors.

        Returns:
            True if no errors were encountered during recovery
        """
        return len(self.errors) == 0


class RecoveryStateTracker:
    """Tracks recovery state for resuming interrupted jobs.

    Uses a marker file in the workspace to detect if the orchestrator
    crashed and needs to perform recovery on next startup.
    """

    def __init__(self, workspace_dir: Path) -> None:
        """Initialize recovery state tracker.

        Args:
            workspace_dir: Workspace directory for marker file
        """
        self._recovery_marker = workspace_dir / ".orchestrator_recovery"

    def mark_crash(self) -> None:
        """Mark that orchestrator is running (for crash detection).

        This marker is created when the orchestrator starts and cleared
        on graceful shutdown. If it exists on startup, it indicates
        the previous run crashed.
        """
        self._recovery_marker.write_text(
            json.dumps(
                {
                    "crashed_at": datetime.utcnow().isoformat(),
                    "pid": os.getpid(),
                }
            )
        )
        logger.debug(
            "Created recovery marker",
            extra={"marker_path": str(self._recovery_marker)},
        )

    def is_recovering_from_crash(self) -> bool:
        """Check if orchestrator is recovering from crash.

        Returns:
            True if recovery marker exists, indicating previous crash
        """
        return self._recovery_marker.exists()

    def get_crash_info(self) -> Optional[Dict[str, Any]]:
        """Get crash metadata from marker file.

        Returns:
            Dictionary with crash timestamp and PID, or None if no marker
        """
        if not self._recovery_marker.exists():
            return None

        try:
            return json.loads(self._recovery_marker.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read crash info from marker",
                extra={"error": str(exc)},
            )
            return None

    def clear_recovery_marker(self) -> None:
        """Clear recovery marker after successful recovery or shutdown.

        Called after successful recovery or on graceful shutdown to
        indicate the orchestrator is in a clean state.
        """
        if self._recovery_marker.exists():
            self._recovery_marker.unlink()
            logger.debug(
                "Cleared recovery marker",
                extra={"marker_path": str(self._recovery_marker)},
            )


class CrashRecoveryManager:
    """Manages crash recovery process for orchestrator.

    Orchestrates the full recovery workflow including:
    - WAL file analysis
    - Job state inspection
    - Interrupted job reset (RUNNING → PENDING)
    - WAL checkpoint
    - Database integrity verification
    - Recovery metrics and audit logging
    """

    def __init__(
        self,
        *,
        job_queue: JobQueue,
        workspace_dir: Path,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        """Initialize crash recovery manager.

        Args:
            job_queue: The persistent job queue to recover
            workspace_dir: Workspace directory for recovery marker
            audit_logger: Optional audit logger for recovery events
        """
        self._queue = job_queue
        self._workspace = workspace_dir
        self._audit = audit_logger
        self._recovery_tracker = RecoveryStateTracker(workspace_dir)

    def recover_from_crash(self) -> CrashRecoveryReport:
        """Execute crash recovery procedure.

        Performs full recovery workflow:
        1. Analyze WAL file size
        2. Count jobs by status
        3. Reset RUNNING jobs to PENDING
        4. Checkpoint WAL
        5. Verify database integrity
        6. Generate recovery report
        7. Clear recovery marker
        8. Record audit event

        Returns:
            CrashRecoveryReport with recovery metrics and any errors
        """
        start_time = time.perf_counter()

        logger.info("Starting crash recovery")

        # Get crash info
        crash_info = self._recovery_tracker.get_crash_info()
        crashed_at = crash_info.get("crashed_at") if crash_info else None

        # Check WAL size
        wal_path = Path(str(self._queue._path) + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0

        logger.info(
            "WAL file status",
            extra={"wal_size_bytes": wal_size},
        )

        # Count jobs in various states
        all_jobs = self._queue.snapshot()
        jobs_pending = len([j for j in all_jobs if j["status"] == "pending"])
        jobs_running = len([j for j in all_jobs if j["status"] == "running"])

        logger.info(
            "Job state before recovery",
            extra={
                "jobs_pending": jobs_pending,
                "jobs_running": jobs_running,
                "jobs_total": len(all_jobs),
            },
        )

        # Reset RUNNING jobs to PENDING (they were interrupted)
        jobs_reset = self._reset_interrupted_jobs()

        # Re-count after reset
        jobs_recovered = len(self._queue.snapshot())

        # Checkpoint WAL to ensure all changes persisted
        self._checkpoint_wal()

        # Verify database integrity
        integrity_errors = self._verify_database_integrity()

        duration = time.perf_counter() - start_time

        report = CrashRecoveryReport(
            recovered_at=datetime.utcnow(),
            jobs_recovered=jobs_recovered,
            jobs_pending=jobs_pending + jobs_reset,
            jobs_running_before_crash=jobs_running,
            jobs_reset_to_pending=jobs_reset,
            wal_size_before_recovery_bytes=wal_size,
            recovery_duration_seconds=duration,
            errors=integrity_errors,
        )

        # Clear recovery marker
        self._recovery_tracker.clear_recovery_marker()

        # Audit log
        if self._audit:
            from ..privacy.audit import AuditEvent

            self._audit.record(
                AuditEvent(
                    job_id=f"recovery_{datetime.utcnow().isoformat()}",
                    source="crash_recovery",
                    action="recover_from_crash",
                    status="succeeded" if report.was_successful() else "failed",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "jobs_recovered": jobs_recovered,
                        "jobs_reset": jobs_reset,
                        "duration_seconds": duration,
                        "crashed_at": crashed_at,
                        "wal_size_bytes": wal_size,
                    },
                )
            )

        logger.info(
            "Crash recovery completed",
            extra={
                "jobs_recovered": jobs_recovered,
                "jobs_reset": jobs_reset,
                "duration_seconds": duration,
                "success": report.was_successful(),
            },
        )

        return report

    def _reset_interrupted_jobs(self) -> int:
        """Reset RUNNING jobs to PENDING for retry.

        Jobs that were in RUNNING state when the orchestrator crashed
        are incomplete and need to be retried. This method transitions
        them back to PENDING state with no delay.

        Returns:
            Number of jobs reset
        """
        from .queue import JobStatus

        running_jobs = self._queue.snapshot(status=JobStatus.RUNNING)

        for job in running_jobs:
            logger.info(
                "Resetting interrupted job",
                extra={"job_id": job["job_id"]},
            )
            # Reset to pending with no delay
            # Use mark_failed first to transition from RUNNING
            self._queue.mark_failed(job["job_id"])
            # Then reschedule immediately
            self._queue.reschedule(job["job_id"], retry_delay_seconds=0)

        if running_jobs:
            logger.info(
                "Reset interrupted jobs",
                extra={"count": len(running_jobs)},
            )

        return len(running_jobs)

    def _checkpoint_wal(self) -> None:
        """Force WAL checkpoint to persist all changes.

        Executes PRAGMA wal_checkpoint(FULL) to ensure all changes
        in the WAL are merged into the main database file for
        maximum durability.
        """
        try:
            self._queue._conn.execute("PRAGMA wal_checkpoint(FULL)")
            logger.info("WAL checkpoint completed")
        except Exception as exc:
            logger.error("WAL checkpoint failed", exc_info=exc)

    def _verify_database_integrity(self) -> List[str]:
        """Verify database integrity after recovery.

        Runs comprehensive integrity checks:
        - SQLite integrity_check
        - Foreign key violations
        - Application-level validation via integrity module

        Returns:
            List of error messages (empty if all checks pass)
        """
        errors = []

        try:
            # SQLite integrity check
            result = self._queue._conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] != "ok":
                errors.append(f"Integrity check failed: {result[0]}")

            # Foreign key check
            result = self._queue._conn.execute("PRAGMA foreign_key_check").fetchall()
            if result:
                errors.append(f"Foreign key violations: {result}")

            # Application-level integrity validation
            from .integrity import validate_database_integrity

            app_errors = validate_database_integrity(self._queue)
            if app_errors:
                errors.extend(app_errors)

        except Exception as exc:
            errors.append(f"Integrity verification failed: {exc}")

        if errors:
            logger.error(
                "Database integrity issues detected",
                extra={"error_count": len(errors), "errors": errors},
            )
        else:
            logger.info("Database integrity verification passed")

        return errors

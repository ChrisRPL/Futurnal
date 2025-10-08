"""Deadlock detection and recovery for stalled job processing.

This module detects jobs that are stuck in the RUNNING state beyond
a configurable timeout and provides mechanisms to recover them by
resetting to PENDING for retry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .queue import JobQueue, JobStatus


logger = logging.getLogger(__name__)


class DeadlockDetector:
    """Detects and resolves job processing deadlocks.

    Jobs can become stuck in the RUNNING state due to crashes, network
    failures, or other unexpected conditions. This detector identifies
    such jobs based on a configurable timeout and provides recovery
    mechanisms.
    """

    def __init__(self, queue: JobQueue, timeout_seconds: int = 600):
        """Initialize deadlock detector.

        Args:
            queue: The job queue to monitor
            timeout_seconds: Time in seconds before a RUNNING job is
                considered stalled (default: 600 = 10 minutes)
        """
        self._queue = queue
        self._timeout_seconds = timeout_seconds

    def detect_stalled_jobs(self) -> List[str]:
        """Detect jobs stuck in RUNNING state.

        Scans the job queue for jobs that have been in RUNNING state
        longer than the configured timeout period.

        Returns:
            List of job IDs that appear to be stalled
        """
        from .queue import JobStatus

        stalled = []
        running_jobs = self._queue.snapshot(status=JobStatus.RUNNING)

        for job in running_jobs:
            try:
                updated_at = datetime.fromisoformat(job["updated_at"])
                age = (datetime.utcnow() - updated_at).total_seconds()

                if age > self._timeout_seconds:
                    stalled.append(job["job_id"])
                    logger.warning(
                        "Detected stalled job",
                        extra={
                            "job_id": job["job_id"],
                            "age_seconds": age,
                            "timeout_seconds": self._timeout_seconds,
                        },
                    )
            except (ValueError, KeyError) as exc:
                logger.error(
                    "Error checking job staleness",
                    extra={
                        "job_id": job.get("job_id"),
                        "error": str(exc),
                    },
                )
                continue

        return stalled

    def recover_stalled_job(self, job_id: str) -> None:
        """Recover a stalled job by resetting to PENDING.

        This method marks the job as FAILED and then reschedules it
        for retry, effectively resetting it from the stalled RUNNING
        state.

        Args:
            job_id: ID of the stalled job to recover
        """
        logger.info(
            "Recovering stalled job",
            extra={"job_id": job_id},
        )

        # Reset to PENDING for retry via FAILED state
        try:
            self._queue.mark_failed(job_id)
            self._queue.reschedule(job_id, retry_delay_seconds=60)

            logger.info(
                "Stalled job recovered and rescheduled",
                extra={
                    "job_id": job_id,
                    "retry_delay_seconds": 60,
                },
            )
        except Exception as exc:
            logger.error(
                "Failed to recover stalled job",
                extra={
                    "job_id": job_id,
                    "error": str(exc),
                },
            )
            raise

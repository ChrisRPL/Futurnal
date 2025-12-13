"""Orchestrator-specific audit event types and logging helpers.

This module defines orchestrator-specific audit event types and provides helper
functions for logging job lifecycle operations with privacy-compliant metadata.

Event Types:
- System events (startup, shutdown)
- Job events (enqueue, start, complete, fail, retry, quarantine)
- State transition events (pending->running, running->succeeded, etc.)

Integration:
- Uses existing AuditLogger infrastructure
- Ensures no sensitive data in audit logs
- Follows IMAP audit pattern for consistency
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from ..privacy.audit import AuditEvent, AuditLogger


class OrchestratorAuditEvents:
    """Audit event types for orchestrator operations.

    Constants for standardized orchestrator audit event types, ensuring
    consistent event naming across the job lifecycle.
    """

    # System events
    SYSTEM_STARTED = "orchestrator_started"
    SYSTEM_SHUTDOWN = "orchestrator_shutdown"
    SYSTEM_CRASH_RECOVERY = "orchestrator_crash_recovery"

    # Job lifecycle events
    JOB_ENQUEUED = "job_enqueued"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_RETRY_SCHEDULED = "job_retry_scheduled"
    JOB_QUARANTINED = "job_quarantined"
    JOB_CANCELLED = "job_cancelled"

    # State transition events
    STATE_TRANSITION = "state_transition"
    STATE_TRANSITION_INVALID = "state_transition_invalid"

    # Resource events
    RESOURCE_PRESSURE_DETECTED = "resource_pressure_detected"
    CONCURRENCY_ADJUSTED = "concurrency_adjusted"


def log_system_start(
    audit_logger: AuditLogger,
    *,
    workspace_dir: str,
    configured_workers: int,
    registered_sources: int,
) -> None:
    """Log orchestrator system startup event.

    Args:
        audit_logger: Audit logger instance
        workspace_dir: Workspace directory path (redacted in logs)
        configured_workers: Number of configured workers
        registered_sources: Number of registered sources
    """
    audit_logger.record(
        AuditEvent(
            job_id=f"system_start_{int(datetime.utcnow().timestamp())}",
            source="orchestrator",
            action=OrchestratorAuditEvents.SYSTEM_STARTED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "configured_workers": configured_workers,
                "registered_sources": registered_sources,
                "workspace_hash": _hash_path(workspace_dir),
            },
        )
    )


def log_system_shutdown(
    audit_logger: AuditLogger,
    *,
    jobs_completed: int,
    jobs_failed: int,
    jobs_pending: int,
    uptime_seconds: Optional[float] = None,
) -> None:
    """Log orchestrator system shutdown event.

    Args:
        audit_logger: Audit logger instance
        jobs_completed: Total completed jobs during session
        jobs_failed: Total failed jobs during session
        jobs_pending: Remaining pending jobs
        uptime_seconds: Optional uptime duration
    """
    metadata: Dict[str, object] = {
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "jobs_pending": jobs_pending,
    }
    if uptime_seconds is not None:
        metadata["uptime_seconds"] = round(uptime_seconds, 2)

    audit_logger.record(
        AuditEvent(
            job_id=f"system_shutdown_{int(datetime.utcnow().timestamp())}",
            source="orchestrator",
            action=OrchestratorAuditEvents.SYSTEM_SHUTDOWN,
            status="success",
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_crash_recovery(
    audit_logger: AuditLogger,
    *,
    jobs_reset: int,
    recovery_duration_seconds: float,
    errors: int = 0,
) -> None:
    """Log crash recovery event.

    Args:
        audit_logger: Audit logger instance
        jobs_reset: Number of jobs reset to pending
        recovery_duration_seconds: Time taken for recovery
        errors: Number of errors during recovery
    """
    audit_logger.record(
        AuditEvent(
            job_id=f"crash_recovery_{int(datetime.utcnow().timestamp())}",
            source="orchestrator",
            action=OrchestratorAuditEvents.SYSTEM_CRASH_RECOVERY,
            status="success" if errors == 0 else "partial_success",
            timestamp=datetime.utcnow(),
            metadata={
                "jobs_reset": jobs_reset,
                "recovery_duration_seconds": round(recovery_duration_seconds, 2),
                "errors": errors,
            },
        )
    )


def log_job_enqueued(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    source_name: str,
    job_type: str,
    trigger: str,
    priority: str,
) -> None:
    """Log job enqueue event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        source_name: Name of the data source
        job_type: Type of job (LOCAL_FILES, OBSIDIAN_VAULT, etc.)
        trigger: What triggered the job (schedule, manual, watcher, interval)
        priority: Job priority
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name,
            action=OrchestratorAuditEvents.JOB_ENQUEUED,
            status="pending",
            timestamp=datetime.utcnow(),
            metadata={
                "job_type": job_type,
                "trigger": trigger,
                "priority": priority,
            },
        )
    )


def log_job_started(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    source_name: str,
    job_type: str,
    attempt: int,
) -> None:
    """Log job start event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        source_name: Name of the data source
        job_type: Type of job
        attempt: Current attempt number
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name,
            action=OrchestratorAuditEvents.JOB_STARTED,
            status="running",
            timestamp=datetime.utcnow(),
            attempt=attempt,
            metadata={
                "job_type": job_type,
            },
        )
    )


def log_job_completed(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    source_name: str,
    files_processed: int,
    bytes_processed: int,
    duration_seconds: float,
) -> None:
    """Log job completion event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        source_name: Name of the data source
        files_processed: Number of files processed
        bytes_processed: Total bytes processed
        duration_seconds: Job duration in seconds
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name,
            action=OrchestratorAuditEvents.JOB_COMPLETED,
            status="succeeded",
            timestamp=datetime.utcnow(),
            metadata={
                "files_processed": files_processed,
                "bytes_processed": bytes_processed,
                "duration_seconds": round(duration_seconds, 2),
                "throughput_bytes_per_sec": (
                    round(bytes_processed / duration_seconds, 2)
                    if duration_seconds > 0
                    else 0
                ),
            },
        )
    )


def log_job_failed(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    source_name: str,
    error_type: str,
    error_message: str,
    attempt: int,
    files_processed: int = 0,
    bytes_processed: int = 0,
) -> None:
    """Log job failure event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        source_name: Name of the data source
        error_type: Classification of error (network, parse, permission, etc.)
        error_message: Error message (sanitized, no sensitive data)
        attempt: Attempt number when failure occurred
        files_processed: Files processed before failure
        bytes_processed: Bytes processed before failure
    """
    # Sanitize error message - remove any potential file paths or sensitive data
    sanitized_error = _sanitize_error_message(error_message)

    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name,
            action=OrchestratorAuditEvents.JOB_FAILED,
            status="failed",
            timestamp=datetime.utcnow(),
            attempt=attempt,
            metadata={
                "error_type": error_type,
                "error_summary": sanitized_error[:200],  # Limit length
                "files_processed": files_processed,
                "bytes_processed": bytes_processed,
            },
        )
    )


def log_retry_scheduled(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    source_name: str,
    attempt: int,
    delay_seconds: float,
    failure_type: str,
    retry_strategy: str,
) -> None:
    """Log retry scheduling event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        source_name: Name of the data source
        attempt: Attempt number being scheduled
        delay_seconds: Delay before retry
        failure_type: Type of failure causing retry
        retry_strategy: Strategy being used (exponential, linear, etc.)
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name,
            action=OrchestratorAuditEvents.JOB_RETRY_SCHEDULED,
            status="pending",
            timestamp=datetime.utcnow(),
            attempt=attempt,
            metadata={
                "delay_seconds": round(delay_seconds, 2),
                "failure_type": failure_type,
                "retry_strategy": retry_strategy,
            },
        )
    )


def log_job_quarantined(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    source_name: str,
    reason: str,
    total_attempts: int,
) -> None:
    """Log job quarantine event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        source_name: Name of the data source
        reason: Quarantine reason classification
        total_attempts: Total attempts before quarantine
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name,
            action=OrchestratorAuditEvents.JOB_QUARANTINED,
            status="quarantined",
            timestamp=datetime.utcnow(),
            metadata={
                "reason": reason,
                "total_attempts": total_attempts,
            },
        )
    )


def log_state_transition(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    from_status: str,
    to_status: str,
    source_name: Optional[str] = None,
) -> None:
    """Log job state transition event.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        from_status: Previous job status
        to_status: New job status
        source_name: Optional source name
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source=source_name or "job_queue",
            action=OrchestratorAuditEvents.STATE_TRANSITION,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "from_status": from_status,
                "to_status": to_status,
            },
        )
    )


def log_invalid_state_transition(
    audit_logger: AuditLogger,
    *,
    job_id: str,
    from_status: str,
    to_status: str,
    reason: str,
) -> None:
    """Log invalid state transition attempt.

    Args:
        audit_logger: Audit logger instance
        job_id: Job identifier
        from_status: Current job status
        to_status: Attempted target status
        reason: Why the transition is invalid
    """
    audit_logger.record(
        AuditEvent(
            job_id=job_id,
            source="job_queue",
            action=OrchestratorAuditEvents.STATE_TRANSITION_INVALID,
            status="blocked",
            timestamp=datetime.utcnow(),
            metadata={
                "from_status": from_status,
                "to_status": to_status,
                "reason": reason,
            },
        )
    )


def _hash_path(path: str) -> str:
    """Create a privacy-safe hash of a path.

    Args:
        path: File system path

    Returns:
        SHA256 hash of the path (first 16 chars)
    """
    from hashlib import sha256

    return sha256(path.encode("utf-8")).hexdigest()[:16]


def _sanitize_error_message(message: str) -> str:
    """Sanitize error message to remove sensitive data.

    Args:
        message: Raw error message

    Returns:
        Sanitized error message safe for audit logs
    """
    import re

    # Remove file paths (Unix and Windows)
    sanitized = re.sub(r"(/[a-zA-Z0-9_\-./]+)+", "[PATH]", message)
    sanitized = re.sub(r"([A-Z]:\\[a-zA-Z0-9_\-\\./]+)+", "[PATH]", sanitized)

    # Remove email addresses
    sanitized = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", sanitized)

    # Remove potential API keys / tokens (long hex strings)
    sanitized = re.sub(r"\b[a-fA-F0-9]{32,}\b", "[TOKEN]", sanitized)

    return sanitized


__all__ = [
    "OrchestratorAuditEvents",
    "log_system_start",
    "log_system_shutdown",
    "log_crash_recovery",
    "log_job_enqueued",
    "log_job_started",
    "log_job_completed",
    "log_job_failed",
    "log_retry_scheduled",
    "log_job_quarantined",
    "log_state_transition",
    "log_invalid_state_transition",
]

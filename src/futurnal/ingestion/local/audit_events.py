"""Local files-specific audit event types and logging helpers.

This module defines audit event types for local file system operations
and provides helper functions for logging with privacy-compliant redaction.

Event Types:
- Source events (registered, unregistered, scanned)
- File system events (file discovered, processed, skipped)
- State events (state loaded, state saved)
- Watch events (watcher started, stopped, event received)
- Privacy events (consent granted/revoked)

Integration:
- Uses existing AuditLogger infrastructure
- Applies path redaction using RedactionPolicy
- Ensures no file content in audit logs
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...privacy.audit import AuditLogger, AuditEvent
    from ...privacy.redaction import RedactionPolicy


class LocalFilesAuditEvents:
    """Audit event types for local file operations.

    Constants for standardized local files audit event types, ensuring
    consistent event naming across the local files connector.
    """

    # Source events
    SOURCE_REGISTERED = "local_source_registered"
    SOURCE_UNREGISTERED = "local_source_unregistered"
    SOURCE_SCANNED = "local_source_scanned"
    SOURCE_SCAN_FAILED = "local_source_scan_failed"

    # File system events
    FILE_DISCOVERED = "local_file_discovered"
    FILE_PROCESSED = "local_file_processed"
    FILE_SKIPPED = "local_file_skipped"
    FILE_PROCESSING_FAILED = "local_file_processing_failed"
    FILE_MODIFIED = "local_file_modified"
    FILE_DELETED = "local_file_deleted"

    # State events
    STATE_LOADED = "local_state_loaded"
    STATE_SAVED = "local_state_saved"
    STATE_CORRUPTED = "local_state_corrupted"

    # Watch events
    WATCHER_STARTED = "local_watcher_started"
    WATCHER_STOPPED = "local_watcher_stopped"
    WATCH_EVENT_RECEIVED = "local_watch_event_received"

    # Sync events
    SYNC_STARTED = "local_sync_started"
    SYNC_COMPLETED = "local_sync_completed"
    SYNC_FAILED = "local_sync_failed"

    # Privacy events
    CONSENT_GRANTED = "local_consent_granted"
    CONSENT_REVOKED = "local_consent_revoked"
    CONSENT_CHECK_FAILED = "local_consent_check_failed"


def _redact_path(
    path: Path,
    source_path: Optional[Path] = None,
    redaction_policy: Optional["RedactionPolicy"] = None,
) -> str:
    """Redact file path for privacy.

    Uses RedactionPolicy if available, otherwise creates relative paths.

    Args:
        path: File path to redact
        source_path: Optional source root path for relative paths
        redaction_policy: Optional redaction policy to apply

    Returns:
        Redacted path string
    """
    if redaction_policy:
        return redaction_policy.redact_path(path)

    if source_path:
        try:
            relative = path.relative_to(source_path)
            return str(relative)
        except ValueError:
            pass

    # Return just filename with parent indicator for privacy
    return f".../{path.name}"


def log_source_event(
    audit_logger: "AuditLogger",
    *,
    source_id: str,
    source_name: str,
    source_path: Path,
    action: str,
    status: str = "success",
    file_count: Optional[int] = None,
    total_size_bytes: Optional[int] = None,
    redaction_policy: Optional["RedactionPolicy"] = None,
    error: Optional[str] = None,
) -> None:
    """Log source registration/scan event.

    Args:
        audit_logger: Audit logger instance
        source_id: Source identifier
        source_name: Human-readable source name
        source_path: Path to the source directory
        action: Event action (registered, unregistered, scanned)
        status: Event status (success, failed)
        file_count: Optional count of files in source
        total_size_bytes: Optional total size of files
        redaction_policy: Optional path redaction policy
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    redacted_path = _redact_path(source_path, redaction_policy=redaction_policy)

    metadata: Dict[str, object] = {
        "source_id": source_id,
        "source_name": source_name,
        "source_path": redacted_path,
    }

    if file_count is not None:
        metadata["file_count"] = file_count

    if total_size_bytes is not None:
        metadata["total_size_bytes"] = total_size_bytes

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"local_source_{source_id}_{int(datetime.utcnow().timestamp())}",
            source="local_files_connector",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_file_event(
    audit_logger: "AuditLogger",
    *,
    source_id: str,
    file_path: Path,
    source_path: Optional[Path] = None,
    action: str,
    status: str = "success",
    file_type: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    file_hash: Optional[str] = None,
    redaction_policy: Optional["RedactionPolicy"] = None,
    error: Optional[str] = None,
) -> None:
    """Log file processing event with privacy redaction.

    Args:
        audit_logger: Audit logger instance
        source_id: Source identifier
        file_path: Path to the processed file
        source_path: Optional source root for relative paths
        action: Processing action (discovered, processed, skipped, failed)
        status: Event status
        file_type: File type/extension
        file_size_bytes: File size in bytes
        file_hash: Content hash (first 8 chars only for privacy)
        redaction_policy: Optional path redaction policy
        error: Optional error message

    Privacy Guarantee:
        - File paths redacted to relative paths
        - File content NEVER logged
        - Only first 8 chars of hash logged
    """
    from ...privacy.audit import AuditEvent

    redacted_path = _redact_path(file_path, source_path, redaction_policy)

    metadata: Dict[str, object] = {
        "source_id": source_id,
        "file_path": redacted_path,
        "file_extension": file_path.suffix.lower(),
    }

    if file_type:
        metadata["file_type"] = file_type

    if file_size_bytes is not None:
        metadata["file_size_bytes"] = file_size_bytes

    if file_hash:
        # Only log first 8 chars for privacy
        metadata["file_hash_prefix"] = file_hash[:8]

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"local_file_{source_id}_{hash(str(file_path)) % 100000}",
            source="local_file_processor",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_sync_event(
    audit_logger: "AuditLogger",
    *,
    source_id: str,
    action: str,
    status: str = "success",
    files_discovered: int = 0,
    files_processed: int = 0,
    files_skipped: int = 0,
    files_failed: int = 0,
    bytes_processed: int = 0,
    duration_seconds: Optional[float] = None,
    is_incremental: bool = False,
    error: Optional[str] = None,
) -> None:
    """Log sync event with statistics.

    Args:
        audit_logger: Audit logger instance
        source_id: Source identifier
        action: Sync action (started, completed, failed)
        status: Event status
        files_discovered: Count of discovered files
        files_processed: Count of processed files
        files_skipped: Count of skipped files
        files_failed: Count of failed files
        bytes_processed: Total bytes processed
        duration_seconds: Sync duration in seconds
        is_incremental: Whether this was an incremental sync
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "source_id": source_id,
        "files_discovered": files_discovered,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "files_failed": files_failed,
        "total_files": files_processed + files_skipped + files_failed,
        "bytes_processed": bytes_processed,
        "is_incremental": is_incremental,
    }

    if duration_seconds is not None:
        metadata["duration_seconds"] = round(duration_seconds, 2)

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"local_sync_{source_id}_{int(datetime.utcnow().timestamp())}",
            source="local_sync_engine",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_state_event(
    audit_logger: "AuditLogger",
    *,
    source_id: str,
    action: str,
    status: str = "success",
    record_count: Optional[int] = None,
    state_size_bytes: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Log state management event.

    Args:
        audit_logger: Audit logger instance
        source_id: Source identifier
        action: State action (loaded, saved, corrupted)
        status: Event status
        record_count: Number of file records in state
        state_size_bytes: State file size
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "source_id": source_id,
    }

    if record_count is not None:
        metadata["record_count"] = record_count

    if state_size_bytes is not None:
        metadata["state_size_bytes"] = state_size_bytes

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"local_state_{source_id}_{int(datetime.utcnow().timestamp())}",
            source="local_state_store",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_watch_event(
    audit_logger: "AuditLogger",
    *,
    source_id: str,
    action: str,
    status: str = "success",
    event_type: Optional[str] = None,
    affected_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
    redaction_policy: Optional["RedactionPolicy"] = None,
) -> None:
    """Log file system watch event.

    Args:
        audit_logger: Audit logger instance
        source_id: Source identifier
        action: Watch action (started, stopped, event_received)
        status: Event status
        event_type: File system event type (created, modified, deleted)
        affected_path: Path that triggered the event
        source_path: Source root for relative paths
        redaction_policy: Optional path redaction policy
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "source_id": source_id,
    }

    if event_type:
        metadata["fs_event_type"] = event_type

    if affected_path:
        redacted = _redact_path(affected_path, source_path, redaction_policy)
        metadata["affected_path"] = redacted

    audit_logger.record(
        AuditEvent(
            job_id=f"local_watch_{source_id}_{int(datetime.utcnow().timestamp())}",
            source="local_file_watcher",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_consent_check_failed(
    audit_logger: "AuditLogger",
    *,
    source_id: str,
    scope: str,
    operation: str,
) -> None:
    """Log consent check failure.

    Args:
        audit_logger: Audit logger instance
        source_id: Source identifier
        scope: Required consent scope
        operation: Operation that was blocked
    """
    from ...privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"local_consent_fail_{source_id}_{int(datetime.utcnow().timestamp())}",
            source="local_consent_manager",
            action=LocalFilesAuditEvents.CONSENT_CHECK_FAILED,
            status="blocked",
            timestamp=datetime.utcnow(),
            metadata={
                "source_id": source_id,
                "scope": scope,
                "operation": operation,
            },
        )
    )


__all__ = [
    "LocalFilesAuditEvents",
    "log_source_event",
    "log_file_event",
    "log_sync_event",
    "log_state_event",
    "log_watch_event",
    "log_consent_check_failed",
]

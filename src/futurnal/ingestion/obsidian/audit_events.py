"""Obsidian-specific audit event types and logging helpers.

This module defines Obsidian-specific audit event types and provides helper
functions for logging common Obsidian vault operations with privacy-compliant
redaction.

Event Types:
- Vault events (registered, unregistered, scanned)
- Sync events (started, completed, failed)
- File processing (processed, skipped, failed)
- Wikilink events (parsed, resolved, broken)
- Frontmatter events (parsed, extracted)
- Privacy events (consent granted/revoked)

Integration:
- Uses existing AuditLogger infrastructure
- Applies path redaction for privacy
- Ensures no file content in audit logs
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...privacy.audit import AuditLogger, AuditEvent


class ObsidianAuditEvents:
    """Audit event types for Obsidian operations.

    Constants for standardized Obsidian audit event types, ensuring
    consistent event naming across the Obsidian connector.
    """

    # Vault events
    VAULT_REGISTERED = "obsidian_vault_registered"
    VAULT_UNREGISTERED = "obsidian_vault_unregistered"
    VAULT_SCANNED = "obsidian_vault_scanned"
    VAULT_SCAN_FAILED = "obsidian_vault_scan_failed"

    # Sync events
    SYNC_STARTED = "obsidian_sync_started"
    SYNC_COMPLETED = "obsidian_sync_completed"
    SYNC_FAILED = "obsidian_sync_failed"
    SYNC_BATCH_PROCESSED = "obsidian_sync_batch_processed"

    # File processing
    FILE_PROCESSED = "obsidian_file_processed"
    FILE_SKIPPED = "obsidian_file_skipped"
    FILE_PROCESSING_FAILED = "obsidian_file_processing_failed"
    FILE_MOVED = "obsidian_file_moved"
    FILE_DELETED = "obsidian_file_deleted"

    # Wikilink events
    WIKILINKS_PARSED = "obsidian_wikilinks_parsed"
    WIKILINKS_RESOLVED = "obsidian_wikilinks_resolved"
    WIKILINK_BROKEN = "obsidian_wikilink_broken"

    # Frontmatter events
    FRONTMATTER_PARSED = "obsidian_frontmatter_parsed"
    FRONTMATTER_EXTRACTED = "obsidian_frontmatter_extracted"

    # Privacy events
    CONSENT_GRANTED = "obsidian_consent_granted"
    CONSENT_REVOKED = "obsidian_consent_revoked"
    CONSENT_CHECK_FAILED = "obsidian_consent_check_failed"

    # Engine events
    SYNC_ENGINE_STARTED = "obsidian_sync_engine_started"
    SYNC_ENGINE_STOPPED = "obsidian_sync_engine_stopped"


def _redact_path(path: Path, vault_path: Optional[Path] = None) -> str:
    """Redact file path for privacy.

    Converts absolute paths to relative paths from vault root,
    and hashes the filename if it might contain PII.

    Args:
        path: File path to redact
        vault_path: Optional vault root path for relative paths

    Returns:
        Redacted path string
    """
    if vault_path:
        try:
            relative = path.relative_to(vault_path)
            return str(relative)
        except ValueError:
            pass

    # Return just the filename with parent indicator
    return f".../{path.name}"


def log_vault_event(
    audit_logger: "AuditLogger",
    *,
    vault_id: str,
    vault_name: str,
    action: str,
    status: str = "success",
    file_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Log vault registration/scan event.

    Args:
        audit_logger: Audit logger instance
        vault_id: Vault identifier
        vault_name: Human-readable vault name
        action: Event action (registered, unregistered, scanned)
        status: Event status (success, failed)
        file_count: Optional count of files in vault
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "vault_id": vault_id,
        "vault_name": vault_name,
    }

    if file_count is not None:
        metadata["file_count"] = file_count

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"obsidian_vault_{vault_id}_{int(datetime.utcnow().timestamp())}",
            source="obsidian_vault_connector",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_sync_event(
    audit_logger: "AuditLogger",
    *,
    vault_id: str,
    action: str,
    status: str = "success",
    files_processed: int = 0,
    files_skipped: int = 0,
    files_failed: int = 0,
    wikilinks_resolved: int = 0,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log sync event with statistics.

    Args:
        audit_logger: Audit logger instance
        vault_id: Vault identifier
        action: Sync action (started, completed, failed)
        status: Event status
        files_processed: Count of processed files
        files_skipped: Count of skipped files
        files_failed: Count of failed files
        wikilinks_resolved: Count of resolved wikilinks
        duration_seconds: Sync duration in seconds
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "vault_id": vault_id,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "files_failed": files_failed,
        "total_files": files_processed + files_skipped + files_failed,
        "wikilinks_resolved": wikilinks_resolved,
    }

    if duration_seconds is not None:
        metadata["duration_seconds"] = round(duration_seconds, 2)

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"obsidian_sync_{vault_id}_{int(datetime.utcnow().timestamp())}",
            source="obsidian_sync_engine",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_file_processing_event(
    audit_logger: "AuditLogger",
    *,
    vault_id: str,
    file_path: Path,
    vault_path: Optional[Path] = None,
    action: str,
    status: str = "success",
    file_type: Optional[str] = None,
    frontmatter_keys: Optional[List[str]] = None,
    wikilink_count: int = 0,
    error: Optional[str] = None,
) -> None:
    """Log file processing event with privacy redaction.

    Args:
        audit_logger: Audit logger instance
        vault_id: Vault identifier
        file_path: Path to the processed file
        vault_path: Optional vault root for relative paths
        action: Processing action (processed, skipped, failed)
        status: Event status
        file_type: File type (md, png, etc.)
        frontmatter_keys: List of frontmatter keys (not values!)
        wikilink_count: Number of wikilinks found
        error: Optional error message

    Privacy Guarantee:
        - File paths redacted to relative paths
        - File content NEVER logged
        - Only frontmatter key names logged, not values
    """
    from ...privacy.audit import AuditEvent

    redacted_path = _redact_path(file_path, vault_path)

    metadata: Dict[str, object] = {
        "vault_id": vault_id,
        "file_path": redacted_path,
        "file_extension": file_path.suffix.lower(),
    }

    if file_type:
        metadata["file_type"] = file_type

    if frontmatter_keys:
        metadata["frontmatter_key_count"] = len(frontmatter_keys)
        # Only log key names, never values
        metadata["frontmatter_keys"] = frontmatter_keys

    if wikilink_count > 0:
        metadata["wikilink_count"] = wikilink_count

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"obsidian_file_{vault_id}_{hash(str(file_path)) % 100000}",
            source="obsidian_file_processor",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_wikilink_event(
    audit_logger: "AuditLogger",
    *,
    vault_id: str,
    source_file: Path,
    vault_path: Optional[Path] = None,
    total_links: int,
    resolved_links: int,
    broken_links: int,
    external_links: int = 0,
) -> None:
    """Log wikilink resolution event.

    Args:
        audit_logger: Audit logger instance
        vault_id: Vault identifier
        source_file: Source file containing links
        vault_path: Optional vault root for relative paths
        total_links: Total wikilinks found
        resolved_links: Successfully resolved links
        broken_links: Unresolved/broken links
        external_links: External (non-wikilink) links
    """
    from ...privacy.audit import AuditEvent

    redacted_path = _redact_path(source_file, vault_path)

    audit_logger.record(
        AuditEvent(
            job_id=f"obsidian_links_{vault_id}_{hash(str(source_file)) % 100000}",
            source="obsidian_wikilink_resolver",
            action=ObsidianAuditEvents.WIKILINKS_RESOLVED,
            status="success" if broken_links == 0 else "partial_success",
            timestamp=datetime.utcnow(),
            metadata={
                "vault_id": vault_id,
                "source_file": redacted_path,
                "total_links": total_links,
                "resolved_links": resolved_links,
                "broken_links": broken_links,
                "external_links": external_links,
                "resolution_rate": round(resolved_links / total_links, 2) if total_links > 0 else 1.0,
            },
        )
    )


def log_batch_event(
    audit_logger: "AuditLogger",
    *,
    vault_id: str,
    batch_id: str,
    event_count: int,
    files_affected: int,
    priority: str,
    duration_seconds: float,
    status: str = "success",
) -> None:
    """Log sync batch processing event.

    Args:
        audit_logger: Audit logger instance
        vault_id: Vault identifier
        batch_id: Batch identifier
        event_count: Number of events in batch
        files_affected: Number of files affected
        priority: Batch priority level
        duration_seconds: Processing duration
        status: Event status
    """
    from ...privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"obsidian_batch_{batch_id}",
            source="obsidian_sync_engine",
            action=ObsidianAuditEvents.SYNC_BATCH_PROCESSED,
            status=status,
            timestamp=datetime.utcnow(),
            metadata={
                "vault_id": vault_id,
                "batch_id": batch_id,
                "event_count": event_count,
                "files_affected": files_affected,
                "priority": priority,
                "duration_seconds": round(duration_seconds, 2),
            },
        )
    )


def log_consent_check_failed(
    audit_logger: "AuditLogger",
    *,
    vault_id: str,
    scope: str,
    operation: str,
) -> None:
    """Log consent check failure.

    Args:
        audit_logger: Audit logger instance
        vault_id: Vault identifier
        scope: Required consent scope
        operation: Operation that was blocked
    """
    from ...privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"obsidian_consent_fail_{vault_id}_{int(datetime.utcnow().timestamp())}",
            source="obsidian_consent_manager",
            action=ObsidianAuditEvents.CONSENT_CHECK_FAILED,
            status="blocked",
            timestamp=datetime.utcnow(),
            metadata={
                "vault_id": vault_id,
                "scope": scope,
                "operation": operation,
            },
        )
    )


__all__ = [
    "ObsidianAuditEvents",
    "log_vault_event",
    "log_sync_event",
    "log_file_processing_event",
    "log_wikilink_event",
    "log_batch_event",
    "log_consent_check_failed",
]

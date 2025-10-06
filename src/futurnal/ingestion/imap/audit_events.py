"""IMAP-specific audit event types and logging helpers.

This module defines IMAP-specific audit event types and provides helper
functions for logging common IMAP operations with privacy-compliant
redaction.

Event Types:
- Connection events (establish, fail, close)
- Sync events (started, completed, failed)
- Email processing (fetched, parsed, failed)
- Attachment events (extracted, processed, skipped)
- Thread reconstruction
- Privacy events (consent granted/revoked)

Integration:
- Uses existing AuditLogger infrastructure
- Applies EmailHeaderRedactionPolicy for email events
- Ensures no PII in audit logs
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel

from ...privacy.audit import AuditEvent, AuditLogger
from .email_redaction import EmailHeaderRedactionPolicy


class ImapAuditEvents:
    """Audit event types for IMAP operations.

    Constants for standardized IMAP audit event types, ensuring
    consistent event naming across the IMAP connector.
    """

    # Connection events
    CONNECTION_ESTABLISHED = "imap_connection_established"
    CONNECTION_FAILED = "imap_connection_failed"
    CONNECTION_CLOSED = "imap_connection_closed"

    # Sync events
    SYNC_STARTED = "imap_sync_started"
    SYNC_COMPLETED = "imap_sync_completed"
    SYNC_FAILED = "imap_sync_failed"

    # Email processing
    EMAIL_FETCHED = "imap_email_fetched"
    EMAIL_PARSED = "imap_email_parsed"
    EMAIL_PROCESSING_FAILED = "imap_email_processing_failed"

    # Attachment processing
    ATTACHMENT_EXTRACTED = "imap_attachment_extracted"
    ATTACHMENT_PROCESSED = "imap_attachment_processed"
    ATTACHMENT_SKIPPED = "imap_attachment_skipped"

    # Thread reconstruction
    THREAD_RECONSTRUCTED = "imap_thread_reconstructed"

    # Privacy events
    CONSENT_GRANTED = "imap_consent_granted"
    CONSENT_REVOKED = "imap_consent_revoked"
    CONSENT_CHECK_FAILED = "imap_consent_check_failed"


def log_connection_event(
    audit_logger: AuditLogger,
    *,
    mailbox_id: str,
    host: str,
    port: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Log IMAP connection event.

    Args:
        audit_logger: Audit logger instance
        mailbox_id: Mailbox identifier
        host: IMAP host (domain preserved for debugging)
        port: IMAP port
        status: Event status (success, failed)
        error: Optional error message
    """
    action = (
        ImapAuditEvents.CONNECTION_ESTABLISHED
        if status == "success"
        else ImapAuditEvents.CONNECTION_FAILED
    )

    metadata: Dict[str, object] = {
        "mailbox_id": mailbox_id,
        "host": host,  # Domain OK for debugging
        "port": port,
    }

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"imap_conn_{mailbox_id}_{int(datetime.utcnow().timestamp())}",
            source="imap_connection_manager",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_email_sync_event(
    audit_logger: AuditLogger,
    *,
    mailbox_id: str,
    folder: str,
    new_messages: int,
    updated_messages: int,
    deleted_messages: int,
    sync_duration_seconds: float,
    errors: int = 0,
) -> None:
    """Log email sync event with privacy redaction.

    Args:
        audit_logger: Audit logger instance
        mailbox_id: Mailbox identifier
        folder: Folder name
        new_messages: Count of new messages
        updated_messages: Count of updated messages
        deleted_messages: Count of deleted messages
        sync_duration_seconds: Sync duration in seconds
        errors: Error count
    """
    audit_logger.record(
        AuditEvent(
            job_id=f"imap_sync_{mailbox_id}_{int(datetime.utcnow().timestamp())}",
            source="imap_sync_engine",
            action=ImapAuditEvents.SYNC_COMPLETED,
            status="success" if errors == 0 else "partial_success",
            timestamp=datetime.utcnow(),
            metadata={
                "mailbox_id": mailbox_id,
                "folder": folder,
                "new_messages": new_messages,
                "updated_messages": updated_messages,
                "deleted_messages": deleted_messages,
                "total_changes": new_messages + updated_messages + deleted_messages,
                "sync_duration_seconds": round(sync_duration_seconds, 2),
                "errors": errors,
            },
        )
    )


def log_email_processing_event(
    audit_logger: AuditLogger,
    *,
    email_message: BaseModel,
    redaction_policy: EmailHeaderRedactionPolicy,
    status: str = "success",
    error: Optional[str] = None,
) -> None:
    """Log email processing event with redaction.

    This function logs email parsing/processing events with full
    privacy redaction applied. Email addresses and optionally subjects
    are redacted according to the policy.

    Args:
        audit_logger: Audit logger instance
        email_message: EmailMessage object to log
        redaction_policy: Redaction policy to apply
        status: Event status (success, failed)
        error: Optional error message

    Privacy Guarantee:
        - Email addresses redacted per policy
        - Subject optionally redacted per policy
        - Email body NEVER logged
    """
    # Apply redaction policy
    redacted_email = redaction_policy.redact_email_message(email_message)

    # Add error if present
    if error:
        redacted_email["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"imap_email_{email_message.uid}",
            source="imap_email_processor",
            action=ImapAuditEvents.EMAIL_PARSED,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=redacted_email,
        )
    )


def log_attachment_event(
    audit_logger: AuditLogger,
    *,
    mailbox_id: str,
    message_id_hash: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    action: str,
    status: str = "success",
) -> None:
    """Log attachment processing event.

    Args:
        audit_logger: Audit logger instance
        mailbox_id: Mailbox identifier
        message_id_hash: Hashed message ID
        filename: Attachment filename
        content_type: MIME content type
        size_bytes: Attachment size
        action: Event action (extracted, processed, skipped)
        status: Event status
    """
    audit_logger.record(
        AuditEvent(
            job_id=f"imap_attach_{message_id_hash}",
            source="imap_attachment_processor",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata={
                "mailbox_id": mailbox_id,
                "message_id_hash": message_id_hash,
                "filename": filename,
                "content_type": content_type,
                "size_bytes": size_bytes,
            },
        )
    )


def log_thread_reconstruction_event(
    audit_logger: AuditLogger,
    *,
    mailbox_id: str,
    thread_id: str,
    message_count: int,
    participant_count: int,
    duration_seconds: float,
) -> None:
    """Log thread reconstruction event.

    Args:
        audit_logger: Audit logger instance
        mailbox_id: Mailbox identifier
        thread_id: Thread identifier
        message_count: Messages in thread
        participant_count: Unique participants
        duration_seconds: Processing duration
    """
    audit_logger.record(
        AuditEvent(
            job_id=f"imap_thread_{thread_id}",
            source="imap_thread_reconstructor",
            action=ImapAuditEvents.THREAD_RECONSTRUCTED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "mailbox_id": mailbox_id,
                "thread_id": thread_id,
                "message_count": message_count,
                "participant_count": participant_count,
                "duration_seconds": round(duration_seconds, 2),
            },
        )
    )


def log_consent_check_failed(
    audit_logger: AuditLogger,
    *,
    mailbox_id: str,
    scope: str,
    operation: str,
) -> None:
    """Log consent check failure.

    Args:
        audit_logger: Audit logger instance
        mailbox_id: Mailbox identifier
        scope: Required consent scope
        operation: Operation that was blocked
    """
    audit_logger.record(
        AuditEvent(
            job_id=f"imap_consent_fail_{mailbox_id}_{int(datetime.utcnow().timestamp())}",
            source="imap_consent_manager",
            action=ImapAuditEvents.CONSENT_CHECK_FAILED,
            status="blocked",
            timestamp=datetime.utcnow(),
            metadata={
                "mailbox_id": mailbox_id,
                "scope": scope,
                "operation": operation,
            },
        )
    )


__all__ = [
    "ImapAuditEvents",
    "log_connection_event",
    "log_email_sync_event",
    "log_email_processing_event",
    "log_attachment_event",
    "log_thread_reconstruction_event",
    "log_consent_check_failed",
]

"""Comprehensive tests for IMAP audit events.

Tests cover:
- Event type constants
- Connection event logging
- Email sync event logging
- Email processing event logging with redaction
- Attachment event logging
- Thread reconstruction event logging
- Consent check failure logging
- No PII in audit events
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from futurnal.ingestion.imap.audit_events import (
    ImapAuditEvents,
    log_attachment_event,
    log_connection_event,
    log_consent_check_failed,
    log_email_processing_event,
    log_email_sync_event,
    log_thread_reconstruction_event,
)
from futurnal.ingestion.imap.email_parser import EmailAddress, EmailMessage
from futurnal.ingestion.imap.email_redaction import EmailHeaderRedactionPolicy
from futurnal.privacy.audit import AuditLogger


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    """Create audit logger for testing."""
    return AuditLogger(output_dir=tmp_path / "audit")


@pytest.fixture
def sample_email_message() -> EmailMessage:
    """Create sample EmailMessage for testing."""
    return EmailMessage(
        message_id="<test@example.com>",
        uid=123,
        folder="INBOX",
        subject="Test Subject",
        from_address=EmailAddress(address="sender@example.com"),
        to_addresses=[EmailAddress(address="recipient@example.com")],
        cc_addresses=[],
        bcc_addresses=[],
        reply_to_addresses=[],
        date=datetime(2024, 1, 1, 12, 0, 0),
        in_reply_to=None,
        references=[],
        body_plain="Test body",
        body_html=None,
        body_normalized="Test body",
        size_bytes=100,
        flags=[],
        labels=[],
        attachments=[],
        contains_sensitive_keywords=False,
        privacy_classification="standard",
        retrieved_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )


# ============================================================================
# Event Type Constants Tests
# ============================================================================


def test_audit_event_types_defined():
    """Test all audit event types are defined."""
    assert ImapAuditEvents.CONNECTION_ESTABLISHED == "imap_connection_established"
    assert ImapAuditEvents.CONNECTION_FAILED == "imap_connection_failed"
    assert ImapAuditEvents.CONNECTION_CLOSED == "imap_connection_closed"
    assert ImapAuditEvents.SYNC_STARTED == "imap_sync_started"
    assert ImapAuditEvents.SYNC_COMPLETED == "imap_sync_completed"
    assert ImapAuditEvents.SYNC_FAILED == "imap_sync_failed"
    assert ImapAuditEvents.EMAIL_FETCHED == "imap_email_fetched"
    assert ImapAuditEvents.EMAIL_PARSED == "imap_email_parsed"
    assert ImapAuditEvents.EMAIL_PROCESSING_FAILED == "imap_email_processing_failed"
    assert ImapAuditEvents.ATTACHMENT_EXTRACTED == "imap_attachment_extracted"
    assert ImapAuditEvents.ATTACHMENT_PROCESSED == "imap_attachment_processed"
    assert ImapAuditEvents.ATTACHMENT_SKIPPED == "imap_attachment_skipped"
    assert ImapAuditEvents.THREAD_RECONSTRUCTED == "imap_thread_reconstructed"
    assert ImapAuditEvents.CONSENT_GRANTED == "imap_consent_granted"
    assert ImapAuditEvents.CONSENT_REVOKED == "imap_consent_revoked"
    assert ImapAuditEvents.CONSENT_CHECK_FAILED == "imap_consent_check_failed"


# ============================================================================
# Connection Event Logging Tests
# ============================================================================


def test_log_connection_event_success(audit_logger: AuditLogger):
    """Test logging successful connection event."""
    log_connection_event(
        audit_logger,
        mailbox_id="test-mailbox",
        host="imap.example.com",
        port=993,
        status="success",
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    assert audit_path.exists()

    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.CONNECTION_ESTABLISHED
    assert event["status"] == "success"
    assert event["metadata"]["mailbox_id"] == "test-mailbox"
    assert event["metadata"]["host"] == "imap.example.com"
    assert event["metadata"]["port"] == 993


def test_log_connection_event_failure(audit_logger: AuditLogger):
    """Test logging failed connection event."""
    log_connection_event(
        audit_logger,
        mailbox_id="test-mailbox",
        host="imap.example.com",
        port=993,
        status="failed",
        error="Connection timeout",
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.CONNECTION_FAILED
    assert event["status"] == "failed"
    assert event["metadata"]["error"] == "Connection timeout"


# ============================================================================
# Email Sync Event Logging Tests
# ============================================================================


def test_log_email_sync_event(audit_logger: AuditLogger):
    """Test logging email sync event."""
    log_email_sync_event(
        audit_logger,
        mailbox_id="test-mailbox",
        folder="INBOX",
        new_messages=10,
        updated_messages=5,
        deleted_messages=2,
        sync_duration_seconds=3.5,
        errors=0,
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.SYNC_COMPLETED
    assert event["status"] == "success"
    assert event["metadata"]["mailbox_id"] == "test-mailbox"
    assert event["metadata"]["folder"] == "INBOX"
    assert event["metadata"]["new_messages"] == 10
    assert event["metadata"]["updated_messages"] == 5
    assert event["metadata"]["deleted_messages"] == 2
    assert event["metadata"]["total_changes"] == 17
    assert event["metadata"]["sync_duration_seconds"] == 3.5
    assert event["metadata"]["errors"] == 0


def test_log_email_sync_event_with_errors(audit_logger: AuditLogger):
    """Test logging sync event with errors."""
    log_email_sync_event(
        audit_logger,
        mailbox_id="test-mailbox",
        folder="INBOX",
        new_messages=10,
        updated_messages=0,
        deleted_messages=0,
        sync_duration_seconds=2.0,
        errors=3,
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["status"] == "partial_success"
    assert event["metadata"]["errors"] == 3


# ============================================================================
# Email Processing Event Logging Tests
# ============================================================================


def test_log_email_processing_event(
    audit_logger: AuditLogger,
    sample_email_message: EmailMessage,
):
    """Test logging email processing event with redaction."""
    policy = EmailHeaderRedactionPolicy(redact_sender=True, redact_recipients=True)

    log_email_processing_event(
        audit_logger,
        email_message=sample_email_message,
        redaction_policy=policy,
        status="success",
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.EMAIL_PARSED
    assert event["status"] == "success"

    # Verify redaction applied
    metadata = event["metadata"]
    assert "@example.com" in metadata["from"]  # Domain preserved
    assert "sender" not in metadata["from"].lower()  # Local part redacted

    # Recipients should be counts
    assert metadata["to_count"] == 1
    assert "to" not in metadata  # Full addresses not included


def test_log_email_processing_event_no_pii(
    audit_logger: AuditLogger,
    sample_email_message: EmailMessage,
):
    """Test that email processing event contains no PII."""
    policy = EmailHeaderRedactionPolicy(redact_sender=True, redact_recipients=True)

    log_email_processing_event(
        audit_logger,
        email_message=sample_email_message,
        redaction_policy=policy,
    )

    # Read audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Should NOT contain sensitive info
    assert "sender@example.com" not in content
    assert "recipient@example.com" not in content
    assert "Test body" not in content  # Body never logged


def test_log_email_processing_event_with_error(
    audit_logger: AuditLogger,
    sample_email_message: EmailMessage,
):
    """Test logging email processing event with error."""
    policy = EmailHeaderRedactionPolicy()

    log_email_processing_event(
        audit_logger,
        email_message=sample_email_message,
        redaction_policy=policy,
        status="failed",
        error="Parsing failed",
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["status"] == "failed"
    assert event["metadata"]["error"] == "Parsing failed"


# ============================================================================
# Attachment Event Logging Tests
# ============================================================================


def test_log_attachment_event(audit_logger: AuditLogger):
    """Test logging attachment event."""
    log_attachment_event(
        audit_logger,
        mailbox_id="test-mailbox",
        message_id_hash="abc123",
        filename="document.pdf",
        content_type="application/pdf",
        size_bytes=50000,
        action=ImapAuditEvents.ATTACHMENT_EXTRACTED,
        status="success",
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.ATTACHMENT_EXTRACTED
    assert event["status"] == "success"
    assert event["metadata"]["mailbox_id"] == "test-mailbox"
    assert event["metadata"]["message_id_hash"] == "abc123"
    assert event["metadata"]["filename"] == "document.pdf"
    assert event["metadata"]["content_type"] == "application/pdf"
    assert event["metadata"]["size_bytes"] == 50000


# ============================================================================
# Thread Reconstruction Event Logging Tests
# ============================================================================


def test_log_thread_reconstruction_event(audit_logger: AuditLogger):
    """Test logging thread reconstruction event."""
    log_thread_reconstruction_event(
        audit_logger,
        mailbox_id="test-mailbox",
        thread_id="thread-123",
        message_count=15,
        participant_count=5,
        duration_seconds=1.5,
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.THREAD_RECONSTRUCTED
    assert event["status"] == "success"
    assert event["metadata"]["mailbox_id"] == "test-mailbox"
    assert event["metadata"]["thread_id"] == "thread-123"
    assert event["metadata"]["message_count"] == 15
    assert event["metadata"]["participant_count"] == 5
    assert event["metadata"]["duration_seconds"] == 1.5


# ============================================================================
# Consent Check Failure Logging Tests
# ============================================================================


def test_log_consent_check_failed(audit_logger: AuditLogger):
    """Test logging consent check failure."""
    log_consent_check_failed(
        audit_logger,
        mailbox_id="test-mailbox",
        scope="imap:mailbox:access",
        operation="folder_sync",
    )

    # Verify event logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == ImapAuditEvents.CONSENT_CHECK_FAILED
    assert event["status"] == "blocked"
    assert event["metadata"]["mailbox_id"] == "test-mailbox"
    assert event["metadata"]["scope"] == "imap:mailbox:access"
    assert event["metadata"]["operation"] == "folder_sync"


# ============================================================================
# Privacy Compliance Tests
# ============================================================================


def test_no_email_bodies_in_logs(
    audit_logger: AuditLogger,
    sample_email_message: EmailMessage,
):
    """Test that email bodies are NEVER logged."""
    policy = EmailHeaderRedactionPolicy()

    log_email_processing_event(
        audit_logger,
        email_message=sample_email_message,
        redaction_policy=policy,
    )

    # Read audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Body content should NEVER appear
    assert "Test body" not in content
    assert "body_plain" not in content
    assert "body_html" not in content
    assert "body_normalized" not in content


def test_email_addresses_redacted_in_logs(
    audit_logger: AuditLogger,
    sample_email_message: EmailMessage,
):
    """Test that email addresses are redacted in logs."""
    policy = EmailHeaderRedactionPolicy(redact_sender=True, redact_recipients=True)

    log_email_processing_event(
        audit_logger,
        email_message=sample_email_message,
        redaction_policy=policy,
    )

    # Read audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Original email addresses should NOT appear
    assert "sender@example.com" not in content
    assert "recipient@example.com" not in content

    # Domain should be preserved for debugging
    assert "@example.com" in content


# ============================================================================
# Multiple Event Logging Tests
# ============================================================================


def test_multiple_events_logged_sequentially(audit_logger: AuditLogger):
    """Test logging multiple events in sequence."""
    # Log connection event
    log_connection_event(
        audit_logger,
        mailbox_id="test-mailbox",
        host="imap.example.com",
        port=993,
        status="success",
    )

    # Log sync event
    log_email_sync_event(
        audit_logger,
        mailbox_id="test-mailbox",
        folder="INBOX",
        new_messages=5,
        updated_messages=0,
        deleted_messages=0,
        sync_duration_seconds=1.0,
    )

    # Verify both events logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    lines = audit_path.read_text().strip().split("\n")

    assert len(lines) == 2

    event1 = json.loads(lines[0])
    event2 = json.loads(lines[1])

    assert event1["action"] == ImapAuditEvents.CONNECTION_ESTABLISHED
    assert event2["action"] == ImapAuditEvents.SYNC_COMPLETED


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_audit_flow(
    audit_logger: AuditLogger,
    sample_email_message: EmailMessage,
):
    """Test complete audit flow for email processing."""
    policy = EmailHeaderRedactionPolicy()

    # Step 1: Connection
    log_connection_event(
        audit_logger,
        mailbox_id="test-mailbox",
        host="imap.example.com",
        port=993,
        status="success",
    )

    # Step 2: Sync
    log_email_sync_event(
        audit_logger,
        mailbox_id="test-mailbox",
        folder="INBOX",
        new_messages=1,
        updated_messages=0,
        deleted_messages=0,
        sync_duration_seconds=0.5,
    )

    # Step 3: Email processing
    log_email_processing_event(
        audit_logger,
        email_message=sample_email_message,
        redaction_policy=policy,
    )

    # Verify all events logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    lines = audit_path.read_text().strip().split("\n")

    assert len(lines) == 3

    # Verify chain integrity (each event should have chain_hash)
    for line in lines:
        event = json.loads(line)
        assert "chain_hash" in event
        assert "chain_prev" in event

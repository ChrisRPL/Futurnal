"""Privacy integration tests for IMAP connector.

Tests the complete privacy & audit flow:
- End-to-end consent workflow
- Consent enforcement in EmailParser
- Consent enforcement in SyncEngine
- Audit trail generation
- GDPR compliance (data export/deletion)
- No PII leaks in any logs
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import pytest

from futurnal.ingestion.imap.consent_manager import (
    ImapConsentManager,
    ImapConsentScopes,
)
from futurnal.ingestion.imap.email_parser import EmailParser
from futurnal.ingestion.imap.email_redaction import EmailHeaderRedactionPolicy
from futurnal.ingestion.imap.descriptor import MailboxPrivacySettings
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.audit import AuditLogger


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def consent_registry(tmp_path: Path) -> ConsentRegistry:
    """Create consent registry for testing."""
    return ConsentRegistry(directory=tmp_path / "consent")


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    """Create audit logger for testing."""
    return AuditLogger(output_dir=tmp_path / "audit")


@pytest.fixture
def consent_manager(
    consent_registry: ConsentRegistry,
    audit_logger: AuditLogger,
) -> ImapConsentManager:
    """Create consent manager for testing."""
    return ImapConsentManager(
        consent_registry=consent_registry,
        audit_logger=audit_logger,
    )


@pytest.fixture
def privacy_settings() -> MailboxPrivacySettings:
    """Create privacy settings for testing."""
    return MailboxPrivacySettings(
        enable_sender_anonymization=True,
        enable_recipient_anonymization=True,
        enable_subject_redaction=False,
        privacy_subject_keywords=["confidential", "private"],
    )


@pytest.fixture
def email_parser(
    privacy_settings: MailboxPrivacySettings,
    audit_logger: AuditLogger,
    consent_manager: ImapConsentManager,
) -> EmailParser:
    """Create email parser with privacy components."""
    return EmailParser(
        privacy_policy=privacy_settings,
        audit_logger=audit_logger,
        consent_manager=consent_manager,
    )


@pytest.fixture
def sample_email_bytes() -> bytes:
    """Create sample email message bytes."""
    msg = MIMEMultipart()
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = "Test Email"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

    body = MIMEText("This is a test email body.", "plain")
    msg.attach(body)

    return msg.as_bytes()


# ============================================================================
# End-to-End Consent Flow Tests
# ============================================================================


@patch("builtins.input", side_effect=["yes", "yes"])
@patch("builtins.print")
def test_consent_flow_email_parsing_allowed(
    mock_print: MagicMock,
    mock_input: MagicMock,
    email_parser: EmailParser,
    consent_manager: ImapConsentManager,
    sample_email_bytes: bytes,
):
    """Test that email parsing works with consent granted."""
    # Grant consent interactively
    consent_manager.request_mailbox_consent(
        mailbox_id="test-mailbox",
        email_address="test@example.com",
        required_scopes=[
            ImapConsentScopes.MAILBOX_ACCESS.value,
            ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
        ],
    )

    # Parse email - should succeed
    email_message = email_parser.parse_message(
        raw_message=sample_email_bytes,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_message is not None
    assert email_message.uid == 1


def test_consent_flow_email_parsing_blocked(
    email_parser: EmailParser,
    sample_email_bytes: bytes,
):
    """Test that email parsing is blocked without consent."""
    # Do NOT grant consent

    # Attempt to parse email - should raise ConsentRequiredError
    with pytest.raises(ConsentRequiredError, match="Consent required"):
        email_parser.parse_message(
            raw_message=sample_email_bytes,
            uid=1,
            folder="INBOX",
            mailbox_id="test-mailbox",
        )


def test_consent_revoked_blocks_parsing(
    email_parser: EmailParser,
    consent_manager: ImapConsentManager,
    sample_email_bytes: bytes,
):
    """Test that revoking consent blocks operations."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Parse email - should succeed
    email_message = email_parser.parse_message(
        raw_message=sample_email_bytes,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )
    assert email_message is not None

    # Revoke consent
    consent_manager.revoke_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Attempt to parse again - should fail
    with pytest.raises(ConsentRequiredError):
        email_parser.parse_message(
            raw_message=sample_email_bytes,
            uid=2,
            folder="INBOX",
            mailbox_id="test-mailbox",
        )


# ============================================================================
# Audit Trail Generation Tests
# ============================================================================


def test_audit_trail_for_consent_decisions(
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
):
    """Test that consent decisions generate audit events."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        operator="test_user",
    )

    # Verify audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    assert audit_path.exists()

    content = audit_path.read_text()
    event = json.loads(content.strip())

    assert event["action"] == "consent:imap:mailbox:access"
    assert event["status"] == "granted"
    assert event["operator_action"] == "test_user"
    assert "consent_token_hash" in event


def test_audit_trail_for_parsing_with_redaction(
    email_parser: EmailParser,
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
    sample_email_bytes: bytes,
):
    """Test that email parsing generates redacted audit events."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Parse email
    email_parser.parse_message(
        raw_message=sample_email_bytes,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    # Verify audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Find parsing event (skip consent event)
    lines = content.strip().split("\n")
    parsing_event = None
    for line in lines:
        event = json.loads(line)
        if event["action"] == "imap_email_parsed":
            parsing_event = event
            break

    assert parsing_event is not None

    # Verify redaction applied
    metadata = parsing_event["metadata"]
    assert "@example.com" in metadata["from"]  # Domain preserved
    assert "sender" not in metadata["from"].lower()  # Local part redacted


def test_consent_check_failure_logged(
    email_parser: EmailParser,
    audit_logger: AuditLogger,
    sample_email_bytes: bytes,
):
    """Test that consent check failures are logged."""
    # Do NOT grant consent

    # Attempt to parse
    try:
        email_parser.parse_message(
            raw_message=sample_email_bytes,
            uid=1,
            folder="INBOX",
            mailbox_id="test-mailbox",
        )
    except ConsentRequiredError:
        pass

    # Verify consent failure logged
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    assert "imap_consent_check_failed" in content
    assert "email_parsing" in content


# ============================================================================
# GDPR Compliance Tests
# ============================================================================


def test_gdpr_data_export(
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
):
    """Test GDPR data export (retrieve all audit events for mailbox)."""
    mailbox_id = "test-mailbox"

    # Generate some consent events
    consent_manager.grant_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )
    consent_manager.grant_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Read all audit events
    audit_path = audit_logger.output_dir / audit_logger.filename
    lines = audit_path.read_text().strip().split("\n")

    # Filter events for this mailbox
    mailbox_events = []
    for line in lines:
        event = json.loads(line)
        if mailbox_id in event.get("source", ""):
            mailbox_events.append(event)

    # Should have 2 consent events
    assert len(mailbox_events) == 2


def test_gdpr_data_deletion(
    consent_registry: ConsentRegistry,
    consent_manager: ImapConsentManager,
):
    """Test GDPR data deletion (remove all records for mailbox)."""
    mailbox_id = "test-mailbox"

    # Grant consents
    consent_manager.grant_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )
    consent_manager.grant_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Verify consents exist
    assert (
        consent_manager.check_consent(
            mailbox_id=mailbox_id,
            scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        )
        is True
    )

    # Delete all consents for mailbox (simulated)
    consent_manager.revoke_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )
    consent_manager.revoke_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Verify consents revoked
    assert (
        consent_manager.check_consent(
            mailbox_id=mailbox_id,
            scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        )
        is False
    )


# ============================================================================
# PII Leak Prevention Tests
# ============================================================================


def test_no_pii_in_any_logs(
    email_parser: EmailParser,
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
    sample_email_bytes: bytes,
):
    """Test that NO PII appears in any audit logs."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Parse email
    email_parser.parse_message(
        raw_message=sample_email_bytes,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    # Read entire audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Verify NO PII present
    assert "sender@example.com" not in content  # Email addresses
    assert "recipient@example.com" not in content
    assert "This is a test email body" not in content  # Email body
    assert "body_plain" not in content
    assert "body_html" not in content


def test_domain_preserved_for_debugging(
    email_parser: EmailParser,
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
    sample_email_bytes: bytes,
):
    """Test that email domains are preserved for debugging."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Parse email
    email_parser.parse_message(
        raw_message=sample_email_bytes,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    # Read audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Domain should be present (for debugging)
    assert "@example.com" in content

    # But local parts should be hashed
    assert "sender" not in content.lower() or "sender@" not in content


# ============================================================================
# Privacy Keyword Detection Tests
# ============================================================================


def test_sensitive_keyword_detection():
    """Test that sensitive keywords are detected."""
    # Create email with sensitive keyword
    msg = MIMEMultipart()
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = "Confidential: Q4 Results"
    msg["Message-ID"] = "<test@example.com>"
    msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

    body = MIMEText("This is confidential information.", "plain")
    msg.attach(body)

    email_bytes = msg.as_bytes()

    # Create parser with keyword detection
    privacy_settings = MailboxPrivacySettings(
        privacy_subject_keywords=["confidential", "private"]
    )
    parser = EmailParser(privacy_policy=privacy_settings)

    # Grant consent (create minimal consent manager)
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        consent_reg = ConsentRegistry(directory=Path(tmpdir))
        cm = ImapConsentManager(consent_registry=consent_reg)
        cm.grant_consent(
            mailbox_id="test-mailbox",
            scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
        )

        parser.consent_manager = cm

        # Parse email
        email_message = parser.parse_message(
            raw_message=email_bytes,
            uid=1,
            folder="INBOX",
            mailbox_id="test-mailbox",
        )

        # Verify sensitive keywords detected
        assert email_message.contains_sensitive_keywords is True
        assert email_message.privacy_classification == "sensitive"


# ============================================================================
# Multi-Mailbox Isolation Tests
# ============================================================================


def test_multi_mailbox_consent_isolation(consent_manager: ImapConsentManager):
    """Test that consents are isolated between mailboxes."""
    # Grant consent for mailbox1
    consent_manager.grant_consent(
        mailbox_id="mailbox1",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Check both mailboxes
    assert (
        consent_manager.check_consent(
            mailbox_id="mailbox1",
            scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        )
        is True
    )
    assert (
        consent_manager.check_consent(
            mailbox_id="mailbox2",
            scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        )
        is False
    )


# ============================================================================
# Audit Chain Integrity Tests
# ============================================================================


@pytest.mark.skip(reason="Audit chain verification tested in privacy module")
def test_audit_chain_integrity(tmp_path: Path):
    """Test that audit log maintains chain integrity."""
    # Create fresh audit logger for this test
    fresh_audit_logger = AuditLogger(output_dir=tmp_path / "audit_chain_test")
    consent_registry = ConsentRegistry(directory=tmp_path / "consent_chain_test")
    consent_manager = ImapConsentManager(
        consent_registry=consent_registry,
        audit_logger=fresh_audit_logger,
    )

    # Generate multiple events
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )
    consent_manager.revoke_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Verify chain integrity
    verified = fresh_audit_logger.verify()
    assert verified is True

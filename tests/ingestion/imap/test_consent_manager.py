"""Comprehensive tests for IMAP consent manager.

Tests cover:
- Consent scope definitions
- Interactive consent flow (with mocked input)
- Consent checking (granted/revoked)
- Consent enforcement (raises ConsentRequiredError)
- Programmatic consent grant/revoke
- Audit logging of consent decisions
- Integration with ConsentRegistry
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from futurnal.ingestion.imap.consent_manager import (
    ImapConsentManager,
    ImapConsentScopes,
)
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


# ============================================================================
# ImapConsentScopes Tests
# ============================================================================


def test_consent_scopes_values():
    """Test consent scope enum values."""
    assert ImapConsentScopes.MAILBOX_ACCESS.value == "imap:mailbox:access"
    assert ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value == "imap:email:content_analysis"
    assert ImapConsentScopes.EMAIL_METADATA_EXTRACTION.value == "imap:email:metadata_extraction"
    assert ImapConsentScopes.ATTACHMENT_EXTRACTION.value == "imap:email:attachment_extraction"
    assert ImapConsentScopes.THREAD_RECONSTRUCTION.value == "imap:email:thread_reconstruction"
    assert ImapConsentScopes.PARTICIPANT_ANALYSIS.value == "imap:email:participant_analysis"
    assert ImapConsentScopes.CLOUD_MODELS.value == "imap:email:cloud_models"


def test_get_default_scopes():
    """Test default consent scopes."""
    scopes = ImapConsentScopes.get_default_scopes()

    assert len(scopes) == 4
    assert ImapConsentScopes.MAILBOX_ACCESS.value in scopes
    assert ImapConsentScopes.EMAIL_METADATA_EXTRACTION.value in scopes
    assert ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value in scopes
    assert ImapConsentScopes.THREAD_RECONSTRUCTION.value in scopes


def test_get_minimal_scopes():
    """Test minimal consent scopes."""
    scopes = ImapConsentScopes.get_minimal_scopes()

    assert len(scopes) == 2
    assert ImapConsentScopes.MAILBOX_ACCESS.value in scopes
    assert ImapConsentScopes.EMAIL_METADATA_EXTRACTION.value in scopes


# ============================================================================
# Consent Checking Tests
# ============================================================================


def test_check_consent_not_granted(consent_manager: ImapConsentManager):
    """Test checking consent that hasn't been granted."""
    result = consent_manager.check_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    assert result is False


def test_check_consent_granted(consent_manager: ImapConsentManager):
    """Test checking consent that has been granted."""
    # Grant consent programmatically
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Check consent
    result = consent_manager.check_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    assert result is True


def test_check_consent_revoked(consent_manager: ImapConsentManager):
    """Test checking consent that has been revoked."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Revoke consent
    consent_manager.revoke_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Check consent - should be False
    result = consent_manager.check_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    assert result is False


# ============================================================================
# Consent Enforcement Tests
# ============================================================================


def test_require_consent_not_granted(consent_manager: ImapConsentManager):
    """Test requiring consent that hasn't been granted."""
    with pytest.raises(ConsentRequiredError, match="Consent required"):
        consent_manager.require_consent(
            mailbox_id="test-mailbox",
            scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        )


def test_require_consent_granted(consent_manager: ImapConsentManager):
    """Test requiring consent that has been granted."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Require consent - should not raise
    consent_manager.require_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )


# ============================================================================
# Programmatic Consent Management Tests
# ============================================================================


def test_grant_consent(
    consent_manager: ImapConsentManager,
    consent_registry: ConsentRegistry,
):
    """Test programmatic consent grant."""
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        operator="test_operator",
    )

    # Verify in registry
    record = consent_registry.get(
        source="mailbox:test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    assert record is not None
    assert record.granted is True
    assert record.operator == "test_operator"


def test_revoke_consent(
    consent_manager: ImapConsentManager,
    consent_registry: ConsentRegistry,
):
    """Test programmatic consent revoke."""
    # Grant first
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Revoke
    consent_manager.revoke_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        operator="test_operator",
    )

    # Verify in registry
    record = consent_registry.get(
        source="mailbox:test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    assert record is not None
    assert record.granted is False
    assert record.operator == "test_operator"


def test_grant_consent_with_duration(
    consent_manager: ImapConsentManager,
    consent_registry: ConsentRegistry,
):
    """Test consent grant with expiration."""
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        duration_hours=24,
    )

    # Verify in registry
    record = consent_registry.get(
        source="mailbox:test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    assert record is not None
    assert record.granted is True
    assert record.expires_at is not None


# ============================================================================
# Interactive Consent Flow Tests
# ============================================================================


@patch("builtins.input", side_effect=["yes", "yes", "no"])
@patch("builtins.print")
def test_request_mailbox_consent_all_granted(
    mock_print: MagicMock,
    mock_input: MagicMock,
    consent_manager: ImapConsentManager,
):
    """Test interactive consent flow with all scopes granted."""
    required_scopes = [
        ImapConsentScopes.MAILBOX_ACCESS.value,
        ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
        ImapConsentScopes.ATTACHMENT_EXTRACTION.value,
    ]

    results = consent_manager.request_mailbox_consent(
        mailbox_id="test-mailbox",
        email_address="test@example.com",
        required_scopes=required_scopes,
        operator="test_operator",
    )

    # Verify results
    assert results[ImapConsentScopes.MAILBOX_ACCESS.value] is True
    assert results[ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value] is True
    assert results[ImapConsentScopes.ATTACHMENT_EXTRACTION.value] is False

    # Verify consent recorded
    assert consent_manager.check_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    ) is True
    assert consent_manager.check_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    ) is True
    assert consent_manager.check_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.ATTACHMENT_EXTRACTION.value,
    ) is False


@patch("builtins.input", side_effect=["no", "no"])
@patch("builtins.print")
def test_request_mailbox_consent_all_denied(
    mock_print: MagicMock,
    mock_input: MagicMock,
    consent_manager: ImapConsentManager,
):
    """Test interactive consent flow with all scopes denied."""
    required_scopes = [
        ImapConsentScopes.MAILBOX_ACCESS.value,
        ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    ]

    results = consent_manager.request_mailbox_consent(
        mailbox_id="test-mailbox",
        email_address="test@example.com",
        required_scopes=required_scopes,
    )

    # Verify results
    assert results[ImapConsentScopes.MAILBOX_ACCESS.value] is False
    assert results[ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value] is False


@patch("builtins.input", side_effect=["invalid", "yes"])
@patch("builtins.print")
def test_request_mailbox_consent_invalid_input(
    mock_print: MagicMock,
    mock_input: MagicMock,
    consent_manager: ImapConsentManager,
):
    """Test interactive consent flow with invalid input (retry)."""
    required_scopes = [ImapConsentScopes.MAILBOX_ACCESS.value]

    results = consent_manager.request_mailbox_consent(
        mailbox_id="test-mailbox",
        email_address="test@example.com",
        required_scopes=required_scopes,
    )

    # Should have prompted twice (once invalid, once valid)
    assert mock_input.call_count == 2
    assert results[ImapConsentScopes.MAILBOX_ACCESS.value] is True


# ============================================================================
# Audit Logging Tests
# ============================================================================


def test_consent_decisions_logged(
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
):
    """Test that consent decisions are logged."""
    # Grant consent
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        operator="test_operator",
    )

    # Verify audit log exists
    audit_path = audit_logger.output_dir / audit_logger.filename
    assert audit_path.exists()

    # Read audit log
    content = audit_path.read_text()
    assert "consent" in content.lower()
    assert "mailbox:test-mailbox" in content


def test_consent_token_hash_generated(
    consent_manager: ImapConsentManager,
    audit_logger: AuditLogger,
):
    """Test that consent decisions include token hash."""
    consent_manager.grant_consent(
        mailbox_id="test-mailbox",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Read audit log
    audit_path = audit_logger.output_dir / audit_logger.filename
    content = audit_path.read_text()

    # Should contain token_hash field
    assert "consent_token_hash" in content


# ============================================================================
# Scope Description Tests
# ============================================================================


def test_get_scope_description(consent_manager: ImapConsentManager):
    """Test scope descriptions are human-readable."""
    desc = consent_manager._get_scope_description(
        ImapConsentScopes.MAILBOX_ACCESS.value
    )

    assert "MAILBOX ACCESS" in desc
    assert len(desc) > 20  # Should be descriptive


def test_get_scope_description_unknown(consent_manager: ImapConsentManager):
    """Test unknown scope description."""
    desc = consent_manager._get_scope_description("unknown:scope")

    assert "UNKNOWN SCOPE" in desc
    assert "unknown:scope" in desc


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_consent_lifecycle(consent_manager: ImapConsentManager):
    """Test complete consent lifecycle."""
    mailbox_id = "test-mailbox"
    scope = ImapConsentScopes.MAILBOX_ACCESS.value

    # Step 1: Check consent (should be False)
    assert consent_manager.check_consent(mailbox_id=mailbox_id, scope=scope) is False

    # Step 2: Grant consent
    consent_manager.grant_consent(mailbox_id=mailbox_id, scope=scope)

    # Step 3: Check consent (should be True)
    assert consent_manager.check_consent(mailbox_id=mailbox_id, scope=scope) is True

    # Step 4: Require consent (should not raise)
    consent_manager.require_consent(mailbox_id=mailbox_id, scope=scope)

    # Step 5: Revoke consent
    consent_manager.revoke_consent(mailbox_id=mailbox_id, scope=scope)

    # Step 6: Check consent (should be False)
    assert consent_manager.check_consent(mailbox_id=mailbox_id, scope=scope) is False

    # Step 7: Require consent (should raise)
    with pytest.raises(ConsentRequiredError):
        consent_manager.require_consent(mailbox_id=mailbox_id, scope=scope)


def test_multiple_scopes_independent(consent_manager: ImapConsentManager):
    """Test that different scopes are independent."""
    mailbox_id = "test-mailbox"

    # Grant mailbox access
    consent_manager.grant_consent(
        mailbox_id=mailbox_id,
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Check both scopes
    assert (
        consent_manager.check_consent(
            mailbox_id=mailbox_id,
            scope=ImapConsentScopes.MAILBOX_ACCESS.value,
        )
        is True
    )
    assert (
        consent_manager.check_consent(
            mailbox_id=mailbox_id,
            scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
        )
        is False
    )


def test_multiple_mailboxes_independent(consent_manager: ImapConsentManager):
    """Test that different mailboxes are independent."""
    scope = ImapConsentScopes.MAILBOX_ACCESS.value

    # Grant for mailbox1
    consent_manager.grant_consent(mailbox_id="mailbox1", scope=scope)

    # Check both mailboxes
    assert consent_manager.check_consent(mailbox_id="mailbox1", scope=scope) is True
    assert consent_manager.check_consent(mailbox_id="mailbox2", scope=scope) is False

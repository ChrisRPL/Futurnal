"""Comprehensive tests for email header redaction.

Tests cover:
- Email address redaction (hash + domain)
- Subject line redaction
- Full EmailMessage redaction
- Hash stability (same input → same hash)
- Privacy keyword detection
- Different privacy levels
- Edge cases (malformed emails, missing fields)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from futurnal.ingestion.imap.email_parser import EmailAddress, EmailMessage
from futurnal.ingestion.imap.email_redaction import (
    EmailHeaderRedactionPolicy,
    create_redaction_policy_for_privacy_level,
)


# ============================================================================
# Email Address Redaction Tests
# ============================================================================


def test_redact_email_address_basic():
    """Test basic email address redaction."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_address("john.doe@example.com")

    # Should preserve domain
    assert "@example.com" in redacted

    # Should not contain original local part
    assert "john.doe" not in redacted

    # Should have hash prefix
    parts = redacted.split("@")
    assert len(parts) == 2
    assert len(parts[0]) == 8  # Default hash length


def test_redact_email_address_preserves_domain():
    """Test that domain is preserved for debugging."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_address("user@gmail.com")

    assert redacted.endswith("@gmail.com")


def test_redact_email_address_hash_stable():
    """Test that same email produces same hash."""
    policy = EmailHeaderRedactionPolicy()

    redacted1 = policy.redact_email_address("test@example.com")
    redacted2 = policy.redact_email_address("test@example.com")

    assert redacted1 == redacted2


def test_redact_email_address_different_emails():
    """Test that different emails produce different hashes."""
    policy = EmailHeaderRedactionPolicy()

    redacted1 = policy.redact_email_address("user1@example.com")
    redacted2 = policy.redact_email_address("user2@example.com")

    assert redacted1 != redacted2


def test_redact_email_address_malformed():
    """Test redaction of malformed email (no @)."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_address("notanemail")

    # Should return just hash
    assert "@" not in redacted
    assert len(redacted) == 8


def test_redact_email_address_custom_hash_length():
    """Test custom hash length."""
    policy = EmailHeaderRedactionPolicy(hash_length=12)

    redacted = policy.redact_email_address("user@example.com")

    hash_part = redacted.split("@")[0]
    assert len(hash_part) == 12


# ============================================================================
# Subject Line Redaction Tests
# ============================================================================


def test_redact_subject_basic():
    """Test subject line redaction."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_subject("Confidential Meeting")

    # Should be hashed
    assert len(redacted) == 8
    assert "Confidential" not in redacted
    assert "Meeting" not in redacted


def test_redact_subject_hash_stable():
    """Test subject hash stability."""
    policy = EmailHeaderRedactionPolicy()

    redacted1 = policy.redact_subject("Same Subject")
    redacted2 = policy.redact_subject("Same Subject")

    assert redacted1 == redacted2


def test_redact_subject_different_subjects():
    """Test different subjects produce different hashes."""
    policy = EmailHeaderRedactionPolicy()

    redacted1 = policy.redact_subject("Subject 1")
    redacted2 = policy.redact_subject("Subject 2")

    assert redacted1 != redacted2


# ============================================================================
# Email Message Redaction Tests
# ============================================================================


@pytest.fixture
def sample_email_message() -> EmailMessage:
    """Create sample EmailMessage for testing."""
    return EmailMessage(
        message_id="<test@example.com>",
        uid=1,
        folder="INBOX",
        subject="Test Subject",
        from_address=EmailAddress(address="sender@example.com", display_name="Sender"),
        to_addresses=[
            EmailAddress(address="recipient1@example.com", display_name="Recipient 1"),
            EmailAddress(address="recipient2@example.com", display_name="Recipient 2"),
        ],
        cc_addresses=[EmailAddress(address="cc@example.com", display_name="CC User")],
        bcc_addresses=[],
        reply_to_addresses=[],
        date=datetime(2024, 1, 1, 12, 0, 0),
        in_reply_to=None,
        references=[],
        body_plain="Test body content",
        body_html=None,
        body_normalized="Test body content",
        size_bytes=1000,
        flags=["\\Seen"],
        labels=[],
        attachments=[],
        contains_sensitive_keywords=False,
        privacy_classification="standard",
        retrieved_at=datetime(2024, 1, 1, 12, 0, 0),
        mailbox_id="test-mailbox",
    )


def test_redact_email_message_sender_redacted(sample_email_message: EmailMessage):
    """Test email message redaction with sender redacted."""
    policy = EmailHeaderRedactionPolicy(redact_sender=True, redact_recipients=False)

    redacted = policy.redact_email_message(sample_email_message)

    # Sender should be redacted
    assert "@example.com" in redacted["from"]
    assert "sender" not in redacted["from"].lower()

    # Recipients should not be redacted
    assert redacted["to"] == ["recipient1@example.com", "recipient2@example.com"]
    assert redacted["cc"] == ["cc@example.com"]


def test_redact_email_message_recipients_redacted(sample_email_message: EmailMessage):
    """Test email message redaction with recipients redacted."""
    policy = EmailHeaderRedactionPolicy(redact_sender=False, redact_recipients=True)

    redacted = policy.redact_email_message(sample_email_message)

    # Sender should not be redacted
    assert redacted["from"] == "sender@example.com"

    # Recipients should be counts only
    assert redacted["to_count"] == 2
    assert redacted["cc_count"] == 1
    assert "to" not in redacted
    assert "cc" not in redacted


def test_redact_email_message_subject_not_redacted(sample_email_message: EmailMessage):
    """Test email message with subject not redacted (default)."""
    policy = EmailHeaderRedactionPolicy(redact_subject=False)

    redacted = policy.redact_email_message(sample_email_message)

    # Subject should be plaintext
    assert redacted["subject"] == "Test Subject"
    assert "subject_hash" not in redacted


def test_redact_email_message_subject_redacted(sample_email_message: EmailMessage):
    """Test email message with subject redacted."""
    policy = EmailHeaderRedactionPolicy(redact_subject=True)

    redacted = policy.redact_email_message(sample_email_message)

    # Subject should be hashed
    assert "subject_hash" in redacted
    assert len(redacted["subject_hash"]) == 8
    assert "subject" not in redacted


def test_redact_email_message_no_body_content(sample_email_message: EmailMessage):
    """Test that email body is NEVER included in redacted output."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_message(sample_email_message)

    # Body should NEVER be in redacted output
    assert "body_plain" not in redacted
    assert "body_html" not in redacted
    assert "body_normalized" not in redacted
    assert "Test body content" not in str(redacted)


def test_redact_email_message_metadata_included(sample_email_message: EmailMessage):
    """Test that metadata is included in redacted output."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_message(sample_email_message)

    # Metadata should be present
    assert "message_id_hash" in redacted
    assert "folder" in redacted
    assert "date" in redacted
    assert "size_bytes" in redacted
    assert "has_attachments" in redacted
    assert "attachment_count" in redacted

    # Values should be correct
    assert redacted["folder"] == "INBOX"
    assert redacted["size_bytes"] == 1000
    assert redacted["has_attachments"] is False
    assert redacted["attachment_count"] == 0


def test_redact_email_message_privacy_classification(
    sample_email_message: EmailMessage,
):
    """Test that privacy classification is included."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_message(sample_email_message)

    assert "contains_sensitive" in redacted
    assert "privacy_classification" in redacted
    assert redacted["privacy_classification"] == "standard"


# ============================================================================
# Email List Redaction Tests
# ============================================================================


def test_redact_email_list():
    """Test redacting a list of email addresses."""
    policy = EmailHeaderRedactionPolicy()

    emails = ["user1@example.com", "user2@example.com", "user3@example.com"]

    redacted = policy.redact_email_list(emails)

    assert len(redacted) == 3
    for redacted_email in redacted:
        assert "@example.com" in redacted_email
        assert "user" not in redacted_email.lower()


def test_redact_email_list_empty():
    """Test redacting empty email list."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_list([])

    assert redacted == []


# ============================================================================
# Sensitive Subject Detection Tests
# ============================================================================


def test_check_sensitive_subject_no_keywords():
    """Test sensitive subject check with no keywords."""
    policy = EmailHeaderRedactionPolicy()

    is_sensitive = policy.check_sensitive_subject("Normal Subject", None)

    assert is_sensitive is False


def test_check_sensitive_subject_no_match():
    """Test sensitive subject check with keywords but no match."""
    policy = EmailHeaderRedactionPolicy()

    keywords = ["confidential", "private", "secret"]
    is_sensitive = policy.check_sensitive_subject("Normal Subject", keywords)

    assert is_sensitive is False


def test_check_sensitive_subject_match():
    """Test sensitive subject check with keyword match."""
    policy = EmailHeaderRedactionPolicy()

    keywords = ["confidential", "private", "secret"]
    is_sensitive = policy.check_sensitive_subject(
        "Confidential: Q4 Results", keywords
    )

    assert is_sensitive is True


def test_check_sensitive_subject_case_insensitive():
    """Test sensitive subject check is case-insensitive."""
    policy = EmailHeaderRedactionPolicy()

    keywords = ["confidential"]
    is_sensitive = policy.check_sensitive_subject("CONFIDENTIAL MEETING", keywords)

    assert is_sensitive is True


def test_check_sensitive_subject_partial_match():
    """Test sensitive subject check with partial keyword match."""
    policy = EmailHeaderRedactionPolicy()

    keywords = ["private"]
    is_sensitive = policy.check_sensitive_subject(
        "This is a private matter", keywords
    )

    assert is_sensitive is True


# ============================================================================
# Privacy Level Policy Creation Tests
# ============================================================================


def test_create_policy_strict():
    """Test creating strict privacy policy."""
    policy = create_redaction_policy_for_privacy_level("strict")

    assert policy.redact_sender is True
    assert policy.redact_recipients is True
    assert policy.redact_subject_enabled is True


def test_create_policy_standard():
    """Test creating standard privacy policy."""
    policy = create_redaction_policy_for_privacy_level(
        "standard",
        enable_sender_anonymization=True,
        enable_recipient_anonymization=False,
        enable_subject_redaction=False,
    )

    assert policy.redact_sender is True
    assert policy.redact_recipients is False
    assert policy.redact_subject_enabled is False


def test_create_policy_permissive():
    """Test creating permissive privacy policy."""
    policy = create_redaction_policy_for_privacy_level("permissive")

    assert policy.redact_sender is False
    assert policy.redact_recipients is False
    assert policy.redact_subject_enabled is False


# ============================================================================
# Edge Cases Tests
# ============================================================================


def test_redact_email_with_special_characters():
    """Test redacting email with special characters."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_email_address("user+tag@example.com")

    assert "@example.com" in redacted
    assert "user+tag" not in redacted


def test_redact_empty_subject():
    """Test redacting empty subject."""
    policy = EmailHeaderRedactionPolicy()

    redacted = policy.redact_subject("")

    assert len(redacted) == 8  # Should still produce hash


def test_redact_message_with_no_recipients(sample_email_message: EmailMessage):
    """Test redacting message with no recipients."""
    sample_email_message.to_addresses = []
    sample_email_message.cc_addresses = []

    policy = EmailHeaderRedactionPolicy(redact_recipients=True)

    redacted = policy.redact_email_message(sample_email_message)

    assert redacted["to_count"] == 0
    assert redacted["cc_count"] == 0


def test_redact_message_with_none_subject(sample_email_message: EmailMessage):
    """Test redacting message with None subject."""
    sample_email_message.subject = None

    policy = EmailHeaderRedactionPolicy(redact_subject=False)

    redacted = policy.redact_email_message(sample_email_message)

    assert redacted["subject"] is None


# ============================================================================
# Hash Consistency Tests
# ============================================================================


def test_hash_function_consistent():
    """Test that internal hash function is consistent."""
    policy1 = EmailHeaderRedactionPolicy()
    policy2 = EmailHeaderRedactionPolicy()

    hash1 = policy1._hash("test@example.com")
    hash2 = policy2._hash("test@example.com")

    assert hash1 == hash2


def test_hash_function_different_inputs():
    """Test that different inputs produce different hashes."""
    policy = EmailHeaderRedactionPolicy()

    hash1 = policy._hash("input1")
    hash2 = policy._hash("input2")

    assert hash1 != hash2


def test_hash_length_respected():
    """Test that hash length parameter is respected."""
    policy = EmailHeaderRedactionPolicy(hash_length=16)

    redacted = policy.redact_email_address("user@example.com")
    hash_part = redacted.split("@")[0]

    assert len(hash_part) == 16

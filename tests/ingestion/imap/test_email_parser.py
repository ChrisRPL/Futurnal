"""Comprehensive tests for email parser and models.

Tests cover:
- EmailAddress parsing with/without display names
- RFC822/MIME message parsing
- Header extraction (subject, from, to, cc, date, threading)
- Body extraction (plain, HTML, multipart)
- Attachment metadata extraction
- Character encoding handling
- Privacy keyword detection
- Edge cases (missing headers, malformed data, encoding errors)
"""

from __future__ import annotations

import email
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from pathlib import Path

import pytest

from futurnal.ingestion.imap.email_parser import (
    AttachmentMetadata,
    EmailAddress,
    EmailMessage,
    EmailParser,
)
from futurnal.ingestion.imap.descriptor import MailboxPrivacySettings
from futurnal.privacy.audit import AuditLogger


# ============================================================================
# EmailAddress Tests
# ============================================================================


def test_email_address_simple():
    """Test simple email address parsing."""
    addr = EmailAddress(address="john@example.com")
    assert addr.address == "john@example.com"
    assert addr.display_name is None


def test_email_address_with_display_name():
    """Test email address with display name."""
    addr = EmailAddress(address="john@example.com", display_name="John Doe")
    assert addr.address == "john@example.com"
    assert addr.display_name == "John Doe"


def test_email_address_validation_missing_at():
    """Test email address validation rejects missing @."""
    with pytest.raises(ValueError, match="Invalid email address"):
        EmailAddress(address="invalid-email")


def test_email_address_validation_multiple_at():
    """Test email address validation rejects multiple @."""
    with pytest.raises(ValueError, match="Invalid email address"):
        EmailAddress(address="invalid@@example.com")


def test_email_address_from_header_single():
    """Test parsing single address from header."""
    addrs = EmailAddress.from_header("john@example.com")
    assert len(addrs) == 1
    assert addrs[0].address == "john@example.com"
    assert addrs[0].display_name is None


def test_email_address_from_header_with_display_name():
    """Test parsing address with display name."""
    addrs = EmailAddress.from_header("John Doe <john@example.com>")
    assert len(addrs) == 1
    assert addrs[0].address == "john@example.com"
    assert addrs[0].display_name == "John Doe"


def test_email_address_from_header_multiple():
    """Test parsing multiple addresses from header."""
    addrs = EmailAddress.from_header(
        "John Doe <john@example.com>, Jane Smith <jane@example.com>"
    )
    assert len(addrs) == 2
    assert addrs[0].address == "john@example.com"
    assert addrs[0].display_name == "John Doe"
    assert addrs[1].address == "jane@example.com"
    assert addrs[1].display_name == "Jane Smith"


def test_email_address_from_header_mixed_format():
    """Test parsing mixed format addresses."""
    addrs = EmailAddress.from_header(
        "John Doe <john@example.com>, jane@example.com"
    )
    assert len(addrs) == 2
    assert addrs[0].address == "john@example.com"
    assert addrs[0].display_name == "John Doe"
    assert addrs[1].address == "jane@example.com"
    assert addrs[1].display_name is None


def test_email_address_from_header_empty():
    """Test parsing empty header returns empty list."""
    addrs = EmailAddress.from_header("")
    assert len(addrs) == 0


def test_email_address_from_header_whitespace_only():
    """Test parsing whitespace-only header returns empty list."""
    addrs = EmailAddress.from_header("   ")
    assert len(addrs) == 0


# ============================================================================
# EmailParser Basic Tests
# ============================================================================


def _create_simple_email(
    subject="Test Subject",
    from_addr="sender@example.com",
    to_addr="recipient@example.com",
    body="Test body",
    message_id="<test123@example.com>",
) -> bytes:
    """Create a simple test email message."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Message-ID"] = message_id
    msg["Date"] = email.utils.formatdate(localtime=True)
    return msg.as_bytes()


def test_parse_simple_email():
    """Test parsing a simple plain text email."""
    raw_message = _create_simple_email()
    parser = EmailParser()

    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.message_id == "test123@example.com"
    assert email_msg.subject == "Test Subject"
    assert email_msg.from_address.address == "sender@example.com"
    assert len(email_msg.to_addresses) == 1
    assert email_msg.to_addresses[0].address == "recipient@example.com"
    assert email_msg.body_plain == "Test body"
    assert email_msg.body_normalized == "Test body"
    assert email_msg.uid == 1
    assert email_msg.folder == "INBOX"
    assert email_msg.mailbox_id == "test-mailbox"


def test_parse_email_with_cc():
    """Test parsing email with Cc recipients."""
    msg = MIMEText("Test body")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Cc"] = "cc1@example.com, cc2@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert len(email_msg.cc_addresses) == 2
    assert email_msg.cc_addresses[0].address == "cc1@example.com"
    assert email_msg.cc_addresses[1].address == "cc2@example.com"


# ============================================================================
# Threading Header Tests
# ============================================================================


def test_parse_email_with_threading_headers():
    """Test parsing email with In-Reply-To and References."""
    msg = MIMEText("Reply body")
    msg["Subject"] = "Re: Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<reply123@example.com>"
    msg["In-Reply-To"] = "<original123@example.com>"
    msg["References"] = "<thread1@example.com> <original123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.in_reply_to == "original123@example.com"
    assert len(email_msg.references) == 2
    assert email_msg.references[0] == "thread1@example.com"
    assert email_msg.references[1] == "original123@example.com"
    assert email_msg.is_reply is True


def test_parse_email_without_threading_headers():
    """Test email without threading headers."""
    raw_message = _create_simple_email()
    parser = EmailParser()

    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.in_reply_to is None
    assert len(email_msg.references) == 0
    assert email_msg.is_reply is False


# ============================================================================
# Message-ID Tests
# ============================================================================


def test_parse_email_missing_message_id():
    """Test fallback Message-ID generation for messages without Message-ID."""
    msg = MIMEText("Test body")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Date"] = email.utils.formatdate(localtime=True)
    # No Message-ID header

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.message_id.startswith("generated-")
    assert "@futurnal.local" in email_msg.message_id


# ============================================================================
# Subject Decoding Tests
# ============================================================================


def test_parse_email_with_encoded_subject():
    """Test decoding RFC 2047 encoded subject."""
    msg = MIMEText("Test body")
    msg["Subject"] = "=?utf-8?B?VGVzdCBTdWJqZWN0?="  # "Test Subject" in base64
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.subject == "Test Subject"


def test_parse_email_with_non_ascii_subject():
    """Test parsing subject with non-ASCII characters."""
    msg = MIMEText("Test body", _charset="utf-8")
    msg["Subject"] = "Tëst Sübject 日本語"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert "Tëst Sübject" in email_msg.subject or "日本語" in email_msg.subject


def test_parse_email_missing_subject():
    """Test parsing email without subject."""
    msg = MIMEText("Test body")
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)
    # No Subject header

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.subject is None


# ============================================================================
# Multipart MIME Tests
# ============================================================================


def test_parse_multipart_email_plain_and_html():
    """Test parsing multipart email with plain and HTML parts."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    plain_part = MIMEText("Plain text body", "plain")
    html_part = MIMEText("<html><body>HTML body</body></html>", "html")

    msg.attach(plain_part)
    msg.attach(html_part)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.body_plain == "Plain text body"
    assert "<html>" in email_msg.body_html
    assert email_msg.body_normalized == "Plain text body"  # Prefers plain


def test_parse_html_only_email():
    """Test parsing email with only HTML body."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    html_part = MIMEText("<html><body><p>HTML body</p></body></html>", "html")
    msg.attach(html_part)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.body_plain is None
    assert "<html>" in email_msg.body_html
    assert "HTML body" in email_msg.body_normalized  # Converted from HTML


# ============================================================================
# Attachment Tests
# ============================================================================


def test_parse_email_with_attachment():
    """Test parsing email with attachment."""
    msg = MIMEMultipart()
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    # Add body
    body_part = MIMEText("Email body")
    msg.attach(body_part)

    # Add attachment
    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(b"Test file content")
    attachment.add_header(
        "Content-Disposition", "attachment", filename="test.txt"
    )
    msg.attach(attachment)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.has_attachments is True
    assert len(email_msg.attachments) == 1
    assert email_msg.attachments[0].filename == "test.txt"
    assert email_msg.attachments[0].content_type == "application/octet-stream"
    assert email_msg.attachments[0].is_inline is False
    assert email_msg.attachments[0].size_bytes > 0


def test_parse_email_with_inline_image():
    """Test parsing email with inline image."""
    msg = MIMEMultipart()
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    # Add body
    body_part = MIMEText("Email body")
    msg.attach(body_part)

    # Add inline image
    image = MIMEBase("image", "png")
    image.set_payload(b"Fake PNG data")
    image.add_header("Content-Disposition", "inline", filename="image.png")
    image.add_header("Content-ID", "<image1@example.com>")
    msg.attach(image)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.has_attachments is True
    assert len(email_msg.attachments) == 1
    assert email_msg.attachments[0].filename == "image.png"
    assert email_msg.attachments[0].is_inline is True
    assert email_msg.attachments[0].content_id == "image1@example.com"


# ============================================================================
# Privacy Tests
# ============================================================================


def test_privacy_keyword_detection():
    """Test detection of privacy keywords in subject and body."""
    privacy_settings = MailboxPrivacySettings(
        privacy_subject_keywords=["confidential", "private", "nda"]
    )

    raw_message = _create_simple_email(
        subject="Confidential Meeting Notes",
        body="This is a private message",
    )

    parser = EmailParser(privacy_policy=privacy_settings)
    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.contains_sensitive_keywords is True
    assert email_msg.privacy_classification == "sensitive"


def test_no_privacy_keywords():
    """Test message without privacy keywords."""
    privacy_settings = MailboxPrivacySettings(
        privacy_subject_keywords=["confidential", "private", "nda"]
    )

    raw_message = _create_simple_email(
        subject="Regular Meeting",
        body="This is a regular message",
    )

    parser = EmailParser(privacy_policy=privacy_settings)
    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.contains_sensitive_keywords is False
    assert email_msg.privacy_classification == "standard"


def test_audit_logging_without_content(tmp_path: Path):
    """Test audit logging doesn't include email content."""
    audit_logger = AuditLogger(output_dir=tmp_path / "audit")

    raw_message = _create_simple_email(
        subject="Secret Meeting",
        body="Secret body content",
    )

    parser = EmailParser(audit_logger=audit_logger)
    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    # Read audit log
    audit_file = tmp_path / "audit" / "audit.log"
    assert audit_file.exists()

    audit_content = audit_file.read_text()

    # Verify sensitive data is NOT in audit log
    assert "Secret Meeting" not in audit_content
    assert "Secret body content" not in audit_content

    # Verify metadata IS in audit log
    assert "email_parsed" in audit_content
    assert "INBOX" in audit_content


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_parse_email_missing_from_header():
    """Test parsing email with missing From header."""
    msg = MIMEText("Test body")
    msg["Subject"] = "Test"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)
    # No From header

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.from_address.address == "unknown@unknown"


def test_parse_email_missing_date():
    """Test parsing email with missing Date header falls back to current time."""
    msg = MIMEText("Test body")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    # No Date header

    parser = EmailParser()
    before = datetime.utcnow()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )
    after = datetime.utcnow()

    # Date should be between before and after
    assert before <= email_msg.date <= after


def test_parse_empty_body():
    """Test parsing email with empty body."""
    msg = MIMEText("")
    msg["Subject"] = "Test"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    parser = EmailParser()
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    assert email_msg.body_plain == ""
    assert email_msg.body_normalized == ""


def test_email_message_properties():
    """Test EmailMessage computed properties."""
    raw_message = _create_simple_email()
    parser = EmailParser()

    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
    )

    # Test properties
    assert email_msg.has_attachments is False
    assert email_msg.participant_count == 2  # 1 from + 1 to
    assert email_msg.is_reply is False


def test_email_with_flags_and_labels():
    """Test parsing email with IMAP flags and Gmail labels."""
    raw_message = _create_simple_email()
    parser = EmailParser()

    email_msg = parser.parse_message(
        raw_message=raw_message,
        uid=1,
        folder="INBOX",
        mailbox_id="test-mailbox",
        flags=["\\Seen", "\\Flagged"],
        labels=["Important", "Work"],
    )

    assert email_msg.flags == ["\\Seen", "\\Flagged"]
    assert email_msg.labels == ["Important", "Work"]


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_parsing_pipeline():
    """Test complete parsing pipeline with all features."""
    # Create complex multipart email with all features
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "=?utf-8?B?VGVzdCBTdWJqZWN0?="
    msg["From"] = "John Doe <john@example.com>"
    msg["To"] = "Jane Smith <jane@example.com>, bob@example.com"
    msg["Cc"] = "Alice <alice@example.com>"
    msg["Message-ID"] = "<complex123@example.com>"
    msg["In-Reply-To"] = "<parent456@example.com>"
    msg["References"] = "<thread789@example.com> <parent456@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    # Add plain and HTML parts
    plain_part = MIMEText("Plain text body", "plain")
    html_part = MIMEText("<html><body>HTML body</body></html>", "html")
    msg.attach(plain_part)
    msg.attach(html_part)

    # Add attachment
    attachment = MIMEBase("application", "pdf")
    attachment.set_payload(b"PDF content")
    attachment.add_header("Content-Disposition", "attachment", filename="doc.pdf")
    msg.attach(attachment)

    # Parse with privacy settings and audit logger
    privacy_settings = MailboxPrivacySettings(
        privacy_subject_keywords=["confidential"]
    )

    parser = EmailParser(privacy_policy=privacy_settings)
    email_msg = parser.parse_message(
        raw_message=msg.as_bytes(),
        uid=42,
        folder="INBOX/Work",
        mailbox_id="work-mailbox",
        flags=["\\Seen"],
        labels=["Important"],
    )

    # Verify all components
    assert email_msg.message_id == "complex123@example.com"
    assert email_msg.subject == "Test Subject"
    assert email_msg.from_address.address == "john@example.com"
    assert email_msg.from_address.display_name == "John Doe"
    assert len(email_msg.to_addresses) == 2
    assert len(email_msg.cc_addresses) == 1
    assert email_msg.in_reply_to == "parent456@example.com"
    assert len(email_msg.references) == 2
    assert email_msg.body_plain == "Plain text body"
    assert "<html>" in email_msg.body_html
    assert email_msg.body_normalized == "Plain text body"
    assert email_msg.has_attachments is True
    assert len(email_msg.attachments) == 1
    assert email_msg.attachments[0].filename == "doc.pdf"
    assert email_msg.uid == 42
    assert email_msg.folder == "INBOX/Work"
    assert email_msg.mailbox_id == "work-mailbox"
    assert email_msg.flags == ["\\Seen"]
    assert email_msg.labels == ["Important"]
    assert email_msg.is_reply is True
    assert email_msg.participant_count == 4  # 1 from + 2 to + 1 cc

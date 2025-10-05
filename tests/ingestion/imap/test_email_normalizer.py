"""Tests for email normalizer.

Tests cover:
- Normalized format generation for Unstructured.io
- Metadata header formatting
- Threading context preservation
- Attachment metadata formatting
- Edge cases (missing fields, empty content)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from futurnal.ingestion.imap.email_parser import (
    AttachmentMetadata,
    EmailAddress,
    EmailMessage,
)
from futurnal.ingestion.imap.email_normalizer import EmailNormalizer


def _create_test_email_message(**overrides) -> EmailMessage:
    """Create a test EmailMessage with default values."""
    defaults = {
        "message_id": "test123@example.com",
        "uid": 1,
        "folder": "INBOX",
        "subject": "Test Subject",
        "from_address": EmailAddress(
            address="sender@example.com", display_name="John Doe"
        ),
        "to_addresses": [
            EmailAddress(address="recipient@example.com", display_name="Jane Smith")
        ],
        "cc_addresses": [],
        "bcc_addresses": [],
        "reply_to_addresses": [],
        "date": datetime(2024, 1, 15, 10, 30, 0),
        "in_reply_to": None,
        "references": [],
        "body_plain": "Test email body",
        "body_html": None,
        "body_normalized": "Test email body",
        "size_bytes": 1024,
        "flags": [],
        "labels": [],
        "attachments": [],
        "contains_sensitive_keywords": False,
        "privacy_classification": "standard",
        "retrieved_at": datetime(2024, 1, 15, 10, 35, 0),
        "mailbox_id": "test-mailbox",
    }
    defaults.update(overrides)
    return EmailMessage(**defaults)


def test_normalize_simple_email():
    """Test normalizing a simple email message."""
    email_msg = _create_test_email_message()
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    # Verify header section
    assert "From: sender@example.com" in normalized
    assert "From Name: John Doe" in normalized
    assert "To: recipient@example.com" in normalized
    assert "Date: 2024-01-15T10:30:00" in normalized
    assert "Subject: Test Subject" in normalized

    # Verify separator
    assert "\n---\n" in normalized

    # Verify body
    assert "Test email body" in normalized


def test_normalize_email_without_display_name():
    """Test normalizing email with sender without display name."""
    email_msg = _create_test_email_message(
        from_address=EmailAddress(address="sender@example.com", display_name=None)
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "From: sender@example.com" in normalized
    assert "From Name:" not in normalized


def test_normalize_email_with_multiple_recipients():
    """Test normalizing email with multiple To and Cc recipients."""
    email_msg = _create_test_email_message(
        to_addresses=[
            EmailAddress(address="recipient1@example.com"),
            EmailAddress(address="recipient2@example.com"),
        ],
        cc_addresses=[
            EmailAddress(address="cc1@example.com"),
            EmailAddress(address="cc2@example.com"),
        ],
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "To: recipient1@example.com, recipient2@example.com" in normalized
    assert "Cc: cc1@example.com, cc2@example.com" in normalized


def test_normalize_email_without_cc():
    """Test normalizing email without Cc recipients."""
    email_msg = _create_test_email_message(cc_addresses=[])
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "Cc:" not in normalized


def test_normalize_email_with_threading():
    """Test normalizing email with threading headers."""
    email_msg = _create_test_email_message(
        in_reply_to="parent123@example.com",
        references=["thread1@example.com", "parent123@example.com"],
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "In-Reply-To: parent123@example.com" in normalized
    assert "References: thread1@example.com, parent123@example.com" in normalized


def test_normalize_email_without_threading():
    """Test normalizing email without threading headers."""
    email_msg = _create_test_email_message(
        in_reply_to=None,
        references=[],
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "In-Reply-To:" not in normalized
    assert "References:" not in normalized


def test_normalize_email_without_subject():
    """Test normalizing email without subject."""
    email_msg = _create_test_email_message(subject=None)
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "Subject:" not in normalized


def test_normalize_email_with_attachments():
    """Test normalizing email with attachments."""
    email_msg = _create_test_email_message(
        attachments=[
            AttachmentMetadata(
                filename="document.pdf",
                content_type="application/pdf",
                size_bytes=1024 * 512,  # 512 KB
                part_id="1",
                is_inline=False,
            ),
            AttachmentMetadata(
                filename="image.png",
                content_type="image/png",
                size_bytes=1024 * 1024 * 2,  # 2 MB
                part_id="2",
                is_inline=True,
                content_id="image1@example.com",
            ),
        ]
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "Attachments:" in normalized
    assert "document.pdf (application/pdf, 512.0 KB)" in normalized
    assert "image.png (image/png, 2.0 MB) (inline)" in normalized


def test_normalize_email_without_attachments():
    """Test normalizing email without attachments."""
    email_msg = _create_test_email_message(attachments=[])
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "Attachments:" not in normalized


def test_format_size_bytes():
    """Test human-readable size formatting."""
    normalizer = EmailNormalizer()

    assert normalizer._format_size(500) == "500 bytes"
    assert normalizer._format_size(1024) == "1.0 KB"
    assert normalizer._format_size(1024 * 1024) == "1.0 MB"
    assert normalizer._format_size(1024 * 1024 * 1024) == "1.0 GB"
    assert normalizer._format_size(1536) == "1.5 KB"  # 1.5 KB
    assert normalizer._format_size(1024 * 1024 * 2) == "2.0 MB"  # 2 MB


def test_normalize_email_empty_body():
    """Test normalizing email with empty body."""
    email_msg = _create_test_email_message(
        body_plain="",
        body_html=None,
        body_normalized="",
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    # Should still have headers and separator
    assert "From: sender@example.com" in normalized
    assert "\n---\n" in normalized


def test_normalize_email_with_html_body():
    """Test normalizing email where body_normalized comes from HTML."""
    email_msg = _create_test_email_message(
        body_plain=None,
        body_html="<html><body><p>HTML content</p></body></html>",
        body_normalized="HTML content",
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    assert "HTML content" in normalized


def test_normalize_preserves_structure():
    """Test that normalized format has correct structure."""
    email_msg = _create_test_email_message(
        subject="Important Meeting",
        from_address=EmailAddress(
            address="boss@example.com", display_name="The Boss"
        ),
        to_addresses=[EmailAddress(address="team@example.com")],
        cc_addresses=[EmailAddress(address="hr@example.com")],
        in_reply_to="previous@example.com",
        references=["thread@example.com", "previous@example.com"],
        body_normalized="Meeting notes here",
        attachments=[
            AttachmentMetadata(
                filename="agenda.pdf",
                content_type="application/pdf",
                size_bytes=2048,
                part_id="1",
            )
        ],
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    lines = normalized.split("\n")

    # Find separator position
    separator_idx = lines.index("---")

    # Headers should be before separator
    headers_section = "\n".join(lines[:separator_idx])
    assert "From: boss@example.com" in headers_section
    assert "From Name: The Boss" in headers_section
    assert "To: team@example.com" in headers_section
    assert "Cc: hr@example.com" in headers_section
    assert "Subject: Important Meeting" in headers_section
    assert "In-Reply-To: previous@example.com" in headers_section
    assert "References:" in headers_section

    # Body should be after separator
    body_section = "\n".join(lines[separator_idx + 1 :])
    assert "Meeting notes here" in body_section
    assert "Attachments:" in body_section
    assert "agenda.pdf" in body_section


def test_normalize_full_email():
    """Test normalizing a complete email with all features."""
    email_msg = _create_test_email_message(
        message_id="full-test@example.com",
        subject="Re: Project Update",
        from_address=EmailAddress(
            address="alice@example.com", display_name="Alice Johnson"
        ),
        to_addresses=[
            EmailAddress(address="bob@example.com", display_name="Bob Smith"),
            EmailAddress(address="charlie@example.com"),
        ],
        cc_addresses=[EmailAddress(address="manager@example.com")],
        date=datetime(2024, 3, 20, 14, 30, 0),
        in_reply_to="original@example.com",
        references=["thread1@example.com", "original@example.com"],
        body_normalized="Here are the project updates...\n\nBest regards,\nAlice",
        attachments=[
            AttachmentMetadata(
                filename="report.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size_bytes=1024 * 256,  # 256 KB
                part_id="1",
            ),
            AttachmentMetadata(
                filename="chart.png",
                content_type="image/png",
                size_bytes=1024 * 128,  # 128 KB
                part_id="2",
                is_inline=True,
                content_id="chart1",
            ),
        ],
    )
    normalizer = EmailNormalizer()

    normalized = normalizer.normalize(email_msg)

    # Verify all components present
    assert "From: alice@example.com" in normalized
    assert "From Name: Alice Johnson" in normalized
    assert "To: bob@example.com, charlie@example.com" in normalized
    assert "Cc: manager@example.com" in normalized
    assert "Date: 2024-03-20T14:30:00" in normalized
    assert "Subject: Re: Project Update" in normalized
    assert "In-Reply-To: original@example.com" in normalized
    assert "References: thread1@example.com, original@example.com" in normalized
    assert "---" in normalized
    assert "Here are the project updates..." in normalized
    assert "Best regards,\nAlice" in normalized
    assert "Attachments:" in normalized
    assert "report.docx" in normalized
    assert "256.0 KB" in normalized
    assert "chart.png" in normalized
    assert "128.0 KB" in normalized
    assert "(inline)" in normalized


def test_normalizer_initialization():
    """Test EmailNormalizer can be initialized."""
    normalizer = EmailNormalizer()
    assert normalizer is not None

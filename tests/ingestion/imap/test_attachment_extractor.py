"""Comprehensive tests for attachment extraction and storage.

Tests cover:
- MIME attachment extraction from multipart messages
- Size limit enforcement
- Format filtering (supported/unsupported extensions)
- Content hash calculation and deduplication
- Storage with hash-based filenames
- Inline image detection
- Edge cases (missing filename, empty content, corrupted data)
- Privacy-aware audit logging
"""

from __future__ import annotations

import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from pathlib import Path

import pytest

from futurnal.ingestion.imap.attachment_extractor import AttachmentExtractor
from futurnal.ingestion.imap.attachment_models import (
    EmailAttachment,
    AttachmentProcessingStatus,
)
from futurnal.privacy.audit import AuditLogger


# ============================================================================
# Test Email Creation Helpers
# ============================================================================


def _create_email_with_attachment(
    filename="test.pdf",
    content=b"Test PDF content",
    content_type="application/pdf",
    inline=False,
) -> bytes:
    """Create a test email with an attachment."""
    msg = MIMEMultipart()
    msg["Subject"] = "Test with attachment"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    # Add body
    body = MIMEText("This is the email body")
    msg.attach(body)

    # Add attachment
    attachment = MIMEBase(*content_type.split("/"))
    attachment.set_payload(content)
    email.encoders.encode_base64(attachment)
    attachment.add_header(
        "Content-Disposition",
        "inline" if inline else "attachment",
        filename=filename,
    )
    if inline:
        attachment.add_header("Content-ID", f"<{filename}>")

    msg.attach(attachment)

    return msg.as_bytes()


def _create_email_without_attachment() -> bytes:
    """Create a simple email without attachments."""
    msg = MIMEText("Simple email body")
    msg["Subject"] = "No attachments"
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Message-ID"] = "<noattach@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    return msg.as_bytes()


# ============================================================================
# AttachmentExtractor Basic Tests
# ============================================================================


def test_extractor_initialization(tmp_path: Path):
    """Test extractor initializes with correct settings."""
    storage_dir = tmp_path / "attachments"

    extractor = AttachmentExtractor(
        max_size_bytes=10 * 1024 * 1024,  # 10MB
        storage_dir=storage_dir,
    )

    assert extractor.max_size_bytes == 10 * 1024 * 1024
    assert extractor.storage_dir == storage_dir
    assert storage_dir.exists()
    assert len(extractor.supported_extensions) > 0


def test_extract_pdf_attachment(tmp_path: Path):
    """Test extracting a PDF attachment."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    raw_message = _create_email_with_attachment(
        filename="document.pdf",
        content=b"PDF content here",
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="test123@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    att = attachments[0]

    assert att.filename == "document.pdf"
    assert att.content_type == "application/pdf"
    assert att.size_bytes == 16  # len(b"PDF content here")
    assert att.processing_status == AttachmentProcessingStatus.PENDING
    assert att.content_hash != ""
    assert att.storage_path is not None
    assert att.storage_path.exists()
    assert not att.is_inline
    assert att.mailbox_id == "test-mailbox"


def test_extract_inline_image(tmp_path: Path):
    """Test extracting an inline image attachment."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    raw_message = _create_email_with_attachment(
        filename="image.png",
        content=b"PNG image data",
        content_type="image/png",
        inline=True,
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="test456@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    att = attachments[0]

    assert att.filename == "image.png"
    assert att.is_inline is True
    assert att.content_id == "image.png"
    assert att.processing_status == AttachmentProcessingStatus.PENDING


def test_extract_no_attachments(tmp_path: Path):
    """Test extracting from email without attachments."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    raw_message = _create_email_without_attachment()

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="noattach@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 0


# ============================================================================
# Size Filtering Tests
# ============================================================================


def test_skip_large_attachment(tmp_path: Path):
    """Test that large attachments are skipped."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(
        max_size_bytes=100,  # Very small limit
        storage_dir=storage_dir,
    )

    large_content = b"x" * 1000  # 1KB content
    raw_message = _create_email_with_attachment(
        filename="large.pdf",
        content=large_content,
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="test789@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    att = attachments[0]

    assert att.processing_status == AttachmentProcessingStatus.SKIPPED
    assert "too_large" in att.processing_error
    assert att.storage_path is None
    assert att.content_hash == ""


def test_size_limit_boundary(tmp_path: Path):
    """Test attachment at exact size limit is accepted."""
    storage_dir = tmp_path / "attachments"
    max_size = 1000
    extractor = AttachmentExtractor(
        max_size_bytes=max_size,
        storage_dir=storage_dir,
    )

    content = b"x" * max_size  # Exactly at limit
    raw_message = _create_email_with_attachment(
        filename="boundary.pdf",
        content=content,
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="boundary@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    att = attachments[0]

    assert att.processing_status == AttachmentProcessingStatus.PENDING
    assert att.storage_path is not None
    assert att.storage_path.exists()


# ============================================================================
# Format Filtering Tests
# ============================================================================


def test_skip_unsupported_format(tmp_path: Path):
    """Test that unsupported file formats are skipped."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    raw_message = _create_email_with_attachment(
        filename="data.dat",  # Unsupported extension
        content=b"Binary data",
        content_type="application/octet-stream",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="unsupported@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    att = attachments[0]

    assert att.processing_status == AttachmentProcessingStatus.SKIPPED
    assert "unsupported_format" in att.processing_error
    assert att.storage_path is None


def test_supported_formats(tmp_path: Path):
    """Test various supported file formats are extracted."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    supported_files = [
        ("document.pdf", "application/pdf"),
        ("spreadsheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("image.jpg", "image/jpeg"),
        ("text.txt", "text/plain"),
        ("webpage.html", "text/html"),
    ]

    for filename, content_type in supported_files:
        raw_message = _create_email_with_attachment(
            filename=filename,
            content=b"Test content",
            content_type=content_type,
        )

        attachments = extractor.extract_attachments(
            raw_message=raw_message,
            message_id=f"test-{filename}@example.com",
            mailbox_id="test-mailbox",
        )

        assert len(attachments) == 1
        assert attachments[0].processing_status == AttachmentProcessingStatus.PENDING
        assert attachments[0].storage_path is not None


def test_custom_supported_extensions(tmp_path: Path):
    """Test extractor with custom supported extensions."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(
        storage_dir=storage_dir,
        supported_extensions={'.custom', '.special'},
    )

    # Should accept .custom
    raw_message = _create_email_with_attachment(
        filename="file.custom",
        content=b"Custom data",
        content_type="application/custom",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="custom@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    assert attachments[0].processing_status == AttachmentProcessingStatus.PENDING

    # Should skip .pdf (not in custom list)
    raw_message = _create_email_with_attachment(
        filename="file.pdf",
        content=b"PDF data",
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="pdf@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 1
    assert attachments[0].processing_status == AttachmentProcessingStatus.SKIPPED


# ============================================================================
# Content Hash and Deduplication Tests
# ============================================================================


def test_content_hash_calculation(tmp_path: Path):
    """Test content hash is calculated correctly."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    content = b"Test content for hashing"
    raw_message = _create_email_with_attachment(
        filename="test.pdf",
        content=content,
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="hash@example.com",
        mailbox_id="test-mailbox",
    )

    att = attachments[0]
    expected_hash = EmailAttachment.compute_content_hash(content)

    assert att.content_hash == expected_hash
    assert len(att.content_hash) == 64  # SHA256 is 64 hex chars


def test_deduplication_same_content(tmp_path: Path):
    """Test that identical content is deduplicated."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    content = b"Identical content"

    # Extract first attachment
    raw_message1 = _create_email_with_attachment(
        filename="file1.pdf",
        content=content,
        content_type="application/pdf",
    )

    attachments1 = extractor.extract_attachments(
        raw_message=raw_message1,
        message_id="msg1@example.com",
        mailbox_id="test-mailbox",
    )

    # Extract second attachment with same content
    raw_message2 = _create_email_with_attachment(
        filename="file2.pdf",  # Different filename, same content
        content=content,
        content_type="application/pdf",
    )

    attachments2 = extractor.extract_attachments(
        raw_message=raw_message2,
        message_id="msg2@example.com",
        mailbox_id="test-mailbox",
    )

    # Both should have same hash and storage path
    assert attachments1[0].content_hash == attachments2[0].content_hash
    assert attachments1[0].storage_path == attachments2[0].storage_path

    # Storage file should exist only once
    assert attachments1[0].storage_path.exists()
    assert attachments2[0].storage_path.exists()


def test_different_content_different_storage(tmp_path: Path):
    """Test that different content gets different storage paths."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    raw_message1 = _create_email_with_attachment(
        filename="file1.pdf",
        content=b"Content A",
        content_type="application/pdf",
    )

    raw_message2 = _create_email_with_attachment(
        filename="file2.pdf",
        content=b"Content B",
        content_type="application/pdf",
    )

    attachments1 = extractor.extract_attachments(
        raw_message=raw_message1,
        message_id="msg1@example.com",
        mailbox_id="test-mailbox",
    )

    attachments2 = extractor.extract_attachments(
        raw_message=raw_message2,
        message_id="msg2@example.com",
        mailbox_id="test-mailbox",
    )

    assert attachments1[0].content_hash != attachments2[0].content_hash
    assert attachments1[0].storage_path != attachments2[0].storage_path


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_attachment_without_filename(tmp_path: Path):
    """Test handling attachment without filename."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    msg = MIMEMultipart()
    msg["Message-ID"] = "<nofilename@example.com>"

    attachment = MIMEBase("application", "pdf")
    attachment.set_payload(b"Content")
    # Don't set filename
    attachment.add_header("Content-Disposition", "attachment")
    msg.attach(attachment)

    attachments = extractor.extract_attachments(
        raw_message=msg.as_bytes(),
        message_id="nofilename@example.com",
        mailbox_id="test-mailbox",
    )

    # Should skip attachment without filename
    assert len(attachments) == 0


def test_empty_attachment_content(tmp_path: Path):
    """Test handling empty attachment content."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    raw_message = _create_email_with_attachment(
        filename="empty.pdf",
        content=b"",  # Empty content
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="empty@example.com",
        mailbox_id="test-mailbox",
    )

    # Should skip empty attachment
    assert len(attachments) == 0


def test_multiple_attachments(tmp_path: Path):
    """Test extracting multiple attachments from one email."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    msg = MIMEMultipart()
    msg["Subject"] = "Multiple attachments"
    msg["Message-ID"] = "<multi@example.com>"

    # Add body
    msg.attach(MIMEText("Email body"))

    # Add multiple attachments
    for i in range(3):
        attachment = MIMEBase("application", "pdf")
        attachment.set_payload(f"Content {i}".encode())
        email.encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename=f"file{i}.pdf")
        msg.attach(attachment)

    attachments = extractor.extract_attachments(
        raw_message=msg.as_bytes(),
        message_id="multi@example.com",
        mailbox_id="test-mailbox",
    )

    assert len(attachments) == 3
    filenames = [att.filename for att in attachments]
    assert "file0.pdf" in filenames
    assert "file1.pdf" in filenames
    assert "file2.pdf" in filenames


# ============================================================================
# Storage Tests
# ============================================================================


def test_storage_path_format(tmp_path: Path):
    """Test storage path uses hash-based naming."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    content = b"Test content"
    raw_message = _create_email_with_attachment(
        filename="document.pdf",
        content=content,
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="storage@example.com",
        mailbox_id="test-mailbox",
    )

    att = attachments[0]
    expected_hash = EmailAttachment.compute_content_hash(content)

    # Storage filename should be: {hash}.{extension}
    assert att.storage_path.name == f"{expected_hash}.pdf"
    assert att.storage_path.parent == storage_dir


def test_storage_content_verification(tmp_path: Path):
    """Test stored content matches original."""
    storage_dir = tmp_path / "attachments"
    extractor = AttachmentExtractor(storage_dir=storage_dir)

    content = b"Original content to verify"
    raw_message = _create_email_with_attachment(
        filename="verify.pdf",
        content=content,
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="verify@example.com",
        mailbox_id="test-mailbox",
    )

    att = attachments[0]
    stored_content = att.storage_path.read_bytes()

    assert stored_content == content


# ============================================================================
# Audit Logging Tests
# ============================================================================


def test_audit_logging(tmp_path: Path):
    """Test privacy-aware audit logging."""
    storage_dir = tmp_path / "attachments"
    audit_dir = tmp_path / "audit"
    audit_logger = AuditLogger(output_dir=audit_dir)

    extractor = AttachmentExtractor(
        storage_dir=storage_dir,
        audit_logger=audit_logger,
    )

    raw_message = _create_email_with_attachment(
        filename="audit.pdf",
        content=b"Content for audit test",
        content_type="application/pdf",
    )

    attachments = extractor.extract_attachments(
        raw_message=raw_message,
        message_id="audit@example.com",
        mailbox_id="test-mailbox",
    )

    # Check audit log was created
    audit_log_path = audit_dir / "audit.log"
    assert audit_log_path.exists()

    # Read audit log
    audit_content = audit_log_path.read_text()

    # Should contain extraction event
    assert "attachment_extracted" in audit_content

    # Should NOT contain actual content
    assert "Content for audit test" not in audit_content

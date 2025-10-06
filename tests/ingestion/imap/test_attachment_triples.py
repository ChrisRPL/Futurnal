"""Comprehensive tests for attachment semantic triple extraction.

Tests cover:
- Triple generation from attachment metadata
- Bidirectional email-attachment linking
- Inline image detection in triples
- Processing status representation
- Content hash and deduplication metadata
- Error state representation in triples
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from futurnal.ingestion.imap.attachment_triples import extract_attachment_triples
from futurnal.ingestion.imap.attachment_models import (
    EmailAttachment,
    AttachmentProcessingStatus,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def pending_attachment(tmp_path: Path) -> EmailAttachment:
    """Create a pending attachment for testing."""
    storage_path = tmp_path / "test.pdf"
    storage_path.write_text("Mock content")

    return EmailAttachment(
        attachment_id="att-123",
        message_id="email@example.com",
        part_id="part-1",
        filename="document.pdf",
        content_type="application/pdf",
        size_bytes=1024000,
        is_inline=False,
        content_id=None,
        content_hash="abcd1234" * 8,  # 64 chars
        storage_path=storage_path,
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime(2024, 1, 15, 10, 30),
        mailbox_id="test-mailbox",
    )


@pytest.fixture
def completed_attachment(tmp_path: Path) -> EmailAttachment:
    """Create a completed attachment for testing."""
    storage_path = tmp_path / "completed.pdf"
    storage_path.write_text("Processed content")

    attachment = EmailAttachment(
        attachment_id="att-456",
        message_id="email@example.com",
        part_id="part-2",
        filename="processed.pdf",
        content_type="application/pdf",
        size_bytes=2048000,
        content_hash="ef125678" * 8,
        storage_path=storage_path,
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime(2024, 1, 15, 11, 0),
        mailbox_id="test-mailbox",
    )

    # Mark as completed
    attachment.mark_completed(extraction_elements=15)

    return attachment


@pytest.fixture
def inline_image(tmp_path: Path) -> EmailAttachment:
    """Create an inline image attachment for testing."""
    storage_path = tmp_path / "image.png"
    storage_path.write_text("PNG data")

    return EmailAttachment(
        attachment_id="att-789",
        message_id="email@example.com",
        part_id="part-3",
        filename="logo.png",
        content_type="image/png",
        size_bytes=51200,
        is_inline=True,
        content_id="logo-123",
        content_hash="1234ab12" * 8,
        storage_path=storage_path,
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime(2024, 1, 15, 12, 0),
        mailbox_id="test-mailbox",
    )


# ============================================================================
# Basic Triple Extraction Tests
# ============================================================================


def test_extract_basic_triples(pending_attachment: EmailAttachment):
    """Test basic triple extraction from attachment."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    assert len(triples) > 0

    # Find specific triples
    type_triples = [t for t in triples if t.predicate == "rdf:type"]
    assert len(type_triples) == 1
    assert type_triples[0].object == "futurnal:EmailAttachment"

    filename_triples = [t for t in triples if t.predicate == "attachment:filename"]
    assert len(filename_triples) == 1
    assert filename_triples[0].object == "document.pdf"


def test_attachment_email_linking(pending_attachment: EmailAttachment):
    """Test bidirectional linking between attachment and email."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    # Check attachment → email link
    part_of_triples = [t for t in triples if t.predicate == "attachment:partOfEmail"]
    assert len(part_of_triples) == 1
    assert "email" in part_of_triples[0].object.lower()

    # Check email → attachment link
    has_attachment_triples = [t for t in triples if t.predicate == "email:hasAttachment"]
    assert len(has_attachment_triples) == 1
    assert "email" in has_attachment_triples[0].subject.lower()
    assert "attachment" in has_attachment_triples[0].object.lower()


def test_content_metadata_triples(pending_attachment: EmailAttachment):
    """Test content metadata is captured in triples."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    # Content type
    content_type_triples = [t for t in triples if t.predicate == "attachment:contentType"]
    assert len(content_type_triples) == 1
    assert content_type_triples[0].object == "application/pdf"

    # Size
    size_triples = [t for t in triples if t.predicate == "attachment:sizeBytes"]
    assert len(size_triples) == 1
    assert size_triples[0].object == "1024000"

    # Content hash
    hash_triples = [t for t in triples if t.predicate == "attachment:contentHash"]
    assert len(hash_triples) == 1
    assert len(hash_triples[0].object) == 64  # SHA256 is 64 hex chars


# ============================================================================
# Processing Status Tests
# ============================================================================


def test_pending_status_triples(pending_attachment: EmailAttachment):
    """Test triples for pending attachment."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    status_triples = [t for t in triples if t.predicate == "attachment:processingStatus"]
    assert len(status_triples) == 1
    assert status_triples[0].object == "pending"

    # Should NOT have processing metadata
    elements_triples = [t for t in triples if t.predicate == "attachment:extractionElements"]
    assert len(elements_triples) == 0


def test_completed_status_triples(completed_attachment: EmailAttachment):
    """Test triples for completed attachment include processing metadata."""
    triples = extract_attachment_triples(
        attachment=completed_attachment,
        email_message_id="email@example.com",
    )

    status_triples = [t for t in triples if t.predicate == "attachment:processingStatus"]
    assert len(status_triples) == 1
    assert status_triples[0].object == "completed"

    # Should have extraction elements
    elements_triples = [t for t in triples if t.predicate == "attachment:extractionElements"]
    assert len(elements_triples) == 1
    assert elements_triples[0].object == "15"

    # Should have processed timestamp
    processed_triples = [t for t in triples if t.predicate == "attachment:processedAt"]
    assert len(processed_triples) == 1


def test_failed_status_triples(tmp_path: Path):
    """Test triples for failed attachment include error info."""
    attachment = EmailAttachment(
        attachment_id="failed-123",
        message_id="email@example.com",
        part_id="part-fail",
        filename="corrupt.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        content_hash="",
        storage_path=None,
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )

    attachment.mark_failed("Processing timeout")

    triples = extract_attachment_triples(
        attachment=attachment,
        email_message_id="email@example.com",
    )

    status_triples = [t for t in triples if t.predicate == "attachment:processingStatus"]
    assert len(status_triples) == 1
    assert status_triples[0].object == "failed"

    # Should have error message
    error_triples = [t for t in triples if t.predicate == "attachment:processingError"]
    assert len(error_triples) == 1
    assert "timeout" in error_triples[0].object.lower()


def test_skipped_status_triples(tmp_path: Path):
    """Test triples for skipped attachment include skip reason."""
    attachment = EmailAttachment(
        attachment_id="skip-123",
        message_id="email@example.com",
        part_id="part-skip",
        filename="large.pdf",
        content_type="application/pdf",
        size_bytes=100 * 1024 * 1024,  # 100MB
        content_hash="",
        storage_path=None,
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )

    attachment.mark_skipped("too_large")

    triples = extract_attachment_triples(
        attachment=attachment,
        email_message_id="email@example.com",
    )

    status_triples = [t for t in triples if t.predicate == "attachment:processingStatus"]
    assert len(status_triples) == 1
    assert status_triples[0].object == "skipped"

    # Should have error reason
    error_triples = [t for t in triples if t.predicate == "attachment:processingError"]
    assert len(error_triples) == 1
    assert "too_large" in error_triples[0].object


# ============================================================================
# Inline Image Tests
# ============================================================================


def test_inline_image_triples(inline_image: EmailAttachment):
    """Test triples for inline image attachments."""
    triples = extract_attachment_triples(
        attachment=inline_image,
        email_message_id="email@example.com",
    )

    # Should have isInline triple
    inline_triples = [t for t in triples if t.predicate == "attachment:isInline"]
    assert len(inline_triples) == 1
    assert inline_triples[0].object == "true"

    # Should have content ID
    content_id_triples = [t for t in triples if t.predicate == "attachment:contentId"]
    assert len(content_id_triples) == 1
    assert content_id_triples[0].object == "logo-123"


def test_non_inline_attachment_no_inline_triples(pending_attachment: EmailAttachment):
    """Test that non-inline attachments don't have inline triples."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    inline_triples = [t for t in triples if t.predicate == "attachment:isInline"]
    assert len(inline_triples) == 0

    content_id_triples = [t for t in triples if t.predicate == "attachment:contentId"]
    assert len(content_id_triples) == 0


# ============================================================================
# Temporal and Provenance Tests
# ============================================================================


def test_temporal_triples(pending_attachment: EmailAttachment):
    """Test temporal metadata triples."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    extracted_triples = [t for t in triples if t.predicate == "attachment:extractedAt"]
    assert len(extracted_triples) == 1
    assert "2024-01-15" in extracted_triples[0].object


def test_provenance_triples(pending_attachment: EmailAttachment):
    """Test provenance metadata triples."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    mailbox_triples = [t for t in triples if t.predicate == "attachment:mailboxId"]
    assert len(mailbox_triples) == 1
    assert mailbox_triples[0].object == "test-mailbox"


# ============================================================================
# Triple Structure Tests
# ============================================================================


def test_all_triples_have_source_element_id(pending_attachment: EmailAttachment):
    """Test that all triples have source_element_id set."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    for triple in triples:
        assert triple.source_element_id is not None
        assert triple.source_element_id == "att-123"


def test_all_triples_have_extraction_method(pending_attachment: EmailAttachment):
    """Test that all triples have extraction_method set."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    for triple in triples:
        assert triple.extraction_method == "attachment_metadata"


def test_attachment_uri_format(pending_attachment: EmailAttachment):
    """Test attachment URI follows expected format."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    # All triples with attachment as subject should have proper URI
    for triple in triples:
        if "attachment" in triple.subject.lower():
            assert triple.subject.startswith("futurnal:attachment/")
            assert "att-123" in triple.subject


def test_email_uri_format(pending_attachment: EmailAttachment):
    """Test email URI follows expected format."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    # Email references should have proper URI
    email_triples = [t for t in triples if t.predicate == "attachment:partOfEmail"]
    for triple in email_triples:
        assert triple.object.startswith("futurnal:email/")


# ============================================================================
# Multiple Attachments Test
# ============================================================================


def test_multiple_attachments_distinct_triples(tmp_path: Path):
    """Test that different attachments generate distinct triples."""
    attachment1 = EmailAttachment(
        attachment_id="att-001",
        message_id="email@example.com",
        part_id="part-1",
        filename="file1.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        content_hash="1111" + "0" * 60,
        storage_path=tmp_path / "file1.pdf",
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )

    attachment2 = EmailAttachment(
        attachment_id="att-002",
        message_id="email@example.com",
        part_id="part-2",
        filename="file2.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        content_hash="2222" + "0" * 60,
        storage_path=tmp_path / "file2.pdf",
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )

    triples1 = extract_attachment_triples(attachment1, "email@example.com")
    triples2 = extract_attachment_triples(attachment2, "email@example.com")

    # Should have different attachment URIs
    att_uris1 = {t.subject for t in triples1 if "attachment" in t.subject.lower()}
    att_uris2 = {t.subject for t in triples2 if "attachment" in t.subject.lower()}

    assert att_uris1 != att_uris2

    # Should have different filenames
    filename_triples1 = [t for t in triples1 if t.predicate == "attachment:filename"]
    filename_triples2 = [t for t in triples2 if t.predicate == "attachment:filename"]

    assert filename_triples1[0].object == "file1.pdf"
    assert filename_triples2[0].object == "file2.pdf"


# ============================================================================
# Content Hash Tests
# ============================================================================


def test_content_hash_in_triples(pending_attachment: EmailAttachment):
    """Test that content hash is included in triples."""
    triples = extract_attachment_triples(
        attachment=pending_attachment,
        email_message_id="email@example.com",
    )

    hash_triples = [t for t in triples if t.predicate == "attachment:contentHash"]
    assert len(hash_triples) == 1
    assert hash_triples[0].object == pending_attachment.content_hash


def test_empty_content_hash_still_in_triples(tmp_path: Path):
    """Test that empty content hash (skipped attachment) is handled."""
    attachment = EmailAttachment(
        attachment_id="skip-123",
        message_id="email@example.com",
        part_id="part-skip",
        filename="skipped.dat",
        content_type="application/octet-stream",
        size_bytes=1024,
        content_hash="",  # Empty for skipped
        storage_path=None,
        processing_status=AttachmentProcessingStatus.SKIPPED,
        processing_error="unsupported_format",
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )

    triples = extract_attachment_triples(
        attachment=attachment,
        email_message_id="email@example.com",
    )

    # Should not have hash triple when empty
    hash_triples = [t for t in triples if t.predicate == "attachment:contentHash"]
    assert len(hash_triples) == 0

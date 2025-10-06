"""Comprehensive tests for attachment processing through Unstructured.io.

Tests cover:
- Successful processing through Unstructured.io
- Timeout enforcement
- Status tracking (pending → processing → completed)
- Element enrichment with attachment metadata
- Error handling and failure modes
- Privacy-aware audit logging
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.ingestion.imap.attachment_processor import AttachmentProcessor
from futurnal.ingestion.imap.attachment_models import (
    EmailAttachment,
    AttachmentProcessingStatus,
)
from futurnal.privacy.audit import AuditLogger


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_attachment(tmp_path: Path) -> EmailAttachment:
    """Create a sample attachment for testing."""
    storage_path = tmp_path / "test.pdf"
    storage_path.write_text("Mock PDF content")

    return EmailAttachment(
        attachment_id="test-123",
        message_id="email@example.com",
        part_id="part-1",
        filename="document.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        is_inline=False,
        content_id=None,
        content_hash="abcd1234" * 8,  # 64 chars
        storage_path=storage_path,
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )


@pytest.fixture
def mock_unstructured_elements():
    """Create mock Unstructured.io element objects."""
    element1 = MagicMock()
    element1.to_dict.return_value = {
        "type": "Title",
        "text": "Document Title",
        "metadata": {"page_number": 1},
    }

    element2 = MagicMock()
    element2.to_dict.return_value = {
        "type": "NarrativeText",
        "text": "This is the document content.",
        "metadata": {"page_number": 1},
    }

    return [element1, element2]


# ============================================================================
# AttachmentProcessor Basic Tests
# ============================================================================


def test_processor_initialization():
    """Test processor initializes with correct settings."""
    processor = AttachmentProcessor(
        ocr_languages="eng+fra",
        processing_timeout=120,
    )

    assert processor.ocr_languages == "eng+fra"
    assert processor.processing_timeout == 120


@pytest.mark.asyncio
async def test_process_attachment_success(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
):
    """Test successful attachment processing."""
    processor = AttachmentProcessor()

    # Mock Unstructured.io partition function
    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        elements = await processor.process_attachment(sample_attachment)

        # Verify processing succeeded
        assert len(elements) == 2
        assert sample_attachment.processing_status == AttachmentProcessingStatus.COMPLETED
        assert sample_attachment.extraction_elements == 2
        assert sample_attachment.processed_at is not None
        assert sample_attachment.processing_error is None


@pytest.mark.asyncio
async def test_process_attachment_element_enrichment(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
):
    """Test that elements are enriched with attachment metadata."""
    processor = AttachmentProcessor()

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        elements = await processor.process_attachment(sample_attachment)

        # Verify metadata enrichment
        for element in elements:
            assert "metadata" in element
            assert element["metadata"]["source_attachment_id"] == "test-123"
            assert element["metadata"]["source_message_id"] == "email@example.com"
            assert element["metadata"]["attachment_filename"] == "document.pdf"
            assert element["metadata"]["attachment_content_type"] == "application/pdf"
            assert element["metadata"]["attachment_size_bytes"] == 1024


@pytest.mark.asyncio
async def test_process_attachment_already_processed(sample_attachment: EmailAttachment):
    """Test that already-processed attachments are skipped."""
    sample_attachment.processing_status = AttachmentProcessingStatus.COMPLETED

    processor = AttachmentProcessor()
    elements = await processor.process_attachment(sample_attachment)

    assert len(elements) == 0
    assert sample_attachment.processing_status == AttachmentProcessingStatus.COMPLETED


@pytest.mark.asyncio
async def test_process_attachment_missing_file(tmp_path: Path):
    """Test error handling when attachment file is missing."""
    # Create attachment with non-existent file
    attachment = EmailAttachment(
        attachment_id="missing-123",
        message_id="email@example.com",
        part_id="part-1",
        filename="missing.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        content_hash="abcd1234" * 8,
        storage_path=tmp_path / "nonexistent.pdf",  # Doesn't exist
        processing_status=AttachmentProcessingStatus.PENDING,
        extracted_at=datetime.utcnow(),
        mailbox_id="test-mailbox",
    )

    processor = AttachmentProcessor()
    elements = await processor.process_attachment(attachment)

    assert len(elements) == 0
    assert attachment.processing_status == AttachmentProcessingStatus.FAILED
    assert "not found" in attachment.processing_error.lower()


# ============================================================================
# Timeout Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processing_timeout(sample_attachment: EmailAttachment):
    """Test that processing timeout is enforced."""
    processor = AttachmentProcessor(processing_timeout=0.1)  # Very short timeout

    # Mock partition to take longer than timeout
    async def slow_partition(*args, **kwargs):
        await asyncio.sleep(1)  # 1 second delay
        return []

    with patch("unstructured.partition.auto.partition", side_effect=slow_partition):
        elements = await processor.process_attachment(sample_attachment)

        assert len(elements) == 0
        assert sample_attachment.processing_status == AttachmentProcessingStatus.FAILED
        assert "timeout" in sample_attachment.processing_error.lower()


@pytest.mark.asyncio
async def test_processing_within_timeout(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
):
    """Test that processing completes within timeout."""
    processor = AttachmentProcessor(processing_timeout=10)  # Generous timeout

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        elements = await processor.process_attachment(sample_attachment)

        assert len(elements) == 2
        assert sample_attachment.processing_status == AttachmentProcessingStatus.COMPLETED


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processing_error_handling(sample_attachment: EmailAttachment):
    """Test error handling during processing."""
    processor = AttachmentProcessor()

    # Mock partition to raise an error
    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.side_effect = RuntimeError("Unstructured.io error")

        elements = await processor.process_attachment(sample_attachment)

        assert len(elements) == 0
        assert sample_attachment.processing_status == AttachmentProcessingStatus.FAILED
        assert "Unstructured.io error" in sample_attachment.processing_error


@pytest.mark.asyncio
async def test_processing_recovers_from_temporary_failure(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
):
    """Test that attachment can be reprocessed after failure."""
    processor = AttachmentProcessor()

    # First attempt fails
    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.side_effect = RuntimeError("Temporary error")
        elements = await processor.process_attachment(sample_attachment)

        assert sample_attachment.processing_status == AttachmentProcessingStatus.FAILED

    # Reset to PENDING for retry
    sample_attachment.processing_status = AttachmentProcessingStatus.PENDING
    sample_attachment.processing_error = None

    # Second attempt succeeds
    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements
        elements = await processor.process_attachment(sample_attachment)

        assert len(elements) == 2
        assert sample_attachment.processing_status == AttachmentProcessingStatus.COMPLETED


# ============================================================================
# Status Tracking Tests
# ============================================================================


@pytest.mark.asyncio
async def test_status_transitions(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
):
    """Test status transitions during processing."""
    processor = AttachmentProcessor()

    # Initial status
    assert sample_attachment.processing_status == AttachmentProcessingStatus.PENDING
    assert sample_attachment.processed_at is None
    assert sample_attachment.extraction_elements == 0

    # Process
    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        # Capture status during processing (would be PROCESSING internally)
        elements = await processor.process_attachment(sample_attachment)

        # Final status
        assert sample_attachment.processing_status == AttachmentProcessingStatus.COMPLETED
        assert sample_attachment.processed_at is not None
        assert sample_attachment.extraction_elements == 2
        assert sample_attachment.processing_error is None


# ============================================================================
# OCR Language Configuration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_ocr_language_configuration(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
):
    """Test that OCR languages are passed to Unstructured.io."""
    processor = AttachmentProcessor(ocr_languages="eng+fra+deu")

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        await processor.process_attachment(sample_attachment)

        # Verify partition was called with correct language config
        mock_partition.assert_called_once()
        call_kwargs = mock_partition.call_args[1]
        assert call_kwargs["languages"] == ["eng+fra+deu"]


# ============================================================================
# Audit Logging Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audit_logging_success(
    sample_attachment: EmailAttachment,
    mock_unstructured_elements,
    tmp_path: Path,
):
    """Test privacy-aware audit logging on success."""
    audit_dir = tmp_path / "audit"
    audit_logger = AuditLogger(output_dir=audit_dir)

    processor = AttachmentProcessor(audit_logger=audit_logger)

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        await processor.process_attachment(sample_attachment)

        # Check audit log
        audit_log_path = audit_dir / "audit.log"
        assert audit_log_path.exists()
        audit_content = audit_log_path.read_text()

        assert "attachment_processed" in audit_content
        assert "success" in audit_content

        # Should NOT contain file content
        assert "Mock PDF content" not in audit_content


@pytest.mark.asyncio
async def test_audit_logging_failure(
    sample_attachment: EmailAttachment,
    tmp_path: Path,
):
    """Test privacy-aware audit logging on failure."""
    audit_dir = tmp_path / "audit"
    audit_logger = AuditLogger(output_dir=audit_dir)

    processor = AttachmentProcessor(audit_logger=audit_logger)

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.side_effect = RuntimeError("Processing error")

        await processor.process_attachment(sample_attachment)

        # Check audit log
        audit_log_path = audit_dir / "audit.log"
        assert audit_log_path.exists()
        audit_content = audit_log_path.read_text()

        assert "attachment_processed" in audit_content
        assert "failed" in audit_content


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_multiple_attachments(
    tmp_path: Path,
    mock_unstructured_elements,
):
    """Test processing multiple attachments sequentially."""
    processor = AttachmentProcessor()

    attachments = []
    for i in range(3):
        storage_path = tmp_path / f"file{i}.pdf"
        storage_path.write_text(f"Content {i}")

        attachment = EmailAttachment(
            attachment_id=f"test-{i}",
            message_id="email@example.com",
            part_id=f"part-{i}",
            filename=f"file{i}.pdf",
            content_type="application/pdf",
            size_bytes=100 + i,
            content_hash=f"123{i}abcd" + "0" * 56,
            storage_path=storage_path,
            processing_status=AttachmentProcessingStatus.PENDING,
            extracted_at=datetime.utcnow(),
            mailbox_id="test-mailbox",
        )
        attachments.append(attachment)

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = mock_unstructured_elements

        # Process all
        for attachment in attachments:
            elements = await processor.process_attachment(attachment)
            assert len(elements) == 2
            assert attachment.processing_status == AttachmentProcessingStatus.COMPLETED

    # All should be completed
    assert all(att.is_processed for att in attachments)


@pytest.mark.asyncio
async def test_empty_elements_list(sample_attachment: EmailAttachment):
    """Test handling when Unstructured.io returns empty elements."""
    processor = AttachmentProcessor()

    with patch("unstructured.partition.auto.partition") as mock_partition:
        mock_partition.return_value = []  # Empty list

        elements = await processor.process_attachment(sample_attachment)

        assert len(elements) == 0
        assert sample_attachment.processing_status == AttachmentProcessingStatus.COMPLETED
        assert sample_attachment.extraction_elements == 0

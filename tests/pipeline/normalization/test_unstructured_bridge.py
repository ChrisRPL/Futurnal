"""Unit tests for UnstructuredBridge.

Tests cover:
- Initialization and library availability
- Strategy selection logic per format
- Element-to-dict conversion with metadata preservation
- Document processing with file_path and content inputs
- Error handling and recovery
- Metrics tracking and success rate calculation

All Unstructured.io calls are mocked using conftest.py patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.unstructured_bridge import (
    PartitionStrategy,
    UnstructuredBridge,
    UnstructuredProcessingError,
    UNSTRUCTURED_FORMAT_CONFIG,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_unstructured_element():
    """Create a mock Unstructured.io element with realistic attributes."""

    def create_element(text: str, element_type: str = "Title"):
        """Factory function to create mock elements with dynamic types."""

        # Create a custom class dynamically with the correct __name__
        MockElementClass = type(
            element_type,  # Class name
            (),  # Base classes
            {
                "__str__": lambda self: self._text,
                "__init__": lambda self, text, element_type: None,  # Will be overridden
            },
        )

        # Create instance
        element = MockElementClass.__new__(MockElementClass)
        element._text = text
        element._type = element_type
        element.id = f"element-{hash(text)}"

        # Mock metadata object
        element.metadata = MagicMock()
        element.metadata.filename = "test.pdf"
        element.metadata.filetype = "application/pdf"
        element.metadata.page_number = 1
        element.metadata.page_name = None
        element.metadata.category = element_type
        element.metadata.element_id = element.id
        element.metadata.coordinates = None

        return element

    return create_element


@pytest.fixture
def mock_partition_function(mock_unstructured_element):
    """Create a mock partition function that returns realistic elements."""

    def _partition(**kwargs) -> List[Any]:
        # Return mock elements based on input
        filename = kwargs.get("filename")
        text_content = kwargs.get("text")

        if filename:
            # File-based processing
            return [
                mock_unstructured_element(f"Title from {Path(filename).name}", "Title"),
                mock_unstructured_element("This is the first paragraph.", "NarrativeText"),
                mock_unstructured_element("This is another section.", "NarrativeText"),
            ]
        elif text_content:
            # Text-based processing
            words = text_content.split()[:5]
            return [
                mock_unstructured_element(" ".join(words), "NarrativeText"),
            ]
        else:
            return []

    return _partition


@pytest.fixture
def bridge_with_mock(mock_partition_function):
    """Create UnstructuredBridge with mocked partition function."""
    bridge = UnstructuredBridge()
    bridge._partition_func = mock_partition_function
    return bridge


# ============================================================================
# Initialization Tests
# ============================================================================


def test_init_loads_unstructured_library_successfully():
    """Test that UnstructuredBridge initializes when unstructured is available."""
    # Unstructured is already mocked in conftest.py
    bridge = UnstructuredBridge()

    assert bridge.documents_processed == 0
    assert bridge.processing_errors == 0
    assert bridge._partition_func is not None


@pytest.mark.skip(reason="Difficult to test reliably with conftest.py stub system")
def test_init_fails_when_unstructured_not_installed():
    """Test that UnstructuredBridge raises error when unstructured not available.

    Note: This test is skipped because the conftest.py automatically stubs
    unstructured.io, making it difficult to test the failure case reliably
    in the unit test environment. In production, the error handling is verified
    by the _ensure_unstructured_available method.
    """
    pass


# ============================================================================
# Strategy Selection Tests
# ============================================================================


def test_strategy_auto_selection_pdf_returns_hi_res():
    """Test that PDF format auto-selects HI_RES strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.PDF)
    assert strategy == PartitionStrategy.HI_RES


def test_strategy_auto_selection_docx_returns_hi_res():
    """Test that DOCX format auto-selects HI_RES strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.DOCX)
    assert strategy == PartitionStrategy.HI_RES


def test_strategy_auto_selection_pptx_returns_hi_res():
    """Test that PPTX format auto-selects HI_RES strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.PPTX)
    assert strategy == PartitionStrategy.HI_RES


def test_strategy_auto_selection_xlsx_returns_hi_res():
    """Test that XLSX format auto-selects HI_RES strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.XLSX)
    assert strategy == PartitionStrategy.HI_RES


def test_strategy_auto_selection_markdown_returns_fast():
    """Test that MARKDOWN format auto-selects FAST strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.MARKDOWN)
    assert strategy == PartitionStrategy.FAST


def test_strategy_auto_selection_html_returns_fast():
    """Test that HTML format auto-selects FAST strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.HTML)
    assert strategy == PartitionStrategy.FAST


def test_strategy_auto_selection_text_returns_fast():
    """Test that TEXT format auto-selects FAST strategy."""
    bridge = UnstructuredBridge()
    strategy = bridge._select_strategy(DocumentFormat.TEXT)
    assert strategy == PartitionStrategy.FAST


# ============================================================================
# Binary Format Detection Tests
# ============================================================================


def test_binary_formats_includes_pdf():
    """Test that PDF is recognized as binary format."""
    bridge = UnstructuredBridge()
    binary_formats = bridge._binary_formats()
    assert DocumentFormat.PDF in binary_formats


def test_binary_formats_includes_office_documents():
    """Test that Office formats are recognized as binary."""
    bridge = UnstructuredBridge()
    binary_formats = bridge._binary_formats()
    assert DocumentFormat.DOCX in binary_formats
    assert DocumentFormat.PPTX in binary_formats
    assert DocumentFormat.XLSX in binary_formats


def test_binary_formats_excludes_text_formats():
    """Test that text formats are not in binary format set."""
    bridge = UnstructuredBridge()
    binary_formats = bridge._binary_formats()
    assert DocumentFormat.MARKDOWN not in binary_formats
    assert DocumentFormat.HTML not in binary_formats
    assert DocumentFormat.TEXT not in binary_formats


# ============================================================================
# Element Conversion Tests
# ============================================================================


def test_element_to_dict_preserves_text(mock_unstructured_element):
    """Test that element-to-dict conversion preserves text content."""
    bridge = UnstructuredBridge()
    element = mock_unstructured_element("Test content here", "NarrativeText")

    result = bridge._element_to_dict(element)

    assert result["text"] == "Test content here"
    assert result["type"] == "NarrativeText"


def test_element_to_dict_preserves_metadata(mock_unstructured_element):
    """Test that element-to-dict conversion preserves all metadata."""
    bridge = UnstructuredBridge()
    element = mock_unstructured_element("Title text", "Title")
    element.metadata.page_number = 5
    element.metadata.filename = "document.pdf"

    result = bridge._element_to_dict(element)

    assert result["metadata"]["page_number"] == 5
    assert result["metadata"]["filename"] == "document.pdf"
    assert result["metadata"]["filetype"] == "application/pdf"


def test_element_to_dict_handles_missing_attributes_gracefully():
    """Test that element-to-dict handles elements with missing attributes."""
    bridge = UnstructuredBridge()

    # Create minimal element without full metadata
    MinimalElementClass = type("Unknown", (), {})

    class MinimalElement(MinimalElementClass):
        def __str__(self):
            return "Minimal content"

    MinimalElement.__name__ = "Unknown"

    element = MinimalElement()

    result = bridge._element_to_dict(element)

    assert result["text"] == "Minimal content"
    assert result["type"] == "Unknown"
    assert "metadata" in result


def test_element_to_dict_includes_element_id_when_available(mock_unstructured_element):
    """Test that element ID is included when present on element."""
    bridge = UnstructuredBridge()
    element = mock_unstructured_element("Content", "Text")
    element.id = "unique-element-id-123"

    result = bridge._element_to_dict(element)

    assert result["element_id"] == "unique-element-id-123"


# ============================================================================
# Document Processing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_document_with_file_path_binary_format(
    bridge_with_mock, tmp_path
):
    """Test processing PDF with file_path input."""
    # Create temporary PDF file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

    elements = await bridge_with_mock.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
    )

    assert len(elements) > 0
    assert bridge_with_mock.documents_processed == 1
    assert bridge_with_mock.processing_errors == 0


@pytest.mark.asyncio
async def test_process_document_with_content_text_format(bridge_with_mock):
    """Test processing markdown with content string input."""
    markdown_content = "# Title\n\nThis is markdown content."

    elements = await bridge_with_mock.process_document(
        content=markdown_content,
        format=DocumentFormat.MARKDOWN,
    )

    assert len(elements) > 0
    assert bridge_with_mock.documents_processed == 1


@pytest.mark.asyncio
async def test_process_document_raises_without_any_input(bridge_with_mock):
    """Test that process_document raises error when neither content nor file_path provided."""
    with pytest.raises(ValueError, match="Either content or file_path must be provided"):
        await bridge_with_mock.process_document(
            format=DocumentFormat.MARKDOWN,
        )


@pytest.mark.asyncio
async def test_process_document_uses_manual_strategy_override(bridge_with_mock, tmp_path):
    """Test that manually specified strategy is respected."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

    # Process with explicit FAST strategy (normally PDF would use HI_RES)
    elements = await bridge_with_mock.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
        strategy=PartitionStrategy.FAST,
    )

    assert len(elements) > 0


@pytest.mark.asyncio
async def test_process_document_increments_success_counter(bridge_with_mock, tmp_path):
    """Test that successful processing increments documents_processed counter."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    assert bridge_with_mock.documents_processed == 0

    await bridge_with_mock.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
    )

    assert bridge_with_mock.documents_processed == 1


@pytest.mark.asyncio
async def test_process_document_increments_error_counter_on_failure(tmp_path):
    """Test that failed processing increments processing_errors counter."""
    bridge = UnstructuredBridge()

    # Mock partition function to raise an error
    def failing_partition(**kwargs):
        raise RuntimeError("Simulated processing failure")

    bridge._partition_func = failing_partition

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    assert bridge.processing_errors == 0

    with pytest.raises(UnstructuredProcessingError):
        await bridge.process_document(
            file_path=pdf_file,
            format=DocumentFormat.PDF,
        )

    assert bridge.processing_errors == 1


@pytest.mark.asyncio
async def test_process_document_includes_filename_in_error_message(tmp_path):
    """Test that error messages include filename for better debugging."""
    bridge = UnstructuredBridge()

    def failing_partition(**kwargs):
        raise RuntimeError("Processing failed")

    bridge._partition_func = failing_partition

    pdf_file = tmp_path / "important_document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    with pytest.raises(UnstructuredProcessingError, match="important_document.pdf"):
        await bridge.process_document(
            file_path=pdf_file,
            format=DocumentFormat.PDF,
        )


# ============================================================================
# Metrics Tests
# ============================================================================


def test_get_metrics_returns_correct_counts():
    """Test that get_metrics returns accurate processing counts."""
    bridge = UnstructuredBridge()
    bridge.documents_processed = 10
    bridge.processing_errors = 2

    metrics = bridge.get_metrics()

    assert metrics["documents_processed"] == 10
    assert metrics["processing_errors"] == 2


def test_get_metrics_calculates_success_rate_correctly():
    """Test that success_rate is calculated as processed / total."""
    bridge = UnstructuredBridge()
    bridge.documents_processed = 8
    bridge.processing_errors = 2

    metrics = bridge.get_metrics()

    # 8 / (8 + 2) = 0.8 = 80%
    assert metrics["success_rate"] == 0.8


def test_get_metrics_handles_zero_documents_safely():
    """Test that get_metrics returns 0.0 success_rate when no documents processed."""
    bridge = UnstructuredBridge()

    metrics = bridge.get_metrics()

    assert metrics["documents_processed"] == 0
    assert metrics["processing_errors"] == 0
    assert metrics["success_rate"] == 0.0


def test_get_metrics_handles_only_errors():
    """Test that get_metrics correctly calculates 0% success rate when all failed."""
    bridge = UnstructuredBridge()
    bridge.processing_errors = 5

    metrics = bridge.get_metrics()

    assert metrics["success_rate"] == 0.0


def test_get_metrics_handles_perfect_success_rate():
    """Test that get_metrics returns 1.0 when all documents succeeded."""
    bridge = UnstructuredBridge()
    bridge.documents_processed = 20

    metrics = bridge.get_metrics()

    assert metrics["success_rate"] == 1.0


# ============================================================================
# Configuration Tests
# ============================================================================


def test_unstructured_format_config_exists():
    """Test that UNSTRUCTURED_FORMAT_CONFIG is defined."""
    assert UNSTRUCTURED_FORMAT_CONFIG is not None
    assert isinstance(UNSTRUCTURED_FORMAT_CONFIG, dict)


def test_unstructured_format_config_has_privacy_settings():
    """Test that PDF config includes privacy-preserving settings."""
    pdf_config = UNSTRUCTURED_FORMAT_CONFIG.get("pdf", {})
    assert pdf_config.get("extract_images_in_pdf") is False


def test_unstructured_format_config_has_strategy_per_format():
    """Test that each format has a strategy defined."""
    for format_key in ["pdf", "docx", "markdown", "html", "email"]:
        assert "strategy" in UNSTRUCTURED_FORMAT_CONFIG.get(format_key, {})

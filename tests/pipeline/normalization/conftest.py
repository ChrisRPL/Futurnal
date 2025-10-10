"""Test fixtures and configuration for normalization pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# Test Document

## Introduction
This is a test document with multiple sections.

## Section 1
Some content in section 1 with **bold** and *italic* text.

### Subsection 1.1
Nested content here.

## Section 2
More content in section 2.

#tag1 #tag2
"""


@pytest.fixture
def sample_text_content():
    """Sample plain text content."""
    return "This is a simple text document.\nWith multiple lines.\nAnd some content."


@pytest.fixture
def mock_unstructured_elements():
    """Mock Unstructured.io element output."""
    return [
        {
            "type": "Title",
            "text": "Test Document",
            "metadata": {
                "filename": "test.pdf",
                "page_number": 1,
                "category": "Title",
            },
        },
        {
            "type": "NarrativeText",
            "text": "This is the first paragraph of the document.",
            "metadata": {"filename": "test.pdf", "page_number": 1},
        },
        {
            "type": "NarrativeText",
            "text": "This is the second paragraph on page 2.",
            "metadata": {"filename": "test.pdf", "page_number": 2},
        },
    ]


@pytest.fixture
def mock_unstructured_partition(monkeypatch):
    """Mock unstructured.partition.auto.partition function."""
    mock_element = MagicMock()
    mock_element.__class__.__name__ = "Element"
    mock_element.__str__ = lambda self: "Mock element text"
    mock_element.metadata = MagicMock()
    mock_element.metadata.filename = "test.pdf"
    mock_element.metadata.page_number = 1

    def mock_partition(**kwargs):
        return [mock_element]

    # Patch the partition function
    if "unstructured.partition.auto" in sys.modules:
        monkeypatch.setattr(
            "unstructured.partition.auto.partition", mock_partition
        )

    return mock_partition


@pytest.fixture
def mock_language_detector(monkeypatch):
    """Mock fasttext language detector."""

    def mock_detect(text: str, **kwargs):
        return {"lang": "en", "score": 0.95}

    # Mock ftlangdetect if available
    try:
        monkeypatch.setattr("ftlangdetect.detect", mock_detect)
    except (ImportError, AttributeError):
        pass

    return mock_detect


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""

    def _create_file(content: str, name: str = "test.txt") -> Path:
        file_path = tmp_path / name
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create_file


@pytest.fixture
def temp_pdf_file(tmp_path):
    """Create a temporary PDF file placeholder."""

    def _create_pdf(name: str = "test.pdf") -> Path:
        file_path = tmp_path / name
        # Create a minimal PDF-like file
        file_path.write_bytes(b"%PDF-1.4\n")
        return file_path

    return _create_pdf


@pytest.fixture
def mock_adapter():
    """Mock format adapter for testing."""

    class MockAdapter:
        name = "MockAdapter"
        supported_formats = []
        requires_unstructured_processing = False

        async def normalize(self, **kwargs):
            from futurnal.pipeline.models import (
                DocumentFormat,
                NormalizedDocument,
                NormalizedMetadata,
            )
            from datetime import datetime, timezone

            metadata = NormalizedMetadata(
                source_path=str(kwargs.get("file_path", "test.txt")),
                source_id=kwargs.get("source_id", "test-123"),
                source_type=kwargs.get("source_type", "test"),
                format=DocumentFormat.TEXT,
                content_type="text/plain",
                character_count=100,
                word_count=20,
                line_count=5,
                content_hash="abcd1234" * 8,
            )

            return NormalizedDocument(
                document_id="test-doc-123",
                sha256="abcd1234" * 8,
                content="Mock content",
                metadata=metadata,
            )

        async def validate(self, file_path):
            return True

    return MockAdapter()


@pytest.fixture
def mock_normalization_sink():
    """Mock normalization sink for testing."""

    class MockSink:
        def __init__(self):
            self.handled_documents = []
            self.deleted_documents = []

        def handle(self, element: dict):
            self.handled_documents.append(element)

        def handle_deletion(self, element: dict):
            self.deleted_documents.append(element)

    return MockSink()


@pytest.fixture
def mock_audit_logger(tmp_path):
    """Create a real audit logger for testing."""
    from futurnal.privacy.audit import AuditLogger

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(exist_ok=True)

    return AuditLogger(output_dir=audit_dir)


@pytest.fixture
def mock_quarantine_manager(tmp_path):
    """Create a real quarantine manager for testing."""
    from futurnal.orchestrator.quarantine import QuarantineStore

    quarantine_db = tmp_path / "quarantine" / "quarantine.db"
    quarantine_db.parent.mkdir(exist_ok=True)

    return QuarantineStore(path=quarantine_db)


# ---------------------------------------------------------------------------
# Import Quality Gates Fixtures
# ---------------------------------------------------------------------------

# Import all format sample fixtures
from tests.pipeline.normalization.fixtures.format_samples import (
    markdown_simple,
    markdown_complex,
    markdown_large,
    text_simple,
    text_large,
    code_python,
    code_javascript,
    json_simple,
    json_large,
    yaml_simple,
    csv_simple,
    csv_large,
    html_simple,
    xml_simple,
    email_simple,
    jupyter_simple,
    all_format_fixtures,
)

# Import all edge case fixtures
from tests.pipeline.normalization.fixtures.edge_case_data import (
    empty_file,
    whitespace_only_file,
    single_char_file,
    single_line_file,
    unicode_emoji_file,
    unicode_bom_file,
    mixed_encoding_file,
    large_file_10mb,
    large_file_100mb,
    large_markdown_50mb,
    truncated_json_file,
    invalid_json_file,
    malformed_xml_file,
    malformed_yaml_file,
    binary_as_text_file,
    null_bytes_file,
    deeply_nested_json,
    very_long_lines_file,
    many_small_lines_file,
    special_filename_file,
    very_long_filename_file,
    markdown_no_headings,
    markdown_only_headings,
    csv_inconsistent_columns,
    html_malformed,
    all_edge_case_fixtures,
)


# ---------------------------------------------------------------------------
# Pytest Hooks for Custom Markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers for quality gates testing."""
    config.addinivalue_line(
        "markers",
        "determinism: Tests for deterministic outputs (byte-identical)"
    )
    config.addinivalue_line(
        "markers",
        "format_coverage: Tests for all 16+ DocumentFormat types"
    )
    config.addinivalue_line(
        "markers",
        "edge_case: Tests for edge cases (empty, large, corrupted files)"
    )
    config.addinivalue_line(
        "markers",
        "privacy_audit: Tests for privacy compliance (no content leakage)"
    )
    config.addinivalue_line(
        "markers",
        "production_readiness: Tests for production readiness validation"
    )
    config.addinivalue_line(
        "markers",
        "slow: Slow-running tests (use --run-slow to include)"
    )

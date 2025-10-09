"""Integration tests for format adapter registry and pipeline.

Tests end-to-end adapter workflows, format detection, error propagation,
and interaction with the normalization pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.adapters import (
    CodeAdapter,
    EmailAdapter,
    GenericAdapter,
    HTMLAdapter,
    MarkdownAdapter,
    PDFAdapter,
)
from futurnal.pipeline.normalization.registry import (
    AdapterError,
    FormatAdapterRegistry,
)


# ============================================================================
# Format Detection and Selection Tests
# ============================================================================


@pytest.mark.asyncio
async def test_format_detection_and_adapter_selection(temp_file):
    """Test registry selects correct adapter based on format."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Create test files for different formats
    test_cases = [
        ("test.md", DocumentFormat.MARKDOWN, "# Test"),
        ("test.html", DocumentFormat.HTML, "<html><body>Test</body></html>"),
        ("test.py", DocumentFormat.CODE, "def test(): pass"),
    ]

    for filename, expected_format, content in test_cases:
        file_path = temp_file(content, filename)
        adapter = registry.get_adapter(expected_format)

        doc = await adapter.normalize(
            file_path=file_path,
            source_id="test-123",
            source_type="local_files",
            source_metadata={},
        )

        assert doc.metadata.format == expected_format


@pytest.mark.asyncio
async def test_fallback_adapter_for_unknown_format(temp_file):
    """Test fallback adapter is used for unknown formats."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Create file with unknown extension
    file_path = temp_file("unknown content", "test.xyz")
    adapter = registry.get_adapter(DocumentFormat.UNKNOWN)

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={},
    )

    assert "generic_adapter" in doc.metadata.extra
    assert doc.metadata.extra["generic_adapter"]["used_fallback"] is True


# ============================================================================
# Multi-Format Processing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_multiple_formats_sequentially(temp_file):
    """Test processing multiple file formats in sequence."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Create various files
    files = [
        ("note.md", DocumentFormat.MARKDOWN, "# Note"),
        ("code.py", DocumentFormat.CODE, "def test(): pass"),
        ("page.html", DocumentFormat.HTML, "<html><body>Test</body></html>"),
    ]

    results = []
    for filename, fmt, content in files:
        file_path = temp_file(content, filename)
        adapter = registry.get_adapter(fmt)

        doc = await adapter.normalize(
            file_path=file_path,
            source_id=f"test-{filename}",
            source_type="local_files",
            source_metadata={},
        )

        results.append((filename, doc))

    # Verify all processed successfully
    assert len(results) == 3
    for filename, doc in results:
        assert doc.document_id is not None
        assert doc.metadata.source_path.endswith(filename)


# ============================================================================
# Error Propagation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_adapter_error_propagates():
    """Test adapter errors propagate correctly."""
    adapter = MarkdownAdapter()
    fake_path = Path("/nonexistent/file.md")

    with pytest.raises(AdapterError) as exc_info:
        await adapter.normalize(
            file_path=fake_path,
            source_id="test-123",
            source_type="local_files",
            source_metadata={},
        )

    assert "Failed to normalize" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validation_failure_handling(temp_file):
    """Test adapter validation failures are handled correctly."""
    adapter = MarkdownAdapter()

    # Create non-markdown file
    txt_file = temp_file("content", "test.txt")

    # Validation should fail
    is_valid = await adapter.validate(txt_file)
    assert is_valid is False


# ============================================================================
# Schema Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_normalized_document_schema_compliance(temp_file):
    """Test adapter output complies with NormalizedDocument schema."""
    content = "# Test Document\n\nContent here."
    file_path = temp_file(content, "test.md")
    adapter = MarkdownAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={"custom": "value"},
    )

    # Check required fields
    assert doc.document_id is not None
    assert doc.sha256 is not None
    assert doc.content is not None
    assert doc.metadata is not None

    # Check metadata fields
    assert doc.metadata.source_path == str(file_path)
    assert doc.metadata.source_id == "test-123"
    assert doc.metadata.source_type == "local_files"
    assert doc.metadata.format == DocumentFormat.MARKDOWN
    assert doc.metadata.character_count > 0
    assert doc.metadata.word_count > 0
    assert doc.metadata.line_count > 0

    # Check timestamps
    assert doc.metadata.created_at is not None
    assert doc.metadata.modified_at is not None
    assert doc.metadata.ingested_at is not None
    assert doc.normalized_at is not None


@pytest.mark.asyncio
async def test_metadata_extra_field_population(temp_file):
    """Test adapters populate format-specific metadata."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    test_cases = [
        (
            "test.md",
            DocumentFormat.MARKDOWN,
            "# Test\n\n#tag",
            "markdown",
        ),
        (
            "test.html",
            DocumentFormat.HTML,
            "<html><head><title>Test</title></head><body>Content</body></html>",
            "html",
        ),
        (
            "test.py",
            DocumentFormat.CODE,
            "# Comment\ndef test(): pass",
            "code",
        ),
    ]

    for filename, fmt, content, extra_key in test_cases:
        file_path = temp_file(content, filename)
        adapter = registry.get_adapter(fmt)

        doc = await adapter.normalize(
            file_path=file_path,
            source_id="test-123",
            source_type="local_files",
            source_metadata={},
        )

        assert extra_key in doc.metadata.extra
        assert isinstance(doc.metadata.extra[extra_key], dict)


# ============================================================================
# Unstructured.io Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_adapters_requiring_unstructured_flag(temp_pdf_file):
    """Test adapters requiring Unstructured.io are flagged correctly."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Adapters requiring Unstructured.io
    pdf_adapter = registry.get_adapter(DocumentFormat.PDF)
    html_adapter = registry.get_adapter(DocumentFormat.HTML)

    assert pdf_adapter.requires_unstructured_processing is True
    assert html_adapter.requires_unstructured_processing is True

    # Adapters NOT requiring Unstructured.io
    md_adapter = registry.get_adapter(DocumentFormat.MARKDOWN)
    code_adapter = registry.get_adapter(DocumentFormat.CODE)

    assert md_adapter.requires_unstructured_processing is False
    assert code_adapter.requires_unstructured_processing is False


# ============================================================================
# Comprehensive Pipeline Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_markdown(temp_file):
    """Test full pipeline for Markdown document."""
    content = """---
title: Test Document
tags: [test, example]
---

# Main Heading

This is a test document with **bold** and *italic* text.

## Section 1

Content in section 1.

#inline-tag
"""
    file_path = temp_file(content, "test.md")
    adapter = MarkdownAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="md-123",
        source_type="obsidian_vault",
        source_metadata={"vault": "test_vault"},
    )

    # Verify complete processing
    assert doc.document_id is not None
    assert doc.content is not None
    assert "markdown" in doc.metadata.extra

    # Verify metadata extraction
    markdown_meta = doc.metadata.extra["markdown"]
    assert markdown_meta["has_frontmatter"] is True
    assert markdown_meta["has_tags"] is True


@pytest.mark.asyncio
async def test_full_pipeline_code(temp_file):
    """Test full pipeline for code document."""
    code_content = '''"""
This is a module docstring.
"""

import sys

# This is a comment
def main():
    """Function docstring."""
    print("Hello, world!")
    return 0

class TestClass:
    """Class docstring."""
    pass

if __name__ == "__main__":
    sys.exit(main())
'''
    file_path = temp_file(code_content, "script.py")
    adapter = CodeAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="code-123",
        source_type="local_files",
        source_metadata={},
    )

    # Verify complete processing
    assert doc.document_id is not None
    assert "code" in doc.metadata.extra

    # Verify code-specific metadata
    code_meta = doc.metadata.extra["code"]
    assert code_meta["language"] == "python"
    assert code_meta["comment_count"] > 0
    assert code_meta["has_functions"] is True
    assert code_meta["has_classes"] is True
    assert code_meta["has_imports"] is True


@pytest.mark.asyncio
async def test_full_pipeline_html(temp_file):
    """Test full pipeline for HTML document."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Test page description">
    <meta name="keywords" content="test, html, adapter">
    <title>Test HTML Page</title>
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a test paragraph.</p>

    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
        </tr>
    </table>

    <form action="/submit">
        <input type="text" name="field">
    </form>
</body>
</html>
"""
    file_path = temp_file(html_content, "page.html")
    adapter = HTMLAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="html-123",
        source_type="local_files",
        source_metadata={},
    )

    # Verify complete processing
    assert doc.document_id is not None
    assert "html" in doc.metadata.extra

    # Verify HTML-specific metadata
    html_meta = doc.metadata.extra["html"]
    assert html_meta["title"] == "Test HTML Page"
    assert html_meta["meta_description"] == "Test page description"
    assert html_meta["meta_keywords"] == "test, html, adapter"
    assert html_meta["heading_count"] > 0
    assert html_meta["table_count"] > 0
    assert html_meta["form_count"] > 0
    assert html_meta["has_tables"] is True
    assert html_meta["has_forms"] is True
    assert html_meta["detected_language"] == "en"

    # Verify language was set
    assert doc.metadata.language == "en"

    # Verify tags extracted from keywords
    assert len(doc.metadata.tags) > 0
    assert "test" in doc.metadata.tags


# ============================================================================
# Edge Case Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_empty_file_handling(temp_file):
    """Test adapters handle empty files gracefully."""
    empty_file = temp_file("", "empty.txt")
    adapter = GenericAdapter()

    doc = await adapter.normalize(
        file_path=empty_file,
        source_id="empty-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.content == ""
    assert doc.metadata.character_count == 0
    assert doc.metadata.word_count == 0


@pytest.mark.asyncio
async def test_large_file_metadata(temp_file):
    """Test adapters compute correct metadata for larger files."""
    # Create a larger file
    large_content = "\n".join([f"Line {i}" for i in range(1000)])
    file_path = temp_file(large_content, "large.txt")
    adapter = GenericAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="large-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.line_count == 1000
    assert doc.metadata.word_count == 2000  # "Line" + number per line
    assert doc.metadata.character_count == len(large_content)


# ============================================================================
# Registry Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_registry_adapter_interoperability():
    """Test all registered adapters work together correctly."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Get all registered formats
    formats = registry.list_supported_formats()

    # Should have all standard formats
    assert DocumentFormat.MARKDOWN in formats
    assert DocumentFormat.EMAIL in formats
    assert DocumentFormat.HTML in formats
    assert DocumentFormat.CODE in formats
    assert DocumentFormat.PDF in formats

    # All should be retrievable
    for fmt in formats:
        adapter = registry.get_adapter(fmt)
        assert adapter is not None
        assert hasattr(adapter, "normalize")
        assert hasattr(adapter, "validate")

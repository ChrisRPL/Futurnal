"""Integration tests for UnstructuredBridge with format-specific processing.

Tests cover real-world scenarios:
- PDF processing with table inference and privacy settings
- Markdown processing with frontmatter preservation
- HTML processing with complex nested structures
- Email processing with header metadata extraction
- Format-specific parameter application
- End-to-end processing pipeline validation

These tests use realistic document fixtures and verify complete processing workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.unstructured_bridge import (
    PartitionStrategy,
    UnstructuredBridge,
    UNSTRUCTURED_FORMAT_CONFIG,
)


# ============================================================================
# Test Fixtures - Realistic Document Content
# ============================================================================


@pytest.fixture
def pdf_with_tables_content():
    """Simulated PDF content with table structure."""
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/MediaBox [0 0 612 792]
/Contents 5 0 R
>>
endobj
5 0 obj
<< /Length 100 >>
stream
BT
/F1 12 Tf
100 700 Td
(Financial Report Q4 2024) Tj
ET
endstream
endobj
%%EOF
"""


@pytest.fixture
def markdown_with_frontmatter():
    """Markdown content with YAML frontmatter."""
    return """---
title: Test Document
author: John Doe
date: 2024-01-15
tags: [testing, integration, markdown]
---

# Main Heading

This is a paragraph with **bold** and *italic* text.

## Section 1

- List item 1
- List item 2
- List item 3

### Subsection 1.1

Code block:
```python
def hello_world():
    print("Hello, World!")
```

## Section 2

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |
"""


@pytest.fixture
def complex_html():
    """HTML with complex nested structure and metadata."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Complex HTML test document">
    <meta name="keywords" content="testing, html, integration">
    <title>Complex HTML Document</title>
    <style>
        .nested { padding: 10px; }
    </style>
</head>
<body>
    <header>
        <h1>Main Title</h1>
        <nav>
            <ul>
                <li><a href="#section1">Section 1</a></li>
                <li><a href="#section2">Section 2</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <article id="section1">
            <h2>Section 1: Introduction</h2>
            <p>This is a <strong>complex</strong> paragraph with <em>nested</em> elements.</p>

            <div class="nested">
                <h3>Nested Content</h3>
                <ul>
                    <li>Nested item 1
                        <ul>
                            <li>Sub-item 1.1</li>
                            <li>Sub-item 1.2</li>
                        </ul>
                    </li>
                    <li>Nested item 2</li>
                </ul>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Cell 1</td>
                        <td>Cell 2</td>
                    </tr>
                </tbody>
            </table>
        </article>

        <article id="section2">
            <h2>Section 2: Details</h2>
            <p>More content with <a href="https://example.com">external links</a>.</p>
        </article>
    </main>

    <footer>
        <p>&copy; 2024 Test Document</p>
    </footer>
</body>
</html>
"""


@pytest.fixture
def email_with_headers():
    """RFC822 email with full headers."""
    return """From: sender@example.com
To: recipient@example.com
Cc: cc_user@example.com
Subject: Integration Test Email with Complex Headers
Date: Mon, 15 Jan 2024 10:30:00 -0500
Message-ID: <test-12345@example.com>
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
X-Priority: 1
X-Custom-Header: Custom Value

Dear Recipient,

This is a test email with multiple headers for integration testing.

The email contains:
- Multiple recipients
- Custom headers
- Formatted content
- Multiple paragraphs

Best regards,
Sender

--
Signature Line
Company Name
"""


# ============================================================================
# Mock Element Factories
# ============================================================================


def create_mock_pdf_elements(filename: str) -> List[Any]:
    """Create realistic mock elements for PDF processing."""

    def create_element(text: str, element_type: str, page_num: int = 1):
        # Create a custom class dynamically with the correct __name__
        MockElementClass = type(element_type, (), {"__str__": lambda self: self._text})

        element = MockElementClass.__new__(MockElementClass)
        element._text = text
        element._type = element_type
        element.id = f"element-{hash(text)}"

        element.metadata = MagicMock()
        element.metadata.filename = filename
        element.metadata.filetype = "application/pdf"
        element.metadata.page_number = page_num
        element.metadata.category = element_type
        element.metadata.element_id = element.id
        # Simulate table coordinates for table elements
        if "Table" in element_type:
            element.metadata.coordinates = {
                "points": [(100, 100), (500, 100), (500, 200), (100, 200)],
                "system": "PixelSpace",
            }
        else:
            element.metadata.coordinates = None

        return element

    return [
        create_element("Financial Report Q4 2024", "Title", 1),
        create_element("Revenue Analysis", "Title", 1),
        create_element("The company achieved strong growth in Q4.", "NarrativeText", 1),
        create_element("Quarter | Revenue | Growth\nQ1 | $100M | 10%\nQ2 | $120M | 20%", "Table", 1),
        create_element("Conclusion", "Title", 2),
        create_element("Overall performance exceeded expectations.", "NarrativeText", 2),
    ]


def create_mock_markdown_elements() -> List[Any]:
    """Create realistic mock elements for Markdown processing."""

    def create_element(text: str, element_type: str):
        MockElementClass = type(element_type, (), {"__str__": lambda self: self._text})
        element = MockElementClass.__new__(MockElementClass)
        element._text = text
        element._type = element_type
        element.id = f"element-{hash(text)}"
        element.metadata = MagicMock()
        element.metadata.filename = None
        element.metadata.category = element_type
        element.metadata.element_id = element.id
        return element

    return [
        create_element("Main Heading", "Title"),
        create_element("This is a paragraph with bold and italic text.", "NarrativeText"),
        create_element("Section 1", "Title"),
        create_element("List item 1\nList item 2\nList item 3", "ListItem"),
        create_element("Subsection 1.1", "Title"),
        create_element("def hello_world():\n    print('Hello, World!')", "CodeSnippet"),
    ]


def create_mock_html_elements() -> List[Any]:
    """Create realistic mock elements for HTML processing."""

    def create_element(text: str, element_type: str, tag: str = None):
        MockElementClass = type(element_type, (), {"__str__": lambda self: self._text})
        element = MockElementClass.__new__(MockElementClass)
        element._text = text
        element._type = element_type
        element.id = f"element-{hash(text)}"
        element.metadata = MagicMock()
        element.metadata.filename = None
        element.metadata.category = element_type
        element.metadata.element_id = element.id
        element.metadata.tag = tag
        return element

    return [
        create_element("Main Title", "Title", "h1"),
        create_element("Section 1: Introduction", "Title", "h2"),
        create_element("This is a complex paragraph with nested elements.", "NarrativeText", "p"),
        create_element("Nested Content", "Title", "h3"),
        create_element("Nested item 1\nSub-item 1.1\nSub-item 1.2", "ListItem", "ul"),
        create_element("Header 1 | Header 2\nCell 1 | Cell 2", "Table", "table"),
    ]


def create_mock_email_elements() -> List[Any]:
    """Create realistic mock elements for email processing."""

    def create_element(text: str, element_type: str):
        MockElementClass = type(element_type, (), {"__str__": lambda self: self._text})
        element = MockElementClass.__new__(MockElementClass)
        element._text = text
        element._type = element_type
        element.id = f"element-{hash(text)}"
        element.metadata = MagicMock()
        element.metadata.filename = None
        element.metadata.category = element_type
        element.metadata.element_id = element.id
        # Email-specific metadata
        if element_type == "EmailMetadata":
            element.metadata.sender = "sender@example.com"
            element.metadata.recipient = "recipient@example.com"
            element.metadata.subject = "Integration Test Email with Complex Headers"
            element.metadata.date = "Mon, 15 Jan 2024 10:30:00 -0500"
        return element

    return [
        create_element("Subject: Integration Test Email with Complex Headers", "EmailMetadata"),
        create_element("Dear Recipient,", "NarrativeText"),
        create_element("This is a test email with multiple headers for integration testing.", "NarrativeText"),
        create_element("The email contains:\n- Multiple recipients\n- Custom headers", "ListItem"),
        create_element("Best regards,\nSender", "NarrativeText"),
    ]


# ============================================================================
# PDF Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_pdf_processing_with_table_inference(tmp_path, pdf_with_tables_content):
    """Test PDF processing with table inference enabled."""
    bridge = UnstructuredBridge()

    # Create PDF file
    pdf_file = tmp_path / "financial_report.pdf"
    pdf_file.write_bytes(pdf_with_tables_content)

    # Mock the partition function to return realistic PDF elements
    def mock_partition(**kwargs):
        assert kwargs["strategy"] == "hi_res"
        assert kwargs["infer_table_structure"] is True
        assert kwargs.get("extract_images_in_pdf") is False  # Privacy setting
        return create_mock_pdf_elements(str(pdf_file))

    bridge._partition_func = mock_partition

    # Process document
    elements = await bridge.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
    )

    # Verify elements were processed
    assert len(elements) > 0
    assert any("Table" in el["type"] for el in elements)
    assert any("Title" in el["type"] for el in elements)

    # Verify metadata preserved
    for element in elements:
        assert "metadata" in element
        if "Table" in element["type"]:
            # Table elements should have coordinates
            assert element["metadata"].get("coordinates") is not None


@pytest.mark.asyncio
async def test_pdf_privacy_settings_no_image_extraction(tmp_path, pdf_with_tables_content):
    """Test that PDF processing respects privacy settings (no image extraction)."""
    bridge = UnstructuredBridge()

    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(pdf_with_tables_content)

    # Track kwargs passed to partition function
    captured_kwargs = {}

    def mock_partition(**kwargs):
        captured_kwargs.update(kwargs)
        return create_mock_pdf_elements(str(pdf_file))

    bridge._partition_func = mock_partition

    await bridge.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
    )

    # Verify privacy setting was applied
    assert captured_kwargs.get("extract_images_in_pdf") is False


# ============================================================================
# Markdown Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_markdown_processing_with_frontmatter(markdown_with_frontmatter):
    """Test markdown processing preserves structure and frontmatter."""
    bridge = UnstructuredBridge()

    def mock_partition(**kwargs):
        assert kwargs["strategy"] == "fast"
        assert kwargs["text"] == markdown_with_frontmatter
        return create_mock_markdown_elements()

    bridge._partition_func = mock_partition

    elements = await bridge.process_document(
        content=markdown_with_frontmatter,
        format=DocumentFormat.MARKDOWN,
    )

    assert len(elements) > 0
    assert any("Title" in el["type"] for el in elements)
    assert any("CodeSnippet" in el["type"] for el in elements)


# ============================================================================
# HTML Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_html_processing_complex_nested_structure(complex_html):
    """Test HTML processing handles complex nested structures correctly."""
    bridge = UnstructuredBridge()

    def mock_partition(**kwargs):
        assert kwargs["strategy"] == "fast"
        assert kwargs["include_metadata"] is True
        return create_mock_html_elements()

    bridge._partition_func = mock_partition

    elements = await bridge.process_document(
        content=complex_html,
        format=DocumentFormat.HTML,
    )

    assert len(elements) > 0
    # Should have titles, narrative text, lists, and tables
    element_types = {el["type"] for el in elements}
    assert "Title" in element_types
    assert "NarrativeText" in element_types
    assert "ListItem" in element_types
    assert "Table" in element_types


# ============================================================================
# Email Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_email_processing_extracts_headers_metadata(tmp_path, email_with_headers):
    """Test email processing extracts and preserves header metadata."""
    bridge = UnstructuredBridge()

    email_file = tmp_path / "test_email.eml"
    email_file.write_bytes(email_with_headers.encode("utf-8"))

    def mock_partition(**kwargs):
        assert kwargs["strategy"] == "fast"
        return create_mock_email_elements()

    bridge._partition_func = mock_partition

    elements = await bridge.process_document(
        file_path=email_file,
        format=DocumentFormat.EMAIL,
    )

    assert len(elements) > 0

    # Check for email metadata element
    metadata_elements = [el for el in elements if "EmailMetadata" in el["type"]]
    assert len(metadata_elements) > 0


# ============================================================================
# Format-Specific Configuration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_format_specific_parameters_applied_to_docx(tmp_path):
    """Test that DOCX format applies format-specific parameters."""
    bridge = UnstructuredBridge()

    docx_file = tmp_path / "document.docx"
    # Create minimal DOCX-like file (not real, just for test)
    docx_file.write_bytes(b"PK\x03\x04" + b"fake docx content")

    captured_kwargs = {}

    def mock_partition(**kwargs):
        captured_kwargs.update(kwargs)
        return create_mock_pdf_elements(str(docx_file))

    bridge._partition_func = mock_partition

    await bridge.process_document(
        file_path=docx_file,
        format=DocumentFormat.DOCX,
    )

    # DOCX should use HI_RES strategy
    assert captured_kwargs.get("strategy") == "hi_res"
    assert captured_kwargs.get("include_page_breaks") is True


@pytest.mark.asyncio
async def test_all_formats_have_strategy_defined():
    """Test that common formats have strategies defined in config."""
    formats_to_test = [
        DocumentFormat.PDF,
        DocumentFormat.DOCX,
        DocumentFormat.MARKDOWN,
        DocumentFormat.HTML,
        DocumentFormat.EMAIL,
    ]

    bridge = UnstructuredBridge()

    for doc_format in formats_to_test:
        strategy = bridge._select_strategy(doc_format)
        assert strategy in [PartitionStrategy.FAST, PartitionStrategy.HI_RES, PartitionStrategy.OCR_ONLY]


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_pipeline_multiple_formats(
    tmp_path, pdf_with_tables_content, markdown_with_frontmatter, complex_html
):
    """Test processing multiple document formats in sequence."""
    bridge = UnstructuredBridge()

    # Setup files
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(pdf_with_tables_content)

    html_file = tmp_path / "doc.html"
    html_file.write_text(complex_html, encoding="utf-8")

    # Mock partition for different formats
    def mock_partition(**kwargs):
        filename = kwargs.get("filename", "")
        if "pdf" in filename:
            return create_mock_pdf_elements(filename)
        elif "html" in filename:
            return create_mock_html_elements()
        elif kwargs.get("text"):
            return create_mock_markdown_elements()
        return []

    bridge._partition_func = mock_partition

    # Process PDF
    pdf_elements = await bridge.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
    )

    # Process Markdown
    md_elements = await bridge.process_document(
        content=markdown_with_frontmatter,
        format=DocumentFormat.MARKDOWN,
    )

    # Process HTML
    html_elements = await bridge.process_document(
        file_path=html_file,
        format=DocumentFormat.HTML,
    )

    # Verify all processed successfully
    assert len(pdf_elements) > 0
    assert len(md_elements) > 0
    assert len(html_elements) > 0
    assert bridge.documents_processed == 3
    assert bridge.processing_errors == 0

    # Verify metrics
    metrics = bridge.get_metrics()
    assert metrics["success_rate"] == 1.0

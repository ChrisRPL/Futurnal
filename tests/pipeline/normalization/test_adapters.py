"""Unit tests for format adapters.

Tests all adapter implementations for normalize() and validate() methods,
format-specific metadata extraction, and error handling.
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
    TextAdapter,
)
from futurnal.pipeline.normalization.registry import AdapterError


# ============================================================================
# MarkdownAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_markdown_adapter_normalize(temp_file):
    """Test MarkdownAdapter normalize with valid markdown."""
    content = """# Test Document

## Section 1
Some content with **bold** and *italic*.

#tag1 #tag2
"""
    file_path = temp_file(content, "test.md")
    adapter = MarkdownAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert doc.metadata.format == DocumentFormat.MARKDOWN
    assert doc.metadata.source_path == str(file_path)
    assert "markdown" in doc.metadata.extra
    assert doc.metadata.character_count > 0


@pytest.mark.asyncio
async def test_markdown_adapter_validate(temp_file):
    """Test MarkdownAdapter validate method."""
    adapter = MarkdownAdapter()

    # Valid markdown file
    md_file = temp_file("# Test", "test.md")
    assert await adapter.validate(md_file) is True

    # Invalid extension
    txt_file = temp_file("# Test", "test.txt")
    assert await adapter.validate(txt_file) is False

    # Non-existent file
    fake_file = Path("/nonexistent/file.md")
    assert await adapter.validate(fake_file) is False


# ============================================================================
# EmailAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_email_adapter_normalize(tmp_path):
    """Test EmailAdapter normalize with valid email."""
    # Create a simple RFC822 email
    email_content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000

This is the email body.
"""
    file_path = tmp_path / "test.eml"
    file_path.write_bytes(email_content.encode())
    adapter = EmailAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="email-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert doc.metadata.format == DocumentFormat.EMAIL
    assert "email" in doc.metadata.extra
    assert doc.metadata.extra["email"]["subject"] == "Test Email"
    assert doc.metadata.extra["email"]["from"] == "sender@example.com"


@pytest.mark.asyncio
async def test_email_adapter_validate(tmp_path):
    """Test EmailAdapter validate method."""
    adapter = EmailAdapter()

    # Valid .eml file
    email_content = b"From: test@example.com\nSubject: Test\n\nBody"
    eml_file = tmp_path / "test.eml"
    eml_file.write_bytes(email_content)
    assert await adapter.validate(eml_file) is True

    # Invalid extension
    txt_file = tmp_path / "test.txt"
    txt_file.write_bytes(b"test")
    assert await adapter.validate(txt_file) is False


# ============================================================================
# HTMLAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_html_adapter_normalize(temp_file):
    """Test HTMLAdapter normalize with valid HTML."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Page</title>
    <meta name="description" content="Test description">
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph.</p>
    <table>
        <tr><td>Cell 1</td></tr>
    </table>
</body>
</html>
"""
    file_path = temp_file(html_content, "test.html")
    adapter = HTMLAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="html-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert doc.metadata.format == DocumentFormat.HTML
    assert "html" in doc.metadata.extra
    assert doc.metadata.extra["html"]["title"] == "Test Page"
    assert doc.metadata.extra["html"]["table_count"] > 0
    assert doc.metadata.extra["html"]["heading_count"] > 0


@pytest.mark.asyncio
async def test_html_adapter_language_detection(temp_file):
    """Test HTMLAdapter language detection from HTML tag."""
    html_content = '<html lang="fr"><body>Content</body></html>'
    file_path = temp_file(html_content, "test.html")
    adapter = HTMLAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="html-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.language == "fr"


@pytest.mark.asyncio
async def test_html_adapter_validate(temp_file):
    """Test HTMLAdapter validate method."""
    adapter = HTMLAdapter()

    # Valid HTML file
    html_file = temp_file("<html><body>Test</body></html>", "test.html")
    assert await adapter.validate(html_file) is True

    # Valid .htm file
    htm_file = temp_file("<html><body>Test</body></html>", "test.htm")
    assert await adapter.validate(htm_file) is True

    # Invalid extension
    txt_file = temp_file("<html>test</html>", "test.txt")
    assert await adapter.validate(txt_file) is False


# ============================================================================
# CodeAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_code_adapter_normalize_python(temp_file):
    """Test CodeAdapter normalize with Python code."""
    code_content = '''"""Module docstring."""

def test_function():
    # This is a comment
    return 42

class TestClass:
    """Class docstring."""
    pass
'''
    file_path = temp_file(code_content, "test.py")
    adapter = CodeAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="code-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert doc.metadata.format == DocumentFormat.CODE
    assert "code" in doc.metadata.extra
    assert doc.metadata.extra["code"]["language"] == "python"
    assert doc.metadata.extra["code"]["comment_count"] > 0
    assert doc.metadata.extra["code"]["has_functions"] is True
    assert doc.metadata.extra["code"]["has_classes"] is True


@pytest.mark.asyncio
async def test_code_adapter_normalize_javascript(temp_file):
    """Test CodeAdapter normalize with JavaScript code."""
    code_content = """// Single line comment
function testFunc() {
    /* Multi-line
       comment */
    return true;
}
"""
    file_path = temp_file(code_content, "test.js")
    adapter = CodeAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="code-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.extra["code"]["language"] == "javascript"
    assert doc.metadata.extra["code"]["comment_count"] >= 2


@pytest.mark.asyncio
async def test_code_adapter_validate(temp_file):
    """Test CodeAdapter validate method."""
    adapter = CodeAdapter()

    # Valid Python file
    py_file = temp_file("print('hello')", "test.py")
    assert await adapter.validate(py_file) is True

    # Valid JavaScript file
    js_file = temp_file("console.log('hello');", "test.js")
    assert await adapter.validate(js_file) is True

    # Unknown extension
    unknown_file = temp_file("code", "test.xyz")
    assert await adapter.validate(unknown_file) is False


# ============================================================================
# PDFAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_pdf_adapter_normalize(temp_pdf_file):
    """Test PDFAdapter normalize with PDF file."""
    file_path = temp_pdf_file("test.pdf")
    adapter = PDFAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="pdf-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert doc.metadata.format == DocumentFormat.PDF
    assert "pdf" in doc.metadata.extra
    assert adapter.requires_unstructured_processing is True


@pytest.mark.asyncio
async def test_pdf_adapter_validate(temp_pdf_file):
    """Test PDFAdapter validate method."""
    adapter = PDFAdapter()

    # Valid PDF file
    pdf_file = temp_pdf_file("test.pdf")
    assert await adapter.validate(pdf_file) is True

    # Invalid extension
    from pathlib import Path

    txt_file = Path("/tmp/test.txt")
    assert await adapter.validate(txt_file) is False


# ============================================================================
# TextAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_text_adapter_normalize(temp_file):
    """Test TextAdapter normalize with plain text."""
    content = "This is plain text content.\nWith multiple lines."
    file_path = temp_file(content, "test.txt")
    adapter = TextAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="text-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert doc.metadata.format == DocumentFormat.TEXT
    assert "text" in doc.metadata.extra
    assert doc.content == content


@pytest.mark.asyncio
async def test_text_adapter_code_detection(temp_file):
    """Test TextAdapter detects code files."""
    code_content = """import sys

def main():
    print("Hello")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
    file_path = temp_file(code_content, "script.py")
    adapter = TextAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="text-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.format == DocumentFormat.CODE
    assert doc.metadata.extra["text"]["is_code"] is True


@pytest.mark.asyncio
async def test_text_adapter_validate(temp_file):
    """Test TextAdapter validate method."""
    adapter = TextAdapter()

    # Any existing file is valid for TextAdapter
    txt_file = temp_file("content", "test.txt")
    assert await adapter.validate(txt_file) is True

    # Non-existent file
    fake_file = Path("/nonexistent/file.txt")
    assert await adapter.validate(fake_file) is False


# ============================================================================
# GenericAdapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generic_adapter_normalize(temp_file):
    """Test GenericAdapter normalize with unknown format."""
    content = "Unknown format content"
    file_path = temp_file(content, "test.xyz")
    adapter = GenericAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="generic-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.document_id is not None
    assert "generic_adapter" in doc.metadata.extra
    assert doc.metadata.extra["generic_adapter"]["used_fallback"] is True


@pytest.mark.asyncio
async def test_generic_adapter_validate(temp_file, tmp_path):
    """Test GenericAdapter validate method."""
    adapter = GenericAdapter()

    # Any existing file is valid
    unknown_file = temp_file("content", "test.xyz")
    assert await adapter.validate(unknown_file) is True

    # Binary file is also valid
    binary_file = tmp_path / "test.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")
    assert await adapter.validate(binary_file) is True


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_adapter_error_nonexistent_file():
    """Test adapters handle nonexistent files gracefully."""
    fake_path = Path("/nonexistent/file.md")
    adapter = MarkdownAdapter()

    with pytest.raises(AdapterError):
        await adapter.normalize(
            file_path=fake_path,
            source_id="test-123",
            source_type="local_files",
            source_metadata={},
        )


@pytest.mark.asyncio
async def test_adapter_requires_unstructured_flags():
    """Test adapters correctly set requires_unstructured_processing flag."""
    # Adapters that don't require Unstructured.io
    assert MarkdownAdapter().requires_unstructured_processing is False
    assert EmailAdapter().requires_unstructured_processing is False
    assert CodeAdapter().requires_unstructured_processing is False
    assert TextAdapter().requires_unstructured_processing is False
    assert GenericAdapter().requires_unstructured_processing is False

    # Adapters that require Unstructured.io
    assert PDFAdapter().requires_unstructured_processing is True
    assert HTMLAdapter().requires_unstructured_processing is True


# ============================================================================
# Metadata Extraction Tests
# ============================================================================


@pytest.mark.asyncio
async def test_adapter_temporal_metadata(temp_file):
    """Test adapters extract temporal metadata correctly."""
    content = "Test content"
    file_path = temp_file(content, "test.txt")
    adapter = TextAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.created_at is not None
    assert doc.metadata.modified_at is not None
    assert doc.metadata.ingested_at is not None


@pytest.mark.asyncio
async def test_adapter_content_statistics(temp_file):
    """Test adapters compute content statistics correctly."""
    content = "Word one two three.\nLine two.\nLine three."
    file_path = temp_file(content, "test.txt")
    adapter = TextAdapter()

    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.character_count == len(content)
    assert doc.metadata.word_count == len(content.split())
    assert doc.metadata.line_count == content.count("\n") + 1


# ============================================================================
# Encoding Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_text_adapter_encoding_handling(temp_file, tmp_path):
    """Test TextAdapter handles different encodings."""
    # UTF-8 content
    content_utf8 = "Héllo wörld 日本語"
    file_path = tmp_path / "test_utf8.txt"
    file_path.write_text(content_utf8, encoding="utf-8")

    adapter = TextAdapter()
    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={},
    )

    assert "Héllo" in doc.content or "Hello" in doc.content  # May have encoding fallback


@pytest.mark.asyncio
async def test_code_adapter_encoding_handling(temp_file, tmp_path):
    """Test CodeAdapter handles different encodings."""
    # Latin-1 encoded Python code
    content = "# Cómment\ndef func():\n    pass"
    file_path = tmp_path / "test.py"
    file_path.write_text(content, encoding="latin-1")

    adapter = CodeAdapter()
    doc = await adapter.normalize(
        file_path=file_path,
        source_id="test-123",
        source_type="local_files",
        source_metadata={},
    )

    assert doc.metadata.format == DocumentFormat.CODE
    assert "code" in doc.metadata.extra

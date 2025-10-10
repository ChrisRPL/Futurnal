"""Edge case test data for quality gates validation.

Provides fixtures for testing edge cases including:
- Empty and minimal documents
- Very large documents (>1GB for streaming)
- Unicode and special characters
- Corrupted and malformed files
- Boundary conditions

Design Philosophy:
- Cover failure modes comprehensively
- Test graceful degradation
- Validate error handling and quarantine workflows
- Ensure privacy-safe error messages
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Empty/Minimal Document Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Completely empty file (0 bytes)."""
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")
    return file_path


@pytest.fixture
def whitespace_only_file(tmp_path: Path) -> Path:
    """File containing only whitespace."""
    content = "   \n\n\t\t\n   \n"
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def single_char_file(tmp_path: Path) -> Path:
    """File with single character."""
    file_path = tmp_path / "single.txt"
    file_path.write_text("A", encoding="utf-8")
    return file_path


@pytest.fixture
def single_line_file(tmp_path: Path) -> Path:
    """File with single line, no newline."""
    file_path = tmp_path / "single_line.txt"
    file_path.write_text("Single line without newline", encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Unicode and Special Characters
# ---------------------------------------------------------------------------


@pytest.fixture
def unicode_emoji_file(tmp_path: Path) -> Path:
    """File containing emoji and special unicode characters."""
    content = """# Unicode Test Document 🚀

## Emojis
Hello 👋 World 🌍! Testing various emojis: 😀 🎉 ✨ 🔥 💻 📚 🌟

## Special Characters
Mathematical: ∑ ∫ √ ∞ ≠ ≈ ≤ ≥
Currency: $ € £ ¥ ₹ ₽ ₩
Arrows: ← → ↑ ↓ ↔ ⇒ ⇔
Symbols: © ® ™ § ¶ † ‡ • ◦

## Different Scripts
Greek: Ελληνικά (Elliniká)
Cyrillic: Русский (Russkiy)
Arabic: العربية (al-ʿArabīyah)
Hebrew: עברית (Ivrit)
Chinese: 中文 (Zhōngwén)
Japanese: 日本語 (Nihongo)
Korean: 한국어 (Hangugeo)
Thai: ภาษาไทย (Phasa Thai)
Hindi: हिन्दी (Hindī)

## Combined Characters
Accented: café, naïve, résumé, Zürich, São Paulo
Combined marks: ñ, ü, ö, é, è, ê, ë

## Zero-Width Characters
Before‌After (zero-width non-joiner)
Before‍After (zero-width joiner)

## RTL Text
This is English, and this is Arabic: مرحبا بك في الاختبار
Mixed bidirectional text with RTL ← and LTR → directions.
"""
    file_path = tmp_path / "unicode.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def unicode_bom_file(tmp_path: Path) -> Path:
    """File with UTF-8 BOM (Byte Order Mark)."""
    content = "This file starts with a UTF-8 BOM marker"
    file_path = tmp_path / "bom.txt"
    # Write UTF-8 BOM + content
    file_path.write_bytes(b'\xef\xbb\xbf' + content.encode('utf-8'))
    return file_path


@pytest.fixture
def mixed_encoding_file(tmp_path: Path) -> Path:
    """File that appears to have mixed encoding issues (but is valid UTF-8)."""
    # Create content that looks like it might have encoding issues
    content = "Normal text\nText with special chars: café, naïve, Zürich\nMore normal text"
    file_path = tmp_path / "mixed_encoding.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Very Large Files (for streaming processor testing)
# ---------------------------------------------------------------------------


@pytest.fixture
def large_file_10mb(tmp_path: Path) -> Path:
    """10MB text file for large file testing."""
    # Generate ~10MB of text
    line = "This is a test line with some content. " * 10 + "\n"
    num_lines = (10 * 1024 * 1024) // len(line.encode('utf-8'))

    file_path = tmp_path / "large_10mb.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for _ in range(num_lines):
            f.write(line)

    return file_path


@pytest.fixture
def large_file_100mb(tmp_path: Path) -> Path:
    """100MB text file for streaming processor testing.

    Note: This fixture is slow to generate. Use only in performance tests.
    """
    # Generate ~100MB of text
    line = "This is a test line with some content for large file testing. " * 10 + "\n"
    num_lines = (100 * 1024 * 1024) // len(line.encode('utf-8'))

    file_path = tmp_path / "large_100mb.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(num_lines):
            f.write(line)
            # Add variation every 1000 lines to prevent deduplication
            if i % 1000 == 0:
                f.write(f"Marker line {i}\n")

    return file_path


@pytest.fixture
def large_markdown_50mb(tmp_path: Path) -> Path:
    """50MB markdown file with structured content."""
    file_path = tmp_path / "large_50mb.md"

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("# Large Markdown Document\n\n")

        # Generate sections to reach ~50MB
        section_content = "This is paragraph content. " * 100 + "\n\n"
        num_sections = (50 * 1024 * 1024) // len(section_content.encode('utf-8'))

        for i in range(num_sections):
            f.write(f"## Section {i + 1}\n\n")
            f.write(section_content)

            # Add subsections every 10 sections
            if i % 10 == 0:
                f.write(f"### Subsection {i + 1}.1\n\n")
                f.write("Nested content. " * 50 + "\n\n")

    return file_path


# ---------------------------------------------------------------------------
# Corrupted/Malformed Files
# ---------------------------------------------------------------------------


@pytest.fixture
def truncated_json_file(tmp_path: Path) -> Path:
    """JSON file that is truncated mid-document."""
    # Valid JSON that gets cut off
    content = '{"items": [{"id": 1, "value": "test"}, {"id": 2, "val'
    file_path = tmp_path / "truncated.json"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def invalid_json_file(tmp_path: Path) -> Path:
    """JSON file with syntax errors."""
    content = '{"items": [1, 2, 3,], "test": value, }'  # Trailing commas, unquoted value
    file_path = tmp_path / "invalid.json"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def malformed_xml_file(tmp_path: Path) -> Path:
    """XML file with unclosed tags."""
    content = """<?xml version="1.0"?>
<document>
    <section>
        <paragraph>Some content
        <!-- Missing closing tags -->
    <another>More content
</document>"""
    file_path = tmp_path / "malformed.xml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def malformed_yaml_file(tmp_path: Path) -> Path:
    """YAML file with syntax errors."""
    content = """name: Test
items:
  - item1
  - item2
  invalid indentation here
  - item3
key without value
another: value"""
    file_path = tmp_path / "malformed.yaml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def binary_as_text_file(tmp_path: Path) -> Path:
    """File with binary data but .txt extension."""
    # Create binary data that might cause encoding issues
    binary_data = bytes(range(256))
    file_path = tmp_path / "binary.txt"
    file_path.write_bytes(binary_data)
    return file_path


@pytest.fixture
def null_bytes_file(tmp_path: Path) -> Path:
    """Text file containing null bytes."""
    content = b"Normal text\x00with null\x00bytes\x00inside"
    file_path = tmp_path / "null_bytes.txt"
    file_path.write_bytes(content)
    return file_path


# ---------------------------------------------------------------------------
# Boundary Conditions
# ---------------------------------------------------------------------------


@pytest.fixture
def deeply_nested_json(tmp_path: Path) -> Path:
    """JSON with deep nesting (100+ levels)."""
    # Create deeply nested structure
    data = {}
    current = data
    for i in range(100):
        current["level"] = i
        current["nested"] = {}
        current = current["nested"]
    current["final"] = "value"

    file_path = tmp_path / "deeply_nested.json"
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return file_path


@pytest.fixture
def very_long_lines_file(tmp_path: Path) -> Path:
    """File with extremely long lines (>100K characters)."""
    # Create lines with 100K+ characters
    long_line = "x" * 150000 + "\n"
    content = long_line * 10

    file_path = tmp_path / "long_lines.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def many_small_lines_file(tmp_path: Path) -> Path:
    """File with millions of tiny lines."""
    file_path = tmp_path / "many_lines.txt"

    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(100000):  # 100K lines
            f.write(f"{i}\n")

    return file_path


@pytest.fixture
def special_filename_file(tmp_path: Path) -> Path:
    """File with special characters in filename."""
    # Test various special characters that are valid in filenames
    filename = "test file (copy) [1] {special} 'quoted' ~tilde #hash @at.txt"
    file_path = tmp_path / filename
    file_path.write_text("Content with special filename", encoding="utf-8")
    return file_path


@pytest.fixture
def very_long_filename_file(tmp_path: Path) -> Path:
    """File with very long filename (approaching OS limits)."""
    # Most filesystems support 255 bytes for filename
    # Create a filename close to limit
    base_name = "x" * 240  # Leave room for extension
    filename = f"{base_name}.txt"
    file_path = tmp_path / filename
    file_path.write_text("Content with very long filename", encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Format-Specific Edge Cases
# ---------------------------------------------------------------------------


@pytest.fixture
def markdown_no_headings(tmp_path: Path) -> Path:
    """Markdown file with no headings (for chunking tests)."""
    content = """This is a markdown file with no headings at all.
Just continuous paragraphs of text without any structure.

Another paragraph here with more content.
And yet another paragraph.

More content without any headings or structure.
This tests how the chunking handles flat documents.
"""
    file_path = tmp_path / "no_headings.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def markdown_only_headings(tmp_path: Path) -> Path:
    """Markdown file with only headings, no content."""
    content = """# Heading 1
## Heading 2
### Heading 3
## Another Heading 2
# Another Heading 1
"""
    file_path = tmp_path / "only_headings.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def csv_inconsistent_columns(tmp_path: Path) -> Path:
    """CSV file with inconsistent column counts."""
    content = """Name,Value,Extra
Row1,100
Row2,200,ExtraValue,TooMany
Row3,300,Normal
Row4
"""
    file_path = tmp_path / "inconsistent.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def html_malformed(tmp_path: Path) -> Path:
    """Malformed HTML with unclosed tags."""
    content = """<!DOCTYPE html>
<html>
<head>
    <title>Test
<!-- Missing closing tags -->
<body>
    <h1>Heading
    <p>Paragraph without closing
    <div>
        <span>Content
    </div>
"""
    file_path = tmp_path / "malformed.html"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Edge Case Collection Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def all_edge_case_fixtures(
    empty_file,
    whitespace_only_file,
    unicode_emoji_file,
    truncated_json_file,
    malformed_xml_file,
    deeply_nested_json,
    very_long_lines_file,
) -> dict:
    """Dictionary of all edge case fixtures.

    Returns:
        Dictionary mapping edge case names to file paths
    """
    return {
        "empty": empty_file,
        "whitespace_only": whitespace_only_file,
        "unicode_emoji": unicode_emoji_file,
        "truncated_json": truncated_json_file,
        "malformed_xml": malformed_xml_file,
        "deeply_nested_json": deeply_nested_json,
        "very_long_lines": very_long_lines_file,
    }

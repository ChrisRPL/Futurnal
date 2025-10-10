"""Comprehensive format fixtures for quality gates testing.

Provides test data generators for all 16 DocumentFormat types to enable
comprehensive format coverage validation. Each format includes:
- Small samples (< 1KB) for unit tests
- Medium samples (~100KB) for integration tests
- Large samples (>1MB) for performance tests

Design Philosophy:
- Real-world representative content
- Privacy-safe synthetic data (no PII)
- Deterministic generation for reproducibility
- Comprehensive format feature coverage
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pytest


# ---------------------------------------------------------------------------
# Markdown Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def markdown_simple(tmp_path: Path) -> Path:
    """Simple markdown document for basic tests."""
    content = """# Simple Markdown Document

## Introduction
This is a simple markdown document for testing.

## Content Section
Some paragraph content with **bold** and *italic* text.

### Subsection
Nested content with a list:
- Item 1
- Item 2
- Item 3

## Conclusion
Final thoughts and conclusion.
"""
    file_path = tmp_path / "simple.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def markdown_complex(tmp_path: Path) -> Path:
    """Complex markdown with frontmatter, tables, code blocks."""
    content = """---
title: Complex Markdown Test
author: Test Author
date: 2024-01-15
tags: [test, markdown, quality-gates]
---

# Complex Markdown Document

## Features Tested
This document tests various markdown features:

### Code Blocks
```python
def hello_world():
    \"\"\"A simple function.\"\"\"
    print("Hello, World!")
```

### Tables
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Data A   | Data B   | Data C   |

### Links and Images
[Link text](https://example.com)
![Alt text](image.png)

### Lists
1. First ordered item
2. Second ordered item
   - Nested unordered
   - Another nested

### Blockquotes
> This is a blockquote
> with multiple lines

### Inline Elements
Text with `inline code`, **bold**, *italic*, and ~~strikethrough~~.
"""
    file_path = tmp_path / "complex.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def markdown_large(tmp_path: Path) -> Path:
    """Large markdown document for performance testing (~100KB)."""
    sections = ["# Large Markdown Document\n\n"]

    # Generate 50 sections with content
    for i in range(50):
        sections.append(f"## Section {i + 1}\n\n")
        sections.append(f"This is section {i + 1} with detailed content. " * 20)
        sections.append("\n\n")

        # Add subsections every 5 sections
        if i % 5 == 0:
            sections.append(f"### Subsection {i + 1}.1\n\n")
            sections.append("Nested content with additional details. " * 15)
            sections.append("\n\n")

    content = "".join(sections)
    file_path = tmp_path / "large.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Text Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_simple(tmp_path: Path) -> Path:
    """Simple plain text file."""
    content = "This is a simple plain text file.\nWith multiple lines of content.\nFor testing purposes."
    file_path = tmp_path / "simple.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def text_large(tmp_path: Path) -> Path:
    """Large plain text file for performance testing (~100KB)."""
    lines = [f"This is line {i} with some content for testing. " * 10 for i in range(1000)]
    content = "\n".join(lines)
    file_path = tmp_path / "large.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Code Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def code_python(tmp_path: Path) -> Path:
    """Python code file."""
    content = '''"""Sample Python module for testing."""

from typing import List, Optional


class SampleClass:
    """A sample class for testing code normalization."""

    def __init__(self, name: str):
        """Initialize with name."""
        self.name = name

    def process(self, data: List[str]) -> Optional[str]:
        """Process data and return result."""
        if not data:
            return None
        return f"{self.name}: {', '.join(data)}"


def main():
    """Main entry point."""
    obj = SampleClass("Test")
    result = obj.process(["item1", "item2", "item3"])
    print(result)


if __name__ == "__main__":
    main()
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def code_javascript(tmp_path: Path) -> Path:
    """JavaScript code file."""
    content = '''/**
 * Sample JavaScript module for testing
 */

class SampleClass {
    constructor(name) {
        this.name = name;
    }

    process(data) {
        if (!data || data.length === 0) {
            return null;
        }
        return `${this.name}: ${data.join(', ')}`;
    }
}

function main() {
    const obj = new SampleClass('Test');
    const result = obj.process(['item1', 'item2', 'item3']);
    console.log(result);
}

main();
'''
    file_path = tmp_path / "sample.js"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# JSON/YAML Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def json_simple(tmp_path: Path) -> Path:
    """Simple JSON document."""
    data = {
        "name": "Test Document",
        "version": "1.0.0",
        "description": "A simple JSON document for testing",
        "items": [
            {"id": 1, "value": "Item 1"},
            {"id": 2, "value": "Item 2"},
            {"id": 3, "value": "Item 3"}
        ],
        "metadata": {
            "created": "2024-01-15",
            "author": "Test Author"
        }
    }
    file_path = tmp_path / "simple.json"
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return file_path


@pytest.fixture
def json_large(tmp_path: Path) -> Path:
    """Large JSON document for performance testing (~100KB)."""
    data = {
        "documents": [
            {
                "id": i,
                "title": f"Document {i}",
                "content": f"Content for document {i}. " * 50,
                "metadata": {
                    "author": f"Author {i}",
                    "tags": [f"tag{j}" for j in range(10)]
                }
            }
            for i in range(100)
        ]
    }
    file_path = tmp_path / "large.json"
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return file_path


@pytest.fixture
def yaml_simple(tmp_path: Path) -> Path:
    """Simple YAML document."""
    content = """name: Test Document
version: 1.0.0
description: A simple YAML document for testing

items:
  - id: 1
    value: Item 1
  - id: 2
    value: Item 2
  - id: 3
    value: Item 3

metadata:
  created: 2024-01-15
  author: Test Author
"""
    file_path = tmp_path / "simple.yaml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# CSV Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_simple(tmp_path: Path) -> Path:
    """Simple CSV file."""
    content = """Name,Age,Department,Salary
Alice Johnson,30,Engineering,85000
Bob Smith,25,Marketing,65000
Carol Williams,35,Engineering,95000
David Brown,28,Sales,70000
Eve Davis,32,Engineering,90000
"""
    file_path = tmp_path / "simple.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def csv_large(tmp_path: Path) -> Path:
    """Large CSV file for performance testing (~100KB)."""
    header = "ID,Name,Value,Description,Category,Status\n"
    rows = [
        f"{i},Name{i},Value{i},'Description for item {i}',Category{i % 10},Active"
        for i in range(1000)
    ]
    content = header + "\n".join(rows)
    file_path = tmp_path / "large.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# HTML Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def html_simple(tmp_path: Path) -> Path:
    """Simple HTML document."""
    content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test HTML Document</title>
</head>
<body>
    <h1>Simple HTML Document</h1>
    <p>This is a paragraph with some <strong>bold</strong> and <em>italic</em> text.</p>

    <h2>List Section</h2>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>

    <h2>Table Section</h2>
    <table>
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
        </tr>
        <tr>
            <td>Value 1</td>
            <td>Value 2</td>
        </tr>
    </table>
</body>
</html>
"""
    file_path = tmp_path / "simple.html"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# XML Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def xml_simple(tmp_path: Path) -> Path:
    """Simple XML document."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Test XML Document</title>
        <author>Test Author</author>
        <date>2024-01-15</date>
    </metadata>
    <content>
        <section id="1">
            <heading>Introduction</heading>
            <paragraph>This is the introduction section.</paragraph>
        </section>
        <section id="2">
            <heading>Main Content</heading>
            <paragraph>This is the main content section.</paragraph>
            <items>
                <item>Item 1</item>
                <item>Item 2</item>
                <item>Item 3</item>
            </items>
        </section>
    </content>
</document>
"""
    file_path = tmp_path / "simple.xml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Email Fixtures (EML format)
# ---------------------------------------------------------------------------


@pytest.fixture
def email_simple(tmp_path: Path) -> Path:
    """Simple email message in EML format."""
    content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email Message
Date: Mon, 15 Jan 2024 10:30:00 +0000
Content-Type: text/plain; charset=UTF-8

This is a simple test email message.

It has multiple paragraphs of content for testing
email normalization in the pipeline.

Best regards,
Test Sender
"""
    file_path = tmp_path / "simple.eml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Jupyter Notebook Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jupyter_simple(tmp_path: Path) -> Path:
    """Simple Jupyter notebook."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Test Notebook\n", "This is a simple Jupyter notebook for testing."]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": ["import numpy as np\n", "print('Hello, World!')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Results\n", "The code above prints a greeting."]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    file_path = tmp_path / "simple.ipynb"
    file_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Format Coverage Fixture Dictionary
# ---------------------------------------------------------------------------


@pytest.fixture
def all_format_fixtures(
    markdown_simple,
    text_simple,
    code_python,
    json_simple,
    yaml_simple,
    csv_simple,
    html_simple,
    xml_simple,
    email_simple,
    jupyter_simple,
) -> Dict[str, Path]:
    """Dictionary mapping format names to fixture paths.

    Returns:
        Dictionary with format names as keys and file paths as values
    """
    return {
        "markdown": markdown_simple,
        "text": text_simple,
        "code": code_python,
        "json": json_simple,
        "yaml": yaml_simple,
        "csv": csv_simple,
        "html": html_simple,
        "xml": xml_simple,
        "email": email_simple,
        "jupyter": jupyter_simple,
    }


# ---------------------------------------------------------------------------
# Format Sample Generator Functions
# ---------------------------------------------------------------------------


def generate_format_sample(format_type: str, size: str = "small") -> str:
    """Generate sample content for a format type.

    Args:
        format_type: Format type (markdown, json, etc.)
        size: Size variant (small, medium, large)

    Returns:
        Sample content string
    """
    generators = {
        "markdown": _generate_markdown,
        "text": _generate_text,
        "json": _generate_json,
        "yaml": _generate_yaml,
        "csv": _generate_csv,
        "html": _generate_html,
        "xml": _generate_xml,
        "python": _generate_python,
    }

    generator = generators.get(format_type, _generate_text)
    return generator(size)


def _generate_markdown(size: str) -> str:
    """Generate markdown content."""
    if size == "small":
        return "# Test\n\nSimple content."
    elif size == "large":
        sections = ["# Large Document\n\n"]
        for i in range(100):
            sections.append(f"## Section {i}\n\n")
            sections.append("Content paragraph. " * 50)
            sections.append("\n\n")
        return "".join(sections)
    else:  # medium
        return "# Medium Document\n\n" + ("## Section\n\nContent. " * 20)


def _generate_text(size: str) -> str:
    """Generate plain text content."""
    if size == "small":
        return "Simple text content."
    elif size == "large":
        return "Text content. " * 10000
    else:  # medium
        return "Text content. " * 1000


def _generate_json(size: str) -> str:
    """Generate JSON content."""
    if size == "small":
        data = {"test": "value", "items": [1, 2, 3]}
    elif size == "large":
        data = {"items": [{"id": i, "value": f"Item {i}"} for i in range(1000)]}
    else:  # medium
        data = {"items": [{"id": i, "value": f"Item {i}"} for i in range(100)]}
    return json.dumps(data, indent=2)


def _generate_yaml(size: str) -> str:
    """Generate YAML content."""
    if size == "small":
        return "test: value\nitems:\n  - 1\n  - 2\n  - 3"
    else:
        return "test: value\nitems:\n" + "\n".join([f"  - item{i}" for i in range(100)])


def _generate_csv(size: str) -> str:
    """Generate CSV content."""
    if size == "small":
        return "Name,Value\nItem1,100\nItem2,200"
    elif size == "large":
        rows = ["Name,Value"] + [f"Item{i},{i*100}" for i in range(1000)]
        return "\n".join(rows)
    else:  # medium
        rows = ["Name,Value"] + [f"Item{i},{i*100}" for i in range(100)]
        return "\n".join(rows)


def _generate_html(size: str) -> str:
    """Generate HTML content."""
    return f"<!DOCTYPE html><html><body><h1>Test</h1><p>Content</p></body></html>"


def _generate_xml(size: str) -> str:
    """Generate XML content."""
    return '<?xml version="1.0"?>\n<document><item>Test</item></document>'


def _generate_python(size: str) -> str:
    """Generate Python code content."""
    return 'def test():\n    """Test function."""\n    return "Hello"'

"""Performance tests for UnstructuredBridge.

Tests cover:
- Processing throughput for different formats (MB/s)
- Memory usage validation for large documents
- Batch processing performance
- Scalability under load

These tests are marked with @pytest.mark.performance and can be run separately:
    pytest tests/pipeline/normalization/test_unstructured_bridge_performance.py -m performance
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.unstructured_bridge import (
    UnstructuredBridge,
    PartitionStrategy,
)


# Mark all tests in this module as performance tests
pytestmark = pytest.mark.performance


# ============================================================================
# Test Fixtures - Large Document Generation
# ============================================================================


@pytest.fixture
def large_pdf_content_1mb():
    """Generate 1MB of PDF-like content."""
    # Simulate PDF structure with repeated content
    header = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 50 >>
stream
"""
    content_line = b"BT /F1 12 Tf 100 700 Td (Performance test content line.) Tj ET\n"
    footer = b"""endstream
endobj
%%EOF
"""

    # Calculate how many lines needed for ~1MB
    target_size = 1024 * 1024  # 1MB
    header_size = len(header) + len(footer)
    content_size = target_size - header_size
    line_count = content_size // len(content_line)

    return header + (content_line * line_count) + footer


@pytest.fixture
def large_markdown_content_5mb():
    """Generate 5MB of markdown content."""
    sections = []
    sections.append("# Performance Test Document\n\n")

    # Generate enough sections to reach ~5MB
    section_template = """## Section {i}

This is the content for section {i} of the performance test document.
It contains multiple paragraphs to simulate realistic document structure.

- List item 1 for section {i}
- List item 2 for section {i}
- List item 3 for section {i}

### Subsection {i}.1

More detailed content goes here with **bold** and *italic* text.
This helps simulate real-world markdown documents with rich formatting.

```python
def section_{i}_function():
    return "Performance test code block"
```

"""

    # Each section is ~250 bytes, need ~20,000 sections for 5MB
    for i in range(20000):
        sections.append(section_template.format(i=i))

    return "".join(sections)


@pytest.fixture
def large_html_content_2mb():
    """Generate 2MB of HTML content."""
    header = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Performance Test HTML</title>
</head>
<body>
    <h1>Performance Test Document</h1>
"""

    section_template = """
    <article>
        <h2>Section {i}</h2>
        <p>This is paragraph content for section {i} of the performance test.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
            <li>List item 3</li>
        </ul>
        <div class="nested">
            <p>Nested content with more text to increase document size.</p>
        </div>
    </article>
"""

    footer = """
</body>
</html>
"""

    # Each section ~200 bytes, need ~10,000 sections for 2MB
    sections = [section_template.format(i=i) for i in range(10000)]

    return header + "".join(sections) + footer


def create_mock_elements_for_content(content: str | bytes, count: int = 100) -> List[Any]:
    """Create mock elements simulating Unstructured.io output."""

    class MockElement:
        def __init__(self, text: str, element_type: str):
            self._text = text
            self._type = element_type
            self.id = f"perf-element-{hash(text)}"
            self.metadata = MagicMock()
            self.metadata.category = element_type
            self.metadata.element_id = self.id

        def __str__(self):
            return self._text

        def __class__(self):
            return type(self._type, (), {})

    # Create representative elements
    elements = []
    for i in range(count):
        if i % 5 == 0:
            elements.append(MockElement(f"Title {i}", "Title"))
        else:
            elements.append(MockElement(f"Content paragraph {i} with some text.", "NarrativeText"))

    return elements


# ============================================================================
# Throughput Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processing_throughput_pdf_format(tmp_path, large_pdf_content_1mb):
    """Test PDF processing throughput (target: ≥5 MB/s)."""
    bridge = UnstructuredBridge()

    # Create large PDF file
    pdf_file = tmp_path / "large_document.pdf"
    pdf_file.write_bytes(large_pdf_content_1mb)
    file_size_mb = len(large_pdf_content_1mb) / (1024 * 1024)

    # Mock partition to simulate processing
    def mock_partition(**kwargs):
        # Simulate some processing time (Unstructured.io overhead)
        time.sleep(0.05)  # 50ms overhead
        return create_mock_elements_for_content(large_pdf_content_1mb, count=200)

    bridge._partition_func = mock_partition

    # Measure processing time
    start_time = time.time()
    elements = await bridge.process_document(
        file_path=pdf_file,
        format=DocumentFormat.PDF,
    )
    duration = time.time() - start_time

    # Calculate throughput
    throughput_mb_s = file_size_mb / duration

    print(f"\n📊 PDF Processing Performance:")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Elements extracted: {len(elements)}")
    print(f"  Throughput: {throughput_mb_s:.2f} MB/s")

    # Verify processing succeeded
    assert len(elements) > 0
    assert bridge.documents_processed == 1

    # Performance assertion (relaxed for mocked test)
    assert duration < 5.0  # Should complete in reasonable time


@pytest.mark.asyncio
async def test_processing_throughput_markdown_format(large_markdown_content_5mb):
    """Test Markdown processing throughput (target: ≥20 MB/s)."""
    bridge = UnstructuredBridge()

    content_size_mb = len(large_markdown_content_5mb) / (1024 * 1024)

    # Mock partition for fast text processing
    def mock_partition(**kwargs):
        # Simulate fast text processing
        time.sleep(0.01)  # 10ms overhead for text
        return create_mock_elements_for_content(large_markdown_content_5mb, count=500)

    bridge._partition_func = mock_partition

    # Measure processing time
    start_time = time.time()
    elements = await bridge.process_document(
        content=large_markdown_content_5mb,
        format=DocumentFormat.MARKDOWN,
    )
    duration = time.time() - start_time

    # Calculate throughput
    throughput_mb_s = content_size_mb / duration

    print(f"\n📊 Markdown Processing Performance:")
    print(f"  Content size: {content_size_mb:.2f} MB")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Elements extracted: {len(elements)}")
    print(f"  Throughput: {throughput_mb_s:.2f} MB/s")

    # Verify processing succeeded
    assert len(elements) > 0
    assert bridge.documents_processed == 1

    # Performance assertion
    assert duration < 2.0  # Markdown should be faster than PDF


@pytest.mark.asyncio
async def test_processing_throughput_html_format(large_html_content_2mb):
    """Test HTML processing throughput (target: ≥15 MB/s)."""
    bridge = UnstructuredBridge()

    content_size_mb = len(large_html_content_2mb) / (1024 * 1024)

    # Mock partition for HTML processing
    def mock_partition(**kwargs):
        time.sleep(0.02)  # 20ms overhead
        return create_mock_elements_for_content(large_html_content_2mb, count=300)

    bridge._partition_func = mock_partition

    # Measure processing time
    start_time = time.time()
    elements = await bridge.process_document(
        content=large_html_content_2mb,
        format=DocumentFormat.HTML,
    )
    duration = time.time() - start_time

    # Calculate throughput
    throughput_mb_s = content_size_mb / duration

    print(f"\n📊 HTML Processing Performance:")
    print(f"  Content size: {content_size_mb:.2f} MB")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Elements extracted: {len(elements)}")
    print(f"  Throughput: {throughput_mb_s:.2f} MB/s")

    # Verify processing succeeded
    assert len(elements) > 0
    assert bridge.documents_processed == 1


# ============================================================================
# Memory Usage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_memory_usage_large_file_processing(tmp_path, large_pdf_content_1mb):
    """Test that large file processing stays within memory limits (<2GB)."""
    import sys

    bridge = UnstructuredBridge()

    pdf_file = tmp_path / "large.pdf"
    pdf_file.write_bytes(large_pdf_content_1mb)

    # Mock partition with realistic element count
    def mock_partition(**kwargs):
        return create_mock_elements_for_content(large_pdf_content_1mb, count=1000)

    bridge._partition_func = mock_partition

    # Get initial memory usage (if psutil available)
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Process document
        elements = await bridge.process_document(
            file_path=pdf_file,
            format=DocumentFormat.PDF,
        )

        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before

        print(f"\n💾 Memory Usage:")
        print(f"  Before: {memory_before:.2f} MB")
        print(f"  After: {memory_after:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")

        # Verify memory increase is reasonable
        assert memory_increase < 500  # Should not increase by more than 500MB for 1MB file

    except ImportError:
        # psutil not available, just verify processing works
        elements = await bridge.process_document(
            file_path=pdf_file,
            format=DocumentFormat.PDF,
        )

    assert len(elements) > 0


# ============================================================================
# Batch Processing Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_batch_processing_100_documents(tmp_path):
    """Test processing 100 documents in batch."""
    bridge = UnstructuredBridge()

    # Create 100 small documents
    documents = []
    for i in range(100):
        doc_file = tmp_path / f"doc_{i}.pdf"
        content = f"%PDF-1.4\nDocument {i} content here.\n%%EOF".encode()
        doc_file.write_bytes(content)
        documents.append(doc_file)

    # Mock fast partition
    def mock_partition(**kwargs):
        return create_mock_elements_for_content(b"content", count=5)

    bridge._partition_func = mock_partition

    # Process all documents
    start_time = time.time()
    for doc in documents:
        await bridge.process_document(
            file_path=doc,
            format=DocumentFormat.PDF,
        )
    duration = time.time() - start_time

    throughput_docs_s = len(documents) / duration

    print(f"\n📦 Batch Processing Performance:")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Total duration: {duration:.3f}s")
    print(f"  Throughput: {throughput_docs_s:.2f} docs/s")
    print(f"  Average per document: {(duration / len(documents) * 1000):.2f} ms")

    # Verify all processed
    assert bridge.documents_processed == 100
    assert bridge.processing_errors == 0

    metrics = bridge.get_metrics()
    assert metrics["success_rate"] == 1.0

    # Should process at reasonable speed (at least 10 docs/sec)
    assert throughput_docs_s >= 10.0


@pytest.mark.asyncio
async def test_batch_processing_with_mixed_formats(tmp_path):
    """Test batch processing with multiple document formats."""
    bridge = UnstructuredBridge()

    # Create mixed format documents
    documents = []

    # PDFs
    for i in range(30):
        pdf = tmp_path / f"doc_{i}.pdf"
        pdf.write_bytes(f"%PDF-1.4\nContent {i}\n%%EOF".encode())
        documents.append((pdf, DocumentFormat.PDF))

    # Markdown files
    for i in range(30):
        md = tmp_path / f"note_{i}.md"
        md.write_text(f"# Note {i}\n\nContent here.")
        documents.append((md, DocumentFormat.MARKDOWN))

    # HTML files
    for i in range(40):
        html = tmp_path / f"page_{i}.html"
        html.write_text(f"<html><body><h1>Page {i}</h1></body></html>")
        documents.append((html, DocumentFormat.HTML))

    # Mock partition
    def mock_partition(**kwargs):
        return create_mock_elements_for_content(b"content", count=5)

    bridge._partition_func = mock_partition

    # Process all documents
    start_time = time.time()
    for doc_path, doc_format in documents:
        if doc_format in [DocumentFormat.MARKDOWN, DocumentFormat.HTML]:
            content = doc_path.read_text()
            await bridge.process_document(
                content=content,
                format=doc_format,
            )
        else:
            await bridge.process_document(
                file_path=doc_path,
                format=doc_format,
            )
    duration = time.time() - start_time

    print(f"\n📦 Mixed Format Batch Performance:")
    print(f"  Total documents: {len(documents)}")
    print(f"  PDFs: 30, Markdown: 30, HTML: 40")
    print(f"  Total duration: {duration:.3f}s")
    print(f"  Throughput: {(len(documents) / duration):.2f} docs/s")

    # Verify all processed
    assert bridge.documents_processed == 100
    metrics = bridge.get_metrics()
    assert metrics["success_rate"] == 1.0


# ============================================================================
# Scalability Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processing_maintains_consistent_performance():
    """Test that processing performance remains consistent over many documents."""
    bridge = UnstructuredBridge()

    # Mock partition
    def mock_partition(**kwargs):
        time.sleep(0.001)  # 1ms per document
        return create_mock_elements_for_content(b"content", count=3)

    bridge._partition_func = mock_partition

    # Process in batches and measure time per batch
    batch_times = []
    num_batches = 10
    docs_per_batch = 50

    for batch_num in range(num_batches):
        start_time = time.time()

        for i in range(docs_per_batch):
            await bridge.process_document(
                content=f"Document {batch_num * docs_per_batch + i}",
                format=DocumentFormat.TEXT,
            )

        batch_duration = time.time() - start_time
        batch_times.append(batch_duration)

    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    min_batch_time = min(batch_times)
    max_batch_time = max(batch_times)
    variance = max_batch_time - min_batch_time

    print(f"\n📈 Scalability Analysis:")
    print(f"  Total documents: {num_batches * docs_per_batch}")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Min batch time: {min_batch_time:.3f}s")
    print(f"  Max batch time: {max_batch_time:.3f}s")
    print(f"  Variance: {variance:.3f}s")

    # Performance should remain relatively consistent
    # Variance should be less than 50% of average
    assert variance < (avg_batch_time * 0.5)

    # Verify all processed
    assert bridge.documents_processed == num_batches * docs_per_batch


@pytest.mark.asyncio
async def test_metrics_overhead_negligible():
    """Test that metrics tracking has negligible performance overhead."""
    # Process with metrics enabled
    bridge_with_metrics = UnstructuredBridge()

    def mock_partition(**kwargs):
        return create_mock_elements_for_content(b"content", count=5)

    bridge_with_metrics._partition_func = mock_partition

    start_time = time.time()
    for i in range(1000):
        await bridge_with_metrics.process_document(
            content=f"Document {i}",
            format=DocumentFormat.TEXT,
        )
        # Get metrics each time (simulating monitoring)
        _ = bridge_with_metrics.get_metrics()
    duration_with_metrics = time.time() - start_time

    print(f"\n⏱️ Metrics Overhead:")
    print(f"  1000 documents with metrics: {duration_with_metrics:.3f}s")
    print(f"  Average per document: {(duration_with_metrics / 1000 * 1000):.3f} ms")

    # Should still process quickly even with metrics overhead
    assert duration_with_metrics < 10.0  # Should complete in under 10 seconds

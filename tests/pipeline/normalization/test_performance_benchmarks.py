"""Performance benchmark tests for normalization pipeline.

These tests validate production performance targets:
- Overall throughput ≥5 MB/s for mixed document types
- Per-format throughput targets
- Memory usage <2GB (validated by streaming processor tests)
- Offline operation (validated by enrichment tests)

Run with: pytest -m performance tests/pipeline/normalization/test_performance_benchmarks.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.performance import PerformanceMonitor
from futurnal.pipeline.normalization.factory import create_normalization_service
from futurnal.pipeline.normalization.service import NormalizationConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def performance_config():
    """Configuration optimized for performance testing."""
    return NormalizationConfig(
        enable_chunking=True,
        default_chunk_strategy="by_title",
        max_chunk_size_chars=4000,
        chunk_overlap_chars=200,
        enable_language_detection=True,
        enable_content_classification=True,
        compute_content_hash=True,
        enable_streaming=True,
        streaming_threshold_mb=100.0,
        max_memory_mb=2048.0,
    )


@pytest.fixture
def benchmark_service(performance_config):
    """Create normalization service for benchmarking."""
    service = create_normalization_service(config=performance_config)
    # Ensure performance monitor is running
    if service.performance_monitor:
        service.performance_monitor.start()
    return service


@pytest.fixture
def test_files(tmp_path):
    """Create test files of various formats and sizes."""
    files = {}

    # Create markdown file (lightweight, ~1MB)
    md_file = tmp_path / "test.md"
    md_content = "# Test Document\n\n" + ("This is test content. " * 10000)
    md_file.write_text(md_content, encoding="utf-8")
    files["markdown"] = md_file

    # Create larger markdown file (~5MB)
    md_large = tmp_path / "large.md"
    md_large_content = "# Large Document\n\n" + ("Content paragraph. " * 100000)
    md_large.write_text(md_large_content, encoding="utf-8")
    files["markdown_large"] = md_large

    # Create text file (~2MB)
    txt_file = tmp_path / "test.txt"
    txt_content = "Plain text content. " * 50000
    txt_file.write_text(txt_content, encoding="utf-8")
    files["text"] = txt_file

    # Create JSON file (~1MB)
    json_file = tmp_path / "test.json"
    json_content = '{"data": [' + ','.join(['{"id": %d, "value": "test"}' % i for i in range(10000)]) + ']}'
    json_file.write_text(json_content, encoding="utf-8")
    files["json"] = json_file

    return files


# ---------------------------------------------------------------------------
# Overall Throughput Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestOverallThroughput:
    """Tests validating overall throughput target (≥5 MB/s)."""

    @pytest.mark.asyncio
    async def test_mixed_format_throughput(self, benchmark_service, test_files):
        """Test throughput with mixed document formats meets ≥5 MB/s target.

        This is the primary production performance requirement.
        """
        monitor = PerformanceMonitor()
        monitor.start()

        total_bytes = 0
        start_time = time.time()

        # Process all test files
        for file_type, file_path in test_files.items():
            file_size = file_path.stat().st_size
            total_bytes += file_size

            # Normalize document
            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"bench_{file_type}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            # Record in monitor
            monitor.record_document(
                size_bytes=file_size,
                format=normalized.metadata.format,
                duration_ms=doc_duration_ms,
            )

        elapsed_seconds = time.time() - start_time
        throughput_mbps = (total_bytes / (1024 * 1024)) / elapsed_seconds

        # Get monitor metrics
        metrics = monitor.get_metrics()

        # Print benchmark results
        print(f"\n{'=' * 60}")
        print("MIXED FORMAT THROUGHPUT BENCHMARK")
        print(f"{'=' * 60}")
        print(f"Total Files: {len(test_files)}")
        print(f"Total Size: {metrics['mb_processed']:.2f} MB")
        print(f"Duration: {elapsed_seconds:.2f}s")
        print(f"Throughput: {throughput_mbps:.2f} MB/s")
        print(f"Target: 5.0 MB/s")
        print(f"Status: {'✓ PASS' if throughput_mbps >= 5.0 else '✗ FAIL'}")
        print(f"{'=' * 60}\n")

        # Assert performance target
        assert throughput_mbps >= 5.0, (
            f"Throughput {throughput_mbps:.2f} MB/s is below target 5.0 MB/s"
        )

    @pytest.mark.asyncio
    async def test_sustained_throughput(self, benchmark_service, tmp_path):
        """Test sustained throughput over many documents."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Create and process 20 documents (~20MB total)
        total_bytes = 0
        num_docs = 20

        for i in range(num_docs):
            # Create 1MB file
            file_path = tmp_path / f"doc_{i}.md"
            content = f"# Document {i}\n\n" + ("Test content. " * 20000)
            file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size
            total_bytes += file_size

            # Process
            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"sustained_{i}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            monitor.record_document(
                size_bytes=file_size,
                format=normalized.metadata.format,
                duration_ms=doc_duration_ms,
            )

        metrics = monitor.get_metrics()
        throughput = metrics["throughput_mbps"]

        print(f"\nSustained Throughput: {throughput:.2f} MB/s ({num_docs} docs)")

        assert throughput >= 5.0, f"Sustained throughput {throughput:.2f} MB/s below target"


# ---------------------------------------------------------------------------
# Per-Format Throughput Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestPerFormatThroughput:
    """Tests validating per-format throughput targets."""

    @pytest.mark.asyncio
    async def test_markdown_throughput(self, benchmark_service, tmp_path):
        """Test Markdown processing throughput (target ≥6 MB/s).

        Markdown is lightweight and should have high throughput.
        """
        monitor = PerformanceMonitor()
        monitor.start()

        # Create 10 markdown files (~10MB total)
        total_bytes = 0
        for i in range(10):
            file_path = tmp_path / f"markdown_{i}.md"
            content = f"# Document {i}\n\n" + ("Markdown content. " * 20000)
            file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size
            total_bytes += file_size

            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"md_{i}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            monitor.record_document(
                size_bytes=file_size,
                format=DocumentFormat.MARKDOWN,
                duration_ms=doc_duration_ms,
            )

        metrics = monitor.get_metrics()
        md_stats = metrics["format_stats"]["markdown"]
        throughput = md_stats["avg_throughput_mbps"]

        print(f"\nMarkdown Throughput: {throughput:.2f} MB/s")

        assert throughput >= 6.0, f"Markdown throughput {throughput:.2f} MB/s below target 6.0 MB/s"

    @pytest.mark.asyncio
    async def test_text_throughput(self, benchmark_service, tmp_path):
        """Test plain text processing throughput (target ≥10 MB/s).

        Plain text should be fastest due to simplicity.
        """
        monitor = PerformanceMonitor()
        monitor.start()

        # Create 10 text files (~10MB total)
        for i in range(10):
            file_path = tmp_path / f"text_{i}.txt"
            content = "Plain text content. " * 20000
            file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size

            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"txt_{i}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            monitor.record_document(
                size_bytes=file_size,
                format=DocumentFormat.TEXT,
                duration_ms=doc_duration_ms,
            )

        metrics = monitor.get_metrics()
        txt_stats = metrics["format_stats"]["text"]
        throughput = txt_stats["avg_throughput_mbps"]

        print(f"\nText Throughput: {throughput:.2f} MB/s")

        assert throughput >= 10.0, f"Text throughput {throughput:.2f} MB/s below target 10.0 MB/s"


# ---------------------------------------------------------------------------
# Large File Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestLargeFileThroughput:
    """Tests for large file processing performance."""

    @pytest.mark.asyncio
    async def test_large_markdown_file(self, benchmark_service, tmp_path):
        """Test processing large markdown file (>10MB)."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Create 15MB markdown file
        file_path = tmp_path / "large.md"
        content = "# Large Document\n\n" + ("Content section. " * 300000)
        file_path.write_text(content, encoding="utf-8")

        file_size = file_path.stat().st_size
        print(f"\nLarge file size: {file_size / (1024 * 1024):.2f} MB")

        doc_start = time.time()
        normalized = await benchmark_service.normalize_document(
            file_path=file_path,
            source_id="large_md",
            source_type="benchmark",
        )
        doc_duration_ms = (time.time() - doc_start) * 1000

        monitor.record_document(
            size_bytes=file_size,
            format=DocumentFormat.MARKDOWN,
            duration_ms=doc_duration_ms,
        )

        metrics = monitor.get_metrics()
        throughput = metrics["throughput_mbps"]

        print(f"Large file throughput: {throughput:.2f} MB/s")
        print(f"Processing time: {doc_duration_ms / 1000:.2f}s")

        # Large files should still meet minimum target
        assert throughput >= 3.0, f"Large file throughput {throughput:.2f} MB/s too low"


# ---------------------------------------------------------------------------
# Efficiency Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestProcessingEfficiency:
    """Tests for processing efficiency metrics."""

    @pytest.mark.asyncio
    async def test_processing_time_per_mb(self, benchmark_service, tmp_path):
        """Test average processing time per MB is reasonable."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Process 5MB of content
        total_mb = 0.0
        for i in range(5):
            file_path = tmp_path / f"efficiency_{i}.md"
            content = f"# Document {i}\n\n" + ("Content. " * 20000)
            file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size
            total_mb += file_size / (1024 * 1024)

            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"eff_{i}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            monitor.record_document(
                size_bytes=file_size,
                format=normalized.metadata.format,
                duration_ms=doc_duration_ms,
            )

        metrics = monitor.get_metrics()
        avg_time_per_doc = metrics["avg_processing_time_ms"]
        avg_doc_size_mb = metrics["avg_document_size_mb"]

        time_per_mb = avg_time_per_doc / avg_doc_size_mb if avg_doc_size_mb > 0 else 0

        print(f"\nAvg processing time: {avg_time_per_doc:.2f}ms/doc")
        print(f"Avg document size: {avg_doc_size_mb:.2f}MB")
        print(f"Time per MB: {time_per_mb:.2f}ms/MB")

        # Should process 1MB in < 250ms (roughly 4MB/s)
        assert time_per_mb < 250, f"Processing too slow: {time_per_mb:.2f}ms/MB"

    @pytest.mark.asyncio
    async def test_documents_per_second(self, benchmark_service, tmp_path):
        """Test document processing rate (docs/sec)."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Process 30 small documents quickly
        for i in range(30):
            file_path = tmp_path / f"rate_{i}.txt"
            content = "Small document content. " * 100
            file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size

            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"rate_{i}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            monitor.record_document(
                size_bytes=file_size,
                format=DocumentFormat.TEXT,
                duration_ms=doc_duration_ms,
            )

        metrics = monitor.get_metrics()
        docs_per_sec = metrics["documents_per_second"]

        print(f"\nProcessing rate: {docs_per_sec:.2f} docs/sec")

        # Should process at least 10 docs/sec
        assert docs_per_sec >= 10.0, f"Processing rate {docs_per_sec:.2f} docs/sec too low"


# ---------------------------------------------------------------------------
# Integration Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestIntegrationPerformance:
    """Tests validating performance with full pipeline integration."""

    @pytest.mark.asyncio
    async def test_full_pipeline_throughput(self, benchmark_service, tmp_path):
        """Test throughput with all pipeline features enabled.

        Validates performance with:
        - Language detection
        - Content classification
        - Chunking
        - Hash computation
        - Metadata enrichment
        """
        monitor = PerformanceMonitor()
        monitor.start()

        # Create mixed content (10MB)
        total_bytes = 0
        for i in range(10):
            file_path = tmp_path / f"full_{i}.md"
            content = f"# Section {i}\n\n" + ("Full pipeline test content. " * 20000)
            file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size
            total_bytes += file_size

            doc_start = time.time()
            normalized = await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"full_{i}",
                source_type="benchmark",
            )
            doc_duration_ms = (time.time() - doc_start) * 1000

            # Verify all features executed
            assert normalized.metadata.language is not None
            assert normalized.metadata.content_type is not None
            assert normalized.metadata.content_hash is not None
            assert normalized.is_chunked

            monitor.record_document(
                size_bytes=file_size,
                format=normalized.metadata.format,
                duration_ms=doc_duration_ms,
            )

        metrics = monitor.get_metrics()
        throughput = metrics["throughput_mbps"]

        print(f"\nFull Pipeline Throughput: {throughput:.2f} MB/s")
        print(f"Features: Language detection, Classification, Chunking, Hashing")

        # Even with all features, should meet minimum target
        assert throughput >= 5.0, f"Full pipeline throughput {throughput:.2f} MB/s below target"

    @pytest.mark.asyncio
    async def test_service_performance_metrics(self, benchmark_service, tmp_path):
        """Test NormalizationService exports performance metrics correctly."""
        # Process some documents
        for i in range(5):
            file_path = tmp_path / f"service_{i}.md"
            content = f"# Document {i}\n\n" + ("Content. " * 10000)
            file_path.write_text(content, encoding="utf-8")

            await benchmark_service.normalize_document(
                file_path=file_path,
                source_id=f"svc_{i}",
                source_type="benchmark",
            )

        # Get service metrics
        metrics = benchmark_service.get_metrics()

        # Verify performance metrics included
        assert "performance" in metrics
        perf_metrics = metrics["performance"]

        assert "throughput_mbps" in perf_metrics
        assert "documents_processed" in perf_metrics
        assert "meets_throughput_target" in perf_metrics
        assert "format_stats" in perf_metrics

        print(f"\nService Throughput: {perf_metrics['throughput_mbps']:.2f} MB/s")
        print(f"Target Met: {perf_metrics['meets_throughput_target']}")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestPerformanceSummary:
    """Generate comprehensive performance report."""

    @pytest.mark.asyncio
    async def test_generate_performance_report(self, benchmark_service, tmp_path):
        """Generate comprehensive performance benchmark report."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Test various scenarios
        scenarios = {
            "Small Markdown": (10, "md", 1024 * 100),  # 100KB files
            "Medium Text": (5, "txt", 1024 * 1024),    # 1MB files
            "Large Markdown": (2, "md", 5 * 1024 * 1024),  # 5MB files
        }

        results = {}

        for scenario_name, (num_files, ext, file_size) in scenarios.items():
            scenario_start = time.time()
            scenario_bytes = 0

            for i in range(num_files):
                file_path = tmp_path / f"{scenario_name.replace(' ', '_')}_{i}.{ext}"
                # Generate content to reach target size
                content = "Test content. " * (file_size // 14)
                file_path.write_text(content, encoding="utf-8")

                actual_size = file_path.stat().st_size
                scenario_bytes += actual_size

                doc_start = time.time()
                normalized = await benchmark_service.normalize_document(
                    file_path=file_path,
                    source_id=f"{scenario_name}_{i}",
                    source_type="benchmark",
                )
                doc_duration_ms = (time.time() - doc_start) * 1000

                monitor.record_document(
                    size_bytes=actual_size,
                    format=normalized.metadata.format,
                    duration_ms=doc_duration_ms,
                )

            scenario_duration = time.time() - scenario_start
            scenario_throughput = (scenario_bytes / (1024 * 1024)) / scenario_duration

            results[scenario_name] = {
                "files": num_files,
                "total_mb": scenario_bytes / (1024 * 1024),
                "duration_s": scenario_duration,
                "throughput_mbps": scenario_throughput,
            }

        # Print comprehensive report
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        for scenario, data in results.items():
            print(f"\n{scenario}:")
            print(f"  Files: {data['files']}")
            print(f"  Total: {data['total_mb']:.2f} MB")
            print(f"  Duration: {data['duration_s']:.2f}s")
            print(f"  Throughput: {data['throughput_mbps']:.2f} MB/s")
            print(f"  Status: {'✓ PASS' if data['throughput_mbps'] >= 5.0 else '⚠ BELOW TARGET'}")

        # Overall metrics
        overall_metrics = monitor.get_metrics()
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Documents: {overall_metrics['documents_processed']}")
        print(f"  Total Data: {overall_metrics['mb_processed']:.2f} MB")
        print(f"  Average Throughput: {overall_metrics['throughput_mbps']:.2f} MB/s")
        print(f"  Meets Target (≥5 MB/s): {'✓ YES' if overall_metrics['meets_throughput_target'] else '✗ NO'}")

        print("\n" + "=" * 80 + "\n")

        # Assert overall target met
        assert overall_metrics["meets_throughput_target"], (
            f"Overall throughput {overall_metrics['throughput_mbps']:.2f} MB/s below 5.0 MB/s target"
        )

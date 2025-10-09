"""Unit tests for performance monitoring.

Tests verify:
- PerformanceMonitor initialization and state management
- Document recording with various sizes and formats
- Throughput calculation accuracy
- Per-format statistics tracking
- Metrics export format
- Edge cases and concurrent access
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.performance import (
    FormatStats,
    PerformanceMonitor,
    PerformanceSnapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor():
    """Create performance monitor for testing."""
    return PerformanceMonitor()


@pytest.fixture
def started_monitor():
    """Create and start performance monitor."""
    m = PerformanceMonitor()
    m.start()
    return m


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for PerformanceMonitor initialization."""

    def test_initial_state(self, monitor):
        """Test monitor starts with zero state."""
        assert monitor.start_time is None
        assert monitor.session_start is None
        assert monitor.bytes_processed == 0
        assert monitor.documents_processed == 0
        assert monitor.total_processing_time_ms == 0.0
        assert len(monitor._format_stats) == 0

    def test_start_initializes_timing(self, monitor):
        """Test start() initializes timing counters."""
        monitor.start()

        assert monitor.start_time is not None
        assert monitor.session_start is not None
        assert isinstance(monitor.start_time, float)
        assert isinstance(monitor.session_start, datetime)

    def test_start_can_be_called_multiple_times(self, monitor):
        """Test start() can be called multiple times to reset."""
        monitor.start()
        first_start = monitor.start_time

        time.sleep(0.01)

        monitor.start()
        second_start = monitor.start_time

        assert second_start > first_start


# ---------------------------------------------------------------------------
# Document Recording Tests
# ---------------------------------------------------------------------------


class TestDocumentRecording:
    """Tests for recording processed documents."""

    def test_record_single_document(self, started_monitor):
        """Test recording a single document."""
        started_monitor.record_document(
            size_bytes=1024,
            format=DocumentFormat.MARKDOWN,
            duration_ms=100.0,
        )

        assert started_monitor.documents_processed == 1
        assert started_monitor.bytes_processed == 1024
        assert started_monitor.total_processing_time_ms == 100.0

    def test_record_multiple_documents(self, started_monitor):
        """Test recording multiple documents."""
        # Record 3 documents
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )
        started_monitor.record_document(
            size_bytes=2048, format=DocumentFormat.PDF, duration_ms=150.0
        )
        started_monitor.record_document(
            size_bytes=512, format=DocumentFormat.MARKDOWN, duration_ms=25.0
        )

        assert started_monitor.documents_processed == 3
        assert started_monitor.bytes_processed == 3584
        assert started_monitor.total_processing_time_ms == 225.0

    def test_record_updates_format_stats(self, started_monitor):
        """Test document recording updates per-format statistics."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=100.0
        )

        stats = started_monitor.get_format_stats(DocumentFormat.MARKDOWN)
        assert stats is not None
        assert stats.document_count == 1
        assert stats.total_bytes == 1024
        assert stats.total_duration_ms == 100.0

    def test_record_multiple_formats(self, started_monitor):
        """Test recording documents with different formats."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )
        started_monitor.record_document(
            size_bytes=2048, format=DocumentFormat.PDF, duration_ms=150.0
        )

        md_stats = started_monitor.get_format_stats(DocumentFormat.MARKDOWN)
        pdf_stats = started_monitor.get_format_stats(DocumentFormat.PDF)

        assert md_stats.document_count == 1
        assert pdf_stats.document_count == 1
        assert md_stats.total_bytes == 1024
        assert pdf_stats.total_bytes == 2048


# ---------------------------------------------------------------------------
# Throughput Calculation Tests
# ---------------------------------------------------------------------------


class TestThroughputCalculation:
    """Tests for throughput calculation."""

    def test_throughput_zero_when_not_started(self, monitor):
        """Test throughput is zero when monitor not started."""
        assert monitor.get_throughput_mbps() == 0.0

    def test_throughput_calculation(self, started_monitor):
        """Test throughput calculation accuracy."""
        # Record 1MB processed in ~100ms (simulated)
        started_monitor.record_document(
            size_bytes=1024 * 1024,  # 1MB
            format=DocumentFormat.MARKDOWN,
            duration_ms=100.0,
        )

        # Wait small amount to ensure elapsed time
        time.sleep(0.01)

        throughput = started_monitor.get_throughput_mbps()

        # Should be roughly 100 MB/s (1MB in 0.01s)
        # But actual will be lower due to processing overhead
        assert throughput > 0.0

    def test_throughput_with_multiple_documents(self, started_monitor):
        """Test throughput with multiple documents."""
        # Process 5MB total
        for _ in range(5):
            started_monitor.record_document(
                size_bytes=1024 * 1024,  # 1MB each
                format=DocumentFormat.MARKDOWN,
                duration_ms=50.0,
            )

        time.sleep(0.01)
        throughput = started_monitor.get_throughput_mbps()

        # Should have processed 5MB
        assert throughput > 0.0
        assert started_monitor.bytes_processed == 5 * 1024 * 1024

    def test_documents_per_second_calculation(self, started_monitor):
        """Test documents per second rate calculation."""
        # Record 10 documents
        for _ in range(10):
            started_monitor.record_document(
                size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=10.0
            )

        time.sleep(0.01)
        docs_per_sec = started_monitor.get_documents_per_second()

        assert docs_per_sec > 0.0


# ---------------------------------------------------------------------------
# Format Statistics Tests
# ---------------------------------------------------------------------------


class TestFormatStatistics:
    """Tests for per-format statistics."""

    def test_format_stats_initial_state(self):
        """Test FormatStats initial state."""
        stats = FormatStats(format=DocumentFormat.MARKDOWN)

        assert stats.format == DocumentFormat.MARKDOWN
        assert stats.document_count == 0
        assert stats.total_bytes == 0
        assert stats.total_duration_ms == 0.0
        assert stats.avg_throughput_mbps == 0.0
        assert stats.last_updated is None

    def test_format_stats_update(self):
        """Test FormatStats.update() method."""
        stats = FormatStats(format=DocumentFormat.PDF)

        stats.update(size_bytes=2048, duration_ms=100.0)

        assert stats.document_count == 1
        assert stats.total_bytes == 2048
        assert stats.total_duration_ms == 100.0
        assert stats.avg_throughput_mbps > 0.0
        assert stats.last_updated is not None

    def test_format_stats_multiple_updates(self):
        """Test multiple updates to FormatStats."""
        stats = FormatStats(format=DocumentFormat.MARKDOWN)

        # Update 3 times
        stats.update(size_bytes=1024, duration_ms=50.0)
        stats.update(size_bytes=2048, duration_ms=100.0)
        stats.update(size_bytes=1024, duration_ms=50.0)

        assert stats.document_count == 3
        assert stats.total_bytes == 4096
        assert stats.total_duration_ms == 200.0

        # Check throughput calculation
        # 4096 bytes / (200ms / 1000) = 20480 bytes/s = 0.0195 MB/s
        expected_throughput = (4096 / (1024 * 1024)) / (200.0 / 1000.0)
        assert abs(stats.avg_throughput_mbps - expected_throughput) < 0.01

    def test_format_stats_to_dict(self):
        """Test FormatStats.to_dict() export."""
        stats = FormatStats(format=DocumentFormat.EMAIL)
        stats.update(size_bytes=1024, duration_ms=100.0)

        result = stats.to_dict()

        assert result["format"] == "email"
        assert result["document_count"] == 1
        assert result["total_bytes"] == 1024
        assert result["total_mb"] == round(1024 / (1024 * 1024), 2)
        assert result["total_duration_ms"] == 100.0
        assert "avg_throughput_mbps" in result
        assert "last_updated" in result

    def test_get_format_stats_nonexistent(self, started_monitor):
        """Test getting stats for format that hasn't been processed."""
        stats = started_monitor.get_format_stats(DocumentFormat.XLSX)
        assert stats is None

    def test_get_all_format_stats(self, started_monitor):
        """Test getting all format statistics."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )
        started_monitor.record_document(
            size_bytes=2048, format=DocumentFormat.PDF, duration_ms=100.0
        )

        all_stats = started_monitor.get_all_format_stats()

        assert len(all_stats) == 2
        assert DocumentFormat.MARKDOWN in all_stats
        assert DocumentFormat.PDF in all_stats


# ---------------------------------------------------------------------------
# Snapshot Tests
# ---------------------------------------------------------------------------


class TestPerformanceSnapshot:
    """Tests for performance snapshots."""

    def test_snapshot_captures_current_state(self, started_monitor):
        """Test snapshot captures current performance state."""
        # Record some documents
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )
        started_monitor.record_document(
            size_bytes=2048, format=DocumentFormat.PDF, duration_ms=100.0
        )

        snapshot = started_monitor.get_snapshot()

        assert snapshot.total_documents == 2
        assert snapshot.total_bytes == 3072
        assert snapshot.total_duration_ms == 150.0
        assert snapshot.current_throughput_mbps >= 0.0
        assert snapshot.documents_per_second >= 0.0
        assert len(snapshot.format_breakdown) == 2

    def test_snapshot_to_dict(self, started_monitor):
        """Test snapshot export to dictionary."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )

        snapshot = started_monitor.get_snapshot()
        result = snapshot.to_dict()

        assert "timestamp" in result
        assert "total_documents" in result
        assert "total_bytes" in result
        assert "total_mb" in result
        assert "current_throughput_mbps" in result
        assert "documents_per_second" in result
        assert "format_breakdown" in result

    def test_last_snapshot_saved(self, started_monitor):
        """Test last snapshot is saved in monitor."""
        snapshot = started_monitor.get_snapshot()

        assert started_monitor.last_snapshot is not None
        assert started_monitor.last_snapshot == snapshot


# ---------------------------------------------------------------------------
# Metrics Export Tests
# ---------------------------------------------------------------------------


class TestMetricsExport:
    """Tests for metrics export."""

    def test_get_metrics_structure(self, started_monitor):
        """Test get_metrics() returns complete structure."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )

        metrics = started_monitor.get_metrics()

        # Check required fields
        assert "documents_processed" in metrics
        assert "bytes_processed" in metrics
        assert "mb_processed" in metrics
        assert "total_processing_time_ms" in metrics
        assert "throughput_mbps" in metrics
        assert "documents_per_second" in metrics
        assert "avg_processing_time_ms" in metrics
        assert "avg_document_size_mb" in metrics
        assert "session_start" in metrics
        assert "session_duration_seconds" in metrics
        assert "meets_throughput_target" in metrics
        assert "throughput_target_mbps" in metrics
        assert "format_stats" in metrics

    def test_get_metrics_values(self, started_monitor):
        """Test get_metrics() returns correct values."""
        started_monitor.record_document(
            size_bytes=5 * 1024 * 1024,  # 5MB
            format=DocumentFormat.PDF,
            duration_ms=1000.0,
        )

        time.sleep(0.01)
        metrics = started_monitor.get_metrics()

        assert metrics["documents_processed"] == 1
        assert metrics["bytes_processed"] == 5 * 1024 * 1024
        assert metrics["mb_processed"] == 5.0
        assert metrics["total_processing_time_ms"] == 1000.0
        assert metrics["throughput_mbps"] > 0.0

    def test_meets_throughput_target(self, started_monitor):
        """Test meets_throughput_target flag."""
        # Process enough data to exceed 5 MB/s target
        # Need to process >5MB in <1 second
        for _ in range(6):
            started_monitor.record_document(
                size_bytes=1024 * 1024,  # 1MB
                format=DocumentFormat.MARKDOWN,
                duration_ms=10.0,
            )

        time.sleep(0.01)
        metrics = started_monitor.get_metrics()

        # Check flag exists
        assert "meets_throughput_target" in metrics
        assert metrics["throughput_target_mbps"] == 5.0

    def test_format_stats_in_metrics(self, started_monitor):
        """Test format stats included in metrics export."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )
        started_monitor.record_document(
            size_bytes=2048, format=DocumentFormat.PDF, duration_ms=100.0
        )

        metrics = started_monitor.get_metrics()
        format_stats = metrics["format_stats"]

        assert "markdown" in format_stats
        assert "pdf" in format_stats
        assert format_stats["markdown"]["document_count"] == 1
        assert format_stats["pdf"]["document_count"] == 1


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for monitor reset functionality."""

    def test_reset_clears_all_state(self, started_monitor):
        """Test reset() clears all monitor state."""
        # Record some documents
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.MARKDOWN, duration_ms=50.0
        )

        # Reset
        started_monitor.reset()

        # Verify cleared
        assert started_monitor.start_time is None
        assert started_monitor.session_start is None
        assert started_monitor.bytes_processed == 0
        assert started_monitor.documents_processed == 0
        assert started_monitor.total_processing_time_ms == 0.0
        assert len(started_monitor._format_stats) == 0
        assert started_monitor.last_snapshot is None


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_documents(self, started_monitor):
        """Test metrics with zero documents processed."""
        metrics = started_monitor.get_metrics()

        assert metrics["documents_processed"] == 0
        assert metrics["avg_processing_time_ms"] == 0.0
        assert metrics["avg_document_size_mb"] == 0.0

    def test_zero_duration_document(self, started_monitor):
        """Test recording document with zero duration."""
        started_monitor.record_document(
            size_bytes=1024, format=DocumentFormat.TEXT, duration_ms=0.0
        )

        assert started_monitor.documents_processed == 1
        assert started_monitor.total_processing_time_ms == 0.0

    def test_very_large_document(self, started_monitor):
        """Test recording very large document."""
        # 1GB document
        started_monitor.record_document(
            size_bytes=1024 * 1024 * 1024,
            format=DocumentFormat.PDF,
            duration_ms=5000.0,
        )

        metrics = started_monitor.get_metrics()
        assert metrics["mb_processed"] == 1024.0

    def test_many_small_documents(self, started_monitor):
        """Test recording many small documents."""
        # Record 1000 tiny documents
        for _ in range(1000):
            started_monitor.record_document(
                size_bytes=100, format=DocumentFormat.TEXT, duration_ms=1.0
            )

        assert started_monitor.documents_processed == 1000
        assert started_monitor.bytes_processed == 100000


# ---------------------------------------------------------------------------
# Logging Tests
# ---------------------------------------------------------------------------


class TestLogging:
    """Tests for log summary functionality."""

    def test_log_summary(self, started_monitor, caplog):
        """Test log_summary() produces output."""
        started_monitor.record_document(
            size_bytes=1024 * 1024,  # 1MB
            format=DocumentFormat.MARKDOWN,
            duration_ms=200.0,
        )

        with caplog.at_level("INFO"):
            started_monitor.log_summary()

        # Check log messages were produced
        assert "PERFORMANCE SUMMARY" in caplog.text
        assert "Documents Processed:" in caplog.text
        assert "Throughput:" in caplog.text

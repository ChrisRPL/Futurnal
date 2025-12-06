"""Tests for EmbeddingPerformanceProfiler.

Tests performance tracking, report generation, and bottleneck identification.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import threading
import time

from futurnal.embeddings.quality.profiler import (
    EmbeddingPerformanceProfiler,
    PerformanceMetric,
    LATENCY_TARGET_P95_MS,
    LATENCY_WARNING_P95_MS,
    THROUGHPUT_TARGET_PER_MIN,
    MEMORY_WARNING_MB,
)


@pytest.fixture
def profiler():
    """Create a fresh profiler instance."""
    return EmbeddingPerformanceProfiler()


@pytest.fixture
def profiler_with_data(profiler):
    """Create profiler with some recorded data."""
    # Record various metrics
    for i in range(100):
        profiler.profile_embedding_request(
            model_id=f"model_{i % 3}",
            entity_type=f"type_{i % 2}",
            content_length=100 + i * 10,
            latency_ms=100 + i * 5,  # 100-595ms
            memory_mb=500 + i * 2,  # 500-698MB
        )
    return profiler


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""

    def test_creation(self):
        """Test basic creation."""
        metric = PerformanceMetric(
            model_id="test-model",
            entity_type="Person",
            content_length=500,
            latency_ms=150.0,
            memory_mb=800.0,
        )

        assert metric.model_id == "test-model"
        assert metric.entity_type == "Person"
        assert metric.content_length == 500
        assert metric.latency_ms == 150.0
        assert metric.memory_mb == 800.0
        assert metric.timestamp is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metric = PerformanceMetric(
            model_id="test-model",
            entity_type="Person",
            content_length=500,
            latency_ms=150.0,
            memory_mb=800.0,
        )

        data = metric.to_dict()

        assert data["model_id"] == "test-model"
        assert data["entity_type"] == "Person"
        assert data["content_length"] == 500
        assert data["latency_ms"] == 150.0
        assert data["memory_mb"] == 800.0
        assert "timestamp" in data


class TestEmbeddingPerformanceProfiler:
    """Tests for EmbeddingPerformanceProfiler."""

    def test_profile_embedding_request(self, profiler):
        """Test recording performance metrics."""
        profiler.profile_embedding_request(
            model_id="instructor-large-entity",
            entity_type="Person",
            content_length=500,
            latency_ms=150.0,
            memory_mb=800.0,
        )

        assert profiler.get_metric_count() == 1

    def test_profile_with_auto_memory(self, profiler):
        """Test profiling with automatic memory detection."""
        with patch.object(profiler, "_get_current_memory_mb", return_value=1000.0):
            profiler.profile_embedding_request(
                model_id="test-model",
                entity_type="Person",
                content_length=500,
                latency_ms=150.0,
                memory_mb=None,  # Auto-detect
            )

        assert profiler.get_metric_count() == 1

    def test_generate_performance_report_empty(self, profiler):
        """Test report generation with no data."""
        report = profiler.generate_performance_report()

        assert report["total_requests"] == 0
        assert report["avg_latency_ms"] == 0.0
        assert report["throughput_per_minute"] == 0.0

    def test_generate_performance_report(self, profiler_with_data):
        """Test performance report generation."""
        report = profiler_with_data.generate_performance_report()

        assert report["total_requests"] == 100
        assert report["avg_latency_ms"] > 0
        assert report["p50_latency_ms"] > 0
        assert report["p95_latency_ms"] > 0
        assert report["p99_latency_ms"] > 0
        assert report["max_latency_ms"] > 0
        assert report["avg_memory_mb"] > 0
        assert report["max_memory_mb"] > 0
        assert report["throughput_per_minute"] > 0
        assert "by_model" in report
        assert "by_entity_type" in report

    def test_by_model_stats(self, profiler_with_data):
        """Test per-model statistics."""
        report = profiler_with_data.generate_performance_report()

        assert len(report["by_model"]) == 3  # model_0, model_1, model_2

        for model_id, stats in report["by_model"].items():
            assert "avg_latency_ms" in stats
            assert "p95_latency_ms" in stats
            assert "count" in stats

    def test_by_entity_type_stats(self, profiler_with_data):
        """Test per-entity-type statistics."""
        report = profiler_with_data.generate_performance_report()

        assert len(report["by_entity_type"]) == 2  # type_0, type_1

        for entity_type, stats in report["by_entity_type"].items():
            assert "avg_latency_ms" in stats
            assert "count" in stats

    def test_identify_performance_bottlenecks_no_issues(self, profiler):
        """Test bottleneck detection with good performance."""
        # Record fast operations
        for i in range(50):
            profiler.profile_embedding_request(
                model_id="fast-model",
                entity_type="Person",
                content_length=100,
                latency_ms=100,  # Well under 2s target
                memory_mb=500,  # Under memory warning
            )

        bottlenecks = profiler.identify_performance_bottlenecks()

        # Should report all within targets
        assert len(bottlenecks) == 1
        assert "within targets" in bottlenecks[0].lower()

    def test_identify_performance_bottlenecks_high_latency(self, profiler):
        """Test detection of high latency bottleneck."""
        # Record slow operations
        for i in range(100):
            profiler.profile_embedding_request(
                model_id="slow-model",
                entity_type="Person",
                content_length=1000,
                latency_ms=2500,  # Above 2s target
                memory_mb=500,
            )

        bottlenecks = profiler.identify_performance_bottlenecks()

        # Should detect latency issue
        latency_issues = [b for b in bottlenecks if "latency" in b.lower()]
        assert len(latency_issues) > 0

    def test_identify_performance_bottlenecks_high_memory(self, profiler):
        """Test detection of high memory bottleneck."""
        # Record high memory operations
        for i in range(50):
            profiler.profile_embedding_request(
                model_id="memory-heavy",
                entity_type="Person",
                content_length=1000,
                latency_ms=500,
                memory_mb=3000,  # Above 2GB warning
            )

        bottlenecks = profiler.identify_performance_bottlenecks()

        # Should detect memory issue
        memory_issues = [b for b in bottlenecks if "memory" in b.lower()]
        assert len(memory_issues) > 0

    def test_identify_performance_bottlenecks_low_throughput(self, profiler):
        """Test detection of low throughput."""
        # Record with artificially slow timestamps to simulate low throughput
        now = datetime.utcnow()

        # Manually add metrics with time gaps
        for i in range(10):
            metric = PerformanceMetric(
                model_id="test-model",
                entity_type="Person",
                content_length=100,
                latency_ms=100,
                memory_mb=500,
                timestamp=now + timedelta(minutes=i),  # 1 per minute
            )
            profiler._metrics.append(metric)

        bottlenecks = profiler.identify_performance_bottlenecks()

        # Should detect low throughput (10/10min = 1/min << 100/min)
        throughput_issues = [b for b in bottlenecks if "throughput" in b.lower()]
        assert len(throughput_issues) > 0

    def test_identify_performance_bottlenecks_model_specific(self, profiler):
        """Test model-specific bottleneck detection."""
        # Record with one slow model
        for i in range(50):
            profiler.profile_embedding_request(
                model_id="slow-model",
                entity_type="Person",
                content_length=100,
                latency_ms=1800,  # Near warning level
                memory_mb=500,
            )

        for i in range(50):
            profiler.profile_embedding_request(
                model_id="fast-model",
                entity_type="Person",
                content_length=100,
                latency_ms=200,
                memory_mb=500,
            )

        bottlenecks = profiler.identify_performance_bottlenecks()

        # Should mention slow-model specifically
        model_issues = [b for b in bottlenecks if "slow-model" in b]
        assert len(model_issues) > 0

    def test_get_recent_metrics(self, profiler_with_data):
        """Test getting recent metrics."""
        recent = profiler_with_data.get_recent_metrics(count=10)

        assert len(recent) == 10
        assert all(isinstance(m, dict) for m in recent)

    def test_reset(self, profiler_with_data):
        """Test resetting profiler."""
        assert profiler_with_data.get_metric_count() > 0

        profiler_with_data.reset()

        assert profiler_with_data.get_metric_count() == 0

    def test_history_pruning(self):
        """Test that history is pruned when max is exceeded."""
        profiler = EmbeddingPerformanceProfiler(max_metrics_history=100)

        # Add more than max
        for i in range(150):
            profiler.profile_embedding_request(
                model_id="test-model",
                entity_type="Person",
                content_length=100,
                latency_ms=100,
                memory_mb=500,
            )

        # Should be pruned to half of max
        assert profiler.get_metric_count() <= 100

    def test_thread_safety(self, profiler):
        """Test thread-safe concurrent profiling."""
        errors = []

        def record_metrics(thread_id):
            try:
                for i in range(50):
                    profiler.profile_embedding_request(
                        model_id=f"model_{thread_id}",
                        entity_type="Person",
                        content_length=100,
                        latency_ms=100 + thread_id,
                        memory_mb=500,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_metrics, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All metrics should be recorded (or pruned if over max)
        assert profiler.get_metric_count() > 0

    def test_percentile_calculations(self, profiler):
        """Test accurate percentile calculations."""
        # Record known latencies
        latencies = list(range(1, 101))  # 1 to 100ms
        for latency in latencies:
            profiler.profile_embedding_request(
                model_id="test-model",
                entity_type="Person",
                content_length=100,
                latency_ms=float(latency),
                memory_mb=500,
            )

        report = profiler.generate_performance_report()

        # P50 should be around 50
        assert 45 <= report["p50_latency_ms"] <= 55

        # P95 should be around 95
        assert 90 <= report["p95_latency_ms"] <= 100

        # P99 should be around 99
        assert 95 <= report["p99_latency_ms"] <= 100

    def test_memory_profiling_psutil_fallback(self, profiler):
        """Test memory profiling with psutil unavailable."""
        with patch.dict("sys.modules", {"psutil": None}):
            # Should not raise, returns 0.0
            memory = profiler._get_current_memory_mb()
            # Either returns actual memory or 0.0 on error
            assert isinstance(memory, float)


class TestPerformanceTargets:
    """Tests verifying performance target constants."""

    def test_latency_targets_defined(self):
        """Verify latency targets match production plan."""
        assert LATENCY_TARGET_P95_MS == 2000  # <2s
        assert LATENCY_WARNING_P95_MS == 1500

    def test_throughput_targets_defined(self):
        """Verify throughput targets match production plan."""
        assert THROUGHPUT_TARGET_PER_MIN == 100  # >100/min

    def test_memory_targets_defined(self):
        """Verify memory targets are reasonable."""
        assert MEMORY_WARNING_MB == 2000  # 2GB warning

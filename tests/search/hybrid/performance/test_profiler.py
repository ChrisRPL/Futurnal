"""Tests for PerformanceProfiler.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

import pytest

from futurnal.search.hybrid.performance.profiler import (
    LatencyBreakdown,
    PerformanceProfiler,
    PerformanceSnapshot,
    LATENCY_TARGET_P95_MS,
    LATENCY_TARGET_P50_MS,
    CACHE_HIT_RATE_TARGET,
)


class TestLatencyBreakdown:
    """Tests for LatencyBreakdown dataclass."""

    def test_creation(self):
        """Test basic creation."""
        from tests.search.hybrid.performance.conftest import create_latency_breakdown

        breakdown = create_latency_breakdown()

        assert breakdown.query_id == "test_query"
        assert breakdown.total_ms == 150.0
        assert "embed_query" in breakdown.components
        assert breakdown.strategy == "hybrid_parallel"


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""

    def test_record_query(self, profiler: PerformanceProfiler):
        """Test recording query metrics."""
        profiler.record_query(
            query_id="q1",
            total_ms=100,
            components={"embed": 30, "search": 70},
            strategy="vector",
            intent_type="factual",
        )

        assert profiler.get_record_count() == 1

    def test_avg_latency(self, profiler_with_data: PerformanceProfiler):
        """Test average latency calculation."""
        avg = profiler_with_data.get_avg_latency()
        assert avg is not None
        assert avg > 0

    def test_avg_latency_by_strategy(self, profiler_with_data: PerformanceProfiler):
        """Test average latency filtered by strategy."""
        avg_vector = profiler_with_data.get_avg_latency(strategy="vector")
        avg_hybrid = profiler_with_data.get_avg_latency(strategy="hybrid_parallel")

        assert avg_vector is not None
        assert avg_hybrid is not None
        # Both should have some data
        assert avg_vector > 0
        assert avg_hybrid > 0

    def test_percentile_latency(self, profiler_with_data: PerformanceProfiler):
        """Test percentile calculations."""
        p50 = profiler_with_data.get_percentile_latency(50)
        p95 = profiler_with_data.get_percentile_latency(95)
        p99 = profiler_with_data.get_percentile_latency(99)

        assert p50 is not None
        assert p95 is not None
        assert p99 is not None

        # Verify ordering: p50 < p95 < p99
        assert p50 <= p95 <= p99

    def test_component_breakdown(self, profiler_with_data: PerformanceProfiler):
        """Test component-level statistics."""
        breakdown = profiler_with_data.get_component_breakdown()

        assert "embed_query" in breakdown
        assert "vector_search" in breakdown
        assert "rank_results" in breakdown

        for component, stats in breakdown.items():
            assert "mean" in stats
            assert "p50" in stats
            assert "p95" in stats

    def test_identify_bottlenecks(self, profiler_with_data: PerformanceProfiler):
        """Test bottleneck identification."""
        bottlenecks = profiler_with_data.identify_bottlenecks()

        # Should identify some bottlenecks
        assert isinstance(bottlenecks, list)

        if bottlenecks:
            bottleneck = bottlenecks[0]
            assert "component" in bottleneck
            assert "recommendation" in bottleneck
            assert "ratio_of_total" in bottleneck

    def test_create_snapshot(self, profiler_with_data: PerformanceProfiler):
        """Test snapshot creation."""
        snapshot = profiler_with_data.create_snapshot()

        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.p50_latency >= 0
        assert snapshot.p95_latency >= 0
        assert snapshot.p99_latency >= 0

    def test_export_metrics(self, profiler_with_data: PerformanceProfiler):
        """Test metrics export."""
        metrics = profiler_with_data.export_metrics()

        assert "latency" in metrics
        assert "cache" in metrics
        assert "throughput" in metrics
        assert "targets" in metrics

        # Verify target tracking
        assert "p95_target_ms" in metrics["targets"]
        assert "p95_meets_target" in metrics["targets"]

    def test_reset(self, profiler_with_data: PerformanceProfiler):
        """Test profiler reset."""
        assert profiler_with_data.get_record_count() > 0

        profiler_with_data.reset()

        assert profiler_with_data.get_record_count() == 0

    def test_cache_hit_rate_calculation(self, profiler: PerformanceProfiler):
        """Test cache hit rate calculation."""
        # Record queries with known cache patterns
        for i in range(10):
            profiler.record_query(
                query_id=f"q{i}",
                total_ms=100,
                components={},
                strategy="vector",
                intent_type="factual",
                cache_hits=["embedding"] if i < 7 else [],  # 70% hit
                cache_misses=[] if i < 7 else ["embedding"],
            )

        hit_rate = profiler._calculate_cache_hit_rate()

        assert hit_rate == pytest.approx(0.7, rel=0.1)

    def test_empty_profiler_noop(self, profiler: PerformanceProfiler):
        """Test empty profiler handles operations gracefully."""
        assert profiler.get_avg_latency() is None
        assert profiler.get_percentile_latency(50) is None
        assert profiler.get_component_breakdown() == {}
        assert profiler.identify_bottlenecks() == []


class TestPerformanceTargets:
    """Tests verifying performance target constants."""

    def test_targets_defined(self):
        """Verify targets match production plan."""
        assert LATENCY_TARGET_P95_MS == 1000  # <1s
        assert LATENCY_TARGET_P50_MS == 200   # <200ms
        assert CACHE_HIT_RATE_TARGET == 0.60  # >60%

    def test_target_tracking(self, profiler_with_data: PerformanceProfiler):
        """Test target compliance tracking."""
        metrics = profiler_with_data.export_metrics()

        assert "p95_meets_target" in metrics["targets"]
        assert "p50_meets_target" in metrics["targets"]
        assert "cache_meets_target" in metrics["targets"]

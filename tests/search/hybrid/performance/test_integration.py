"""Integration tests for Performance & Caching module.

Tests end-to-end performance and validates success metrics.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from futurnal.search.hybrid.performance.cache import (
    CacheLayer,
    MultiLayerCache,
)
from futurnal.search.hybrid.performance.optimizer import (
    QueryPlanOptimizer,
    RetrievalStrategy,
)
from futurnal.search.hybrid.performance.profiler import (
    PerformanceProfiler,
    LATENCY_TARGET_P95_MS,
    CACHE_HIT_RATE_TARGET,
)


class TestCachePerformance:
    """Tests for cache performance characteristics."""

    def test_cache_set_get_latency(self, multi_layer_cache: MultiLayerCache):
        """Verify cache operations are fast."""
        # Measure set operation
        start = time.perf_counter()
        for i in range(100):
            multi_layer_cache.set(
                CacheLayer.QUERY_RESULT,
                f"query_{i}",
                {"results": [f"r{i}"]},
            )
        set_time = (time.perf_counter() - start) * 1000

        # Measure get operation
        start = time.perf_counter()
        for i in range(100):
            multi_layer_cache.get(CacheLayer.QUERY_RESULT, f"query_{i}")
        get_time = (time.perf_counter() - start) * 1000

        # Cache operations should be very fast (<1ms per operation)
        assert set_time / 100 < 1.0, f"Set latency {set_time/100:.2f}ms exceeds 1ms"
        assert get_time / 100 < 1.0, f"Get latency {get_time/100:.2f}ms exceeds 1ms"

    def test_semantic_cache_lookup_performance(self, multi_layer_cache: MultiLayerCache):
        """Verify semantic cache lookup is reasonably fast."""
        # Populate with embeddings
        for i in range(100):
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            multi_layer_cache.set(
                CacheLayer.QUERY_RESULT,
                f"query_{i}",
                f"result_{i}",
                query_embedding=embedding,
            )

        # Measure semantic lookup
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        start = time.perf_counter()
        multi_layer_cache.get(
            CacheLayer.QUERY_RESULT,
            "different_query",
            query_embedding=query_embedding,
        )
        lookup_time = (time.perf_counter() - start) * 1000

        # Semantic lookup should be <50ms even with 100 entries
        assert lookup_time < 50, f"Semantic lookup {lookup_time:.2f}ms exceeds 50ms"


class TestEndToEndCaching:
    """End-to-end tests for caching behavior."""

    @pytest.mark.performance
    def test_cache_hit_rate_target(self, multi_layer_cache: MultiLayerCache):
        """Validate >60% cache hit rate with realistic query pattern."""
        # Simulate realistic query pattern
        # 50 unique queries with Zipf-like distribution (popular queries repeated more)
        queries = []
        for i in range(200):
            # More popular queries (lower indices) appear more frequently
            idx = int(np.random.zipf(1.5)) % 50
            queries.append(f"query_{idx}")

        for query in queries:
            value, hit = multi_layer_cache.get(CacheLayer.QUERY_RESULT, query)
            if not hit:
                multi_layer_cache.set(
                    CacheLayer.QUERY_RESULT,
                    query,
                    f"result_for_{query}",
                )

        hit_rate = multi_layer_cache.stats.overall_hit_rate()

        assert hit_rate > CACHE_HIT_RATE_TARGET, (
            f"Cache hit rate {hit_rate:.2%} below {CACHE_HIT_RATE_TARGET:.0%} target"
        )


class TestIntegratedOptimization:
    """Tests for integrated cache + optimizer + profiler."""

    def test_profiler_with_optimizer(
        self,
        multi_layer_cache: MultiLayerCache,
        profiler: PerformanceProfiler,
    ):
        """Test profiler records optimizer-influenced queries."""
        optimizer = QueryPlanOptimizer(cache=multi_layer_cache, profiler=profiler)

        # Simulate queries with different intents
        intents = ["temporal", "causal", "exploratory", "factual"]

        for i, intent_type in enumerate(intents * 25):
            mock_intent = MagicMock()
            mock_intent.primary_intent = intent_type

            # Get plan
            plan = optimizer.optimize(f"test query {i}", mock_intent)

            # Simulate execution with profiling
            profiler.record_query(
                query_id=f"q{i}",
                total_ms=plan.estimated_cost.latency_ms + np.random.normal(0, 10),
                components={"plan": plan.estimated_cost.latency_ms},
                strategy=plan.strategy.value,
                intent_type=intent_type,
                cache_hits=[],
                cache_misses=["query_result"],
            )

        # Verify profiler has data
        assert profiler.get_record_count() == 100

        # Verify strategy distribution is tracked
        metrics = profiler.export_metrics()
        assert len(metrics["by_strategy"]) > 0

    @pytest.mark.performance
    def test_simulated_p95_latency(self, profiler: PerformanceProfiler):
        """Simulate P95 latency validation."""
        # Simulate 100 queries with realistic latency distribution
        np.random.seed(42)  # Reproducible

        for i in range(100):
            # Most queries fast, some slow (simulating real distribution)
            base_latency = 100  # Base 100ms
            if i < 90:
                # 90% fast queries
                latency = base_latency + np.random.exponential(50)
            else:
                # 10% slower queries (tail)
                latency = base_latency + np.random.exponential(200)

            profiler.record_query(
                query_id=f"q{i}",
                total_ms=latency,
                components={"search": latency * 0.7, "rank": latency * 0.3},
                strategy="hybrid_parallel",
                intent_type="exploratory",
            )

        p95 = profiler.get_percentile_latency(95)

        # With this distribution, P95 should be under 1s
        assert p95 is not None
        assert p95 < LATENCY_TARGET_P95_MS, (
            f"P95 latency {p95:.0f}ms exceeds {LATENCY_TARGET_P95_MS}ms target"
        )


class TestCacheInvalidation:
    """Tests for cache invalidation scenarios."""

    def test_entity_mutation_invalidates_cache(self, multi_layer_cache: MultiLayerCache):
        """Verify entity mutations invalidate related cache entries."""
        # Set up cache entries with related entities
        multi_layer_cache.set(
            CacheLayer.QUERY_RESULT,
            "query about entity_x",
            "result",
            related_entities=["entity_x"],
        )
        multi_layer_cache.set(
            CacheLayer.GRAPH_TRAVERSAL,
            "traversal for entity_x",
            {"path": ["a", "b"]},
            related_entities=["entity_x"],
        )
        multi_layer_cache.set(
            CacheLayer.QUERY_RESULT,
            "unrelated query",
            "result2",
            related_entities=["entity_y"],
        )

        # Trigger invalidation
        multi_layer_cache.invalidate_for_entities(["entity_x"])

        # entity_x entries should be invalidated
        value, hit = multi_layer_cache.get(CacheLayer.QUERY_RESULT, "query about entity_x")
        assert hit is False

        value, hit = multi_layer_cache.get(CacheLayer.GRAPH_TRAVERSAL, "traversal for entity_x")
        assert hit is False

        # Unrelated entry should remain
        value, hit = multi_layer_cache.get(CacheLayer.QUERY_RESULT, "unrelated query")
        assert hit is True

    def test_schema_change_invalidates_layers(self, multi_layer_cache: MultiLayerCache):
        """Verify schema changes invalidate appropriate layers."""
        # Populate caches
        multi_layer_cache.set(CacheLayer.QUERY_RESULT, "q1", "r1")
        multi_layer_cache.set(CacheLayer.EMBEDDING, "e1", [0.1, 0.2])
        multi_layer_cache.set(CacheLayer.LLM_INTENT, "i1", "factual")

        # Trigger schema change invalidation
        multi_layer_cache.invalidate_on_schema_change("v2")

        # QUERY_RESULT should be invalidated (invalidate_on_schema_change=True)
        value, hit = multi_layer_cache.get(CacheLayer.QUERY_RESULT, "q1")
        assert hit is False

        # EMBEDDING should be invalidated
        value, hit = multi_layer_cache.get(CacheLayer.EMBEDDING, "e1")
        assert hit is False

        # LLM_INTENT should remain (invalidate_on_schema_change=False)
        value, hit = multi_layer_cache.get(CacheLayer.LLM_INTENT, "i1")
        assert hit is True

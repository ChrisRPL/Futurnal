"""Tests for MultiLayerCache.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from futurnal.search.hybrid.performance.cache import (
    CacheConfig,
    CacheEntry,
    CacheLayer,
    CacheStatistics,
    DEFAULT_CACHE_CONFIGS,
    MultiLayerCache,
)


class TestCacheLayer:
    """Tests for CacheLayer enum."""

    def test_layer_values(self):
        """Verify all expected cache layers exist."""
        assert CacheLayer.QUERY_RESULT == "query_result"
        assert CacheLayer.EMBEDDING == "embedding"
        assert CacheLayer.LLM_INTENT == "llm_intent"
        assert CacheLayer.GRAPH_TRAVERSAL == "graph_traversal"
        assert CacheLayer.TEMPORAL_INDEX == "temporal_index"


class TestCacheStatistics:
    """Tests for CacheStatistics."""

    def test_initialization(self):
        """Test default initialization of all counters."""
        stats = CacheStatistics()
        for layer in CacheLayer:
            assert stats.hits[layer] == 0
            assert stats.misses[layer] == 0
            assert stats.sets[layer] == 0

    def test_record_hit(self):
        """Test recording cache hits."""
        stats = CacheStatistics()
        stats.record_hit(CacheLayer.QUERY_RESULT)
        assert stats.hits[CacheLayer.QUERY_RESULT] == 1

    def test_record_semantic_hit(self):
        """Test recording semantic cache hits."""
        stats = CacheStatistics()
        stats.record_hit(CacheLayer.QUERY_RESULT, semantic=True)
        assert stats.hits[CacheLayer.QUERY_RESULT] == 1
        assert stats.semantic_hits[CacheLayer.QUERY_RESULT] == 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStatistics()
        stats.record_hit(CacheLayer.QUERY_RESULT)
        stats.record_hit(CacheLayer.QUERY_RESULT)
        stats.record_miss(CacheLayer.QUERY_RESULT)

        assert stats.hit_rate(CacheLayer.QUERY_RESULT) == pytest.approx(2/3)

    def test_overall_hit_rate(self):
        """Test overall hit rate across layers."""
        stats = CacheStatistics()
        stats.record_hit(CacheLayer.QUERY_RESULT)
        stats.record_hit(CacheLayer.EMBEDDING)
        stats.record_miss(CacheLayer.QUERY_RESULT)
        stats.record_miss(CacheLayer.EMBEDDING)

        assert stats.overall_hit_rate() == pytest.approx(0.5)

    def test_semantic_hit_ratio(self):
        """Test semantic hit ratio calculation."""
        stats = CacheStatistics()
        stats.record_hit(CacheLayer.QUERY_RESULT, semantic=True)
        stats.record_hit(CacheLayer.QUERY_RESULT, semantic=False)
        stats.record_hit(CacheLayer.EMBEDDING, semantic=True)

        assert stats.semantic_hit_ratio() == pytest.approx(2/3)

    def test_to_dict(self):
        """Test export to dictionary."""
        stats = CacheStatistics()
        stats.record_hit(CacheLayer.QUERY_RESULT)
        data = stats.to_dict()

        assert "hits" in data
        assert "overall_hit_rate" in data
        assert "by_layer" in data


class TestMultiLayerCache:
    """Tests for MultiLayerCache."""

    def test_basic_set_get(self, multi_layer_cache: MultiLayerCache):
        """Test basic set and get operations."""
        multi_layer_cache.set(
            CacheLayer.QUERY_RESULT,
            "test query",
            {"results": ["a", "b", "c"]},
        )

        value, hit = multi_layer_cache.get(CacheLayer.QUERY_RESULT, "test query")

        assert hit is True
        assert value == {"results": ["a", "b", "c"]}

    def test_cache_miss(self, multi_layer_cache: MultiLayerCache):
        """Test cache miss returns None and records miss."""
        value, hit = multi_layer_cache.get(CacheLayer.QUERY_RESULT, "nonexistent")

        assert hit is False
        assert value is None
        assert multi_layer_cache.stats.misses[CacheLayer.QUERY_RESULT] == 1

    def test_exact_match_priority(self, multi_layer_cache: MultiLayerCache):
        """Test exact match is tried before semantic match."""
        embedding = np.random.randn(768).astype(np.float32)

        multi_layer_cache.set(
            CacheLayer.QUERY_RESULT,
            "exact match query",
            "exact_value",
            query_embedding=embedding,
        )

        value, hit = multi_layer_cache.get(
            CacheLayer.QUERY_RESULT,
            "exact match query",
            query_embedding=embedding,
        )

        assert hit is True
        assert value == "exact_value"
        # Should be exact hit, not semantic
        assert multi_layer_cache.stats.semantic_hits[CacheLayer.QUERY_RESULT] == 0

    def test_semantic_similarity_match(self, multi_layer_cache: MultiLayerCache):
        """Test semantic similarity matching."""
        base_embedding = np.ones(768, dtype=np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        multi_layer_cache.set(
            CacheLayer.QUERY_RESULT,
            "original query",
            "original_value",
            query_embedding=base_embedding,
        )

        # Create similar embedding (slight perturbation)
        similar_embedding = base_embedding + np.random.randn(768).astype(np.float32) * 0.01
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

        value, hit = multi_layer_cache.get(
            CacheLayer.QUERY_RESULT,
            "different query text",
            query_embedding=similar_embedding,
        )

        assert hit is True
        assert value == "original_value"
        assert multi_layer_cache.stats.semantic_hits[CacheLayer.QUERY_RESULT] == 1

    def test_entity_based_invalidation(self, cache_with_entries: MultiLayerCache):
        """Test invalidation based on entity IDs."""
        initial_size = cache_with_entries.get_size(CacheLayer.QUERY_RESULT)
        assert initial_size > 0

        # Invalidate entries related to entity_5
        cache_with_entries.invalidate_for_entities(["entity_5"])

        new_size = cache_with_entries.get_size(CacheLayer.QUERY_RESULT)
        assert new_size == initial_size - 1

    def test_layer_invalidation(self, cache_with_entries: MultiLayerCache):
        """Test invalidation of entire cache layer."""
        cache_with_entries.invalidate_layer(CacheLayer.QUERY_RESULT)

        assert cache_with_entries.get_size(CacheLayer.QUERY_RESULT) == 0

    def test_lru_eviction(self):
        """Test LRU eviction when max entries exceeded."""
        # Create cache with small max_entries
        config = DEFAULT_CACHE_CONFIGS.copy()
        config[CacheLayer.QUERY_RESULT] = CacheConfig(
            layer=CacheLayer.QUERY_RESULT,
            ttl_seconds=300,
            max_entries=10,  # Small limit
            invalidate_on_pkg_mutation=True,
            invalidate_on_schema_change=True,
            use_semantic_similarity=False,
            similarity_threshold=1.0,
        )

        cache = MultiLayerCache(configs=config)

        # Add more than max entries
        for i in range(15):
            cache.set(CacheLayer.QUERY_RESULT, f"query_{i}", f"value_{i}")

        # Should have evicted some entries
        assert cache.get_size(CacheLayer.QUERY_RESULT) <= 10

    def test_ttl_expiration(self):
        """Test entries expire after TTL."""
        # Create cache with very short TTL
        config = DEFAULT_CACHE_CONFIGS.copy()
        config[CacheLayer.QUERY_RESULT] = CacheConfig(
            layer=CacheLayer.QUERY_RESULT,
            ttl_seconds=1,  # 1 second TTL
            max_entries=100,
            invalidate_on_pkg_mutation=True,
            invalidate_on_schema_change=True,
            use_semantic_similarity=False,
            similarity_threshold=1.0,
        )

        cache = MultiLayerCache(configs=config)
        cache.set(CacheLayer.QUERY_RESULT, "test", "value")

        # Should hit immediately
        value, hit = cache.get(CacheLayer.QUERY_RESULT, "test")
        assert hit is True

        # Wait for expiration
        time.sleep(1.1)

        # Should miss after expiration
        value, hit = cache.get(CacheLayer.QUERY_RESULT, "test")
        assert hit is False

    def test_create_event_handler(self, multi_layer_cache: MultiLayerCache):
        """Test event handler creation for PKG integration."""
        existing_handler = MagicMock()

        combined = multi_layer_cache.create_event_handler(existing_handler)

        # Create mock event
        mock_event = MagicMock()
        mock_event.entity_id = "entity_1"
        mock_event.event_type = MagicMock()
        mock_event.event_type.value = "entity_updated"

        # Combined handler should call both
        combined(mock_event)
        existing_handler.assert_called_once_with(mock_event)

    def test_clear(self, cache_with_entries: MultiLayerCache):
        """Test clearing all cache layers."""
        assert cache_with_entries.get_size() > 0

        cache_with_entries.clear()

        assert cache_with_entries.get_size() == 0

    def test_multiple_layers(self, multi_layer_cache: MultiLayerCache):
        """Test operations across multiple cache layers."""
        multi_layer_cache.set(CacheLayer.QUERY_RESULT, "query1", "result1")
        multi_layer_cache.set(CacheLayer.EMBEDDING, "embed1", [0.1, 0.2, 0.3])
        multi_layer_cache.set(CacheLayer.LLM_INTENT, "intent1", "factual")

        assert multi_layer_cache.get_size(CacheLayer.QUERY_RESULT) == 1
        assert multi_layer_cache.get_size(CacheLayer.EMBEDDING) == 1
        assert multi_layer_cache.get_size(CacheLayer.LLM_INTENT) == 1
        assert multi_layer_cache.get_size() == 3


class TestCacheTargets:
    """Tests validating cache performance targets."""

    def test_hit_rate_target(self, cache_with_entries: MultiLayerCache):
        """Validate >60% cache hit rate target is achievable."""
        # Simulate realistic query pattern (some repeats)
        for i in range(100):
            query = f"test query {i % 10}"  # 10 unique queries repeated
            value, hit = cache_with_entries.get(CacheLayer.QUERY_RESULT, query)
            if not hit:
                cache_with_entries.set(CacheLayer.QUERY_RESULT, query, f"result_{i}")

        hit_rate = cache_with_entries.stats.overall_hit_rate()

        # With repeated queries, we should achieve >60%
        assert hit_rate > 0.6, f"Cache hit rate {hit_rate:.2%} below 60% target"

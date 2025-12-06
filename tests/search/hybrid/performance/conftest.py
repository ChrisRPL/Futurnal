"""Test fixtures for Performance & Caching module.

Provides mocks and fixtures for testing:
- MultiLayerCache
- OllamaConnectionPool
- ModelWarmUpManager
- QueryPlanOptimizer
- PerformanceProfiler

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

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
from futurnal.search.hybrid.performance.ollama_pool import (
    HealthStatus,
    OllamaConnectionConfig,
    OllamaConnectionPool,
)
from futurnal.search.hybrid.performance.optimizer import (
    QueryPlan,
    QueryPlanCost,
    QueryPlanOptimizer,
    RetrievalStrategy,
)
from futurnal.search.hybrid.performance.profiler import (
    LatencyBreakdown,
    PerformanceProfiler,
    PerformanceSnapshot,
)
from futurnal.search.hybrid.performance.warmup import (
    ModelWarmUpManager,
    WarmUpConfig,
)


# ---------------------------------------------------------------------------
# Cache Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_config() -> Dict[CacheLayer, CacheConfig]:
    """Default cache configuration for testing."""
    return DEFAULT_CACHE_CONFIGS.copy()


@pytest.fixture
def multi_layer_cache() -> MultiLayerCache:
    """Fresh MultiLayerCache instance for testing."""
    return MultiLayerCache()


@pytest.fixture
def cache_with_entries(multi_layer_cache: MultiLayerCache) -> MultiLayerCache:
    """MultiLayerCache with some pre-populated entries."""
    for i in range(10):
        multi_layer_cache.set(
            CacheLayer.QUERY_RESULT,
            f"test query {i}",
            {"results": [f"result_{i}"]},
            related_entities=[f"entity_{i}"],
        )
    return multi_layer_cache


# ---------------------------------------------------------------------------
# Ollama Pool Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_config() -> OllamaConnectionConfig:
    """Default Ollama connection config for testing."""
    return OllamaConnectionConfig(
        base_url="http://localhost:11434",
        pool_size=2,
        request_timeout=5.0,
        max_retries=2,
    )


@pytest.fixture
def mock_ollama_session() -> MagicMock:
    """Mock aiohttp session for Ollama pool testing."""
    session = MagicMock()
    session.post = AsyncMock()
    session.get = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def health_status() -> HealthStatus:
    """Default health status for testing."""
    return HealthStatus()


# ---------------------------------------------------------------------------
# Warm-up Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def warmup_config() -> WarmUpConfig:
    """Default warm-up configuration for testing."""
    return WarmUpConfig(
        models=["test-model"],
        warmup_prompts={"test-model": "test prompt"},
        parallel_warmup=True,
        timeout_seconds=5.0,
    )


@pytest.fixture
def mock_ollama_pool() -> MagicMock:
    """Mock OllamaConnectionPool for testing."""
    pool = MagicMock(spec=OllamaConnectionPool)
    pool.generate = AsyncMock(return_value="response")
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    pool.is_healthy = True
    return pool


# ---------------------------------------------------------------------------
# Optimizer Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizer() -> QueryPlanOptimizer:
    """Default QueryPlanOptimizer for testing."""
    return QueryPlanOptimizer()


@pytest.fixture
def optimizer_with_cache(multi_layer_cache: MultiLayerCache) -> QueryPlanOptimizer:
    """QueryPlanOptimizer with cache integration."""
    return QueryPlanOptimizer(cache=multi_layer_cache)


@pytest.fixture
def mock_query_intent() -> MagicMock:
    """Mock QueryIntent for testing."""
    intent = MagicMock()
    intent.primary_intent = "temporal"
    intent.confidence = 0.9
    return intent


# ---------------------------------------------------------------------------
# Profiler Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profiler() -> PerformanceProfiler:
    """Fresh PerformanceProfiler instance for testing."""
    return PerformanceProfiler()


@pytest.fixture
def profiler_with_data(profiler: PerformanceProfiler) -> PerformanceProfiler:
    """PerformanceProfiler with sample data."""
    for i in range(100):
        profiler.record_query(
            query_id=f"query_{i}",
            total_ms=100 + i * 5,
            components={
                "embed_query": 20 + i % 10,
                "vector_search": 50 + i % 20,
                "rank_results": 30 + i % 15,
            },
            strategy="hybrid_parallel" if i % 2 else "vector",
            intent_type="temporal" if i % 3 else "factual",
            cache_hits=["embedding"] if i % 2 else [],
            cache_misses=["query_result"] if i % 3 else [],
        )
    return profiler


# ---------------------------------------------------------------------------
# Test Data Factories
# ---------------------------------------------------------------------------


def create_embedding(dim: int = 768) -> np.ndarray:
    """Create a random embedding vector for testing."""
    return np.random.randn(dim).astype(np.float32)


def create_cache_entry(
    key: str = "test_key",
    value: Any = {"result": "test"},
    layer: CacheLayer = CacheLayer.QUERY_RESULT,
    ttl_seconds: int = 300,
) -> CacheEntry:
    """Create a CacheEntry for testing."""
    now = datetime.utcnow()
    from datetime import timedelta

    return CacheEntry(
        key=key,
        value=value,
        embedding=None,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
        hit_count=0,
        layer=layer,
        query_hash=key,
        related_entities=[],
    )


def create_latency_breakdown(
    query_id: str = "test_query",
    total_ms: float = 150.0,
) -> LatencyBreakdown:
    """Create a LatencyBreakdown for testing."""
    return LatencyBreakdown(
        query_id=query_id,
        total_ms=total_ms,
        components={"embed_query": 30, "vector_search": 80, "rank_results": 40},
        timestamp=datetime.utcnow(),
        strategy="hybrid_parallel",
        intent_type="temporal",
        cache_hits=["embedding"],
        cache_misses=["query_result"],
    )

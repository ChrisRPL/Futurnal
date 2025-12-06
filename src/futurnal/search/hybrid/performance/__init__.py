"""Performance & Caching Module for Hybrid Search API.

This module provides comprehensive caching strategies and performance optimizations
to achieve sub-1s latency for 95% of queries.

Components:
- MultiLayerCache: Multi-layer caching with semantic similarity matching
- OllamaConnectionPool: Async connection pool for LLM inference
- ModelWarmUpManager: Cold start mitigation through model preloading
- QueryPlanOptimizer: Cost-based query plan optimization
- PerformanceProfiler: Latency tracking and bottleneck detection

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md

Option B Compliance:
- Ghost model frozen, Ollama used for inference only
- Local-first processing, all caching runs on-device
- Performance gates: P95 <1s validated before production
"""

from futurnal.search.hybrid.performance.cache import (
    CacheConfig,
    CacheEntry,
    CacheLayer,
    CacheStatistics,
    MultiLayerCache,
    DEFAULT_CACHE_CONFIGS,
)
from futurnal.search.hybrid.performance.ollama_pool import (
    HealthStatus,
    OllamaConnectionConfig,
    OllamaConnectionError,
    OllamaConnectionPool,
)
from futurnal.search.hybrid.performance.warmup import (
    ModelWarmUpManager,
    WarmUpConfig,
    DEFAULT_WARMUP_PROMPTS,
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

__all__ = [
    # Cache
    "CacheConfig",
    "CacheEntry",
    "CacheLayer",
    "CacheStatistics",
    "MultiLayerCache",
    "DEFAULT_CACHE_CONFIGS",
    # Ollama Pool
    "HealthStatus",
    "OllamaConnectionConfig",
    "OllamaConnectionError",
    "OllamaConnectionPool",
    # Warmup
    "ModelWarmUpManager",
    "WarmUpConfig",
    "DEFAULT_WARMUP_PROMPTS",
    # Optimizer
    "QueryPlan",
    "QueryPlanCost",
    "QueryPlanOptimizer",
    "RetrievalStrategy",
    # Profiler
    "LatencyBreakdown",
    "PerformanceProfiler",
    "PerformanceSnapshot",
]

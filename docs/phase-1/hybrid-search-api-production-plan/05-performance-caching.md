Summary: Implement multi-layer caching, LLM inference optimization, query plan optimization, and performance profiling for sub-1s latency with Ollama backend integration.

# 05 · Performance & Caching

## Purpose
Implement comprehensive caching strategies and performance optimizations to achieve sub-1s latency for 95% of queries while maintaining consistency with PKG/schema updates and leveraging Ollama backend for fast LLM inference.

**Criticality**: HIGH - Direct impact on user experience and Option B quality targets

## Scope
- Multi-layer caching architecture (query results, embeddings, LLM intent, graph traversal)
- LLM inference optimization with Ollama connection pooling
- Query plan cost-based optimization
- Performance profiling and bottleneck identification
- Cache invalidation on PKG/schema mutations
- Warm-up strategies for cold start mitigation

## Requirements Alignment
- **Option B Requirement**: "Sub-1s latency for typical queries"
- **Performance Target**: <1s for 95% of queries, <100ms for cached queries
- **Cache Consistency**: Invalidate on relevant PKG/schema updates
- **Quality Gate**: P95 latency validated before production deployment

## Research Foundation

### Caching in RAG Systems
- **Semantic Cache**: Cache by query embedding similarity, not exact match
- **Hierarchical Caching**: Multiple layers with different TTLs and invalidation strategies
- **Predictive Prefetching**: Anticipate follow-up queries based on patterns

### LLM Inference Optimization
- **Connection Pooling**: Reuse HTTP connections to Ollama server
- **Request Batching**: Combine multiple intent classifications
- **Model Warmup**: Pre-load models to avoid cold start latency

---

## Component Design

### 1. Multi-Layer Cache Architecture

```python
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import numpy as np


class CacheLayer(str, Enum):
    """Cache layers with different characteristics."""
    QUERY_RESULT = "query_result"      # Final search results (TTL: 5 min)
    EMBEDDING = "embedding"             # Query/entity embeddings (TTL: 1 hour)
    LLM_INTENT = "llm_intent"          # Intent classification results (TTL: 30 min)
    GRAPH_TRAVERSAL = "graph_traversal" # PKG traversal results (invalidate on mutation)
    TEMPORAL_INDEX = "temporal_index"   # Temporal query indexes (TTL: 15 min)


@dataclass
class CacheConfig:
    """Configuration for each cache layer."""
    layer: CacheLayer
    ttl_seconds: int
    max_entries: int
    invalidate_on_pkg_mutation: bool
    invalidate_on_schema_change: bool
    use_semantic_similarity: bool  # For embedding-based cache lookup
    similarity_threshold: float    # Min similarity for cache hit (0.95 default)


DEFAULT_CACHE_CONFIGS = {
    CacheLayer.QUERY_RESULT: CacheConfig(
        layer=CacheLayer.QUERY_RESULT,
        ttl_seconds=300,           # 5 minutes
        max_entries=10000,
        invalidate_on_pkg_mutation=True,
        invalidate_on_schema_change=True,
        use_semantic_similarity=True,
        similarity_threshold=0.95
    ),
    CacheLayer.EMBEDDING: CacheConfig(
        layer=CacheLayer.EMBEDDING,
        ttl_seconds=3600,          # 1 hour
        max_entries=50000,
        invalidate_on_pkg_mutation=False,
        invalidate_on_schema_change=True,
        use_semantic_similarity=False,
        similarity_threshold=1.0
    ),
    CacheLayer.LLM_INTENT: CacheConfig(
        layer=CacheLayer.LLM_INTENT,
        ttl_seconds=1800,          # 30 minutes
        max_entries=5000,
        invalidate_on_pkg_mutation=False,
        invalidate_on_schema_change=False,
        use_semantic_similarity=True,
        similarity_threshold=0.98  # Higher threshold for intent
    ),
    CacheLayer.GRAPH_TRAVERSAL: CacheConfig(
        layer=CacheLayer.GRAPH_TRAVERSAL,
        ttl_seconds=900,           # 15 minutes (but invalidated on mutation)
        max_entries=20000,
        invalidate_on_pkg_mutation=True,
        invalidate_on_schema_change=True,
        use_semantic_similarity=False,
        similarity_threshold=1.0
    ),
    CacheLayer.TEMPORAL_INDEX: CacheConfig(
        layer=CacheLayer.TEMPORAL_INDEX,
        ttl_seconds=900,           # 15 minutes
        max_entries=10000,
        invalidate_on_pkg_mutation=True,
        invalidate_on_schema_change=False,
        use_semantic_similarity=False,
        similarity_threshold=1.0
    )
}


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    embedding: Optional[np.ndarray]  # For semantic cache lookup
    created_at: datetime
    expires_at: datetime
    hit_count: int
    layer: CacheLayer
    query_hash: str                   # For exact match fallback
    related_entities: List[str]       # For invalidation targeting


class MultiLayerCache:
    """
    Hierarchical cache with semantic similarity matching.

    Implements multi-layer caching strategy:
    1. L1: Query results (fast, short TTL)
    2. L2: Embeddings (medium, longer TTL)
    3. L3: LLM intents (slow to compute, medium TTL)
    4. L4: Graph traversals (invalidate on mutation)

    Integration Points:
    - PKGEventBus: Subscribe to mutations for invalidation
    - SchemaVersionManager: Subscribe to schema changes
    - QueryEmbeddingRouter: Get embeddings for semantic matching
    """

    def __init__(
        self,
        configs: Dict[CacheLayer, CacheConfig] = None,
        embedding_router: "QueryEmbeddingRouter" = None,
        pkg_event_bus: "PKGEventBus" = None
    ):
        self.configs = configs or DEFAULT_CACHE_CONFIGS
        self.embedding_router = embedding_router
        self.caches: Dict[CacheLayer, Dict[str, CacheEntry]] = {
            layer: {} for layer in CacheLayer
        }
        self.stats = CacheStatistics()

        # Subscribe to PKG mutations for invalidation
        if pkg_event_bus:
            pkg_event_bus.subscribe("entity_created", self._on_pkg_mutation)
            pkg_event_bus.subscribe("entity_updated", self._on_pkg_mutation)
            pkg_event_bus.subscribe("entity_deleted", self._on_pkg_mutation)
            pkg_event_bus.subscribe("relationship_created", self._on_pkg_mutation)
            pkg_event_bus.subscribe("schema_evolved", self._on_schema_change)

    def get(
        self,
        layer: CacheLayer,
        query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Any], bool]:
        """
        Get value from cache with semantic similarity matching.

        Returns:
            Tuple of (value, is_hit). Value is None if miss.
        """
        config = self.configs[layer]
        cache = self.caches[layer]

        # Generate cache key
        query_hash = self._hash_query(query)

        # Try exact match first (fast path)
        if query_hash in cache:
            entry = cache[query_hash]
            if not self._is_expired(entry):
                entry.hit_count += 1
                self.stats.record_hit(layer)
                return entry.value, True
            else:
                # Expired, remove
                del cache[query_hash]

        # Try semantic similarity match if enabled
        if config.use_semantic_similarity and query_embedding is not None:
            best_match = self._find_semantic_match(
                layer, query_embedding, config.similarity_threshold
            )
            if best_match:
                best_match.hit_count += 1
                self.stats.record_hit(layer, semantic=True)
                return best_match.value, True

        self.stats.record_miss(layer)
        return None, False

    def set(
        self,
        layer: CacheLayer,
        query: str,
        value: Any,
        query_embedding: Optional[np.ndarray] = None,
        related_entities: List[str] = None
    ):
        """Store value in cache with optional embedding for semantic matching."""
        config = self.configs[layer]
        cache = self.caches[layer]

        # Enforce max entries (LRU eviction)
        if len(cache) >= config.max_entries:
            self._evict_lru(layer)

        query_hash = self._hash_query(query)
        now = datetime.utcnow()

        entry = CacheEntry(
            key=query_hash,
            value=value,
            embedding=query_embedding,
            created_at=now,
            expires_at=now + timedelta(seconds=config.ttl_seconds),
            hit_count=0,
            layer=layer,
            query_hash=query_hash,
            related_entities=related_entities or []
        )

        cache[query_hash] = entry
        self.stats.record_set(layer)

    def invalidate_for_entities(self, entity_ids: List[str]):
        """
        Invalidate cache entries related to specific entities.

        Called when PKG mutations affect specific entities.
        """
        for layer, config in self.configs.items():
            if config.invalidate_on_pkg_mutation:
                cache = self.caches[layer]
                keys_to_remove = []

                for key, entry in cache.items():
                    if any(eid in entry.related_entities for eid in entity_ids):
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del cache[key]
                    self.stats.record_invalidation(layer, reason="entity_mutation")

    def invalidate_layer(self, layer: CacheLayer, reason: str = "manual"):
        """Invalidate entire cache layer."""
        count = len(self.caches[layer])
        self.caches[layer] = {}
        self.stats.record_invalidation(layer, reason=reason, count=count)

    def invalidate_on_schema_change(self, schema_version: str):
        """Invalidate caches affected by schema evolution."""
        for layer, config in self.configs.items():
            if config.invalidate_on_schema_change:
                self.invalidate_layer(layer, reason=f"schema_change_{schema_version}")

    def _find_semantic_match(
        self,
        layer: CacheLayer,
        query_embedding: np.ndarray,
        threshold: float
    ) -> Optional[CacheEntry]:
        """Find best semantic match above threshold."""
        cache = self.caches[layer]
        best_entry = None
        best_similarity = threshold

        for entry in cache.values():
            if entry.embedding is not None and not self._is_expired(entry):
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

        return best_entry

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _hash_query(self, query: str) -> str:
        """Generate deterministic hash for query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() > entry.expires_at

    def _evict_lru(self, layer: CacheLayer):
        """Evict least recently used entries."""
        cache = self.caches[layer]
        if not cache:
            return

        # Sort by hit_count (LFU) and created_at (LRU tiebreaker)
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: (x[1].hit_count, x[1].created_at)
        )

        # Remove bottom 10%
        to_remove = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:to_remove]:
            del cache[key]
            self.stats.record_eviction(layer)

    def _on_pkg_mutation(self, event: Dict[str, Any]):
        """Handle PKG mutation events."""
        entity_ids = event.get("entity_ids", [])
        if entity_ids:
            self.invalidate_for_entities(entity_ids)

    def _on_schema_change(self, event: Dict[str, Any]):
        """Handle schema evolution events."""
        schema_version = event.get("version", "unknown")
        self.invalidate_on_schema_change(schema_version)


@dataclass
class CacheStatistics:
    """Cache performance statistics for monitoring."""
    hits: Dict[CacheLayer, int] = None
    misses: Dict[CacheLayer, int] = None
    semantic_hits: Dict[CacheLayer, int] = None
    sets: Dict[CacheLayer, int] = None
    evictions: Dict[CacheLayer, int] = None
    invalidations: Dict[CacheLayer, int] = None

    def __post_init__(self):
        self.hits = {layer: 0 for layer in CacheLayer}
        self.misses = {layer: 0 for layer in CacheLayer}
        self.semantic_hits = {layer: 0 for layer in CacheLayer}
        self.sets = {layer: 0 for layer in CacheLayer}
        self.evictions = {layer: 0 for layer in CacheLayer}
        self.invalidations = {layer: 0 for layer in CacheLayer}

    def record_hit(self, layer: CacheLayer, semantic: bool = False):
        self.hits[layer] += 1
        if semantic:
            self.semantic_hits[layer] += 1

    def record_miss(self, layer: CacheLayer):
        self.misses[layer] += 1

    def record_set(self, layer: CacheLayer):
        self.sets[layer] += 1

    def record_eviction(self, layer: CacheLayer):
        self.evictions[layer] += 1

    def record_invalidation(self, layer: CacheLayer, reason: str, count: int = 1):
        self.invalidations[layer] += count

    def hit_rate(self, layer: CacheLayer) -> float:
        total = self.hits[layer] + self.misses[layer]
        return self.hits[layer] / total if total > 0 else 0.0

    def overall_hit_rate(self) -> float:
        total_hits = sum(self.hits.values())
        total_misses = sum(self.misses.values())
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0
```

---

### 2. Ollama LLM Inference Optimization

```python
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import aiohttp
import time


@dataclass
class OllamaConnectionConfig:
    """Configuration for Ollama connection pool."""
    base_url: str = "http://localhost:11434"
    pool_size: int = 4                  # Concurrent connections
    request_timeout: float = 30.0       # Seconds
    keepalive_timeout: float = 300.0    # Connection reuse window
    max_retries: int = 3
    retry_delay: float = 0.5
    batch_size: int = 8                 # Max requests per batch
    batch_timeout: float = 0.05         # Max wait for batch fill (50ms)


class OllamaConnectionPool:
    """
    Connection pool for Ollama LLM server.

    Optimizations:
    - Connection reuse with keepalive
    - Request batching for intent classification
    - Automatic retry with exponential backoff
    - Health monitoring with circuit breaker

    Integration Points:
    - OllamaIntentClassifier: Primary consumer
    - QueryRouter: Intent classification requests
    """

    def __init__(self, config: OllamaConnectionConfig = None):
        self.config = config or OllamaConnectionConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self.config.pool_size)
        self._batch_queue: List[Dict[str, Any]] = []
        self._batch_event = asyncio.Event()
        self._health_status = HealthStatus()

    async def initialize(self):
        """Initialize connection pool and warm up."""
        connector = aiohttp.TCPConnector(
            limit=self.config.pool_size,
            keepalive_timeout=self.config.keepalive_timeout,
            enable_cleanup_closed=True
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )

        # Warm up connection
        await self._health_check()

    async def close(self):
        """Clean shutdown of connection pool."""
        if self._session:
            await self._session.close()
            self._session = None

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self._semaphore:
            yield self._session

    async def generate(
        self,
        model: str,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Single generation request with retry logic.

        Args:
            model: Ollama model name (e.g., "llama3.1:8b")
            prompt: Input prompt
            **kwargs: Additional Ollama parameters

        Returns:
            Generated text response
        """
        for attempt in range(self.config.max_retries):
            try:
                async with self.acquire() as session:
                    async with session.post(
                        f"{self.config.base_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            **kwargs
                        }
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        self._health_status.record_success()
                        return data.get("response", "")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self._health_status.record_failure()
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay * (2 ** attempt)
                    )
                else:
                    raise OllamaConnectionError(f"Failed after {attempt + 1} attempts: {e}")

    async def batch_generate(
        self,
        model: str,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Batch generation for multiple prompts.

        Optimizes throughput by processing multiple prompts
        with concurrent connections.
        """
        tasks = [
            self.generate(model, prompt, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _health_check(self) -> bool:
        """Check Ollama server health."""
        try:
            async with self._session.get(
                f"{self.config.base_url}/api/tags"
            ) as response:
                response.raise_for_status()
                self._health_status.mark_healthy()
                return True
        except Exception:
            self._health_status.mark_unhealthy()
            return False


@dataclass
class HealthStatus:
    """Circuit breaker health tracking."""
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    is_healthy: bool = True
    last_check: float = 0.0
    failure_threshold: int = 5
    recovery_threshold: int = 3

    def record_success(self):
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        if self.consecutive_successes >= self.recovery_threshold:
            self.is_healthy = True

    def record_failure(self):
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        if self.consecutive_failures >= self.failure_threshold:
            self.is_healthy = False

    def mark_healthy(self):
        self.is_healthy = True
        self.consecutive_failures = 0

    def mark_unhealthy(self):
        self.is_healthy = False


class OllamaConnectionError(Exception):
    """Raised when Ollama connection fails."""
    pass


class LLMInferenceCache:
    """
    Specialized cache for LLM inference results.

    Caches:
    - Intent classifications (high reuse)
    - Query rewrites (medium reuse)
    - Strategy selections (low reuse but expensive)
    """

    def __init__(
        self,
        multi_layer_cache: MultiLayerCache,
        embedding_router: "QueryEmbeddingRouter"
    ):
        self.cache = multi_layer_cache
        self.embedding_router = embedding_router

    async def get_or_compute_intent(
        self,
        query: str,
        classifier: "OllamaIntentClassifier"
    ) -> "QueryIntent":
        """
        Get intent from cache or compute via LLM.

        Uses semantic similarity for cache lookup since
        similar queries should have similar intents.
        """
        # Get query embedding for semantic matching
        query_embedding = await self.embedding_router.embed_query(query)

        # Try cache
        cached, hit = self.cache.get(
            CacheLayer.LLM_INTENT,
            query,
            query_embedding
        )

        if hit:
            return cached

        # Compute via LLM
        intent = await classifier.classify(query)

        # Cache result
        self.cache.set(
            CacheLayer.LLM_INTENT,
            query,
            intent,
            query_embedding
        )

        return intent
```

---

### 3. Model Warm-Up System

```python
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WarmUpConfig:
    """Configuration for model warm-up."""
    models: List[str]                   # Models to warm up
    warmup_prompts: Dict[str, str]      # Model-specific warm-up prompts
    parallel_warmup: bool = True        # Warm up models in parallel
    timeout_seconds: float = 60.0       # Max time for warm-up
    retry_on_failure: bool = True


DEFAULT_WARMUP_PROMPTS = {
    "llama3.1:8b": "Classify this query: 'test query'",
    "phi3:mini": "Intent: 'hello'",
    "instructor-large": "Embed: 'test embedding'"
}


class ModelWarmUpManager:
    """
    Manages model warm-up to avoid cold start latency.

    Cold start costs:
    - Ollama model load: 2-5 seconds
    - Embedding model load: 1-3 seconds
    - First inference: 0.5-1 second (compilation)

    Warm-up strategy:
    1. On application start: warm up primary models
    2. On idle: keep models warm with periodic pings
    3. On model switch: pre-warm next likely model
    """

    def __init__(
        self,
        ollama_pool: OllamaConnectionPool,
        config: WarmUpConfig = None
    ):
        self.ollama_pool = ollama_pool
        self.config = config or WarmUpConfig(
            models=["llama3.1:8b"],
            warmup_prompts=DEFAULT_WARMUP_PROMPTS
        )
        self._warm_models: Dict[str, float] = {}  # model -> last warm time
        self._keep_warm_task: Optional[asyncio.Task] = None

    async def warm_up_all(self):
        """
        Warm up all configured models.

        Called during application startup.
        """
        logger.info(f"Starting model warm-up for: {self.config.models}")
        start_time = asyncio.get_event_loop().time()

        if self.config.parallel_warmup:
            tasks = [
                self._warm_up_model(model)
                for model in self.config.models
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for model in self.config.models:
                result = await self._warm_up_model(model)
                results.append(result)

        elapsed = asyncio.get_event_loop().time() - start_time
        success_count = sum(1 for r in results if r is True)

        logger.info(
            f"Model warm-up complete: {success_count}/{len(self.config.models)} "
            f"models warmed in {elapsed:.2f}s"
        )

        return success_count == len(self.config.models)

    async def _warm_up_model(self, model: str) -> bool:
        """Warm up a single model."""
        prompt = self.config.warmup_prompts.get(model, "Hello")

        try:
            await self.ollama_pool.generate(
                model=model,
                prompt=prompt,
                num_predict=1  # Minimal generation
            )
            self._warm_models[model] = asyncio.get_event_loop().time()
            logger.debug(f"Model {model} warmed up successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to warm up model {model}: {e}")
            if self.config.retry_on_failure:
                await asyncio.sleep(1.0)
                try:
                    await self.ollama_pool.generate(
                        model=model,
                        prompt=prompt,
                        num_predict=1
                    )
                    self._warm_models[model] = asyncio.get_event_loop().time()
                    return True
                except Exception:
                    pass
            return False

    async def start_keep_warm(self, interval_seconds: float = 60.0):
        """Start background task to keep models warm."""
        self._keep_warm_task = asyncio.create_task(
            self._keep_warm_loop(interval_seconds)
        )

    async def stop_keep_warm(self):
        """Stop the keep-warm background task."""
        if self._keep_warm_task:
            self._keep_warm_task.cancel()
            try:
                await self._keep_warm_task
            except asyncio.CancelledError:
                pass

    async def _keep_warm_loop(self, interval: float):
        """Periodic ping to keep models loaded."""
        while True:
            await asyncio.sleep(interval)
            for model in self.config.models:
                if model in self._warm_models:
                    # Send minimal request to keep model in memory
                    try:
                        await self.ollama_pool.generate(
                            model=model,
                            prompt="ping",
                            num_predict=1
                        )
                        self._warm_models[model] = asyncio.get_event_loop().time()
                    except Exception:
                        pass  # Silent failure for keep-warm
```

---

### 4. Query Plan Optimizer

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR_ONLY = "vector"
    GRAPH_ONLY = "graph"
    TEMPORAL_FIRST = "temporal_first"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    CAUSAL_CHAIN = "causal_chain"


@dataclass
class QueryPlanCost:
    """Estimated cost for a query plan."""
    latency_ms: float          # Estimated latency
    resource_usage: float      # 0-1 scale
    cache_probability: float   # Likelihood of cache hit
    accuracy_score: float      # Expected relevance

    @property
    def total_cost(self) -> float:
        """Weighted cost function."""
        return (
            self.latency_ms * 0.4 +
            self.resource_usage * 100 * 0.2 +
            (1 - self.cache_probability) * 100 * 0.2 +
            (1 - self.accuracy_score) * 100 * 0.2
        )


@dataclass
class QueryPlan:
    """Execution plan for a query."""
    strategy: RetrievalStrategy
    estimated_cost: QueryPlanCost
    steps: List[Dict[str, Any]]
    parallel_steps: List[List[str]]  # Steps that can run in parallel
    early_termination_threshold: float  # Stop if confidence exceeds
    timeout_ms: float


class QueryPlanOptimizer:
    """
    Cost-based query plan optimization.

    Optimizations:
    1. Strategy selection based on query type and history
    2. Parallel execution where possible
    3. Early termination for high-confidence results
    4. Cache-aware planning

    Integration Points:
    - QueryRouter: Receives intent classification
    - MultiLayerCache: Checks cache state for planning
    - PerformanceProfiler: Uses historical latency data
    """

    def __init__(
        self,
        cache: MultiLayerCache,
        profiler: "PerformanceProfiler"
    ):
        self.cache = cache
        self.profiler = profiler
        self.strategy_costs = self._initialize_strategy_costs()

    def _initialize_strategy_costs(self) -> Dict[RetrievalStrategy, QueryPlanCost]:
        """Initialize baseline costs for each strategy."""
        return {
            RetrievalStrategy.VECTOR_ONLY: QueryPlanCost(
                latency_ms=50, resource_usage=0.3,
                cache_probability=0.7, accuracy_score=0.75
            ),
            RetrievalStrategy.GRAPH_ONLY: QueryPlanCost(
                latency_ms=100, resource_usage=0.4,
                cache_probability=0.5, accuracy_score=0.85
            ),
            RetrievalStrategy.TEMPORAL_FIRST: QueryPlanCost(
                latency_ms=150, resource_usage=0.5,
                cache_probability=0.4, accuracy_score=0.90
            ),
            RetrievalStrategy.HYBRID_PARALLEL: QueryPlanCost(
                latency_ms=200, resource_usage=0.7,
                cache_probability=0.3, accuracy_score=0.92
            ),
            RetrievalStrategy.HYBRID_SEQUENTIAL: QueryPlanCost(
                latency_ms=300, resource_usage=0.5,
                cache_probability=0.4, accuracy_score=0.93
            ),
            RetrievalStrategy.CAUSAL_CHAIN: QueryPlanCost(
                latency_ms=500, resource_usage=0.8,
                cache_probability=0.2, accuracy_score=0.95
            ),
        }

    def optimize(
        self,
        query: str,
        intent: "QueryIntent",
        constraints: Dict[str, Any] = None
    ) -> QueryPlan:
        """
        Generate optimized query plan.

        Args:
            query: User query
            intent: Classified intent
            constraints: Optional constraints (max_latency, min_accuracy)

        Returns:
            Optimized QueryPlan
        """
        constraints = constraints or {}
        max_latency = constraints.get("max_latency_ms", 1000)
        min_accuracy = constraints.get("min_accuracy", 0.7)

        # Get candidate strategies for this intent
        candidates = self._get_candidate_strategies(intent)

        # Adjust costs based on cache state
        adjusted_costs = self._adjust_for_cache_state(candidates, query)

        # Adjust costs based on historical performance
        adjusted_costs = self._adjust_for_history(adjusted_costs, intent)

        # Select best strategy within constraints
        best_strategy = self._select_best_strategy(
            adjusted_costs, max_latency, min_accuracy
        )

        # Build execution plan
        plan = self._build_plan(best_strategy, intent, max_latency)

        return plan

    def _get_candidate_strategies(
        self,
        intent: "QueryIntent"
    ) -> List[RetrievalStrategy]:
        """Get strategies suitable for this intent."""
        intent_type = intent.primary_intent

        strategy_map = {
            "temporal": [
                RetrievalStrategy.TEMPORAL_FIRST,
                RetrievalStrategy.HYBRID_SEQUENTIAL
            ],
            "causal": [
                RetrievalStrategy.CAUSAL_CHAIN,
                RetrievalStrategy.GRAPH_ONLY
            ],
            "exploratory": [
                RetrievalStrategy.HYBRID_PARALLEL,
                RetrievalStrategy.VECTOR_ONLY
            ],
            "factual": [
                RetrievalStrategy.VECTOR_ONLY,
                RetrievalStrategy.HYBRID_PARALLEL
            ],
            "code": [
                RetrievalStrategy.VECTOR_ONLY,  # CodeBERT embeddings
                RetrievalStrategy.GRAPH_ONLY
            ]
        }

        return strategy_map.get(intent_type, [RetrievalStrategy.HYBRID_PARALLEL])

    def _adjust_for_cache_state(
        self,
        candidates: List[RetrievalStrategy],
        query: str
    ) -> Dict[RetrievalStrategy, QueryPlanCost]:
        """Adjust costs based on cache state."""
        adjusted = {}

        for strategy in candidates:
            base_cost = self.strategy_costs[strategy]

            # Check if results likely cached
            cache_hit_probability = self._estimate_cache_hit(query, strategy)

            adjusted[strategy] = QueryPlanCost(
                latency_ms=base_cost.latency_ms * (1 - cache_hit_probability * 0.8),
                resource_usage=base_cost.resource_usage * (1 - cache_hit_probability * 0.5),
                cache_probability=cache_hit_probability,
                accuracy_score=base_cost.accuracy_score
            )

        return adjusted

    def _estimate_cache_hit(
        self,
        query: str,
        strategy: RetrievalStrategy
    ) -> float:
        """Estimate probability of cache hit."""
        # Use cache statistics
        layer_map = {
            RetrievalStrategy.VECTOR_ONLY: CacheLayer.EMBEDDING,
            RetrievalStrategy.GRAPH_ONLY: CacheLayer.GRAPH_TRAVERSAL,
            RetrievalStrategy.TEMPORAL_FIRST: CacheLayer.TEMPORAL_INDEX,
        }

        layer = layer_map.get(strategy, CacheLayer.QUERY_RESULT)
        return self.cache.stats.hit_rate(layer)

    def _adjust_for_history(
        self,
        costs: Dict[RetrievalStrategy, QueryPlanCost],
        intent: "QueryIntent"
    ) -> Dict[RetrievalStrategy, QueryPlanCost]:
        """Adjust costs based on historical performance."""
        adjusted = {}

        for strategy, cost in costs.items():
            historical_latency = self.profiler.get_avg_latency(
                strategy.value, intent.primary_intent
            )

            if historical_latency:
                # Blend historical with baseline
                blended_latency = (cost.latency_ms + historical_latency) / 2
                adjusted[strategy] = QueryPlanCost(
                    latency_ms=blended_latency,
                    resource_usage=cost.resource_usage,
                    cache_probability=cost.cache_probability,
                    accuracy_score=cost.accuracy_score
                )
            else:
                adjusted[strategy] = cost

        return adjusted

    def _select_best_strategy(
        self,
        costs: Dict[RetrievalStrategy, QueryPlanCost],
        max_latency: float,
        min_accuracy: float
    ) -> RetrievalStrategy:
        """Select best strategy within constraints."""
        valid_strategies = [
            (strategy, cost)
            for strategy, cost in costs.items()
            if cost.latency_ms <= max_latency and cost.accuracy_score >= min_accuracy
        ]

        if not valid_strategies:
            # Fallback to fastest strategy
            return min(costs.items(), key=lambda x: x[1].latency_ms)[0]

        # Select lowest total cost
        return min(valid_strategies, key=lambda x: x[1].total_cost)[0]

    def _build_plan(
        self,
        strategy: RetrievalStrategy,
        intent: "QueryIntent",
        timeout_ms: float
    ) -> QueryPlan:
        """Build detailed execution plan."""
        cost = self.strategy_costs[strategy]

        # Define steps based on strategy
        steps = self._get_strategy_steps(strategy)
        parallel_steps = self._get_parallel_groups(strategy)

        return QueryPlan(
            strategy=strategy,
            estimated_cost=cost,
            steps=steps,
            parallel_steps=parallel_steps,
            early_termination_threshold=0.95,
            timeout_ms=timeout_ms
        )

    def _get_strategy_steps(self, strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
        """Get execution steps for strategy."""
        step_definitions = {
            RetrievalStrategy.VECTOR_ONLY: [
                {"name": "embed_query", "component": "embedding_router"},
                {"name": "vector_search", "component": "vector_store"},
                {"name": "rank_results", "component": "ranker"}
            ],
            RetrievalStrategy.TEMPORAL_FIRST: [
                {"name": "extract_temporal", "component": "temporal_parser"},
                {"name": "temporal_filter", "component": "temporal_index"},
                {"name": "embed_query", "component": "embedding_router"},
                {"name": "vector_search", "component": "vector_store"},
                {"name": "temporal_boost", "component": "ranker"},
                {"name": "rank_results", "component": "ranker"}
            ],
            RetrievalStrategy.HYBRID_PARALLEL: [
                {"name": "embed_query", "component": "embedding_router"},
                {"name": "vector_search", "component": "vector_store"},
                {"name": "graph_search", "component": "pkg"},
                {"name": "fuse_results", "component": "fusion"},
                {"name": "rank_results", "component": "ranker"}
            ],
            RetrievalStrategy.CAUSAL_CHAIN: [
                {"name": "identify_anchor", "component": "causal_retriever"},
                {"name": "traverse_causes", "component": "pkg"},
                {"name": "traverse_effects", "component": "pkg"},
                {"name": "build_chain", "component": "causal_retriever"},
                {"name": "rank_by_relevance", "component": "ranker"}
            ]
        }

        return step_definitions.get(strategy, [])

    def _get_parallel_groups(self, strategy: RetrievalStrategy) -> List[List[str]]:
        """Get groups of steps that can execute in parallel."""
        parallel_definitions = {
            RetrievalStrategy.HYBRID_PARALLEL: [
                ["vector_search", "graph_search"]
            ],
            RetrievalStrategy.CAUSAL_CHAIN: [
                ["traverse_causes", "traverse_effects"]
            ]
        }

        return parallel_definitions.get(strategy, [])
```

---

### 5. Performance Profiler

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import json


@dataclass
class LatencyBreakdown:
    """Detailed latency breakdown for a query."""
    query_id: str
    total_ms: float
    components: Dict[str, float]  # Component -> latency
    timestamp: datetime
    strategy: str
    intent_type: str
    cache_hits: List[str]
    cache_misses: List[str]


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    timestamp: datetime
    p50_latency: float
    p95_latency: float
    p99_latency: float
    cache_hit_rate: float
    queries_per_second: float
    error_rate: float


class PerformanceProfiler:
    """
    Performance monitoring and profiling.

    Tracks:
    - Latency breakdown by component
    - Cache effectiveness
    - Query throughput
    - Error rates

    Integration Points:
    - QueryPlanOptimizer: Provides historical data for planning
    - SearchQualityFeedback: Correlates performance with quality
    - Monitoring/Alerting: Exports metrics
    """

    def __init__(
        self,
        retention_hours: int = 24,
        snapshot_interval_minutes: int = 5
    ):
        self.retention_hours = retention_hours
        self.snapshot_interval = snapshot_interval_minutes

        self.latency_records: List[LatencyBreakdown] = []
        self.snapshots: List[PerformanceSnapshot] = []

        # Aggregated statistics
        self._latency_by_strategy: Dict[str, List[float]] = defaultdict(list)
        self._latency_by_intent: Dict[str, List[float]] = defaultdict(list)
        self._latency_by_component: Dict[str, List[float]] = defaultdict(list)

    def record_query(
        self,
        query_id: str,
        total_ms: float,
        components: Dict[str, float],
        strategy: str,
        intent_type: str,
        cache_hits: List[str],
        cache_misses: List[str]
    ):
        """Record latency breakdown for a query."""
        record = LatencyBreakdown(
            query_id=query_id,
            total_ms=total_ms,
            components=components,
            timestamp=datetime.utcnow(),
            strategy=strategy,
            intent_type=intent_type,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )

        self.latency_records.append(record)

        # Update aggregates
        self._latency_by_strategy[strategy].append(total_ms)
        self._latency_by_intent[intent_type].append(total_ms)
        for component, latency in components.items():
            self._latency_by_component[component].append(latency)

        # Cleanup old records
        self._cleanup_old_records()

    def get_avg_latency(
        self,
        strategy: str = None,
        intent_type: str = None
    ) -> Optional[float]:
        """Get average latency, optionally filtered."""
        if strategy and strategy in self._latency_by_strategy:
            values = self._latency_by_strategy[strategy]
            return statistics.mean(values) if values else None

        if intent_type and intent_type in self._latency_by_intent:
            values = self._latency_by_intent[intent_type]
            return statistics.mean(values) if values else None

        if self.latency_records:
            return statistics.mean(r.total_ms for r in self.latency_records)

        return None

    def get_percentile_latency(self, percentile: float) -> Optional[float]:
        """Get latency at specified percentile."""
        if not self.latency_records:
            return None

        latencies = sorted(r.total_ms for r in self.latency_records)
        index = int(len(latencies) * percentile / 100)
        return latencies[min(index, len(latencies) - 1)]

    def get_component_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics by component."""
        breakdown = {}

        for component, latencies in self._latency_by_component.items():
            if latencies:
                breakdown[component] = {
                    "mean": statistics.mean(latencies),
                    "p50": statistics.median(latencies),
                    "p95": self._percentile(latencies, 95),
                    "p99": self._percentile(latencies, 99),
                    "min": min(latencies),
                    "max": max(latencies)
                }

        return breakdown

    def identify_bottlenecks(
        self,
        threshold_ratio: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_ratio: Component is bottleneck if it takes
                            more than this ratio of total latency

        Returns:
            List of bottleneck components with recommendations
        """
        bottlenecks = []
        breakdown = self.get_component_breakdown()

        avg_total = self.get_avg_latency() or 1.0

        for component, stats in breakdown.items():
            ratio = stats["mean"] / avg_total
            if ratio > threshold_ratio:
                bottlenecks.append({
                    "component": component,
                    "mean_latency_ms": stats["mean"],
                    "ratio_of_total": ratio,
                    "recommendation": self._get_recommendation(component, stats)
                })

        return sorted(bottlenecks, key=lambda x: x["ratio_of_total"], reverse=True)

    def _get_recommendation(
        self,
        component: str,
        stats: Dict[str, float]
    ) -> str:
        """Get optimization recommendation for bottleneck."""
        recommendations = {
            "embed_query": "Consider caching embeddings or using faster embedding model",
            "vector_search": "Optimize vector index or reduce search scope",
            "graph_search": "Add graph indexes or limit traversal depth",
            "temporal_filter": "Pre-compute temporal indexes",
            "llm_intent": "Enable LLM inference caching or use faster model",
            "rank_results": "Simplify ranking function or reduce result set"
        }

        return recommendations.get(
            component,
            f"Review {component} implementation for optimization opportunities"
        )

    def create_snapshot(self) -> PerformanceSnapshot:
        """Create point-in-time performance snapshot."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            p50_latency=self.get_percentile_latency(50) or 0.0,
            p95_latency=self.get_percentile_latency(95) or 0.0,
            p99_latency=self.get_percentile_latency(99) or 0.0,
            cache_hit_rate=self._calculate_cache_hit_rate(),
            queries_per_second=self._calculate_qps(),
            error_rate=0.0  # TODO: Implement error tracking
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent queries."""
        if not self.latency_records:
            return 0.0

        total_cache_ops = 0
        total_hits = 0

        for record in self.latency_records[-100:]:  # Last 100 queries
            total_cache_ops += len(record.cache_hits) + len(record.cache_misses)
            total_hits += len(record.cache_hits)

        return total_hits / total_cache_ops if total_cache_ops > 0 else 0.0

    def _calculate_qps(self) -> float:
        """Calculate queries per second."""
        if len(self.latency_records) < 2:
            return 0.0

        recent = self.latency_records[-100:]
        time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()

        return len(recent) / time_span if time_span > 0 else 0.0

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _cleanup_old_records(self):
        """Remove records older than retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)

        self.latency_records = [
            r for r in self.latency_records
            if r.timestamp > cutoff
        ]

        # Rebuild aggregates periodically
        if len(self.latency_records) % 1000 == 0:
            self._rebuild_aggregates()

    def _rebuild_aggregates(self):
        """Rebuild aggregate statistics from records."""
        self._latency_by_strategy = defaultdict(list)
        self._latency_by_intent = defaultdict(list)
        self._latency_by_component = defaultdict(list)

        for record in self.latency_records:
            self._latency_by_strategy[record.strategy].append(record.total_ms)
            self._latency_by_intent[record.intent_type].append(record.total_ms)
            for component, latency in record.components.items():
                self._latency_by_component[component].append(latency)

    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for monitoring integration."""
        return {
            "latency": {
                "p50": self.get_percentile_latency(50),
                "p95": self.get_percentile_latency(95),
                "p99": self.get_percentile_latency(99),
                "mean": self.get_avg_latency()
            },
            "cache": {
                "hit_rate": self._calculate_cache_hit_rate()
            },
            "throughput": {
                "qps": self._calculate_qps()
            },
            "bottlenecks": self.identify_bottlenecks(),
            "by_strategy": {
                strategy: statistics.mean(latencies)
                for strategy, latencies in self._latency_by_strategy.items()
                if latencies
            },
            "by_component": self.get_component_breakdown()
        }
```

---

## Testing Strategy

### Unit Tests

```python
class TestMultiLayerCache:
    """Unit tests for multi-layer caching."""

    def test_cache_set_get_exact_match(self):
        """Test exact match cache operations."""
        cache = MultiLayerCache()

        cache.set(CacheLayer.QUERY_RESULT, "test query", {"results": []})
        value, hit = cache.get(CacheLayer.QUERY_RESULT, "test query")

        assert hit is True
        assert value == {"results": []}

    def test_cache_semantic_similarity_match(self):
        """Test semantic similarity cache lookup."""
        cache = MultiLayerCache()

        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.99, 0.1, 0.0])  # Very similar

        cache.set(
            CacheLayer.LLM_INTENT,
            "what happened yesterday",
            "temporal",
            embedding1
        )

        value, hit = cache.get(
            CacheLayer.LLM_INTENT,
            "what occurred yesterday",
            embedding2
        )

        assert hit is True
        assert value == "temporal"

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        config = {
            CacheLayer.QUERY_RESULT: CacheConfig(
                layer=CacheLayer.QUERY_RESULT,
                ttl_seconds=1,  # 1 second TTL
                max_entries=100,
                invalidate_on_pkg_mutation=True,
                invalidate_on_schema_change=True,
                use_semantic_similarity=False,
                similarity_threshold=1.0
            )
        }
        cache = MultiLayerCache(configs=config)

        cache.set(CacheLayer.QUERY_RESULT, "test", "value")

        time.sleep(1.5)

        value, hit = cache.get(CacheLayer.QUERY_RESULT, "test")
        assert hit is False

    def test_entity_invalidation(self):
        """Test invalidation by entity IDs."""
        cache = MultiLayerCache()

        cache.set(
            CacheLayer.GRAPH_TRAVERSAL,
            "query1",
            "result1",
            related_entities=["entity_123"]
        )
        cache.set(
            CacheLayer.GRAPH_TRAVERSAL,
            "query2",
            "result2",
            related_entities=["entity_456"]
        )

        cache.invalidate_for_entities(["entity_123"])

        _, hit1 = cache.get(CacheLayer.GRAPH_TRAVERSAL, "query1")
        _, hit2 = cache.get(CacheLayer.GRAPH_TRAVERSAL, "query2")

        assert hit1 is False  # Invalidated
        assert hit2 is True   # Still valid


class TestOllamaConnectionPool:
    """Tests for Ollama connection management."""

    @pytest.mark.asyncio
    async def test_connection_reuse(self):
        """Test connection pooling reuses connections."""
        pool = OllamaConnectionPool()
        await pool.initialize()

        # Multiple requests should reuse connections
        results = await asyncio.gather(
            pool.generate("llama3.1:8b", "test1"),
            pool.generate("llama3.1:8b", "test2"),
            pool.generate("llama3.1:8b", "test3")
        )

        assert len(results) == 3
        await pool.close()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test automatic retry on transient failures."""
        # Mock server with intermittent failures
        pool = OllamaConnectionPool(
            OllamaConnectionConfig(max_retries=3, retry_delay=0.1)
        )

        # Should succeed after retries
        # (requires mock server setup in integration tests)


class TestQueryPlanOptimizer:
    """Tests for query plan optimization."""

    def test_strategy_selection_temporal(self):
        """Test temporal query selects temporal strategy."""
        optimizer = QueryPlanOptimizer(
            cache=MultiLayerCache(),
            profiler=PerformanceProfiler()
        )

        intent = QueryIntent(primary_intent="temporal", confidence=0.9)
        plan = optimizer.optimize("what happened last week", intent)

        assert plan.strategy in [
            RetrievalStrategy.TEMPORAL_FIRST,
            RetrievalStrategy.HYBRID_SEQUENTIAL
        ]

    def test_latency_constraint_respected(self):
        """Test optimizer respects latency constraints."""
        optimizer = QueryPlanOptimizer(
            cache=MultiLayerCache(),
            profiler=PerformanceProfiler()
        )

        intent = QueryIntent(primary_intent="causal", confidence=0.9)
        plan = optimizer.optimize(
            "why did this happen",
            intent,
            constraints={"max_latency_ms": 200}
        )

        assert plan.estimated_cost.latency_ms <= 200
```

### Integration Tests

```python
class TestPerformanceIntegration:
    """Integration tests for performance systems."""

    @pytest.mark.integration
    async def test_end_to_end_with_caching(self):
        """Test full query with caching enabled."""
        api = create_hybrid_search_api(caching_enabled=True)

        # First query - cache miss
        start1 = time.time()
        results1 = await api.search("test query", top_k=10)
        latency1 = (time.time() - start1) * 1000

        # Second identical query - cache hit
        start2 = time.time()
        results2 = await api.search("test query", top_k=10)
        latency2 = (time.time() - start2) * 1000

        assert results1 == results2  # Same results
        assert latency2 < latency1 * 0.5  # Significant speedup

    @pytest.mark.integration
    async def test_warm_up_reduces_cold_start(self):
        """Test model warm-up reduces cold start latency."""
        pool = OllamaConnectionPool()
        warmup = ModelWarmUpManager(pool)

        await pool.initialize()

        # Cold query (no warm-up)
        start_cold = time.time()
        await pool.generate("llama3.1:8b", "cold test")
        cold_latency = time.time() - start_cold

        # Warm up
        await warmup.warm_up_all()

        # Warm query
        start_warm = time.time()
        await pool.generate("llama3.1:8b", "warm test")
        warm_latency = time.time() - start_warm

        # Warm queries should be faster (model already loaded)
        assert warm_latency < cold_latency

        await pool.close()


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.performance
    async def test_p95_latency_target(self):
        """Validate <1s latency for 95% of queries."""
        api = create_hybrid_search_api()
        test_queries = load_benchmark_queries(n=100)

        latencies = []
        for query in test_queries:
            start = time.time()
            await api.search(query, top_k=10)
            latencies.append((time.time() - start) * 1000)

        p95 = np.percentile(latencies, 95)

        assert p95 < 1000, f"P95 latency {p95}ms exceeds 1000ms target"

    @pytest.mark.performance
    def test_cache_hit_rate_target(self):
        """Validate >60% cache hit rate."""
        api = create_hybrid_search_api()

        # Simulate realistic query pattern (some repeats)
        queries = ["query_" + str(i % 50) for i in range(200)]

        for query in queries:
            api.search(query, top_k=10)

        hit_rate = api.cache.stats.overall_hit_rate()

        assert hit_rate > 0.6, f"Cache hit rate {hit_rate:.2%} below 60% target"
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| P95 Latency | <1000ms | `profiler.get_percentile_latency(95)` |
| P50 Latency | <200ms | `profiler.get_percentile_latency(50)` |
| Cache Hit Rate | >60% | `cache.stats.overall_hit_rate()` |
| Semantic Cache Hits | >20% of hits | `cache.stats.semantic_hits / cache.stats.hits` |
| Cold Start | <5s | Time from startup to first query |
| Warm Query | <100ms | Cached query latency |

---

## Dependencies

- **MultiLayerCache**: Multi-layer cache architecture
- **OllamaConnectionPool**: LLM connection management
- **ModelWarmUpManager**: Cold start mitigation
- **QueryPlanOptimizer**: Cost-based planning
- **PerformanceProfiler**: Latency tracking and bottleneck detection

### External Dependencies
- Ollama server (localhost:11434)
- PKGEventBus for cache invalidation
- SchemaVersionManager for schema change detection
- QueryEmbeddingRouter for semantic cache matching

---

## Option B Compliance

- **Ghost Model Frozen**: Ollama models used for inference only, no fine-tuning
- **Experiential Learning Integration**: Cache patterns inform query templates
- **Performance Gates**: P95 <1s validated before production
- **Local-First**: All caching and optimization runs on-device

---

**This module delivers production-grade performance with intelligent caching and LLM optimization.**

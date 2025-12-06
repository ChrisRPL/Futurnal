"""Multi-Layer Cache for Hybrid Search.

Provides hierarchical caching with semantic similarity matching for
efficient query result, embedding, LLM intent, and graph traversal caching.

Key Features:
- 5 cache layers with configurable TTLs
- Semantic similarity matching for cache lookups
- Entity-based cache invalidation on PKG mutations
- LRU eviction with hit count tracking
- Integration with PKGEventEmitter for real-time invalidation

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md

Option B Compliance:
- Local-first caching on-device
- Invalidation on PKG/schema mutations maintains consistency
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from futurnal.pkg.sync.emitter import PKGEventEmitter
    from futurnal.pkg.sync.events import PKGEvent
    from futurnal.search.hybrid.query_router import QueryEmbeddingRouter

logger = logging.getLogger(__name__)


class CacheLayer(str, Enum):
    """Cache layers with different characteristics."""

    QUERY_RESULT = "query_result"  # Final search results (TTL: 5 min)
    EMBEDDING = "embedding"  # Query/entity embeddings (TTL: 1 hour)
    LLM_INTENT = "llm_intent"  # Intent classification results (TTL: 30 min)
    GRAPH_TRAVERSAL = "graph_traversal"  # PKG traversal results (invalidate on mutation)
    TEMPORAL_INDEX = "temporal_index"  # Temporal query indexes (TTL: 15 min)


@dataclass
class CacheConfig:
    """Configuration for each cache layer."""

    layer: CacheLayer
    ttl_seconds: int
    max_entries: int
    invalidate_on_pkg_mutation: bool
    invalidate_on_schema_change: bool
    use_semantic_similarity: bool  # For embedding-based cache lookup
    similarity_threshold: float  # Min similarity for cache hit (0.95 default)


DEFAULT_CACHE_CONFIGS: Dict[CacheLayer, CacheConfig] = {
    CacheLayer.QUERY_RESULT: CacheConfig(
        layer=CacheLayer.QUERY_RESULT,
        ttl_seconds=300,  # 5 minutes
        max_entries=10000,
        invalidate_on_pkg_mutation=True,
        invalidate_on_schema_change=True,
        use_semantic_similarity=True,
        similarity_threshold=0.95,
    ),
    CacheLayer.EMBEDDING: CacheConfig(
        layer=CacheLayer.EMBEDDING,
        ttl_seconds=3600,  # 1 hour
        max_entries=50000,
        invalidate_on_pkg_mutation=False,
        invalidate_on_schema_change=True,
        use_semantic_similarity=False,
        similarity_threshold=1.0,
    ),
    CacheLayer.LLM_INTENT: CacheConfig(
        layer=CacheLayer.LLM_INTENT,
        ttl_seconds=1800,  # 30 minutes
        max_entries=5000,
        invalidate_on_pkg_mutation=False,
        invalidate_on_schema_change=False,
        use_semantic_similarity=True,
        similarity_threshold=0.98,  # Higher threshold for intent
    ),
    CacheLayer.GRAPH_TRAVERSAL: CacheConfig(
        layer=CacheLayer.GRAPH_TRAVERSAL,
        ttl_seconds=900,  # 15 minutes (but invalidated on mutation)
        max_entries=20000,
        invalidate_on_pkg_mutation=True,
        invalidate_on_schema_change=True,
        use_semantic_similarity=False,
        similarity_threshold=1.0,
    ),
    CacheLayer.TEMPORAL_INDEX: CacheConfig(
        layer=CacheLayer.TEMPORAL_INDEX,
        ttl_seconds=900,  # 15 minutes
        max_entries=10000,
        invalidate_on_pkg_mutation=True,
        invalidate_on_schema_change=False,
        use_semantic_similarity=False,
        similarity_threshold=1.0,
    ),
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
    query_hash: str  # For exact match fallback
    related_entities: List[str]  # For invalidation targeting


@dataclass
class CacheStatistics:
    """Cache performance statistics for monitoring."""

    hits: Dict[CacheLayer, int] = field(default_factory=dict)
    misses: Dict[CacheLayer, int] = field(default_factory=dict)
    semantic_hits: Dict[CacheLayer, int] = field(default_factory=dict)
    sets: Dict[CacheLayer, int] = field(default_factory=dict)
    evictions: Dict[CacheLayer, int] = field(default_factory=dict)
    invalidations: Dict[CacheLayer, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize counters for all layers."""
        for layer in CacheLayer:
            self.hits.setdefault(layer, 0)
            self.misses.setdefault(layer, 0)
            self.semantic_hits.setdefault(layer, 0)
            self.sets.setdefault(layer, 0)
            self.evictions.setdefault(layer, 0)
            self.invalidations.setdefault(layer, 0)

    def record_hit(self, layer: CacheLayer, semantic: bool = False) -> None:
        """Record a cache hit."""
        self.hits[layer] += 1
        if semantic:
            self.semantic_hits[layer] += 1

    def record_miss(self, layer: CacheLayer) -> None:
        """Record a cache miss."""
        self.misses[layer] += 1

    def record_set(self, layer: CacheLayer) -> None:
        """Record a cache set operation."""
        self.sets[layer] += 1

    def record_eviction(self, layer: CacheLayer) -> None:
        """Record a cache eviction."""
        self.evictions[layer] += 1

    def record_invalidation(
        self, layer: CacheLayer, reason: str = "manual", count: int = 1
    ) -> None:
        """Record cache invalidation."""
        self.invalidations[layer] += count
        logger.debug(f"Cache invalidation: layer={layer.value}, reason={reason}, count={count}")

    def hit_rate(self, layer: CacheLayer) -> float:
        """Calculate hit rate for a specific layer."""
        total = self.hits[layer] + self.misses[layer]
        return self.hits[layer] / total if total > 0 else 0.0

    def overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all layers."""
        total_hits = sum(self.hits.values())
        total_misses = sum(self.misses.values())
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0

    def semantic_hit_ratio(self) -> float:
        """Calculate ratio of semantic hits to total hits."""
        total_hits = sum(self.hits.values())
        total_semantic = sum(self.semantic_hits.values())
        return total_semantic / total_hits if total_hits > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary."""
        return {
            "hits": {k.value: v for k, v in self.hits.items()},
            "misses": {k.value: v for k, v in self.misses.items()},
            "semantic_hits": {k.value: v for k, v in self.semantic_hits.items()},
            "sets": {k.value: v for k, v in self.sets.items()},
            "evictions": {k.value: v for k, v in self.evictions.items()},
            "invalidations": {k.value: v for k, v in self.invalidations.items()},
            "overall_hit_rate": self.overall_hit_rate(),
            "semantic_hit_ratio": self.semantic_hit_ratio(),
            "by_layer": {
                layer.value: {"hit_rate": self.hit_rate(layer)}
                for layer in CacheLayer
            },
        }


class MultiLayerCache:
    """Hierarchical cache with semantic similarity matching.

    Implements multi-layer caching strategy:
    1. L1: Query results (fast, short TTL)
    2. L2: Embeddings (medium, longer TTL)
    3. L3: LLM intents (slow to compute, medium TTL)
    4. L4: Graph traversals (invalidate on mutation)
    5. L5: Temporal indexes (invalidate on mutation)

    Integration Points:
    - PKGEventEmitter: Subscribe to mutations for invalidation
    - SchemaVersionManager: Subscribe to schema changes
    - QueryEmbeddingRouter: Get embeddings for semantic matching

    Example:
        >>> cache = MultiLayerCache()
        >>> cache.set(CacheLayer.QUERY_RESULT, "test query", {"results": []})
        >>> value, hit = cache.get(CacheLayer.QUERY_RESULT, "test query")
        >>> assert hit is True
    """

    def __init__(
        self,
        configs: Optional[Dict[CacheLayer, CacheConfig]] = None,
        embedding_router: Optional["QueryEmbeddingRouter"] = None,
        event_handler_chain: Optional[Callable[["PKGEvent"], None]] = None,
    ) -> None:
        """Initialize multi-layer cache.

        Args:
            configs: Optional custom cache configurations per layer
            embedding_router: Optional QueryEmbeddingRouter for semantic matching
            event_handler_chain: Optional existing event handler to chain with

        Note:
            To integrate with PKGEventEmitter, create the cache first, then
            create the emitter with cache.create_event_handler(existing_handler).
        """
        self.configs = configs or DEFAULT_CACHE_CONFIGS.copy()
        self.embedding_router = embedding_router
        self._event_handler_chain = event_handler_chain

        self.caches: Dict[CacheLayer, Dict[str, CacheEntry]] = {
            layer: {} for layer in CacheLayer
        }
        self.stats = CacheStatistics()

        logger.info("Initialized MultiLayerCache with %d layers", len(CacheLayer))

    def create_event_handler(
        self,
        existing_handler: Optional[Callable[["PKGEvent"], None]] = None,
    ) -> Callable[["PKGEvent"], None]:
        """Create event handler that combines cache invalidation with existing handler.

        This allows chaining the cache's invalidation logic with an existing
        PKGEventEmitter handler (e.g., embedding sync handler).

        Args:
            existing_handler: Optional existing handler to chain

        Returns:
            Combined event handler function

        Example:
            >>> cache = MultiLayerCache()
            >>> combined_handler = cache.create_event_handler(sync_handler.handle_event)
            >>> emitter = PKGEventEmitter(event_handler=combined_handler)
        """

        def combined_handler(event: "PKGEvent") -> None:
            # First call existing handler if provided
            if existing_handler:
                existing_handler(event)

            # Then handle cache invalidation
            self._handle_pkg_event(event)

        return combined_handler

    def _handle_pkg_event(self, event: "PKGEvent") -> None:
        """Handle PKG mutation event for cache invalidation."""
        from futurnal.pkg.sync.events import SyncEventType

        entity_ids = []
        if event.entity_id:
            entity_ids.append(event.entity_id)

        if event.event_type in (
            SyncEventType.ENTITY_CREATED,
            SyncEventType.ENTITY_UPDATED,
            SyncEventType.ENTITY_DELETED,
        ):
            if entity_ids:
                self.invalidate_for_entities(entity_ids)
        elif event.event_type == SyncEventType.RELATIONSHIP_CREATED:
            # Invalidate entries related to relationship endpoints
            if event.entity_id:
                entity_ids.append(event.entity_id)
            if hasattr(event, "target_id") and event.target_id:
                entity_ids.append(event.target_id)
            if entity_ids:
                self.invalidate_for_entities(entity_ids)

    def get(
        self,
        layer: CacheLayer,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Any], bool]:
        """Get value from cache with semantic similarity matching.

        Args:
            layer: Cache layer to query
            query: Query string for cache key
            query_embedding: Optional embedding for semantic matching

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
        related_entities: Optional[List[str]] = None,
    ) -> None:
        """Store value in cache with optional embedding for semantic matching.

        Args:
            layer: Cache layer to store in
            query: Query string for cache key
            value: Value to cache
            query_embedding: Optional embedding for semantic matching
            related_entities: Optional list of entity IDs for invalidation targeting
        """
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
            related_entities=related_entities or [],
        )

        cache[query_hash] = entry
        self.stats.record_set(layer)

    def invalidate_for_entities(self, entity_ids: List[str]) -> None:
        """Invalidate cache entries related to specific entities.

        Called when PKG mutations affect specific entities.

        Args:
            entity_ids: List of entity IDs to invalidate caches for
        """
        entity_set = set(entity_ids)

        for layer, config in self.configs.items():
            if config.invalidate_on_pkg_mutation:
                cache = self.caches[layer]
                keys_to_remove = []

                for key, entry in cache.items():
                    if any(eid in entity_set for eid in entry.related_entities):
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del cache[key]

                if keys_to_remove:
                    self.stats.record_invalidation(
                        layer,
                        reason="entity_mutation",
                        count=len(keys_to_remove),
                    )

    def invalidate_layer(self, layer: CacheLayer, reason: str = "manual") -> None:
        """Invalidate entire cache layer.

        Args:
            layer: Cache layer to invalidate
            reason: Reason for invalidation (for logging)
        """
        count = len(self.caches[layer])
        self.caches[layer] = {}
        self.stats.record_invalidation(layer, reason=reason, count=count)
        logger.info(f"Invalidated entire cache layer {layer.value}: {count} entries")

    def invalidate_on_schema_change(self, schema_version: str) -> None:
        """Invalidate caches affected by schema evolution.

        Args:
            schema_version: New schema version string
        """
        for layer, config in self.configs.items():
            if config.invalidate_on_schema_change:
                self.invalidate_layer(layer, reason=f"schema_change_{schema_version}")

    def _find_semantic_match(
        self,
        layer: CacheLayer,
        query_embedding: np.ndarray,
        threshold: float,
    ) -> Optional[CacheEntry]:
        """Find best semantic match above threshold.

        Args:
            layer: Cache layer to search
            query_embedding: Query embedding vector
            threshold: Minimum similarity threshold

        Returns:
            Best matching CacheEntry or None
        """
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
        """Compute cosine similarity between embeddings.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _hash_query(self, query: str) -> str:
        """Generate deterministic hash for query.

        Args:
            query: Query string

        Returns:
            16-character hex hash
        """
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired.

        Args:
            entry: Cache entry to check

        Returns:
            True if expired, False otherwise
        """
        return datetime.utcnow() > entry.expires_at

    def _evict_lru(self, layer: CacheLayer) -> None:
        """Evict least recently used entries.

        Uses combined LFU+LRU strategy: sort by hit count (LFU)
        then by creation time (LRU) as tiebreaker.

        Args:
            layer: Cache layer to evict from
        """
        cache = self.caches[layer]
        if not cache:
            return

        # Sort by hit_count (LFU) and created_at (LRU tiebreaker)
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: (x[1].hit_count, x[1].created_at),
        )

        # Remove bottom 10%
        to_remove = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:to_remove]:
            del cache[key]
            self.stats.record_eviction(layer)

    def get_size(self, layer: Optional[CacheLayer] = None) -> int:
        """Get number of entries in cache.

        Args:
            layer: Optional specific layer, or None for total

        Returns:
            Number of cache entries
        """
        if layer:
            return len(self.caches[layer])
        return sum(len(c) for c in self.caches.values())

    def clear(self) -> None:
        """Clear all cache layers."""
        for layer in CacheLayer:
            self.invalidate_layer(layer, reason="clear")

"""Schema-Aware Hybrid Retrieval Engine.

Main engine for hybrid search combining vector similarity and graph traversal
with schema version awareness and multi-model embedding integration.

Integrates:
- TemporalQueryEngine: For temporal-aware graph expansion
- CausalChainRetrieval: For causal graph expansion
- SchemaVersionedEmbeddingStore: For schema-filtered vector search
- QueryEmbeddingRouter: For intent-based query embedding

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Temporal-first design (temporal context in queries)
- Schema evolves autonomously
- Uses frozen models (via embedding service)
- Local-first processing
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import (
    GraphExpansionError,
    HybridSearchError,
    InvalidHybridQueryError,
    VectorSearchError,
)
from futurnal.search.hybrid.fusion import ResultFusion
from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
from futurnal.search.hybrid.schema_compat import SchemaVersionCompatibility
from futurnal.search.hybrid.types import (
    GraphSearchResult,
    HybridSearchQuery,
    HybridSearchResult,
    TemporalQueryContext,
    VectorSearchResult,
)

if TYPE_CHECKING:
    from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
    from futurnal.pkg.queries.temporal import TemporalGraphQueries
    from futurnal.privacy.audit import AuditLogger
    from futurnal.search.causal.retrieval import CausalChainRetrieval
    from futurnal.search.temporal.engine import TemporalQueryEngine

logger = logging.getLogger(__name__)


class SchemaAwareRetrieval:
    """Hybrid retrieval with schema version awareness and multi-model embeddings.

    Combines vector similarity and graph traversal, adapting strategies
    based on schema version and query intent.

    Search Flow:
    1. Determine query embedding type based on intent
    2. Generate query embedding using appropriate model
    3. Vector search with schema version filtering
    4. Graph expansion from top vector results (strategy based on intent)
    5. Adaptive weight adjustment based on intent and result counts
    6. Fuse and rank results

    Intent Strategies:
    - temporal: Favor graph expansion via temporal relationships
    - causal: Favor graph expansion via causal chains
    - lookup: Favor vector similarity (direct semantic match)
    - exploratory: Balanced vector + graph

    Example:
        >>> from futurnal.pkg.queries.temporal import TemporalGraphQueries
        >>> from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
        >>> from futurnal.search.temporal import TemporalQueryEngine
        >>> from futurnal.search.causal import CausalChainRetrieval
        >>> from futurnal.search.hybrid import (
        ...     SchemaAwareRetrieval,
        ...     QueryEmbeddingRouter,
        ...     HybridSearchConfig,
        ... )

        >>> # Initialize with all dependencies
        >>> retrieval = SchemaAwareRetrieval(
        ...     pkg_queries=pkg_queries,
        ...     embedding_store=embedding_store,
        ...     temporal_engine=temporal_engine,
        ...     causal_retrieval=causal_retrieval,
        ...     embedding_router=query_router,
        ... )

        >>> # Temporal search
        >>> results = retrieval.hybrid_search(
        ...     query="What happened in Q1?",
        ...     intent="temporal",
        ...     top_k=10,
        ... )

        >>> # Causal search
        >>> results = retrieval.hybrid_search(
        ...     query="What caused the project delay?",
        ...     intent="causal",
        ...     top_k=10,
        ... )
    """

    def __init__(
        self,
        pkg_queries: "TemporalGraphQueries",
        embedding_store: "SchemaVersionedEmbeddingStore",
        temporal_engine: Optional["TemporalQueryEngine"] = None,
        causal_retrieval: Optional["CausalChainRetrieval"] = None,
        embedding_router: Optional[QueryEmbeddingRouter] = None,
        config: Optional[HybridSearchConfig] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ) -> None:
        """Initialize schema-aware hybrid retrieval engine.

        Args:
            pkg_queries: PKG temporal queries service (required)
            embedding_store: Schema-versioned embedding store (required)
            temporal_engine: Temporal query engine (optional, for temporal expansion)
            causal_retrieval: Causal chain retrieval (optional, for causal expansion)
            embedding_router: Query embedding router (optional, uses fallback if None)
            config: Hybrid search configuration
            audit_logger: Optional audit logger for query tracking
        """
        self._pkg = pkg_queries
        self._embeddings = embedding_store
        self._temporal = temporal_engine
        self._causal = causal_retrieval
        self._router = embedding_router
        self._config = config or HybridSearchConfig()
        self._audit = audit_logger

        # Initialize schema compatibility checker
        self._schema_compat = SchemaVersionCompatibility(
            embedding_store=embedding_store,
            config=self._config,
        )

        # Initialize result fusion
        self._fusion = ResultFusion(config=self._config)

        logger.info(
            f"Initialized SchemaAwareRetrieval with "
            f"temporal_engine={temporal_engine is not None}, "
            f"causal_retrieval={causal_retrieval is not None}, "
            f"embedding_router={embedding_router is not None}"
        )

    @property
    def config(self) -> HybridSearchConfig:
        """Get the configuration."""
        return self._config

    def hybrid_search(
        self,
        query: str,
        intent: str = "exploratory",
        top_k: int = 20,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
        temporal_context: Optional[TemporalQueryContext] = None,
    ) -> List[HybridSearchResult]:
        """Execute hybrid search combining vector and graph retrieval.

        Main entry point for hybrid search. Combines vector similarity
        and graph traversal with adaptive weighting based on intent.

        Args:
            query: Natural language search query
            intent: Search intent: temporal, causal, lookup, exploratory
            top_k: Maximum number of results to return
            vector_weight: Weight for vector similarity (0-1)
            graph_weight: Weight for graph traversal (0-1)
            temporal_context: Optional temporal context for time-aware queries

        Returns:
            List of HybridSearchResult sorted by combined_score

        Raises:
            InvalidHybridQueryError: If query parameters are invalid
            VectorSearchError: If vector search fails
            GraphExpansionError: If graph expansion fails
            HybridSearchError: For other errors
        """
        start_time = time.perf_counter()

        # Validate inputs
        self._validate_query(query, intent, top_k, vector_weight, graph_weight)

        try:
            # Get current schema version
            current_version = self._schema_compat.get_current_schema_version()

            # Generate query embedding
            query_embedding = self._get_query_embedding(
                query, intent, temporal_context
            )

            # Vector retrieval with schema filtering
            vector_results = self._vector_search(
                query_embedding=query_embedding,
                top_k=top_k * self._config.vector_top_k_multiplier,
                current_version=current_version,
                intent=intent,
            )

            # Graph expansion from top vector results
            seed_entities = [
                r.entity_id for r in vector_results[: self._config.max_seed_entities]
            ]
            graph_results = self._graph_expansion(
                seed_entities=seed_entities,
                intent=intent,
            )

            # Adaptive weight adjustment
            adjusted_weights = self._adjust_weights(
                intent=intent,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
                vector_result_count=len(vector_results),
                graph_result_count=len(graph_results),
            )

            # Fuse results
            fused = self._fusion.fuse_results(
                vector_results=vector_results,
                graph_results=graph_results,
                vector_weight=adjusted_weights["vector"],
                graph_weight=adjusted_weights["graph"],
            )

            # Take top_k results
            results = fused[:top_k]

            # Log performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._log_search(
                query=query,
                intent=intent,
                elapsed_ms=elapsed_ms,
                result_count=len(results),
                vector_count=len(vector_results),
                graph_count=len(graph_results),
            )

            return results

        except (InvalidHybridQueryError, VectorSearchError, GraphExpansionError):
            raise
        except Exception as e:
            raise HybridSearchError(f"Hybrid search failed: {e}") from e

    def hybrid_search_with_query(
        self,
        query: HybridSearchQuery,
    ) -> List[HybridSearchResult]:
        """Execute hybrid search using HybridSearchQuery model.

        Convenience method that accepts a validated query model.

        Args:
            query: HybridSearchQuery with all parameters

        Returns:
            List of HybridSearchResult
        """
        return self.hybrid_search(
            query=query.query_text,
            intent=query.intent,
            top_k=query.top_k,
            vector_weight=query.vector_weight,
            graph_weight=query.graph_weight,
            temporal_context=query.temporal_context,
        )

    def _validate_query(
        self,
        query: str,
        intent: str,
        top_k: int,
        vector_weight: float,
        graph_weight: float,
    ) -> None:
        """Validate query parameters."""
        if not query or not query.strip():
            raise InvalidHybridQueryError("Query cannot be empty")

        valid_intents = {"temporal", "causal", "lookup", "exploratory"}
        if intent.lower() not in valid_intents:
            raise InvalidHybridQueryError(
                f"Invalid intent '{intent}'. Must be one of: {valid_intents}"
            )

        if top_k < 1 or top_k > 100:
            raise InvalidHybridQueryError("top_k must be between 1 and 100")

        if not 0.0 <= vector_weight <= 1.0:
            raise InvalidHybridQueryError("vector_weight must be between 0.0 and 1.0")

        if not 0.0 <= graph_weight <= 1.0:
            raise InvalidHybridQueryError("graph_weight must be between 0.0 and 1.0")

    def _get_query_embedding(
        self,
        query: str,
        intent: str,
        temporal_context: Optional[TemporalQueryContext],
    ) -> List[float]:
        """Generate query embedding using router or fallback.

        Args:
            query: Natural language query
            intent: Search intent
            temporal_context: Optional temporal context

        Returns:
            Query embedding vector
        """
        if self._router is not None:
            embedding, _ = self._router.embed_query_with_type_detection(
                query=query,
                intent=intent,
                temporal_context=temporal_context,
            )
            return embedding

        # Fallback: use embedding store's encode method
        return self._embed_query_fallback(query)

    def _embed_query_fallback(self, query: str) -> List[float]:
        """Fallback embedding when router not available.

        Uses embedding store's encode_query method if available,
        otherwise raises error.

        Args:
            query: Natural language query

        Returns:
            Query embedding vector
        """
        if hasattr(self._embeddings, "encode_query"):
            return self._embeddings.encode_query(query)

        # Use the wrapped writer's embedding functionality
        if hasattr(self._embeddings, "_writer"):
            writer = self._embeddings._writer
            if hasattr(writer, "_embed_text"):
                return list(writer._embed_text(query))

        raise HybridSearchError(
            "No query embedding method available. "
            "Provide a QueryEmbeddingRouter or embedding store with encode_query method."
        )

    def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        current_version: int,
        intent: str,
    ) -> List[VectorSearchResult]:
        """Execute vector similarity search with schema filtering.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum results to retrieve
            current_version: Current schema version
            intent: Search intent

        Returns:
            List of VectorSearchResult
        """
        try:
            # Calculate minimum compatible schema version
            min_version = self._schema_compat.get_minimum_compatible_version(
                current_version
            )

            # Query embedding store
            raw_results = self._embeddings.query_embeddings(
                query_vector=query_embedding,
                top_k=top_k,
                min_schema_version=min_version,
                exclude_needs_reembedding=True,
            )

            # Convert to VectorSearchResult
            vector_results = [
                VectorSearchResult(
                    entity_id=r.entity_id,
                    entity_type=r.entity_type or "Unknown",
                    content=r.document or "",
                    similarity_score=r.similarity_score,
                    schema_version=r.metadata.get("schema_version", current_version),
                    metadata=r.metadata,
                )
                for r in raw_results
            ]

            # Apply schema compatibility filtering and score adjustment
            filtered_results = self._schema_compat.filter_compatible_results(
                results=vector_results,
                current_version=current_version,
                apply_score_adjustment=True,
            )

            logger.debug(
                f"Vector search: {len(raw_results)} raw -> "
                f"{len(filtered_results)} compatible results"
            )

            return filtered_results

        except Exception as e:
            raise VectorSearchError(f"Vector search failed: {e}") from e

    def _graph_expansion(
        self,
        seed_entities: List[str],
        intent: str,
    ) -> List[GraphSearchResult]:
        """Expand from seed entities via graph traversal based on intent.

        Dispatches to appropriate expansion strategy based on intent.

        Args:
            seed_entities: Entity IDs to expand from
            intent: Search intent (determines expansion strategy)

        Returns:
            List of GraphSearchResult from expansion
        """
        if not seed_entities:
            return []

        try:
            if intent == "causal" and self._causal is not None:
                return self._causal_expansion(seed_entities)
            elif intent == "temporal" and self._temporal is not None:
                return self._temporal_expansion(seed_entities)
            else:
                return self._neighborhood_expansion(seed_entities)
        except Exception as e:
            raise GraphExpansionError(f"Graph expansion failed: {e}") from e

    def _causal_expansion(
        self,
        seed_entities: List[str],
    ) -> List[GraphSearchResult]:
        """Expand via causal relationships using CausalChainRetrieval.

        Follows CAUSES, ENABLES, TRIGGERS relationships.

        Args:
            seed_entities: Entity IDs to expand from

        Returns:
            List of GraphSearchResult from causal expansion
        """
        if self._causal is None:
            logger.debug("No causal retrieval available, falling back to neighborhood")
            return self._neighborhood_expansion(seed_entities)

        results = []
        for seed_id in seed_entities:
            try:
                # Find effects (forward causal expansion)
                effects_result = self._causal.find_effects(
                    event_id=seed_id,
                    max_hops=2,
                )

                for effect in effects_result.effects:
                    results.append(
                        GraphSearchResult(
                            entity_id=effect.effect_id,
                            entity_type="Event",
                            path_from_seed=[seed_id, effect.effect_id],
                            path_score=effect.aggregate_confidence,
                            relationship_types=["CAUSES"],
                        )
                    )

                # Optionally find causes (backward causal expansion)
                causes_result = self._causal.find_causes(
                    event_id=seed_id,
                    max_hops=2,
                )

                for cause in causes_result.causes:
                    results.append(
                        GraphSearchResult(
                            entity_id=cause.cause_id,
                            entity_type="Event",
                            path_from_seed=[cause.cause_id, seed_id],
                            path_score=cause.aggregate_confidence,
                            relationship_types=["CAUSES"],
                        )
                    )

            except Exception as e:
                # Entity might not be an event or not found
                logger.debug(f"Causal expansion failed for {seed_id}: {e}")
                continue

        # Limit results
        return results[: self._config.graph_expansion_limit]

    def _temporal_expansion(
        self,
        seed_entities: List[str],
    ) -> List[GraphSearchResult]:
        """Expand via temporal relationships using TemporalQueryEngine.

        Finds entities in temporal neighborhood.

        Args:
            seed_entities: Entity IDs to expand from

        Returns:
            List of GraphSearchResult from temporal expansion
        """
        if self._temporal is None:
            logger.debug("No temporal engine available, falling back to neighborhood")
            return self._neighborhood_expansion(seed_entities)

        results = []
        for seed_id in seed_entities:
            try:
                # Query temporal neighborhood
                neighborhood = self._temporal.query_temporal_neighborhood(
                    entity_id=seed_id,
                    time_window=timedelta(days=30),
                )

                # Add event neighbors
                for event in neighborhood.event_neighbors:
                    path_length = 2  # seed -> relationship -> event
                    results.append(
                        GraphSearchResult(
                            entity_id=event.id,
                            entity_type="Event",
                            path_from_seed=[seed_id, event.id],
                            path_score=1.0 / path_length,
                            relationship_types=["TEMPORAL"],
                        )
                    )

                # Add entity neighbors
                for entity in neighborhood.entity_neighbors:
                    path_length = 2
                    results.append(
                        GraphSearchResult(
                            entity_id=entity.id,
                            entity_type=entity.type if hasattr(entity, "type") else "Entity",
                            path_from_seed=[seed_id, entity.id],
                            path_score=1.0 / path_length,
                            relationship_types=["TEMPORAL"],
                        )
                    )

            except Exception as e:
                logger.debug(f"Temporal expansion failed for {seed_id}: {e}")
                continue

        return results[: self._config.graph_expansion_limit]

    def _neighborhood_expansion(
        self,
        seed_entities: List[str],
    ) -> List[GraphSearchResult]:
        """General N-hop neighborhood expansion via PKG queries.

        Fallback when temporal/causal engines not available.

        Args:
            seed_entities: Entity IDs to expand from

        Returns:
            List of GraphSearchResult from neighborhood expansion
        """
        results = []

        for seed_id in seed_entities:
            try:
                # Use PKG queries for 1-2 hop neighborhood
                neighbors = self._pkg.query_temporal_neighborhood(
                    entity_id=seed_id,
                    time_window=timedelta(days=365),  # Wider window for general search
                    include_events=True,
                    include_entities=True,
                )

                for event in neighbors.event_neighbors:
                    results.append(
                        GraphSearchResult(
                            entity_id=event.id,
                            entity_type="Event",
                            path_from_seed=[seed_id, event.id],
                            path_score=0.5,  # Default score for neighborhood
                            relationship_types=["RELATED_TO"],
                        )
                    )

                for entity in neighbors.entity_neighbors:
                    results.append(
                        GraphSearchResult(
                            entity_id=entity.id,
                            entity_type=getattr(entity, "type", "Entity"),
                            path_from_seed=[seed_id, entity.id],
                            path_score=0.5,
                            relationship_types=["RELATED_TO"],
                        )
                    )

            except Exception as e:
                logger.debug(f"Neighborhood expansion failed for {seed_id}: {e}")
                continue

        return results[: self._config.graph_expansion_limit]

    def _adjust_weights(
        self,
        intent: str,
        vector_weight: float,
        graph_weight: float,
        vector_result_count: int,
        graph_result_count: int,
    ) -> Dict[str, float]:
        """Adaptively adjust fusion weights based on intent and result counts.

        Intent-based adjustments:
        - temporal/causal: Favor graph (relationships matter more)
        - lookup: Favor vector (semantic similarity matters)
        - exploratory: Balanced

        Args:
            intent: Search intent
            vector_weight: Base vector weight
            graph_weight: Base graph weight
            vector_result_count: Number of vector results
            graph_result_count: Number of graph results

        Returns:
            Dict with "vector" and "graph" adjusted weights
        """
        # Intent-based base adjustment
        intent_adjustments = {
            "temporal": {"vector": -0.1, "graph": +0.1},
            "causal": {"vector": -0.15, "graph": +0.15},
            "lookup": {"vector": +0.1, "graph": -0.1},
            "exploratory": {"vector": 0, "graph": 0},
        }

        adjustment = intent_adjustments.get(
            intent.lower(), {"vector": 0, "graph": 0}
        )

        # Apply intent adjustment
        adjusted_vector = vector_weight + adjustment["vector"]
        adjusted_graph = graph_weight + adjustment["graph"]

        # Result-based adjustment (if one source has few results)
        if vector_result_count < 5 and graph_result_count > 10:
            adjusted_graph += 0.1
            adjusted_vector -= 0.1
        elif graph_result_count < 5 and vector_result_count > 10:
            adjusted_vector += 0.1
            adjusted_graph -= 0.1

        # Clamp to valid range
        adjusted_vector = max(0.0, min(1.0, adjusted_vector))
        adjusted_graph = max(0.0, min(1.0, adjusted_graph))

        # Normalize to sum to 1
        total = adjusted_vector + adjusted_graph
        if total > 0:
            adjusted_vector /= total
            adjusted_graph /= total
        else:
            adjusted_vector = 0.5
            adjusted_graph = 0.5

        logger.debug(
            f"Adjusted weights for intent={intent}: "
            f"vector={adjusted_vector:.2f}, graph={adjusted_graph:.2f}"
        )

        return {"vector": adjusted_vector, "graph": adjusted_graph}

    def _log_search(
        self,
        query: str,
        intent: str,
        elapsed_ms: float,
        result_count: int,
        vector_count: int,
        graph_count: int,
    ) -> None:
        """Log search performance and audit if enabled."""
        # Performance warning if exceeds target
        if elapsed_ms > self._config.target_latency_ms:
            logger.warning(
                f"Hybrid search exceeded latency target: "
                f"{elapsed_ms:.0f}ms > {self._config.target_latency_ms:.0f}ms"
            )
        else:
            logger.debug(
                f"Hybrid search completed: {elapsed_ms:.0f}ms, "
                f"{result_count} results (vector={vector_count}, graph={graph_count})"
            )

        # Audit logging (without query content for privacy)
        if self._audit is not None:
            try:
                self._audit.log_search_query(
                    query_type="hybrid_search",
                    intent=intent,
                    result_count=result_count,
                    latency_ms=elapsed_ms,
                )
            except Exception as e:
                logger.debug(f"Audit logging failed: {e}")

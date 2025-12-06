"""Schema-Aware Hybrid Retrieval Module.

Provides hybrid search combining vector similarity and graph traversal
with schema version awareness and multi-model embedding integration.

This is Module 03 of the Hybrid Search API, implementing:
- Multi-model query embedding routing
- Schema version compatibility handling
- Vector + graph fusion strategies
- Entity type-specific retrieval strategies
- Adaptive weight adjustment

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Temporal-first design (temporal context in queries)
- Schema evolves autonomously
- Uses frozen models (via embedding service)
- Local-first processing
- Deterministic results for reproducibility

Example:
    >>> from futurnal.pkg.queries.temporal import TemporalGraphQueries
    >>> from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
    >>> from futurnal.search.temporal import TemporalQueryEngine
    >>> from futurnal.search.causal import CausalChainRetrieval
    >>> from futurnal.search.hybrid import (
    ...     SchemaAwareRetrieval,
    ...     QueryEmbeddingRouter,
    ...     EntityTypeRetrievalStrategy,
    ...     HybridSearchConfig,
    ... )

    >>> # Initialize retrieval engine
    >>> retrieval = SchemaAwareRetrieval(
    ...     pkg_queries=pkg_queries,
    ...     embedding_store=embedding_store,
    ...     temporal_engine=temporal_engine,
    ...     causal_retrieval=causal_retrieval,
    ...     embedding_router=query_router,
    ... )

    >>> # Execute hybrid search
    >>> results = retrieval.hybrid_search(
    ...     query="What happened in the project?",
    ...     intent="temporal",
    ...     top_k=10,
    ... )

    >>> # Entity type-specific search
    >>> strategy = EntityTypeRetrievalStrategy(retrieval)
    >>> events = strategy.search_events("project kickoff", top_k=5)
"""

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import (
    FusionError,
    GraphExpansionError,
    HybridSearchError,
    InvalidHybridQueryError,
    QueryRoutingError,
    SchemaCompatibilityError,
    VectorSearchError,
)
from futurnal.search.hybrid.fusion import ResultFusion
from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
from futurnal.search.hybrid.schema_compat import SchemaVersionCompatibility
from futurnal.search.hybrid.strategies import EntityTypeRetrievalStrategy
from futurnal.search.hybrid.types import (
    GraphSearchResult,
    HybridSearchQuery,
    HybridSearchResult,
    QueryEmbeddingType,
    SchemaCompatibilityResult,
    TemporalQueryContext,
    VectorSearchResult,
)

__all__ = [
    # Core classes
    "SchemaAwareRetrieval",
    "QueryEmbeddingRouter",
    "EntityTypeRetrievalStrategy",
    "SchemaVersionCompatibility",
    "ResultFusion",
    # Configuration
    "HybridSearchConfig",
    # Types
    "QueryEmbeddingType",
    "HybridSearchQuery",
    "VectorSearchResult",
    "GraphSearchResult",
    "HybridSearchResult",
    "TemporalQueryContext",
    "SchemaCompatibilityResult",
    # Exceptions
    "HybridSearchError",
    "QueryRoutingError",
    "SchemaCompatibilityError",
    "VectorSearchError",
    "GraphExpansionError",
    "FusionError",
    "InvalidHybridQueryError",
]

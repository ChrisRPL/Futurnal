"""Schema-Aware Hybrid Retrieval Module.

Provides hybrid search combining vector similarity and graph traversal
with schema version awareness and multi-model embedding integration.

Modules:
- Module 03: Schema-Aware Hybrid Retrieval
- Module 04: Query Routing & Orchestration (routing subpackage)

Module 03 implements:
- Multi-model query embedding routing
- Schema version compatibility handling
- Vector + graph fusion strategies
- Entity type-specific retrieval strategies
- Adaptive weight adjustment

Module 04 implements:
- LLM-based intent classification (Ollama/HuggingFace)
- Dynamic model selection by query characteristics
- Multi-strategy orchestration
- GRPO experiential learning integration
- Query understanding templates

Production Plan References:
- docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md
- docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Option B Compliance:
- Temporal-first design (temporal context in queries)
- Schema evolves autonomously
- Ghost model FROZEN (classification only, no fine-tuning)
- Experiential learning via token priors
- Local-first processing
- Deterministic results for reproducibility

Example:
    >>> from futurnal.search.hybrid import (
    ...     SchemaAwareRetrieval,
    ...     QueryRouter,
    ...     get_intent_classifier,
    ...     SearchQualityFeedback,
    ... )

    >>> # Initialize with Module 04 routing
    >>> router = QueryRouter(
    ...     temporal_engine=temporal_engine,
    ...     causal_retrieval=causal_retrieval,
    ...     schema_retrieval=schema_retrieval,
    ... )

    >>> # Route and execute query
    >>> result = router.route_and_execute("What happened in January 2024?")
    >>> print(result.entities)
"""

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import (
    FusionError,
    GraphExpansionError,
    GRPOFeedbackError,
    HybridSearchError,
    IntentClassificationError,
    InvalidHybridQueryError,
    ModelSelectionError,
    QueryRoutingError,
    SchemaCompatibilityError,
    StrategyExecutionError,
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
    QueryIntent,
    QueryPlan,
    QueryResult,
    SchemaCompatibilityResult,
    TemporalQueryContext,
    VectorSearchResult,
)

# Module 04: Query Routing & Orchestration
from futurnal.search.hybrid.routing import (
    DynamicModelRouter,
    HuggingFaceIntentClassifier,
    IntentClassifierLLM,
    LLMBackendType,
    OllamaIntentClassifier,
    QueryRouter,
    QueryRouterLLMConfig,
    QueryTemplate,
    QueryTemplateDatabase,
    SearchQualityFeedback,
    SearchQualitySignal,
    get_intent_classifier,
)

__all__ = [
    # Module 03: Core classes
    "SchemaAwareRetrieval",
    "QueryEmbeddingRouter",
    "EntityTypeRetrievalStrategy",
    "SchemaVersionCompatibility",
    "ResultFusion",
    # Module 04: Query Routing
    "QueryRouter",
    "DynamicModelRouter",
    "OllamaIntentClassifier",
    "HuggingFaceIntentClassifier",
    "IntentClassifierLLM",
    "get_intent_classifier",
    "SearchQualityFeedback",
    "SearchQualitySignal",
    "QueryTemplate",
    "QueryTemplateDatabase",
    # Configuration
    "HybridSearchConfig",
    "LLMBackendType",
    "QueryRouterLLMConfig",
    # Types (Module 03)
    "QueryEmbeddingType",
    "HybridSearchQuery",
    "VectorSearchResult",
    "GraphSearchResult",
    "HybridSearchResult",
    "TemporalQueryContext",
    "SchemaCompatibilityResult",
    # Types (Module 04)
    "QueryIntent",
    "QueryPlan",
    "QueryResult",
    # Exceptions
    "HybridSearchError",
    "QueryRoutingError",
    "SchemaCompatibilityError",
    "VectorSearchError",
    "GraphExpansionError",
    "FusionError",
    "InvalidHybridQueryError",
    "IntentClassificationError",
    "ModelSelectionError",
    "StrategyExecutionError",
    "GRPOFeedbackError",
]

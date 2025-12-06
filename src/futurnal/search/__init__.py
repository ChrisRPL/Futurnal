"""Hybrid Search API Module.

Provides high-level search capabilities integrating PKG and vector embeddings:
- Temporal query engine with decay scoring
- Causal chain retrieval for cause/effect analysis
- Pattern matching and correlation detection
- Schema-aware hybrid retrieval

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/

Option B Compliance:
- Temporal-first design for Phase 2/3 preparation
- Causal chain retrieval with 100% temporal validation
- Local-first processing on-device
- No model fine-tuning, experiential learning only
"""

from futurnal.search.temporal import (
    TemporalQueryEngine,
    TemporalQueryType,
    TemporalQuery,
    TemporalDecayScorer,
    TemporalPatternMatcher,
    TemporalCorrelationDetector,
    SequenceMatch,
    RecurringPattern,
    TemporalCorrelationResult,
)
from futurnal.search.causal import (
    CausalChainRetrieval,
    CausalQuery,
    CausalQueryType,
    CausalSearchPath,
    FindCausesResult,
    FindEffectsResult,
    CausalPathResult,
    CorrelationPatternResult,
    TemporalOrderingValidator,
)
from futurnal.search.hybrid import (
    SchemaAwareRetrieval,
    QueryEmbeddingRouter,
    EntityTypeRetrievalStrategy,
    SchemaVersionCompatibility,
    ResultFusion,
    HybridSearchConfig,
    QueryEmbeddingType,
    HybridSearchQuery,
    VectorSearchResult,
    GraphSearchResult,
    HybridSearchResult,
    TemporalQueryContext,
    SchemaCompatibilityResult,
    HybridSearchError,
    QueryRoutingError,
    SchemaCompatibilityError,
)
from futurnal.search.config import (
    SearchConfig,
    TemporalSearchConfig,
    CausalSearchConfig,
)

__all__ = [
    # Temporal Engine (Module 01)
    "TemporalQueryEngine",
    "TemporalQueryType",
    "TemporalQuery",
    "TemporalDecayScorer",
    "TemporalPatternMatcher",
    "TemporalCorrelationDetector",
    "SequenceMatch",
    "RecurringPattern",
    "TemporalCorrelationResult",
    # Causal Retrieval (Module 02)
    "CausalChainRetrieval",
    "CausalQuery",
    "CausalQueryType",
    "CausalSearchPath",
    "FindCausesResult",
    "FindEffectsResult",
    "CausalPathResult",
    "CorrelationPatternResult",
    "TemporalOrderingValidator",
    # Schema-Aware Hybrid Retrieval (Module 03)
    "SchemaAwareRetrieval",
    "QueryEmbeddingRouter",
    "EntityTypeRetrievalStrategy",
    "SchemaVersionCompatibility",
    "ResultFusion",
    "HybridSearchConfig",
    "QueryEmbeddingType",
    "HybridSearchQuery",
    "VectorSearchResult",
    "GraphSearchResult",
    "HybridSearchResult",
    "TemporalQueryContext",
    "SchemaCompatibilityResult",
    "HybridSearchError",
    "QueryRoutingError",
    "SchemaCompatibilityError",
    # Config
    "SearchConfig",
    "TemporalSearchConfig",
    "CausalSearchConfig",
]

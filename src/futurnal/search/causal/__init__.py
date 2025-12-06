"""Causal Chain Retrieval Module.

Implements Module 02 of the Hybrid Search API:
- Causal path finding (A -> B -> C)
- Finding causes and effects of events
- Correlation pattern detection for Phase 2
- Bradford Hill criteria support for Phase 3

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- Temporal validation required for ALL paths (100%)
- Causal confidence scoring on relationships
- Phase 2/3 foundation established

Example:
    >>> from futurnal.search.causal import CausalChainRetrieval, CausalQuery, CausalQueryType

    >>> retrieval = CausalChainRetrieval(pkg_queries, temporal_engine)

    >>> # What caused this decision?
    >>> causes = retrieval.find_causes("decision_123", max_hops=3)

    >>> # How did the meeting lead to the publication?
    >>> path = retrieval.find_causal_path("meeting_1", "publication_1")

    >>> # Use unified query interface
    >>> query = CausalQuery(
    ...     query_type=CausalQueryType.FIND_CAUSES,
    ...     event_id="decision_123",
    ... )
    >>> result = retrieval.query(query)
"""

from futurnal.search.causal.types import CausalQuery, CausalQueryType
from futurnal.search.causal.retrieval import CausalChainRetrieval
from futurnal.search.causal.validation import TemporalOrderingValidator
from futurnal.search.causal.results import (
    CausalCauseResult,
    CausalEffectResult,
    CausalPathResult,
    CausalSearchPath,
    CorrelationPatternResult,
    FindCausesResult,
    FindEffectsResult,
)
from futurnal.search.causal.exceptions import (
    CausalChainDepthExceeded,
    CausalPathNotFoundError,
    CausalSearchError,
    CorrelationDetectionError,
    EventNotFoundError,
    InvalidCausalQueryError,
    TemporalOrderingViolation,
)

__all__ = [
    # Engine
    "CausalChainRetrieval",
    # Types
    "CausalQuery",
    "CausalQueryType",
    # Validation
    "TemporalOrderingValidator",
    # Results
    "CausalCauseResult",
    "CausalEffectResult",
    "CausalPathResult",
    "CausalSearchPath",
    "CorrelationPatternResult",
    "FindCausesResult",
    "FindEffectsResult",
    # Exceptions
    "CausalChainDepthExceeded",
    "CausalPathNotFoundError",
    "CausalSearchError",
    "CorrelationDetectionError",
    "EventNotFoundError",
    "InvalidCausalQueryError",
    "TemporalOrderingViolation",
]

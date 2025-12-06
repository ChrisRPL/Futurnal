"""Hybrid Search API Module.

Provides high-level search capabilities integrating PKG and vector embeddings:
- Temporal query engine with decay scoring
- Pattern matching and correlation detection
- Schema-aware hybrid retrieval

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/

Option B Compliance:
- Temporal-first design for Phase 2/3 preparation
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
from futurnal.search.config import SearchConfig, TemporalSearchConfig

__all__ = [
    # Engine
    "TemporalQueryEngine",
    # Types
    "TemporalQueryType",
    "TemporalQuery",
    # Components
    "TemporalDecayScorer",
    "TemporalPatternMatcher",
    "TemporalCorrelationDetector",
    # Results
    "SequenceMatch",
    "RecurringPattern",
    "TemporalCorrelationResult",
    # Config
    "SearchConfig",
    "TemporalSearchConfig",
]

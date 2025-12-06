"""Temporal Query Engine Module.

Implements Module 01 of the Hybrid Search API:
- Time range queries with decay scoring
- Temporal pattern matching
- Correlation detection for Phase 2 preparation

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- Temporal-first: All queries centered on temporal metadata
- Phase 2/3 preparation: Correlation detection enables causal inference
- Uses existing PKG temporal queries as foundation
"""

from futurnal.search.temporal.types import TemporalQueryType, TemporalQuery
from futurnal.search.temporal.decay import TemporalDecayScorer
from futurnal.search.temporal.patterns import TemporalPatternMatcher
from futurnal.search.temporal.correlation import TemporalCorrelationDetector
from futurnal.search.temporal.results import (
    SequenceMatch,
    RecurringPattern,
    TemporalCorrelationResult,
    HybridNeighborhoodResult,
    ScoredEvent,
)
from futurnal.search.temporal.engine import TemporalQueryEngine

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
    "HybridNeighborhoodResult",
    "ScoredEvent",
]

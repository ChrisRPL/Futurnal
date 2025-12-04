"""PKG Temporal Query Support (Module 04).

Implements temporal query capabilities critical for Option B including:
- Time range queries (events within period)
- Temporal relationship traversal (BEFORE/AFTER chains)
- Causal chain queries (A->B->C causation paths)
- Temporal neighborhood (entities/events within time window)

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/04-temporal-query-support.md

Option B Compliance:
- Enables Phase 2 correlation detection
- Enables Phase 3 causal inference
- Temporal-first design enforced
- Causal relationships include Bradford Hill metadata
"""

from futurnal.pkg.queries.models import (
    CausalPath,
    CausalChainResult,
    TemporalNeighborhood,
    TemporalQueryResult,
)
from futurnal.pkg.queries.exceptions import (
    TemporalQueryError,
    InvalidTimeRangeError,
    EventNotFoundError,
    CausalChainDepthError,
)
from futurnal.pkg.queries.temporal import TemporalGraphQueries

__all__ = [
    # Main service
    "TemporalGraphQueries",
    # Result models
    "CausalPath",
    "CausalChainResult",
    "TemporalNeighborhood",
    "TemporalQueryResult",
    # Exceptions
    "TemporalQueryError",
    "InvalidTimeRangeError",
    "EventNotFoundError",
    "CausalChainDepthError",
]

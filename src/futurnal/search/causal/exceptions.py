"""Causal Search Exceptions.

Defines exception hierarchy for causal chain retrieval:
- CausalSearchError: Base exception
- InvalidCausalQueryError: Query parameter validation errors
- CausalPathNotFoundError: No path between events
- TemporalOrderingViolation: Causal path violates temporal ordering
- CausalChainDepthExceeded: Requested depth exceeds maximum
- EventNotFoundError: Event does not exist

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md
"""

from __future__ import annotations

from typing import List, Optional


class CausalSearchError(Exception):
    """Base exception for causal search operations.

    All causal search exceptions inherit from this class.

    Attributes:
        message: Error description
        query_type: Type of query that failed (optional)
        cause: Underlying exception that caused this error (optional)
    """

    def __init__(
        self,
        message: str,
        query_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.query_type = query_type
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation with context."""
        parts = [self.message]
        if self.query_type:
            parts.append(f"(query_type={self.query_type})")
        return " ".join(parts)


class InvalidCausalQueryError(CausalSearchError):
    """Causal query parameters are invalid.

    Raised when query validation fails due to missing or inconsistent parameters.
    """

    pass


class CausalPathNotFoundError(CausalSearchError):
    """No causal path exists between specified events.

    Attributes:
        start_event_id: Start event ID
        end_event_id: End event ID
        max_hops_searched: Maximum hops that were searched
    """

    def __init__(
        self,
        start_event_id: str,
        end_event_id: str,
        max_hops_searched: int,
    ):
        self.start_event_id = start_event_id
        self.end_event_id = end_event_id
        self.max_hops_searched = max_hops_searched
        message = (
            f"No causal path found from '{start_event_id}' to '{end_event_id}' "
            f"within {max_hops_searched} hops"
        )
        super().__init__(message, query_type="causal_path")


class TemporalOrderingViolation(CausalSearchError):
    """Causal path violates temporal ordering (cause after effect).

    Critical for Option B compliance - all paths must be temporally valid.
    Bradford Hill criterion 1 (temporality) requires cause to precede effect.

    Attributes:
        path: Event IDs in the violating path
        violation_index: Index where violation occurred
    """

    def __init__(
        self,
        path: List[str],
        violation_index: int,
    ):
        self.path = path
        self.violation_index = violation_index
        message = (
            f"Temporal ordering violated at index {violation_index} in path: "
            f"{' -> '.join(path)}"
        )
        super().__init__(message, query_type="temporal_validation")


class CausalChainDepthExceeded(CausalSearchError):
    """Requested chain depth exceeds maximum allowed.

    Maximum causal chain depth is capped to prevent expensive graph traversals.

    Attributes:
        requested: Requested depth
        maximum: Maximum allowed depth
    """

    def __init__(self, requested: int, maximum: int):
        self.requested = requested
        self.maximum = maximum
        message = f"Requested depth {requested} exceeds maximum {maximum}"
        super().__init__(message, query_type="causal_chain")


class EventNotFoundError(CausalSearchError):
    """Event with specified ID does not exist.

    Raised when a query references an event that cannot be found in the PKG.

    Attributes:
        event_id: ID of the missing event
    """

    def __init__(self, event_id: str):
        self.event_id = event_id
        message = f"Event not found: {event_id}"
        super().__init__(message, query_type="event_lookup")


class CorrelationDetectionError(CausalSearchError):
    """Correlation detection failed.

    Raised when correlation pattern detection encounters an error,
    typically when the temporal engine is not configured.
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, query_type="correlation_pattern", cause=cause)

"""Temporal Query Exceptions.

Defines exception hierarchy for temporal and causal query operations:
- TemporalQueryError: Base exception for all temporal query errors
- InvalidTimeRangeError: Invalid time range (start > end)
- EventNotFoundError: Referenced event does not exist
- CausalChainDepthError: Requested depth exceeds limits

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/04-temporal-query-support.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


class TemporalQueryError(Exception):
    """Base exception for temporal query operations.

    All temporal query exceptions inherit from this class,
    enabling broad exception catching when needed.

    Attributes:
        message: Human-readable error description
        query_type: Type of query that failed (time_range, causal_chain, etc.)
    """

    def __init__(
        self,
        message: str,
        query_type: Optional[str] = None,
    ):
        self.message = message
        self.query_type = query_type
        super().__init__(message)

    def __str__(self) -> str:
        if self.query_type:
            return f"[{self.query_type}] {self.message}"
        return self.message


class InvalidTimeRangeError(TemporalQueryError):
    """Raised when time range is invalid (start > end).

    Time range queries require start <= end. This exception
    provides clear feedback when the range is inverted.

    Attributes:
        start: Requested start time
        end: Requested end time
    """

    def __init__(
        self,
        start: datetime,
        end: datetime,
        message: Optional[str] = None,
    ):
        self.start = start
        self.end = end
        if message is None:
            message = (
                f"Invalid time range: start ({start.isoformat()}) "
                f"must be before or equal to end ({end.isoformat()})"
            )
        super().__init__(message, query_type="time_range")


class EventNotFoundError(TemporalQueryError):
    """Raised when a referenced event does not exist.

    Causal chain queries require a valid starting event ID.
    This exception indicates the event was not found in the database.

    Attributes:
        event_id: ID of the event that was not found
    """

    def __init__(
        self,
        event_id: str,
        message: Optional[str] = None,
    ):
        self.event_id = event_id
        if message is None:
            message = f"Event not found: {event_id}"
        super().__init__(message, query_type="causal_chain")


class EntityNotFoundError(TemporalQueryError):
    """Raised when a referenced entity does not exist.

    Temporal neighborhood queries require a valid entity ID.
    This exception indicates the entity was not found in the database.

    Attributes:
        entity_id: ID of the entity that was not found
    """

    def __init__(
        self,
        entity_id: str,
        message: Optional[str] = None,
    ):
        self.entity_id = entity_id
        if message is None:
            message = f"Entity not found: {entity_id}"
        super().__init__(message, query_type="temporal_neighborhood")


class CausalChainDepthError(TemporalQueryError):
    """Raised when requested causal chain depth exceeds limits.

    To prevent runaway queries, causal chain depth is limited.
    This exception indicates the requested depth is too large.

    Attributes:
        requested_depth: Depth requested by caller
        max_allowed: Maximum allowed depth
    """

    def __init__(
        self,
        requested_depth: int,
        max_allowed: int,
        message: Optional[str] = None,
    ):
        self.requested_depth = requested_depth
        self.max_allowed = max_allowed
        if message is None:
            message = (
                f"Causal chain depth {requested_depth} exceeds maximum allowed {max_allowed}"
            )
        super().__init__(message, query_type="causal_chain")


class TemporalNeighborhoodError(TemporalQueryError):
    """Raised when temporal neighborhood query fails.

    Generic error for temporal neighborhood query issues
    not covered by more specific exceptions.

    Attributes:
        entity_id: ID of the center entity
        time_window: Time window requested
    """

    def __init__(
        self,
        entity_id: str,
        time_window: timedelta,
        message: Optional[str] = None,
    ):
        self.entity_id = entity_id
        self.time_window = time_window
        if message is None:
            message = (
                f"Temporal neighborhood query failed for entity {entity_id} "
                f"with time window {time_window}"
            )
        super().__init__(message, query_type="temporal_neighborhood")


class QueryTimeoutError(TemporalQueryError):
    """Raised when a query exceeds the timeout threshold.

    Long-running queries are terminated to prevent resource exhaustion.
    This exception indicates the query took too long.

    Attributes:
        timeout_ms: Timeout threshold in milliseconds
        actual_ms: Actual execution time before termination
    """

    def __init__(
        self,
        timeout_ms: float,
        actual_ms: Optional[float] = None,
        message: Optional[str] = None,
    ):
        self.timeout_ms = timeout_ms
        self.actual_ms = actual_ms
        if message is None:
            if actual_ms is not None:
                message = f"Query timed out after {actual_ms:.1f}ms (limit: {timeout_ms:.1f}ms)"
            else:
                message = f"Query exceeded timeout of {timeout_ms:.1f}ms"
        super().__init__(message)

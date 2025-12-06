"""Search-Specific Exceptions.

Defines exception hierarchy for the temporal search module.
Extends PKG query exceptions for consistency.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


class TemporalSearchError(Exception):
    """Base exception for temporal search operations.

    All search-specific exceptions inherit from this class.
    Wraps underlying PKG and embedding exceptions with search context.
    """

    def __init__(
        self,
        message: str,
        query_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error description
            query_type: Type of query that failed (optional)
            cause: Underlying exception (optional)
        """
        super().__init__(message)
        self.query_type = query_type
        self.cause = cause


class InvalidTemporalQueryError(TemporalSearchError):
    """Query parameters are invalid or inconsistent.

    Raised when query validation fails before execution.

    Example:
        >>> raise InvalidTemporalQueryError(
        ...     "TIME_RANGE requires start_time and end_time",
        ...     query_type="time_range",
        ... )
    """

    pass


class TemporalQueryTimeoutError(TemporalSearchError):
    """Query execution exceeded timeout.

    Raised when a query takes longer than configured timeout.

    Attributes:
        timeout_ms: Configured timeout in milliseconds
        elapsed_ms: Actual elapsed time in milliseconds
    """

    def __init__(
        self,
        message: str,
        timeout_ms: float,
        elapsed_ms: float,
        query_type: Optional[str] = None,
    ):
        super().__init__(message, query_type)
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms


class PatternNotFoundError(TemporalSearchError):
    """No matching patterns found for query.

    Raised when pattern matching finds no results.
    Not necessarily an error - may indicate data doesn't contain pattern.

    Attributes:
        pattern: The pattern that was searched for
    """

    def __init__(
        self,
        pattern: list[str],
        time_range: Optional[tuple[datetime, datetime]] = None,
    ):
        self.pattern = pattern
        self.time_range = time_range
        pattern_str = " -> ".join(pattern)
        message = f"No sequences matching pattern '{pattern_str}' found"
        if time_range:
            message += f" in range {time_range[0]} to {time_range[1]}"
        super().__init__(message, query_type="temporal_sequence")


class InsufficientDataError(TemporalSearchError):
    """Not enough data for correlation analysis.

    Raised when correlation detection doesn't have enough
    data points for statistically significant results.

    Attributes:
        event_type_a: First event type in correlation
        event_type_b: Second event type in correlation
        found_count: Number of co-occurrences found
        required_count: Minimum co-occurrences required
    """

    def __init__(
        self,
        event_type_a: str,
        event_type_b: str,
        found_count: int,
        required_count: int,
    ):
        self.event_type_a = event_type_a
        self.event_type_b = event_type_b
        self.found_count = found_count
        self.required_count = required_count
        message = (
            f"Insufficient data for correlation between '{event_type_a}' and "
            f"'{event_type_b}': found {found_count} co-occurrences, "
            f"need at least {required_count}"
        )
        super().__init__(message, query_type="temporal_correlation")


class HybridSearchError(TemporalSearchError):
    """Error in hybrid graph+vector search.

    Raised when combining graph and vector results fails.

    Attributes:
        graph_error: Error from graph query (if any)
        vector_error: Error from vector query (if any)
    """

    def __init__(
        self,
        message: str,
        graph_error: Optional[Exception] = None,
        vector_error: Optional[Exception] = None,
    ):
        super().__init__(message, query_type="hybrid")
        self.graph_error = graph_error
        self.vector_error = vector_error


class DecayScoringError(TemporalSearchError):
    """Error in decay scoring calculation.

    Raised when decay scoring fails (e.g., invalid timestamps).
    """

    def __init__(self, message: str, event_id: Optional[str] = None):
        super().__init__(message, query_type="decay_scoring")
        self.event_id = event_id


class CorrelationAnalysisError(TemporalSearchError):
    """Error during correlation analysis.

    Raised when statistical correlation analysis fails.
    """

    def __init__(
        self,
        message: str,
        event_type_a: Optional[str] = None,
        event_type_b: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, query_type="temporal_correlation", cause=cause)
        self.event_type_a = event_type_a
        self.event_type_b = event_type_b

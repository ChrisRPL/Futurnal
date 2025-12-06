"""Temporal Query Types and Models.

Defines the core types for temporal queries:
- TemporalQueryType: Enum of supported query types
- TemporalQuery: Pydantic model for query specification

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- Temporal-first design with required timestamp parameters
- Supports Phase 2 correlation detection via TEMPORAL_CORRELATION
- Supports Phase 3 causal inference via CAUSAL_CHAIN
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class TemporalQueryType(str, Enum):
    """Types of temporal queries supported by the engine.

    Aligned with production plan specification:
    - Basic temporal: TIME_RANGE, BEFORE, AFTER, DURING
    - Advanced: TEMPORAL_NEIGHBORHOOD, TEMPORAL_SEQUENCE
    - Phase 2/3: TEMPORAL_CORRELATION, CAUSAL_CHAIN
    """

    TIME_RANGE = "time_range"
    """Events within a specific time range."""

    BEFORE = "before"
    """Events before a reference timestamp or event."""

    AFTER = "after"
    """Events after a reference timestamp or event."""

    DURING = "during"
    """Events that occurred during another event's duration."""

    TEMPORAL_NEIGHBORHOOD = "temporal_neighborhood"
    """Events around a reference point within a time window."""

    TEMPORAL_SEQUENCE = "temporal_sequence"
    """Event sequences matching a pattern (e.g., Meeting -> Decision)."""

    TEMPORAL_CORRELATION = "temporal_correlation"
    """Statistical correlation between event types (Phase 2 foundation)."""

    CAUSAL_CHAIN = "causal_chain"
    """Causal relationship traversal (Phase 3 foundation)."""


class TemporalQuery(BaseModel):
    """Temporal query specification.

    Unified query model that encapsulates all temporal query parameters.
    The query_type determines which parameters are required.

    Example:
        >>> # Time range query
        >>> query = TemporalQuery(
        ...     query_type=TemporalQueryType.TIME_RANGE,
        ...     start_time=datetime(2024, 1, 1),
        ...     end_time=datetime(2024, 3, 31),
        ...     event_types=["meeting"],
        ... )

        >>> # Temporal correlation query
        >>> query = TemporalQuery(
        ...     query_type=TemporalQueryType.TEMPORAL_CORRELATION,
        ...     event_type_a="Meeting",
        ...     event_type_b="Decision",
        ...     max_gap=timedelta(days=7),
        ... )
    """

    query_type: TemporalQueryType = Field(
        ...,
        description="Type of temporal query to execute"
    )

    # Time range parameters
    start_time: Optional[datetime] = Field(
        None,
        description="Start of time range (inclusive)"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End of time range (inclusive)"
    )

    # Reference parameters
    reference_timestamp: Optional[datetime] = Field(
        None,
        description="Reference timestamp for BEFORE/AFTER/NEIGHBORHOOD queries"
    )
    reference_event_id: Optional[str] = Field(
        None,
        description="Reference event ID for relative queries"
    )

    # Filtering parameters
    time_window: Optional[timedelta] = Field(
        None,
        description="Time window for neighborhood or before/after queries"
    )
    event_types: Optional[List[str]] = Field(
        None,
        description="Filter by event types (meeting, decision, etc.)"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results"
    )

    # Pattern matching parameters (TEMPORAL_SEQUENCE)
    pattern: Optional[List[str]] = Field(
        None,
        description="Event type pattern for sequence matching (e.g., ['Meeting', 'Decision'])"
    )
    max_gap: Optional[timedelta] = Field(
        None,
        description="Maximum gap between events in sequence or correlation"
    )

    # Correlation parameters (TEMPORAL_CORRELATION)
    event_type_a: Optional[str] = Field(
        None,
        description="First event type for correlation analysis"
    )
    event_type_b: Optional[str] = Field(
        None,
        description="Second event type for correlation analysis"
    )
    min_occurrences: int = Field(
        default=3,
        ge=1,
        description="Minimum co-occurrences for correlation significance"
    )

    # Causal chain parameters (CAUSAL_CHAIN)
    max_hops: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum hops for causal chain traversal"
    )

    # Decay scoring parameters
    enable_decay_scoring: bool = Field(
        default=True,
        description="Apply temporal decay scoring to results"
    )
    decay_half_life_days: float = Field(
        default=30.0,
        gt=0,
        description="Half-life in days for decay scoring"
    )

    # Pagination parameters
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum results to return"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset for pagination"
    )

    # Hybrid search parameters
    query_text: Optional[str] = Field(
        None,
        description="Text query for hybrid vector+graph search"
    )
    vector_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid results"
    )

    @model_validator(mode="after")
    def validate_query_parameters(self) -> "TemporalQuery":
        """Validate that required parameters are provided for query type."""
        qt = self.query_type

        if qt == TemporalQueryType.TIME_RANGE:
            if self.start_time is None or self.end_time is None:
                raise ValueError(
                    "TIME_RANGE query requires start_time and end_time"
                )
            if self.start_time > self.end_time:
                raise ValueError(
                    "start_time must be before or equal to end_time"
                )

        elif qt in (TemporalQueryType.BEFORE, TemporalQueryType.AFTER):
            if self.reference_timestamp is None and self.reference_event_id is None:
                raise ValueError(
                    f"{qt.value} query requires reference_timestamp or reference_event_id"
                )

        elif qt == TemporalQueryType.DURING:
            if self.reference_event_id is None:
                raise ValueError(
                    "DURING query requires reference_event_id"
                )

        elif qt == TemporalQueryType.TEMPORAL_NEIGHBORHOOD:
            if self.reference_event_id is None and self.reference_timestamp is None:
                raise ValueError(
                    "TEMPORAL_NEIGHBORHOOD requires reference_event_id or reference_timestamp"
                )
            if self.time_window is None:
                # Default to 7 days if not specified
                self.time_window = timedelta(days=7)

        elif qt == TemporalQueryType.TEMPORAL_SEQUENCE:
            if not self.pattern or len(self.pattern) < 2:
                raise ValueError(
                    "TEMPORAL_SEQUENCE requires pattern with at least 2 event types"
                )
            if self.start_time is None or self.end_time is None:
                raise ValueError(
                    "TEMPORAL_SEQUENCE requires start_time and end_time"
                )
            if self.max_gap is None:
                self.max_gap = timedelta(days=30)

        elif qt == TemporalQueryType.TEMPORAL_CORRELATION:
            if not self.event_type_a or not self.event_type_b:
                raise ValueError(
                    "TEMPORAL_CORRELATION requires event_type_a and event_type_b"
                )
            if self.max_gap is None:
                self.max_gap = timedelta(days=30)

        elif qt == TemporalQueryType.CAUSAL_CHAIN:
            if self.reference_event_id is None:
                raise ValueError(
                    "CAUSAL_CHAIN requires reference_event_id"
                )

        return self

    class Config:
        """Pydantic configuration."""
        frozen = False  # Allow mutation for default value assignment in validator
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }

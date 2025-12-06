"""Causal Query Types and Models.

Defines the core types for causal chain queries:
- CausalQueryType: Enum of supported causal query types
- CausalQuery: Pydantic model for query specification

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- Causal-first design with required temporal validation
- Supports Phase 2 correlation detection via CORRELATION_PATTERN
- Supports Phase 3 causal inference via FIND_CAUSES/FIND_EFFECTS
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CausalQueryType(str, Enum):
    """Types of causal queries supported by the retrieval engine.

    Aligned with production plan specification:
    - FIND_CAUSES: What caused this event?
    - FIND_EFFECTS: What did this event cause?
    - CAUSAL_PATH: Path from A to B
    - CAUSAL_CHAIN: Full causal chain from start
    - CORRELATION_PATTERN: Detect correlations in time range
    """

    FIND_CAUSES = "find_causes"
    """Find events that caused the target event."""

    FIND_EFFECTS = "find_effects"
    """Find events caused by the target event."""

    CAUSAL_PATH = "causal_path"
    """Find specific causal path from start to end event."""

    CAUSAL_CHAIN = "causal_chain"
    """Find all causal chains from a start event."""

    CORRELATION_PATTERN = "correlation_pattern"
    """Detect correlation patterns in time range."""


class CausalQuery(BaseModel):
    """Causal query specification.

    Unified query model for all causal query types.
    query_type determines which parameters are required.

    Example:
        >>> query = CausalQuery(
        ...     query_type=CausalQueryType.FIND_CAUSES,
        ...     event_id="decision_123",
        ...     max_hops=3,
        ...     min_confidence=0.6,
        ... )
    """

    query_type: CausalQueryType = Field(
        ...,
        description="Type of causal query to execute",
    )

    # Event reference parameters
    event_id: Optional[str] = Field(
        default=None,
        description="Target event ID for FIND_CAUSES/FIND_EFFECTS/CAUSAL_CHAIN",
    )
    start_event_id: Optional[str] = Field(
        default=None,
        description="Start event ID for CAUSAL_PATH",
    )
    end_event_id: Optional[str] = Field(
        default=None,
        description="End event ID for CAUSAL_PATH",
    )

    # Traversal parameters
    max_hops: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum hops for causal traversal (1-10)",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum causal confidence threshold",
    )

    # Time range for correlation detection
    time_range_start: Optional[datetime] = Field(
        default=None,
        description="Start of time range for CORRELATION_PATTERN",
    )
    time_range_end: Optional[datetime] = Field(
        default=None,
        description="End of time range for CORRELATION_PATTERN",
    )
    min_correlation_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum correlation strength for pattern detection",
    )

    # Pagination
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results to return",
    )

    @model_validator(mode="after")
    def validate_query_parameters(self) -> "CausalQuery":
        """Validate required parameters for query type."""
        qt = self.query_type

        if qt in (
            CausalQueryType.FIND_CAUSES,
            CausalQueryType.FIND_EFFECTS,
            CausalQueryType.CAUSAL_CHAIN,
        ):
            if self.event_id is None:
                raise ValueError(f"{qt.value} query requires event_id")

        elif qt == CausalQueryType.CAUSAL_PATH:
            if self.start_event_id is None or self.end_event_id is None:
                raise ValueError(
                    "CAUSAL_PATH requires start_event_id and end_event_id"
                )

        elif qt == CausalQueryType.CORRELATION_PATTERN:
            if self.time_range_start is None or self.time_range_end is None:
                raise ValueError(
                    "CORRELATION_PATTERN requires time_range_start and time_range_end"
                )
            if self.time_range_start > self.time_range_end:
                raise ValueError("time_range_start must be before time_range_end")

        return self

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }

"""Causal Search Result Models.

Extends PKG causal models with search-specific fields:
- CausalSearchPath: Extended CausalPath with search metadata
- FindCausesResult: Result for FIND_CAUSES queries
- FindEffectsResult: Result for FIND_EFFECTS queries
- CausalPathResult: Result for CAUSAL_PATH queries
- CorrelationPatternResult: Result for CORRELATION_PATTERN queries

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- Temporal ordering validation for ALL paths (100%)
- Causal confidence scoring on relationships
- Bradford Hill criteria support (temporality validated)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from futurnal.pkg.schema.models import EventNode


class CausalSearchPath(BaseModel):
    """Causal path with search-specific metadata.

    Extends PKG CausalPath concept with:
    - Explicit start/end event IDs for API consistency
    - path_length computed field
    - temporal_ordering_valid field (required for Option B)

    Example:
        >>> path = CausalSearchPath(
        ...     start_event_id="meeting_1",
        ...     end_event_id="publication_1",
        ...     path=["meeting_1", "decision_1", "publication_1"],
        ...     causal_confidence=0.72,
        ...     temporal_ordering_valid=True,
        ... )
    """

    start_event_id: str = Field(..., description="ID of the starting event")
    end_event_id: str = Field(..., description="ID of the ending event")
    path: List[str] = Field(
        ...,
        min_length=2,
        description="Event IDs in causal order [cause, ..., effect]",
    )
    events: Optional[List[EventNode]] = Field(
        default=None,
        description="Full event objects (if fetched)",
    )
    causal_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate causal confidence (min of all relationships)",
    )
    confidence_scores: List[float] = Field(
        default_factory=list,
        description="Individual confidence scores for each causal link",
    )
    temporal_ordering_valid: bool = Field(
        ...,
        description="True if cause precedes effect for ALL links (Option B required)",
    )
    causal_evidence: List[str] = Field(
        default_factory=list,
        description="Text evidence for each causal link",
    )

    @computed_field
    @property
    def path_length(self) -> int:
        """Number of causal hops in this path."""
        return len(self.path) - 1 if self.path else 0

    @computed_field
    @property
    def temporal_span(self) -> Optional[timedelta]:
        """Time between first and last event."""
        if self.events and len(self.events) >= 2:
            return self.events[-1].timestamp - self.events[0].timestamp
        return None


class CausalCauseResult(BaseModel):
    """Single cause result from find_causes query.

    Represents an event that caused the target event, with distance
    and confidence information.
    """

    cause_id: str = Field(..., description="ID of the cause event")
    cause_name: str = Field(..., description="Name of the cause event")
    cause_timestamp: datetime = Field(..., description="Timestamp of cause event")
    cause_event: Optional[EventNode] = Field(
        default=None,
        description="Full event object",
    )
    distance: int = Field(..., ge=1, description="Causal distance (hops)")
    confidence_scores: List[float] = Field(
        default_factory=list,
        description="Confidence scores along the path",
    )
    aggregate_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Min confidence along the path",
    )
    temporal_ordering_valid: bool = Field(
        ...,
        description="True if path respects temporal ordering",
    )


class FindCausesResult(BaseModel):
    """Result for FIND_CAUSES query.

    Contains all events that caused the target event, ordered by distance.

    Example:
        >>> result = retrieval.find_causes("decision_123", max_hops=3)
        >>> print(f"Found {result.total_causes} causes")
        >>> for cause in result.causes:
        ...     print(f"{cause.cause_name} at distance {cause.distance}")
    """

    target_event_id: str = Field(..., description="ID of the target event")
    target_event: Optional[EventNode] = Field(
        default=None,
        description="Target event object",
    )
    causes: List[CausalCauseResult] = Field(
        default_factory=list,
        description="List of cause events",
    )
    max_hops_requested: int = Field(..., description="Max hops requested")
    min_confidence_requested: float = Field(
        ...,
        description="Min confidence requested",
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query time in ms",
    )

    @computed_field
    @property
    def total_causes(self) -> int:
        """Total number of causes found."""
        return len(self.causes)

    @computed_field
    @property
    def unique_root_causes(self) -> List[str]:
        """Unique root cause IDs (events at max distance)."""
        if not self.causes:
            return []
        max_dist = max(c.distance for c in self.causes)
        return list(set(c.cause_id for c in self.causes if c.distance == max_dist))


class CausalEffectResult(BaseModel):
    """Single effect result from find_effects query.

    Represents an event that was caused by the source event, with distance
    and confidence information.
    """

    effect_id: str = Field(..., description="ID of the effect event")
    effect_name: str = Field(..., description="Name of the effect event")
    effect_timestamp: datetime = Field(..., description="Timestamp of effect event")
    effect_event: Optional[EventNode] = Field(
        default=None,
        description="Full event object",
    )
    distance: int = Field(..., ge=1, description="Causal distance (hops)")
    confidence_scores: List[float] = Field(
        default_factory=list,
        description="Confidence scores along the path",
    )
    aggregate_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Min confidence along the path",
    )
    temporal_ordering_valid: bool = Field(
        ...,
        description="True if path respects temporal ordering",
    )


class FindEffectsResult(BaseModel):
    """Result for FIND_EFFECTS query.

    Contains all events caused by the source event, ordered by distance.

    Example:
        >>> result = retrieval.find_effects("meeting_123", max_hops=3)
        >>> print(f"Found {result.total_effects} effects")
        >>> for effect in result.effects:
        ...     print(f"{effect.effect_name} at distance {effect.distance}")
    """

    source_event_id: str = Field(..., description="ID of the source event")
    source_event: Optional[EventNode] = Field(
        default=None,
        description="Source event object",
    )
    effects: List[CausalEffectResult] = Field(
        default_factory=list,
        description="List of effect events",
    )
    max_hops_requested: int = Field(..., description="Max hops requested")
    min_confidence_requested: float = Field(
        ...,
        description="Min confidence requested",
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query time in ms",
    )

    @computed_field
    @property
    def total_effects(self) -> int:
        """Total number of effects found."""
        return len(self.effects)


class CausalPathResult(BaseModel):
    """Result for CAUSAL_PATH query.

    Contains the causal path between two events (if found).

    Example:
        >>> result = retrieval.find_causal_path("meeting_1", "publication_1")
        >>> if result.path_found:
        ...     print(f"Path: {' -> '.join(result.path.path)}")
        ...     print(f"Confidence: {result.path.causal_confidence:.2f}")
    """

    path_found: bool = Field(..., description="Whether a path was found")
    path: Optional[CausalSearchPath] = Field(
        default=None,
        description="The causal path (if found)",
    )
    start_event_id: str = Field(..., description="Start event ID")
    end_event_id: str = Field(..., description="End event ID")
    max_hops_requested: int = Field(..., description="Max hops requested")
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query time in ms",
    )


class CorrelationPatternResult(BaseModel):
    """Result for CORRELATION_PATTERN query.

    Contains correlation patterns detected in the specified time range,
    with causal candidates flagged for Phase 3 validation.

    Example:
        >>> result = retrieval.detect_correlation_patterns(
        ...     time_range_start=datetime(2024, 1, 1),
        ...     time_range_end=datetime(2024, 6, 30),
        ... )
        >>> print(f"Found {result.patterns_found} patterns")
        >>> print(f"  {result.causal_candidate_count} are causal candidates")
    """

    patterns_found: int = Field(..., ge=0, description="Number of patterns found")
    correlations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of correlation results from TemporalCorrelationDetector",
    )
    causal_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Correlations flagged as causal candidates for Phase 3",
    )
    time_range_start: datetime = Field(..., description="Start of analyzed range")
    time_range_end: datetime = Field(..., description="End of analyzed range")
    min_correlation_strength: float = Field(
        ...,
        description="Min strength threshold",
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query time in ms",
    )

    @computed_field
    @property
    def causal_candidate_count(self) -> int:
        """Number of causal candidates found."""
        return len(self.causal_candidates)

    @computed_field
    @property
    def time_span(self) -> timedelta:
        """Duration of the analyzed time range."""
        return self.time_range_end - self.time_range_start

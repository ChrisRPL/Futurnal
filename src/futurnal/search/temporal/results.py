"""Temporal Search Result Models.

Defines Pydantic models for temporal query results:
- ScoredEvent: Event with decay score
- SequenceMatch: Matched event sequence
- RecurringPattern: Discovered recurring pattern
- TemporalCorrelationResult: Correlation analysis result
- HybridNeighborhoodResult: Combined graph+vector results

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- TemporalCorrelationResult includes is_causal_candidate for Phase 2/3
- All results include confidence/strength metrics for quality evolution
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field

from futurnal.pkg.schema.models import BaseNode, EventNode


class ScoredEvent(BaseModel):
    """Event with temporal decay score.

    Wraps an EventNode with its computed decay score for ranked results.

    Example:
        >>> scored = ScoredEvent(
        ...     event=event_node,
        ...     decay_score=0.85,
        ...     base_score=1.0,
        ... )
        >>> print(f"Event {scored.event.name}: {scored.final_score:.2f}")
    """

    event: EventNode = Field(
        ...,
        description="The event node"
    )
    decay_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Decay factor based on recency (0-1)"
    )
    base_score: float = Field(
        default=1.0,
        ge=0.0,
        description="Base relevance score before decay"
    )
    vector_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Vector similarity score (if hybrid query)"
    )

    @computed_field
    @property
    def final_score(self) -> float:
        """Combined score: base_score * decay_score."""
        return self.base_score * self.decay_score

    @computed_field
    @property
    def event_id(self) -> str:
        """Convenience accessor for event ID."""
        return self.event.id

    @computed_field
    @property
    def event_timestamp(self) -> datetime:
        """Convenience accessor for event timestamp."""
        return self.event.timestamp


class SequenceMatch(BaseModel):
    """A matched event sequence.

    Represents a sequence of events matching a specified pattern,
    including timing information for Phase 2 analysis.

    Example:
        >>> match = SequenceMatch(
        ...     events=[meeting, decision, publication],
        ...     pattern=["Meeting", "Decision", "Publication"],
        ...     gaps=[timedelta(days=2), timedelta(days=5)],
        ... )
        >>> print(f"Sequence spans {match.total_span.days} days")
    """

    events: List[EventNode] = Field(
        ...,
        min_length=2,
        description="Events in the sequence, ordered temporally"
    )
    pattern: List[str] = Field(
        ...,
        min_length=2,
        description="Event type pattern that was matched"
    )
    gaps: List[timedelta] = Field(
        default_factory=list,
        description="Time gaps between consecutive events"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence that this is a meaningful sequence"
    )
    match_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How well events match the pattern (exact=1.0, fuzzy<1.0)"
    )

    @computed_field
    @property
    def total_span(self) -> timedelta:
        """Total time from first to last event."""
        if len(self.events) >= 2:
            return self.events[-1].timestamp - self.events[0].timestamp
        return timedelta(0)

    @computed_field
    @property
    def average_gap(self) -> timedelta:
        """Average time between events in sequence."""
        if self.gaps:
            total_seconds = sum(g.total_seconds() for g in self.gaps)
            return timedelta(seconds=total_seconds / len(self.gaps))
        return timedelta(0)

    @computed_field
    @property
    def start_time(self) -> datetime:
        """Timestamp of first event in sequence."""
        return self.events[0].timestamp if self.events else datetime.min

    @computed_field
    @property
    def end_time(self) -> datetime:
        """Timestamp of last event in sequence."""
        return self.events[-1].timestamp if self.events else datetime.min

    @computed_field
    @property
    def pattern_description(self) -> str:
        """Human-readable pattern description."""
        return " -> ".join(self.pattern)


class RecurringPattern(BaseModel):
    """A discovered recurring temporal pattern.

    Represents a pattern that occurs multiple times in the data,
    discovered through automatic pattern mining.

    Critical for Phase 2 correlation detection.

    Example:
        >>> pattern = RecurringPattern(
        ...     pattern=["Meeting", "Decision"],
        ...     occurrences=5,
        ...     average_gap=timedelta(days=3),
        ... )
        >>> print(f"'{pattern.pattern_description}' occurs {pattern.occurrences} times")
    """

    pattern: List[str] = Field(
        ...,
        min_length=2,
        description="Event type sequence that recurs"
    )
    occurrences: int = Field(
        ...,
        ge=1,
        description="Number of times this pattern occurs"
    )
    average_gap: timedelta = Field(
        ...,
        description="Average time between events in pattern"
    )
    min_gap: Optional[timedelta] = Field(
        None,
        description="Minimum observed gap"
    )
    max_gap: Optional[timedelta] = Field(
        None,
        description="Maximum observed gap"
    )
    examples: List[SequenceMatch] = Field(
        default_factory=list,
        max_length=5,
        description="Example sequence matches (up to 5)"
    )
    statistical_significance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Statistical significance of pattern (0-1)"
    )
    is_causal_candidate: bool = Field(
        default=False,
        description="Flag for Phase 2/3: potential causal pattern"
    )

    @computed_field
    @property
    def pattern_description(self) -> str:
        """Human-readable pattern description."""
        return " -> ".join(self.pattern)

    @computed_field
    @property
    def pattern_length(self) -> int:
        """Number of event types in pattern."""
        return len(self.pattern)


class TemporalCorrelationResult(BaseModel):
    """Result of temporal correlation analysis.

    Captures statistical correlation between two event types,
    forming the foundation for Phase 2 correlation detection
    and Phase 3 causal inference.

    Example:
        >>> result = TemporalCorrelationResult(
        ...     correlation_found=True,
        ...     event_type_a="Meeting",
        ...     event_type_b="Decision",
        ...     co_occurrences=8,
        ...     avg_gap_days=3.5,
        ... )
        >>> print(result.temporal_pattern)
        "Meeting typically precedes Decision by 3.5 days"
    """

    correlation_found: bool = Field(
        ...,
        description="Whether significant correlation was detected"
    )
    event_type_a: str = Field(
        ...,
        description="First event type in correlation"
    )
    event_type_b: str = Field(
        ...,
        description="Second event type in correlation"
    )

    # Statistics (populated if correlation_found=True)
    co_occurrences: Optional[int] = Field(
        None,
        ge=0,
        description="Number of co-occurrences found"
    )
    avg_gap_days: Optional[float] = Field(
        None,
        ge=0,
        description="Average gap between A and B in days"
    )
    min_gap_days: Optional[float] = Field(
        None,
        ge=0,
        description="Minimum observed gap in days"
    )
    max_gap_days: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum observed gap in days"
    )
    std_gap_days: Optional[float] = Field(
        None,
        ge=0,
        description="Standard deviation of gap in days"
    )

    # Correlation strength
    correlation_strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Correlation strength score (0-1)"
    )

    # Phase 2/3 preparation
    is_causal_candidate: bool = Field(
        default=False,
        description="Flag for Phase 2/3: potential causal relationship"
    )
    causal_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in causal interpretation (Phase 3)"
    )

    # Example co-occurrences
    examples: List[Tuple[str, str, float]] = Field(
        default_factory=list,
        description="Example (event_a_id, event_b_id, gap_days) tuples"
    )

    @computed_field
    @property
    def temporal_pattern(self) -> Optional[str]:
        """Human-readable pattern description."""
        if not self.correlation_found or self.avg_gap_days is None:
            return None
        return (
            f"{self.event_type_a} typically precedes "
            f"{self.event_type_b} by {self.avg_gap_days:.1f} days"
        )

    @computed_field
    @property
    def gap_consistency(self) -> Optional[float]:
        """Measure of gap consistency (1 - normalized_std)."""
        if self.std_gap_days is None or self.avg_gap_days is None:
            return None
        if self.avg_gap_days == 0:
            return 1.0
        # Normalized coefficient of variation inverted
        cv = self.std_gap_days / self.avg_gap_days
        return max(0.0, 1.0 - min(cv, 1.0))


class HybridNeighborhoodResult(BaseModel):
    """Combined graph + vector neighborhood result.

    Merges results from PKG temporal neighborhood query
    with vector similarity search for hybrid retrieval.

    Example:
        >>> result = HybridNeighborhoodResult(
        ...     center_event_id="evt_123",
        ...     graph_neighbors=graph_results,
        ...     vector_neighbors=vector_results,
        ... )
        >>> for neighbor in result.merged_neighbors:
        ...     print(f"{neighbor.event.name}: {neighbor.final_score:.2f}")
    """

    center_event_id: str = Field(
        ...,
        description="ID of the center event"
    )
    center_event: Optional[EventNode] = Field(
        None,
        description="The center event node"
    )

    # Graph results
    graph_neighbors: List[ScoredEvent] = Field(
        default_factory=list,
        description="Neighbors from graph traversal"
    )

    # Vector results
    vector_neighbors: List[ScoredEvent] = Field(
        default_factory=list,
        description="Neighbors from vector similarity"
    )

    # Merge settings
    vector_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in merge"
    )

    # Time bounds
    time_window: Optional[timedelta] = Field(
        None,
        description="Time window used for query"
    )
    time_bounds: Optional[Tuple[datetime, datetime]] = Field(
        None,
        description="Actual time range (start, end)"
    )

    # Performance
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total query execution time in milliseconds"
    )
    graph_query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Graph query time in milliseconds"
    )
    vector_query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Vector query time in milliseconds"
    )

    @computed_field
    @property
    def merged_neighbors(self) -> List[ScoredEvent]:
        """Merged and ranked neighbors using RRF-like fusion.

        Combines graph and vector results using reciprocal rank fusion,
        weighted by the vector_weight parameter.
        """
        # Build score maps
        graph_scores: Dict[str, float] = {
            n.event_id: n.final_score for n in self.graph_neighbors
        }
        vector_scores: Dict[str, float] = {
            n.event_id: n.vector_similarity or 0.0 for n in self.vector_neighbors
        }

        # Get all unique event IDs
        all_ids = set(graph_scores.keys()) | set(vector_scores.keys())

        # Build merged results
        merged: Dict[str, ScoredEvent] = {}
        for event_id in all_ids:
            g_score = graph_scores.get(event_id, 0.0)
            v_score = vector_scores.get(event_id, 0.0)

            # Weighted combination
            graph_weight = 1.0 - self.vector_weight
            combined_score = (graph_weight * g_score) + (self.vector_weight * v_score)

            # Get event from either source
            event = None
            for n in self.graph_neighbors:
                if n.event_id == event_id:
                    event = n.event
                    break
            if event is None:
                for n in self.vector_neighbors:
                    if n.event_id == event_id:
                        event = n.event
                        break

            if event:
                merged[event_id] = ScoredEvent(
                    event=event,
                    base_score=combined_score,
                    decay_score=1.0,  # Already applied
                    vector_similarity=vector_scores.get(event_id),
                )

        # Sort by final score descending
        return sorted(
            merged.values(),
            key=lambda x: x.final_score,
            reverse=True,
        )

    @computed_field
    @property
    def total_neighbors(self) -> int:
        """Total unique neighbors found."""
        ids = set()
        for n in self.graph_neighbors:
            ids.add(n.event_id)
        for n in self.vector_neighbors:
            ids.add(n.event_id)
        return len(ids)


class TemporalSearchResult(BaseModel):
    """Generic wrapper for temporal search results.

    Provides consistent structure for all temporal query results
    with pagination and performance metadata.
    """

    items: List[ScoredEvent] = Field(
        default_factory=list,
        description="Search result items"
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total matching items (may exceed items length)"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset used in query"
    )
    limit: int = Field(
        default=100,
        ge=1,
        description="Limit used in query"
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds"
    )
    decay_applied: bool = Field(
        default=False,
        description="Whether decay scoring was applied"
    )

    @computed_field
    @property
    def has_more(self) -> bool:
        """Are there more results beyond current page?"""
        return (self.offset + len(self.items)) < self.total_count

    @computed_field
    @property
    def next_offset(self) -> int:
        """Offset for next page of results."""
        return self.offset + len(self.items)

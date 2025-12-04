"""Temporal Query Result Models.

Defines Pydantic models for temporal and causal query results:
- CausalPath: Single causal chain path
- CausalChainResult: Collection of causal paths
- TemporalNeighborhood: Entities/events within temporal window
- TemporalQueryResult: Generic paginated result wrapper

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/04-temporal-query-support.md

Option B Compliance:
- CausalPath includes Bradford Hill metadata for Phase 3
- Causal confidence aggregation supports validation
- Temporal bounds tracked for correlation analysis
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import reduce
from operator import mul
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, computed_field

from futurnal.pkg.schema.models import BaseNode, EventNode


T = TypeVar("T", bound=BaseNode)


class CausalPath(BaseModel):
    """Single causal chain path.

    Represents a path A -> B -> C where each arrow is a CAUSES relationship.
    Includes confidence scores for each link and aggregate confidence.

    Option B Compliance:
    - Supports Phase 3 Bradford Hill validation via is_validated field
    - Tracks causal evidence for each relationship
    - Aggregate confidence enables filtering weak chains

    Example:
        >>> path = CausalPath(
        ...     events=[event_a, event_b, event_c],
        ...     confidences=[0.8, 0.9],
        ...     causal_evidence=["A led to B", "B caused C"]
        ... )
        >>> print(path.aggregate_confidence)  # 0.72
        >>> print(path.depth)  # 2
    """

    events: List[EventNode] = Field(
        ...,
        min_length=2,
        description="Ordered list of events in the causal chain [cause, ..., effect]"
    )
    confidences: List[float] = Field(
        ...,
        description="Causal confidence for each CAUSES relationship"
    )
    causal_evidence: List[str] = Field(
        default_factory=list,
        description="Text evidence for each causal link"
    )
    is_validated: bool = Field(
        default=False,
        description="Has this path been validated by Phase 3?"
    )
    validation_method: Optional[str] = Field(
        None,
        description="Method used for Phase 3 validation"
    )

    @computed_field
    @property
    def aggregate_confidence(self) -> float:
        """Calculate aggregate confidence as product of all link confidences.

        Returns 1.0 if no confidences (should not happen with valid data).
        """
        if not self.confidences:
            return 1.0
        return reduce(mul, self.confidences, 1.0)

    @computed_field
    @property
    def depth(self) -> int:
        """Number of causal hops in this path."""
        return len(self.events) - 1 if self.events else 0

    @computed_field
    @property
    def start_event(self) -> Optional[EventNode]:
        """First event in the causal chain (root cause)."""
        return self.events[0] if self.events else None

    @computed_field
    @property
    def end_event(self) -> Optional[EventNode]:
        """Last event in the causal chain (final effect)."""
        return self.events[-1] if self.events else None

    @computed_field
    @property
    def temporal_span(self) -> Optional[timedelta]:
        """Time between first and last event."""
        if len(self.events) >= 2:
            return self.events[-1].timestamp - self.events[0].timestamp
        return None


class CausalChainResult(BaseModel):
    """Result of a causal chain query.

    Contains all causal paths found from a starting event, with metadata
    for Phase 2 analysis and Phase 3 validation.

    Example:
        >>> result = query_causal_chain(start_event_id="event_123", max_hops=5)
        >>> print(f"Found {result.total_paths} chains, max depth {result.max_depth_found}")
        >>> for path in result.paths:
        ...     if path.aggregate_confidence > 0.7:
        ...         print(f"Strong chain: {path.start_event.name} -> {path.end_event.name}")
    """

    paths: List[CausalPath] = Field(
        default_factory=list,
        description="All causal paths found"
    )
    start_event_id: str = Field(
        ...,
        description="ID of the starting event"
    )
    start_event: Optional[EventNode] = Field(
        None,
        description="The starting event node"
    )
    max_hops_requested: int = Field(
        ...,
        description="Maximum hops requested in query"
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds"
    )

    @computed_field
    @property
    def total_paths(self) -> int:
        """Total number of causal paths found."""
        return len(self.paths)

    @computed_field
    @property
    def max_depth_found(self) -> int:
        """Maximum depth among all paths found."""
        if not self.paths:
            return 0
        return max(path.depth for path in self.paths)

    @computed_field
    @property
    def unique_effects(self) -> List[str]:
        """Unique end event IDs reachable from start."""
        seen = set()
        effects = []
        for path in self.paths:
            if path.end_event and path.end_event.id not in seen:
                seen.add(path.end_event.id)
                effects.append(path.end_event.id)
        return effects

    def filter_by_confidence(self, min_confidence: float) -> "CausalChainResult":
        """Return new result with only paths above confidence threshold.

        Args:
            min_confidence: Minimum aggregate confidence (0.0 to 1.0)

        Returns:
            New CausalChainResult with filtered paths
        """
        filtered_paths = [
            path for path in self.paths
            if path.aggregate_confidence >= min_confidence
        ]
        return CausalChainResult(
            paths=filtered_paths,
            start_event_id=self.start_event_id,
            start_event=self.start_event,
            max_hops_requested=self.max_hops_requested,
            query_time_ms=self.query_time_ms,
        )

    def filter_by_depth(self, max_depth: int) -> "CausalChainResult":
        """Return new result with only paths up to specified depth.

        Args:
            max_depth: Maximum number of causal hops

        Returns:
            New CausalChainResult with filtered paths
        """
        filtered_paths = [
            path for path in self.paths
            if path.depth <= max_depth
        ]
        return CausalChainResult(
            paths=filtered_paths,
            start_event_id=self.start_event_id,
            start_event=self.start_event,
            max_hops_requested=self.max_hops_requested,
            query_time_ms=self.query_time_ms,
        )


class NeighborRelationship(BaseModel):
    """Relationship connecting center to neighbor in temporal neighborhood.

    Stores relationship metadata for Phase 2 correlation analysis.
    """

    relationship_type: str = Field(
        ...,
        description="Type of relationship (WORKS_AT, PARTICIPATED_IN, etc.)"
    )
    direction: str = Field(
        ...,
        description="Direction relative to center: 'outgoing' or 'incoming'"
    )
    neighbor_id: str = Field(
        ...,
        description="ID of the neighbor node"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relationship properties including valid_from, valid_to, confidence"
    )

    @property
    def valid_from(self) -> Optional[datetime]:
        """When this relationship started."""
        val = self.properties.get("valid_from")
        if isinstance(val, datetime):
            return val
        return None

    @property
    def valid_to(self) -> Optional[datetime]:
        """When this relationship ended (None = ongoing)."""
        val = self.properties.get("valid_to")
        if isinstance(val, datetime):
            return val
        return None

    @property
    def confidence(self) -> float:
        """Relationship confidence score."""
        return float(self.properties.get("confidence", 1.0))


class TemporalNeighborhood(BaseModel):
    """Entities and events within temporal window of a center entity.

    Used for Phase 2 correlation detection - finds all entities/events
    that are related to a given entity within a specified time window.

    Option B Compliance:
    - Temporal bounds explicit for correlation analysis
    - Includes both entity and event neighbors
    - Relationship metadata preserved for analysis

    Example:
        >>> neighborhood = query_temporal_neighborhood(
        ...     entity_id="person_123",
        ...     time_window=timedelta(days=30)
        ... )
        >>> events = [n for n in neighborhood.neighbors if isinstance(n, EventNode)]
        >>> print(f"Found {len(events)} related events in time window")
    """

    center_id: str = Field(
        ...,
        description="ID of the center entity"
    )
    center_entity: Optional[BaseNode] = Field(
        None,
        description="The center entity node"
    )
    neighbors: List[BaseNode] = Field(
        default_factory=list,
        description="All entities and events in the temporal neighborhood"
    )
    relationships: List[NeighborRelationship] = Field(
        default_factory=list,
        description="Relationships connecting center to neighbors"
    )
    time_window: timedelta = Field(
        ...,
        description="Temporal window used for the query"
    )
    time_bounds: Tuple[datetime, datetime] = Field(
        ...,
        description="Actual time range (start, end) of the query"
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds"
    )

    @computed_field
    @property
    def total_neighbors(self) -> int:
        """Total number of neighbors found."""
        return len(self.neighbors)

    @computed_field
    @property
    def event_neighbors(self) -> List[EventNode]:
        """Filter to only EventNode neighbors."""
        return [n for n in self.neighbors if isinstance(n, EventNode)]

    @computed_field
    @property
    def entity_neighbors(self) -> List[BaseNode]:
        """Filter to non-Event neighbors (Person, Organization, etc.)."""
        return [n for n in self.neighbors if not isinstance(n, EventNode)]

    def get_neighbors_by_relationship(
        self, relationship_type: str
    ) -> List[BaseNode]:
        """Get neighbors connected by specific relationship type.

        Args:
            relationship_type: Type of relationship to filter by

        Returns:
            List of neighbors connected by that relationship type
        """
        neighbor_ids = {
            rel.neighbor_id
            for rel in self.relationships
            if rel.relationship_type == relationship_type
        }
        return [n for n in self.neighbors if n.id in neighbor_ids]


class TemporalQueryResult(BaseModel, Generic[T]):
    """Generic paginated result wrapper for temporal queries.

    Provides consistent pagination interface for all temporal query results.

    Example:
        >>> result = query_events_in_timerange(start, end)
        >>> for event in result.items:
        ...     print(event.name)
        >>> if result.has_more:
        ...     next_result = query_events_in_timerange(start, end, offset=result.next_offset)
    """

    items: List[T] = Field(
        default_factory=list,
        description="Query result items"
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of matching items (may exceed items length)"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset used in this query"
    )
    limit: int = Field(
        default=100,
        ge=1,
        description="Limit used in this query"
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Query execution time in milliseconds"
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

    @computed_field
    @property
    def page_count(self) -> int:
        """Number of items in current page."""
        return len(self.items)

"""Search module test fixtures.

Provides fixtures for testing the temporal search engine:
- Mock PKG queries for unit tests
- Test event data generators
- Integration test fixtures using testcontainers

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

from futurnal.pkg.schema.models import EventNode
from futurnal.pkg.queries.models import (
    CausalChainResult,
    CausalPath,
    TemporalNeighborhood,
)
from futurnal.search.config import SearchConfig, TemporalSearchConfig
from futurnal.search.temporal.types import TemporalQuery, TemporalQueryType


# ---------------------------------------------------------------------------
# Test Markers (import from PKG conftest for consistency)
# ---------------------------------------------------------------------------

try:
    from tests.pkg.conftest import (
        requires_neo4j,
        requires_testcontainers,
        requires_docker,
    )
except ImportError:
    # Define markers if PKG conftest not available
    requires_neo4j = pytest.mark.skipif(
        True, reason="Neo4j tests require testcontainers"
    )
    requires_testcontainers = pytest.mark.skipif(
        True, reason="Testcontainers not available"
    )
    requires_docker = pytest.mark.skipif(
        True, reason="Docker not available"
    )


# ---------------------------------------------------------------------------
# Test Event Factory
# ---------------------------------------------------------------------------


def create_test_event(
    event_id: str,
    name: str,
    event_type: str,
    timestamp: datetime,
    description: str = "",
    confidence: float = 1.0,
) -> EventNode:
    """Create a test EventNode.

    Args:
        event_id: Unique event ID
        name: Event name
        event_type: Event type (meeting, decision, etc.)
        timestamp: Event timestamp
        description: Optional description
        confidence: Extraction confidence (default 1.0)

    Returns:
        EventNode instance
    """
    return EventNode(
        id=event_id,
        name=name,
        event_type=event_type,
        timestamp=timestamp,
        source_document="test_doc",
        description=description or f"Test event: {name}",
        confidence=confidence,
    )


def create_event_series(
    base_id: str,
    event_type: str,
    start_date: datetime,
    count: int,
    gap_days: int = 7,
) -> List[EventNode]:
    """Create a series of events with regular spacing.

    Args:
        base_id: Base ID prefix (will be suffixed with _0, _1, etc.)
        event_type: Event type for all events
        start_date: First event date
        count: Number of events
        gap_days: Days between events

    Returns:
        List of EventNode instances
    """
    events = []
    for i in range(count):
        events.append(create_test_event(
            event_id=f"{base_id}_{i}",
            name=f"{event_type.title()} {i + 1}",
            event_type=event_type,
            timestamp=start_date + timedelta(days=i * gap_days),
        ))
    return events


def create_pattern_events(
    pattern: List[str],
    start_date: datetime,
    occurrence_count: int = 3,
    gap_days: int = 3,
    occurrence_gap_days: int = 30,
) -> List[EventNode]:
    """Create events forming a recurring pattern.

    Args:
        pattern: Event types in pattern (e.g., ["Meeting", "Decision"])
        start_date: First occurrence start date
        occurrence_count: Number of pattern occurrences
        gap_days: Days between events within pattern
        occurrence_gap_days: Days between pattern occurrences

    Returns:
        List of EventNode instances
    """
    events = []
    for occ in range(occurrence_count):
        occ_start = start_date + timedelta(days=occ * occurrence_gap_days)
        for i, event_type in enumerate(pattern):
            events.append(create_test_event(
                event_id=f"pattern_{occ}_{i}",
                name=f"{event_type} (Occurrence {occ + 1})",
                event_type=event_type,
                timestamp=occ_start + timedelta(days=i * gap_days),
            ))
    return sorted(events, key=lambda e: e.timestamp)


# ---------------------------------------------------------------------------
# Mock PKG Queries
# ---------------------------------------------------------------------------


class MockTemporalGraphQueries:
    """Mock TemporalGraphQueries for unit testing.

    Provides controllable responses for PKG query methods
    without requiring actual Neo4j connection.
    """

    def __init__(self, events: Optional[List[EventNode]] = None):
        """Initialize with optional preset events.

        Args:
            events: List of events to return from queries
        """
        self._events = events or []
        self._events_by_type: Dict[str, List[EventNode]] = {}
        for e in self._events:
            if e.event_type:
                if e.event_type not in self._events_by_type:
                    self._events_by_type[e.event_type] = []
                self._events_by_type[e.event_type].append(e)

    def set_events(self, events: List[EventNode]) -> None:
        """Set events for queries."""
        self._events = events
        self._events_by_type = {}
        for e in events:
            if e.event_type:
                if e.event_type not in self._events_by_type:
                    self._events_by_type[e.event_type] = []
                self._events_by_type[e.event_type].append(e)

    def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[EventNode]:
        """Mock time range query."""
        events = self._events
        if event_type:
            events = self._events_by_type.get(event_type, [])

        # Filter by time range
        filtered = [
            e for e in events
            if start <= self._strip_tz(e.timestamp) <= end
        ]

        # Sort by timestamp
        filtered.sort(key=lambda e: e.timestamp)

        # Apply pagination
        return filtered[offset:offset + limit]

    def query_events_before(
        self,
        reference_event_id: str,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Mock before query."""
        ref_event = self._get_event_by_id(reference_event_id)
        if not ref_event:
            return []

        ref_time = self._strip_tz(ref_event.timestamp)
        start_time = ref_time - time_window if time_window else datetime.min

        events = [
            e for e in self._events
            if start_time <= self._strip_tz(e.timestamp) < ref_time
        ]
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    def query_events_after(
        self,
        reference_event_id: str,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Mock after query."""
        ref_event = self._get_event_by_id(reference_event_id)
        if not ref_event:
            return []

        ref_time = self._strip_tz(ref_event.timestamp)
        end_time = ref_time + time_window if time_window else datetime.max

        events = [
            e for e in self._events
            if ref_time < self._strip_tz(e.timestamp) <= end_time
        ]
        return sorted(events, key=lambda e: e.timestamp)[:limit]

    def query_temporal_neighborhood(
        self,
        entity_id: str,
        time_window: timedelta,
        include_events: bool = True,
        include_entities: bool = True,
    ) -> TemporalNeighborhood:
        """Mock neighborhood query."""
        center = self._get_event_by_id(entity_id)
        if not center:
            # Return empty neighborhood
            return TemporalNeighborhood(
                center_id=entity_id,
                center_entity=None,
                neighbors=[],
                relationships=[],
                time_window=time_window,
                time_bounds=(datetime.min, datetime.max),
            )

        ref_time = self._strip_tz(center.timestamp)
        start = ref_time - time_window
        end = ref_time + time_window

        neighbors = [
            e for e in self._events
            if e.id != entity_id and start <= self._strip_tz(e.timestamp) <= end
        ]

        return TemporalNeighborhood(
            center_id=entity_id,
            center_entity=center,
            neighbors=neighbors,
            relationships=[],
            time_window=time_window,
            time_bounds=(start, end),
        )

    def query_causal_chain(
        self,
        start_event_id: str,
        max_hops: int = 5,
    ) -> CausalChainResult:
        """Mock causal chain query - returns empty result."""
        start_event = self._get_event_by_id(start_event_id)
        return CausalChainResult(
            paths=[],
            start_event_id=start_event_id,
            start_event=start_event,
            max_hops_requested=max_hops,
        )

    def _get_event_by_id(self, event_id: str) -> Optional[EventNode]:
        """Get event by ID."""
        for e in self._events:
            if e.id == event_id:
                return e
        return None

    def _strip_tz(self, dt: datetime) -> datetime:
        """Strip timezone for comparison."""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temporal_config() -> TemporalSearchConfig:
    """Provide default temporal search config."""
    return TemporalSearchConfig(
        decay_half_life_days=30.0,
        correlation_min_occurrences=3,
        default_max_gap_days=30,
    )


@pytest.fixture
def search_config(temporal_config) -> SearchConfig:
    """Provide full search config."""
    return SearchConfig(temporal=temporal_config)


@pytest.fixture
def sample_events() -> List[EventNode]:
    """Provide sample events for testing."""
    return [
        create_test_event("e1", "Meeting 1", "meeting", datetime(2024, 1, 1, 10, 0)),
        create_test_event("e2", "Decision 1", "decision", datetime(2024, 1, 3, 14, 0)),
        create_test_event("e3", "Meeting 2", "meeting", datetime(2024, 1, 15, 9, 0)),
        create_test_event("e4", "Decision 2", "decision", datetime(2024, 1, 18, 11, 0)),
        create_test_event("e5", "Publication 1", "publication", datetime(2024, 1, 25, 16, 0)),
        create_test_event("e6", "Meeting 3", "meeting", datetime(2024, 2, 1, 10, 0)),
        create_test_event("e7", "Decision 3", "decision", datetime(2024, 2, 4, 15, 0)),
    ]


@pytest.fixture
def mock_pkg_queries(sample_events) -> MockTemporalGraphQueries:
    """Provide mock PKG queries with sample events."""
    return MockTemporalGraphQueries(events=sample_events)


@pytest.fixture
def pattern_events() -> List[EventNode]:
    """Provide events forming Meeting -> Decision pattern."""
    return create_pattern_events(
        pattern=["meeting", "decision"],
        start_date=datetime(2024, 1, 1),
        occurrence_count=5,
        gap_days=3,
        occurrence_gap_days=14,
    )

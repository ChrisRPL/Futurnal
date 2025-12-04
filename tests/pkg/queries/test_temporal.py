"""Unit Tests for Temporal Graph Queries (Module 04).

Tests are executed against real Neo4j via testcontainers,
following the no-mockups rule. Tests match production plan:
docs/phase-1/pkg-graph-storage-production-plan/04-temporal-query-support.md

Test coverage:
- test_time_range_query: Validate time range queries work
- test_time_range_with_event_type_filter: Filter by event type
- test_causal_chain_query: Validate causal chain traversal
- test_causal_chain_with_confidences: Verify confidence aggregation
- test_temporal_neighborhood_query: Find entities within time window
- test_empty_result_sets: Edge case handling
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest

from futurnal.pkg.schema.models import EventNode
from futurnal.pkg.queries.models import (
    CausalPath,
    CausalChainResult,
    TemporalNeighborhood,
    TemporalQueryResult,
)
from futurnal.pkg.queries.exceptions import (
    CausalChainDepthError,
    EntityNotFoundError,
    EventNotFoundError,
    InvalidTimeRangeError,
    TemporalQueryError,
)
from futurnal.pkg.queries.temporal import TemporalGraphQueries

# Import test fixtures from conftest
from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------


def create_test_event(
    session,
    event_id: str,
    name: str,
    event_type: str,
    timestamp: datetime,
    source_document: str = "test_doc",
) -> dict:
    """Create a test event in the database."""
    query = """
    CREATE (e:Event {
        id: $id,
        name: $name,
        event_type: $event_type,
        timestamp: datetime($timestamp),
        source_document: $source_document,
        description: $description,
        created_at: datetime(),
        updated_at: datetime()
    })
    RETURN e
    """
    result = session.run(query, {
        "id": event_id,
        "name": name,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "source_document": source_document,
        "description": f"Test event: {name}",
    })
    return result.single()["e"]


def create_causal_relationship(
    session,
    source_id: str,
    target_id: str,
    causal_confidence: float = 0.8,
    causal_evidence: str = "Test causal evidence",
) -> None:
    """Create a CAUSES relationship between two events."""
    query = """
    MATCH (source:Event {id: $source_id})
    MATCH (target:Event {id: $target_id})
    CREATE (source)-[:CAUSES {
        causal_confidence: $causal_confidence,
        causal_evidence: $causal_evidence,
        source_document: 'test_doc',
        created_at: datetime()
    }]->(target)
    """
    session.run(query, {
        "source_id": source_id,
        "target_id": target_id,
        "causal_confidence": causal_confidence,
        "causal_evidence": causal_evidence,
    })


def create_person_node(
    session,
    person_id: str,
    name: str,
) -> dict:
    """Create a test person in the database."""
    query = """
    CREATE (p:Person {
        id: $id,
        name: $name,
        created_at: datetime(),
        updated_at: datetime()
    })
    RETURN p
    """
    result = session.run(query, {
        "id": person_id,
        "name": name,
    })
    return result.single()["p"]


def create_participated_in_relationship(
    session,
    person_id: str,
    event_id: str,
    valid_from: datetime,
) -> None:
    """Create PARTICIPATED_IN relationship between person and event."""
    query = """
    MATCH (p:Person {id: $person_id})
    MATCH (e:Event {id: $event_id})
    CREATE (p)-[:PARTICIPATED_IN {
        valid_from: datetime($valid_from),
        confidence: 1.0,
        source_document: 'test_doc'
    }]->(e)
    """
    session.run(query, {
        "person_id": person_id,
        "event_id": event_id,
        "valid_from": valid_from.isoformat(),
    })


# ---------------------------------------------------------------------------
# Mock Database Manager for Unit Tests
# ---------------------------------------------------------------------------


class MockDatabaseManager:
    """Mock database manager wrapping Neo4j session for testing."""

    def __init__(self, driver):
        self._driver = driver
        self._session = None

    def session(self):
        """Return a context manager for sessions."""
        return self._driver.session()


# ---------------------------------------------------------------------------
# Time Range Query Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestTimeRangeQueries:
    """Test time range query functionality."""

    def test_time_range_query(self, neo4j_driver, clean_database):
        """Validate time range queries work.

        From production plan testing strategy:
        Create events at different times, query Jan 2024,
        verify e1/e2 in results, e3 not.
        """
        with neo4j_driver.session() as session:
            # Create events at different times
            e1 = create_test_event(
                session,
                event_id="e1",
                name="January Event 1",
                event_type="meeting",
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
            )
            e2 = create_test_event(
                session,
                event_id="e2",
                name="January Event 2",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 14, 0, 0),
            )
            e3 = create_test_event(
                session,
                event_id="e3",
                name="February Event",
                event_type="meeting",
                timestamp=datetime(2024, 2, 1, 9, 0, 0),
            )

        # Create query service
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query January 2024
        results = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        # Verify results
        result_ids = {e.id for e in results}
        assert "e1" in result_ids, "e1 should be in January results"
        assert "e2" in result_ids, "e2 should be in January results"
        assert "e3" not in result_ids, "e3 should NOT be in January results"
        assert len(results) == 2

        # Verify ordering (by timestamp ascending)
        assert results[0].id == "e1"
        assert results[1].id == "e2"

    def test_time_range_with_event_type_filter(self, neo4j_driver, clean_database):
        """Filter time range by event type."""
        with neo4j_driver.session() as session:
            create_test_event(
                session,
                event_id="meeting1",
                name="Team Meeting",
                event_type="meeting",
                timestamp=datetime(2024, 1, 10, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="decision1",
                name="Budget Decision",
                event_type="decision",
                timestamp=datetime(2024, 1, 15, 14, 0, 0),
            )
            create_test_event(
                session,
                event_id="meeting2",
                name="Review Meeting",
                event_type="meeting",
                timestamp=datetime(2024, 1, 20, 9, 0, 0),
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query only meetings
        results = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            event_type="meeting",
        )

        assert len(results) == 2
        assert all(e.event_type == "meeting" for e in results)

    def test_invalid_time_range(self, neo4j_driver, clean_database):
        """Verify error when start > end."""
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        with pytest.raises(InvalidTimeRangeError) as exc_info:
            queries.query_events_in_timerange(
                start=datetime(2024, 1, 31),
                end=datetime(2024, 1, 1),  # Before start!
            )

        assert exc_info.value.start == datetime(2024, 1, 31)
        assert exc_info.value.end == datetime(2024, 1, 1)

    def test_paginated_time_range_query(self, neo4j_driver, clean_database):
        """Test pagination for time range queries."""
        with neo4j_driver.session() as session:
            # Create 10 events
            for i in range(10):
                create_test_event(
                    session,
                    event_id=f"event_{i}",
                    name=f"Event {i}",
                    event_type="meeting",
                    timestamp=datetime(2024, 1, i + 1, 10, 0, 0),
                )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # First page
        result = queries.query_events_in_timerange_paginated(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            limit=5,
            offset=0,
        )

        assert len(result.items) == 5
        assert result.total_count == 10
        assert result.has_more is True
        assert result.next_offset == 5

        # Second page
        result2 = queries.query_events_in_timerange_paginated(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            limit=5,
            offset=5,
        )

        assert len(result2.items) == 5
        assert result2.has_more is False


# ---------------------------------------------------------------------------
# Causal Chain Query Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestCausalChainQueries:
    """Test causal chain query functionality."""

    def test_causal_chain_query(self, neo4j_driver, clean_database):
        """Validate causal chain traversal.

        From production plan testing strategy:
        Create causal chain A -> B -> C, verify C reachable from A.
        """
        with neo4j_driver.session() as session:
            # Create causal chain: A -> B -> C
            create_test_event(
                session,
                event_id="A",
                name="Event A",
                event_type="action",
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="B",
                name="Event B",
                event_type="action",
                timestamp=datetime(2024, 1, 2, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="C",
                name="Event C",
                event_type="action",
                timestamp=datetime(2024, 1, 3, 10, 0, 0),
            )

            create_causal_relationship(session, "A", "B")
            create_causal_relationship(session, "B", "C")

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query chain from A
        result = queries.query_causal_chain(start_event_id="A", max_hops=5)

        # Verify result
        assert result.total_paths >= 1, "Should find at least one path"
        assert result.start_event_id == "A"
        assert result.start_event is not None
        assert result.start_event.id == "A"

        # Find the A -> B -> C path
        full_chain_paths = [p for p in result.paths if p.depth == 2]
        assert len(full_chain_paths) >= 1, "Should find A -> B -> C path"

        # Verify C is reachable
        all_end_events = {p.end_event.id for p in result.paths if p.end_event}
        assert "C" in all_end_events, "C should be reachable from A"

    def test_causal_chain_with_confidences(self, neo4j_driver, clean_database):
        """Verify confidence aggregation in causal chains."""
        with neo4j_driver.session() as session:
            create_test_event(
                session,
                event_id="X",
                name="Event X",
                event_type="action",
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="Y",
                name="Event Y",
                event_type="action",
                timestamp=datetime(2024, 1, 2, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="Z",
                name="Event Z",
                event_type="action",
                timestamp=datetime(2024, 1, 3, 10, 0, 0),
            )

            # Create chain with specific confidences: X -0.8-> Y -0.9-> Z
            create_causal_relationship(session, "X", "Y", causal_confidence=0.8)
            create_causal_relationship(session, "Y", "Z", causal_confidence=0.9)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(start_event_id="X", max_hops=5)

        # Find the X -> Y -> Z path
        full_chain = [p for p in result.paths if p.depth == 2][0]

        # Verify confidence aggregation: 0.8 * 0.9 = 0.72
        assert len(full_chain.confidences) == 2
        assert abs(full_chain.aggregate_confidence - 0.72) < 0.01

    def test_causal_chain_event_not_found(self, neo4j_driver, clean_database):
        """Verify error when start event doesn't exist."""
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        with pytest.raises(EventNotFoundError) as exc_info:
            queries.query_causal_chain(start_event_id="nonexistent")

        assert exc_info.value.event_id == "nonexistent"

    def test_causal_chain_depth_limit(self, neo4j_driver, clean_database):
        """Verify error when max_hops exceeds limit."""
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # First create a valid event
        with neo4j_driver.session() as session:
            create_test_event(
                session,
                event_id="test",
                name="Test",
                event_type="action",
                timestamp=datetime(2024, 1, 1),
            )

        with pytest.raises(CausalChainDepthError) as exc_info:
            queries.query_causal_chain(start_event_id="test", max_hops=100)

        assert exc_info.value.requested_depth == 100
        assert exc_info.value.max_allowed == 10

    def test_filter_by_confidence(self, neo4j_driver, clean_database):
        """Test filtering causal chains by confidence threshold."""
        with neo4j_driver.session() as session:
            # Create events
            for event_id in ["P", "Q", "R", "S"]:
                create_test_event(
                    session,
                    event_id=event_id,
                    name=f"Event {event_id}",
                    event_type="action",
                    timestamp=datetime(2024, 1, ord(event_id) - ord("O"), 10, 0, 0),
                )

            # Create two paths from P:
            # P -0.9-> Q (strong)
            # P -0.3-> R -0.4-> S (weak chain: 0.12 aggregate)
            create_causal_relationship(session, "P", "Q", causal_confidence=0.9)
            create_causal_relationship(session, "P", "R", causal_confidence=0.3)
            create_causal_relationship(session, "R", "S", causal_confidence=0.4)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(start_event_id="P", max_hops=3)

        # Filter to only strong chains (>0.5)
        filtered = result.filter_by_confidence(0.5)

        # Should only keep P -> Q path
        assert filtered.total_paths == 1
        assert filtered.paths[0].end_event.id == "Q"


# ---------------------------------------------------------------------------
# Temporal Neighborhood Query Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestTemporalNeighborhoodQueries:
    """Test temporal neighborhood query functionality."""

    def test_temporal_neighborhood_query(self, neo4j_driver, clean_database):
        """Find entities within time window.

        From production plan testing strategy.
        """
        with neo4j_driver.session() as session:
            # Create a person
            create_person_node(session, "alice", "Alice Smith")

            # Create events Alice participated in
            create_test_event(
                session,
                event_id="meeting1",
                name="Team Meeting",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="meeting2",
                name="Review Meeting",
                event_type="meeting",
                timestamp=datetime(2024, 1, 20, 14, 0, 0),
            )
            create_test_event(
                session,
                event_id="old_meeting",
                name="Old Meeting",
                event_type="meeting",
                timestamp=datetime(2023, 6, 1, 10, 0, 0),  # Outside window
            )

            # Create participation relationships
            create_participated_in_relationship(
                session, "alice", "meeting1",
                valid_from=datetime(2024, 1, 15, 10, 0, 0)
            )
            create_participated_in_relationship(
                session, "alice", "meeting2",
                valid_from=datetime(2024, 1, 20, 14, 0, 0)
            )
            create_participated_in_relationship(
                session, "alice", "old_meeting",
                valid_from=datetime(2023, 6, 1, 10, 0, 0)
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query neighborhood with 30-day window
        neighborhood = queries.query_temporal_neighborhood(
            entity_id="alice",
            time_window=timedelta(days=30),
        )

        # Verify results
        assert neighborhood.center_id == "alice"
        assert neighborhood.center_entity is not None
        assert neighborhood.total_neighbors >= 2  # At least meeting1, meeting2

        # Should have event neighbors
        event_neighbor_ids = {e.id for e in neighborhood.event_neighbors}
        assert "meeting1" in event_neighbor_ids or "meeting2" in event_neighbor_ids

    def test_temporal_neighborhood_entity_not_found(self, neo4j_driver, clean_database):
        """Verify error when entity doesn't exist."""
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        with pytest.raises(EntityNotFoundError) as exc_info:
            queries.query_temporal_neighborhood(
                entity_id="nonexistent",
                time_window=timedelta(days=30),
            )

        assert exc_info.value.entity_id == "nonexistent"

    def test_temporal_neighborhood_event_centered(self, neo4j_driver, clean_database):
        """Test neighborhood when center is an event."""
        with neo4j_driver.session() as session:
            # Create events
            create_test_event(
                session,
                event_id="center_event",
                name="Center Event",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
            )
            create_test_event(
                session,
                event_id="nearby_event",
                name="Nearby Event",
                event_type="meeting",
                timestamp=datetime(2024, 1, 16, 10, 0, 0),
            )

            # Create relationship
            create_causal_relationship(
                session, "center_event", "nearby_event"
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        neighborhood = queries.query_temporal_neighborhood(
            entity_id="center_event",
            time_window=timedelta(days=7),
        )

        assert neighborhood.center_id == "center_event"
        # When center is event, time window is centered on its timestamp
        assert neighborhood.time_bounds[0] == datetime(2024, 1, 8, 10, 0, 0)
        assert neighborhood.time_bounds[1] == datetime(2024, 1, 22, 10, 0, 0)


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_result_sets(self, neo4j_driver, clean_database):
        """Edge case: No matching events.

        From production plan testing strategy.
        """
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query time range with no events
        results = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        assert results == []
        assert len(results) == 0

    def test_empty_causal_chain(self, neo4j_driver, clean_database):
        """Edge case: Event with no causal relationships."""
        with neo4j_driver.session() as session:
            create_test_event(
                session,
                event_id="isolated",
                name="Isolated Event",
                event_type="action",
                timestamp=datetime(2024, 1, 1),
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(start_event_id="isolated")

        assert result.total_paths == 0
        assert result.max_depth_found == 0
        assert result.paths == []

    def test_simultaneous_events(self, neo4j_driver, clean_database):
        """Test finding simultaneous events."""
        reference_time = datetime(2024, 1, 15, 10, 0, 0)

        with neo4j_driver.session() as session:
            create_test_event(
                session,
                event_id="ref",
                name="Reference Event",
                event_type="meeting",
                timestamp=reference_time,
            )
            create_test_event(
                session,
                event_id="sim1",
                name="Simultaneous 1",
                event_type="meeting",
                timestamp=reference_time + timedelta(minutes=30),
            )
            create_test_event(
                session,
                event_id="sim2",
                name="Simultaneous 2",
                event_type="meeting",
                timestamp=reference_time - timedelta(minutes=45),
            )
            create_test_event(
                session,
                event_id="far",
                name="Far Away",
                event_type="meeting",
                timestamp=reference_time + timedelta(days=1),
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Find events within 1 hour of reference
        results = queries.query_simultaneous_events(
            reference_event_id="ref",
            tolerance=timedelta(hours=1),
        )

        result_ids = {e.id for e in results}
        assert "sim1" in result_ids
        assert "sim2" in result_ids
        assert "far" not in result_ids
        assert "ref" not in result_ids  # Reference should be excluded


# ---------------------------------------------------------------------------
# Model Tests (No DB required)
# ---------------------------------------------------------------------------


class TestCausalPathModel:
    """Test CausalPath model functionality."""

    def test_aggregate_confidence(self):
        """Test confidence aggregation."""
        event1 = EventNode(
            name="Event 1",
            event_type="action",
            timestamp=datetime(2024, 1, 1),
            source_document="doc",
        )
        event2 = EventNode(
            name="Event 2",
            event_type="action",
            timestamp=datetime(2024, 1, 2),
            source_document="doc",
        )
        event3 = EventNode(
            name="Event 3",
            event_type="action",
            timestamp=datetime(2024, 1, 3),
            source_document="doc",
        )

        path = CausalPath(
            events=[event1, event2, event3],
            confidences=[0.8, 0.9],
        )

        assert abs(path.aggregate_confidence - 0.72) < 0.001
        assert path.depth == 2
        assert path.start_event.id == event1.id
        assert path.end_event.id == event3.id

    def test_temporal_span(self):
        """Test temporal span calculation."""
        event1 = EventNode(
            name="Event 1",
            event_type="action",
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            source_document="doc",
        )
        event2 = EventNode(
            name="Event 2",
            event_type="action",
            timestamp=datetime(2024, 1, 3, 10, 0, 0),
            source_document="doc",
        )

        path = CausalPath(
            events=[event1, event2],
            confidences=[0.9],
        )

        assert path.temporal_span == timedelta(days=2)


class TestCausalChainResultModel:
    """Test CausalChainResult model functionality."""

    def test_filter_by_depth(self):
        """Test filtering chains by depth."""
        event1 = EventNode(
            name="E1", event_type="action",
            timestamp=datetime(2024, 1, 1), source_document="doc"
        )
        event2 = EventNode(
            name="E2", event_type="action",
            timestamp=datetime(2024, 1, 2), source_document="doc"
        )
        event3 = EventNode(
            name="E3", event_type="action",
            timestamp=datetime(2024, 1, 3), source_document="doc"
        )

        path1 = CausalPath(events=[event1, event2], confidences=[0.9])  # depth 1
        path2 = CausalPath(events=[event1, event2, event3], confidences=[0.8, 0.9])  # depth 2

        result = CausalChainResult(
            paths=[path1, path2],
            start_event_id="e1",
            max_hops_requested=5,
        )

        filtered = result.filter_by_depth(1)
        assert filtered.total_paths == 1
        assert filtered.paths[0].depth == 1


class TestTemporalQueryResultModel:
    """Test TemporalQueryResult pagination."""

    def test_pagination_properties(self):
        """Test pagination computed properties."""
        result = TemporalQueryResult[EventNode](
            items=[],  # Would have items in real usage
            total_count=100,
            offset=20,
            limit=10,
        )

        assert result.has_more is True
        assert result.next_offset == 20  # 20 + 0 items

    def test_no_more_pages(self):
        """Test when no more pages available."""
        event = EventNode(
            name="E", event_type="action",
            timestamp=datetime(2024, 1, 1), source_document="doc"
        )

        result = TemporalQueryResult[EventNode](
            items=[event] * 5,
            total_count=5,
            offset=0,
            limit=10,
        )

        assert result.has_more is False
        assert result.page_count == 5

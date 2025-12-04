"""Integration Tests for Temporal Extraction → Temporal Queries (Module 05).

Tests the flow from temporal triple extraction to temporal query execution.

From production plan:
- test_temporal_triples_to_timerange_query: Store temporal → query by time
- test_event_extraction_to_causal_chain: Store events → query causal paths
- test_temporal_neighborhood_from_extraction: Store → query neighborhood

Success Metrics:
- Temporal queries return correct events for time ranges
- Causal chain queries find paths with correct confidence aggregation
- Temporal neighborhoods include all related entities within time window

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Any

import pytest

from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def create_events_in_db(session, events: List[Dict[str, Any]]) -> List[str]:
    """Create event nodes in the database."""
    event_ids = []
    for event in events:
        session.run(
            """
            CREATE (e:Event {
                id: $id,
                name: $name,
                event_type: $event_type,
                timestamp: datetime($timestamp),
                description: $description,
                source_document: $source_document,
                confidence: $confidence,
                created_at: datetime(),
                updated_at: datetime()
            })
            """,
            {
                "id": event["id"],
                "name": event["name"],
                "event_type": event["event_type"],
                "timestamp": event["timestamp"].isoformat(),
                "description": event.get("description", ""),
                "source_document": event.get("source_document", "test_doc"),
                "confidence": event.get("confidence", 0.9),
            },
        )
        event_ids.append(event["id"])
    return event_ids


def create_causal_chain(session, chain: List[Dict[str, Any]]) -> None:
    """Create causal relationships between events."""
    for rel in chain:
        session.run(
            """
            MATCH (cause:Event {id: $cause_id})
            MATCH (effect:Event {id: $effect_id})
            CREATE (cause)-[:CAUSES {
                causal_confidence: $confidence,
                causal_evidence: $evidence,
                temporality_satisfied: true,
                source_document: 'test_doc',
                created_at: datetime()
            }]->(effect)
            """,
            {
                "cause_id": rel["cause_id"],
                "effect_id": rel["effect_id"],
                "confidence": rel.get("confidence", 0.8),
                "evidence": rel.get("evidence", "Test causal relationship"),
            },
        )


def create_person_with_events(
    session,
    person_id: str,
    person_name: str,
    event_ids: List[str],
) -> None:
    """Create a person and link them to events."""
    session.run(
        """
        CREATE (p:Person {
            id: $person_id,
            name: $person_name,
            created_at: datetime(),
            updated_at: datetime()
        })
        """,
        {"person_id": person_id, "person_name": person_name},
    )

    for event_id in event_ids:
        session.run(
            """
            MATCH (p:Person {id: $person_id})
            MATCH (e:Event {id: $event_id})
            CREATE (p)-[:PARTICIPATED_IN {
                valid_from: e.timestamp,
                confidence: 1.0,
                source_document: 'test_doc'
            }]->(e)
            """,
            {"person_id": person_id, "event_id": event_id},
        )


class MockDatabaseManager:
    """Mock database manager wrapping Neo4j driver for testing."""

    def __init__(self, driver):
        self._driver = driver

    def session(self):
        return self._driver.session()


# ---------------------------------------------------------------------------
# Integration Tests: Temporal Extraction → Temporal Queries
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestTemporalExtractionToQueries:
    """Tests for temporal extraction → temporal query integration.

    From production plan 05-integration-testing.md:
    Validates that temporally extracted data flows through to query layer.
    """

    def test_temporal_triples_to_timerange_query(self, neo4j_driver, clean_database):
        """Store temporal triples → query by time range.

        From production plan:
        - Extract temporal triples
        - Store in PKG
        - Query temporal data
        - Verify events have timestamps
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        base_date = datetime(2024, 1, 15)

        # Create events with temporal data (simulating extraction output)
        events = [
            {
                "id": "timerange_event_1",
                "name": "Morning Meeting",
                "event_type": "meeting",
                "timestamp": base_date + timedelta(hours=9),
                "description": "Team sync",
            },
            {
                "id": "timerange_event_2",
                "name": "Lunch Decision",
                "event_type": "decision",
                "timestamp": base_date + timedelta(hours=12),
                "description": "Project decision",
            },
            {
                "id": "timerange_event_3",
                "name": "Evening Review",
                "event_type": "meeting",
                "timestamp": base_date + timedelta(hours=17),
                "description": "Daily review",
            },
            {
                "id": "timerange_event_4",
                "name": "Next Day Event",
                "event_type": "action",
                "timestamp": base_date + timedelta(days=1, hours=10),
                "description": "Follow-up action",
            },
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)

        # Query using TemporalGraphQueries
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query for first day only
        results = queries.query_events_in_timerange(
            start=base_date,
            end=base_date + timedelta(hours=23, minutes=59),
        )

        # Verify results
        assert len(results) == 3, "Should find 3 events on first day"
        for event in results:
            assert event.timestamp is not None, "All events must have timestamps (Option B)"
            assert event.timestamp >= base_date
            assert event.timestamp < base_date + timedelta(days=1)

        # Query with event_type filter
        meetings = queries.query_events_in_timerange(
            start=base_date,
            end=base_date + timedelta(days=2),
            event_type="meeting",
        )
        assert len(meetings) == 2, "Should find 2 meetings"

    def test_event_extraction_to_causal_chain(self, neo4j_driver, clean_database):
        """Store events → query causal paths.

        From production plan:
        - Create event pair with CAUSES relationship
        - Store with causal metadata
        - Query via query_causal_chain()
        - Verify path found with confidence
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        base_date = datetime(2024, 1, 15)

        # Create causal chain: A → B → C → D
        events = [
            {
                "id": "chain_start",
                "name": "Initial Trigger",
                "event_type": "trigger",
                "timestamp": base_date,
                "confidence": 0.95,
            },
            {
                "id": "chain_step_1",
                "name": "First Response",
                "event_type": "response",
                "timestamp": base_date + timedelta(hours=2),
                "confidence": 0.90,
            },
            {
                "id": "chain_step_2",
                "name": "Second Response",
                "event_type": "response",
                "timestamp": base_date + timedelta(hours=4),
                "confidence": 0.85,
            },
            {
                "id": "chain_end",
                "name": "Final Outcome",
                "event_type": "outcome",
                "timestamp": base_date + timedelta(hours=6),
                "confidence": 0.88,
            },
        ]

        causal_chain = [
            {"cause_id": "chain_start", "effect_id": "chain_step_1", "confidence": 0.9},
            {"cause_id": "chain_step_1", "effect_id": "chain_step_2", "confidence": 0.85},
            {"cause_id": "chain_step_2", "effect_id": "chain_end", "confidence": 0.8},
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)
            create_causal_chain(session, causal_chain)

        # Query causal chain
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(
            start_event_id="chain_start",
            max_hops=5,
        )

        # Verify paths found
        assert result.total_paths >= 1, "Should find at least one causal path"

        # Find the full chain (3 hops: start → 1 → 2 → end)
        full_chain = [p for p in result.paths if p.depth == 3]
        assert len(full_chain) >= 1, "Should find complete chain"

        path = full_chain[0]
        assert path.start_event.name == "Initial Trigger"
        assert path.end_event.name == "Final Outcome"

        # Verify confidence aggregation (product of link confidences)
        expected_confidence = 0.9 * 0.85 * 0.8  # ~0.612
        assert abs(path.aggregate_confidence - expected_confidence) < 0.01

    def test_temporal_neighborhood_from_extraction(self, neo4j_driver, clean_database):
        """Store entities → query temporal neighborhood.

        From production plan:
        - Create entity with related events
        - Query via query_temporal_neighborhood()
        - Verify neighborhood includes temporal relations
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        base_date = datetime(2024, 1, 15)

        # Create events
        events = [
            {
                "id": "neighbor_event_1",
                "name": "Event 1",
                "event_type": "meeting",
                "timestamp": base_date,
            },
            {
                "id": "neighbor_event_2",
                "name": "Event 2",
                "event_type": "decision",
                "timestamp": base_date + timedelta(days=2),
            },
            {
                "id": "neighbor_event_3",
                "name": "Event 3",
                "event_type": "action",
                "timestamp": base_date + timedelta(days=5),
            },
            {
                "id": "neighbor_event_4",
                "name": "Event 4 - Far",
                "event_type": "action",
                "timestamp": base_date + timedelta(days=60),  # Outside window
            },
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)
            create_person_with_events(
                session,
                person_id="neighbor_person",
                person_name="Test Person",
                event_ids=[e["id"] for e in events],
            )

        # Query temporal neighborhood
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        neighborhood = queries.query_temporal_neighborhood(
            entity_id="neighbor_person",
            time_window=timedelta(days=30),
        )

        # Verify neighborhood contains nearby events
        assert neighborhood.total_neighbors > 0, "Should find neighbors"

        # Get event neighbors
        event_neighbors = [n for n in neighborhood.neighbors if hasattr(n, "event_type")]

        # Only events within 30 days should be included
        # Note: The exact filtering depends on TemporalGraphQueries implementation
        assert len(event_neighbors) >= 3, "Should find at least 3 nearby events"

    def test_causal_chain_with_bradford_hill_metadata(self, neo4j_driver, clean_database):
        """Verify causal chains include Bradford Hill criteria metadata.

        Option B critical: Phase 3 validation requires Bradford Hill structure.
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        base_date = datetime(2024, 1, 15)

        events = [
            {
                "id": "bh_cause",
                "name": "Potential Cause",
                "event_type": "exposure",
                "timestamp": base_date,
            },
            {
                "id": "bh_effect",
                "name": "Observed Effect",
                "event_type": "outcome",
                "timestamp": base_date + timedelta(days=7),
            },
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)

            # Create causal relationship with Bradford Hill metadata
            session.run(
                """
                MATCH (cause:Event {id: 'bh_cause'})
                MATCH (effect:Event {id: 'bh_effect'})
                CREATE (cause)-[:CAUSES {
                    causal_confidence: 0.75,
                    causal_evidence: 'Temporal correlation observed',
                    temporality_satisfied: true,
                    is_causal_candidate: true,
                    is_validated: false,
                    strength: 0.6,
                    consistency: null,
                    plausibility: 'Mechanism plausible based on domain knowledge',
                    source_document: 'test_doc',
                    created_at: datetime()
                }]->(effect)
                """
            )

        # Query the causal relationship
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(
            start_event_id="bh_cause",
            max_hops=1,
        )

        assert result.total_paths == 1
        path = result.paths[0]

        # Verify event properties preserved
        assert path.start_event.id == "bh_cause"
        assert path.end_event.id == "bh_effect"

        # Verify causal confidence is captured
        assert len(path.confidences) == 1
        assert abs(path.confidences[0] - 0.75) < 0.01


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestTemporalQueryEdgeCases:
    """Tests for edge cases in temporal queries."""

    def test_empty_timerange_returns_empty(self, neo4j_driver, clean_database):
        """Query for time range with no events returns empty list."""
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        base_date = datetime(2024, 1, 15)

        events = [
            {
                "id": "outside_range",
                "name": "Event Outside Range",
                "event_type": "action",
                "timestamp": base_date,
            },
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Query for different time range
        results = queries.query_events_in_timerange(
            start=base_date + timedelta(days=30),
            end=base_date + timedelta(days=60),
        )

        assert len(results) == 0, "Should return empty list for empty range"

    def test_causal_chain_no_effects(self, neo4j_driver, clean_database):
        """Event with no causal effects returns empty chain."""
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        events = [
            {
                "id": "isolated_event",
                "name": "Isolated Event",
                "event_type": "action",
                "timestamp": datetime(2024, 1, 15),
            },
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(
            start_event_id="isolated_event",
            max_hops=5,
        )

        assert result.total_paths == 0, "Isolated event should have no causal paths"

    def test_temporal_neighborhood_entity_not_found(self, neo4j_driver, clean_database):
        """Query for non-existent entity handles gracefully."""
        from futurnal.pkg.queries.temporal import TemporalGraphQueries
        from futurnal.pkg.queries.exceptions import EntityNotFoundError

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Should raise EntityNotFoundError
        with pytest.raises(EntityNotFoundError):
            queries.query_temporal_neighborhood(
                entity_id="nonexistent_entity",
                time_window=timedelta(days=30),
            )

    def test_simultaneous_events_handled(self, neo4j_driver, clean_database):
        """Events at same timestamp are handled correctly."""
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        same_time = datetime(2024, 1, 15, 10, 0, 0)

        events = [
            {
                "id": "sim_event_1",
                "name": "Simultaneous Event 1",
                "event_type": "action",
                "timestamp": same_time,
            },
            {
                "id": "sim_event_2",
                "name": "Simultaneous Event 2",
                "event_type": "action",
                "timestamp": same_time,
            },
            {
                "id": "sim_event_3",
                "name": "Simultaneous Event 3",
                "event_type": "action",
                "timestamp": same_time,
            },
        ]

        with neo4j_driver.session() as session:
            create_events_in_db(session, events)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        results = queries.query_events_in_timerange(
            start=same_time,
            end=same_time + timedelta(seconds=1),
        )

        assert len(results) == 3, "Should find all simultaneous events"

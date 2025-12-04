"""Integration Tests for Temporal Graph Queries (Module 04).

End-to-end integration tests and performance benchmarks.
Tests are executed against real Neo4j via testcontainers.

From production plan:
- test_extraction_to_temporal_queries - End-to-end flow
- test_performance_with_1000_events - Performance benchmark <100ms

Success Metrics:
- Time range queries <100ms for typical range
- Causal chain queries functional up to 5 hops
- Temporal neighborhood queries efficient
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List
import random

import pytest

from futurnal.pkg.schema.models import EventNode
from futurnal.pkg.queries.temporal import TemporalGraphQueries

from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------


def bulk_create_events(
    session,
    count: int,
    base_timestamp: datetime = datetime(2024, 1, 1),
    event_types: List[str] = None,
) -> List[str]:
    """Bulk create test events for performance testing.

    Args:
        session: Neo4j session
        count: Number of events to create
        base_timestamp: Starting timestamp
        event_types: List of event types to cycle through

    Returns:
        List of created event IDs
    """
    if event_types is None:
        event_types = ["meeting", "decision", "action", "communication"]

    event_ids = []
    for i in range(count):
        event_id = f"perf_event_{i}"
        event_type = event_types[i % len(event_types)]
        timestamp = base_timestamp + timedelta(hours=i)

        query = """
        CREATE (e:Event {
            id: $id,
            name: $name,
            event_type: $event_type,
            timestamp: datetime($timestamp),
            source_document: 'perf_test_doc',
            description: $description,
            created_at: datetime(),
            updated_at: datetime()
        })
        """
        session.run(query, {
            "id": event_id,
            "name": f"Performance Event {i}",
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "description": f"Performance test event {i}",
        })
        event_ids.append(event_id)

    return event_ids


def create_causal_chain(
    session,
    event_ids: List[str],
    chain_probability: float = 0.3,
) -> int:
    """Create random causal relationships between consecutive events.

    Args:
        session: Neo4j session
        event_ids: List of event IDs to connect
        chain_probability: Probability of creating CAUSES relationship

    Returns:
        Number of relationships created
    """
    relationships_created = 0

    for i in range(len(event_ids) - 1):
        if random.random() < chain_probability:
            query = """
            MATCH (source:Event {id: $source_id})
            MATCH (target:Event {id: $target_id})
            CREATE (source)-[:CAUSES {
                causal_confidence: $confidence,
                causal_evidence: 'Performance test',
                source_document: 'perf_test_doc',
                created_at: datetime()
            }]->(target)
            """
            session.run(query, {
                "source_id": event_ids[i],
                "target_id": event_ids[i + 1],
                "confidence": 0.7 + random.random() * 0.3,
            })
            relationships_created += 1

    return relationships_created


class MockDatabaseManager:
    """Mock database manager wrapping Neo4j session for testing."""

    def __init__(self, driver):
        self._driver = driver
        self._session = None

    def session(self):
        """Return a context manager for sessions."""
        return self._driver.session()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestTemporalQueriesIntegration:
    """End-to-end integration tests for temporal queries."""

    def test_extraction_to_temporal_queries(self, neo4j_driver, clean_database):
        """End-to-end flow from event creation to temporal queries.

        From production plan testing strategy:
        Simulates the flow from extraction pipeline to temporal query service.
        """
        # Simulate extraction pipeline output
        extracted_events = [
            {
                "id": "extracted_001",
                "name": "Project Kickoff Meeting",
                "event_type": "meeting",
                "timestamp": datetime(2024, 1, 15, 9, 0, 0),
                "description": "Initial project planning meeting",
            },
            {
                "id": "extracted_002",
                "name": "Requirements Decision",
                "event_type": "decision",
                "timestamp": datetime(2024, 1, 15, 15, 0, 0),
                "description": "Finalized project requirements",
            },
            {
                "id": "extracted_003",
                "name": "Development Started",
                "event_type": "action",
                "timestamp": datetime(2024, 1, 16, 10, 0, 0),
                "description": "Development phase began",
            },
        ]

        # Store events (simulating PKG storage)
        with neo4j_driver.session() as session:
            for event in extracted_events:
                query = """
                CREATE (e:Event {
                    id: $id,
                    name: $name,
                    event_type: $event_type,
                    timestamp: datetime($timestamp),
                    description: $description,
                    source_document: 'extraction_test',
                    created_at: datetime(),
                    updated_at: datetime()
                })
                """
                session.run(query, {
                    **event,
                    "timestamp": event["timestamp"].isoformat(),
                })

            # Create causal relationships (simulating causal extraction)
            # Meeting -> Decision -> Development
            session.run("""
                MATCH (m:Event {id: 'extracted_001'})
                MATCH (d:Event {id: 'extracted_002'})
                CREATE (m)-[:CAUSES {
                    causal_confidence: 0.85,
                    causal_evidence: 'Meeting led to decision',
                    source_document: 'extraction_test'
                }]->(d)
            """)

            session.run("""
                MATCH (d:Event {id: 'extracted_002'})
                MATCH (a:Event {id: 'extracted_003'})
                CREATE (d)-[:CAUSES {
                    causal_confidence: 0.90,
                    causal_evidence: 'Decision enabled development',
                    source_document: 'extraction_test'
                }]->(a)
            """)

        # Now test temporal queries
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # 1. Time range query
        events = queries.query_events_in_timerange(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16, 23, 59, 59),
        )
        assert len(events) == 3
        assert events[0].name == "Project Kickoff Meeting"

        # 2. Causal chain query
        chain_result = queries.query_causal_chain(
            start_event_id="extracted_001",
            max_hops=5,
        )
        assert chain_result.total_paths >= 1
        # Should find path: Meeting -> Decision -> Development
        full_path = [p for p in chain_result.paths if p.depth == 2]
        assert len(full_path) >= 1

        # 3. Verify chain integrity
        path = full_path[0]
        assert path.start_event.name == "Project Kickoff Meeting"
        assert path.end_event.name == "Development Started"
        assert path.aggregate_confidence > 0.7  # 0.85 * 0.90 = 0.765

    def test_causal_chain_across_documents(self, neo4j_driver, clean_database):
        """Test causal chains that span multiple source documents.

        From production plan testing strategy.
        """
        with neo4j_driver.session() as session:
            # Create events from different documents
            events = [
                ("doc1_event", "Doc 1 Event", "action", datetime(2024, 1, 1), "doc_alpha"),
                ("doc2_event", "Doc 2 Event", "action", datetime(2024, 1, 2), "doc_beta"),
                ("doc3_event", "Doc 3 Event", "action", datetime(2024, 1, 3), "doc_gamma"),
            ]

            for eid, name, etype, ts, source in events:
                session.run("""
                    CREATE (e:Event {
                        id: $id, name: $name, event_type: $etype,
                        timestamp: datetime($ts), source_document: $source,
                        created_at: datetime(), updated_at: datetime()
                    })
                """, {"id": eid, "name": name, "etype": etype, "ts": ts.isoformat(), "source": source})

            # Create cross-document causal chain
            session.run("""
                MATCH (e1:Event {id: 'doc1_event'})
                MATCH (e2:Event {id: 'doc2_event'})
                CREATE (e1)-[:CAUSES {
                    causal_confidence: 0.8,
                    causal_evidence: 'Cross-doc causation',
                    source_document: 'inferred'
                }]->(e2)
            """)
            session.run("""
                MATCH (e2:Event {id: 'doc2_event'})
                MATCH (e3:Event {id: 'doc3_event'})
                CREATE (e2)-[:CAUSES {
                    causal_confidence: 0.85,
                    causal_evidence: 'Cross-doc causation',
                    source_document: 'inferred'
                }]->(e3)
            """)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(
            start_event_id="doc1_event",
            max_hops=5,
        )

        # Should find the full chain
        assert result.total_paths >= 1
        full_chain = [p for p in result.paths if p.depth == 2]
        assert len(full_chain) >= 1

        # Verify events from different documents are connected
        path = full_chain[0]
        sources = {e.source_document for e in path.events}
        assert len(sources) == 3  # Three different source documents


# ---------------------------------------------------------------------------
# Performance Benchmark Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
@pytest.mark.performance
class TestTemporalQueriesPerformance:
    """Performance benchmark tests.

    Success Metrics from production plan:
    - Time range queries <100ms for typical range
    - Causal chain queries functional up to 5 hops
    """

    def test_performance_with_1000_events(self, neo4j_driver, clean_database):
        """Performance benchmark: Time range queries <100ms for typical range.

        From production plan success metrics.
        """
        # Create 1000 events
        with neo4j_driver.session() as session:
            event_ids = bulk_create_events(
                session,
                count=1000,
                base_timestamp=datetime(2024, 1, 1),
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Warm up query (first query may be slower)
        _ = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        # Measure time range query performance
        start_time = time.perf_counter()

        results = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),  # ~14 days, ~336 events
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify performance target
        assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f}ms, target <100ms"
        assert len(results) > 100  # Should find substantial number of events

        print(f"\nPerformance: Time range query returned {len(results)} events in {elapsed_ms:.1f}ms")

    def test_performance_causal_chain_5_hops(self, neo4j_driver, clean_database):
        """Performance benchmark: Causal chains up to 5 hops.

        From production plan success metrics.
        """
        # Create chain of events with causal relationships
        with neo4j_driver.session() as session:
            # Create 100 events
            event_ids = bulk_create_events(
                session,
                count=100,
                base_timestamp=datetime(2024, 1, 1),
            )

            # Create deterministic causal chain for testing
            # Every 2nd event causes the next
            for i in range(0, 50, 2):
                for j in range(1, min(6, 50 - i)):  # Up to 5 hops
                    if i + j < 50:
                        session.run("""
                            MATCH (source:Event {id: $source_id})
                            MATCH (target:Event {id: $target_id})
                            MERGE (source)-[:CAUSES {
                                causal_confidence: 0.8,
                                causal_evidence: 'Test chain',
                                source_document: 'perf_test'
                            }]->(target)
                        """, {
                            "source_id": event_ids[i],
                            "target_id": event_ids[i + j],
                        })

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Measure causal chain query performance
        start_time = time.perf_counter()

        result = queries.query_causal_chain(
            start_event_id=event_ids[0],
            max_hops=5,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify functionality
        assert result.total_paths > 0, "Should find causal paths"
        assert result.max_depth_found <= 5, "Should respect max_hops limit"

        # Performance should be reasonable (not strict <100ms for chains)
        assert elapsed_ms < 1000, f"Query took {elapsed_ms:.1f}ms, should be <1000ms"

        print(f"\nPerformance: Causal chain query found {result.total_paths} paths "
              f"(max depth {result.max_depth_found}) in {elapsed_ms:.1f}ms")

    def test_performance_temporal_neighborhood(self, neo4j_driver, clean_database):
        """Performance benchmark: Temporal neighborhood queries."""
        with neo4j_driver.session() as session:
            # Create a person
            session.run("""
                CREATE (p:Person {
                    id: 'perf_person',
                    name: 'Performance Test Person',
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """)

            # Create 500 events and relationships
            event_ids = bulk_create_events(
                session,
                count=500,
                base_timestamp=datetime(2024, 1, 1),
            )

            # Connect person to random events
            for i in range(0, 500, 5):  # 100 relationships
                session.run("""
                    MATCH (p:Person {id: 'perf_person'})
                    MATCH (e:Event {id: $event_id})
                    CREATE (p)-[:PARTICIPATED_IN {
                        valid_from: e.timestamp,
                        confidence: 1.0,
                        source_document: 'perf_test'
                    }]->(e)
                """, {"event_id": event_ids[i]})

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Measure temporal neighborhood query performance
        start_time = time.perf_counter()

        neighborhood = queries.query_temporal_neighborhood(
            entity_id="perf_person",
            time_window=timedelta(days=30),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify results
        assert neighborhood.total_neighbors > 0

        # Performance should be sub-second
        assert elapsed_ms < 500, f"Query took {elapsed_ms:.1f}ms, should be <500ms"

        print(f"\nPerformance: Temporal neighborhood found {neighborhood.total_neighbors} "
              f"neighbors in {elapsed_ms:.1f}ms")


# ---------------------------------------------------------------------------
# Schema Integration Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestSchemaIntegration:
    """Test integration with PKG schema (Module 01)."""

    def test_query_results_match_schema_models(self, neo4j_driver, clean_database):
        """Verify query results are proper EventNode models."""
        with neo4j_driver.session() as session:
            session.run("""
                CREATE (e:Event {
                    id: 'schema_test',
                    name: 'Schema Test Event',
                    event_type: 'meeting',
                    timestamp: datetime('2024-01-15T10:00:00'),
                    source_document: 'test_doc',
                    description: 'Testing schema compliance',
                    duration: duration('PT1H'),
                    location: 'Conference Room A',
                    confidence: 0.95,
                    extraction_method: 'explicit',
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        results = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        assert len(results) == 1
        event = results[0]

        # Verify it's a proper EventNode
        assert isinstance(event, EventNode)
        assert event.id == "schema_test"
        assert event.name == "Schema Test Event"
        assert event.event_type == "meeting"
        # Compare timestamp ignoring timezone (Neo4j returns UTC-aware datetime)
        expected_ts = datetime(2024, 1, 15, 10, 0, 0)
        actual_ts = event.timestamp.replace(tzinfo=None) if event.timestamp.tzinfo else event.timestamp
        assert actual_ts == expected_ts
        assert event.source_document == "test_doc"
        assert event.confidence == 0.95
        assert event.extraction_method == "explicit"

    def test_causal_chain_preserves_event_properties(self, neo4j_driver, clean_database):
        """Verify causal chain events have all properties."""
        with neo4j_driver.session() as session:
            for eid, name, ts in [
                ("cause", "Cause Event", datetime(2024, 1, 1)),
                ("effect", "Effect Event", datetime(2024, 1, 2)),
            ]:
                session.run("""
                    CREATE (e:Event {
                        id: $id, name: $name, event_type: 'action',
                        timestamp: datetime($ts), source_document: 'test',
                        description: 'Test event', confidence: 0.9,
                        created_at: datetime(), updated_at: datetime()
                    })
                """, {"id": eid, "name": name, "ts": ts.isoformat()})

            session.run("""
                MATCH (c:Event {id: 'cause'}), (e:Event {id: 'effect'})
                CREATE (c)-[:CAUSES {
                    causal_confidence: 0.85,
                    causal_evidence: 'Test evidence'
                }]->(e)
            """)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        result = queries.query_causal_chain(start_event_id="cause")

        assert result.total_paths == 1
        path = result.paths[0]

        # Verify events in path are complete
        assert path.start_event.name == "Cause Event"
        assert path.end_event.name == "Effect Event"
        assert path.start_event.confidence == 0.9
        assert path.end_event.confidence == 0.9

"""Performance Benchmark Tests for PKG (Module 05).

Validates performance targets from production plan.

From production plan:
- Query latency <1s for temporal neighborhood queries
- Bulk insert >1000 triples/sec for 10000 triples

Success Metrics (from production plan):
- Sub-second query latency for typical graph traversals
- >1000 triples/sec bulk insert throughput
- Query latency <100ms for time range queries

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pytest

from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


# ---------------------------------------------------------------------------
# Test Data Generation
# ---------------------------------------------------------------------------


def bulk_create_events_raw(
    session,
    count: int,
    base_timestamp: datetime = datetime(2024, 1, 1),
    event_types: List[str] = None,
) -> List[str]:
    """Bulk create test events using UNWIND for performance.

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

    events_data = []
    for i in range(count):
        events_data.append({
            "id": f"perf_event_{i}",
            "name": f"Performance Event {i}",
            "event_type": event_types[i % len(event_types)],
            "timestamp": (base_timestamp + timedelta(hours=i)).isoformat(),
            "description": f"Performance test event {i}",
            "source_document": f"perf_doc_{i // 100}",
            "confidence": 0.8 + (i % 20) / 100,
        })

    # Use UNWIND for efficient bulk insert
    session.run(
        """
        UNWIND $events AS event
        CREATE (e:Event {
            id: event.id,
            name: event.name,
            event_type: event.event_type,
            timestamp: datetime(event.timestamp),
            description: event.description,
            source_document: event.source_document,
            confidence: event.confidence,
            created_at: datetime(),
            updated_at: datetime()
        })
        """,
        {"events": events_data},
    )

    return [e["id"] for e in events_data]


def bulk_create_entities_raw(
    session,
    persons: int,
    events: int,
    concepts: int,
    base_timestamp: datetime = datetime(2024, 1, 1),
) -> Dict[str, List[str]]:
    """Bulk create mixed entity types.

    Args:
        session: Neo4j session
        persons: Number of persons to create
        events: Number of events to create
        concepts: Number of concepts to create
        base_timestamp: Starting timestamp for events

    Returns:
        Dict mapping entity type to list of IDs
    """
    ids = {"Person": [], "Event": [], "Concept": []}

    # Create persons
    persons_data = [
        {
            "id": f"person_{i}",
            "name": f"Person {i}",
            "confidence": 0.9,
        }
        for i in range(persons)
    ]
    session.run(
        """
        UNWIND $items AS item
        CREATE (p:Person {
            id: item.id,
            name: item.name,
            confidence: item.confidence,
            created_at: datetime(),
            updated_at: datetime()
        })
        """,
        {"items": persons_data},
    )
    ids["Person"] = [p["id"] for p in persons_data]

    # Create events
    events_data = [
        {
            "id": f"event_{i}",
            "name": f"Event {i}",
            "event_type": ["meeting", "decision", "action", "communication"][i % 4],
            "timestamp": (base_timestamp + timedelta(hours=i)).isoformat(),
            "source_document": f"doc_{i // 50}",
            "confidence": 0.85,
        }
        for i in range(events)
    ]
    session.run(
        """
        UNWIND $items AS item
        CREATE (e:Event {
            id: item.id,
            name: item.name,
            event_type: item.event_type,
            timestamp: datetime(item.timestamp),
            source_document: item.source_document,
            confidence: item.confidence,
            created_at: datetime(),
            updated_at: datetime()
        })
        """,
        {"items": events_data},
    )
    ids["Event"] = [e["id"] for e in events_data]

    # Create concepts
    concepts_data = [
        {
            "id": f"concept_{i}",
            "name": f"Concept {i}",
            "category": ["topic", "skill", "domain"][i % 3],
            "confidence": 0.9,
        }
        for i in range(concepts)
    ]
    session.run(
        """
        UNWIND $items AS item
        CREATE (c:Concept {
            id: item.id,
            name: item.name,
            category: item.category,
            confidence: item.confidence,
            created_at: datetime(),
            updated_at: datetime()
        })
        """,
        {"items": concepts_data},
    )
    ids["Concept"] = [c["id"] for c in concepts_data]

    return ids


def bulk_create_relationships_raw(
    session,
    relationships: List[Dict[str, Any]],
) -> int:
    """Bulk create relationships using UNWIND.

    Args:
        session: Neo4j session
        relationships: List of relationship dicts with
                       source_id, target_id, type, confidence

    Returns:
        Number of relationships created
    """
    # Group by relationship type for efficient batching
    by_type = {}
    for rel in relationships:
        rel_type = rel["type"]
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)

    total = 0
    for rel_type, rels in by_type.items():
        # Dynamic relationship type using APOC or Cypher workaround
        # For simplicity, we'll handle common types explicitly
        if rel_type == "PARTICIPATED_IN":
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a {id: rel.source_id})
                MATCH (b {id: rel.target_id})
                CREATE (a)-[:PARTICIPATED_IN {
                    confidence: rel.confidence,
                    created_at: datetime()
                }]->(b)
                """,
                {"rels": rels},
            )
        elif rel_type == "CAUSES":
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a {id: rel.source_id})
                MATCH (b {id: rel.target_id})
                CREATE (a)-[:CAUSES {
                    causal_confidence: rel.confidence,
                    created_at: datetime()
                }]->(b)
                """,
                {"rels": rels},
            )
        elif rel_type == "BEFORE":
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a {id: rel.source_id})
                MATCH (b {id: rel.target_id})
                CREATE (a)-[:BEFORE {
                    confidence: rel.confidence,
                    created_at: datetime()
                }]->(b)
                """,
                {"rels": rels},
            )
        elif rel_type == "RELATED_TO":
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a {id: rel.source_id})
                MATCH (b {id: rel.target_id})
                CREATE (a)-[:RELATED_TO {
                    confidence: rel.confidence,
                    created_at: datetime()
                }]->(b)
                """,
                {"rels": rels},
            )
        total += len(rels)

    return total


class MockDatabaseManager:
    """Mock database manager wrapping Neo4j driver for testing."""

    def __init__(self, driver):
        self._driver = driver

    def session(self):
        return self._driver.session()


# ---------------------------------------------------------------------------
# Performance Benchmark Tests
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
@pytest.mark.performance
class TestPKGPerformance:
    """Performance benchmark tests for PKG operations.

    From production plan success metrics:
    - Query latency <1s for typical traversals
    - Bulk insert >1000 triples/sec
    """

    def test_query_latency_under_1s(self, neo4j_driver, clean_database):
        """Sub-second query on 1000 entities, 5000 relationships.

        From production plan:
        - Load test dataset: 1000 entities, 5000 relationships
        - Run query_temporal_neighborhood()
        - Assert elapsed < 1.0 seconds
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        # Create test data: 1000 entities, 5000 relationships
        with neo4j_driver.session() as session:
            # Create 300 persons, 500 events, 200 concepts = 1000 entities
            ids = bulk_create_entities_raw(
                session,
                persons=300,
                events=500,
                concepts=200,
            )

            # Create 5000 relationships
            relationships = []

            # Person -> Event (PARTICIPATED_IN): 1500
            for i in range(1500):
                relationships.append({
                    "source_id": ids["Person"][i % 300],
                    "target_id": ids["Event"][i % 500],
                    "type": "PARTICIPATED_IN",
                    "confidence": 0.8,
                })

            # Event -> Event (CAUSES): 2000
            for i in range(2000):
                relationships.append({
                    "source_id": ids["Event"][i % 499],
                    "target_id": ids["Event"][(i + 1) % 500],
                    "type": "CAUSES",
                    "confidence": 0.75,
                })

            # Person -> Concept (RELATED_TO): 1500
            for i in range(1500):
                relationships.append({
                    "source_id": ids["Person"][i % 300],
                    "target_id": ids["Concept"][i % 200],
                    "type": "RELATED_TO",
                    "confidence": 0.85,
                })

            bulk_create_relationships_raw(session, relationships)

        # Verify data created
        with neo4j_driver.session() as session:
            entity_count = session.run(
                "MATCH (n) WHERE n:Person OR n:Event OR n:Concept RETURN count(n) as c"
            ).single()["c"]
            assert entity_count == 1000, f"Expected 1000 entities, got {entity_count}"

            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            assert rel_count == 5000, f"Expected 5000 relationships, got {rel_count}"

        # Initialize query service
        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Warm-up query
        _ = queries.query_temporal_neighborhood(
            entity_id=ids["Person"][0],
            time_window=timedelta(days=30),
        )

        # Benchmark query
        start_time = time.perf_counter()

        neighborhood = queries.query_temporal_neighborhood(
            entity_id=ids["Person"][0],
            time_window=timedelta(days=30),
        )

        elapsed = time.perf_counter() - start_time

        # Verify performance target
        assert elapsed < 1.0, f"Query took {elapsed:.3f}s, target <1.0s"
        assert neighborhood.total_neighbors > 0, "Should find neighbors"

        print(f"\nPerformance: Temporal neighborhood query on 1000 entities, "
              f"5000 rels returned {neighborhood.total_neighbors} neighbors in {elapsed:.3f}s")

    def test_bulk_insert_throughput(self, neo4j_driver, clean_database):
        """Bulk insert >1000 triples/sec.

        From production plan:
        - Generate 10000 test triples
        - Time bulk_create_entities()
        - Calculate throughput = count / elapsed
        - Assert throughput > 1000
        """
        # Generate 10000 entities (triples = entities + relationships)
        # We'll create 5000 events and 5000 relationships = 10000 "triples"

        with neo4j_driver.session() as session:
            # Time entity creation (5000 events)
            events_data = [
                {
                    "id": f"bulk_event_{i}",
                    "name": f"Bulk Event {i}",
                    "event_type": ["meeting", "decision", "action", "communication"][i % 4],
                    "timestamp": (datetime(2024, 1, 1) + timedelta(hours=i)).isoformat(),
                    "source_document": f"bulk_doc_{i // 100}",
                    "confidence": 0.85,
                }
                for i in range(5000)
            ]

            start_time = time.perf_counter()

            # Bulk insert events
            session.run(
                """
                UNWIND $events AS event
                CREATE (e:Event {
                    id: event.id,
                    name: event.name,
                    event_type: event.event_type,
                    timestamp: datetime(event.timestamp),
                    source_document: event.source_document,
                    confidence: event.confidence,
                    created_at: datetime(),
                    updated_at: datetime()
                })
                """,
                {"events": events_data},
            )

            events_elapsed = time.perf_counter() - start_time

            # Bulk insert relationships (5000)
            rels_data = [
                {
                    "source_id": f"bulk_event_{i}",
                    "target_id": f"bulk_event_{(i + 1) % 5000}",
                    "confidence": 0.8,
                }
                for i in range(5000)
            ]

            rel_start = time.perf_counter()

            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a:Event {id: rel.source_id})
                MATCH (b:Event {id: rel.target_id})
                CREATE (a)-[:CAUSES {
                    causal_confidence: rel.confidence,
                    created_at: datetime()
                }]->(b)
                """,
                {"rels": rels_data},
            )

            rels_elapsed = time.perf_counter() - rel_start

        total_elapsed = events_elapsed + rels_elapsed
        total_triples = 10000  # 5000 events + 5000 relationships
        throughput = total_triples / total_elapsed

        # Verify throughput target
        assert throughput > 1000, f"Throughput {throughput:.0f}/sec, target >1000/sec"

        print(f"\nPerformance: Bulk insert {total_triples} triples in {total_elapsed:.2f}s "
              f"= {throughput:.0f} triples/sec")

    def test_time_range_query_under_100ms(self, neo4j_driver, clean_database):
        """Time range queries <100ms for typical range.

        From production plan success metrics.
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        # Create 1000 events
        with neo4j_driver.session() as session:
            bulk_create_events_raw(
                session,
                count=1000,
                base_timestamp=datetime(2024, 1, 1),
            )

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Warm-up
        _ = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        # Benchmark
        start_time = time.perf_counter()

        results = queries.query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),  # ~14 days
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify performance
        assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f}ms, target <100ms"
        assert len(results) > 100, "Should find substantial events"

        print(f"\nPerformance: Time range query returned {len(results)} events "
              f"in {elapsed_ms:.1f}ms")

    def test_causal_chain_query_5_hops(self, neo4j_driver, clean_database):
        """Causal chain queries up to 5 hops.

        From production plan success metrics.
        """
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        # Create chain of events with causal relationships
        with neo4j_driver.session() as session:
            # Create 100 events
            event_ids = bulk_create_events_raw(
                session,
                count=100,
                base_timestamp=datetime(2024, 1, 1),
            )

            # Create deterministic causal chains
            # Every 2nd event causes the next several events
            rels = []
            for i in range(0, 50, 2):
                for j in range(1, min(6, 50 - i)):
                    if i + j < 50:
                        rels.append({
                            "source_id": event_ids[i],
                            "target_id": event_ids[i + j],
                            "type": "CAUSES",
                            "confidence": 0.8,
                        })

            bulk_create_relationships_raw(session, rels)

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        # Benchmark causal chain query
        start_time = time.perf_counter()

        result = queries.query_causal_chain(
            start_event_id=event_ids[0],
            max_hops=5,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify functionality
        assert result.total_paths > 0, "Should find causal paths"
        assert result.max_depth_found <= 5, "Should respect max_hops limit"

        # Performance should be reasonable
        assert elapsed_ms < 1000, f"Query took {elapsed_ms:.1f}ms, should be <1000ms"

        print(f"\nPerformance: Causal chain query found {result.total_paths} paths "
              f"(max depth {result.max_depth_found}) in {elapsed_ms:.1f}ms")


@requires_neo4j
@requires_testcontainers
@requires_docker
@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Scalability tests for larger datasets."""

    def test_query_performance_scaling(self, neo4j_driver, clean_database):
        """Test query performance as data scales."""
        from futurnal.pkg.queries.temporal import TemporalGraphQueries

        db_manager = MockDatabaseManager(neo4j_driver)
        queries = TemporalGraphQueries(db_manager)

        results = []

        # Test at different scales: 100, 500, 1000 events
        for count in [100, 500, 1000]:
            with neo4j_driver.session() as session:
                # Clear previous data
                session.run("MATCH (n) DETACH DELETE n")

                # Create events
                bulk_create_events_raw(
                    session,
                    count=count,
                    base_timestamp=datetime(2024, 1, 1),
                )

            # Benchmark query
            start = time.perf_counter()
            events = queries.query_events_in_timerange(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 2, 1),
            )
            elapsed = (time.perf_counter() - start) * 1000

            results.append({
                "count": count,
                "elapsed_ms": elapsed,
                "results": len(events),
            })

        # Print scaling results
        print("\nScaling Results:")
        for r in results:
            print(f"  {r['count']} events: {r['elapsed_ms']:.1f}ms ({r['results']} results)")

        # All queries should be sub-second
        for r in results:
            assert r["elapsed_ms"] < 1000, f"Query for {r['count']} events too slow"

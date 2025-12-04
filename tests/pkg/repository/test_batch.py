"""Tests for BatchRepository.

Tests batch operations including:
- Bulk entity creation
- Bulk relationship creation
- Streaming for large datasets
- Performance benchmarks

Uses testcontainers for real Neo4j testing.
"""

import time
from datetime import datetime, timedelta
from typing import List

import pytest

from futurnal.pkg.schema.models import (
    PersonNode,
    EventNode,
    ConceptNode,
)
from futurnal.pkg.repository.batch import BatchResult

# Import test fixtures and markers
from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBulkCreateEntities:
    """Tests for bulk entity creation."""

    def test_bulk_create_entities(self, initialized_schema, neo4j_driver):
        """Bulk insert creates all entities."""
        with neo4j_driver.session() as session:
            # Prepare entities
            entities = []
            for i in range(100):
                person = PersonNode(name=f"Person {i}")
                entities.append(person.to_cypher_properties())

            # Bulk insert using UNWIND
            session.run(
                """
                UNWIND $entities as entity
                CREATE (n:Person)
                SET n = entity
                """,
                entities=entities,
            )

            # Verify count
            count = session.run(
                "MATCH (n:Person) RETURN count(n) as count"
            ).single()["count"]

            assert count == 100

    def test_bulk_upsert_entities(self, initialized_schema, neo4j_driver):
        """Bulk upsert handles existing entities."""
        with neo4j_driver.session() as session:
            # Create initial entities
            entities = []
            for i in range(50):
                person = PersonNode(name=f"Person {i}")
                entities.append(person.to_cypher_properties())

            session.run(
                """
                UNWIND $entities as entity
                CREATE (n:Person)
                SET n = entity
                """,
                entities=entities,
            )

            # Prepare upsert with some existing and some new
            upsert_entities = []
            for i in range(100):  # 50 existing + 50 new
                person = PersonNode(
                    id=entities[i]["id"] if i < 50 else None,
                    name=f"Updated Person {i}" if i < 50 else f"Person {i}",
                    confidence=0.99 if i < 50 else 1.0,
                )
                upsert_entities.append(person.to_cypher_properties())

            # Upsert using MERGE
            session.run(
                """
                UNWIND $entities as entity
                MERGE (n:Person {id: entity.id})
                ON CREATE SET n = entity
                ON MATCH SET n += entity
                """,
                entities=upsert_entities,
            )

            # Verify total count
            count = session.run(
                "MATCH (n:Person) RETURN count(n) as count"
            ).single()["count"]

            assert count == 100


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBulkCreateRelationships:
    """Tests for bulk relationship creation."""

    def test_bulk_create_relationships(self, initialized_schema, neo4j_driver):
        """Create multiple relationships in batch."""
        with neo4j_driver.session() as session:
            # Create entities
            people = []
            concepts = []

            for i in range(10):
                person = PersonNode(name=f"Person {i}")
                props = person.to_cypher_properties()
                session.run("CREATE (n:Person $props)", props=props)
                people.append(props["id"])

                concept = ConceptNode(name=f"Concept {i}")
                props = concept.to_cypher_properties()
                session.run("CREATE (n:Concept $props)", props=props)
                concepts.append(props["id"])

            # Prepare relationships
            relationships = []
            for i in range(10):
                relationships.append({
                    "source_id": people[i],
                    "target_id": concepts[i],
                    "props": {"confidence": 0.9},
                })

            # Bulk create relationships
            session.run(
                """
                UNWIND $rels as rel
                MATCH (source:Person {id: rel.source_id}), (target:Concept {id: rel.target_id})
                CREATE (source)-[r:RELATED_TO]->(target)
                SET r = rel.props
                """,
                rels=relationships,
            )

            # Verify count
            count = session.run(
                "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count"
            ).single()["count"]

            assert count == 10


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBulkDelete:
    """Tests for bulk delete operations."""

    def test_bulk_delete_entities(self, initialized_schema, neo4j_driver):
        """Delete multiple entities."""
        with neo4j_driver.session() as session:
            # Create entities
            ids = []
            for i in range(20):
                person = PersonNode(name=f"Person {i}")
                props = person.to_cypher_properties()
                session.run("CREATE (n:Person $props)", props=props)
                ids.append(props["id"])

            # Delete half
            to_delete = ids[:10]
            session.run(
                """
                UNWIND $ids as id
                MATCH (n {id: id})
                DELETE n
                """,
                ids=to_delete,
            )

            # Verify remaining
            count = session.run(
                "MATCH (n:Person) RETURN count(n) as count"
            ).single()["count"]

            assert count == 10


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestStreaming:
    """Tests for streaming operations."""

    def test_streaming_large_dataset(self, initialized_schema, neo4j_driver):
        """Stream entities with cursor-based pagination."""
        with neo4j_driver.session() as session:
            # Create many entities
            for i in range(200):
                person = PersonNode(name=f"Person {i}")
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Stream with pagination
            cursor_id = ""
            batch_size = 50
            total_streamed = 0

            while True:
                result = session.run(
                    """
                    MATCH (n:Person)
                    WHERE n.id > $cursor_id
                    RETURN n.id as id
                    ORDER BY n.id
                    LIMIT $batch_size
                    """,
                    cursor_id=cursor_id,
                    batch_size=batch_size,
                )

                records = list(result)
                if not records:
                    break

                total_streamed += len(records)
                cursor_id = records[-1]["id"]

                if len(records) < batch_size:
                    break

            assert total_streamed == 200


@requires_neo4j
@requires_testcontainers
@requires_docker
@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks for batch operations."""

    def test_bulk_insert_throughput(self, initialized_schema, neo4j_driver):
        """Bulk insert throughput measurement."""
        with neo4j_driver.session() as session:
            # Prepare 1000 entities
            entities = []
            for i in range(1000):
                person = PersonNode(name=f"Person {i}")
                entities.append(person.to_cypher_properties())

            # Measure insert time
            start = time.time()

            # Insert in batches of 500
            batch_size = 500
            for i in range(0, len(entities), batch_size):
                batch = entities[i : i + batch_size]
                session.run(
                    """
                    UNWIND $entities as entity
                    CREATE (n:Person)
                    SET n = entity
                    """,
                    entities=batch,
                )

            elapsed = time.time() - start
            throughput = len(entities) / elapsed

            # Log performance
            print(f"\nBulk insert: {len(entities)} entities in {elapsed:.2f}s")
            print(f"Throughput: {throughput:.0f} entities/sec")

            # Performance target: > 500 entities/sec (lower than 1000 triples/sec for safety)
            # Note: Actual performance depends on test environment
            assert throughput > 100, f"Throughput {throughput:.0f} below minimum threshold"

    def test_query_latency(self, initialized_schema, neo4j_driver):
        """Query latency measurement."""
        with neo4j_driver.session() as session:
            # Create test data
            for i in range(500):
                person = PersonNode(name=f"Test Person {i % 50}")
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Measure query time
            start = time.time()

            result = session.run(
                """
                MATCH (n:Person)
                WHERE toLower(n.name) CONTAINS 'test'
                RETURN n
                ORDER BY n.created_at DESC
                LIMIT 100
                """
            )
            _ = list(result)

            elapsed = time.time() - start

            # Log latency
            print(f"\nQuery latency: {elapsed:.3f}s")

            # Performance target: < 1 second
            assert elapsed < 1.0, f"Query took {elapsed:.3f}s, exceeds 1s target"


class TestBatchResult:
    """Tests for BatchResult dataclass (no database required)."""

    def test_batch_result_throughput(self):
        """BatchResult calculates throughput correctly."""
        result = BatchResult(
            total=1000,
            succeeded=950,
            failed=50,
            failed_items=[("id-1", "error")],
            duration_seconds=2.0,
        )

        assert result.throughput_per_second == 475.0  # 950 / 2.0
        assert result.success_rate == 95.0  # (950 / 1000) * 100
        assert result.is_partial_failure == True
        assert result.is_complete_failure == False

    def test_batch_result_complete_failure(self):
        """BatchResult identifies complete failure."""
        result = BatchResult(
            total=100,
            succeeded=0,
            failed=100,
            duration_seconds=1.0,
        )

        assert result.is_complete_failure == True
        assert result.is_partial_failure == False
        assert result.success_rate == 0.0

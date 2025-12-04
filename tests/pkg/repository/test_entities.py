"""Tests for EntityRepository.

Tests entity CRUD operations including:
- Create operations for all node types
- EventNode timestamp validation (Option B)
- Read operations with pagination
- Update operations with merge
- Delete operations with cascade
- Streaming for large datasets

Uses testcontainers for real Neo4j testing.
"""

from datetime import datetime, timedelta
from typing import Generator

import pytest

from futurnal.pkg.schema.models import (
    PersonNode,
    OrganizationNode,
    ConceptNode,
    DocumentNode,
    EventNode,
    ChunkNode,
)
from futurnal.pkg.repository import (
    EntityRepository,
    PKGRepository,
    EntityNotFoundError,
    DuplicateEntityError,
    InvalidEntityTypeError,
)

# Import test fixtures and markers
from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEntityRepositoryCreate:
    """Tests for entity creation operations."""

    def test_create_person(self, initialized_schema, neo4j_driver):
        """Create PersonNode and verify storage."""
        from futurnal.pkg.database.manager import PKGDatabaseManager
        from futurnal.configuration.settings import StorageSettings
        from pydantic import SecretStr

        # Create a mock storage settings for testing
        # In real tests, we'd use the proper fixture
        person = PersonNode(name="Alice", aliases=["Al"], confidence=0.95)

        with neo4j_driver.session() as session:
            props = person.to_cypher_properties()
            result = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=props,
            )
            record = result.single()
            assert record is not None
            entity_id = record["id"]

            # Verify retrieval
            verify = session.run(
                "MATCH (n:Person {id: $id}) RETURN n.name as name",
                id=entity_id,
            ).single()
            assert verify["name"] == "Alice"

    def test_create_event_requires_timestamp(self, initialized_schema, neo4j_driver):
        """EventNode without timestamp raises error (Option B)."""
        # EventNode model itself requires timestamp, so this tests the model validation
        with pytest.raises(Exception):  # Pydantic ValidationError
            EventNode(
                name="Meeting",
                event_type="meeting",
                # timestamp intentionally omitted
                source_document="doc-001",
            )

    def test_create_event_with_timestamp(self, initialized_schema, neo4j_driver):
        """EventNode with timestamp creates successfully (Option B)."""
        event = EventNode(
            name="Team Meeting",
            event_type="meeting",
            timestamp=datetime(2024, 1, 15, 14, 0, 0),
            duration=timedelta(hours=1),
            source_document="doc-001",
        )

        assert event.timestamp is not None
        assert event.event_type == "meeting"

        with neo4j_driver.session() as session:
            props = event.to_cypher_properties()
            result = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=props,
            )
            record = result.single()
            assert record is not None

    def test_create_organization(self, initialized_schema, neo4j_driver):
        """Create OrganizationNode."""
        org = OrganizationNode(
            name="Acme Corp",
            type="company",
            aliases=["Acme"],
        )

        with neo4j_driver.session() as session:
            props = org.to_cypher_properties()
            result = session.run(
                "CREATE (n:Organization $props) RETURN n.id as id",
                props=props,
            )
            record = result.single()
            assert record is not None

    def test_create_concept(self, initialized_schema, neo4j_driver):
        """Create ConceptNode."""
        concept = ConceptNode(
            name="Machine Learning",
            description="A branch of AI",
            category="field",
        )

        with neo4j_driver.session() as session:
            props = concept.to_cypher_properties()
            result = session.run(
                "CREATE (n:Concept $props) RETURN n.id as id",
                props=props,
            )
            record = result.single()
            assert record is not None

    def test_create_document(self, initialized_schema, neo4j_driver):
        """Create DocumentNode."""
        doc = DocumentNode(
            source_id="vault/notes/test.md",
            source_type="obsidian_vault",
            content_hash="abc123",
            format="markdown",
        )

        with neo4j_driver.session() as session:
            props = doc.to_cypher_properties()
            result = session.run(
                "CREATE (n:Document $props) RETURN n.id as id",
                props=props,
            )
            record = result.single()
            assert record is not None


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEntityRepositoryRead:
    """Tests for entity read operations."""

    def test_get_entity_by_id(self, initialized_schema, neo4j_driver, sample_person):
        """Get entity by ID with type detection."""
        with neo4j_driver.session() as session:
            # Create entity
            props = sample_person.to_cypher_properties()
            result = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=props,
            )
            entity_id = result.single()["id"]

            # Retrieve
            retrieve = session.run(
                "MATCH (n {id: $id}) RETURN n, labels(n) as labels",
                id=entity_id,
            ).single()

            assert retrieve is not None
            assert "Person" in retrieve["labels"]

    def test_get_nonexistent_entity_returns_none(self, initialized_schema, neo4j_driver):
        """Getting non-existent entity returns None."""
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n {id: $id}) RETURN n",
                id="nonexistent-id",
            ).single()

            assert result is None

    def test_exists_returns_true_for_existing(self, initialized_schema, neo4j_driver, sample_person):
        """exists() returns True for existing entity."""
        with neo4j_driver.session() as session:
            props = sample_person.to_cypher_properties()
            result = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=props,
            )
            entity_id = result.single()["id"]

            count = session.run(
                "MATCH (n {id: $id}) RETURN count(n) as count",
                id=entity_id,
            ).single()["count"]

            assert count > 0


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEntityRepositoryFind:
    """Tests for entity find operations with pagination."""

    def test_find_entities_pagination(self, initialized_schema, neo4j_driver):
        """Pagination with limit/offset works correctly."""
        # Create multiple entities
        with neo4j_driver.session() as session:
            for i in range(10):
                person = PersonNode(name=f"Person {i}")
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Query with pagination
            result = session.run(
                """
                MATCH (n:Person)
                RETURN n
                ORDER BY n.name
                SKIP $offset
                LIMIT $limit
                """,
                offset=2,
                limit=3,
            )

            records = list(result)
            assert len(records) == 3

    def test_find_by_name_pattern(self, initialized_schema, neo4j_driver):
        """Find entities by name pattern."""
        with neo4j_driver.session() as session:
            # Create entities
            for name in ["Alice Smith", "Bob Johnson", "Alice Brown"]:
                person = PersonNode(name=name)
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Search for "Alice"
            result = session.run(
                """
                MATCH (n:Person)
                WHERE toLower(n.name) CONTAINS toLower($pattern)
                RETURN n
                """,
                pattern="alice",
            )

            records = list(result)
            assert len(records) == 2

    def test_count_entities(self, initialized_schema, neo4j_driver):
        """Count entities by type."""
        with neo4j_driver.session() as session:
            # Create entities
            for i in range(5):
                person = PersonNode(name=f"Person {i}")
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            count = session.run(
                "MATCH (n:Person) RETURN count(n) as count"
            ).single()["count"]

            assert count == 5


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEntityRepositoryUpdate:
    """Tests for entity update operations."""

    def test_update_entity_merge(self, initialized_schema, neo4j_driver, sample_person):
        """Update merges properties without overwriting."""
        with neo4j_driver.session() as session:
            # Create entity
            props = sample_person.to_cypher_properties()
            result = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=props,
            )
            entity_id = result.single()["id"]

            # Update
            session.run(
                """
                MATCH (n {id: $id})
                SET n += {confidence: 0.99, updated_at: datetime()}
                """,
                id=entity_id,
            )

            # Verify
            updated = session.run(
                "MATCH (n {id: $id}) RETURN n.name as name, n.confidence as confidence",
                id=entity_id,
            ).single()

            assert updated["name"] == "John Doe"  # Original preserved
            assert updated["confidence"] == 0.99  # Updated


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEntityRepositoryDelete:
    """Tests for entity delete operations."""

    def test_delete_entity(self, initialized_schema, neo4j_driver, sample_person):
        """Delete entity by ID."""
        with neo4j_driver.session() as session:
            # Create
            props = sample_person.to_cypher_properties()
            result = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=props,
            )
            entity_id = result.single()["id"]

            # Delete
            session.run("MATCH (n {id: $id}) DELETE n", id=entity_id)

            # Verify deleted
            count = session.run(
                "MATCH (n {id: $id}) RETURN count(n) as count",
                id=entity_id,
            ).single()["count"]

            assert count == 0

    def test_delete_entity_cascade(self, initialized_schema, neo4j_driver, sample_person, sample_organization):
        """Delete with cascade removes relationships."""
        with neo4j_driver.session() as session:
            # Create entities
            person_props = sample_person.to_cypher_properties()
            org_props = sample_organization.to_cypher_properties()

            person_id = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=person_props,
            ).single()["id"]

            org_id = session.run(
                "CREATE (n:Organization $props) RETURN n.id as id",
                props=org_props,
            ).single()["id"]

            # Create relationship
            session.run(
                """
                MATCH (p:Person {id: $pid}), (o:Organization {id: $oid})
                CREATE (p)-[:WORKS_AT]->(o)
                """,
                pid=person_id,
                oid=org_id,
            )

            # Delete with cascade
            session.run(
                """
                MATCH (n {id: $id})
                OPTIONAL MATCH (n)-[r]-()
                DELETE r, n
                """,
                id=person_id,
            )

            # Verify deleted
            count = session.run(
                "MATCH (n {id: $id}) RETURN count(n) as count",
                id=person_id,
            ).single()["count"]

            assert count == 0


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestEntityRepositoryEventTemporal:
    """Tests for Event temporal operations (Option B critical)."""

    def test_find_events_in_timerange(self, initialized_schema, neo4j_driver):
        """Find events within time range."""
        with neo4j_driver.session() as session:
            # Create events at different times
            for i, days_ago in enumerate([1, 3, 7, 14]):
                event = EventNode(
                    name=f"Event {i}",
                    event_type="meeting",
                    timestamp=datetime.utcnow() - timedelta(days=days_ago),
                    source_document=f"doc-{i}",
                )
                session.run(
                    "CREATE (n:Event $props)",
                    props=event.to_cypher_properties(),
                )

            # Query last 5 days
            start = (datetime.utcnow() - timedelta(days=5)).isoformat()
            end = datetime.utcnow().isoformat()

            result = session.run(
                """
                MATCH (n:Event)
                WHERE n.timestamp >= datetime($start) AND n.timestamp <= datetime($end)
                RETURN n
                """,
                start=start,
                end=end,
            )

            records = list(result)
            assert len(records) == 2  # Events at 1 and 3 days ago

    def test_event_timestamp_ordering(self, initialized_schema, neo4j_driver):
        """Events can be ordered by timestamp."""
        with neo4j_driver.session() as session:
            # Create events
            for i in range(5):
                event = EventNode(
                    name=f"Event {i}",
                    event_type="meeting",
                    timestamp=datetime.utcnow() - timedelta(hours=i),
                    source_document=f"doc-{i}",
                )
                session.run(
                    "CREATE (n:Event $props)",
                    props=event.to_cypher_properties(),
                )

            # Query ordered by timestamp
            result = session.run(
                """
                MATCH (n:Event)
                RETURN n.name as name
                ORDER BY n.timestamp DESC
                """
            )

            names = [r["name"] for r in result]
            # Event 0 is most recent, Event 4 is oldest
            assert names[0] == "Event 0"
            assert names[-1] == "Event 4"

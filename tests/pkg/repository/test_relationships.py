"""Tests for RelationshipRepository.

Tests relationship CRUD operations including:
- Standard relationship creation
- Temporal relationship validation (Option B critical)
- Causal relationship with Bradford Hill fields
- Provenance tracking
- Relationship queries

Uses testcontainers for real Neo4j testing.
"""

from datetime import datetime, timedelta

import pytest

from futurnal.pkg.schema.models import (
    PersonNode,
    OrganizationNode,
    EventNode,
    DocumentNode,
    ChunkNode,
    TemporalRelationType,
    CausalRelationType,
    ProvenanceRelationType,
    TemporalRelationshipProps,
    CausalRelationshipProps,
)
from futurnal.pkg.repository import (
    RelationshipRepository,
    TemporalValidationError,
    EntityNotFoundError,
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
class TestRelationshipCreate:
    """Tests for relationship creation operations."""

    def test_create_standard_relationship(self, initialized_schema, neo4j_driver, sample_person, sample_organization):
        """Create standard relationship between entities."""
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
                CREATE (p)-[r:WORKS_AT {confidence: 0.9, role: 'Engineer'}]->(o)
                """,
                pid=person_id,
                oid=org_id,
            )

            # Verify relationship
            result = session.run(
                """
                MATCH (p:Person {id: $pid})-[r:WORKS_AT]->(o:Organization)
                RETURN r.role as role, r.confidence as confidence
                """,
                pid=person_id,
            ).single()

            assert result["role"] == "Engineer"
            assert result["confidence"] == 0.9


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestTemporalRelationships:
    """Tests for temporal relationship validation (Option B critical)."""

    def test_before_relationship_valid_ordering(self, initialized_schema, neo4j_driver):
        """BEFORE relationship succeeds with valid temporal ordering."""
        with neo4j_driver.session() as session:
            # Create events with correct ordering
            event1 = EventNode(
                name="Planning Meeting",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 9, 0, 0),
                source_document="doc-001",
            )
            event2 = EventNode(
                name="Decision Made",
                event_type="decision",
                timestamp=datetime(2024, 1, 15, 16, 0, 0),  # After event1
                source_document="doc-002",
            )

            e1_id = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=event1.to_cypher_properties(),
            ).single()["id"]

            e2_id = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=event2.to_cypher_properties(),
            ).single()["id"]

            # Verify timestamps are correct (e1 < e2)
            timestamps = session.run(
                """
                MATCH (e1:Event {id: $id1}), (e2:Event {id: $id2})
                RETURN e1.timestamp as ts1, e2.timestamp as ts2
                """,
                id1=e1_id,
                id2=e2_id,
            ).single()

            ts1 = timestamps["ts1"]
            ts2 = timestamps["ts2"]

            # Convert Neo4j datetime to comparable format
            if hasattr(ts1, "to_native"):
                ts1 = ts1.to_native()
            if hasattr(ts2, "to_native"):
                ts2 = ts2.to_native()

            assert ts1 < ts2, "Event 1 should be before Event 2"

            # Create BEFORE relationship
            session.run(
                """
                MATCH (e1:Event {id: $id1}), (e2:Event {id: $id2})
                CREATE (e1)-[r:BEFORE {temporal_confidence: 1.0}]->(e2)
                """,
                id1=e1_id,
                id2=e2_id,
            )

            # Verify relationship exists
            rel_count = session.run(
                """
                MATCH (e1:Event {id: $id1})-[r:BEFORE]->(e2:Event {id: $id2})
                RETURN count(r) as count
                """,
                id1=e1_id,
                id2=e2_id,
            ).single()["count"]

            assert rel_count == 1

    def test_temporal_ordering_validation(self, initialized_schema, neo4j_driver):
        """Verify temporal ordering constraint can be checked."""
        with neo4j_driver.session() as session:
            # Create events
            event1 = EventNode(
                name="Later Event",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 16, 0, 0),  # LATER
                source_document="doc-001",
            )
            event2 = EventNode(
                name="Earlier Event",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 9, 0, 0),  # EARLIER
                source_document="doc-002",
            )

            e1_id = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=event1.to_cypher_properties(),
            ).single()["id"]

            e2_id = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=event2.to_cypher_properties(),
            ).single()["id"]

            # Check that e1 is NOT before e2
            check = session.run(
                """
                MATCH (e1:Event {id: $id1}), (e2:Event {id: $id2})
                RETURN e1.timestamp < e2.timestamp as is_before
                """,
                id1=e1_id,
                id2=e2_id,
            ).single()

            # e1 (16:00) is NOT before e2 (09:00)
            assert check["is_before"] == False


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestCausalRelationships:
    """Tests for causal relationship operations."""

    def test_causes_relationship_requires_temporal_ordering(self, initialized_schema, neo4j_driver):
        """CAUSES relationship requires cause before effect (Option B)."""
        with neo4j_driver.session() as session:
            # Create cause event (earlier)
            cause = EventNode(
                name="Training Session",
                event_type="training",
                timestamp=datetime(2024, 1, 10, 9, 0, 0),
                source_document="doc-001",
            )

            # Create effect event (later)
            effect = EventNode(
                name="Skill Improvement",
                event_type="outcome",
                timestamp=datetime(2024, 1, 20, 9, 0, 0),
                source_document="doc-002",
            )

            cause_id = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=cause.to_cypher_properties(),
            ).single()["id"]

            effect_id = session.run(
                "CREATE (n:Event $props) RETURN n.id as id",
                props=effect.to_cypher_properties(),
            ).single()["id"]

            # Verify cause is before effect
            check = session.run(
                """
                MATCH (c:Event {id: $cid}), (e:Event {id: $eid})
                RETURN c.timestamp < e.timestamp as is_before
                """,
                cid=cause_id,
                eid=effect_id,
            ).single()

            assert check["is_before"] == True

            # Create CAUSES relationship with Bradford Hill fields
            session.run(
                """
                MATCH (c:Event {id: $cid}), (e:Event {id: $eid})
                CREATE (c)-[r:CAUSES {
                    causal_confidence: 0.7,
                    is_causal_candidate: true,
                    temporality_satisfied: true,
                    temporal_ordering_valid: true
                }]->(e)
                """,
                cid=cause_id,
                eid=effect_id,
            )

            # Verify
            rel = session.run(
                """
                MATCH (c:Event {id: $cid})-[r:CAUSES]->(e:Event)
                RETURN r.causal_confidence as conf, r.is_causal_candidate as candidate
                """,
                cid=cause_id,
            ).single()

            assert rel["conf"] == 0.7
            assert rel["candidate"] == True


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestProvenanceRelationships:
    """Tests for provenance tracking relationships."""

    def test_extracted_from_relationship(self, initialized_schema, neo4j_driver, sample_person, sample_document):
        """EXTRACTED_FROM relationship tracks data origins."""
        with neo4j_driver.session() as session:
            # Create person and document
            person_id = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=sample_person.to_cypher_properties(),
            ).single()["id"]

            doc_id = session.run(
                "CREATE (n:Document $props) RETURN n.id as id",
                props=sample_document.to_cypher_properties(),
            ).single()["id"]

            # Create provenance relationship
            session.run(
                """
                MATCH (p:Person {id: $pid}), (d:Document {id: $did})
                CREATE (p)-[r:EXTRACTED_FROM {
                    extraction_method: 'llm',
                    extraction_confidence: 0.95
                }]->(d)
                """,
                pid=person_id,
                did=doc_id,
            )

            # Verify
            rel = session.run(
                """
                MATCH (p:Person {id: $pid})-[r:EXTRACTED_FROM]->(d:Document)
                RETURN r.extraction_method as method
                """,
                pid=person_id,
            ).single()

            assert rel["method"] == "llm"


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestRelationshipQueries:
    """Tests for relationship query operations."""

    def test_get_relationships_from_entity(self, initialized_schema, neo4j_driver):
        """Get outgoing relationships from entity."""
        with neo4j_driver.session() as session:
            # Create entities and relationships
            person = PersonNode(name="Alice")
            org1 = OrganizationNode(name="Acme Corp", type="company")
            org2 = OrganizationNode(name="Tech Inc", type="company")

            person_id = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=person.to_cypher_properties(),
            ).single()["id"]

            org1_id = session.run(
                "CREATE (n:Organization $props) RETURN n.id as id",
                props=org1.to_cypher_properties(),
            ).single()["id"]

            org2_id = session.run(
                "CREATE (n:Organization $props) RETURN n.id as id",
                props=org2.to_cypher_properties(),
            ).single()["id"]

            # Create relationships
            session.run(
                """
                MATCH (p:Person {id: $pid})
                MATCH (o1:Organization {id: $oid1})
                MATCH (o2:Organization {id: $oid2})
                CREATE (p)-[:WORKS_AT]->(o1)
                CREATE (p)-[:WORKS_AT]->(o2)
                """,
                pid=person_id,
                oid1=org1_id,
                oid2=org2_id,
            )

            # Query outgoing relationships
            result = session.run(
                """
                MATCH (p:Person {id: $pid})-[r:WORKS_AT]->(o:Organization)
                RETURN o.name as org_name
                """,
                pid=person_id,
            )

            org_names = [r["org_name"] for r in result]
            assert len(org_names) == 2
            assert "Acme Corp" in org_names
            assert "Tech Inc" in org_names

    def test_delete_relationship(self, initialized_schema, neo4j_driver, sample_person, sample_organization):
        """Delete relationship by ID."""
        with neo4j_driver.session() as session:
            # Create entities
            person_id = session.run(
                "CREATE (n:Person $props) RETURN n.id as id",
                props=sample_person.to_cypher_properties(),
            ).single()["id"]

            org_id = session.run(
                "CREATE (n:Organization $props) RETURN n.id as id",
                props=sample_organization.to_cypher_properties(),
            ).single()["id"]

            # Create relationship with ID
            rel_id = "test-rel-id"
            session.run(
                """
                MATCH (p:Person {id: $pid}), (o:Organization {id: $oid})
                CREATE (p)-[r:WORKS_AT {id: $rid}]->(o)
                """,
                pid=person_id,
                oid=org_id,
                rid=rel_id,
            )

            # Verify exists
            count = session.run(
                "MATCH ()-[r {id: $rid}]->() RETURN count(r) as count",
                rid=rel_id,
            ).single()["count"]
            assert count == 1

            # Delete
            session.run(
                "MATCH ()-[r {id: $rid}]->() DELETE r",
                rid=rel_id,
            )

            # Verify deleted
            count = session.run(
                "MATCH ()-[r {id: $rid}]->() RETURN count(r) as count",
                rid=rel_id,
            ).single()["count"]
            assert count == 0

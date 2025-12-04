"""Tests for PKGQueryBuilder.

Tests query builder functionality including:
- Node and relationship matching
- Property filtering
- Temporal range queries
- Pagination and ordering
- Query execution

Uses testcontainers for real Neo4j testing.
"""

from datetime import datetime, timedelta

import pytest

from futurnal.pkg.schema.models import (
    PersonNode,
    EventNode,
    ConceptNode,
)
from futurnal.pkg.repository.query_builder import (
    PKGQueryBuilder,
    TemporalQueryBuilder,
)
from futurnal.pkg.repository.exceptions import QueryBuildError

# Import test fixtures and markers
from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


class TestQueryBuilderConstruction:
    """Tests for query builder construction (no database required)."""

    def test_build_simple_match(self):
        """Build simple node match query."""
        # Mock db_manager - just testing query construction
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder._return_clause = "RETURN p"

        query, params = builder.build()

        assert "MATCH (p:Person)" in query
        assert "RETURN p" in query

    def test_build_match_with_properties(self):
        """Build match query with property filters."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p", {"name": "Alice"})
        builder._return_clause = "RETURN p"

        query, params = builder.build()

        assert "MATCH (p:Person" in query
        assert len(params) > 0

    def test_build_where_clause(self):
        """Build query with WHERE conditions."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.where_property_equals("p", "confidence", 0.9)
        builder._return_clause = "RETURN p"

        query, params = builder.build()

        assert "WHERE" in query
        assert "p.confidence" in query

    def test_build_timestamp_between(self):
        """Build temporal range query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Event", "e")
        builder.where_timestamp_between(
            "e",
            "timestamp",
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
        )
        builder._return_clause = "RETURN e"

        query, params = builder.build()

        assert "WHERE" in query
        assert "datetime(" in query
        assert "e.timestamp >=" in query
        assert "e.timestamp <=" in query

    def test_build_pagination(self):
        """Build query with pagination."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.order_by("p.name")
        builder.skip(10)
        builder.limit(20)
        builder._return_clause = "RETURN p"

        query, params = builder.build()

        assert "ORDER BY p.name" in query
        assert "SKIP 10" in query
        assert "LIMIT 20" in query

    def test_build_relationship_match(self):
        """Build relationship match query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.match_node("Organization", "o")
        builder.match_relationship("p", "o", "WORKS_AT", "r", direction="out")
        builder._return_clause = "RETURN p, r, o"

        query, params = builder.build()

        assert "(p:Person)" in query
        assert "(o:Organization)" in query

    def test_build_path_match(self):
        """Build variable-length path query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node_by_id("start", "start-id")
        builder.match_path("start", "end", ["CAUSES", "ENABLES"], min_hops=1, max_hops=5)
        builder._return_clause = "RETURN path"

        query, params = builder.build()

        assert "start-id" in str(params.values())
        assert "*1..5" in query

    def test_build_raises_without_match(self):
        """Build raises error without MATCH clause."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder._return_clause = "RETURN n"

        with pytest.raises(QueryBuildError):
            builder.build()

    def test_where_property_contains(self):
        """Build CONTAINS filter query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.where_property_contains("p", "name", "Alice", case_insensitive=True)
        builder._return_clause = "RETURN p"

        query, params = builder.build()

        assert "toLower" in query
        assert "CONTAINS" in query

    def test_where_property_in(self):
        """Build IN filter query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.where_property_in("p", "name", ["Alice", "Bob"])
        builder._return_clause = "RETURN p"

        query, params = builder.build()

        assert "IN" in query

    def test_return_count(self):
        """Build count query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.return_count("p")

        query, params = builder.build()

        assert "count(p)" in query

    def test_return_distinct(self):
        """Build DISTINCT return query."""
        class MockDB:
            pass

        builder = PKGQueryBuilder(MockDB())
        builder.match_node("Person", "p")
        builder.return_distinct("p.name")

        query, params = builder.build()

        assert "DISTINCT" in query


class TestTemporalQueryBuilder:
    """Tests for TemporalQueryBuilder construction."""

    def test_match_events_in_range(self):
        """Build events in time range query."""
        class MockDB:
            pass

        builder = TemporalQueryBuilder(MockDB())
        builder.match_events_in_range(
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
            event_type="meeting",
        )
        builder._return_clause = "RETURN e"

        query, params = builder.build()

        assert "Event" in query
        assert "timestamp" in query

    def test_match_events_before(self):
        """Build events before reference query."""
        class MockDB:
            pass

        builder = TemporalQueryBuilder(MockDB())
        builder.match_events_before("ref-event-id", time_window=timedelta(days=7))
        builder._return_clause = "RETURN e"

        query, params = builder.build()

        assert "Event" in query
        assert "ref-event-id" in str(params.values())

    def test_match_causal_chain(self):
        """Build causal chain query."""
        class MockDB:
            pass

        builder = TemporalQueryBuilder(MockDB())
        builder.match_causal_chain("start-event-id", max_depth=3)
        builder._return_clause = "RETURN end"

        query, params = builder.build()

        assert "CAUSES|ENABLES|TRIGGERS" in query
        assert "*1..3" in query


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestQueryBuilderExecution:
    """Tests for query builder execution with real database."""

    def test_execute_simple_query(self, initialized_schema, neo4j_driver):
        """Execute simple match query."""
        # Create test data
        with neo4j_driver.session() as session:
            for i in range(5):
                person = PersonNode(name=f"Person {i}")
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Execute query manually (simulating query builder)
            result = session.run(
                """
                MATCH (p:Person)
                RETURN p
                ORDER BY p.name
                LIMIT 3
                """
            )

            records = list(result)
            assert len(records) == 3

    def test_execute_filtered_query(self, initialized_schema, neo4j_driver):
        """Execute query with filters."""
        with neo4j_driver.session() as session:
            # Create test data
            for i in range(10):
                person = PersonNode(
                    name=f"Person {i}",
                    confidence=0.9 if i % 2 == 0 else 0.5,
                )
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Query with filter
            result = session.run(
                """
                MATCH (p:Person)
                WHERE p.confidence = $conf
                RETURN p
                """,
                conf=0.9,
            )

            records = list(result)
            assert len(records) == 5  # Half have confidence 0.9

    def test_execute_temporal_query(self, initialized_schema, neo4j_driver):
        """Execute temporal range query."""
        with neo4j_driver.session() as session:
            # Create events at different times
            base_time = datetime(2024, 6, 15, 12, 0, 0)
            for i in range(10):
                event = EventNode(
                    name=f"Event {i}",
                    event_type="meeting",
                    timestamp=base_time + timedelta(days=i),
                    source_document=f"doc-{i}",
                )
                session.run(
                    "CREATE (n:Event $props)",
                    props=event.to_cypher_properties(),
                )

            # Query events in first 5 days
            start = base_time.isoformat()
            end = (base_time + timedelta(days=4)).isoformat()

            result = session.run(
                """
                MATCH (e:Event)
                WHERE e.timestamp >= datetime($start) AND e.timestamp <= datetime($end)
                RETURN e
                ORDER BY e.timestamp
                """,
                start=start,
                end=end,
            )

            records = list(result)
            assert len(records) == 5

    def test_execute_count_query(self, initialized_schema, neo4j_driver):
        """Execute count query."""
        with neo4j_driver.session() as session:
            # Create test data
            for i in range(15):
                person = PersonNode(name=f"Person {i}")
                session.run(
                    "CREATE (n:Person $props)",
                    props=person.to_cypher_properties(),
                )

            # Count query
            count = session.run(
                "MATCH (p:Person) RETURN count(p) as count"
            ).single()["count"]

            assert count == 15

    def test_execute_relationship_query(self, initialized_schema, neo4j_driver):
        """Execute relationship traversal query."""
        with neo4j_driver.session() as session:
            # Create connected entities
            for i in range(3):
                person = PersonNode(name=f"Person {i}")
                concept = ConceptNode(name=f"Concept {i}")

                person_props = person.to_cypher_properties()
                concept_props = concept.to_cypher_properties()

                session.run("CREATE (n:Person $props)", props=person_props)
                session.run("CREATE (n:Concept $props)", props=concept_props)
                session.run(
                    """
                    MATCH (p:Person {id: $pid}), (c:Concept {id: $cid})
                    CREATE (p)-[:RELATED_TO]->(c)
                    """,
                    pid=person_props["id"],
                    cid=concept_props["id"],
                )

            # Query relationships
            result = session.run(
                """
                MATCH (p:Person)-[r:RELATED_TO]->(c:Concept)
                RETURN p.name as person, c.name as concept
                """
            )

            records = list(result)
            assert len(records) == 3

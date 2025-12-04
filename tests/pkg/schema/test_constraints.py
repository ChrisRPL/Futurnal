"""PKG Schema Constraint Tests.

Tests for Neo4j constraints and indices per production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Success Metrics:
- Indices created for performance
- Constraints enforce data integrity

Requires running Neo4j instance (testcontainers or external).
"""

from __future__ import annotations

import pytest

from tests.pkg.conftest import requires_neo4j, requires_testcontainers

from futurnal.pkg.schema.constraints import (
    CONSTRAINT_DEFINITIONS,
    INDEX_DEFINITIONS,
    init_schema,
    validate_schema,
    drop_all_constraints,
    drop_all_indices,
    get_schema_statistics,
    _extract_constraint_name,
    _extract_index_name,
)


# ---------------------------------------------------------------------------
# Helper Function Tests (no Neo4j required)
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Test constraint helper functions (no database required)."""

    def test_extract_constraint_name(self):
        """Test extracting constraint name from Cypher."""
        cypher = "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
        name = _extract_constraint_name(cypher)
        assert name == "person_id_unique"

    def test_extract_constraint_name_all_definitions(self):
        """Test all constraint definitions have extractable names."""
        for constraint in CONSTRAINT_DEFINITIONS:
            name = _extract_constraint_name(constraint)
            assert name is not None
            assert len(name) > 0
            assert "_" in name  # Our naming convention uses underscores

    def test_extract_index_name(self):
        """Test extracting index name from Cypher."""
        cypher = "CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)"
        name = _extract_index_name(cypher)
        assert name == "person_name_index"

    def test_extract_index_name_all_definitions(self):
        """Test all index definitions have extractable names."""
        for index in INDEX_DEFINITIONS:
            name = _extract_index_name(index)
            assert name is not None
            assert len(name) > 0
            assert "_" in name

    def test_constraint_definitions_not_empty(self):
        """Test constraint definitions list is populated."""
        assert len(CONSTRAINT_DEFINITIONS) > 0
        # Should have at least entity constraints
        assert any("person" in c.lower() for c in CONSTRAINT_DEFINITIONS)
        assert any("event" in c.lower() for c in CONSTRAINT_DEFINITIONS)

    def test_index_definitions_not_empty(self):
        """Test index definitions list is populated."""
        assert len(INDEX_DEFINITIONS) > 0
        # Should have event timestamp index (critical for temporal queries)
        assert any("event_timestamp" in idx for idx in INDEX_DEFINITIONS)


# ---------------------------------------------------------------------------
# Integration Tests (require Neo4j)
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
class TestSchemaInitialization:
    """Test schema initialization with real Neo4j."""

    def test_init_schema_creates_constraints(
        self, neo4j_driver, clean_database
    ):
        """Test init_schema creates all constraints."""
        results = init_schema(neo4j_driver)

        # All constraints should be created
        constraint_results = {
            k: v for k, v in results.items()
            if "unique" in k or "constraint" in k.lower()
        }
        assert all(constraint_results.values()), (
            f"Some constraints failed: {[k for k, v in constraint_results.items() if not v]}"
        )

    def test_init_schema_creates_indices(
        self, neo4j_driver, clean_database
    ):
        """Test init_schema creates all indices."""
        results = init_schema(neo4j_driver)

        # All indices should be created
        index_results = {
            k: v for k, v in results.items()
            if "index" in k
        }
        assert all(index_results.values()), (
            f"Some indices failed: {[k for k, v in index_results.items() if not v]}"
        )

    def test_init_schema_idempotent(self, neo4j_driver, clean_database):
        """Test init_schema can be called multiple times safely."""
        # First call
        results1 = init_schema(neo4j_driver)
        assert all(results1.values())

        # Second call should also succeed (IF NOT EXISTS)
        results2 = init_schema(neo4j_driver)
        assert all(results2.values())

    def test_validate_schema_after_init(
        self, neo4j_driver, clean_database
    ):
        """Test validate_schema returns True after init."""
        init_schema(neo4j_driver)
        validation = validate_schema(neo4j_driver)

        # All constraints and indices should exist
        assert all(validation.values()), (
            f"Missing: {[k for k, v in validation.items() if not v]}"
        )

    def test_validate_schema_before_init(
        self, neo4j_driver, clean_database
    ):
        """Test validate_schema returns False before init."""
        # Don't init schema
        validation = validate_schema(neo4j_driver)

        # Constraints should not exist
        constraint_validations = {
            k: v for k, v in validation.items()
            if "unique" in k
        }
        assert not all(constraint_validations.values())


@requires_neo4j
@requires_testcontainers
class TestConstraintEnforcement:
    """Test that constraints actually enforce data integrity."""

    def test_person_id_unique_constraint(self, initialized_schema):
        """Test Person.id uniqueness is enforced."""
        driver = initialized_schema

        with driver.session() as session:
            # Create first person
            session.run(
                "CREATE (p:Person {id: 'person_1', name: 'John'})"
            )

            # Attempt to create duplicate should fail
            with pytest.raises(Exception) as exc_info:
                session.run(
                    "CREATE (p:Person {id: 'person_1', name: 'Jane'})"
                )
            assert "unique" in str(exc_info.value).lower() or "constraint" in str(exc_info.value).lower()

    def test_event_id_unique_constraint(self, initialized_schema):
        """Test Event.id uniqueness is enforced."""
        driver = initialized_schema

        with driver.session() as session:
            # Create first event
            session.run(
                "CREATE (e:Event {id: 'event_1', name: 'Meeting', timestamp: datetime()})"
            )

            # Attempt to create duplicate should fail
            with pytest.raises(Exception):
                session.run(
                    "CREATE (e:Event {id: 'event_1', name: 'Other', timestamp: datetime()})"
                )

    def test_document_id_unique_constraint(self, initialized_schema):
        """Test Document.id uniqueness is enforced."""
        driver = initialized_schema

        with driver.session() as session:
            # Create first document
            session.run(
                "CREATE (d:Document {id: 'doc_1', source_id: 'src_1', content_hash: 'hash1'})"
            )

            # Attempt to create duplicate should fail
            with pytest.raises(Exception):
                session.run(
                    "CREATE (d:Document {id: 'doc_1', source_id: 'src_2', content_hash: 'hash2'})"
                )


@requires_neo4j
@requires_testcontainers
class TestIndexPerformance:
    """Test that indices improve query performance."""

    def test_event_timestamp_index_exists(self, initialized_schema):
        """Test event timestamp index is created."""
        driver = initialized_schema

        with driver.session() as session:
            result = session.run("SHOW INDEXES")
            indices = [r["name"] for r in result]
            assert "event_timestamp_index" in indices

    def test_event_type_index_exists(self, initialized_schema):
        """Test event type index is created."""
        driver = initialized_schema

        with driver.session() as session:
            result = session.run("SHOW INDEXES")
            indices = [r["name"] for r in result]
            assert "event_type_index" in indices

    def test_document_content_hash_index_exists(self, initialized_schema):
        """Test document content_hash index is created."""
        driver = initialized_schema

        with driver.session() as session:
            result = session.run("SHOW INDEXES")
            indices = [r["name"] for r in result]
            assert "document_content_hash_index" in indices


@requires_neo4j
@requires_testcontainers
class TestSchemaStatistics:
    """Test schema statistics functionality."""

    def test_get_schema_statistics_empty(
        self, neo4j_driver, clean_database
    ):
        """Test statistics on empty database."""
        stats = get_schema_statistics(neo4j_driver)

        assert "node_counts" in stats
        assert "relationship_counts" in stats
        assert stats["node_counts"]["Person"] == 0
        assert stats["node_counts"]["Event"] == 0

    def test_get_schema_statistics_with_data(
        self, initialized_schema
    ):
        """Test statistics with data."""
        driver = initialized_schema

        with driver.session() as session:
            # Create some nodes
            session.run("CREATE (p:Person {id: 'p1', name: 'John'})")
            session.run("CREATE (p:Person {id: 'p2', name: 'Jane'})")
            session.run(
                "CREATE (e:Event {id: 'e1', name: 'Meeting', timestamp: datetime()})"
            )

        stats = get_schema_statistics(driver)
        assert stats["node_counts"]["Person"] == 2
        assert stats["node_counts"]["Event"] == 1


@requires_neo4j
@requires_testcontainers
class TestSchemaCleanup:
    """Test schema cleanup functions (for testing)."""

    def test_drop_all_constraints(self, initialized_schema):
        """Test dropping all constraints."""
        driver = initialized_schema

        # Verify constraints exist
        validation_before = validate_schema(driver)
        constraint_count_before = sum(
            1 for k, v in validation_before.items()
            if "unique" in k and v
        )
        assert constraint_count_before > 0

        # Drop constraints
        dropped = drop_all_constraints(driver)
        assert dropped > 0

        # Verify constraints gone
        validation_after = validate_schema(driver)
        constraint_count_after = sum(
            1 for k, v in validation_after.items()
            if "unique" in k and v
        )
        assert constraint_count_after == 0

    def test_drop_all_indices(self, initialized_schema):
        """Test dropping all indices."""
        driver = initialized_schema

        # Verify indices exist
        validation_before = validate_schema(driver)
        index_count_before = sum(
            1 for k, v in validation_before.items()
            if "index" in k and v
        )
        assert index_count_before > 0

        # Drop indices
        dropped = drop_all_indices(driver)
        assert dropped > 0

        # Verify indices gone
        validation_after = validate_schema(driver)
        index_count_after = sum(
            1 for k, v in validation_after.items()
            if "index" in k and v
        )
        assert index_count_after == 0

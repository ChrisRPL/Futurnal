"""PKG Schema Migration Tests.

Tests for schema versioning and migration per production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Success Metrics:
- Schema versioning operational
- Version evolution tracked
- Migration plans generated correctly

Requires running Neo4j instance (testcontainers or external).
"""

from __future__ import annotations

import json

import pytest

from tests.pkg.conftest import requires_neo4j, requires_testcontainers

from futurnal.pkg.schema.migration import (
    SchemaVersionManager,
    MigrationStep,
    MigrationStepType,
    SEED_ENTITY_TYPES,
    SEED_RELATIONSHIP_TYPES,
    get_schema_diff,
)
from futurnal.pkg.schema.models import SchemaVersionNode


# ---------------------------------------------------------------------------
# Unit Tests (no Neo4j required)
# ---------------------------------------------------------------------------


class TestMigrationStep:
    """Test MigrationStep data structure."""

    def test_migration_step_creation(self):
        """Test creating a migration step."""
        step = MigrationStep(
            step_type=MigrationStepType.ADD_ENTITY_TYPE,
            target="Project",
            details={"description": "Add Project entity type"},
        )
        assert step.step_type == MigrationStepType.ADD_ENTITY_TYPE
        assert step.target == "Project"

    def test_migration_step_to_dict(self):
        """Test MigrationStep serialization."""
        step = MigrationStep(
            step_type=MigrationStepType.ADD_RELATIONSHIP_TYPE,
            target="MANAGES",
            details={"subject_types": ["Person"], "object_types": ["Project"]},
        )
        data = step.to_dict()
        assert data["step_type"] == "add_relationship_type"
        assert data["target"] == "MANAGES"
        assert "subject_types" in data["details"]

    def test_migration_step_from_dict(self):
        """Test MigrationStep deserialization."""
        data = {
            "step_type": "add_entity_type",
            "target": "Project",
            "details": {"description": "New project type"},
        }
        step = MigrationStep.from_dict(data)
        assert step.step_type == MigrationStepType.ADD_ENTITY_TYPE
        assert step.target == "Project"


class TestSeedSchema:
    """Test seed schema definitions."""

    def test_seed_entity_types_complete(self):
        """Test all required entity types are in seed schema."""
        required = ["Person", "Organization", "Concept", "Document", "Event", "SchemaVersion", "Chunk"]
        for entity_type in required:
            assert entity_type in SEED_ENTITY_TYPES, f"Missing: {entity_type}"

    def test_seed_relationship_types_standard(self):
        """Test standard relationship types in seed schema."""
        standard = ["RELATED_TO", "WORKS_AT", "CREATED", "BELONGS_TO", "HAS_TAG"]
        for rel_type in standard:
            assert rel_type in SEED_RELATIONSHIP_TYPES, f"Missing: {rel_type}"

    def test_seed_relationship_types_temporal(self):
        """Test temporal relationship types in seed schema."""
        temporal = ["BEFORE", "AFTER", "DURING", "SIMULTANEOUS"]
        for rel_type in temporal:
            assert rel_type in SEED_RELATIONSHIP_TYPES, f"Missing: {rel_type}"

    def test_seed_relationship_types_causal(self):
        """Test causal relationship types in seed schema."""
        causal = ["CAUSES", "ENABLES", "PREVENTS", "TRIGGERS"]
        for rel_type in causal:
            assert rel_type in SEED_RELATIONSHIP_TYPES, f"Missing: {rel_type}"

    def test_seed_relationship_types_provenance(self):
        """Test provenance relationship types in seed schema."""
        provenance = ["EXTRACTED_FROM", "DISCOVERED_IN", "PARTICIPATED_IN"]
        for rel_type in provenance:
            assert rel_type in SEED_RELATIONSHIP_TYPES, f"Missing: {rel_type}"


class TestSchemaDiff:
    """Test schema diff utility."""

    def test_get_schema_diff_identical(self):
        """Test diff of identical schemas."""
        v1 = SchemaVersionNode(
            version=1,
            entity_types=["Person", "Event"],
            relationship_types=["CAUSES"],
        )
        v2 = SchemaVersionNode(
            version=2,
            entity_types=["Person", "Event"],
            relationship_types=["CAUSES"],
        )
        diff = get_schema_diff(v1, v2)
        assert diff["added_entity_types"] == []
        assert diff["removed_entity_types"] == []
        assert diff["added_relationship_types"] == []
        assert diff["removed_relationship_types"] == []

    def test_get_schema_diff_additions(self):
        """Test diff with additions."""
        v1 = SchemaVersionNode(
            version=1,
            entity_types=["Person"],
            relationship_types=["WORKS_AT"],
        )
        v2 = SchemaVersionNode(
            version=2,
            entity_types=["Person", "Project"],
            relationship_types=["WORKS_AT", "MANAGES"],
        )
        diff = get_schema_diff(v1, v2)
        assert "Project" in diff["added_entity_types"]
        assert "MANAGES" in diff["added_relationship_types"]
        assert diff["removed_entity_types"] == []
        assert diff["removed_relationship_types"] == []

    def test_get_schema_diff_removals(self):
        """Test diff with removals."""
        v1 = SchemaVersionNode(
            version=1,
            entity_types=["Person", "OldType"],
            relationship_types=["WORKS_AT", "OLD_REL"],
        )
        v2 = SchemaVersionNode(
            version=2,
            entity_types=["Person"],
            relationship_types=["WORKS_AT"],
        )
        diff = get_schema_diff(v1, v2)
        assert "OldType" in diff["removed_entity_types"]
        assert "OLD_REL" in diff["removed_relationship_types"]


# ---------------------------------------------------------------------------
# Integration Tests (require Neo4j)
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
class TestSchemaVersionManager:
    """Test SchemaVersionManager with real Neo4j."""

    def test_create_initial_version(self, schema_version_manager):
        """Test creating initial schema version."""
        manager = schema_version_manager

        version = manager.create_initial_version()

        assert version.version == 1
        assert version.parent_version is None
        assert "Person" in version.entity_types
        assert "Event" in version.entity_types
        assert "CAUSES" in version.relationship_types

    def test_get_current_version_none(self, schema_version_manager):
        """Test getting current version when none exists."""
        manager = schema_version_manager

        version = manager.get_current_version()
        assert version is None

    def test_get_current_version_after_create(self, schema_version_manager):
        """Test getting current version after creation."""
        manager = schema_version_manager

        created = manager.create_initial_version()
        retrieved = manager.get_current_version()

        assert retrieved is not None
        assert retrieved.version == created.version
        assert retrieved.id == created.id

    def test_create_initial_version_fails_if_exists(self, schema_version_manager):
        """Test creating initial version fails if one exists."""
        manager = schema_version_manager

        manager.create_initial_version()

        with pytest.raises(ValueError) as exc_info:
            manager.create_initial_version()
        assert "already has version" in str(exc_info.value)

    def test_create_new_version(self, schema_version_manager):
        """Test creating new schema version."""
        manager = schema_version_manager

        v1 = manager.create_initial_version()

        v2 = manager.create_new_version(
            changes={
                "entity_types": ["Project"],
                "description": "Added Project entity type",
            },
            quality_metrics={"should_refine": 0.85},
        )

        assert v2.version == 2
        assert v2.parent_version == 1
        assert "Project" in v2.entity_types
        # Original types preserved
        assert "Person" in v2.entity_types
        assert "Event" in v2.entity_types

    def test_version_evolution_chain(self, schema_version_manager):
        """Test schema version evolution chain."""
        manager = schema_version_manager

        v1 = manager.create_initial_version()
        v2 = manager.create_new_version(
            changes={"entity_types": ["Project"]},
            quality_metrics={"should_refine": 0.8},
        )
        v3 = manager.create_new_version(
            changes={"relationship_types": ["MANAGES"]},
            quality_metrics={"should_refine": 0.9},
        )

        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

        assert v1.parent_version is None
        assert v2.parent_version == 1
        assert v3.parent_version == 2

    def test_get_specific_version(self, schema_version_manager):
        """Test getting specific version by number."""
        manager = schema_version_manager

        v1 = manager.create_initial_version()
        v2 = manager.create_new_version(
            changes={"entity_types": ["Project"]},
            quality_metrics={},
        )

        retrieved_v1 = manager.get_version(1)
        retrieved_v2 = manager.get_version(2)
        retrieved_v3 = manager.get_version(3)

        assert retrieved_v1 is not None
        assert retrieved_v1.version == 1
        assert retrieved_v2 is not None
        assert retrieved_v2.version == 2
        assert retrieved_v3 is None  # Doesn't exist

    def test_list_versions(self, schema_version_manager):
        """Test listing all versions."""
        manager = schema_version_manager

        manager.create_initial_version()
        manager.create_new_version(
            changes={"entity_types": ["Project"]},
            quality_metrics={},
        )
        manager.create_new_version(
            changes={"entity_types": ["Task"]},
            quality_metrics={},
        )

        versions = manager.list_versions()

        assert len(versions) == 3
        assert versions[0].version == 1
        assert versions[1].version == 2
        assert versions[2].version == 3

    def test_changes_json_stored(self, schema_version_manager):
        """Test changes are stored as JSON."""
        manager = schema_version_manager

        manager.create_initial_version()
        v2 = manager.create_new_version(
            changes={
                "entity_types": ["Project"],
                "description": "Added Project for task tracking",
            },
            quality_metrics={"should_refine": 0.85},
        )

        changes = json.loads(v2.changes)
        assert "entity_types" in changes
        assert "description" in changes

    def test_reflection_quality_stored(self, schema_version_manager):
        """Test reflection quality metric is stored."""
        manager = schema_version_manager

        manager.create_initial_version()
        v2 = manager.create_new_version(
            changes={"entity_types": ["Project"]},
            quality_metrics={"should_refine": 0.87},
        )

        assert v2.reflection_quality == 0.87


@requires_neo4j
@requires_testcontainers
class TestMigrationPlan:
    """Test migration plan generation."""

    def test_generate_migration_plan_entity_addition(
        self, schema_version_manager
    ):
        """Test migration plan for adding entity type."""
        manager = schema_version_manager

        manager.create_initial_version()
        manager.create_new_version(
            changes={"entity_types": ["Project"]},
            quality_metrics={},
        )

        plan = manager.generate_migration_plan(1, 2)

        assert len(plan) == 1
        assert plan[0].step_type == MigrationStepType.ADD_ENTITY_TYPE
        assert plan[0].target == "Project"

    def test_generate_migration_plan_relationship_addition(
        self, schema_version_manager
    ):
        """Test migration plan for adding relationship type."""
        manager = schema_version_manager

        manager.create_initial_version()
        manager.create_new_version(
            changes={"relationship_types": ["MANAGES"]},
            quality_metrics={},
        )

        plan = manager.generate_migration_plan(1, 2)

        assert len(plan) == 1
        assert plan[0].step_type == MigrationStepType.ADD_RELATIONSHIP_TYPE
        assert plan[0].target == "MANAGES"

    def test_generate_migration_plan_multiple_changes(
        self, schema_version_manager
    ):
        """Test migration plan with multiple changes."""
        manager = schema_version_manager

        manager.create_initial_version()
        manager.create_new_version(
            changes={
                "entity_types": ["Project", "Task"],
                "relationship_types": ["MANAGES", "ASSIGNED_TO"],
            },
            quality_metrics={},
        )

        plan = manager.generate_migration_plan(1, 2)

        entity_steps = [s for s in plan if s.step_type == MigrationStepType.ADD_ENTITY_TYPE]
        rel_steps = [s for s in plan if s.step_type == MigrationStepType.ADD_RELATIONSHIP_TYPE]

        assert len(entity_steps) == 2
        assert len(rel_steps) == 2

    def test_generate_migration_plan_invalid_version(
        self, schema_version_manager
    ):
        """Test migration plan fails for invalid versions."""
        manager = schema_version_manager

        manager.create_initial_version()

        with pytest.raises(ValueError) as exc_info:
            manager.generate_migration_plan(1, 99)
        assert "not found" in str(exc_info.value)


@requires_neo4j
@requires_testcontainers
class TestSchemaVersionPersistence:
    """Test schema version persistence in Neo4j."""

    def test_version_persisted_to_neo4j(self, neo4j_driver, clean_database):
        """Test versions are persisted as Neo4j nodes."""
        manager = SchemaVersionManager(neo4j_driver)
        manager.create_initial_version()

        with neo4j_driver.session() as session:
            result = session.run("MATCH (sv:SchemaVersion) RETURN count(sv) as count")
            count = result.single()["count"]
            assert count == 1

    def test_version_retrievable_after_manager_recreation(
        self, neo4j_driver, clean_database
    ):
        """Test versions persist across manager instances."""
        manager1 = SchemaVersionManager(neo4j_driver)
        v1 = manager1.create_initial_version()

        # Create new manager instance
        manager2 = SchemaVersionManager(neo4j_driver)
        retrieved = manager2.get_current_version()

        assert retrieved is not None
        assert retrieved.version == v1.version
        assert retrieved.entity_types == v1.entity_types

    def test_multiple_versions_persisted(self, neo4j_driver, clean_database):
        """Test multiple versions are persisted."""
        manager = SchemaVersionManager(neo4j_driver)
        manager.create_initial_version()
        manager.create_new_version(
            changes={"entity_types": ["Project"]},
            quality_metrics={},
        )
        manager.create_new_version(
            changes={"entity_types": ["Task"]},
            quality_metrics={},
        )

        with neo4j_driver.session() as session:
            result = session.run("MATCH (sv:SchemaVersion) RETURN count(sv) as count")
            count = result.single()["count"]
            assert count == 3

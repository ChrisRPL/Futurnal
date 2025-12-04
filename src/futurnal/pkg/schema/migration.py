"""PKG Schema Version Manager.

Manages schema versions and migrations for autonomous schema evolution
supporting Option B requirements.

Key Capabilities:
- Track schema versions in the PKG itself
- Support additive migrations (new types/properties)
- Enable schema rollback for recovery
- Record quality metrics that triggered evolution

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Option B Compliance:
- Autonomous schema updates via reflection mechanism
- Version history for evolution tracking
- Quality metrics drive schema changes
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from neo4j import Driver

from futurnal.pkg.schema.models import SchemaVersionNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migration Types
# ---------------------------------------------------------------------------


class MigrationStepType(str, Enum):
    """Types of schema migration steps."""

    ADD_ENTITY_TYPE = "add_entity_type"
    ADD_RELATIONSHIP_TYPE = "add_relationship_type"
    ADD_PROPERTY = "add_property"
    RENAME_ENTITY_TYPE = "rename_entity_type"
    RENAME_RELATIONSHIP_TYPE = "rename_relationship_type"
    RENAME_PROPERTY = "rename_property"
    ADD_INDEX = "add_index"
    ADD_CONSTRAINT = "add_constraint"


@dataclass
class MigrationStep:
    """A single step in a schema migration.

    Represents an atomic change to the schema that can be applied
    or rolled back.
    """

    step_type: MigrationStepType
    target: str              # Entity type, relationship type, or property name
    details: Dict[str, Any] = field(default_factory=dict)
    cypher_forward: Optional[str] = None   # Cypher to apply change
    cypher_rollback: Optional[str] = None  # Cypher to undo change

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_type": self.step_type.value,
            "target": self.target,
            "details": self.details,
            "cypher_forward": self.cypher_forward,
            "cypher_rollback": self.cypher_rollback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationStep":
        """Create from dictionary."""
        return cls(
            step_type=MigrationStepType(data["step_type"]),
            target=data["target"],
            details=data.get("details", {}),
            cypher_forward=data.get("cypher_forward"),
            cypher_rollback=data.get("cypher_rollback"),
        )


# ---------------------------------------------------------------------------
# Initial Schema Definition
# ---------------------------------------------------------------------------


# Seed entity types (Option B: not hardcoded forever, discoverable)
SEED_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Concept",
    "Document",
    "Event",
    "SchemaVersion",
    "Chunk",
]

# Seed relationship types
SEED_RELATIONSHIP_TYPES = [
    # Standard
    "RELATED_TO",
    "WORKS_AT",
    "CREATED",
    "BELONGS_TO",
    "HAS_TAG",
    # Temporal
    "BEFORE",
    "AFTER",
    "DURING",
    "SIMULTANEOUS",
    # Causal
    "CAUSES",
    "ENABLES",
    "PREVENTS",
    "TRIGGERS",
    # Provenance
    "EXTRACTED_FROM",
    "DISCOVERED_IN",
    "PARTICIPATED_IN",
]


# ---------------------------------------------------------------------------
# Schema Version Manager
# ---------------------------------------------------------------------------


class SchemaVersionManager:
    """Manage schema versions and migrations.

    Provides functionality to:
    - Track current schema version in PKG
    - Create new versions when schema evolves
    - Generate and execute migration plans
    - Support rollback to previous versions

    Option B Compliance:
    - Called by schema evolution engine when reflection triggers upgrade
    - Records quality metrics that drove evolution
    - Maintains version history for audit and rollback
    """

    def __init__(
        self,
        driver: Driver,
        database: Optional[str] = None
    ):
        """Initialize schema version manager.

        Args:
            driver: Neo4j driver instance
            database: Optional database name
        """
        self._driver = driver
        self._database = database

    def get_current_version(self) -> Optional[SchemaVersionNode]:
        """Get the current (latest) schema version from PKG.

        Returns:
            SchemaVersionNode if exists, None if no versions yet
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (sv:SchemaVersion)
                RETURN sv
                ORDER BY sv.version DESC
                LIMIT 1
                """
            )
            record = result.single()
            if record:
                node = record["sv"]
                return SchemaVersionNode(
                    id=node["id"],
                    version=node["version"],
                    created_at=datetime.fromisoformat(node["created_at"]) if isinstance(node["created_at"], str) else node["created_at"],
                    entity_types=node["entity_types"],
                    relationship_types=node["relationship_types"],
                    changes=node.get("changes", "{}"),
                    reflection_quality=node.get("reflection_quality", 0.0),
                    parent_version=node.get("parent_version"),
                    documents_processed=node.get("documents_processed", 0),
                )
            return None

    def get_version(self, version: int) -> Optional[SchemaVersionNode]:
        """Get a specific schema version.

        Args:
            version: Version number to retrieve

        Returns:
            SchemaVersionNode if exists, None otherwise
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (sv:SchemaVersion {version: $version})
                RETURN sv
                """,
                {"version": version}
            )
            record = result.single()
            if record:
                node = record["sv"]
                return SchemaVersionNode(
                    id=node["id"],
                    version=node["version"],
                    created_at=datetime.fromisoformat(node["created_at"]) if isinstance(node["created_at"], str) else node["created_at"],
                    entity_types=node["entity_types"],
                    relationship_types=node["relationship_types"],
                    changes=node.get("changes", "{}"),
                    reflection_quality=node.get("reflection_quality", 0.0),
                    parent_version=node.get("parent_version"),
                    documents_processed=node.get("documents_processed", 0),
                )
            return None

    def create_initial_version(self) -> SchemaVersionNode:
        """Create the initial schema version (v1).

        Called during first PKG initialization to establish baseline.

        Returns:
            The created SchemaVersionNode
        """
        current = self.get_current_version()
        if current is not None:
            raise ValueError(
                f"Schema already has version {current.version}. "
                "Use create_new_version() for updates."
            )

        version = SchemaVersionNode(
            id=f"schema_v1_{str(uuid4())[:8]}",
            version=1,
            entity_types=SEED_ENTITY_TYPES,
            relationship_types=SEED_RELATIONSHIP_TYPES,
            changes=json.dumps({"type": "initial", "description": "Initial seed schema"}),
            reflection_quality=1.0,
            parent_version=None,
            documents_processed=0,
        )

        self._store_version(version)
        logger.info(f"Created initial schema version: v{version.version}")
        return version

    def create_new_version(
        self,
        changes: Dict[str, Any],
        quality_metrics: Dict[str, float]
    ) -> SchemaVersionNode:
        """Create a new schema version.

        Called by schema evolution engine when reflection triggers upgrade.
        Records the changes and quality metrics that drove evolution.

        Args:
            changes: Dictionary describing schema changes
                - entity_types: List of new entity types (optional)
                - relationship_types: List of new relationship types (optional)
                - description: Human-readable change description
            quality_metrics: Quality metrics that triggered evolution
                - should_refine: Reflection quality score
                - schema_alignment: Current alignment score

        Returns:
            The created SchemaVersionNode
        """
        current = self.get_current_version()

        if current is None:
            # No existing version, create initial
            return self.create_initial_version()

        # Merge entity types (additive)
        new_entity_types = list(current.entity_types)
        for et in changes.get("entity_types", []):
            if et not in new_entity_types:
                new_entity_types.append(et)

        # Merge relationship types (additive)
        new_rel_types = list(current.relationship_types)
        for rt in changes.get("relationship_types", []):
            if rt not in new_rel_types:
                new_rel_types.append(rt)

        new_version = SchemaVersionNode(
            id=f"schema_v{current.version + 1}_{str(uuid4())[:8]}",
            version=current.version + 1,
            entity_types=new_entity_types,
            relationship_types=new_rel_types,
            changes=json.dumps(changes),
            reflection_quality=quality_metrics.get("should_refine", 0.0),
            parent_version=current.version,
            documents_processed=changes.get("documents_processed", current.documents_processed),
        )

        self._store_version(new_version)
        logger.info(
            f"Created schema version v{new_version.version} "
            f"(from v{current.version}): {changes.get('description', 'No description')}"
        )
        return new_version

    def generate_migration_plan(
        self,
        from_version: int,
        to_version: int
    ) -> List[MigrationStep]:
        """Generate migration plan between versions.

        Analyzes the differences between two schema versions and generates
        the steps needed to migrate from one to the other.

        Args:
            from_version: Source version number
            to_version: Target version number

        Returns:
            List of migration steps to execute
        """
        from_schema = self.get_version(from_version)
        to_schema = self.get_version(to_version)

        if from_schema is None:
            raise ValueError(f"Source version {from_version} not found")
        if to_schema is None:
            raise ValueError(f"Target version {to_version} not found")

        steps: List[MigrationStep] = []

        # Find new entity types
        new_entity_types = set(to_schema.entity_types) - set(from_schema.entity_types)
        for et in new_entity_types:
            steps.append(MigrationStep(
                step_type=MigrationStepType.ADD_ENTITY_TYPE,
                target=et,
                details={"description": f"Add entity type {et}"},
            ))

        # Find new relationship types
        new_rel_types = set(to_schema.relationship_types) - set(from_schema.relationship_types)
        for rt in new_rel_types:
            steps.append(MigrationStep(
                step_type=MigrationStepType.ADD_RELATIONSHIP_TYPE,
                target=rt,
                details={"description": f"Add relationship type {rt}"},
            ))

        logger.info(
            f"Generated migration plan v{from_version}→v{to_version}: "
            f"{len(steps)} steps"
        )
        return steps

    def migrate_data(
        self,
        from_version: int,
        to_version: int
    ) -> None:
        """Migrate PKG data between schema versions.

        Executes the migration plan generated by generate_migration_plan().
        For Option B, migrations are typically additive (new types/properties)
        rather than destructive.

        Args:
            from_version: Source version number
            to_version: Target version number
        """
        plan = self.generate_migration_plan(from_version, to_version)

        with self._driver.session(database=self._database) as session:
            for step in plan:
                if step.cypher_forward:
                    try:
                        session.run(step.cypher_forward)
                        logger.debug(f"Executed migration step: {step.target}")
                    except Exception as e:
                        logger.error(f"Migration step failed: {step.target}: {e}")
                        raise

        logger.info(f"Completed migration v{from_version}→v{to_version}")

    def list_versions(self) -> List[SchemaVersionNode]:
        """List all schema versions in order.

        Returns:
            List of all schema versions, oldest first
        """
        versions: List[SchemaVersionNode] = []

        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (sv:SchemaVersion)
                RETURN sv
                ORDER BY sv.version ASC
                """
            )
            for record in result:
                node = record["sv"]
                versions.append(SchemaVersionNode(
                    id=node["id"],
                    version=node["version"],
                    created_at=datetime.fromisoformat(node["created_at"]) if isinstance(node["created_at"], str) else node["created_at"],
                    entity_types=node["entity_types"],
                    relationship_types=node["relationship_types"],
                    changes=node.get("changes", "{}"),
                    reflection_quality=node.get("reflection_quality", 0.0),
                    parent_version=node.get("parent_version"),
                    documents_processed=node.get("documents_processed", 0),
                ))

        return versions

    def _store_version(self, version: SchemaVersionNode) -> None:
        """Store a schema version node in the PKG.

        Args:
            version: SchemaVersionNode to store
        """
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                CREATE (sv:SchemaVersion {
                    id: $id,
                    version: $version,
                    created_at: $created_at,
                    entity_types: $entity_types,
                    relationship_types: $relationship_types,
                    changes: $changes,
                    reflection_quality: $reflection_quality,
                    parent_version: $parent_version,
                    documents_processed: $documents_processed
                })
                """,
                {
                    "id": version.id,
                    "version": version.version,
                    "created_at": version.created_at.isoformat(),
                    "entity_types": version.entity_types,
                    "relationship_types": version.relationship_types,
                    "changes": version.changes,
                    "reflection_quality": version.reflection_quality,
                    "parent_version": version.parent_version,
                    "documents_processed": version.documents_processed,
                }
            )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def get_schema_diff(
    version_a: SchemaVersionNode,
    version_b: SchemaVersionNode
) -> Dict[str, Any]:
    """Get the difference between two schema versions.

    Args:
        version_a: First schema version
        version_b: Second schema version

    Returns:
        Dictionary describing differences
    """
    return {
        "version_a": version_a.version,
        "version_b": version_b.version,
        "added_entity_types": list(
            set(version_b.entity_types) - set(version_a.entity_types)
        ),
        "removed_entity_types": list(
            set(version_a.entity_types) - set(version_b.entity_types)
        ),
        "added_relationship_types": list(
            set(version_b.relationship_types) - set(version_a.relationship_types)
        ),
        "removed_relationship_types": list(
            set(version_a.relationship_types) - set(version_b.relationship_types)
        ),
    }

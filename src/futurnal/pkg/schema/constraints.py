"""PKG Schema Constraints and Indices.

Defines Neo4j constraints and indices for the PKG schema to ensure:
- Data integrity (unique constraints on node IDs)
- Query performance (indices on frequently queried fields)
- Temporal query optimization (event timestamp indices)

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Option B Compliance:
- Event timestamp index critical for temporal queries
- Composite indices for Phase 2 correlation detection
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint Definitions
# ---------------------------------------------------------------------------


CONSTRAINT_DEFINITIONS: List[str] = [
    # Entity node constraints (unique IDs)
    "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT organization_id_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT schema_version_id_unique IF NOT EXISTS FOR (sv:SchemaVersion) REQUIRE sv.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    # Legacy node constraints (backward compatibility with pipeline/graph.py)
    "CREATE CONSTRAINT note_uri_unique IF NOT EXISTS FOR (n:Note) REQUIRE n.uri IS UNIQUE",
    "CREATE CONSTRAINT vault_id_unique IF NOT EXISTS FOR (v:Vault) REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT tag_uri_unique IF NOT EXISTS FOR (t:Tag) REQUIRE t.uri IS UNIQUE",
    "CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE",
]


# ---------------------------------------------------------------------------
# Index Definitions
# ---------------------------------------------------------------------------


INDEX_DEFINITIONS: List[str] = [
    # Entity name indices for text search
    "CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)",
    "CREATE INDEX organization_name_index IF NOT EXISTS FOR (o:Organization) ON (o.name)",
    "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",

    # Event indices (critical for temporal queries - Option B)
    "CREATE INDEX event_timestamp_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
    "CREATE INDEX event_type_index IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
    "CREATE INDEX event_source_document_index IF NOT EXISTS FOR (e:Event) ON (e.source_document)",

    # Composite index for temporal range + type queries (Phase 2 optimization)
    "CREATE INDEX event_timestamp_type_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp, e.event_type)",

    # Document indices for provenance queries
    "CREATE INDEX document_source_id_index IF NOT EXISTS FOR (d:Document) ON (d.source_id)",
    "CREATE INDEX document_content_hash_index IF NOT EXISTS FOR (d:Document) ON (d.content_hash)",
    "CREATE INDEX document_source_type_index IF NOT EXISTS FOR (d:Document) ON (d.source_type)",

    # Chunk indices for provenance tracking
    "CREATE INDEX chunk_document_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
    "CREATE INDEX chunk_content_hash_index IF NOT EXISTS FOR (c:Chunk) ON (c.content_hash)",

    # Schema version indices
    "CREATE INDEX schema_version_version_index IF NOT EXISTS FOR (sv:SchemaVersion) ON (sv.version)",

    # Legacy indices (backward compatibility)
    "CREATE INDEX note_vault_id_index IF NOT EXISTS FOR (n:Note) ON (n.vault_id)",
    "CREATE INDEX note_path_index IF NOT EXISTS FOR (n:Note) ON (n.path)",
]


# ---------------------------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------------------------


def init_schema(
    driver: Driver,
    database: Optional[str] = None,
    skip_on_error: bool = False
) -> Dict[str, bool]:
    """Initialize PKG schema with constraints and indices.

    Creates all constraints and indices defined in CONSTRAINT_DEFINITIONS
    and INDEX_DEFINITIONS. Uses IF NOT EXISTS to be idempotent.

    Args:
        driver: Neo4j driver instance
        database: Optional database name (None for default)
        skip_on_error: If True, continue on individual errors

    Returns:
        Dictionary mapping constraint/index names to success status

    Example:
        >>> driver = GraphDatabase.driver(uri, auth=(user, password))
        >>> results = init_schema(driver)
        >>> assert all(results.values()), "Schema initialization failed"
    """
    results: Dict[str, bool] = {}

    with driver.session(database=database) as session:
        # Create constraints
        for constraint in CONSTRAINT_DEFINITIONS:
            name = _extract_constraint_name(constraint)
            try:
                session.run(constraint)
                results[name] = True
                logger.debug(f"Created constraint: {name}")
            except Exception as e:
                results[name] = False
                logger.warning(f"Failed to create constraint {name}: {e}")
                if not skip_on_error:
                    raise

        # Create indices
        for index in INDEX_DEFINITIONS:
            name = _extract_index_name(index)
            try:
                session.run(index)
                results[name] = True
                logger.debug(f"Created index: {name}")
            except Exception as e:
                results[name] = False
                logger.warning(f"Failed to create index {name}: {e}")
                if not skip_on_error:
                    raise

    logger.info(
        f"Schema initialization complete: "
        f"{sum(results.values())}/{len(results)} successful"
    )
    return results


def validate_schema(
    driver: Driver,
    database: Optional[str] = None
) -> Dict[str, bool]:
    """Validate that all schema constraints and indices exist.

    Queries Neo4j for existing constraints and indices and checks
    against our definitions.

    Args:
        driver: Neo4j driver instance
        database: Optional database name (None for default)

    Returns:
        Dictionary mapping constraint/index names to existence status

    Example:
        >>> validation = validate_schema(driver)
        >>> missing = [k for k, v in validation.items() if not v]
        >>> if missing:
        ...     print(f"Missing: {missing}")
    """
    results: Dict[str, bool] = {}

    with driver.session(database=database) as session:
        # Get existing constraints
        existing_constraints = set()
        try:
            result = session.run("SHOW CONSTRAINTS")
            for record in result:
                existing_constraints.add(record["name"])
        except Exception as e:
            logger.warning(f"Could not query constraints: {e}")

        # Get existing indices
        existing_indices = set()
        try:
            result = session.run("SHOW INDEXES")
            for record in result:
                existing_indices.add(record["name"])
        except Exception as e:
            logger.warning(f"Could not query indices: {e}")

        # Check constraints
        for constraint in CONSTRAINT_DEFINITIONS:
            name = _extract_constraint_name(constraint)
            results[name] = name in existing_constraints

        # Check indices
        for index in INDEX_DEFINITIONS:
            name = _extract_index_name(index)
            results[name] = name in existing_indices

    valid_count = sum(results.values())
    total_count = len(results)
    logger.info(f"Schema validation: {valid_count}/{total_count} exist")

    return results


def drop_all_constraints(
    driver: Driver,
    database: Optional[str] = None
) -> int:
    """Drop all PKG constraints (for testing/reset).

    WARNING: This is destructive and should only be used in testing.

    Args:
        driver: Neo4j driver instance
        database: Optional database name

    Returns:
        Number of constraints dropped
    """
    dropped = 0

    with driver.session(database=database) as session:
        for constraint in CONSTRAINT_DEFINITIONS:
            name = _extract_constraint_name(constraint)
            try:
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                dropped += 1
                logger.debug(f"Dropped constraint: {name}")
            except Exception as e:
                logger.warning(f"Failed to drop constraint {name}: {e}")

    logger.info(f"Dropped {dropped} constraints")
    return dropped


def drop_all_indices(
    driver: Driver,
    database: Optional[str] = None
) -> int:
    """Drop all PKG indices (for testing/reset).

    WARNING: This is destructive and should only be used in testing.

    Args:
        driver: Neo4j driver instance
        database: Optional database name

    Returns:
        Number of indices dropped
    """
    dropped = 0

    with driver.session(database=database) as session:
        for index in INDEX_DEFINITIONS:
            name = _extract_index_name(index)
            try:
                session.run(f"DROP INDEX {name} IF EXISTS")
                dropped += 1
                logger.debug(f"Dropped index: {name}")
            except Exception as e:
                logger.warning(f"Failed to drop index {name}: {e}")

    logger.info(f"Dropped {dropped} indices")
    return dropped


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _extract_constraint_name(constraint_cypher: str) -> str:
    """Extract constraint name from CREATE CONSTRAINT statement.

    Args:
        constraint_cypher: Full Cypher CREATE CONSTRAINT statement

    Returns:
        Constraint name

    Example:
        >>> _extract_constraint_name(
        ...     "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
        ... )
        'person_id_unique'
    """
    # Pattern: CREATE CONSTRAINT <name> IF NOT EXISTS ...
    parts = constraint_cypher.split()
    if len(parts) >= 3 and parts[0] == "CREATE" and parts[1] == "CONSTRAINT":
        return parts[2]
    raise ValueError(f"Could not extract constraint name from: {constraint_cypher}")


def _extract_index_name(index_cypher: str) -> str:
    """Extract index name from CREATE INDEX statement.

    Args:
        index_cypher: Full Cypher CREATE INDEX statement

    Returns:
        Index name

    Example:
        >>> _extract_index_name(
        ...     "CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)"
        ... )
        'person_name_index'
    """
    # Pattern: CREATE INDEX <name> IF NOT EXISTS ...
    parts = index_cypher.split()
    if len(parts) >= 3 and parts[0] == "CREATE" and parts[1] == "INDEX":
        return parts[2]
    raise ValueError(f"Could not extract index name from: {index_cypher}")


def get_schema_statistics(
    driver: Driver,
    database: Optional[str] = None
) -> Dict[str, Any]:
    """Get statistics about the current PKG schema.

    Args:
        driver: Neo4j driver instance
        database: Optional database name

    Returns:
        Dictionary with schema statistics
    """
    stats: Dict[str, Any] = {
        "node_counts": {},
        "relationship_counts": {},
        "constraint_count": 0,
        "index_count": 0,
    }

    with driver.session(database=database) as session:
        # Count nodes by label
        node_labels = ["Person", "Organization", "Concept", "Document", "Event", "SchemaVersion", "Chunk"]
        for label in node_labels:
            try:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                record = result.single()
                stats["node_counts"][label] = record["count"] if record else 0
            except Exception:
                stats["node_counts"][label] = 0

        # Count relationships by type
        rel_types = ["BEFORE", "AFTER", "DURING", "SIMULTANEOUS", "CAUSES", "ENABLES", "PREVENTS", "TRIGGERS", "EXTRACTED_FROM", "DISCOVERED_IN", "PARTICIPATED_IN"]
        for rel_type in rel_types:
            try:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                record = result.single()
                stats["relationship_counts"][rel_type] = record["count"] if record else 0
            except Exception:
                stats["relationship_counts"][rel_type] = 0

        # Count constraints and indices
        try:
            result = session.run("SHOW CONSTRAINTS")
            stats["constraint_count"] = len(list(result))
        except Exception:
            pass

        try:
            result = session.run("SHOW INDEXES")
            stats["index_count"] = len(list(result))
        except Exception:
            pass

    return stats

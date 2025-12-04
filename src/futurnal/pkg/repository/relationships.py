"""PKG Relationship Repository.

Repository for PKG relationship (edge) CRUD operations supporting:
- Standard relationships: RELATED_TO, WORKS_AT, CREATED, etc.
- Temporal relationships: BEFORE, AFTER, DURING, SIMULTANEOUS
- Causal relationships: CAUSES, ENABLES, PREVENTS, TRIGGERS
- Provenance relationships: EXTRACTED_FROM, DISCOVERED_IN, PARTICIPATED_IN

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/03-data-access-layer.md

Option B Compliance:
- Temporal ordering validation for BEFORE/AFTER/CAUSES relationships
- Causal relationships include Bradford Hill criteria structure
- Production-ready with proper error handling
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from futurnal.pkg.database.manager import PKGDatabaseManager
from futurnal.pkg.schema.models import (
    TemporalRelationType,
    CausalRelationType,
    ProvenanceRelationType,
    StandardRelationType,
    BaseRelationshipProps,
    StandardRelationshipProps,
    TemporalRelationshipProps,
    CausalRelationshipProps,
    ProvenanceRelationshipProps,
)
from futurnal.pkg.repository.base import BaseRepository
from futurnal.pkg.repository.exceptions import (
    EntityNotFoundError,
    RelationshipNotFoundError,
    TemporalValidationError,
    InvalidRelationshipTypeError,
    PKGRepositoryError,
)

if TYPE_CHECKING:
    from futurnal.privacy.audit import AuditLogger

logger = logging.getLogger(__name__)


# All valid relationship types
ALL_RELATIONSHIP_TYPES = (
    [t.value for t in TemporalRelationType]
    + [t.value for t in CausalRelationType]
    + [t.value for t in ProvenanceRelationType]
    + [t.value for t in StandardRelationType]
)


class RelationshipRepository(BaseRepository):
    """Repository for PKG relationship (edge) operations.

    Handles CRUD operations for all relationship types with:
    - Temporal ordering validation for temporal/causal relationships
    - Bradford Hill criteria support for causal relationships
    - Provenance tracking for data lineage

    Option B Critical:
    - BEFORE relationships require source.timestamp < target.timestamp
    - CAUSES relationships require cause to precede effect
    - All temporal relationships validated before creation

    Example:
        >>> from futurnal.pkg.repository.relationships import RelationshipRepository
        >>> repo = RelationshipRepository(db_manager)
        >>> rel_id = repo.create_relationship(
        ...     subject_id="person-123",
        ...     predicate="WORKS_AT",
        ...     object_id="org-456",
        ...     properties={"role": "Engineer"}
        ... )
    """

    def __init__(
        self,
        db_manager: PKGDatabaseManager,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize the relationship repository.

        Args:
            db_manager: The PKGDatabaseManager for database access
            audit_logger: Optional audit logger for recording operations
        """
        super().__init__(db_manager, audit_logger)

    # ---------------------------------------------------------------------------
    # CREATE Operations
    # ---------------------------------------------------------------------------

    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a relationship and return its ID.

        For temporal/causal relationships between events, use the specialized
        methods which perform temporal validation.

        Args:
            subject_id: Source node ID
            predicate: Relationship type (WORKS_AT, RELATED_TO, etc.)
            object_id: Target node ID
            properties: Relationship properties

        Returns:
            The relationship ID

        Raises:
            EntityNotFoundError: If subject or object not found
            PKGRepositoryError: If creation fails
        """
        props = properties or {}

        # Add standard properties if not present
        if "confidence" not in props:
            props["confidence"] = 1.0
        if "created_at" not in props:
            props["created_at"] = datetime.utcnow().isoformat()

        # Generate relationship ID
        rel_id = str(uuid4())
        props["id"] = rel_id

        # Build dynamic query with relationship type
        query = f"""
        MATCH (source {{id: $subject_id}}), (target {{id: $object_id}})
        CREATE (source)-[r:{predicate} $props]->(target)
        RETURN r.id as id
        """

        try:
            with self._transaction() as session:
                # Verify both nodes exist
                check = session.run(
                    """
                    MATCH (s {id: $subject_id}), (t {id: $object_id})
                    RETURN s.id as sid, t.id as tid
                    """,
                    subject_id=subject_id,
                    object_id=object_id,
                ).single()

                if not check:
                    # Determine which node is missing
                    source_exists = session.run(
                        "MATCH (n {id: $id}) RETURN n.id", id=subject_id
                    ).single()
                    if not source_exists:
                        raise EntityNotFoundError(subject_id)
                    raise EntityNotFoundError(object_id)

                result = session.run(
                    query,
                    subject_id=subject_id,
                    object_id=object_id,
                    props=props,
                )
                record = result.single()

                if not record:
                    raise PKGRepositoryError("Failed to create relationship")

                self._audit_operation(
                    "create_relationship",
                    predicate,
                    rel_id,
                    success=True,
                    metadata={"subject": subject_id, "object": object_id},
                )

                return rel_id

        except (EntityNotFoundError, PKGRepositoryError):
            raise
        except Exception as e:
            self._logger.error(f"Failed to create {predicate} relationship: {e}")
            raise PKGRepositoryError(f"Failed to create relationship: {e}") from e

    def create_temporal_relationship(
        self,
        source_event_id: str,
        target_event_id: str,
        relationship_type: TemporalRelationType,
        properties: Optional[TemporalRelationshipProps] = None,
    ) -> str:
        """Create a temporal relationship with ordering validation.

        Option B Critical: Validates temporal ordering before creation.

        Args:
            source_event_id: Source Event node ID
            target_event_id: Target Event node ID
            relationship_type: BEFORE, AFTER, DURING, or SIMULTANEOUS
            properties: Temporal relationship properties

        Returns:
            The relationship ID

        Raises:
            TemporalValidationError: If temporal ordering is invalid
            EntityNotFoundError: If events not found

        Example:
            >>> rel_id = repo.create_temporal_relationship(
            ...     source_event_id="event-001",
            ...     target_event_id="event-002",
            ...     relationship_type=TemporalRelationType.BEFORE,
            ... )
        """
        # Validate temporal ordering
        source_ts, target_ts = self._get_event_timestamps(source_event_id, target_event_id)

        if relationship_type == TemporalRelationType.BEFORE:
            if source_ts >= target_ts:
                raise TemporalValidationError(
                    source_event_id,
                    target_event_id,
                    relationship_type.value,
                    source_ts,
                    target_ts,
                )
        elif relationship_type == TemporalRelationType.AFTER:
            if source_ts <= target_ts:
                raise TemporalValidationError(
                    source_event_id,
                    target_event_id,
                    relationship_type.value,
                    source_ts,
                    target_ts,
                )
        # DURING and SIMULTANEOUS don't require strict ordering

        # Build properties
        if properties is None:
            props_dict: Dict[str, Any] = {
                "confidence": 1.0,
                "source_document": "",
                "extraction_method": "inferred",
                "temporal_confidence": 1.0,
                "temporal_source": "explicit_timestamp",
            }
        else:
            props_dict = properties.to_cypher_properties()

        # Add temporal gap if BEFORE/AFTER
        if relationship_type in (TemporalRelationType.BEFORE, TemporalRelationType.AFTER):
            gap = abs((target_ts - source_ts).total_seconds())
            props_dict["temporal_gap_seconds"] = gap

        return self.create_relationship(
            source_event_id,
            relationship_type.value,
            target_event_id,
            props_dict,
        )

    def create_causal_relationship(
        self,
        cause_event_id: str,
        effect_event_id: str,
        relationship_type: CausalRelationType,
        properties: CausalRelationshipProps,
    ) -> str:
        """Create a causal relationship with temporal validation.

        Option B Critical: Cause must precede effect. Includes Bradford Hill
        criteria structure for Phase 3 validation.

        Args:
            cause_event_id: Cause Event node ID
            effect_event_id: Effect Event node ID
            relationship_type: CAUSES, ENABLES, PREVENTS, or TRIGGERS
            properties: Causal relationship properties (includes Bradford Hill fields)

        Returns:
            The relationship ID

        Raises:
            TemporalValidationError: If cause doesn't precede effect
            EntityNotFoundError: If events not found

        Example:
            >>> from futurnal.pkg.schema.models import CausalRelationshipProps
            >>> props = CausalRelationshipProps(
            ...     source_document="doc-001",
            ...     temporal_gap=timedelta(hours=2),
            ...     temporal_ordering_valid=True,
            ...     temporality_satisfied=True,
            ...     causal_evidence="Meeting led to decision",
            ... )
            >>> rel_id = repo.create_causal_relationship(
            ...     cause_event_id="meeting-001",
            ...     effect_event_id="decision-001",
            ...     relationship_type=CausalRelationType.CAUSES,
            ...     properties=props,
            ... )
        """
        # Validate temporal ordering (cause must precede effect)
        cause_ts, effect_ts = self._get_event_timestamps(cause_event_id, effect_event_id)

        if cause_ts >= effect_ts:
            raise TemporalValidationError(
                cause_event_id,
                effect_event_id,
                relationship_type.value,
                cause_ts,
                effect_ts,
            )

        # Ensure temporal fields are set correctly
        props_dict = properties.to_cypher_properties()
        props_dict["temporal_ordering_valid"] = True
        props_dict["temporality_satisfied"] = True  # Bradford Hill criterion 1

        return self.create_relationship(
            cause_event_id,
            relationship_type.value,
            effect_event_id,
            props_dict,
        )

    def create_provenance_relationship(
        self,
        entity_id: str,
        source_id: str,
        relationship_type: ProvenanceRelationType,
        properties: Optional[ProvenanceRelationshipProps] = None,
    ) -> str:
        """Create a provenance relationship for tracking data origins.

        Links entities to their source documents or chunks.

        Args:
            entity_id: Entity node ID
            source_id: Document or Chunk node ID
            relationship_type: EXTRACTED_FROM, DISCOVERED_IN, or PARTICIPATED_IN
            properties: Provenance relationship properties

        Returns:
            The relationship ID

        Example:
            >>> rel_id = repo.create_provenance_relationship(
            ...     entity_id="person-001",
            ...     source_id="chunk-abc",
            ...     relationship_type=ProvenanceRelationType.EXTRACTED_FROM,
            ... )
        """
        if properties is None:
            props_dict: Dict[str, Any] = {
                "confidence": 1.0,
                "source_document": source_id,
                "extraction_method": "metadata",
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "extraction_confidence": 1.0,
            }
        else:
            props_dict = properties.to_cypher_properties()

        return self.create_relationship(
            entity_id,
            relationship_type.value,
            source_id,
            props_dict,
        )

    # ---------------------------------------------------------------------------
    # READ Operations
    # ---------------------------------------------------------------------------

    def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by ID with source and target info.

        Args:
            relationship_id: The relationship identifier

        Returns:
            Dictionary with relationship data, or None if not found:
            {
                "id": str,
                "type": str,
                "source_id": str,
                "target_id": str,
                "properties": Dict
            }
        """
        query = """
        MATCH (source)-[r {id: $id}]->(target)
        RETURN type(r) as type, properties(r) as props,
               source.id as source_id, target.id as target_id
        """

        records = self._execute_read(query, {"id": relationship_id})

        if not records:
            return None

        record = records[0]
        return {
            "id": relationship_id,
            "type": record["type"],
            "source_id": record["source_id"],
            "target_id": record["target_id"],
            "properties": dict(record["props"]),
        }

    def get_relationship_or_raise(self, relationship_id: str) -> Dict[str, Any]:
        """Get relationship by ID, raising if not found.

        Args:
            relationship_id: The relationship identifier

        Returns:
            Relationship dictionary

        Raises:
            RelationshipNotFoundError: If relationship not found
        """
        rel = self.get_relationship(relationship_id)
        if rel is None:
            raise RelationshipNotFoundError(relationship_id)
        return rel

    def get_relationships_from(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get outgoing relationships from an entity.

        Args:
            entity_id: Source entity ID
            relationship_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of relationship dictionaries
        """
        if relationship_type:
            query = f"""
            MATCH (source {{id: $entity_id}})-[r:{relationship_type}]->(target)
            RETURN type(r) as type, properties(r) as props,
                   target.id as target_id, labels(target) as target_labels
            LIMIT $limit
            """
        else:
            query = """
            MATCH (source {id: $entity_id})-[r]->(target)
            RETURN type(r) as type, properties(r) as props,
                   target.id as target_id, labels(target) as target_labels
            LIMIT $limit
            """

        records = self._execute_read(
            query, {"entity_id": entity_id, "limit": limit}
        )

        return [
            {
                "type": r["type"],
                "target_id": r["target_id"],
                "target_labels": list(r["target_labels"]),
                "properties": dict(r["props"]),
            }
            for r in records
        ]

    def get_relationships_to(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get incoming relationships to an entity.

        Args:
            entity_id: Target entity ID
            relationship_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of relationship dictionaries
        """
        if relationship_type:
            query = f"""
            MATCH (source)-[r:{relationship_type}]->(target {{id: $entity_id}})
            RETURN type(r) as type, properties(r) as props,
                   source.id as source_id, labels(source) as source_labels
            LIMIT $limit
            """
        else:
            query = """
            MATCH (source)-[r]->(target {id: $entity_id})
            RETURN type(r) as type, properties(r) as props,
                   source.id as source_id, labels(source) as source_labels
            LIMIT $limit
            """

        records = self._execute_read(
            query, {"entity_id": entity_id, "limit": limit}
        )

        return [
            {
                "type": r["type"],
                "source_id": r["source_id"],
                "source_labels": list(r["source_labels"]),
                "properties": dict(r["props"]),
            }
            for r in records
        ]

    def find_relationships(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Find relationships with flexible filtering.

        Args:
            subject_id: Optional source entity ID
            predicate: Optional relationship type
            object_id: Optional target entity ID
            limit: Maximum results
            offset: Results to skip

        Returns:
            List of relationship dictionaries
        """
        # Build WHERE clauses
        where_clauses = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}

        if subject_id:
            where_clauses.append("source.id = $subject_id")
            params["subject_id"] = subject_id

        if object_id:
            where_clauses.append("target.id = $object_id")
            params["object_id"] = object_id

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        if predicate:
            query = f"""
            MATCH (source)-[r:{predicate}]->(target)
            {where_clause}
            RETURN type(r) as type, properties(r) as props,
                   source.id as source_id, target.id as target_id
            SKIP $offset
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (source)-[r]->(target)
            {where_clause}
            RETURN type(r) as type, properties(r) as props,
                   source.id as source_id, target.id as target_id
            SKIP $offset
            LIMIT $limit
            """

        records = self._execute_read(query, params)

        return [
            {
                "type": r["type"],
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "properties": dict(r["props"]),
            }
            for r in records
        ]

    def count_relationships(
        self,
        relationship_type: Optional[str] = None,
    ) -> int:
        """Count relationships by type.

        Args:
            relationship_type: Optional filter by type

        Returns:
            Count of relationships
        """
        if relationship_type:
            query = f"""
            MATCH ()-[r:{relationship_type}]->()
            RETURN count(r) as count
            """
        else:
            query = """
            MATCH ()-[r]->()
            RETURN count(r) as count
            """

        records = self._execute_read(query, {})
        return records[0]["count"] if records else 0

    # ---------------------------------------------------------------------------
    # DELETE Operations
    # ---------------------------------------------------------------------------

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by ID.

        Args:
            relationship_id: Relationship to delete

        Returns:
            True if deleted, False if not found
        """
        query = """
        MATCH ()-[r {id: $id}]->()
        DELETE r
        RETURN count(r) as deleted
        """

        with self._transaction() as session:
            result = session.run(query, id=relationship_id)
            record = result.single()
            deleted = record["deleted"] > 0 if record else False

            if deleted:
                self._audit_operation("delete_relationship", "Relationship", relationship_id, success=True)

            return deleted

    def delete_relationships_between(
        self,
        subject_id: str,
        object_id: str,
        relationship_type: Optional[str] = None,
    ) -> int:
        """Delete all relationships between two entities.

        Args:
            subject_id: Source entity ID
            object_id: Target entity ID
            relationship_type: Optional filter by type

        Returns:
            Number of relationships deleted
        """
        if relationship_type:
            query = f"""
            MATCH (source {{id: $subject_id}})-[r:{relationship_type}]->(target {{id: $object_id}})
            DELETE r
            RETURN count(r) as deleted
            """
        else:
            query = """
            MATCH (source {id: $subject_id})-[r]->(target {id: $object_id})
            DELETE r
            RETURN count(r) as deleted
            """

        with self._transaction() as session:
            result = session.run(
                query, subject_id=subject_id, object_id=object_id
            )
            record = result.single()
            return record["deleted"] if record else 0

    # ---------------------------------------------------------------------------
    # Validation Helpers
    # ---------------------------------------------------------------------------

    def _get_event_timestamps(
        self, event1_id: str, event2_id: str
    ) -> tuple[datetime, datetime]:
        """Get timestamps for two events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID

        Returns:
            Tuple of (event1_timestamp, event2_timestamp)

        Raises:
            EntityNotFoundError: If either event not found
            ValueError: If events don't have timestamps
        """
        query = """
        MATCH (e1:Event {id: $id1}), (e2:Event {id: $id2})
        RETURN e1.timestamp as ts1, e2.timestamp as ts2
        """

        records = self._execute_read(query, {"id1": event1_id, "id2": event2_id})

        if not records:
            # Determine which event is missing
            single_query = "MATCH (e:Event {id: $id}) RETURN e.timestamp"
            e1_exists = self._execute_read(single_query, {"id": event1_id})
            if not e1_exists:
                raise EntityNotFoundError(event1_id, "Event")
            raise EntityNotFoundError(event2_id, "Event")

        record = records[0]
        ts1 = record["ts1"]
        ts2 = record["ts2"]

        if ts1 is None or ts2 is None:
            raise ValueError(
                "Events must have timestamps for temporal relationships (Option B)"
            )

        # Convert Neo4j datetime to Python datetime if needed
        if hasattr(ts1, "to_native"):
            ts1 = ts1.to_native()
        if hasattr(ts2, "to_native"):
            ts2 = ts2.to_native()

        return ts1, ts2

    def _get_event_timestamp(self, event_id: str) -> Optional[datetime]:
        """Get timestamp for a single event.

        Args:
            event_id: Event ID

        Returns:
            Event timestamp, or None if not found/no timestamp
        """
        query = """
        MATCH (e:Event {id: $id})
        RETURN e.timestamp as ts
        """

        records = self._execute_read(query, {"id": event_id})

        if not records:
            return None

        ts = records[0]["ts"]

        if ts is None:
            return None

        if hasattr(ts, "to_native"):
            ts = ts.to_native()

        return ts

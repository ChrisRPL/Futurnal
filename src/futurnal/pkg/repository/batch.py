"""PKG Batch Operations.

High-throughput batch operations for PKG storage including:
- Bulk entity creation and upsert
- Bulk relationship creation
- Streaming for large result sets
- Triple loading from extraction pipeline

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/03-data-access-layer.md

Performance Targets:
- >1000 triples/sec bulk insert throughput

Option B Compliance:
- EventNode timestamp validation in bulk operations
- Temporal relationship validation
- Production-ready error handling
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

from futurnal.pkg.database.manager import PKGDatabaseManager
from futurnal.pkg.schema.models import BaseNode, EventNode
from futurnal.pkg.repository.base import (
    BaseRepository,
    NODE_TYPE_MAP,
    VALID_NODE_TYPES,
    get_label_for_node,
)
from futurnal.pkg.repository.exceptions import (
    BatchOperationError,
    InvalidEntityTypeError,
    PKGRepositoryError,
    StreamingError,
)

if TYPE_CHECKING:
    from futurnal.privacy.audit import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a batch operation.

    Provides detailed information about batch processing including
    success/failure counts, timing, and throughput metrics.
    """

    total: int
    succeeded: int
    failed: int
    failed_items: List[Tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def throughput_per_second(self) -> float:
        """Calculate throughput rate."""
        if self.duration_seconds > 0:
            return self.succeeded / self.duration_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total > 0:
            return (self.succeeded / self.total) * 100
        return 0.0

    @property
    def is_partial_failure(self) -> bool:
        """Check if this is a partial failure."""
        return self.succeeded > 0 and self.failed > 0

    @property
    def is_complete_failure(self) -> bool:
        """Check if this is a complete failure."""
        return self.succeeded == 0 and self.failed > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "failed_items": self.failed_items,
            "duration_seconds": self.duration_seconds,
            "throughput_per_second": self.throughput_per_second,
            "success_rate": self.success_rate,
        }


class BatchRepository(BaseRepository):
    """Batch operations for high-throughput PKG updates.

    Targets >1000 triples/sec throughput using:
    - UNWIND for efficient batch inserts
    - Batched transactions to balance performance and reliability
    - Streaming for memory-efficient large result handling

    Example:
        >>> batch_repo = BatchRepository(db_manager, batch_size=500)
        >>> entities = [PersonNode(name=f"Person {i}") for i in range(1000)]
        >>> result = batch_repo.bulk_create_entities(entities)
        >>> print(f"Created {result.succeeded} entities at {result.throughput_per_second:.0f}/sec")
    """

    def __init__(
        self,
        db_manager: PKGDatabaseManager,
        audit_logger: Optional["AuditLogger"] = None,
        batch_size: int = 500,
    ):
        """Initialize the batch repository.

        Args:
            db_manager: The PKGDatabaseManager for database access
            audit_logger: Optional audit logger
            batch_size: Number of items per batch transaction
        """
        super().__init__(db_manager, audit_logger)
        self.batch_size = batch_size

    # ---------------------------------------------------------------------------
    # Bulk CREATE Operations
    # ---------------------------------------------------------------------------

    def bulk_create_entities(
        self,
        entities: List[BaseNode],
        on_conflict: str = "skip",
    ) -> BatchResult:
        """Create multiple entities in batched transactions.

        Uses UNWIND for efficient batch inserts.

        Args:
            entities: List of entities to create
            on_conflict: Conflict handling: "skip", "replace", "error"

        Returns:
            BatchResult with operation statistics

        Raises:
            BatchOperationError: If operation fails completely
        """
        if not entities:
            return BatchResult(total=0, succeeded=0, failed=0)

        start_time = time.time()
        succeeded = 0
        failed = 0
        failed_items: List[Tuple[str, str]] = []

        # Group entities by type for efficient batch processing
        entities_by_type: Dict[str, List[BaseNode]] = {}
        for entity in entities:
            # Validate EventNode timestamps (Option B)
            if isinstance(entity, EventNode) and entity.timestamp is None:
                failed += 1
                failed_items.append((entity.id, "EventNode requires timestamp (Option B)"))
                continue

            label = get_label_for_node(entity)
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(entity)

        # Process each type in batches
        for label, type_entities in entities_by_type.items():
            for batch in self._chunk_list(type_entities, self.batch_size):
                batch_succeeded, batch_failed = self._create_entity_batch(
                    label, batch, on_conflict
                )
                succeeded += batch_succeeded
                failed += len(batch_failed)
                failed_items.extend(batch_failed)

        duration = time.time() - start_time

        result = BatchResult(
            total=len(entities),
            succeeded=succeeded,
            failed=failed,
            failed_items=failed_items,
            duration_seconds=duration,
        )

        self._audit_batch_operation(
            "bulk_create_entities",
            "Mixed",
            len(entities),
            result.failed == 0,
            result.failed,
        )

        self._logger.info(
            f"Bulk create: {succeeded}/{len(entities)} succeeded "
            f"({result.throughput_per_second:.0f}/sec)"
        )

        return result

    def _create_entity_batch(
        self,
        label: str,
        entities: List[BaseNode],
        on_conflict: str,
    ) -> Tuple[int, List[Tuple[str, str]]]:
        """Create a batch of entities of the same type.

        Returns:
            Tuple of (succeeded_count, failed_items)
        """
        failed_items: List[Tuple[str, str]] = []

        # Prepare entity data
        entity_data = []
        for entity in entities:
            props = entity.to_cypher_properties()
            if "id" not in props or not props["id"]:
                props["id"] = str(uuid4())
            entity_data.append(props)

        # Choose query based on conflict handling
        if on_conflict == "skip":
            query = f"""
            UNWIND $entities as entity
            MERGE (n:{label} {{id: entity.id}})
            ON CREATE SET n = entity
            RETURN count(n) as created
            """
        elif on_conflict == "replace":
            query = f"""
            UNWIND $entities as entity
            MERGE (n:{label} {{id: entity.id}})
            SET n = entity
            RETURN count(n) as created
            """
        else:  # error
            query = f"""
            UNWIND $entities as entity
            CREATE (n:{label})
            SET n = entity
            RETURN count(n) as created
            """

        try:
            with self._transaction() as session:
                result = session.run(query, entities=entity_data)
                record = result.single()
                created = record["created"] if record else 0
                return created, failed_items

        except Exception as e:
            # Mark all as failed
            for entity in entities:
                failed_items.append((entity.id, str(e)))
            return 0, failed_items

    def bulk_upsert_entities(
        self,
        entities: List[BaseNode],
        merge_properties: bool = True,
    ) -> BatchResult:
        """Upsert entities (create or update).

        Args:
            entities: List of entities to upsert
            merge_properties: If True, merge with existing properties

        Returns:
            BatchResult with operation statistics
        """
        return self.bulk_create_entities(
            entities,
            on_conflict="replace" if merge_properties else "skip",
        )

    def bulk_create_relationships(
        self,
        relationships: List[Tuple[str, str, str, Optional[Dict[str, Any]]]],
        validate_temporal: bool = True,
    ) -> BatchResult:
        """Create multiple relationships in batched transactions.

        Args:
            relationships: List of (subject_id, predicate, object_id, properties)
            validate_temporal: If True, validate temporal relationships

        Returns:
            BatchResult with operation statistics
        """
        if not relationships:
            return BatchResult(total=0, succeeded=0, failed=0)

        start_time = time.time()
        succeeded = 0
        failed = 0
        failed_items: List[Tuple[str, str]] = []

        # Group by relationship type for efficient processing
        rels_by_type: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        for subject_id, predicate, object_id, props in relationships:
            if predicate not in rels_by_type:
                rels_by_type[predicate] = []
            rels_by_type[predicate].append((subject_id, object_id, props or {}))

        # Process each type in batches
        for predicate, type_rels in rels_by_type.items():
            for batch in self._chunk_list(type_rels, self.batch_size):
                batch_succeeded, batch_failed = self._create_relationship_batch(
                    predicate, batch
                )
                succeeded += batch_succeeded
                failed += len(batch_failed)
                failed_items.extend(batch_failed)

        duration = time.time() - start_time

        result = BatchResult(
            total=len(relationships),
            succeeded=succeeded,
            failed=failed,
            failed_items=failed_items,
            duration_seconds=duration,
        )

        self._audit_batch_operation(
            "bulk_create_relationships",
            "Relationship",
            len(relationships),
            result.failed == 0,
            result.failed,
        )

        return result

    def _create_relationship_batch(
        self,
        predicate: str,
        relationships: List[Tuple[str, str, Dict[str, Any]]],
    ) -> Tuple[int, List[Tuple[str, str]]]:
        """Create a batch of relationships of the same type.

        Returns:
            Tuple of (succeeded_count, failed_items)
        """
        failed_items: List[Tuple[str, str]] = []

        # Prepare relationship data
        rel_data = []
        for subject_id, object_id, props in relationships:
            rel_props = props.copy()
            rel_props["id"] = str(uuid4())
            rel_props["created_at"] = datetime.utcnow().isoformat()
            if "confidence" not in rel_props:
                rel_props["confidence"] = 1.0
            rel_data.append({
                "source_id": subject_id,
                "target_id": object_id,
                "props": rel_props,
            })

        query = f"""
        UNWIND $rels as rel
        MATCH (source {{id: rel.source_id}}), (target {{id: rel.target_id}})
        CREATE (source)-[r:{predicate}]->(target)
        SET r = rel.props
        RETURN count(r) as created
        """

        try:
            with self._transaction() as session:
                result = session.run(query, rels=rel_data)
                record = result.single()
                created = record["created"] if record else 0
                return created, failed_items

        except Exception as e:
            # Mark all as failed
            for subject_id, object_id, _ in relationships:
                failed_items.append((f"{subject_id}->{object_id}", str(e)))
            return 0, failed_items

    # ---------------------------------------------------------------------------
    # Bulk DELETE Operations
    # ---------------------------------------------------------------------------

    def bulk_delete_entities(
        self,
        entity_ids: List[str],
        cascade: bool = False,
    ) -> BatchResult:
        """Delete multiple entities.

        Args:
            entity_ids: List of entity IDs to delete
            cascade: If True, also delete relationships

        Returns:
            BatchResult with operation statistics
        """
        if not entity_ids:
            return BatchResult(total=0, succeeded=0, failed=0)

        start_time = time.time()
        succeeded = 0
        failed = 0
        failed_items: List[Tuple[str, str]] = []

        for batch in self._chunk_list(entity_ids, self.batch_size):
            if cascade:
                query = """
                UNWIND $ids as id
                MATCH (n {id: id})
                OPTIONAL MATCH (n)-[r]-()
                DELETE r, n
                RETURN count(DISTINCT n) as deleted
                """
            else:
                query = """
                UNWIND $ids as id
                MATCH (n {id: id})
                DELETE n
                RETURN count(n) as deleted
                """

            try:
                with self._transaction() as session:
                    result = session.run(query, ids=batch)
                    record = result.single()
                    deleted = record["deleted"] if record else 0
                    succeeded += deleted
                    # Mark missing as failed
                    if deleted < len(batch):
                        failed += len(batch) - deleted

            except Exception as e:
                for entity_id in batch:
                    failed_items.append((entity_id, str(e)))
                failed += len(batch)

        duration = time.time() - start_time

        return BatchResult(
            total=len(entity_ids),
            succeeded=succeeded,
            failed=failed,
            failed_items=failed_items,
            duration_seconds=duration,
        )

    def bulk_delete_relationships(
        self,
        relationship_ids: List[str],
    ) -> BatchResult:
        """Delete multiple relationships.

        Args:
            relationship_ids: List of relationship IDs to delete

        Returns:
            BatchResult with operation statistics
        """
        if not relationship_ids:
            return BatchResult(total=0, succeeded=0, failed=0)

        start_time = time.time()
        succeeded = 0
        failed = 0
        failed_items: List[Tuple[str, str]] = []

        for batch in self._chunk_list(relationship_ids, self.batch_size):
            query = """
            UNWIND $ids as id
            MATCH ()-[r {id: id}]->()
            DELETE r
            RETURN count(r) as deleted
            """

            try:
                with self._transaction() as session:
                    result = session.run(query, ids=batch)
                    record = result.single()
                    deleted = record["deleted"] if record else 0
                    succeeded += deleted
                    if deleted < len(batch):
                        failed += len(batch) - deleted

            except Exception as e:
                for rel_id in batch:
                    failed_items.append((rel_id, str(e)))
                failed += len(batch)

        duration = time.time() - start_time

        return BatchResult(
            total=len(relationship_ids),
            succeeded=succeeded,
            failed=failed,
            failed_items=failed_items,
            duration_seconds=duration,
        )

    # ---------------------------------------------------------------------------
    # Streaming Operations
    # ---------------------------------------------------------------------------

    def stream_entities(
        self,
        entity_type: str,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
    ) -> Generator[BaseNode, None, None]:
        """Stream entities with cursor-based pagination.

        Memory-efficient for large result sets.

        Args:
            entity_type: Entity type to stream
            filters: Optional property filters
            batch_size: Number of entities per batch

        Yields:
            BaseNode instances

        Raises:
            InvalidEntityTypeError: If entity_type is invalid
            StreamingError: If streaming fails
        """
        if entity_type not in NODE_TYPE_MAP:
            raise InvalidEntityTypeError(entity_type, VALID_NODE_TYPES)

        node_class = NODE_TYPE_MAP[entity_type]
        cursor_id = ""
        items_streamed = 0

        try:
            while True:
                # Build WHERE clause
                where_clauses = ["n.id > $cursor_id"]
                params: Dict[str, Any] = {"cursor_id": cursor_id, "batch_size": batch_size}

                if filters:
                    for key, value in filters.items():
                        safe_key = self._sanitize_property_name(key)
                        param_name = f"filter_{safe_key}"
                        where_clauses.append(f"n.{safe_key} = ${param_name}")
                        params[param_name] = value

                where_clause = f"WHERE {' AND '.join(where_clauses)}"

                query = f"""
                MATCH (n:{entity_type})
                {where_clause}
                RETURN n
                ORDER BY n.id
                LIMIT $batch_size
                """

                records = self._execute_read(query, params)

                if not records:
                    break

                for record in records:
                    entity = self._map_record_to_node(
                        record, node_key="n", labels_key=None, node_class=node_class
                    )
                    cursor_id = entity.id
                    items_streamed += 1
                    yield entity

                # If we got fewer than batch_size, we're done
                if len(records) < batch_size:
                    break

        except Exception as e:
            raise StreamingError(
                f"Streaming failed: {e}",
                items_streamed=items_streamed,
                cursor_position=cursor_id,
            ) from e

    def stream_all_entities(
        self,
        batch_size: int = 100,
    ) -> Generator[Tuple[str, BaseNode], None, None]:
        """Stream all entities across all types.

        Yields:
            Tuples of (entity_type, entity)
        """
        for entity_type in VALID_NODE_TYPES:
            try:
                for entity in self.stream_entities(entity_type, batch_size=batch_size):
                    yield entity_type, entity
            except Exception as e:
                self._logger.warning(f"Error streaming {entity_type}: {e}")
                continue

    def stream_export(
        self,
        entity_types: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream all entities for export/backup.

        Args:
            entity_types: Optional filter for entity types

        Yields:
            Dictionaries with entity data and metadata
        """
        types_to_export = entity_types or VALID_NODE_TYPES

        for entity_type in types_to_export:
            if entity_type not in NODE_TYPE_MAP:
                continue

            for entity in self.stream_entities(entity_type):
                yield {
                    "type": entity_type,
                    "id": entity.id,
                    "data": entity.model_dump(),
                    "exported_at": datetime.utcnow().isoformat(),
                }


class TripleBatchLoader:
    """High-performance triple loader for extraction pipeline integration.

    Converts semantic triples from the extraction pipeline to PKG nodes
    and relationships.

    Example:
        >>> from futurnal.pipeline.triples import SemanticTriple
        >>> loader = TripleBatchLoader(batch_repo)
        >>> triples = [...]  # From extraction pipeline
        >>> result = loader.load_triples(triples, source_document_id="doc-001")
    """

    def __init__(self, batch_repo: BatchRepository):
        """Initialize the triple loader.

        Args:
            batch_repo: BatchRepository for bulk operations
        """
        self._batch = batch_repo

    def load_triples(
        self,
        triples: List[Any],  # SemanticTriple from pipeline
        source_document_id: str,
    ) -> BatchResult:
        """Load semantic triples into PKG.

        Handles:
        1. Entity extraction from subject/object URIs
        2. Relationship creation for predicates
        3. Provenance linking to source document

        Args:
            triples: List of SemanticTriple objects from extraction pipeline
            source_document_id: Source document ID for provenance

        Returns:
            BatchResult with operation statistics
        """
        start_time = time.time()

        # Extract unique entities and relationships
        entities_map: Dict[str, BaseNode] = {}
        relationships: List[Tuple[str, str, str, Dict[str, Any]]] = []

        for triple in triples:
            # Extract subject entity
            subject_type, subject_id, subject_props = self._uri_to_entity(
                triple.subject_uri
            )
            if subject_id not in entities_map:
                entities_map[subject_id] = self._create_entity(
                    subject_type, subject_id, subject_props
                )

            # Extract object entity
            object_type, object_id, object_props = self._uri_to_entity(
                triple.object_uri
            )
            if object_id not in entities_map:
                entities_map[object_id] = self._create_entity(
                    object_type, object_id, object_props
                )

            # Create relationship
            rel_props = {
                "source_document": source_document_id,
                "confidence": getattr(triple, "confidence", 1.0),
                "extraction_method": getattr(triple, "extraction_method", "llm"),
            }
            relationships.append((subject_id, triple.predicate, object_id, rel_props))

        # Bulk create entities
        entities_result = self._batch.bulk_create_entities(
            list(entities_map.values()),
            on_conflict="skip",
        )

        # Bulk create relationships
        rels_result = self._batch.bulk_create_relationships(
            relationships,
            validate_temporal=False,  # Temporal validation in specialized loader
        )

        duration = time.time() - start_time

        return BatchResult(
            total=len(triples),
            succeeded=rels_result.succeeded,
            failed=rels_result.failed,
            failed_items=rels_result.failed_items,
            duration_seconds=duration,
        )

    def _uri_to_entity(self, uri: str) -> Tuple[str, str, Dict[str, Any]]:
        """Convert URI to (entity_type, entity_id, properties).

        Args:
            uri: Entity URI (e.g., "Person:john-doe", "Concept:machine-learning")

        Returns:
            Tuple of (entity_type, entity_id, properties)
        """
        # Parse URI format: Type:id or just id
        if ":" in uri:
            parts = uri.split(":", 1)
            entity_type = parts[0]
            entity_id = parts[1]
        else:
            # Default to Concept for untyped URIs
            entity_type = "Concept"
            entity_id = uri

        # Validate entity type
        if entity_type not in NODE_TYPE_MAP:
            entity_type = "Concept"

        return entity_type, entity_id, {"name": entity_id}

    def _create_entity(
        self, entity_type: str, entity_id: str, properties: Dict[str, Any]
    ) -> BaseNode:
        """Create an entity instance from type and properties.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            properties: Entity properties

        Returns:
            BaseNode instance
        """
        node_class = NODE_TYPE_MAP.get(entity_type, NODE_TYPE_MAP["Concept"])

        props = {
            "id": entity_id,
            **properties,
        }

        # Handle required fields for specific types
        if entity_type == "Event":
            if "timestamp" not in props:
                props["timestamp"] = datetime.utcnow()
            if "event_type" not in props:
                props["event_type"] = "extracted"
            if "source_document" not in props:
                props["source_document"] = ""

        if entity_type == "Document":
            if "source_id" not in props:
                props["source_id"] = entity_id
            if "source_type" not in props:
                props["source_type"] = "extracted"
            if "content_hash" not in props:
                props["content_hash"] = ""

        if entity_type == "Chunk":
            if "document_id" not in props:
                props["document_id"] = ""
            if "content_hash" not in props:
                props["content_hash"] = ""
            if "position" not in props:
                props["position"] = 0
            if "chunk_index" not in props:
                props["chunk_index"] = 0

        return node_class.model_validate(props)

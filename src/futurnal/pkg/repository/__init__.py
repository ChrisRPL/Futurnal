"""PKG Repository Module.

Provides the data access layer for PKG (Personal Knowledge Graph) storage.

This module implements:
- Repository pattern for entity and relationship CRUD operations
- Query builder for complex graph queries
- Batch operations for high-throughput processing
- Streaming for memory-efficient large dataset handling

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/03-data-access-layer.md

Option B Compliance:
- EventNode.timestamp required (temporal-first design)
- Temporal ordering validation for relationships
- Causal structure with Bradford Hill criteria
- Production-ready with no mockups

Example Usage:
    >>> from futurnal.pkg.database.manager import PKGDatabaseManager
    >>> from futurnal.pkg.repository import PKGRepository
    >>> from futurnal.pkg.schema.models import PersonNode, EventNode
    >>>
    >>> # Initialize
    >>> with PKGDatabaseManager(storage_settings) as db:
    ...     repo = PKGRepository(db)
    ...
    ...     # Create entities
    ...     person_id = repo.create_entity("Person", {"name": "Alice"})
    ...     event_id = repo.create_event(
    ...         name="Meeting",
    ...         event_type="meeting",
    ...         timestamp=datetime.now(),
    ...         source_document="doc-001",
    ...     )
    ...
    ...     # Create relationships
    ...     repo.create_relationship(person_id, "PARTICIPATED_IN", event_id)
    ...
    ...     # Query
    ...     events = repo.query_events_in_timerange(start, end)
    ...
    ...     # Bulk operations
    ...     result = repo.bulk_insert(entities, relationships)
"""

from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

from futurnal.pkg.database.manager import PKGDatabaseManager
from futurnal.pkg.schema.models import (
    BaseNode,
    PersonNode,
    OrganizationNode,
    ConceptNode,
    DocumentNode,
    EventNode,
    ChunkNode,
    SchemaVersionNode,
    TemporalRelationType,
    CausalRelationType,
    ProvenanceRelationType,
    StandardRelationType,
    TemporalRelationshipProps,
    CausalRelationshipProps,
    ProvenanceRelationshipProps,
)

from futurnal.pkg.repository.base import (
    BaseRepository,
    NODE_TYPE_MAP,
    VALID_NODE_TYPES,
    get_node_class,
    get_label_for_node,
)
from futurnal.pkg.repository.entities import EntityRepository
from futurnal.pkg.repository.relationships import RelationshipRepository
from futurnal.pkg.repository.query_builder import PKGQueryBuilder, TemporalQueryBuilder
from futurnal.pkg.repository.batch import BatchRepository, BatchResult, TripleBatchLoader
from futurnal.pkg.repository.emitting_wrapper import (
    EmittingEntityRepository,
    EmittingRelationshipRepository,
)
from futurnal.pkg.repository.exceptions import (
    PKGRepositoryError,
    EntityNotFoundError,
    RelationshipNotFoundError,
    DuplicateEntityError,
    TemporalValidationError,
    InvalidEntityTypeError,
    InvalidRelationshipTypeError,
    BatchOperationError,
    QueryBuildError,
    StreamingError,
)

if TYPE_CHECKING:
    from futurnal.privacy.audit import AuditLogger


__all__ = [
    # Main facade
    "PKGRepository",
    # Sub-repositories
    "EntityRepository",
    "RelationshipRepository",
    "BatchRepository",
    # Emitting wrappers (Module 04 - PKG Sync)
    "EmittingEntityRepository",
    "EmittingRelationshipRepository",
    # Query builders
    "PKGQueryBuilder",
    "TemporalQueryBuilder",
    # Batch helpers
    "BatchResult",
    "TripleBatchLoader",
    # Base utilities
    "NODE_TYPE_MAP",
    "VALID_NODE_TYPES",
    "get_node_class",
    "get_label_for_node",
    # Exceptions
    "PKGRepositoryError",
    "EntityNotFoundError",
    "RelationshipNotFoundError",
    "DuplicateEntityError",
    "TemporalValidationError",
    "InvalidEntityTypeError",
    "InvalidRelationshipTypeError",
    "BatchOperationError",
    "QueryBuildError",
    "StreamingError",
]


class PKGRepository:
    """Main repository facade for PKG operations.

    Combines entity, relationship, and batch repositories into a single
    interface matching the production plan specification.

    This class provides:
    - Entity CRUD operations for all node types
    - Relationship CRUD with temporal validation
    - Query builders for complex queries
    - Batch operations for high throughput
    - Streaming for large datasets

    Option B Compliance:
    - EventNode.timestamp is REQUIRED (temporal-first design)
    - Temporal ordering validation for BEFORE/AFTER/CAUSES relationships
    - Causal relationships include Bradford Hill criteria structure
    - Production-ready with proper error handling

    Example:
        >>> with PKGDatabaseManager(storage_settings) as db:
        ...     repo = PKGRepository(db)
        ...
        ...     # Create a person
        ...     person_id = repo.create_entity("Person", {"name": "Alice"})
        ...
        ...     # Create an event (timestamp required!)
        ...     event_id = repo.create_entity("Event", {
        ...         "name": "Team Meeting",
        ...         "event_type": "meeting",
        ...         "timestamp": datetime.now(),
        ...         "source_document": "doc-001",
        ...     })
        ...
        ...     # Create relationship
        ...     rel_id = repo.create_relationship(person_id, "PARTICIPATED_IN", event_id)
        ...
        ...     # Query events in time range
        ...     events = repo.query_events_in_timerange(start, end)
    """

    def __init__(
        self,
        db_manager: PKGDatabaseManager,
        audit_logger: Optional["AuditLogger"] = None,
        batch_size: int = 500,
    ):
        """Initialize the PKG repository.

        Args:
            db_manager: The PKGDatabaseManager for database access.
                       Must be connected before calling repository methods.
            audit_logger: Optional audit logger for recording operations.
            batch_size: Default batch size for bulk operations.
        """
        self._db = db_manager
        self._audit = audit_logger
        self._entities = EntityRepository(db_manager, audit_logger)
        self._relationships = RelationshipRepository(db_manager, audit_logger)
        self._batch = BatchRepository(db_manager, audit_logger, batch_size)

    # ---------------------------------------------------------------------------
    # Entity Operations (delegates to EntityRepository)
    # ---------------------------------------------------------------------------

    def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create an entity and return its ID.

        Args:
            entity_type: Entity type (Person, Organization, Concept, Event, Document)
            properties: Entity properties

        Returns:
            The entity ID

        Raises:
            InvalidEntityTypeError: If entity_type is invalid
            ValueError: If EventNode missing timestamp (Option B)
            DuplicateEntityError: If entity ID already exists

        Example:
            >>> person_id = repo.create_entity("Person", {"name": "Alice"})
            >>> event_id = repo.create_entity("Event", {
            ...     "name": "Meeting",
            ...     "event_type": "meeting",
            ...     "timestamp": datetime.now(),
            ...     "source_document": "doc-001",
            ... })
        """
        node_class = get_node_class(entity_type)
        entity = node_class.model_validate(properties)
        return self._entities.create_entity(entity)

    def get_entity(self, entity_id: str) -> Optional[BaseNode]:
        """Get entity by ID.

        Args:
            entity_id: The entity identifier

        Returns:
            The entity as a Pydantic model, or None if not found
        """
        return self._entities.get_entity(entity_id)

    def get_entity_or_raise(self, entity_id: str) -> BaseNode:
        """Get entity by ID, raising if not found.

        Args:
            entity_id: The entity identifier

        Returns:
            The entity as a Pydantic model

        Raises:
            EntityNotFoundError: If entity not found
        """
        return self._entities.get_entity_or_raise(entity_id)

    def find_entities(
        self,
        entity_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BaseNode]:
        """Find entities with pagination.

        Args:
            entity_type: Entity type to find
            filters: Property filters (exact match)
            limit: Maximum results
            offset: Results to skip

        Returns:
            List of matching entities
        """
        return self._entities.find_entities(entity_type, filters, limit, offset)

    def count_entities(self, entity_type: Optional[str] = None) -> int:
        """Count entities by type.

        Args:
            entity_type: Optional entity type filter

        Returns:
            Count of entities
        """
        if entity_type:
            return self._entities.count_entities(entity_type)

        # Count all types
        total = 0
        for et in VALID_NODE_TYPES:
            total += self._entities.count_entities(et)
        return total

    def update_entity(
        self, entity_id: str, properties: Dict[str, Any]
    ) -> BaseNode:
        """Update entity properties.

        Args:
            entity_id: Entity to update
            properties: Properties to update (merged with existing)

        Returns:
            Updated entity
        """
        return self._entities.update_entity(entity_id, properties)

    def delete_entity(self, entity_id: str, cascade: bool = False) -> bool:
        """Delete an entity.

        Args:
            entity_id: Entity to delete
            cascade: If True, also delete relationships

        Returns:
            True if deleted, False if not found
        """
        return self._entities.delete_entity(entity_id, cascade)

    # ---------------------------------------------------------------------------
    # Convenience Entity Creators
    # ---------------------------------------------------------------------------

    def create_person(self, name: str, **kwargs: Any) -> str:
        """Create a PersonNode."""
        return self._entities.create_person(name, **kwargs)

    def create_organization(self, name: str, org_type: str = "unknown", **kwargs: Any) -> str:
        """Create an OrganizationNode."""
        return self._entities.create_organization(name, org_type, **kwargs)

    def create_concept(self, name: str, **kwargs: Any) -> str:
        """Create a ConceptNode."""
        return self._entities.create_concept(name, **kwargs)

    def create_event(
        self,
        name: str,
        event_type: str,
        timestamp: datetime,
        source_document: str,
        **kwargs: Any,
    ) -> str:
        """Create an EventNode with required temporal grounding.

        Option B Critical: timestamp is REQUIRED.

        Args:
            name: Event name
            event_type: Event type (meeting, decision, etc.)
            timestamp: When the event occurred (REQUIRED)
            source_document: Source document ID
            **kwargs: Additional properties

        Returns:
            The event ID
        """
        return self._entities.create_event(name, event_type, timestamp, source_document, **kwargs)

    def create_document(
        self,
        source_id: str,
        source_type: str,
        content_hash: str,
        **kwargs: Any,
    ) -> str:
        """Create a DocumentNode."""
        return self._entities.create_document(source_id, source_type, content_hash, **kwargs)

    # ---------------------------------------------------------------------------
    # Relationship Operations (delegates to RelationshipRepository)
    # ---------------------------------------------------------------------------

    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a relationship and return its ID.

        Args:
            subject_id: Source node ID
            predicate: Relationship type
            object_id: Target node ID
            properties: Optional relationship properties

        Returns:
            The relationship ID
        """
        return self._relationships.create_relationship(
            subject_id, predicate, object_id, properties
        )

    def create_temporal_relationship(
        self,
        source_event_id: str,
        target_event_id: str,
        relationship_type: TemporalRelationType,
        properties: Optional[TemporalRelationshipProps] = None,
    ) -> str:
        """Create a temporal relationship with ordering validation.

        Option B Critical: Validates temporal ordering.

        Args:
            source_event_id: Source event ID
            target_event_id: Target event ID
            relationship_type: BEFORE, AFTER, DURING, or SIMULTANEOUS
            properties: Temporal relationship properties

        Returns:
            The relationship ID

        Raises:
            TemporalValidationError: If ordering is invalid
        """
        return self._relationships.create_temporal_relationship(
            source_event_id, target_event_id, relationship_type, properties
        )

    def create_causal_relationship(
        self,
        cause_event_id: str,
        effect_event_id: str,
        relationship_type: CausalRelationType,
        properties: CausalRelationshipProps,
    ) -> str:
        """Create a causal relationship with temporal validation.

        Option B Critical: Cause must precede effect.

        Args:
            cause_event_id: Cause event ID
            effect_event_id: Effect event ID
            relationship_type: CAUSES, ENABLES, PREVENTS, or TRIGGERS
            properties: Causal relationship properties (includes Bradford Hill)

        Returns:
            The relationship ID
        """
        return self._relationships.create_causal_relationship(
            cause_event_id, effect_event_id, relationship_type, properties
        )

    def get_relationships_from(
        self, entity_id: str, relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get outgoing relationships from an entity."""
        return self._relationships.get_relationships_from(entity_id, relationship_type)

    def get_relationships_to(
        self, entity_id: str, relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get incoming relationships to an entity."""
        return self._relationships.get_relationships_to(entity_id, relationship_type)

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        return self._relationships.delete_relationship(relationship_id)

    def count_relationships(self, relationship_type: Optional[str] = None) -> int:
        """Count relationships by type."""
        return self._relationships.count_relationships(relationship_type)

    # ---------------------------------------------------------------------------
    # Temporal Queries (Module 04 Preparation)
    # ---------------------------------------------------------------------------

    def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
    ) -> List[EventNode]:
        """Find events within time range.

        Base implementation for Module 04 (Temporal Query Support) to extend.

        Args:
            start: Start of time range
            end: End of time range
            event_type: Optional event type filter

        Returns:
            List of EventNode instances in the time range
        """
        return self._entities.find_events_in_timerange(start, end, event_type)

    # ---------------------------------------------------------------------------
    # Query Builder Access
    # ---------------------------------------------------------------------------

    def query(self) -> PKGQueryBuilder:
        """Get a query builder for complex queries.

        Returns:
            PKGQueryBuilder instance

        Example:
            >>> results = (
            ...     repo.query()
            ...     .match_node("Person", "p")
            ...     .where_property_contains("p", "name", "Alice")
            ...     .return_nodes("p")
            ...     .execute()
            ... )
        """
        return PKGQueryBuilder(self._db)

    def temporal_query(self) -> TemporalQueryBuilder:
        """Get a temporal query builder.

        Module 04 preparation: Extended query builder with temporal primitives.

        Returns:
            TemporalQueryBuilder instance
        """
        return TemporalQueryBuilder(self._db)

    # ---------------------------------------------------------------------------
    # Batch Operations
    # ---------------------------------------------------------------------------

    def bulk_insert(
        self,
        entities: List[BaseNode],
        relationships: Optional[List[Tuple[str, str, str, Optional[Dict[str, Any]]]]] = None,
    ) -> BatchResult:
        """Bulk insert entities and relationships.

        Args:
            entities: List of entities to create
            relationships: Optional list of (subject_id, predicate, object_id, props)

        Returns:
            BatchResult with operation statistics
        """
        # Create entities first
        entities_result = self._batch.bulk_create_entities(entities)

        if relationships:
            rels_result = self._batch.bulk_create_relationships(relationships)
            return BatchResult(
                total=len(entities) + len(relationships),
                succeeded=entities_result.succeeded + rels_result.succeeded,
                failed=entities_result.failed + rels_result.failed,
                failed_items=entities_result.failed_items + rels_result.failed_items,
                duration_seconds=entities_result.duration_seconds + rels_result.duration_seconds,
            )

        return entities_result

    def bulk_delete_entities(
        self, entity_ids: List[str], cascade: bool = False
    ) -> BatchResult:
        """Bulk delete entities."""
        return self._batch.bulk_delete_entities(entity_ids, cascade)

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
        """
        yield from self._batch.stream_entities(entity_type, filters, batch_size)

    # ---------------------------------------------------------------------------
    # Direct Repository Access
    # ---------------------------------------------------------------------------

    @property
    def entities(self) -> EntityRepository:
        """Get the underlying EntityRepository for advanced operations."""
        return self._entities

    @property
    def relationships(self) -> RelationshipRepository:
        """Get the underlying RelationshipRepository for advanced operations."""
        return self._relationships

    @property
    def batch(self) -> BatchRepository:
        """Get the underlying BatchRepository for advanced operations."""
        return self._batch

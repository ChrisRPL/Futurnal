"""Emitting Repository Wrappers for PKG Synchronization.

Provides wrapper classes for EntityRepository and RelationshipRepository
that emit PKGEvents after successful mutations for embedding synchronization.

The wrappers follow the decorator pattern - they delegate all operations
to the underlying repository while adding event emission after success.

Architecture:
    Client Code
         |
    EmittingEntityRepository (wrapper)
         |
    EntityRepository (actual operations)
         |
    PKGEventEmitter.emit(PKGEvent)
         |
    PKGSyncHandler (embedding sync)

Option B Compliance:
- Events carry full mutation context for proper embedding generation
- Schema version tracked for autonomous evolution
- Temporal metadata preserved in events

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Example Usage:
    >>> from futurnal.pkg.repository import EntityRepository
    >>> from futurnal.pkg.repository.emitting_wrapper import EmittingEntityRepository
    >>> from futurnal.pkg.sync import PKGEventEmitter
    >>>
    >>> # Create base repository
    >>> entity_repo = EntityRepository(db_manager)
    >>>
    >>> # Create emitter with sync handler
    >>> emitter = PKGEventEmitter(event_handler=sync_handler.handle_event)
    >>>
    >>> # Wrap repository for event emission
    >>> emitting_repo = EmittingEntityRepository(
    ...     repo=entity_repo,
    ...     emitter=emitter,
    ...     schema_manager=schema_manager,
    ... )
    >>>
    >>> # Use as normal - events emitted automatically
    >>> entity_id = emitting_repo.create_entity(PersonNode(name="Alice"))
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING
from uuid import uuid4

from futurnal.pkg.schema.models import BaseNode, EventNode
from futurnal.pkg.sync.events import PKGEvent, SyncEventType

if TYPE_CHECKING:
    from futurnal.pkg.repository.entities import EntityRepository
    from futurnal.pkg.repository.relationships import RelationshipRepository
    from futurnal.pkg.sync.emitter import PKGEventEmitter
    from futurnal.pkg.schema.migration import SchemaVersionManager

logger = logging.getLogger(__name__)


class EmittingEntityRepository:
    """Wrapper that emits PKGEvents after EntityRepository mutations.

    All methods delegate to the underlying repository, emitting a PKGEvent
    after successful mutations (create, update, delete).

    Thread Safety:
        This wrapper is thread-safe if the underlying repository is thread-safe.
        Event emission is asynchronous and non-blocking.

    Attributes:
        repo: The wrapped EntityRepository
        emitter: PKGEventEmitter for event emission
        schema_manager: Optional SchemaVersionManager for version tracking

    Example:
        >>> emitting_repo = EmittingEntityRepository(
        ...     repo=entity_repo,
        ...     emitter=emitter,
        ...     schema_manager=schema_manager,
        ... )
        >>> # Create emits ENTITY_CREATED event
        >>> entity_id = emitting_repo.create_entity(person)
    """

    def __init__(
        self,
        repo: "EntityRepository",
        emitter: "PKGEventEmitter",
        schema_manager: Optional["SchemaVersionManager"] = None,
    ) -> None:
        """Initialize the emitting entity repository wrapper.

        Args:
            repo: The underlying EntityRepository to wrap
            emitter: PKGEventEmitter for event emission
            schema_manager: Optional SchemaVersionManager for schema version tracking.
                           If None, defaults to schema_version=1.
        """
        self._repo = repo
        self._emitter = emitter
        self._schema_manager = schema_manager

        logger.info("Initialized EmittingEntityRepository wrapper")

    @property
    def repo(self) -> "EntityRepository":
        """Access the underlying repository."""
        return self._repo

    def _get_schema_version(self) -> int:
        """Get current schema version."""
        if self._schema_manager is None:
            return 1

        version_node = self._schema_manager.get_current_version()
        return version_node.version if version_node else 1

    def _get_entity_type(self, entity: BaseNode) -> str:
        """Get entity type name from node instance."""
        class_name = type(entity).__name__
        # Remove "Node" suffix if present
        if class_name.endswith("Node"):
            return class_name[:-4]
        return class_name

    def _get_entity_type_from_label(self, label: str) -> str:
        """Get entity type name from Neo4j label."""
        # Labels are stored as-is (Person, Organization, etc.)
        return label

    # ---------------------------------------------------------------------------
    # CREATE Operations with Event Emission
    # ---------------------------------------------------------------------------

    def create_entity(self, entity: BaseNode) -> str:
        """Create an entity and emit ENTITY_CREATED event.

        Args:
            entity: The entity to create

        Returns:
            The entity ID

        Raises:
            Same exceptions as EntityRepository.create_entity()
        """
        # Delegate to underlying repository
        entity_id = self._repo.create_entity(entity)

        # Emit event after successful creation
        self._emit_created_event(entity_id, entity)

        return entity_id

    def _emit_created_event(self, entity_id: str, entity: BaseNode) -> None:
        """Emit ENTITY_CREATED event for new entity."""
        entity_type = self._get_entity_type(entity)
        new_data = entity.to_cypher_properties()
        new_data["id"] = entity_id  # Ensure ID is in data

        # Get source document if available
        source_document_id = None
        extraction_confidence = None

        if hasattr(entity, "source_document"):
            source_document_id = getattr(entity, "source_document", None)
        if hasattr(entity, "confidence"):
            extraction_confidence = getattr(entity, "confidence", None)

        event = PKGEvent(
            event_id=f"create_{entity_id}_{uuid4().hex[:8]}",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id=entity_id,
            entity_type=entity_type,
            timestamp=datetime.utcnow(),
            new_data=new_data,
            source_document_id=source_document_id,
            extraction_confidence=extraction_confidence,
            schema_version=self._get_schema_version(),
        )

        self._emitter.emit(event)

    def create_person(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Create a PersonNode with event emission.

        Delegates to EntityRepository.create_person() then emits event.
        """
        entity_id = self._repo.create_person(name, aliases, confidence, **kwargs)

        # Build node data for event
        from futurnal.pkg.schema.models import PersonNode
        person = PersonNode(id=entity_id, name=name, aliases=aliases or [], confidence=confidence)
        self._emit_created_event(entity_id, person)

        return entity_id

    def create_organization(
        self,
        name: str,
        org_type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Create an OrganizationNode with event emission."""
        entity_id = self._repo.create_organization(name, org_type, aliases, confidence, **kwargs)

        from futurnal.pkg.schema.models import OrganizationNode
        org = OrganizationNode(id=entity_id, name=name, type=org_type, aliases=aliases or [], confidence=confidence)
        self._emit_created_event(entity_id, org)

        return entity_id

    def create_event(
        self,
        name: str,
        timestamp: datetime,
        event_type: Optional[str] = None,
        duration: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Create an EventNode with event emission.

        Note: EventNode requires timestamp (Option B compliance).
        """
        entity_id = self._repo.create_event(name, timestamp, event_type, duration, **kwargs)

        from futurnal.pkg.schema.models import EventNode
        event_node = EventNode(id=entity_id, name=name, timestamp=timestamp, event_type=event_type, duration=duration)
        self._emit_created_event(entity_id, event_node)

        return entity_id

    # ---------------------------------------------------------------------------
    # UPDATE Operations with Event Emission
    # ---------------------------------------------------------------------------

    def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any],
        merge: bool = True,
    ) -> BaseNode:
        """Update entity properties and emit ENTITY_UPDATED event.

        Args:
            entity_id: Entity to update
            properties: Properties to update
            merge: If True, merge with existing; if False, replace

        Returns:
            Updated entity

        Raises:
            Same exceptions as EntityRepository.update_entity()
        """
        # Get entity before update for old_data
        old_entity = self._repo.get_entity(entity_id)
        old_data = old_entity.to_cypher_properties() if old_entity else {}

        # Delegate to underlying repository
        updated_entity = self._repo.update_entity(entity_id, properties, merge)

        # Emit event after successful update
        self._emit_updated_event(entity_id, old_data, updated_entity)

        return updated_entity

    def _emit_updated_event(
        self,
        entity_id: str,
        old_data: Dict[str, Any],
        updated_entity: BaseNode,
    ) -> None:
        """Emit ENTITY_UPDATED event for updated entity."""
        entity_type = self._get_entity_type(updated_entity)
        new_data = updated_entity.to_cypher_properties()

        # Get source document if available
        source_document_id = None
        extraction_confidence = None

        if hasattr(updated_entity, "source_document"):
            source_document_id = getattr(updated_entity, "source_document", None)
        if hasattr(updated_entity, "confidence"):
            extraction_confidence = getattr(updated_entity, "confidence", None)

        event = PKGEvent(
            event_id=f"update_{entity_id}_{uuid4().hex[:8]}",
            event_type=SyncEventType.ENTITY_UPDATED,
            entity_id=entity_id,
            entity_type=entity_type,
            timestamp=datetime.utcnow(),
            old_data=old_data,
            new_data=new_data,
            source_document_id=source_document_id,
            extraction_confidence=extraction_confidence,
            schema_version=self._get_schema_version(),
        )

        self._emitter.emit(event)

    # ---------------------------------------------------------------------------
    # DELETE Operations with Event Emission
    # ---------------------------------------------------------------------------

    def delete_entity(self, entity_id: str, cascade: bool = False) -> bool:
        """Delete an entity and emit ENTITY_DELETED event.

        Args:
            entity_id: Entity to delete
            cascade: If True, also delete relationships

        Returns:
            True if entity was deleted, False if not found
        """
        # Get entity before deletion for event data
        entity = self._repo.get_entity(entity_id)
        old_data = entity.to_cypher_properties() if entity else {}
        entity_type = self._get_entity_type(entity) if entity else "Unknown"

        # Delegate to underlying repository
        deleted = self._repo.delete_entity(entity_id, cascade)

        # Emit event after successful deletion
        if deleted:
            self._emit_deleted_event(entity_id, entity_type, old_data)

        return deleted

    def _emit_deleted_event(
        self,
        entity_id: str,
        entity_type: str,
        old_data: Dict[str, Any],
    ) -> None:
        """Emit ENTITY_DELETED event for deleted entity."""
        event = PKGEvent(
            event_id=f"delete_{entity_id}_{uuid4().hex[:8]}",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id=entity_id,
            entity_type=entity_type,
            timestamp=datetime.utcnow(),
            old_data=old_data,
            schema_version=self._get_schema_version(),
        )

        self._emitter.emit(event)

    # ---------------------------------------------------------------------------
    # READ Operations (Direct Delegation - No Events)
    # ---------------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[BaseNode]:
        """Get entity by ID. Delegates to underlying repository."""
        return self._repo.get_entity(entity_id)

    def get_entity_by_type(
        self,
        entity_id: str,
        entity_type: str,
    ) -> Optional[BaseNode]:
        """Get entity by ID and type. Delegates to underlying repository."""
        return self._repo.get_entity_by_type(entity_id, entity_type)

    def get_entity_or_raise(self, entity_id: str) -> BaseNode:
        """Get entity or raise EntityNotFoundError. Delegates to underlying repository."""
        return self._repo.get_entity_or_raise(entity_id)

    def find_entities(
        self,
        entity_type: str,
        filters: Optional[Dict[str, Any]] = None,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = False,
    ) -> List[BaseNode]:
        """Find entities with filtering. Delegates to underlying repository."""
        return self._repo.find_entities(
            entity_type, filters, skip, limit, order_by, order_desc
        )

    def find_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None,
        exact_match: bool = False,
    ) -> List[BaseNode]:
        """Find entities by name. Delegates to underlying repository."""
        return self._repo.find_by_name(name, entity_type, exact_match)

    def find_events_in_timerange(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Find events in time range. Delegates to underlying repository."""
        return self._repo.find_events_in_timerange(start, end, event_type, limit)

    def stream_entities(
        self,
        entity_type: str,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
    ) -> Generator[BaseNode, None, None]:
        """Stream entities. Delegates to underlying repository."""
        return self._repo.stream_entities(entity_type, filters, batch_size)

    def count_entities(self, entity_type: Optional[str] = None) -> int:
        """Count entities. Delegates to underlying repository."""
        return self._repo.count_entities(entity_type)

    # ---------------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying repository.

        This allows the wrapper to be used as a drop-in replacement
        for EntityRepository while only overriding mutation methods.
        """
        return getattr(self._repo, name)


class EmittingRelationshipRepository:
    """Wrapper that emits PKGEvents after RelationshipRepository mutations.

    Similar to EmittingEntityRepository but for relationship operations.
    Emits RELATIONSHIP_CREATED, RELATIONSHIP_UPDATED, RELATIONSHIP_DELETED events.

    Example:
        >>> emitting_rel_repo = EmittingRelationshipRepository(
        ...     repo=relationship_repo,
        ...     emitter=emitter,
        ...     schema_manager=schema_manager,
        ... )
        >>> rel_id = emitting_rel_repo.create_relationship(source_id, target_id, rel_type)
    """

    def __init__(
        self,
        repo: "RelationshipRepository",
        emitter: "PKGEventEmitter",
        schema_manager: Optional["SchemaVersionManager"] = None,
    ) -> None:
        """Initialize the emitting relationship repository wrapper.

        Args:
            repo: The underlying RelationshipRepository to wrap
            emitter: PKGEventEmitter for event emission
            schema_manager: Optional SchemaVersionManager for schema version tracking
        """
        self._repo = repo
        self._emitter = emitter
        self._schema_manager = schema_manager

        logger.info("Initialized EmittingRelationshipRepository wrapper")

    @property
    def repo(self) -> "RelationshipRepository":
        """Access the underlying repository."""
        return self._repo

    def _get_schema_version(self) -> int:
        """Get current schema version."""
        if self._schema_manager is None:
            return 1

        version_node = self._schema_manager.get_current_version()
        return version_node.version if version_node else 1

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a relationship and emit RELATIONSHIP_CREATED event."""
        rel_id = self._repo.create_relationship(source_id, target_id, rel_type, properties)

        event = PKGEvent(
            event_id=f"rel_create_{rel_id}_{uuid4().hex[:8]}",
            event_type=SyncEventType.RELATIONSHIP_CREATED,
            entity_id=rel_id,
            entity_type=rel_type,
            timestamp=datetime.utcnow(),
            new_data={
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
                "properties": properties or {},
            },
            schema_version=self._get_schema_version(),
        )

        self._emitter.emit(event)
        return rel_id

    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship and emit RELATIONSHIP_DELETED event."""
        # Get relationship info before deletion if possible
        deleted = self._repo.delete_relationship(rel_id)

        if deleted:
            event = PKGEvent(
                event_id=f"rel_delete_{rel_id}_{uuid4().hex[:8]}",
                event_type=SyncEventType.RELATIONSHIP_DELETED,
                entity_id=rel_id,
                entity_type="Relationship",
                timestamp=datetime.utcnow(),
                schema_version=self._get_schema_version(),
            )

            self._emitter.emit(event)

        return deleted

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying repository."""
        return getattr(self._repo, name)

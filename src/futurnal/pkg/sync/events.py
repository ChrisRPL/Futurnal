"""Sync Event Models for PKG ↔ Vector Store Synchronization.

Provides event models for tracking and verifying synchronization
between PKG (Neo4j) and Vector Store (ChromaDB).

These models enable:
- Integration testing of sync behavior
- Production monitoring of sync operations
- Debugging sync failures
- Audit trail for data consistency verification
- PKG mutation event emission for embedding synchronization (Module 04)

Implementation follows production plans:
- docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
- docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Option B Compliance:
- Events capture temporal metadata (timestamps)
- Support for experiential event tracking (Phase 2 prep)
- No mocks - real event capture for testing
- PKGEvent carries full mutation context for embedding sync

Example Usage:
    >>> from futurnal.pkg.sync import SyncEvent, SyncEventCapture, SyncEventType
    >>>
    >>> # Create a capture utility
    >>> capture = SyncEventCapture()
    >>>
    >>> # Simulate PKG write triggering sync
    >>> capture.capture(SyncEvent(
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="doc_abc123",
    ...     entity_type="Document",
    ...     source_operation="pkg_write",
    ...     vector_sync_status=SyncStatus.PENDING,
    ... ))
    >>>
    >>> # Verify sync completed
    >>> capture.capture(SyncEvent(
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="doc_abc123",
    ...     entity_type="Document",
    ...     source_operation="vector_write",
    ...     vector_sync_status=SyncStatus.COMPLETED,
    ... ))
    >>>
    >>> # Check sync events
    >>> events = capture.get_events_for_entity("doc_abc123")
    >>> assert len(events) == 2
    >>> assert events[-1].vector_sync_status == SyncStatus.COMPLETED

    >>> # PKG mutation events for embedding sync
    >>> from futurnal.pkg.sync import PKGEvent
    >>> pkg_event = PKGEvent(
    ...     event_id="evt_123",
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="person_456",
    ...     entity_type="Person",
    ...     new_data={"name": "John Doe", "description": "Engineer"},
    ...     schema_version=1,
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SyncEventType(str, Enum):
    """Type of sync event.

    Covers both PKG mutations and batch operations for embedding synchronization.
    """

    # Entity mutations (triggers embedding sync)
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_DELETED = "entity_deleted"

    # Relationship mutations
    RELATIONSHIP_CREATED = "relationship_created"
    RELATIONSHIP_UPDATED = "relationship_updated"
    RELATIONSHIP_DELETED = "relationship_deleted"

    # Schema evolution (triggers selective re-embedding)
    SCHEMA_EVOLVED = "schema_evolved"

    # Batch operations
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    BATCH_FAILED = "batch_failed"


class SyncStatus(str, Enum):
    """Status of vector store synchronization."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SourceOperation(str, Enum):
    """Source of the sync operation."""

    PKG_WRITE = "pkg_write"
    PKG_UPDATE = "pkg_update"
    PKG_DELETE = "pkg_delete"
    VECTOR_WRITE = "vector_write"
    VECTOR_DELETE = "vector_delete"
    BATCH_INSERT = "batch_insert"
    BATCH_DELETE = "batch_delete"


# ---------------------------------------------------------------------------
# Sync Event Model
# ---------------------------------------------------------------------------


@dataclass
class SyncEvent:
    """Represents a synchronization event between PKG and Vector Store.

    Tracks the lifecycle of data synchronization from PKG write
    through to vector store update.

    Attributes:
        event_type: Type of sync event (created, updated, deleted)
        entity_id: Unique identifier of the entity (usually SHA256 or UUID)
        entity_type: Type of entity (Document, Person, Event, etc.)
        timestamp: When the sync event occurred
        source_operation: Which operation triggered this event
        vector_sync_status: Current status of vector synchronization
        metadata: Additional event-specific data
        error_message: Error details if sync failed
        duration_ms: Duration of the operation in milliseconds

    Example:
        >>> event = SyncEvent(
        ...     event_type=SyncEventType.ENTITY_CREATED,
        ...     entity_id="abc123",
        ...     entity_type="Document",
        ...     source_operation=SourceOperation.PKG_WRITE,
        ...     vector_sync_status=SyncStatus.PENDING,
        ... )
    """

    event_type: SyncEventType | str
    entity_id: str
    entity_type: str
    source_operation: SourceOperation | str
    vector_sync_status: SyncStatus | str = SyncStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        """Convert string values to enums if needed."""
        if isinstance(self.event_type, str):
            try:
                self.event_type = SyncEventType(self.event_type)
            except ValueError:
                pass  # Keep as string if not a valid enum

        if isinstance(self.source_operation, str):
            try:
                self.source_operation = SourceOperation(self.source_operation)
            except ValueError:
                pass

        if isinstance(self.vector_sync_status, str):
            try:
                self.vector_sync_status = SyncStatus(self.vector_sync_status)
            except ValueError:
                pass

    @property
    def is_completed(self) -> bool:
        """Check if sync is completed."""
        return self.vector_sync_status == SyncStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if sync failed."""
        return self.vector_sync_status == SyncStatus.FAILED

    @property
    def is_pending(self) -> bool:
        """Check if sync is pending."""
        return self.vector_sync_status == SyncStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "timestamp": self.timestamp.isoformat(),
            "source_operation": (
                self.source_operation.value if isinstance(self.source_operation, Enum) else self.source_operation
            ),
            "vector_sync_status": (
                self.vector_sync_status.value
                if isinstance(self.vector_sync_status, Enum)
                else self.vector_sync_status
            ),
            "metadata": self.metadata,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncEvent":
        """Create SyncEvent from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            event_type=data["event_type"],
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            timestamp=timestamp,
            source_operation=data.get("source_operation", "unknown"),
            vector_sync_status=data.get("vector_sync_status", SyncStatus.PENDING),
            metadata=data.get("metadata", {}),
            error_message=data.get("error_message"),
            duration_ms=data.get("duration_ms"),
        )


# ---------------------------------------------------------------------------
# Sync Event Capture Utility
# ---------------------------------------------------------------------------


@dataclass
class SyncEventCapture:
    """Utility for capturing and analyzing sync events.

    Used for:
    - Integration testing: Verify PKG operations trigger proper sync
    - Production monitoring: Track sync success/failure rates
    - Debugging: Investigate sync issues

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded usage,
        wrap capture operations with appropriate locking.

    Example:
        >>> capture = SyncEventCapture()
        >>>
        >>> # Attach to NormalizationSink
        >>> sink = NormalizationSink(
        ...     pkg_writer=pkg_writer,
        ...     vector_writer=vector_writer,
        ...     sync_event_handler=capture.capture
        ... )
        >>>
        >>> # Process documents
        >>> sink.handle(document1)
        >>> sink.handle(document2)
        >>>
        >>> # Analyze sync events
        >>> print(f"Total events: {capture.count}")
        >>> print(f"Completed: {len(capture.get_completed_events())}")
        >>> print(f"Failed: {len(capture.get_failed_events())}")
    """

    events: List[SyncEvent] = field(default_factory=list)
    max_events: int = 10000  # Prevent unbounded growth
    on_event: Optional[Callable[[SyncEvent], None]] = None  # Callback for real-time processing

    def capture(self, event: SyncEvent) -> None:
        """Capture a sync event.

        Args:
            event: The sync event to capture

        Note:
            If max_events is reached, oldest events are discarded.
        """
        if len(self.events) >= self.max_events:
            # Remove oldest 10% to make room
            remove_count = self.max_events // 10
            self.events = self.events[remove_count:]
            logger.warning(f"SyncEventCapture overflow: removed {remove_count} oldest events")

        self.events.append(event)
        logger.debug(f"Captured sync event: {event.event_type} for {event.entity_id}")

        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Error in sync event callback: {e}")

    def get_events_for_entity(self, entity_id: str) -> List[SyncEvent]:
        """Get all events for a specific entity.

        Args:
            entity_id: The entity identifier

        Returns:
            List of events for the entity, ordered by timestamp
        """
        return sorted(
            [e for e in self.events if e.entity_id == entity_id],
            key=lambda e: e.timestamp,
        )

    def get_events_by_type(self, event_type: SyncEventType | str) -> List[SyncEvent]:
        """Get all events of a specific type.

        Args:
            event_type: The event type to filter by

        Returns:
            List of matching events
        """
        if isinstance(event_type, str):
            return [e for e in self.events if str(e.event_type) == event_type or e.event_type == event_type]
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_status(self, status: SyncStatus | str) -> List[SyncEvent]:
        """Get all events with a specific sync status.

        Args:
            status: The sync status to filter by

        Returns:
            List of matching events
        """
        if isinstance(status, str):
            return [e for e in self.events if str(e.vector_sync_status) == status or e.vector_sync_status == status]
        return [e for e in self.events if e.vector_sync_status == status]

    def get_completed_events(self) -> List[SyncEvent]:
        """Get all events where sync completed successfully."""
        return self.get_events_by_status(SyncStatus.COMPLETED)

    def get_failed_events(self) -> List[SyncEvent]:
        """Get all events where sync failed."""
        return self.get_events_by_status(SyncStatus.FAILED)

    def get_pending_events(self) -> List[SyncEvent]:
        """Get all events where sync is still pending."""
        return self.get_events_by_status(SyncStatus.PENDING)

    def get_latest_event(self) -> Optional[SyncEvent]:
        """Get the most recent event."""
        return self.events[-1] if self.events else None

    def get_latest_event_for_entity(self, entity_id: str) -> Optional[SyncEvent]:
        """Get the most recent event for a specific entity."""
        entity_events = self.get_events_for_entity(entity_id)
        return entity_events[-1] if entity_events else None

    def get_events_in_timerange(self, start: datetime, end: datetime) -> List[SyncEvent]:
        """Get events within a time range.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)

        Returns:
            List of events within the range
        """
        return [e for e in self.events if start <= e.timestamp <= end]

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()
        logger.debug("Cleared sync event capture")

    @property
    def count(self) -> int:
        """Total number of captured events."""
        return len(self.events)

    @property
    def success_rate(self) -> float:
        """Calculate sync success rate.

        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if not self.events:
            return 0.0

        completed = len(self.get_completed_events())
        failed = len(self.get_failed_events())
        total = completed + failed

        if total == 0:
            return 0.0  # No terminal states yet

        return completed / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for captured events.

        Returns:
            Dictionary with event statistics
        """
        return {
            "total_events": self.count,
            "by_type": {
                event_type.value: len(self.get_events_by_type(event_type)) for event_type in SyncEventType
            },
            "by_status": {status.value: len(self.get_events_by_status(status)) for status in SyncStatus},
            "success_rate": self.success_rate,
            "unique_entities": len({e.entity_id for e in self.events}),
        }

    def assert_entity_synced(self, entity_id: str, timeout_events: int = 10) -> None:
        """Assert that an entity has been successfully synced.

        Args:
            entity_id: The entity to check
            timeout_events: Max events to wait for sync completion

        Raises:
            AssertionError: If entity is not synced or sync failed
        """
        events = self.get_events_for_entity(entity_id)

        if not events:
            raise AssertionError(f"No sync events found for entity {entity_id}")

        latest = events[-1]
        if latest.is_failed:
            raise AssertionError(f"Sync failed for entity {entity_id}: {latest.error_message}")

        if not latest.is_completed:
            raise AssertionError(
                f"Sync not completed for entity {entity_id}. Status: {latest.vector_sync_status}"
            )


# ---------------------------------------------------------------------------
# PKG Mutation Event Model (Module 04)
# ---------------------------------------------------------------------------


@dataclass
class PKGEvent:
    """Rich PKG mutation event for embedding synchronization.

    Captures full mutation context to enable proper embedding sync:
    - For ENTITY_CREATED: new_data contains the entity properties to embed
    - For ENTITY_UPDATED: old_data and new_data enable change detection
    - For ENTITY_DELETED: entity_id identifies embedding to remove
    - For SCHEMA_EVOLVED: triggers selective re-embedding

    This model is used by PKGEventEmitter and consumed by PKGSyncHandler
    to maintain consistency between PKG and the Vector Embedding Store.

    Option B Compliance:
    - Temporal context preserved (timestamp field)
    - Schema version tracked for autonomous evolution
    - Extraction confidence for quality tracking

    Production Plan Reference:
    docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

    Example:
        >>> event = PKGEvent(
        ...     event_id="evt_abc123",
        ...     event_type=SyncEventType.ENTITY_CREATED,
        ...     entity_id="person_456",
        ...     entity_type="Person",
        ...     new_data={"name": "John Doe", "description": "Software Engineer"},
        ...     source_document_id="doc_789",
        ...     extraction_confidence=0.95,
        ...     schema_version=1,
        ... )
        >>> sync_event = event.to_sync_event(SyncStatus.PENDING)
    """

    # Required fields
    event_id: str
    event_type: SyncEventType | str
    entity_id: str
    entity_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Mutation data
    old_data: Optional[Dict[str, Any]] = None
    new_data: Optional[Dict[str, Any]] = None

    # Provenance
    source_document_id: Optional[str] = None
    extraction_confidence: Optional[float] = None

    # Schema tracking
    schema_version: int = 1

    def __post_init__(self):
        """Convert string event type to enum if needed."""
        if isinstance(self.event_type, str):
            try:
                self.event_type = SyncEventType(self.event_type)
            except ValueError:
                pass  # Keep as string if not a valid enum

    def to_sync_event(
        self,
        sync_status: SyncStatus = SyncStatus.PENDING,
        source_operation: SourceOperation = SourceOperation.PKG_WRITE,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> SyncEvent:
        """Convert to SyncEvent for backward compatibility.

        Args:
            sync_status: Vector store sync status
            source_operation: Source of the operation
            duration_ms: Operation duration in milliseconds
            error_message: Error message if sync failed

        Returns:
            SyncEvent compatible with SyncEventCapture
        """
        return SyncEvent(
            event_type=self.event_type,
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            timestamp=self.timestamp,
            source_operation=source_operation,
            vector_sync_status=sync_status,
            metadata={
                "event_id": self.event_id,
                "source_document_id": self.source_document_id,
                "extraction_confidence": self.extraction_confidence,
                "schema_version": self.schema_version,
            },
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "timestamp": self.timestamp.isoformat(),
            "old_data": self.old_data,
            "new_data": self.new_data,
            "source_document_id": self.source_document_id,
            "extraction_confidence": self.extraction_confidence,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PKGEvent":
        """Create PKGEvent from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            timestamp=timestamp,
            old_data=data.get("old_data"),
            new_data=data.get("new_data"),
            source_document_id=data.get("source_document_id"),
            extraction_confidence=data.get("extraction_confidence"),
            schema_version=data.get("schema_version", 1),
        )

    @property
    def is_creation(self) -> bool:
        """Check if this is an entity creation event."""
        return self.event_type == SyncEventType.ENTITY_CREATED

    @property
    def is_update(self) -> bool:
        """Check if this is an entity update event."""
        return self.event_type == SyncEventType.ENTITY_UPDATED

    @property
    def is_deletion(self) -> bool:
        """Check if this is an entity deletion event."""
        return self.event_type == SyncEventType.ENTITY_DELETED

    @property
    def is_schema_evolution(self) -> bool:
        """Check if this is a schema evolution event."""
        return self.event_type == SyncEventType.SCHEMA_EVOLVED

    @property
    def requires_embedding(self) -> bool:
        """Check if this event requires embedding generation.

        Events need embedding if:
        - Creation: Always needs new embedding
        - Update: May need re-embedding (checked separately)
        - Schema evolution: May trigger re-embedding (checked separately)
        """
        return self.event_type in (
            SyncEventType.ENTITY_CREATED,
            SyncEventType.ENTITY_UPDATED,
        )

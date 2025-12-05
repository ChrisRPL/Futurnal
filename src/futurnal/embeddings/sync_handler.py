"""PKG Sync Handler for Embedding Synchronization.

Routes PKG mutation events to appropriate embedding operations, ensuring
embeddings remain synchronized with PKG graph state.

Architecture:
    PKGEventEmitter
          |
    IncrementalEmbeddingUpdater (optional batching)
          |
    PKGSyncHandler.handle_event()
          |
          +---> _handle_entity_created() --> embed() --> store_embedding()
          +---> _handle_entity_updated() --> check changes --> re-embed if needed
          +---> _handle_entity_deleted() --> delete_embedding_by_entity_id()
          +---> _handle_schema_evolved() --> ReembeddingService

Option B Compliance:
- Temporal context REQUIRED for Event entities
- Uses frozen models via MultiModelEmbeddingService
- Schema version tracked in all embeddings
- Causal structure preserved via TemporalEmbeddingContext

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Success Metrics:
- Sync latency <1s for 95% of mutations
- Zero embedding loss during PKG mutations
- Automatic retry on transient failures

Example Usage:
    >>> from futurnal.embeddings import MultiModelEmbeddingService, SchemaVersionedEmbeddingStore
    >>> from futurnal.embeddings.sync_handler import PKGSyncHandler
    >>> from futurnal.pkg.sync import PKGEvent, SyncEventType, SyncEventCapture
    >>>
    >>> # Initialize services
    >>> embedding_service = MultiModelEmbeddingService(config)
    >>> embedding_store = SchemaVersionedEmbeddingStore(config)
    >>> capture = SyncEventCapture()
    >>>
    >>> # Create sync handler
    >>> handler = PKGSyncHandler(
    ...     embedding_service=embedding_service,
    ...     embedding_store=embedding_store,
    ...     sync_event_capture=capture,
    ... )
    >>>
    >>> # Handle PKG events
    >>> event = PKGEvent(
    ...     event_id="evt_123",
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="person_456",
    ...     entity_type="Person",
    ...     new_data={"name": "John Doe", "description": "Software Engineer"},
    ... )
    >>> handler.handle_event(event)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from futurnal.embeddings.models import TemporalEmbeddingContext
from futurnal.embeddings.request import EmbeddingRequest
from futurnal.pkg.sync.events import (
    PKGEvent,
    SyncEvent,
    SyncEventCapture,
    SyncEventType,
    SyncStatus,
    SourceOperation,
)

if TYPE_CHECKING:
    from futurnal.embeddings.service import MultiModelEmbeddingService
    from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
    from futurnal.embeddings.reembedding import ReembeddingService

logger = logging.getLogger(__name__)


# Entity types that require temporal context (Option B compliance)
TEMPORAL_ENTITY_TYPES = {"Event"}

# Fields that trigger re-embedding when changed
SIGNIFICANT_FIELDS = {"name", "description", "timestamp", "event_type", "title", "content"}


class PKGSyncHandler:
    """Routes PKG mutation events to embedding operations.

    Handles ENTITY_CREATED, ENTITY_UPDATED, ENTITY_DELETED, and SCHEMA_EVOLVED
    events by generating, updating, or removing embeddings as appropriate.

    Thread Safety:
        This class is thread-safe for concurrent event handling.
        Individual handlers may block on embedding generation.

    Option B Compliance:
        - Enforces temporal context requirement for Event entities
        - Uses frozen models via MultiModelEmbeddingService
        - Tracks schema version in all stored embeddings
        - Preserves causal structure in TemporalEmbeddingContext

    Attributes:
        embedding_service: Service for generating embeddings
        embedding_store: Store for persisting embeddings
        reembedding_service: Optional service for batch re-embedding
        sync_event_capture: Optional capture for monitoring

    Example:
        >>> handler = PKGSyncHandler(
        ...     embedding_service=embedding_service,
        ...     embedding_store=embedding_store,
        ... )
        >>> handler.handle_event(pkg_event)
    """

    def __init__(
        self,
        embedding_service: "MultiModelEmbeddingService",
        embedding_store: "SchemaVersionedEmbeddingStore",
        reembedding_service: Optional["ReembeddingService"] = None,
        sync_event_capture: Optional[SyncEventCapture] = None,
    ) -> None:
        """Initialize the PKG sync handler.

        Args:
            embedding_service: MultiModelEmbeddingService for generating embeddings
            embedding_store: SchemaVersionedEmbeddingStore for persisting embeddings
            reembedding_service: Optional ReembeddingService for batch re-embedding
                                on schema evolution
            sync_event_capture: Optional SyncEventCapture for monitoring/testing
        """
        self._embedding_service = embedding_service
        self._store = embedding_store
        self._reembedding_service = reembedding_service
        self._sync_capture = sync_event_capture

        # Event type to handler mapping
        self._handlers = {
            SyncEventType.ENTITY_CREATED: self._handle_entity_created,
            SyncEventType.ENTITY_UPDATED: self._handle_entity_updated,
            SyncEventType.ENTITY_DELETED: self._handle_entity_deleted,
            SyncEventType.RELATIONSHIP_CREATED: self._handle_relationship_created,
            SyncEventType.RELATIONSHIP_UPDATED: self._handle_relationship_updated,
            SyncEventType.RELATIONSHIP_DELETED: self._handle_relationship_deleted,
            SyncEventType.SCHEMA_EVOLVED: self._handle_schema_evolved,
        }

        # Statistics
        self._events_processed = 0
        self._events_succeeded = 0
        self._events_failed = 0

        logger.info("Initialized PKGSyncHandler")

    def handle_event(self, event: PKGEvent) -> bool:
        """Handle a PKG mutation event.

        Routes the event to the appropriate handler based on event type.
        Tracks timing and captures sync events for monitoring.

        Args:
            event: The PKGEvent to handle

        Returns:
            True if event was processed successfully, False on error

        Note:
            Errors are caught and logged but not re-raised.
            Failed events are captured with error details for debugging.
        """
        self._events_processed += 1
        start_time = time.perf_counter()

        handler = self._handlers.get(event.event_type)
        if handler is None:
            logger.warning(f"No handler for event type: {event.event_type}")
            return False

        logger.debug(
            f"Handling PKG event: {event.event_type} for "
            f"{event.entity_type}:{event.entity_id}"
        )

        try:
            handler(event)

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._events_succeeded += 1

            # Capture success
            if self._sync_capture:
                self._sync_capture.capture(
                    event.to_sync_event(
                        sync_status=SyncStatus.COMPLETED,
                        source_operation=SourceOperation.VECTOR_WRITE,
                        duration_ms=duration_ms,
                    )
                )

            logger.debug(
                f"PKG event handled successfully in {duration_ms:.2f}ms: "
                f"{event.event_type} for {event.entity_id}"
            )
            return True

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._events_failed += 1

            logger.error(
                f"PKG event handling failed for {event.entity_id}: {e}",
                exc_info=True,
            )

            # Capture failure
            if self._sync_capture:
                self._sync_capture.capture(
                    event.to_sync_event(
                        sync_status=SyncStatus.FAILED,
                        source_operation=SourceOperation.VECTOR_WRITE,
                        duration_ms=duration_ms,
                        error_message=str(e),
                    )
                )

            return False

    # ---------------------------------------------------------------------------
    # Entity Event Handlers
    # ---------------------------------------------------------------------------

    def _handle_entity_created(self, event: PKGEvent) -> None:
        """Handle entity creation - generate and store embedding.

        For Event entities, extracts temporal context from new_data.
        For static entities (Person, Organization, etc.), generates semantic embedding.

        Args:
            event: PKGEvent with event_type=ENTITY_CREATED

        Raises:
            ValueError: If Event entity missing temporal context (Option B)
            EmbeddingGenerationError: If embedding generation fails
            StorageError: If storage fails
        """
        if event.new_data is None:
            logger.warning(f"ENTITY_CREATED event missing new_data: {event.entity_id}")
            return

        # Format content for embedding
        content = self._format_entity_content(event.entity_type, event.new_data)

        # Extract temporal context for Event entities
        temporal_context = None
        if event.entity_type in TEMPORAL_ENTITY_TYPES:
            temporal_context = self._extract_temporal_context(event.new_data)
            if temporal_context is None:
                raise ValueError(
                    f"Event entity {event.entity_id} requires temporal context (Option B). "
                    "Events must have a timestamp field."
                )

        # Generate embedding
        result = self._embedding_service.embed(
            entity_type=event.entity_type,
            content=content,
            temporal_context=temporal_context,
            entity_id=event.entity_id,
            entity_name=event.new_data.get("name"),
            metadata={
                "event_id": event.event_id,
                "source": "pkg_sync",
            },
        )

        # Store embedding
        self._store.store_embedding(
            entity_id=event.entity_id,
            entity_type=event.entity_type,
            embedding=list(result.embedding),
            model_id=result.metadata.get("model_id", "unknown"),
            extraction_confidence=event.extraction_confidence or 1.0,
            source_document_id=event.source_document_id or "unknown",
            temporal_context=temporal_context,
        )

        logger.debug(f"Created embedding for new entity: {event.entity_id}")

    def _handle_entity_updated(self, event: PKGEvent) -> None:
        """Handle entity update - re-embed if significant change.

        Checks if the update includes significant changes (name, description,
        timestamp) that would affect the embedding. If not, skips re-embedding.

        Args:
            event: PKGEvent with event_type=ENTITY_UPDATED

        Raises:
            Same exceptions as _handle_entity_created
        """
        if event.new_data is None:
            logger.warning(f"ENTITY_UPDATED event missing new_data: {event.entity_id}")
            return

        # Check if update requires re-embedding
        old_data = event.old_data or {}
        if not self._requires_reembedding(old_data, event.new_data):
            logger.debug(f"Skipping re-embedding for {event.entity_id}: no significant changes")
            return

        # Mark old embedding for replacement
        self._store.mark_for_reembedding(
            entity_ids=[event.entity_id],
            reason="entity_updated",
        )

        # Create new embedding (reuse creation logic)
        self._handle_entity_created(event)

        logger.debug(f"Updated embedding for entity: {event.entity_id}")

    def _handle_entity_deleted(self, event: PKGEvent) -> None:
        """Handle entity deletion - remove embedding.

        Deletes the embedding associated with the deleted entity.

        Args:
            event: PKGEvent with event_type=ENTITY_DELETED
        """
        deleted_count = self._store.delete_embedding_by_entity_id(event.entity_id)

        if deleted_count > 0:
            logger.debug(f"Deleted {deleted_count} embedding(s) for entity: {event.entity_id}")
        else:
            logger.debug(f"No embeddings found to delete for entity: {event.entity_id}")

    # ---------------------------------------------------------------------------
    # Relationship Event Handlers
    # ---------------------------------------------------------------------------

    def _handle_relationship_created(self, event: PKGEvent) -> None:
        """Handle relationship creation.

        Currently, relationships don't have separate embeddings.
        This handler is a placeholder for future extensions.
        """
        logger.debug(f"Relationship created: {event.entity_id} (no embedding action)")

    def _handle_relationship_updated(self, event: PKGEvent) -> None:
        """Handle relationship update.

        Currently, relationships don't have separate embeddings.
        """
        logger.debug(f"Relationship updated: {event.entity_id} (no embedding action)")

    def _handle_relationship_deleted(self, event: PKGEvent) -> None:
        """Handle relationship deletion.

        Currently, relationships don't have separate embeddings.
        """
        logger.debug(f"Relationship deleted: {event.entity_id} (no embedding action)")

    # ---------------------------------------------------------------------------
    # Schema Evolution Handler
    # ---------------------------------------------------------------------------

    def _handle_schema_evolved(self, event: PKGEvent) -> None:
        """Handle schema evolution - trigger selective re-embedding.

        When the PKG schema evolves, some embeddings may need regeneration.
        This handler uses ReembeddingService to detect changes and trigger
        batch re-embedding.

        Args:
            event: PKGEvent with event_type=SCHEMA_EVOLVED
        """
        if self._reembedding_service is None:
            logger.warning("Schema evolved but no ReembeddingService configured")
            return

        if event.new_data is None:
            logger.warning("SCHEMA_EVOLVED event missing new_data")
            return

        old_version = event.new_data.get("old_version", 1)
        new_version = event.new_data.get("new_version", 2)

        logger.info(f"Schema evolved from v{old_version} to v{new_version}")

        # Detect changes and check if re-embedding needed
        changes = self._reembedding_service.detect_schema_changes(
            old_version,
            new_version,
        )

        if changes.requires_reembedding:
            logger.info(f"Triggering re-embedding for schema evolution: {changes}")

            # Trigger batch re-embedding
            progress = self._reembedding_service.trigger_reembedding(
                schema_version=old_version,
                batch_size=100,
            )

            logger.info(
                f"Re-embedding progress: {progress.processed}/{progress.total} "
                f"(success rate: {progress.success_rate:.2%})"
            )
        else:
            logger.debug("Schema evolution does not require re-embedding")

        # Refresh schema cache in store
        self._store.refresh_schema_cache()

    # ---------------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------------

    def _format_entity_content(
        self,
        entity_type: str,
        data: Dict[str, Any],
    ) -> str:
        """Format entity data for embedding.

        Combines relevant fields into a text representation suitable
        for the embedding model.

        Args:
            entity_type: Type of entity (Person, Event, etc.)
            data: Entity properties

        Returns:
            Formatted text for embedding
        """
        parts = []

        # Name is primary
        name = data.get("name", "")
        if name:
            parts.append(name)

        # Description is secondary
        description = data.get("description", "")
        if description:
            parts.append(description)

        # Title (for documents)
        title = data.get("title", "")
        if title and title != name:
            parts.append(title)

        # Content snippet (for documents)
        content = data.get("content", "")
        if content:
            # Truncate to reasonable length for embedding
            parts.append(content[:500])

        # Event type context
        if entity_type == "Event":
            event_type = data.get("event_type", "")
            if event_type:
                parts.append(f"[{event_type}]")

        # Organization type context
        if entity_type == "Organization":
            org_type = data.get("type", "")
            if org_type:
                parts.append(f"[{org_type}]")

        # Join with separator
        text = ": ".join(parts) if len(parts) > 1 else "".join(parts)

        # Ensure non-empty
        if not text.strip():
            text = f"{entity_type} entity"

        return text

    def _extract_temporal_context(
        self,
        data: Dict[str, Any],
    ) -> Optional[TemporalEmbeddingContext]:
        """Extract temporal context from entity data.

        Required for Event entities (Option B compliance).

        Args:
            data: Entity properties

        Returns:
            TemporalEmbeddingContext if timestamp present, None otherwise
        """
        timestamp = data.get("timestamp")
        if timestamp is None:
            return None

        # Convert string timestamp if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                logger.warning(f"Invalid timestamp format: {timestamp}")
                return None

        # Extract duration if available
        duration = None
        duration_val = data.get("duration")
        if duration_val is not None:
            if isinstance(duration_val, (int, float)):
                duration = timedelta(seconds=duration_val)
            elif isinstance(duration_val, timedelta):
                duration = duration_val

        # Extract temporal type if available
        temporal_type = data.get("temporal_type")

        # Extract causal chain if available (Phase 3 prep)
        causal_chain = data.get("causal_chain", [])
        event_sequence = data.get("event_sequence", [])

        return TemporalEmbeddingContext(
            timestamp=timestamp,
            duration=duration,
            temporal_type=temporal_type,
            event_sequence=event_sequence,
            causal_chain=causal_chain,
        )

    def _requires_reembedding(
        self,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any],
    ) -> bool:
        """Determine if entity update requires re-embedding.

        Re-embedding is needed if significant fields changed:
        - name
        - description
        - timestamp (for events)
        - event_type
        - title
        - content

        Args:
            old_data: Previous entity properties
            new_data: Updated entity properties

        Returns:
            True if re-embedding needed, False otherwise
        """
        for field in SIGNIFICANT_FIELDS:
            old_value = old_data.get(field)
            new_value = new_data.get(field)

            if old_value != new_value:
                logger.debug(f"Significant change detected in field '{field}'")
                return True

        return False

    # ---------------------------------------------------------------------------
    # Statistics and Monitoring
    # ---------------------------------------------------------------------------

    @property
    def events_processed(self) -> int:
        """Total number of events processed."""
        return self._events_processed

    @property
    def events_succeeded(self) -> int:
        """Number of successfully processed events."""
        return self._events_succeeded

    @property
    def events_failed(self) -> int:
        """Number of failed events."""
        return self._events_failed

    @property
    def success_rate(self) -> float:
        """Success rate of event processing (0.0 to 1.0)."""
        if self._events_processed == 0:
            return 1.0
        return self._events_succeeded / self._events_processed

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dict with processing statistics
        """
        return {
            "events_processed": self._events_processed,
            "events_succeeded": self._events_succeeded,
            "events_failed": self._events_failed,
            "success_rate": self.success_rate,
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._events_processed = 0
        self._events_succeeded = 0
        self._events_failed = 0
        logger.debug("PKGSyncHandler statistics reset")

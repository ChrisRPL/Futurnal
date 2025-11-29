Summary: Implement synchronization between PKG mutations and embedding updates with event-driven architecture.

# 04 · PKG Synchronization

## Purpose
Implement bidirectional synchronization between PKG graph storage and vector embedding store, ensuring embeddings remain consistent with PKG mutations through event-driven updates.

**Criticality**: HIGH - Maintains consistency between graph and vector representations

## Scope
- Event-driven sync hooks for PKG mutations
- Incremental embedding updates
- Deletion propagation
- Consistency validation mechanisms
- Retry and error handling

## Requirements Alignment
- **Option B Requirement**: "Vector store must stay synchronized with evolving PKG"
- **Consistency Guarantee**: Embeddings reflect current PKG state
- **Performance Target**: <1s sync latency for typical mutations
- **Enables**: Accurate hybrid search results

## Component Design

### PKG Event Types

```python
from enum import Enum
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List


class PKGEventType(str, Enum):
    """Types of PKG mutations."""
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_DELETED = "entity_deleted"
    RELATIONSHIP_CREATED = "relationship_created"
    RELATIONSHIP_UPDATED = "relationship_updated"
    RELATIONSHIP_DELETED = "relationship_deleted"
    SCHEMA_EVOLVED = "schema_evolved"


class PKGEvent(BaseModel):
    """Event emitted by PKG on mutations."""
    event_id: str
    event_type: PKGEventType
    entity_id: str
    entity_type: str
    timestamp: datetime

    # Mutation details
    old_data: Optional[Dict[str, Any]] = None
    new_data: Optional[Dict[str, Any]] = None

    # Context
    source_document_id: Optional[str] = None
    extraction_confidence: Optional[float] = None
    schema_version: int
```

### Sync Event Handler

```python
class PKGSyncHandler:
    """
    Handles PKG mutation events and synchronizes embeddings.

    Architecture:
    - Subscribes to PKG mutation events
    - Routes events to appropriate handlers
    - Ensures eventual consistency
    """

    def __init__(
        self,
        embedding_service: MultiModelEmbeddingService,
        embedding_store: SchemaVersionedEmbeddingStore,
        event_queue
    ):
        self.embedding_service = embedding_service
        self.store = embedding_store
        self.queue = event_queue
        self.handlers = {
            PKGEventType.ENTITY_CREATED: self.handle_entity_created,
            PKGEventType.ENTITY_UPDATED: self.handle_entity_updated,
            PKGEventType.ENTITY_DELETED: self.handle_entity_deleted,
            PKGEventType.SCHEMA_EVOLVED: self.handle_schema_evolved,
        }

    def subscribe_to_pkg_events(self):
        """Subscribe to PKG mutation events."""
        # This would integrate with PKG event emitter
        self.queue.subscribe("pkg_mutations", self.handle_event)

    def handle_event(self, event: PKGEvent):
        """Route event to appropriate handler."""
        handler = self.handlers.get(event.event_type)

        if handler:
            try:
                handler(event)
            except Exception as e:
                self._handle_sync_error(event, e)
        else:
            print(f"No handler for event type: {event.event_type}")

    def handle_entity_created(self, event: PKGEvent):
        """
        Handle entity creation - generate and store embedding.

        Triggered by: New entity extracted from document
        """
        # Extract entity content
        entity_content = self._format_entity_content(event.new_data)

        # Extract temporal context if event
        temporal_context = None
        if event.entity_type == "Event":
            temporal_context = self._extract_temporal_context(event.new_data)

        # Generate embedding
        result = self.embedding_service.embed(
            entity_type=event.entity_type,
            content=entity_content,
            temporal_context=temporal_context,
            metadata={
                "event_id": event.event_id,
                "source": "pkg_sync"
            }
        )

        # Store embedding
        self.store.store_embedding(
            entity_id=event.entity_id,
            entity_type=event.entity_type,
            embedding=result.embedding,
            model_id=result.model_id,
            extraction_confidence=event.extraction_confidence or 0.0,
            source_document_id=event.source_document_id or "unknown"
        )

    def handle_entity_updated(self, event: PKGEvent):
        """
        Handle entity update - regenerate embedding.

        Triggered by: Entity properties changed, schema evolved
        """
        # Check if update requires re-embedding
        if not self._requires_reembedding(event.old_data, event.new_data):
            return  # Skip if insignificant change

        # Delete old embedding
        self.store.chroma.delete(
            collection_name="embeddings",
            where={"entity_id": event.entity_id}
        )

        # Create new embedding (reuse creation logic)
        self.handle_entity_created(event)

    def handle_entity_deleted(self, event: PKGEvent):
        """
        Handle entity deletion - remove embedding.

        Triggered by: Entity removed from PKG
        """
        # Delete embedding from store
        self.store.chroma.delete(
            collection_name="embeddings",
            where={"entity_id": event.entity_id}
        )

    def handle_schema_evolved(self, event: PKGEvent):
        """
        Handle schema evolution - trigger selective re-embedding.

        Triggered by: Schema version incremented
        """
        # Delegate to ReembeddingService
        from schema_versioned_storage import ReembeddingService

        reembedding_service = ReembeddingService(
            self.store,
            self.embedding_service,
            pkg_client=None  # Would be injected
        )

        # Detect changes
        old_version = event.new_data.get("old_version", 1)
        new_version = event.new_data.get("new_version", 2)

        changes = reembedding_service.detect_schema_changes(
            old_version,
            new_version
        )

        # Trigger re-embedding if needed
        if changes["requires_reembedding"]:
            reembedding_service.trigger_reembedding(
                schema_version=old_version,
                batch_size=100
            )

    def _requires_reembedding(
        self,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any]
    ) -> bool:
        """
        Determine if entity update requires re-embedding.

        Re-embed if:
        - Name changed
        - Description changed
        - Temporal properties changed (for events)
        - Significant property changes
        """
        # Check name/description changes
        if old_data.get("name") != new_data.get("name"):
            return True

        if old_data.get("description") != new_data.get("description"):
            return True

        # Check temporal changes for events
        if "timestamp" in old_data and old_data["timestamp"] != new_data.get("timestamp"):
            return True

        return False

    def _format_entity_content(self, entity_data: Dict[str, Any]) -> str:
        """Format entity data for embedding."""
        name = entity_data.get("name", "")
        description = entity_data.get("description", "")
        return f"{name}: {description}"

    def _extract_temporal_context(
        self,
        entity_data: Dict[str, Any]
    ) -> Optional[TemporalEmbeddingContext]:
        """Extract temporal context from event data."""
        if "timestamp" not in entity_data:
            return None

        return TemporalEmbeddingContext(
            timestamp=entity_data["timestamp"],
            duration=entity_data.get("duration"),
            temporal_type=entity_data.get("temporal_type"),
            event_sequence=entity_data.get("event_sequence", []),
            causal_chain=entity_data.get("causal_chain", [])
        )

    def _handle_sync_error(self, event: PKGEvent, error: Exception):
        """Handle synchronization errors with retry logic."""
        print(f"Sync error for event {event.event_id}: {error}")

        # Add to retry queue
        self.queue.publish("sync_retries", {
            "event": event.dict(),
            "error": str(error),
            "retry_count": 0,
            "max_retries": 3
        })
```

### Incremental Update Strategy

```python
class IncrementalEmbeddingUpdater:
    """
    Manages incremental embedding updates.

    Optimizes sync by batching updates and avoiding redundant work.
    """

    def __init__(
        self,
        sync_handler: PKGSyncHandler,
        batch_size: int = 50,
        batch_timeout_seconds: int = 5
    ):
        self.sync_handler = sync_handler
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_seconds
        self.pending_events: List[PKGEvent] = []
        self.last_flush = datetime.now()

    def add_event(self, event: PKGEvent):
        """
        Add event to pending batch.

        Flushes batch if size or timeout threshold reached.
        """
        self.pending_events.append(event)

        # Check flush conditions
        if (len(self.pending_events) >= self.batch_size or
            (datetime.now() - self.last_flush).seconds >= self.batch_timeout):
            self.flush_batch()

    def flush_batch(self):
        """Process pending events as batch."""
        if not self.pending_events:
            return

        # Group events by type for optimization
        grouped = {}
        for event in self.pending_events:
            event_type = event.event_type
            if event_type not in grouped:
                grouped[event_type] = []
            grouped[event_type].append(event)

        # Process each group
        for event_type, events in grouped.items():
            if event_type == PKGEventType.ENTITY_CREATED:
                self._batch_create_embeddings(events)
            elif event_type == PKGEventType.ENTITY_UPDATED:
                self._batch_update_embeddings(events)
            elif event_type == PKGEventType.ENTITY_DELETED:
                self._batch_delete_embeddings(events)

        # Clear pending
        self.pending_events = []
        self.last_flush = datetime.now()

    def _batch_create_embeddings(self, events: List[PKGEvent]):
        """Batch create embeddings for multiple entities."""
        # Prepare embedding requests
        requests = []
        for event in events:
            content = self.sync_handler._format_entity_content(event.new_data)
            temporal_context = None

            if event.entity_type == "Event":
                temporal_context = self.sync_handler._extract_temporal_context(
                    event.new_data
                )

            requests.append(EmbeddingRequest(
                entity_type=event.entity_type,
                content=content,
                temporal_context=temporal_context,
                metadata={"entity_id": event.entity_id}
            ))

        # Batch embed
        results = self.sync_handler.embedding_service.embed_batch(requests)

        # Store all embeddings
        for event, result in zip(events, results):
            self.sync_handler.store.store_embedding(
                entity_id=event.entity_id,
                entity_type=event.entity_type,
                embedding=result.embedding,
                model_id=result.model_id,
                extraction_confidence=event.extraction_confidence or 0.0,
                source_document_id=event.source_document_id or "unknown"
            )

    def _batch_update_embeddings(self, events: List[PKGEvent]):
        """Batch update embeddings."""
        # Filter to events requiring re-embedding
        to_update = [
            e for e in events
            if self.sync_handler._requires_reembedding(e.old_data, e.new_data)
        ]

        # Delete old embeddings
        entity_ids = [e.entity_id for e in to_update]
        for entity_id in entity_ids:
            self.sync_handler.store.chroma.delete(
                collection_name="embeddings",
                where={"entity_id": entity_id}
            )

        # Create new embeddings (reuse batch create)
        self._batch_create_embeddings(to_update)

    def _batch_delete_embeddings(self, events: List[PKGEvent]):
        """Batch delete embeddings."""
        entity_ids = [e.entity_id for e in events]

        for entity_id in entity_ids:
            self.sync_handler.store.chroma.delete(
                collection_name="embeddings",
                where={"entity_id": entity_id}
            )
```

### Consistency Validator

```python
class EmbeddingConsistencyValidator:
    """
    Validates consistency between PKG and embedding store.

    Detects and repairs inconsistencies.
    """

    def __init__(self, pkg_client, embedding_store):
        self.pkg = pkg_client
        self.store = embedding_store

    def validate_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency between PKG and embeddings.

        Returns: {
            "missing_embeddings": [...],
            "orphaned_embeddings": [...],
            "outdated_embeddings": [...],
            "total_entities": int,
            "total_embeddings": int
        }
        """
        # Get all entities from PKG
        pkg_entities = self.pkg.query("""
            MATCH (e)
            WHERE e.id IS NOT NULL
            RETURN e.id AS id, labels(e)[0] AS type
        """)

        pkg_entity_ids = {e["id"] for e in pkg_entities}

        # Get all embeddings
        all_embeddings = self.store.chroma.get(collection_name="embeddings")
        embedding_entity_ids = {m["entity_id"] for m in all_embeddings["metadatas"]}

        # Find missing/orphaned
        missing = pkg_entity_ids - embedding_entity_ids
        orphaned = embedding_entity_ids - pkg_entity_ids

        # Find outdated (schema version mismatch)
        current_schema = self.store.current_schema_version
        outdated = [
            m["entity_id"]
            for m in all_embeddings["metadatas"]
            if m["schema_version"] < current_schema
        ]

        return {
            "missing_embeddings": list(missing),
            "orphaned_embeddings": list(orphaned),
            "outdated_embeddings": outdated,
            "total_entities": len(pkg_entity_ids),
            "total_embeddings": len(embedding_entity_ids)
        }

    def repair_inconsistencies(self, validation_result: Dict[str, Any]):
        """Repair detected inconsistencies."""
        # Create missing embeddings
        for entity_id in validation_result["missing_embeddings"]:
            self._create_missing_embedding(entity_id)

        # Delete orphaned embeddings
        for entity_id in validation_result["orphaned_embeddings"]:
            self.store.chroma.delete(
                collection_name="embeddings",
                where={"entity_id": entity_id}
            )

        # Update outdated embeddings
        self.store.mark_for_reembedding(
            entity_ids=validation_result["outdated_embeddings"],
            reason="schema_version_mismatch"
        )

    def _create_missing_embedding(self, entity_id: str):
        """Create embedding for entity missing from store."""
        # Fetch entity from PKG
        entity = self.pkg.query("""
            MATCH (e)
            WHERE e.id = $entity_id
            RETURN e
        """, entity_id=entity_id)

        if not entity:
            return  # Entity deleted between check and repair

        # Create embedding event
        event = PKGEvent(
            event_id=f"repair_{entity_id}",
            event_type=PKGEventType.ENTITY_CREATED,
            entity_id=entity_id,
            entity_type=entity[0]["e"].labels[0],
            timestamp=datetime.now(),
            new_data=dict(entity[0]["e"]),
            schema_version=self.store.current_schema_version
        )

        # Handle event
        sync_handler = PKGSyncHandler(
            embedding_service=None,  # Would be injected
            embedding_store=self.store,
            event_queue=None
        )
        sync_handler.handle_entity_created(event)
```

## Implementation Details

### Week 4: PKG Synchronization

**Deliverable**: Working PKG-embedding synchronization

1. **Implement event infrastructure**:
   - PKG event emitter integration
   - Event queue setup
   - Event routing

2. **Implement `PKGSyncHandler`**:
   - Event handlers for all mutation types
   - Error handling and retry logic
   - Metrics tracking

3. **Implement incremental updates**:
   - Batching strategy
   - Flush triggers
   - Performance optimization

4. **Implement consistency validation**:
   - Periodic consistency checks
   - Automatic repair workflows
   - Monitoring and alerts

## Testing Strategy

```python
class TestPKGSynchronization:
    def test_entity_creation_sync(self):
        """Validate entity creation triggers embedding."""
        handler = PKGSyncHandler(embedding_service, store, queue)

        event = PKGEvent(
            event_id="test_1",
            event_type=PKGEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.now(),
            new_data={"name": "John Doe", "description": "Software Engineer"},
            schema_version=1
        )

        handler.handle_event(event)

        # Verify embedding created
        results = store.query_embeddings(
            query_vector=np.random.rand(768),
            top_k=10
        )

        assert any(r.metadata.entity_id == "person_123" for r in results)

    def test_consistency_validation(self):
        """Validate consistency checker detects issues."""
        validator = EmbeddingConsistencyValidator(pkg_client, store)

        result = validator.validate_consistency()

        assert "missing_embeddings" in result
        assert "orphaned_embeddings" in result
        assert result["total_entities"] >= 0
```

## Success Metrics

- ✅ Sync latency <1s for 95% of mutations
- ✅ Consistency validation passes (>99.9% consistency)
- ✅ Zero embedding loss during PKG mutations
- ✅ Automatic repair for inconsistencies
- ✅ Batch processing >100 updates/second

## Dependencies

- PKG Graph Storage with event emitters
- Multi-model architecture (02)
- Schema-versioned storage (03)
- Event queue infrastructure

## Next Steps

After PKG synchronization complete:
1. Enable quality evolution tracking (05-quality-evolution.md)
2. Production testing and validation (06-integration-testing.md)

**This module ensures embeddings remain synchronized with the evolving PKG.**

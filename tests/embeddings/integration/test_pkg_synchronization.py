"""PKG synchronization integration tests.

Validates:
- Entity creation triggers embedding generation
- Entity update triggers re-embedding (when significant)
- Entity deletion removes embeddings
- Consistency between PKG and embeddings

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/06-integration-testing.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pytest

from tests.embeddings.integration.conftest import (
    EmbeddingPipeline,
    MockPKGClient,
    create_test_entity,
)
from futurnal.pkg.sync.events import PKGEvent, SyncEventType, SyncStatus


class TestPKGSynchronization:
    """Validate PKG-embedding synchronization."""

    def test_entity_creation_triggers_embedding(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate PKG entity creation triggers embedding."""
        pipeline = embedding_pipeline

        # Simulate PKG entity creation event
        event = PKGEvent(
            event_id="test_creation",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_new",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "New Person", "description": "Test description"},
            extraction_confidence=0.85,
            schema_version=1,
        )

        # Handle event
        result = pipeline.sync_handler.handle_event(event)
        assert result is True

        # Verify embedding created
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )

        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "person_new" in entity_ids

    def test_entity_update_triggers_reembedding(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate PKG entity update triggers re-embedding."""
        pipeline = embedding_pipeline

        # Create initial entity
        create_event = PKGEvent(
            event_id="test_create",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_update",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "Original Name", "description": "Original"},
            extraction_confidence=0.8,
            schema_version=1,
        )
        pipeline.sync_handler.handle_event(create_event)

        # Update entity (significant change - name)
        update_event = PKGEvent(
            event_id="test_update",
            event_type=SyncEventType.ENTITY_UPDATED,
            entity_id="person_update",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            old_data={"name": "Original Name", "description": "Original"},
            new_data={"name": "Updated Name", "description": "Updated description"},
            extraction_confidence=0.9,
            schema_version=1,
        )
        result = pipeline.sync_handler.handle_event(update_event)

        assert result is True

        # Verify entity still has embedding
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )

        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "person_update" in entity_ids

    def test_entity_deletion_removes_embedding(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate PKG entity deletion removes embedding."""
        pipeline = embedding_pipeline

        # Create entity
        create_event = PKGEvent(
            event_id="test_create_del",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_delete",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "To Delete", "description": "Will be deleted"},
            schema_version=1,
        )
        pipeline.sync_handler.handle_event(create_event)

        # Verify embedding exists
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )
        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "person_delete" in entity_ids

        # Delete entity
        delete_event = PKGEvent(
            event_id="test_delete",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="person_delete",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )
        result = pipeline.sync_handler.handle_event(delete_event)

        assert result is True

        # Verify embedding removed
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )

        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "person_delete" not in entity_ids

    def test_event_entity_requires_temporal_context(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate Event entity fails without temporal context (Option B)."""
        pipeline = embedding_pipeline

        # Try to create Event without timestamp
        event = PKGEvent(
            event_id="test_event_no_temporal",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="event_no_time",
            entity_type="Event",
            timestamp=datetime.utcnow(),
            new_data={"name": "Meeting", "event_type": "meeting"},  # No timestamp!
            schema_version=1,
        )

        result = pipeline.sync_handler.handle_event(event)

        # Should fail due to missing temporal context
        assert result is False

    def test_event_entity_with_temporal_context_succeeds(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate Event entity succeeds with temporal context."""
        pipeline = embedding_pipeline

        event = PKGEvent(
            event_id="test_event_with_temporal",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="event_with_time",
            entity_type="Event",
            timestamp=datetime.utcnow(),
            new_data={
                "name": "Team Meeting",
                "event_type": "meeting",
                "timestamp": datetime(2024, 1, 15, 14, 30).isoformat(),
                "duration": 3600,
            },
            schema_version=1,
        )

        result = pipeline.sync_handler.handle_event(event)
        assert result is True

        # Verify embedding was created
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )
        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "event_with_time" in entity_ids

    def test_sync_statistics_tracking(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate sync handler tracks statistics."""
        pipeline = embedding_pipeline

        # Process several events
        for i in range(5):
            event = PKGEvent(
                event_id=f"test_stats_{i}",
                event_type=SyncEventType.ENTITY_CREATED,
                entity_id=f"person_stats_{i}",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                new_data={"name": f"Person {i}"},
                schema_version=1,
            )
            pipeline.sync_handler.handle_event(event)

        stats = pipeline.sync_handler.get_statistics()

        assert stats["events_processed"] >= 5
        assert stats["events_succeeded"] >= 5
        assert stats["success_rate"] >= 1.0

    def test_sync_event_capture(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate sync events are captured."""
        pipeline = embedding_pipeline

        event = PKGEvent(
            event_id="test_capture",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_capture",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "Capture Test"},
            schema_version=1,
        )

        pipeline.sync_handler.handle_event(event)

        # Check captured events
        captured = pipeline.sync_capture.get_events_for_entity("person_capture")
        assert len(captured) >= 1
        assert captured[0].vector_sync_status == SyncStatus.COMPLETED

    def test_batch_entity_creation(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate batch entity creation triggers embeddings."""
        pipeline = embedding_pipeline

        # Create multiple entities
        for i in range(10):
            event = PKGEvent(
                event_id=f"batch_create_{i}",
                event_type=SyncEventType.ENTITY_CREATED,
                entity_id=f"batch_person_{i}",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                new_data={"name": f"Batch Person {i}", "description": f"Description {i}"},
                schema_version=1,
            )
            pipeline.sync_handler.handle_event(event)

        # Verify all embeddings created
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )

        entity_ids = [r.metadata.get("entity_id") for r in results]
        for i in range(10):
            assert f"batch_person_{i}" in entity_ids

    def test_mixed_entity_types_sync(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate sync works for mixed entity types."""
        pipeline = embedding_pipeline

        entities = [
            ("Person", "sync_person", {"name": "John", "description": "Developer"}),
            ("Organization", "sync_org", {"name": "TechCorp", "description": "Company"}),
            ("Event", "sync_event", {"name": "Meeting", "timestamp": datetime.utcnow().isoformat()}),
            ("Concept", "sync_concept", {"name": "AI", "description": "Technology"}),
        ]

        for entity_type, entity_id, data in entities:
            event = PKGEvent(
                event_id=f"mixed_{entity_id}",
                event_type=SyncEventType.ENTITY_CREATED,
                entity_id=entity_id,
                entity_type=entity_type,
                timestamp=datetime.utcnow(),
                new_data=data,
                schema_version=1,
            )
            result = pipeline.sync_handler.handle_event(event)
            assert result is True, f"Failed to sync {entity_type}"

        # Verify all created
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )

        entity_ids = [r.metadata.get("entity_id") for r in results]
        for _, entity_id, _ in entities:
            assert entity_id in entity_ids

    def test_failed_event_tracked(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate failed events are tracked correctly."""
        pipeline = embedding_pipeline

        # Create event that will fail (Event without timestamp)
        event = PKGEvent(
            event_id="test_failed",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="event_failed",
            entity_type="Event",
            timestamp=datetime.utcnow(),
            new_data={"name": "No Timestamp Event"},  # Missing timestamp
            schema_version=1,
        )

        result = pipeline.sync_handler.handle_event(event)
        assert result is False

        stats = pipeline.sync_handler.get_statistics()
        assert stats["events_failed"] >= 1

    def test_update_only_changes_affected_entity(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate update only affects the targeted entity."""
        pipeline = embedding_pipeline

        # Create two entities
        for entity_id in ["entity_a", "entity_b"]:
            event = PKGEvent(
                event_id=f"create_{entity_id}",
                event_type=SyncEventType.ENTITY_CREATED,
                entity_id=entity_id,
                entity_type="Person",
                timestamp=datetime.utcnow(),
                new_data={"name": f"Original {entity_id}"},
                schema_version=1,
            )
            pipeline.sync_handler.handle_event(event)

        # Update only entity_a
        update_event = PKGEvent(
            event_id="update_entity_a",
            event_type=SyncEventType.ENTITY_UPDATED,
            entity_id="entity_a",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "Updated entity_a"},
            schema_version=1,
        )
        pipeline.sync_handler.handle_event(update_event)

        # Both should still exist
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=100,
        )

        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "entity_a" in entity_ids
        assert "entity_b" in entity_ids

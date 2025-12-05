"""Tests for PKGSyncHandler.

Tests PKG ↔ Embedding synchronization event handling including:
- Entity creation triggers embedding generation
- Entity update triggers re-embedding when significant
- Entity deletion removes embeddings
- Schema evolution triggers batch re-embedding
- Temporal context enforcement for Event entities

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Any, Dict

import numpy as np
import pytest

from futurnal.pkg.sync.events import PKGEvent, SyncEventType, SyncEventCapture, SyncStatus
from futurnal.embeddings.sync_handler import PKGSyncHandler, TEMPORAL_ENTITY_TYPES
from futurnal.embeddings.models import EmbeddingResult, TemporalEmbeddingContext, EmbeddingEntityType


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_service():
    """Create a mock MultiModelEmbeddingService."""
    service = MagicMock()

    # Mock embed() to return a result
    def mock_embed(**kwargs):
        embedding = np.random.rand(768).tolist()
        return EmbeddingResult(
            embedding=embedding,
            entity_type=EmbeddingEntityType.STATIC_ENTITY,
            model_version="test-model-v1",
            embedding_dimension=768,
            generation_time_ms=10.0,
            metadata={
                "model_id": "test-model",
                "duration_ms": 10.0,
            },
        )

    service.embed = MagicMock(side_effect=mock_embed)
    service.embed_batch = MagicMock(return_value=[mock_embed() for _ in range(5)])

    return service


@pytest.fixture
def mock_embedding_store():
    """Create a mock SchemaVersionedEmbeddingStore."""
    store = MagicMock()
    store.store_embedding = MagicMock(return_value="emb_123")
    store.delete_embedding_by_entity_id = MagicMock(return_value=1)
    store.mark_for_reembedding = MagicMock(return_value=1)
    store.current_schema_version = 1
    store.refresh_schema_cache = MagicMock()
    return store


@pytest.fixture
def mock_reembedding_service():
    """Create a mock ReembeddingService."""
    service = MagicMock()
    service.detect_schema_changes = MagicMock(return_value=MagicMock(
        requires_reembedding=False,
        new_entity_types=[],
        removed_entity_types=[],
    ))
    service.trigger_reembedding = MagicMock(return_value=MagicMock(
        processed=10,
        total=10,
        success_rate=1.0,
    ))
    return service


@pytest.fixture
def sync_capture():
    """Create a SyncEventCapture for monitoring."""
    return SyncEventCapture()


@pytest.fixture
def sync_handler(
    mock_embedding_service,
    mock_embedding_store,
    mock_reembedding_service,
    sync_capture,
):
    """Create a PKGSyncHandler with mocked dependencies."""
    return PKGSyncHandler(
        embedding_service=mock_embedding_service,
        embedding_store=mock_embedding_store,
        reembedding_service=mock_reembedding_service,
        sync_event_capture=sync_capture,
    )


# -----------------------------------------------------------------------------
# Entity Creation Tests
# -----------------------------------------------------------------------------


class TestEntityCreation:
    """Test ENTITY_CREATED event handling."""

    def test_handle_entity_created_static_entity(self, sync_handler, mock_embedding_service, mock_embedding_store):
        """Person entity creation triggers embedding without temporal context."""
        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "John Doe", "description": "Software Engineer"},
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_embedding_service.embed.assert_called_once()
        mock_embedding_store.store_embedding.assert_called_once()

        # Verify no temporal context for Person
        call_kwargs = mock_embedding_service.embed.call_args[1]
        assert call_kwargs["temporal_context"] is None

    def test_handle_entity_created_event_requires_temporal(self, sync_handler):
        """Event entity creation fails without temporal context (Option B)."""
        event = PKGEvent(
            event_id="evt_002",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="event_123",
            entity_type="Event",
            timestamp=datetime.utcnow(),
            new_data={"name": "Meeting", "event_type": "meeting"},  # No timestamp!
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is False  # Should fail due to missing temporal context
        assert sync_handler.events_failed == 1

    def test_handle_entity_created_event_with_temporal(self, sync_handler, mock_embedding_service, mock_embedding_store):
        """Event entity creation succeeds with temporal context."""
        event = PKGEvent(
            event_id="evt_003",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="event_456",
            entity_type="Event",
            timestamp=datetime.utcnow(),
            new_data={
                "name": "Team Meeting",
                "event_type": "meeting",
                "timestamp": datetime(2024, 1, 15, 14, 30).isoformat(),
                "duration": 3600,  # 1 hour in seconds
            },
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_embedding_service.embed.assert_called_once()

        # Verify temporal context was extracted
        call_kwargs = mock_embedding_service.embed.call_args[1]
        assert call_kwargs["temporal_context"] is not None

    def test_handle_entity_created_missing_data(self, sync_handler):
        """Creation event with missing new_data is handled gracefully."""
        event = PKGEvent(
            event_id="evt_004",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_789",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data=None,  # Missing data
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        # Should not fail but should skip processing
        assert sync_handler.events_succeeded == 1


# -----------------------------------------------------------------------------
# Entity Update Tests
# -----------------------------------------------------------------------------


class TestEntityUpdate:
    """Test ENTITY_UPDATED event handling."""

    def test_handle_entity_updated_significant_change(self, sync_handler, mock_embedding_service, mock_embedding_store):
        """Entity update with name change triggers re-embedding."""
        event = PKGEvent(
            event_id="evt_010",
            event_type=SyncEventType.ENTITY_UPDATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            old_data={"name": "John Doe", "description": "Engineer"},
            new_data={"name": "John Smith", "description": "Engineer"},  # Name changed
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_embedding_store.mark_for_reembedding.assert_called_once()
        mock_embedding_service.embed.assert_called_once()

    def test_handle_entity_updated_no_significant_change(self, sync_handler, mock_embedding_service, mock_embedding_store):
        """Entity update without significant change skips re-embedding."""
        event = PKGEvent(
            event_id="evt_011",
            event_type=SyncEventType.ENTITY_UPDATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            old_data={"name": "John Doe", "aliases": []},
            new_data={"name": "John Doe", "aliases": ["JD"]},  # Only aliases changed
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_embedding_store.mark_for_reembedding.assert_not_called()
        mock_embedding_service.embed.assert_not_called()

    def test_handle_entity_updated_description_change(self, sync_handler, mock_embedding_service):
        """Entity update with description change triggers re-embedding."""
        event = PKGEvent(
            event_id="evt_012",
            event_type=SyncEventType.ENTITY_UPDATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            old_data={"name": "John Doe", "description": "Engineer"},
            new_data={"name": "John Doe", "description": "Senior Engineer"},  # Description changed
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_embedding_service.embed.assert_called_once()


# -----------------------------------------------------------------------------
# Entity Deletion Tests
# -----------------------------------------------------------------------------


class TestEntityDeletion:
    """Test ENTITY_DELETED event handling."""

    def test_handle_entity_deleted(self, sync_handler, mock_embedding_store):
        """Entity deletion removes embedding from store."""
        event = PKGEvent(
            event_id="evt_020",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            old_data={"name": "John Doe"},
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_embedding_store.delete_embedding_by_entity_id.assert_called_once_with("person_123")

    def test_handle_entity_deleted_not_found(self, sync_handler, mock_embedding_store):
        """Entity deletion handles case where embedding doesn't exist."""
        mock_embedding_store.delete_embedding_by_entity_id.return_value = 0

        event = PKGEvent(
            event_id="evt_021",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="nonexistent",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is True  # Should succeed even if nothing deleted


# -----------------------------------------------------------------------------
# Schema Evolution Tests
# -----------------------------------------------------------------------------


class TestSchemaEvolution:
    """Test SCHEMA_EVOLVED event handling."""

    def test_handle_schema_evolved_triggers_reembedding(self, sync_handler, mock_reembedding_service, mock_embedding_store):
        """Schema evolution triggers re-embedding when needed."""
        mock_reembedding_service.detect_schema_changes.return_value = MagicMock(
            requires_reembedding=True,
            new_entity_types=["NewType"],
        )

        event = PKGEvent(
            event_id="evt_030",
            event_type=SyncEventType.SCHEMA_EVOLVED,
            entity_id="schema_v2",
            entity_type="SchemaVersion",
            timestamp=datetime.utcnow(),
            new_data={"old_version": 1, "new_version": 2},
            schema_version=2,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_reembedding_service.trigger_reembedding.assert_called_once()
        mock_embedding_store.refresh_schema_cache.assert_called_once()

    def test_handle_schema_evolved_no_reembedding_needed(self, sync_handler, mock_reembedding_service, mock_embedding_store):
        """Schema evolution skips re-embedding when not needed."""
        mock_reembedding_service.detect_schema_changes.return_value = MagicMock(
            requires_reembedding=False,
        )

        event = PKGEvent(
            event_id="evt_031",
            event_type=SyncEventType.SCHEMA_EVOLVED,
            entity_id="schema_v2",
            entity_type="SchemaVersion",
            timestamp=datetime.utcnow(),
            new_data={"old_version": 1, "new_version": 2},
            schema_version=2,
        )

        result = sync_handler.handle_event(event)

        assert result is True
        mock_reembedding_service.trigger_reembedding.assert_not_called()


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_tracking(self, sync_handler):
        """Statistics are properly tracked."""
        # Process some events
        event1 = PKGEvent(
            event_id="evt_040",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_1",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "Alice"},
            schema_version=1,
        )

        event2 = PKGEvent(
            event_id="evt_041",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="person_2",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        sync_handler.handle_event(event1)
        sync_handler.handle_event(event2)

        assert sync_handler.events_processed == 2
        assert sync_handler.events_succeeded == 2
        assert sync_handler.events_failed == 0
        assert sync_handler.success_rate == 1.0

    def test_get_statistics(self, sync_handler):
        """get_statistics returns proper dict."""
        stats = sync_handler.get_statistics()

        assert "events_processed" in stats
        assert "events_succeeded" in stats
        assert "events_failed" in stats
        assert "success_rate" in stats

    def test_reset_statistics(self, sync_handler):
        """Statistics can be reset."""
        # Process an event
        event = PKGEvent(
            event_id="evt_050",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="person_1",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )
        sync_handler.handle_event(event)

        assert sync_handler.events_processed == 1

        sync_handler.reset_statistics()

        assert sync_handler.events_processed == 0
        assert sync_handler.events_succeeded == 0


# -----------------------------------------------------------------------------
# Sync Event Capture Tests
# -----------------------------------------------------------------------------


class TestSyncEventCapture:
    """Test integration with SyncEventCapture."""

    def test_successful_event_captured(self, sync_handler, sync_capture):
        """Successful events are captured with COMPLETED status."""
        event = PKGEvent(
            event_id="evt_060",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="person_1",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        sync_handler.handle_event(event)

        captured = sync_capture.get_events_for_entity("person_1")
        assert len(captured) == 1
        assert captured[0].vector_sync_status == SyncStatus.COMPLETED

    def test_failed_event_captured(self, sync_handler, sync_capture, mock_embedding_store):
        """Failed events are captured with FAILED status."""
        # Make store raise an error
        mock_embedding_store.delete_embedding_by_entity_id.side_effect = Exception("Test error")

        event = PKGEvent(
            event_id="evt_061",
            event_type=SyncEventType.ENTITY_DELETED,
            entity_id="person_1",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        result = sync_handler.handle_event(event)

        assert result is False
        captured = sync_capture.get_events_for_entity("person_1")
        assert len(captured) == 1
        assert captured[0].vector_sync_status == SyncStatus.FAILED
        assert captured[0].error_message is not None


# -----------------------------------------------------------------------------
# Helper Method Tests
# -----------------------------------------------------------------------------


class TestHelperMethods:
    """Test internal helper methods."""

    def test_format_entity_content_person(self, sync_handler):
        """Format Person content correctly."""
        data = {"name": "John Doe", "description": "Software Engineer"}
        content = sync_handler._format_entity_content("Person", data)

        assert "John Doe" in content
        assert "Software Engineer" in content

    def test_format_entity_content_event(self, sync_handler):
        """Format Event content includes event type."""
        data = {"name": "Meeting", "event_type": "meeting", "description": "Team sync"}
        content = sync_handler._format_entity_content("Event", data)

        assert "Meeting" in content
        assert "[meeting]" in content

    def test_extract_temporal_context_valid(self, sync_handler):
        """Extract temporal context from valid data."""
        data = {
            "timestamp": datetime(2024, 1, 15, 14, 30).isoformat(),
            "duration": 3600,
            "temporal_type": "DURING",
        }

        context = sync_handler._extract_temporal_context(data)

        assert context is not None
        assert context.timestamp == datetime(2024, 1, 15, 14, 30)
        assert context.duration == timedelta(seconds=3600)

    def test_extract_temporal_context_missing_timestamp(self, sync_handler):
        """Extract temporal context returns None without timestamp."""
        data = {"name": "Meeting"}

        context = sync_handler._extract_temporal_context(data)

        assert context is None

    def test_requires_reembedding_name_change(self, sync_handler):
        """Name change requires re-embedding."""
        old = {"name": "Old Name"}
        new = {"name": "New Name"}

        assert sync_handler._requires_reembedding(old, new) is True

    def test_requires_reembedding_no_change(self, sync_handler):
        """No significant change does not require re-embedding."""
        old = {"name": "Same", "aliases": []}
        new = {"name": "Same", "aliases": ["alias"]}

        assert sync_handler._requires_reembedding(old, new) is False

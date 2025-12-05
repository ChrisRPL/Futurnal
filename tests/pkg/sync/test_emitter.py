"""Tests for PKGEventEmitter and EmittingEntityRepository.

Tests event emission infrastructure including:
- PKGEventEmitter event handling
- EmittingEntityRepository wrapper behavior
- SyncEventCapture integration
- Error handling and statistics

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch, call
from typing import List

import pytest

from futurnal.pkg.sync.events import PKGEvent, SyncEventType, SyncEventCapture, SyncStatus
from futurnal.pkg.sync.emitter import PKGEventEmitter
from futurnal.pkg.repository.emitting_wrapper import (
    EmittingEntityRepository,
    EmittingRelationshipRepository,
)
from futurnal.pkg.schema.models import PersonNode, EventNode, OrganizationNode


# -----------------------------------------------------------------------------
# PKGEventEmitter Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def captured_events() -> List[PKGEvent]:
    """Storage for captured events."""
    return []


@pytest.fixture
def mock_event_handler(captured_events):
    """Create a mock event handler that stores events."""
    def handler(event: PKGEvent) -> bool:
        captured_events.append(event)
        return True
    return handler


@pytest.fixture
def sync_capture():
    """Create a SyncEventCapture for monitoring."""
    return SyncEventCapture()


@pytest.fixture
def emitter(mock_event_handler, sync_capture):
    """Create a PKGEventEmitter."""
    return PKGEventEmitter(
        event_handler=mock_event_handler,
        sync_event_capture=sync_capture,
    )


# -----------------------------------------------------------------------------
# PKGEventEmitter Tests
# -----------------------------------------------------------------------------


class TestPKGEventEmitter:
    """Test PKGEventEmitter functionality."""

    def test_emit_calls_handler(self, emitter, captured_events):
        """emit() calls the event handler."""
        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            new_data={"name": "Alice"},
            schema_version=1,
        )

        result = emitter.emit(event)

        assert result is True
        assert len(captured_events) == 1
        assert captured_events[0].entity_id == "person_123"

    def test_emit_updates_statistics(self, emitter):
        """emit() updates statistics."""
        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        emitter.emit(event)

        stats = emitter.get_statistics()
        assert stats["emit_count"] == 1
        assert stats["success_rate"] == 1.0

    def test_emit_captures_to_sync_capture(self, emitter, sync_capture):
        """emit() records to SyncEventCapture."""
        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        emitter.emit(event)

        captured = sync_capture.get_events_for_entity("person_123")
        assert len(captured) == 1

    def test_emit_batch_processes_all(self, emitter, captured_events):
        """emit_batch() processes all events."""
        events = [
            PKGEvent(
                event_id=f"evt_{i}",
                event_type=SyncEventType.ENTITY_CREATED,
                entity_id=f"person_{i}",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                schema_version=1,
            )
            for i in range(5)
        ]

        count = emitter.emit_batch(events)

        assert count == 5
        assert len(captured_events) == 5

    def test_emit_handler_failure(self, sync_capture):
        """emit() handles handler failures gracefully."""
        def failing_handler(event: PKGEvent) -> bool:
            raise ValueError("Handler error")

        emitter = PKGEventEmitter(
            event_handler=failing_handler,
            sync_event_capture=sync_capture,
        )

        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        result = emitter.emit(event)

        assert result is False
        stats = emitter.get_statistics()
        assert stats["error_count"] == 1

    def test_emit_without_sync_capture(self, mock_event_handler, captured_events):
        """emit() works without SyncEventCapture."""
        emitter = PKGEventEmitter(
            event_handler=mock_event_handler,
            sync_event_capture=None,
        )

        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )

        result = emitter.emit(event)

        assert result is True
        assert len(captured_events) == 1


# -----------------------------------------------------------------------------
# EmittingEntityRepository Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_entity_repo():
    """Create a mock EntityRepository."""
    repo = MagicMock()

    # Mock create_entity to return entity ID
    repo.create_entity = MagicMock(return_value="person_123")

    # Mock update_entity to return updated entity
    repo.update_entity = MagicMock(
        return_value=PersonNode(name="Updated", entity_id="person_123")
    )

    # Mock delete_entity
    repo.delete_entity = MagicMock(return_value=True)

    # Mock get_entity for capturing old data
    repo.get_entity = MagicMock(
        return_value=PersonNode(name="Original", entity_id="person_123")
    )

    return repo


@pytest.fixture
def mock_schema_manager():
    """Create a mock SchemaVersionManager."""
    manager = MagicMock()
    # The wrapper calls get_current_version().version
    version_node = MagicMock()
    version_node.version = 1
    manager.get_current_version = MagicMock(return_value=version_node)
    return manager


@pytest.fixture
def emitting_entity_repo(mock_entity_repo, emitter, mock_schema_manager):
    """Create an EmittingEntityRepository."""
    return EmittingEntityRepository(
        repo=mock_entity_repo,
        emitter=emitter,
        schema_manager=mock_schema_manager,
    )


# -----------------------------------------------------------------------------
# EmittingEntityRepository Tests
# -----------------------------------------------------------------------------


class TestEmittingEntityRepository:
    """Test EmittingEntityRepository wrapper."""

    def test_create_entity_emits_event(
        self, emitting_entity_repo, captured_events, mock_entity_repo
    ):
        """create_entity emits ENTITY_CREATED event."""
        entity = PersonNode(name="Alice")

        entity_id = emitting_entity_repo.create_entity(entity)

        assert entity_id == "person_123"
        mock_entity_repo.create_entity.assert_called_once_with(entity)

        # Check event emitted
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == SyncEventType.ENTITY_CREATED
        assert event.entity_type == "Person"

    def test_update_entity_emits_event(
        self, emitting_entity_repo, captured_events, mock_entity_repo
    ):
        """update_entity emits ENTITY_UPDATED event with old_data."""
        result = emitting_entity_repo.update_entity(
            "person_123",
            {"name": "New Name"},
        )

        assert result.name == "Updated"

        # Check event emitted
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == SyncEventType.ENTITY_UPDATED
        assert event.old_data is not None  # Should capture old data
        # new_data contains the full updated entity (not just the update dict)
        assert "name" in event.new_data

    def test_delete_entity_emits_event(
        self, emitting_entity_repo, captured_events, mock_entity_repo
    ):
        """delete_entity emits ENTITY_DELETED event."""
        result = emitting_entity_repo.delete_entity("person_123")

        assert result is True

        # Check event emitted
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == SyncEventType.ENTITY_DELETED
        assert event.entity_id == "person_123"

    def test_create_entity_failure_no_emit(
        self, emitting_entity_repo, captured_events, mock_entity_repo
    ):
        """No event emitted if create_entity fails."""
        mock_entity_repo.create_entity.side_effect = ValueError("Creation failed")

        with pytest.raises(ValueError):
            emitting_entity_repo.create_entity(PersonNode(name="Alice"))

        # No event should be emitted
        assert len(captured_events) == 0

    def test_delegates_find_entities(self, emitting_entity_repo, mock_entity_repo):
        """find_entities delegates to underlying repo."""
        mock_entity_repo.find_entities.return_value = [
            PersonNode(name="Alice", entity_id="p1")
        ]

        result = emitting_entity_repo.find_entities("Person", limit=10)

        # Verify delegation happened (the actual call signature may vary)
        mock_entity_repo.find_entities.assert_called()
        assert len(result) == 1

    def test_delegates_get_entity(self, emitting_entity_repo, mock_entity_repo):
        """get_entity delegates to underlying repo."""
        result = emitting_entity_repo.get_entity("person_123")

        mock_entity_repo.get_entity.assert_called()

    def test_event_includes_schema_version(
        self, emitting_entity_repo, captured_events, mock_schema_manager
    ):
        """Events include current schema version."""
        # Update the mock to return version 5
        version_node = MagicMock()
        version_node.version = 5
        mock_schema_manager.get_current_version.return_value = version_node

        emitting_entity_repo.create_entity(PersonNode(name="Alice"))

        event = captured_events[0]
        assert event.schema_version == 5


# -----------------------------------------------------------------------------
# EmittingRelationshipRepository Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_relationship_repo():
    """Create a mock RelationshipRepository."""
    repo = MagicMock()
    repo.create_relationship = MagicMock(return_value="rel_123")
    repo.delete_relationship = MagicMock(return_value=True)
    repo.get_relationship = MagicMock(return_value={
        "id": "rel_123",
        "type": "KNOWS",
        "source_id": "person_1",
        "target_id": "person_2",
    })
    return repo


@pytest.fixture
def emitting_relationship_repo(mock_relationship_repo, emitter, mock_schema_manager):
    """Create an EmittingRelationshipRepository."""
    return EmittingRelationshipRepository(
        repo=mock_relationship_repo,
        emitter=emitter,
        schema_manager=mock_schema_manager,
    )


class TestEmittingRelationshipRepository:
    """Test EmittingRelationshipRepository wrapper."""

    def test_create_relationship_emits_event(
        self, emitting_relationship_repo, captured_events
    ):
        """create_relationship emits RELATIONSHIP_CREATED event."""
        rel_id = emitting_relationship_repo.create_relationship(
            "person_1", "KNOWS", "person_2"
        )

        assert rel_id == "rel_123"
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == SyncEventType.RELATIONSHIP_CREATED

    def test_delete_relationship_emits_event(
        self, emitting_relationship_repo, captured_events
    ):
        """delete_relationship emits RELATIONSHIP_DELETED event."""
        result = emitting_relationship_repo.delete_relationship("rel_123")

        assert result is True
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == SyncEventType.RELATIONSHIP_DELETED


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestEmitterIntegration:
    """Integration tests for emitter infrastructure."""

    def test_full_entity_lifecycle(
        self, emitting_entity_repo, captured_events, mock_entity_repo, sync_capture
    ):
        """Full entity lifecycle emits correct events."""
        # Create
        entity_id = emitting_entity_repo.create_entity(PersonNode(name="Alice"))
        assert captured_events[-1].event_type == SyncEventType.ENTITY_CREATED

        # Update
        emitting_entity_repo.update_entity(entity_id, {"name": "Alice Smith"})
        assert captured_events[-1].event_type == SyncEventType.ENTITY_UPDATED

        # Delete
        emitting_entity_repo.delete_entity(entity_id)
        assert captured_events[-1].event_type == SyncEventType.ENTITY_DELETED

        # All events captured
        assert len(captured_events) == 3

        # All recorded in sync capture
        events_for_entity = sync_capture.get_events_for_entity(entity_id)
        assert len(events_for_entity) == 3

    def test_batch_entity_creation(
        self, emitting_entity_repo, captured_events, mock_entity_repo
    ):
        """Multiple entity creations emit multiple events."""
        for i in range(5):
            mock_entity_repo.create_entity.return_value = f"person_{i}"
            emitting_entity_repo.create_entity(PersonNode(name=f"Person {i}"))

        assert len(captured_events) == 5
        assert all(e.event_type == SyncEventType.ENTITY_CREATED for e in captured_events)


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestEmitterStatistics:
    """Test emitter statistics."""

    def test_emitter_statistics_tracking(self, emitter):
        """Statistics are properly tracked."""
        # Emit some events
        for i in range(5):
            event = PKGEvent(
                event_id=f"evt_{i}",
                event_type=SyncEventType.ENTITY_CREATED,
                entity_id=f"person_{i}",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                schema_version=1,
            )
            emitter.emit(event)

        stats = emitter.get_statistics()

        assert stats["emit_count"] == 5
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0

    def test_reset_statistics(self, emitter):
        """Statistics can be reset."""
        event = PKGEvent(
            event_id="evt_001",
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="person_123",
            entity_type="Person",
            timestamp=datetime.utcnow(),
            schema_version=1,
        )
        emitter.emit(event)

        assert emitter.get_statistics()["emit_count"] == 1

        emitter.reset_statistics()

        assert emitter.get_statistics()["emit_count"] == 0

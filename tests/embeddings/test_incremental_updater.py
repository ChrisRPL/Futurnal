"""Tests for IncrementalEmbeddingUpdater.

Tests batch processing of PKG events including:
- Event batching by size threshold
- Timeout-based flushing
- Batch create/update/delete operations
- Statistics tracking
- Thread safety

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from futurnal.pkg.sync.events import PKGEvent, SyncEventType
from futurnal.embeddings.incremental_updater import (
    IncrementalEmbeddingUpdater,
    UpdaterConfig,
    BatchStatistics,
)
from futurnal.embeddings.models import EmbeddingResult, EmbeddingEntityType


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_sync_handler():
    """Create a mock PKGSyncHandler."""
    handler = MagicMock()
    handler.handle_event = MagicMock(return_value=True)

    # Mock internal methods used by batch operations
    handler._format_entity_content = MagicMock(return_value="Test content")
    handler._extract_temporal_context = MagicMock(return_value=None)
    handler._requires_reembedding = MagicMock(return_value=True)

    # Mock embedding service
    def create_mock_result():
        return EmbeddingResult(
            embedding=np.random.rand(768).tolist(),
            entity_type=EmbeddingEntityType.STATIC_ENTITY,
            model_version="test-model-v1",
            embedding_dimension=768,
            generation_time_ms=10.0,
            metadata={"model_id": "test-model", "duration_ms": 10.0},
        )

    handler._embedding_service = MagicMock()
    handler._embedding_service.embed_batch = MagicMock(
        return_value=[create_mock_result() for _ in range(10)]
    )

    # Mock store
    handler._store = MagicMock()
    handler._store.store_embedding = MagicMock(return_value="emb_123")
    handler._store.delete_embedding_by_entity_id = MagicMock(return_value=1)
    handler._store.mark_for_reembedding = MagicMock(return_value=1)

    return handler


@pytest.fixture
def small_batch_config():
    """Config with small batch size for testing."""
    return UpdaterConfig(
        batch_size=3,
        batch_timeout_seconds=1.0,
        max_pending_events=10,
        enable_background_flush=False,
    )


@pytest.fixture
def updater(mock_sync_handler, small_batch_config):
    """Create an IncrementalEmbeddingUpdater with small batches."""
    return IncrementalEmbeddingUpdater(
        sync_handler=mock_sync_handler,
        config=small_batch_config,
    )


def create_test_event(
    event_id: str,
    event_type: SyncEventType = SyncEventType.ENTITY_CREATED,
    entity_type: str = "Person",
) -> PKGEvent:
    """Helper to create test events."""
    return PKGEvent(
        event_id=event_id,
        event_type=event_type,
        entity_id=f"entity_{event_id}",
        entity_type=entity_type,
        timestamp=datetime.utcnow(),
        new_data={"name": f"Test {event_id}"},
        schema_version=1,
    )


# -----------------------------------------------------------------------------
# Batch Size Tests
# -----------------------------------------------------------------------------


class TestBatchSizeTriggered:
    """Test automatic flushing when batch size reached."""

    def test_flush_on_batch_size_reached(self, updater, mock_sync_handler):
        """Events are processed when batch size is reached."""
        # Add events up to batch size
        for i in range(3):
            updater.add_event(create_test_event(f"evt_{i}"))

        # Should have triggered flush (3 events, batch_size=3)
        assert updater.pending_count == 0
        assert updater.statistics.total_events == 3

    def test_no_flush_below_batch_size(self, updater):
        """Events are not processed below batch size."""
        # Add fewer than batch size
        for i in range(2):
            updater.add_event(create_test_event(f"evt_{i}"))

        # Should still be pending
        assert updater.pending_count == 2
        assert updater.statistics.total_events == 0

    def test_multiple_batch_flushes(self, updater):
        """Multiple batches are processed correctly."""
        # Add 7 events (2 full batches + 1 pending)
        for i in range(7):
            updater.add_event(create_test_event(f"evt_{i}"))

        # 2 batches processed (6 events), 1 pending
        assert updater.pending_count == 1
        assert updater.statistics.total_batches == 2
        assert updater.statistics.total_events == 6


# -----------------------------------------------------------------------------
# Manual Flush Tests
# -----------------------------------------------------------------------------


class TestManualFlush:
    """Test manual flush() method."""

    def test_manual_flush_processes_pending(self, updater):
        """Manual flush processes all pending events."""
        # Add fewer than batch size
        for i in range(2):
            updater.add_event(create_test_event(f"evt_{i}"))

        assert updater.pending_count == 2

        # Manual flush
        processed = updater.flush()

        assert processed == 2
        assert updater.pending_count == 0
        assert updater.statistics.total_events == 2

    def test_manual_flush_empty_queue(self, updater):
        """Manual flush with no pending events returns 0."""
        processed = updater.flush()

        assert processed == 0
        assert updater.statistics.total_batches == 0


# -----------------------------------------------------------------------------
# Batch Processing Tests
# -----------------------------------------------------------------------------


class TestBatchProcessing:
    """Test batch processing by event type."""

    def test_batch_create_events(self, updater, mock_sync_handler):
        """ENTITY_CREATED events use batch embedding."""
        events = [
            create_test_event(f"evt_{i}", SyncEventType.ENTITY_CREATED)
            for i in range(3)
        ]
        for event in events:
            updater.add_event(event)

        # Verify batch embed was called
        assert mock_sync_handler._embedding_service.embed_batch.called

    def test_batch_delete_events(self, updater, mock_sync_handler):
        """ENTITY_DELETED events are batch deleted."""
        events = [
            PKGEvent(
                event_id=f"evt_{i}",
                event_type=SyncEventType.ENTITY_DELETED,
                entity_id=f"entity_{i}",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                schema_version=1,
            )
            for i in range(3)
        ]
        for event in events:
            updater.add_event(event)

        # Verify delete was called for each
        assert mock_sync_handler._store.delete_embedding_by_entity_id.call_count == 3

    def test_batch_update_filters_significant(self, updater, mock_sync_handler):
        """ENTITY_UPDATED events filter to significant changes."""
        # First update: significant (name changed)
        mock_sync_handler._requires_reembedding.return_value = True

        events = [
            PKGEvent(
                event_id=f"evt_{i}",
                event_type=SyncEventType.ENTITY_UPDATED,
                entity_id=f"entity_{i}",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                old_data={"name": f"Old {i}"},
                new_data={"name": f"New {i}"},
                schema_version=1,
            )
            for i in range(3)
        ]
        for event in events:
            updater.add_event(event)

        # Verify re-embedding was triggered
        assert mock_sync_handler._store.mark_for_reembedding.called

    def test_mixed_event_types_grouped(self, updater, mock_sync_handler):
        """Mixed event types are grouped and processed separately."""
        events = [
            create_test_event("evt_0", SyncEventType.ENTITY_CREATED),
            create_test_event("evt_1", SyncEventType.ENTITY_CREATED),
            PKGEvent(
                event_id="evt_2",
                event_type=SyncEventType.ENTITY_DELETED,
                entity_id="entity_2",
                entity_type="Person",
                timestamp=datetime.utcnow(),
                schema_version=1,
            ),
        ]
        for event in events:
            updater.add_event(event)

        # Both create and delete should be processed
        assert updater.statistics.total_events == 3


# -----------------------------------------------------------------------------
# Backpressure Tests
# -----------------------------------------------------------------------------


class TestBackpressure:
    """Test backpressure handling."""

    def test_max_pending_events_raises(self, mock_sync_handler):
        """Exceeding max_pending_events raises RuntimeError."""
        config = UpdaterConfig(
            batch_size=100,  # High to prevent auto-flush
            max_pending_events=5,
            enable_background_flush=False,
        )
        updater = IncrementalEmbeddingUpdater(mock_sync_handler, config)

        # Add up to max
        for i in range(5):
            updater.add_event(create_test_event(f"evt_{i}"))

        # Next should raise
        with pytest.raises(RuntimeError, match="Event queue full"):
            updater.add_event(create_test_event("evt_overflow"))


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_updated_after_batch(self, updater):
        """Statistics are updated after batch processing."""
        for i in range(3):
            updater.add_event(create_test_event(f"evt_{i}"))

        stats = updater.statistics
        assert stats.total_batches == 1
        assert stats.total_events == 3
        assert stats.total_succeeded > 0

    def test_get_statistics_dict(self, updater):
        """get_statistics returns proper dict."""
        for i in range(3):
            updater.add_event(create_test_event(f"evt_{i}"))

        stats = updater.get_statistics()

        assert "total_batches" in stats
        assert "total_events" in stats
        assert "avg_batch_size" in stats
        assert "throughput_per_second" in stats
        assert "pending_count" in stats

    def test_reset_statistics(self, updater):
        """Statistics can be reset."""
        for i in range(3):
            updater.add_event(create_test_event(f"evt_{i}"))

        assert updater.statistics.total_events == 3

        updater.reset_statistics()

        assert updater.statistics.total_events == 0
        assert updater.statistics.total_batches == 0

    def test_success_rate_calculation(self, updater, mock_sync_handler):
        """Success rate is calculated correctly."""
        # All succeed
        for i in range(3):
            updater.add_event(create_test_event(f"evt_{i}"))

        assert updater.statistics.success_rate > 0

    def test_avg_batch_size_calculation(self, updater):
        """Average batch size is calculated correctly."""
        # 6 events in 2 batches
        for i in range(6):
            updater.add_event(create_test_event(f"evt_{i}"))

        assert updater.statistics.avg_batch_size == 3.0


# -----------------------------------------------------------------------------
# Background Flush Tests
# -----------------------------------------------------------------------------


class TestBackgroundFlush:
    """Test background timeout-based flushing."""

    def test_background_flush_thread_starts(self, mock_sync_handler):
        """Background flush thread starts when enabled."""
        config = UpdaterConfig(
            batch_size=100,
            batch_timeout_seconds=0.5,
            enable_background_flush=True,
        )
        updater = IncrementalEmbeddingUpdater(mock_sync_handler, config)

        try:
            assert updater._running is True
            assert updater._flush_thread is not None
            assert updater._flush_thread.is_alive()
        finally:
            updater.stop()

    def test_background_flush_on_timeout(self, mock_sync_handler):
        """Background thread flushes on timeout."""
        config = UpdaterConfig(
            batch_size=100,  # High to prevent size-triggered flush
            batch_timeout_seconds=0.5,
            enable_background_flush=True,
        )
        updater = IncrementalEmbeddingUpdater(mock_sync_handler, config)

        try:
            # Add events (below batch size)
            updater.add_event(create_test_event("evt_0"))
            updater.add_event(create_test_event("evt_1"))

            assert updater.pending_count == 2

            # Wait for timeout
            time.sleep(1.0)

            # Should have been flushed by background thread
            assert updater.pending_count == 0
            assert updater.statistics.total_events == 2
        finally:
            updater.stop()

    def test_stop_flushes_remaining(self, mock_sync_handler):
        """stop() flushes remaining events."""
        config = UpdaterConfig(
            batch_size=100,
            batch_timeout_seconds=10.0,  # Long timeout
            enable_background_flush=True,
        )
        updater = IncrementalEmbeddingUpdater(mock_sync_handler, config)

        # Add events
        updater.add_event(create_test_event("evt_0"))
        updater.add_event(create_test_event("evt_1"))

        assert updater.pending_count == 2

        # Stop should flush
        updater.stop()

        assert updater.pending_count == 0
        assert updater.statistics.total_events == 2


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of the updater."""

    def test_concurrent_add_events(self, mock_sync_handler):
        """Multiple threads can add events safely."""
        config = UpdaterConfig(
            batch_size=50,
            max_pending_events=1000,
            enable_background_flush=False,
        )
        updater = IncrementalEmbeddingUpdater(mock_sync_handler, config)

        num_threads = 5
        events_per_thread = 10

        def add_events(thread_id):
            for i in range(events_per_thread):
                updater.add_event(create_test_event(f"t{thread_id}_evt_{i}"))

        threads = [
            threading.Thread(target=add_events, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Flush remaining
        updater.flush()

        # All events should be processed
        total_expected = num_threads * events_per_thread
        assert updater.statistics.total_events == total_expected


# -----------------------------------------------------------------------------
# BatchStatistics Tests
# -----------------------------------------------------------------------------


class TestBatchStatistics:
    """Test BatchStatistics dataclass."""

    def test_avg_batch_size_empty(self):
        """avg_batch_size returns 0 when no batches."""
        stats = BatchStatistics()
        assert stats.avg_batch_size == 0.0

    def test_avg_batch_duration_empty(self):
        """avg_batch_duration_ms returns 0 when no batches."""
        stats = BatchStatistics()
        assert stats.avg_batch_duration_ms == 0.0

    def test_success_rate_empty(self):
        """success_rate returns 1.0 when no events."""
        stats = BatchStatistics()
        assert stats.success_rate == 1.0

    def test_throughput_empty(self):
        """throughput_per_second returns 0 when no duration."""
        stats = BatchStatistics()
        assert stats.throughput_per_second == 0.0

    def test_calculations_with_data(self):
        """Statistics calculate correctly with data."""
        stats = BatchStatistics(
            total_batches=2,
            total_events=10,
            total_succeeded=9,
            total_failed=1,
            total_duration_ms=100.0,
        )

        assert stats.avg_batch_size == 5.0
        assert stats.avg_batch_duration_ms == 50.0
        assert stats.success_rate == 0.9
        assert stats.throughput_per_second == 100.0

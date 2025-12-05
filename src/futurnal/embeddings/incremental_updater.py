"""Incremental Embedding Updater for Efficient Batch Processing.

Batches PKG mutation events for efficient embedding updates, optimizing
throughput while maintaining sync latency within acceptable bounds.

Architecture:
    PKGEventEmitter
          |
    IncrementalEmbeddingUpdater.add_event()
          |
    [Event Queue - batches by size/timeout]
          |
    flush_batch() --> PKGSyncHandler (batch operations)

The updater collects events and flushes them when:
1. Batch size threshold reached (default 50)
2. Timeout threshold reached (default 5 seconds)
3. Manual flush() called

Success Metrics:
- Batch processing >100 updates/second
- Sync latency <1s for 95% of mutations
- Efficient use of embed_batch() API

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Example Usage:
    >>> from futurnal.embeddings.incremental_updater import IncrementalEmbeddingUpdater
    >>> from futurnal.embeddings.sync_handler import PKGSyncHandler
    >>>
    >>> # Create updater with sync handler
    >>> updater = IncrementalEmbeddingUpdater(
    ...     sync_handler=sync_handler,
    ...     config=UpdaterConfig(batch_size=50, batch_timeout_seconds=5.0),
    ... )
    >>>
    >>> # Add events (batched automatically)
    >>> for event in pkg_events:
    ...     updater.add_event(event)
    >>>
    >>> # Ensure all events processed
    >>> updater.flush()
    >>> updater.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from futurnal.embeddings.models import TemporalEmbeddingContext
from futurnal.embeddings.request import EmbeddingRequest
from futurnal.pkg.sync.events import PKGEvent, SyncEventType

if TYPE_CHECKING:
    from futurnal.embeddings.sync_handler import PKGSyncHandler

logger = logging.getLogger(__name__)


@dataclass
class UpdaterConfig:
    """Configuration for IncrementalEmbeddingUpdater.

    Attributes:
        batch_size: Number of events to collect before flushing (default 50)
        batch_timeout_seconds: Max time to wait before flushing (default 5.0)
        max_pending_events: Maximum events in queue before blocking (default 1000)
        enable_background_flush: Enable background thread for timeout flushes (default True)
    """
    batch_size: int = 50
    batch_timeout_seconds: float = 5.0
    max_pending_events: int = 1000
    enable_background_flush: bool = True


@dataclass
class BatchStatistics:
    """Statistics for batch processing.

    Attributes:
        total_batches: Number of batches processed
        total_events: Total events processed
        total_succeeded: Events successfully processed
        total_failed: Events that failed processing
        avg_batch_size: Average batch size
        avg_batch_duration_ms: Average batch processing time
    """
    total_batches: int = 0
    total_events: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_duration_ms: float = 0.0

    @property
    def avg_batch_size(self) -> float:
        """Average events per batch."""
        if self.total_batches == 0:
            return 0.0
        return self.total_events / self.total_batches

    @property
    def avg_batch_duration_ms(self) -> float:
        """Average batch processing duration."""
        if self.total_batches == 0:
            return 0.0
        return self.total_duration_ms / self.total_batches

    @property
    def success_rate(self) -> float:
        """Event success rate (0.0 to 1.0)."""
        if self.total_events == 0:
            return 1.0
        return self.total_succeeded / self.total_events

    @property
    def throughput_per_second(self) -> float:
        """Events processed per second."""
        if self.total_duration_ms == 0:
            return 0.0
        return self.total_events / (self.total_duration_ms / 1000.0)


class IncrementalEmbeddingUpdater:
    """Batches PKG events for efficient embedding updates.

    Collects PKGEvents and flushes them in batches to optimize throughput.
    Uses a background thread to handle timeout-based flushes.

    Thread Safety:
        This class is thread-safe. Events can be added from multiple threads.

    Attributes:
        sync_handler: PKGSyncHandler for processing events
        config: UpdaterConfig with batching parameters
        statistics: BatchStatistics with processing metrics

    Example:
        >>> updater = IncrementalEmbeddingUpdater(sync_handler, config)
        >>> updater.add_event(event)  # Batched
        >>> updater.flush()  # Process remaining
        >>> updater.stop()  # Stop background thread
    """

    def __init__(
        self,
        sync_handler: "PKGSyncHandler",
        config: Optional[UpdaterConfig] = None,
    ) -> None:
        """Initialize the incremental updater.

        Args:
            sync_handler: PKGSyncHandler for processing events
            config: Optional UpdaterConfig (uses defaults if not provided)
        """
        self._sync_handler = sync_handler
        self._config = config or UpdaterConfig()
        self._statistics = BatchStatistics()

        # Event queue with thread safety
        self._pending_events: List[PKGEvent] = []
        self._lock = threading.Lock()
        self._last_flush = datetime.utcnow()

        # Background flush thread
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        if self._config.enable_background_flush:
            self._start_background_flush()

        logger.info(
            f"Initialized IncrementalEmbeddingUpdater "
            f"(batch_size={self._config.batch_size}, "
            f"timeout={self._config.batch_timeout_seconds}s)"
        )

    def _start_background_flush(self) -> None:
        """Start background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._background_flush_loop,
            daemon=True,
            name="EmbeddingUpdaterFlush",
        )
        self._flush_thread.start()
        logger.debug("Started background flush thread")

    def _background_flush_loop(self) -> None:
        """Background loop that flushes on timeout."""
        while self._running:
            time.sleep(0.5)  # Check every 500ms

            with self._lock:
                if not self._pending_events:
                    continue

                elapsed = (datetime.utcnow() - self._last_flush).total_seconds()
                if elapsed >= self._config.batch_timeout_seconds:
                    self._flush_batch_internal()

    def add_event(self, event: PKGEvent) -> None:
        """Add an event to the pending batch.

        Events are batched and processed when:
        - Batch size threshold reached
        - Timeout threshold reached (via background thread)

        Args:
            event: PKGEvent to add

        Raises:
            RuntimeError: If max_pending_events reached (backpressure)
        """
        with self._lock:
            # Check backpressure
            if len(self._pending_events) >= self._config.max_pending_events:
                raise RuntimeError(
                    f"Event queue full ({self._config.max_pending_events} events). "
                    "Increase max_pending_events or flush more frequently."
                )

            self._pending_events.append(event)

            # Check if batch size reached
            if len(self._pending_events) >= self._config.batch_size:
                self._flush_batch_internal()

    def flush(self) -> int:
        """Flush all pending events immediately.

        Returns:
            Number of events processed
        """
        with self._lock:
            return self._flush_batch_internal()

    def _flush_batch_internal(self) -> int:
        """Internal flush - assumes lock is held.

        Returns:
            Number of events processed
        """
        if not self._pending_events:
            return 0

        events = self._pending_events.copy()
        self._pending_events = []
        self._last_flush = datetime.utcnow()

        # Release lock during processing
        # Note: We copy events to allow new events during processing
        self._lock.release()
        try:
            return self._process_batch(events)
        finally:
            self._lock.acquire()

    def _process_batch(self, events: List[PKGEvent]) -> int:
        """Process a batch of events.

        Groups events by type and processes them efficiently.

        Args:
            events: List of PKGEvents to process

        Returns:
            Number of events successfully processed
        """
        if not events:
            return 0

        start_time = time.perf_counter()
        success_count = 0

        logger.debug(f"Processing batch of {len(events)} events")

        # Group events by type for efficient processing
        grouped: Dict[SyncEventType, List[PKGEvent]] = {}
        for event in events:
            event_type = event.event_type
            if event_type not in grouped:
                grouped[event_type] = []
            grouped[event_type].append(event)

        # Process each group
        for event_type, group in grouped.items():
            try:
                if event_type == SyncEventType.ENTITY_CREATED:
                    success_count += self._batch_create_embeddings(group)
                elif event_type == SyncEventType.ENTITY_UPDATED:
                    success_count += self._batch_update_embeddings(group)
                elif event_type == SyncEventType.ENTITY_DELETED:
                    success_count += self._batch_delete_embeddings(group)
                else:
                    # Handle individually for other types
                    for event in group:
                        if self._sync_handler.handle_event(event):
                            success_count += 1
            except Exception as e:
                logger.error(f"Error processing batch of {event_type}: {e}")

        # Update statistics
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._statistics.total_batches += 1
        self._statistics.total_events += len(events)
        self._statistics.total_succeeded += success_count
        self._statistics.total_failed += len(events) - success_count
        self._statistics.total_duration_ms += duration_ms

        logger.info(
            f"Batch processed: {success_count}/{len(events)} events "
            f"in {duration_ms:.2f}ms "
            f"({len(events) / (duration_ms / 1000):.1f} events/sec)"
        )

        return success_count

    def _batch_create_embeddings(self, events: List[PKGEvent]) -> int:
        """Batch create embeddings using embed_batch().

        Args:
            events: List of ENTITY_CREATED events

        Returns:
            Number of successful embeddings
        """
        if not events:
            return 0

        # Prepare embedding requests
        requests = []
        valid_events = []

        for event in events:
            if event.new_data is None:
                logger.warning(f"Skipping event {event.event_id}: missing new_data")
                continue

            content = self._sync_handler._format_entity_content(
                event.entity_type,
                event.new_data,
            )

            temporal_context = None
            if event.entity_type == "Event":
                temporal_context = self._sync_handler._extract_temporal_context(
                    event.new_data
                )
                if temporal_context is None:
                    logger.warning(
                        f"Skipping Event {event.entity_id}: missing temporal context"
                    )
                    continue

            requests.append(EmbeddingRequest(
                entity_type=event.entity_type,
                content=content,
                entity_id=event.entity_id,
                entity_name=event.new_data.get("name"),
                temporal_context=temporal_context,
                metadata={"event_id": event.event_id},
            ))
            valid_events.append(event)

        if not requests:
            return 0

        # Batch embed
        try:
            results = self._sync_handler._embedding_service.embed_batch(
                requests,
                fail_fast=False,  # Continue on individual failures
            )

            # Store all embeddings
            success_count = 0
            for event, result in zip(valid_events, results):
                try:
                    temporal_context = None
                    if event.entity_type == "Event" and event.new_data:
                        temporal_context = self._sync_handler._extract_temporal_context(
                            event.new_data
                        )

                    self._sync_handler._store.store_embedding(
                        entity_id=event.entity_id,
                        entity_type=event.entity_type,
                        embedding=list(result.embedding),
                        model_id=result.metadata.get("model_id", "unknown"),
                        extraction_confidence=event.extraction_confidence or 1.0,
                        source_document_id=event.source_document_id or "unknown",
                        temporal_context=temporal_context,
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to store embedding for {event.entity_id}: {e}")

            return success_count

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return 0

    def _batch_update_embeddings(self, events: List[PKGEvent]) -> int:
        """Batch update embeddings.

        Filters to events requiring re-embedding, then batch creates.

        Args:
            events: List of ENTITY_UPDATED events

        Returns:
            Number of successful updates
        """
        # Filter to events requiring re-embedding
        events_to_update = []
        for event in events:
            old_data = event.old_data or {}
            new_data = event.new_data or {}

            if self._sync_handler._requires_reembedding(old_data, new_data):
                events_to_update.append(event)

        if not events_to_update:
            return len(events)  # All skipped successfully

        # Mark old embeddings
        entity_ids = [e.entity_id for e in events_to_update]
        self._sync_handler._store.mark_for_reembedding(
            entity_ids=entity_ids,
            reason="batch_update",
        )

        # Create new embeddings (reuse batch create)
        return self._batch_create_embeddings(events_to_update)

    def _batch_delete_embeddings(self, events: List[PKGEvent]) -> int:
        """Batch delete embeddings.

        Args:
            events: List of ENTITY_DELETED events

        Returns:
            Number of successful deletions
        """
        success_count = 0

        for event in events:
            try:
                deleted = self._sync_handler._store.delete_embedding_by_entity_id(
                    event.entity_id
                )
                if deleted > 0:
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to delete embedding for {event.entity_id}: {e}")

        return success_count

    def stop(self) -> None:
        """Stop the background flush thread and flush remaining events."""
        logger.info("Stopping IncrementalEmbeddingUpdater")

        # Stop background thread
        self._running = False
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)

        # Final flush
        self.flush()

        logger.info("IncrementalEmbeddingUpdater stopped")

    @property
    def pending_count(self) -> int:
        """Number of pending events."""
        with self._lock:
            return len(self._pending_events)

    @property
    def statistics(self) -> BatchStatistics:
        """Get processing statistics."""
        return self._statistics

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics as dictionary.

        Returns:
            Dict with batch processing statistics
        """
        return {
            "total_batches": self._statistics.total_batches,
            "total_events": self._statistics.total_events,
            "total_succeeded": self._statistics.total_succeeded,
            "total_failed": self._statistics.total_failed,
            "avg_batch_size": self._statistics.avg_batch_size,
            "avg_batch_duration_ms": self._statistics.avg_batch_duration_ms,
            "success_rate": self._statistics.success_rate,
            "throughput_per_second": self._statistics.throughput_per_second,
            "pending_count": self.pending_count,
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._statistics = BatchStatistics()
        logger.debug("IncrementalEmbeddingUpdater statistics reset")

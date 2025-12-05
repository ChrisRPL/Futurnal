"""PKG Event Emitter for Embedding Synchronization.

Provides event emission infrastructure for PKG mutations to enable
real-time synchronization with the Vector Embedding Store.

The emitter acts as a bridge between PKG repositories (EntityRepository,
RelationshipRepository) and the embedding sync pipeline (PKGSyncHandler).

Architecture:
    EntityRepository
          |
    EmittingEntityRepository (wrapper)
          |
    PKGEventEmitter.emit(PKGEvent)
          |
    event_handler (PKGSyncHandler or IncrementalEmbeddingUpdater)
          |
    SyncEventCapture (optional monitoring)

Option B Compliance:
- Events carry full mutation context for proper embedding generation
- Schema version tracked for autonomous evolution
- Temporal metadata preserved (timestamp field)

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Example Usage:
    >>> from futurnal.pkg.sync import PKGEventEmitter, PKGEvent, SyncEventCapture
    >>>
    >>> # Create capture for monitoring
    >>> capture = SyncEventCapture()
    >>>
    >>> # Create emitter with sync handler
    >>> emitter = PKGEventEmitter(
    ...     event_handler=sync_handler.handle_event,
    ...     sync_event_capture=capture,
    ... )
    >>>
    >>> # Emit event after entity creation
    >>> event = PKGEvent(
    ...     event_id="evt_123",
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="person_456",
    ...     entity_type="Person",
    ...     new_data={"name": "John Doe"},
    ... )
    >>> emitter.emit(event)
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, TYPE_CHECKING

from futurnal.pkg.sync.events import (
    PKGEvent,
    SyncEventCapture,
    SyncStatus,
    SourceOperation,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PKGEventEmitter:
    """Emits PKG mutation events for embedding synchronization.

    Integrates with EntityRepository and RelationshipRepository via wrapper
    classes to emit events after successful mutations.

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded usage,
        wrap emit() calls with appropriate locking in the wrapper.

    Attributes:
        event_handler: Callable that processes PKGEvents (e.g., PKGSyncHandler.handle_event)
        sync_event_capture: Optional capture utility for monitoring/debugging
        emit_count: Total number of events emitted
        error_count: Total number of emission errors

    Example:
        >>> def my_handler(event: PKGEvent) -> None:
        ...     print(f"Received: {event.event_type} for {event.entity_id}")
        >>>
        >>> emitter = PKGEventEmitter(event_handler=my_handler)
        >>> emitter.emit(PKGEvent(...))
    """

    def __init__(
        self,
        event_handler: Callable[[PKGEvent], None],
        sync_event_capture: Optional[SyncEventCapture] = None,
    ) -> None:
        """Initialize PKG event emitter.

        Args:
            event_handler: Callable that processes PKGEvents. This is typically
                          PKGSyncHandler.handle_event or IncrementalEmbeddingUpdater.add_event.
            sync_event_capture: Optional SyncEventCapture for monitoring/testing.
                               When provided, events are also captured for analysis.
        """
        self._event_handler = event_handler
        self._sync_capture = sync_event_capture
        self._emit_count = 0
        self._error_count = 0

        logger.info("Initialized PKGEventEmitter")

    def emit(self, event: PKGEvent) -> bool:
        """Emit a PKG mutation event.

        Calls the registered event handler with the event, and optionally
        captures the event for monitoring.

        Args:
            event: The PKGEvent to emit

        Returns:
            True if event was processed successfully, False on error

        Note:
            Errors in the event handler are caught and logged but not re-raised.
            This prevents sync failures from blocking PKG mutations.
            Failed events can be identified via sync_event_capture or error_count.
        """
        self._emit_count += 1
        start_time = time.perf_counter()

        logger.debug(
            f"Emitting PKG event: {event.event_type} for "
            f"{event.entity_type}:{event.entity_id}"
        )

        try:
            # Call the registered handler
            self._event_handler(event)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Capture for monitoring (as completed)
            if self._sync_capture:
                self._sync_capture.capture(
                    event.to_sync_event(
                        sync_status=SyncStatus.COMPLETED,
                        source_operation=SourceOperation.PKG_WRITE,
                        duration_ms=duration_ms,
                    )
                )

            logger.debug(
                f"PKG event emitted successfully in {duration_ms:.2f}ms: "
                f"{event.event_type} for {event.entity_id}"
            )
            return True

        except Exception as e:
            self._error_count += 1
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                f"PKG event emission failed for {event.entity_id}: {e}",
                exc_info=True,
            )

            # Capture for monitoring (as failed)
            if self._sync_capture:
                self._sync_capture.capture(
                    event.to_sync_event(
                        sync_status=SyncStatus.FAILED,
                        source_operation=SourceOperation.PKG_WRITE,
                        duration_ms=duration_ms,
                        error_message=str(e),
                    )
                )

            return False

    def emit_batch(self, events: list[PKGEvent]) -> int:
        """Emit multiple PKG events.

        Args:
            events: List of PKGEvents to emit

        Returns:
            Number of events processed successfully
        """
        success_count = 0
        for event in events:
            if self.emit(event):
                success_count += 1

        logger.info(
            f"Batch emission complete: {success_count}/{len(events)} events successful"
        )
        return success_count

    @property
    def emit_count(self) -> int:
        """Total number of events emitted."""
        return self._emit_count

    @property
    def error_count(self) -> int:
        """Total number of emission errors."""
        return self._error_count

    @property
    def success_rate(self) -> float:
        """Success rate of event emissions (0.0 to 1.0)."""
        if self._emit_count == 0:
            return 1.0
        return (self._emit_count - self._error_count) / self._emit_count

    def get_statistics(self) -> dict:
        """Get emission statistics.

        Returns:
            Dict with emit_count, error_count, success_rate
        """
        return {
            "emit_count": self._emit_count,
            "error_count": self._error_count,
            "success_rate": self.success_rate,
        }

    def reset_statistics(self) -> None:
        """Reset emission statistics counters."""
        self._emit_count = 0
        self._error_count = 0
        logger.debug("PKGEventEmitter statistics reset")

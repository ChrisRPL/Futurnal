"""PKG Sync Module - PKG ↔ Vector Store Synchronization.

Provides event models and utilities for tracking synchronization
between PKG (Neo4j) and Vector Store (ChromaDB).

Module Structure:
- events.py: SyncEvent, PKGEvent models and SyncEventCapture utility
- emitter.py: PKGEventEmitter for event emission

Implementation follows production plans:
- docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
- docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Usage:
    >>> from futurnal.pkg.sync import SyncEvent, SyncEventCapture, SyncEventType
    >>>
    >>> # Create capture for testing/monitoring
    >>> capture = SyncEventCapture()
    >>>
    >>> # Create sync event
    >>> event = SyncEvent(
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="doc_123",
    ...     entity_type="Document",
    ...     source_operation="pkg_write",
    ... )
    >>> capture.capture(event)

    >>> # PKG mutation events for embedding sync
    >>> from futurnal.pkg.sync import PKGEvent, PKGEventEmitter
    >>> pkg_event = PKGEvent(
    ...     event_id="evt_123",
    ...     event_type=SyncEventType.ENTITY_CREATED,
    ...     entity_id="person_456",
    ...     entity_type="Person",
    ...     new_data={"name": "John Doe"},
    ... )
    >>> emitter = PKGEventEmitter(event_handler=sync_handler.handle_event)
    >>> emitter.emit(pkg_event)
"""

from futurnal.pkg.sync.events import (
    PKGEvent,
    SourceOperation,
    SyncEvent,
    SyncEventCapture,
    SyncEventType,
    SyncStatus,
)
from futurnal.pkg.sync.emitter import PKGEventEmitter

__all__ = [
    # Event models
    "SyncEvent",
    "SyncEventCapture",
    "SyncEventType",
    "SyncStatus",
    "SourceOperation",
    # PKG mutation events (Module 04)
    "PKGEvent",
    "PKGEventEmitter",
]

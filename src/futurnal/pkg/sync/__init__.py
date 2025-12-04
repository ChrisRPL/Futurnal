"""PKG Sync Module - PKG ↔ Vector Store Synchronization.

Provides event models and utilities for tracking synchronization
between PKG (Neo4j) and Vector Store (ChromaDB).

Module Structure:
- events.py: SyncEvent models and SyncEventCapture utility

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md

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
"""

from futurnal.pkg.sync.events import (
    SyncEvent,
    SyncEventCapture,
    SyncEventType,
    SyncStatus,
)

__all__ = [
    "SyncEvent",
    "SyncEventCapture",
    "SyncEventType",
    "SyncStatus",
]

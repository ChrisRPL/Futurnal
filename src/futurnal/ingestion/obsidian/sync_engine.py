"""Obsidian sync engine for incremental vault synchronization.

This module provides a specialized sync engine that coordinates incremental
synchronization of Obsidian vaults, building on the existing ObsidianVaultConnector
and orchestrator infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue
from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.redaction import RedactionPolicy, build_policy

from ..local.state import FileRecord, StateStore
from .connector import ObsidianVaultConnector, ObsidianVaultSource
from .path_tracker import ObsidianPathTracker, PathChange

logger = logging.getLogger(__name__)


class SyncPriority(Enum):
    """Priority levels for sync operations."""
    CRITICAL = 100   # Path changes, renames that affect graph integrity
    HIGH = 75        # New notes, note content changes
    NORMAL = 50      # Note metadata updates, frontmatter changes
    LOW = 25         # Asset files, attachments
    BACKGROUND = 10  # Cleanup, optimization tasks


class SyncEventType(Enum):
    """Types of sync events."""
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    BATCH_SYNC = "batch_sync"
    FULL_SCAN = "full_scan"


@dataclass
class SyncEvent:
    """Represents a sync event for processing."""
    event_type: SyncEventType
    file_path: Path
    vault_id: str
    priority: SyncPriority = SyncPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, any] = field(default_factory=dict)
    retry_count: int = 0

    def to_job_payload(self) -> Dict[str, any]:
        """Convert sync event to job queue payload."""
        return {
            "event_type": self.event_type.value,
            "file_path": str(self.file_path),
            "vault_id": self.vault_id,
            "sync_priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "retry_count": self.retry_count,
        }


@dataclass
class SyncBatch:
    """Represents a batch of related sync events."""
    batch_id: str
    events: List[SyncEvent]
    priority: SyncPriority
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_path_change_batch(self) -> bool:
        """Check if this batch contains path changes."""
        return any(e.event_type == SyncEventType.FILE_MOVED for e in self.events)

    @property
    def affected_files(self) -> Set[Path]:
        """Get all file paths affected by this batch."""
        return {event.file_path for event in self.events}


class ObsidianSyncEngine:
    """Coordinates incremental synchronization for Obsidian vaults.

    This sync engine provides intelligent batching, prioritization, and
    coordination for vault synchronization operations while building on
    the existing Futurnal architecture.
    """

    def __init__(
        self,
        *,
        vault_connector: ObsidianVaultConnector,
        job_queue: JobQueue,
        state_store: StateStore,
        audit_logger: Optional[AuditLogger] = None,
        batch_window_seconds: float = 2.0,
        max_batch_size: int = 50,
        max_retry_attempts: int = 3,
        retry_backoff_base: float = 2.0,
        performance_monitoring: bool = True,
    ) -> None:
        self._vault_connector = vault_connector
        self._job_queue = job_queue
        self._state_store = state_store
        self._audit_logger = audit_logger

        # Batching configuration
        self._batch_window_seconds = batch_window_seconds
        self._max_batch_size = max_batch_size

        # Retry configuration
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_base = retry_backoff_base

        # Performance monitoring
        self._performance_monitoring = performance_monitoring
        self._sync_metrics: Dict[str, any] = {}
        self._metrics_collector = None

        # Initialize metrics collector if performance monitoring is enabled
        if self._performance_monitoring:
            try:
                from .sync_metrics import create_metrics_collector
                self._metrics_collector = create_metrics_collector()
                logger.info("Sync metrics collector initialized")
            except ImportError:
                logger.warning("Sync metrics not available, continuing without detailed metrics")

        # Internal state
        self._pending_events: List[SyncEvent] = []
        self._active_batches: Dict[str, SyncBatch] = {}
        self._last_batch_time = time.time()
        self._sync_lock = asyncio.Lock()
        self._running = False

        # Event debouncing per file
        self._file_debounce: Dict[Path, float] = {}
        self._debounce_window = 1.0  # seconds

    async def start(self) -> None:
        """Start the sync engine."""
        if self._running:
            return

        self._running = True
        logger.info("ObsidianSyncEngine started")

        if self._audit_logger:
            await self._log_audit_event("sync_engine_started", "success")

        # Initialize metrics
        if self._metrics_collector:
            self._metrics_collector.set_gauge("sync_engine_running", 1.0)

    async def stop(self) -> None:
        """Stop the sync engine and process remaining events."""
        if not self._running:
            return

        self._running = False

        # Process any remaining pending events
        if self._pending_events:
            await self._process_pending_events()

        logger.info("ObsidianSyncEngine stopped")

        if self._audit_logger:
            await self._log_audit_event("sync_engine_stopped", "success")

        # Update metrics
        if self._metrics_collector:
            self._metrics_collector.set_gauge("sync_engine_running", 0.0)

    async def handle_file_event(
        self,
        event_type: SyncEventType,
        file_path: Path,
        vault_id: str,
        metadata: Optional[Dict[str, any]] = None
    ) -> None:
        """Handle a file system event for sync processing.

        Args:
            event_type: Type of file system event
            file_path: Path to the affected file
            vault_id: ID of the vault containing the file
            metadata: Additional event metadata
        """
        if not self._running:
            return

        # Check debouncing to avoid excessive events for the same file
        current_time = time.time()
        last_event_time = self._file_debounce.get(file_path, 0)

        if current_time - last_event_time < self._debounce_window:
            logger.debug(f"Debouncing file event for {file_path}")
            return

        self._file_debounce[file_path] = current_time

        # Determine priority based on file type and event type
        priority = self._determine_sync_priority(file_path, event_type)

        # Create sync event
        sync_event = SyncEvent(
            event_type=event_type,
            file_path=file_path,
            vault_id=vault_id,
            priority=priority,
            metadata=metadata or {}
        )

        async with self._sync_lock:
            self._pending_events.append(sync_event)

            # Record metrics
            if self._metrics_collector:
                self._metrics_collector.record_event(
                    event_type.value,
                    vault_id,
                    priority=priority.name,
                    file_path=str(file_path),
                    metadata_keys=list(metadata.keys()) if metadata else []
                )
                self._metrics_collector.increment_counter("events_received")
                self._metrics_collector.set_gauge("pending_events", len(self._pending_events))

            # Check if we should process a batch
            should_process = (
                len(self._pending_events) >= self._max_batch_size or
                current_time - self._last_batch_time >= self._batch_window_seconds or
                priority == SyncPriority.CRITICAL
            )

            if should_process:
                await self._process_pending_events()

    async def handle_path_changes(
        self,
        path_changes: List[PathChange],
        vault_id: str
    ) -> None:
        """Handle path changes (renames/moves) with special priority.

        Args:
            path_changes: List of detected path changes
            vault_id: ID of the vault containing the changes
        """
        if not path_changes:
            return

        logger.info(f"Processing {len(path_changes)} path changes for vault {vault_id}")

        # Record metrics
        if self._metrics_collector:
            self._metrics_collector.increment_counter("path_changes_processed", len(path_changes))
            self._metrics_collector.record_event(
                "path_changes_batch",
                vault_id,
                path_changes_count=len(path_changes)
            )

        # Create high-priority events for path changes
        sync_events = []
        for change in path_changes:
            event = SyncEvent(
                event_type=SyncEventType.FILE_MOVED,
                file_path=change.new_path,
                vault_id=vault_id,
                priority=SyncPriority.CRITICAL,
                metadata={
                    "path_change": change.to_dict(),
                    "old_path": str(change.old_path),
                    "change_type": change.change_type,
                }
            )
            sync_events.append(event)

        # Process path changes immediately as a high-priority batch
        batch = SyncBatch(
            batch_id=f"path_changes_{uuid.uuid4().hex[:8]}",
            events=sync_events,
            priority=SyncPriority.CRITICAL
        )

        await self._process_sync_batch(batch)

    async def trigger_full_sync(self, vault_source: ObsidianVaultSource) -> str:
        """Trigger a full vault synchronization.

        Args:
            vault_source: Vault source configuration

        Returns:
            Job ID for the full sync operation
        """
        job_id = str(uuid.uuid4())

        # Create full sync job with high priority
        job = IngestionJob(
            job_id=job_id,
            job_type=JobType.LOCAL_FILES,
            payload={
                "source_name": vault_source.name,
                "vault_id": vault_source.vault_id,
                "sync_type": "full_scan",
                "trigger": "manual_full_sync",
            },
            priority=JobPriority.HIGH,
        )

        self._job_queue.enqueue(job)

        logger.info(f"Triggered full sync for vault {vault_source.vault_id} (job: {job_id})")

        if self._audit_logger:
            await self._log_audit_event(
                "full_sync_triggered",
                "success",
                metadata={"vault_id": vault_source.vault_id, "job_id": job_id}
            )

        return job_id

    async def get_sync_status(self, vault_id: str) -> Dict[str, any]:
        """Get current sync status for a vault.

        Args:
            vault_id: ID of the vault

        Returns:
            Dictionary containing sync status information
        """
        pending_count = len([e for e in self._pending_events if e.vault_id == vault_id])
        active_batches = len([b for b in self._active_batches.values()
                             if any(e.vault_id == vault_id for e in b.events)])

        status = {
            "vault_id": vault_id,
            "engine_running": self._running,
            "pending_events": pending_count,
            "active_batches": active_batches,
            "last_batch_time": self._last_batch_time,
            "legacy_metrics": self._sync_metrics.get(vault_id, {}),
        }

        # Add comprehensive metrics if available
        if self._metrics_collector:
            try:
                # Get recent metrics summary
                summary = self._metrics_collector.generate_summary(vault_id, hours=1)
                status["metrics_summary"] = summary.to_dict()

                # Add real-time metrics
                status["real_time_metrics"] = {
                    "events_received": self._metrics_collector.get_counter("events_received"),
                    "batches_processed": self._metrics_collector.get_counter("batches_processed"),
                    "batch_processing_errors": self._metrics_collector.get_counter("batch_processing_errors"),
                    "pending_events_gauge": self._metrics_collector.get_gauge("pending_events"),
                    "sync_engine_running": self._metrics_collector.get_gauge("sync_engine_running"),
                }

                # Add timer statistics
                timer_stats = {
                    "batch_processing_time": self._metrics_collector.get_timer_stats("batch_processing_time"),
                }
                status["timer_stats"] = timer_stats

                # Add histogram statistics
                histogram_stats = {
                    "batch_size": self._metrics_collector.get_histogram_stats("batch_size"),
                }
                status["histogram_stats"] = histogram_stats

            except Exception as e:
                logger.debug(f"Failed to get comprehensive metrics: {e}")
                status["metrics_error"] = str(e)

        return status

    async def export_metrics(self, export_path: Path) -> bool:
        """Export comprehensive metrics to a file.

        Args:
            export_path: Path to export metrics file

        Returns:
            True if export was successful
        """
        if not self._metrics_collector:
            logger.warning("Metrics collector not available for export")
            return False

        try:
            self._metrics_collector.export_metrics(export_path)
            logger.info(f"Sync metrics exported to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    async def cleanup_old_metrics(self) -> None:
        """Clean up old metrics data to manage memory usage."""
        if self._metrics_collector:
            self._metrics_collector.cleanup_old_metrics()
            logger.debug("Cleaned up old sync metrics")

    def get_metrics_collector(self):
        """Get the metrics collector for advanced usage."""
        return self._metrics_collector

    def _determine_sync_priority(self, file_path: Path, event_type: SyncEventType) -> SyncPriority:
        """Determine sync priority based on file characteristics and event type."""
        file_suffix = file_path.suffix.lower()

        # Critical events always get highest priority
        if event_type == SyncEventType.FILE_MOVED:
            return SyncPriority.CRITICAL

        # Markdown files (notes) get high priority
        if file_suffix in {'.md', '.markdown'}:
            if event_type == SyncEventType.FILE_CREATED:
                return SyncPriority.HIGH
            elif event_type == SyncEventType.FILE_MODIFIED:
                return SyncPriority.HIGH
            else:
                return SyncPriority.NORMAL

        # Asset files get lower priority
        if file_suffix in {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.svg', '.webp'}:
            return SyncPriority.LOW

        # Other files get normal priority
        return SyncPriority.NORMAL

    async def _process_pending_events(self) -> None:
        """Process all pending events by creating and executing batches."""
        if not self._pending_events:
            return

        # Group events by priority and vault
        batches = self._create_sync_batches(self._pending_events)
        self._pending_events.clear()
        self._last_batch_time = time.time()

        # Process batches in priority order
        for batch in sorted(batches, key=lambda b: b.priority.value, reverse=True):
            await self._process_sync_batch(batch)

    def _create_sync_batches(self, events: List[SyncEvent]) -> List[SyncBatch]:
        """Create batches from pending events, grouping by priority and vault."""
        vault_priority_groups: Dict[Tuple[str, SyncPriority], List[SyncEvent]] = {}

        # Group events by vault and priority
        for event in events:
            key = (event.vault_id, event.priority)
            if key not in vault_priority_groups:
                vault_priority_groups[key] = []
            vault_priority_groups[key].append(event)

        # Create batches from groups
        batches = []
        for (vault_id, priority), group_events in vault_priority_groups.items():
            # Split large groups into multiple batches
            for i in range(0, len(group_events), self._max_batch_size):
                batch_events = group_events[i:i + self._max_batch_size]
                batch = SyncBatch(
                    batch_id=f"{vault_id}_{priority.name.lower()}_{uuid.uuid4().hex[:8]}",
                    events=batch_events,
                    priority=priority
                )
                batches.append(batch)

        return batches

    async def _process_sync_batch(self, batch: SyncBatch) -> None:
        """Process a sync batch by creating appropriate jobs."""
        batch_start_time = time.time()

        try:
            self._active_batches[batch.batch_id] = batch

            logger.info(
                f"Processing sync batch {batch.batch_id} with {len(batch.events)} events "
                f"(priority: {batch.priority.name})"
            )

            # Start timing for detailed metrics
            if self._metrics_collector:
                self._metrics_collector.start_timer(f"batch_processing_{batch.batch_id}")

            # Convert batch to jobs based on event types
            jobs = self._create_jobs_from_batch(batch)

            # Enqueue jobs with appropriate priority
            job_priority = self._map_sync_to_job_priority(batch.priority)
            for job in jobs:
                job.priority = job_priority
                self._job_queue.enqueue(job)

            processing_time = time.time() - batch_start_time

            # Record comprehensive metrics
            if self._metrics_collector:
                vault_ids = {event.vault_id for event in batch.events}
                for vault_id in vault_ids:
                    self._metrics_collector.record_batch(
                        batch.batch_id,
                        vault_id,
                        len(batch.events),
                        processing_time,
                        priority=batch.priority.name,
                        jobs_created=len(jobs)
                    )

                self._metrics_collector.stop_timer(f"batch_processing_{batch.batch_id}")
                self._metrics_collector.increment_counter("batches_processed")
                self._metrics_collector.record_histogram("batch_size", len(batch.events))
                self._metrics_collector.record_timer("batch_processing_time", processing_time)

            # Record legacy metrics for backwards compatibility
            if self._performance_monitoring:
                vault_ids = {event.vault_id for event in batch.events}

                for vault_id in vault_ids:
                    if vault_id not in self._sync_metrics:
                        self._sync_metrics[vault_id] = {
                            "batches_processed": 0,
                            "events_processed": 0,
                            "average_batch_time": 0.0,
                            "last_sync_time": None,
                        }

                    metrics = self._sync_metrics[vault_id]
                    metrics["batches_processed"] += 1
                    metrics["events_processed"] += len(batch.events)
                    metrics["average_batch_time"] = (
                        (metrics["average_batch_time"] * (metrics["batches_processed"] - 1) + processing_time)
                        / metrics["batches_processed"]
                    )
                    metrics["last_sync_time"] = datetime.utcnow().isoformat()

            logger.debug(f"Completed sync batch {batch.batch_id} in {processing_time:.2f}s")

            if self._audit_logger:
                await self._log_audit_event(
                    "sync_batch_processed",
                    "success",
                    metadata={
                        "batch_id": batch.batch_id,
                        "event_count": len(batch.events),
                        "priority": batch.priority.name,
                        "processing_time_seconds": processing_time,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to process sync batch {batch.batch_id}: {e}", exc_info=True)

            # Record error metrics
            if self._metrics_collector:
                vault_ids = {event.vault_id for event in batch.events}
                for vault_id in vault_ids:
                    self._metrics_collector.record_error(
                        "batch_processing_error",
                        vault_id,
                        str(e),
                        batch_id=batch.batch_id,
                        event_count=len(batch.events)
                    )
                self._metrics_collector.increment_counter("batch_processing_errors")

            if self._audit_logger:
                await self._log_audit_event(
                    "sync_batch_failed",
                    "error",
                    metadata={
                        "batch_id": batch.batch_id,
                        "error": str(e),
                        "event_count": len(batch.events),
                    }
                )

        finally:
            # Remove from active batches
            self._active_batches.pop(batch.batch_id, None)

    def _create_jobs_from_batch(self, batch: SyncBatch) -> List[IngestionJob]:
        """Create ingestion jobs from sync batch events."""
        jobs = []

        # Group events by vault for efficient processing
        vault_events: Dict[str, List[SyncEvent]] = {}
        for event in batch.events:
            if event.vault_id not in vault_events:
                vault_events[event.vault_id] = []
            vault_events[event.vault_id].append(event)

        # Create jobs per vault
        for vault_id, events in vault_events.items():
            job_id = str(uuid.uuid4())

            # Determine job type based on events
            if batch.is_path_change_batch:
                job_type_name = "obsidian_path_changes"
            elif len(events) == 1:
                job_type_name = "obsidian_file_sync"
            else:
                job_type_name = "obsidian_batch_sync"

            job = IngestionJob(
                job_id=job_id,
                job_type=JobType.LOCAL_FILES,
                payload={
                    "vault_id": vault_id,
                    "sync_batch_id": batch.batch_id,
                    "job_type": job_type_name,
                    "events": [event.to_job_payload() for event in events],
                    "trigger": "sync_engine",
                }
            )
            jobs.append(job)

        return jobs

    def _map_sync_to_job_priority(self, sync_priority: SyncPriority) -> JobPriority:
        """Map sync priority to job queue priority."""
        mapping = {
            SyncPriority.CRITICAL: JobPriority.HIGH,
            SyncPriority.HIGH: JobPriority.HIGH,
            SyncPriority.NORMAL: JobPriority.NORMAL,
            SyncPriority.LOW: JobPriority.LOW,
            SyncPriority.BACKGROUND: JobPriority.LOW,
        }
        return mapping.get(sync_priority, JobPriority.NORMAL)

    async def _log_audit_event(
        self,
        action: str,
        status: str,
        metadata: Optional[Dict[str, any]] = None
    ) -> None:
        """Log audit event for sync operations."""
        if not self._audit_logger:
            return

        from futurnal.orchestrator.audit import AuditEvent

        event = AuditEvent(
            job_id="sync_engine",
            source="obsidian_sync_engine",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        try:
            self._audit_logger.record(event)
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")


def create_sync_engine(
    vault_connector: ObsidianVaultConnector,
    job_queue: JobQueue,
    state_store: StateStore,
    audit_logger: Optional[AuditLogger] = None,
    **config
) -> ObsidianSyncEngine:
    """Factory function to create a sync engine with sensible defaults."""
    return ObsidianSyncEngine(
        vault_connector=vault_connector,
        job_queue=job_queue,
        state_store=state_store,
        audit_logger=audit_logger,
        **config
    )
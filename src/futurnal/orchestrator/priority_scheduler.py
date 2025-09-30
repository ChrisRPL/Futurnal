"""Priority-aware job scheduler with batching capabilities for Obsidian sync.

This module provides enhanced job scheduling capabilities that build on the
existing JobQueue to provide intelligent batching, priority handling, and
backpressure management specifically optimized for Obsidian vault synchronization.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .models import IngestionJob, JobPriority, JobType
from .queue import JobQueue

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """Strategies for batching jobs."""
    NONE = "none"
    BY_VAULT = "by_vault"
    BY_PRIORITY = "by_priority"
    BY_TYPE = "by_type"
    ADAPTIVE = "adaptive"


@dataclass
class JobBatch:
    """Represents a batch of related jobs."""
    batch_id: str
    jobs: List[IngestionJob]
    priority: JobPriority
    batch_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, any] = field(default_factory=dict)

    @property
    def job_count(self) -> int:
        """Number of jobs in this batch."""
        return len(self.jobs)

    @property
    def vault_ids(self) -> Set[str]:
        """Get all vault IDs involved in this batch."""
        vault_ids = set()
        for job in self.jobs:
            if isinstance(job.payload, dict) and 'vault_id' in job.payload:
                vault_ids.add(job.payload['vault_id'])
        return vault_ids

    def to_dict(self) -> Dict[str, any]:
        """Convert batch to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "job_count": self.job_count,
            "priority": self.priority.value,
            "batch_type": self.batch_type,
            "vault_ids": list(self.vault_ids),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class QueueMetrics:
    """Metrics for queue performance monitoring."""
    total_jobs_processed: int = 0
    total_batches_processed: int = 0
    average_batch_size: float = 0.0
    queue_depth_by_priority: Dict[JobPriority, int] = field(default_factory=dict)
    processing_time_by_priority: Dict[JobPriority, float] = field(default_factory=dict)
    backpressure_events: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)

    def update_batch_processed(self, batch: JobBatch, processing_time: float) -> None:
        """Update metrics when a batch is processed."""
        self.total_batches_processed += 1
        self.total_jobs_processed += batch.job_count

        # Update average batch size
        self.average_batch_size = (
            (self.average_batch_size * (self.total_batches_processed - 1) + batch.job_count)
            / self.total_batches_processed
        )

        # Update processing time by priority
        if batch.priority not in self.processing_time_by_priority:
            self.processing_time_by_priority[batch.priority] = processing_time
        else:
            current_avg = self.processing_time_by_priority[batch.priority]
            # Simple moving average (could be improved with more sophisticated tracking)
            self.processing_time_by_priority[batch.priority] = (current_avg + processing_time) / 2

        self.last_update = datetime.utcnow()


class PriorityScheduler:
    """Enhanced job scheduler with priority-aware batching and backpressure management.

    Builds on the existing JobQueue to provide sophisticated scheduling
    capabilities optimized for Obsidian vault synchronization patterns.
    """

    def __init__(
        self,
        job_queue: JobQueue,
        *,
        batch_window_seconds: float = 5.0,
        max_batch_size: int = 25,
        max_queue_depth: int = 1000,
        backpressure_threshold: float = 0.8,
        batching_strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE,
        enable_metrics: bool = True,
    ):
        self._job_queue = job_queue
        self._batch_window_seconds = batch_window_seconds
        self._max_batch_size = max_batch_size
        self._max_queue_depth = max_queue_depth
        self._backpressure_threshold = backpressure_threshold
        self._batching_strategy = batching_strategy
        self._enable_metrics = enable_metrics

        # Internal state
        self._pending_batches: Dict[str, JobBatch] = {}
        self._last_batch_time: Dict[str, float] = {}
        self._running = False
        self._batch_lock = asyncio.Lock()

        # Metrics
        self._metrics = QueueMetrics() if enable_metrics else None

        # Backpressure state
        self._backpressure_active = False
        self._last_backpressure_check = time.time()

        # Priority ordering (higher values = higher priority)
        self._priority_order = {
            JobPriority.HIGH: 100,
            JobPriority.NORMAL: 50,
            JobPriority.LOW: 25,
        }

    async def start(self) -> None:
        """Start the priority scheduler."""
        if self._running:
            return

        self._running = True
        logger.info("PriorityScheduler started")

    async def stop(self) -> None:
        """Stop the scheduler and flush pending batches."""
        if not self._running:
            return

        self._running = False

        # Process any pending batches
        await self._flush_all_pending_batches()

        logger.info("PriorityScheduler stopped")

    async def enqueue_job(self, job: IngestionJob, batch_key: Optional[str] = None) -> None:
        """Enqueue a job with intelligent batching.

        Args:
            job: The job to enqueue
            batch_key: Optional key for grouping jobs into batches
        """
        if not self._running:
            # Fallback to direct enqueue if scheduler not running
            self._job_queue.enqueue(job)
            return

        # Check backpressure
        if await self._check_backpressure():
            logger.warning(f"Backpressure active, dropping job {job.job_id}")
            if self._metrics:
                self._metrics.backpressure_events += 1
            return

        # Determine batch key if not provided
        if batch_key is None:
            batch_key = self._determine_batch_key(job)

        async with self._batch_lock:
            # Get or create batch
            batch = self._get_or_create_batch(batch_key, job)
            batch.jobs.append(job)

            # Check if batch should be processed
            current_time = time.time()
            last_batch = self._last_batch_time.get(batch_key, 0)
            should_process = (
                len(batch.jobs) >= self._max_batch_size or
                current_time - last_batch >= self._batch_window_seconds or
                job.priority == JobPriority.HIGH
            )

            if should_process:
                await self._process_batch(batch_key)

    async def enqueue_jobs_batch(self, jobs: List[IngestionJob], batch_key: Optional[str] = None) -> None:
        """Enqueue multiple jobs as a single batch.

        Args:
            jobs: List of jobs to enqueue
            batch_key: Optional key for the batch
        """
        if not jobs:
            return

        if not self._running:
            # Fallback to direct enqueue
            for job in jobs:
                self._job_queue.enqueue(job)
            return

        # Use the first job's characteristics for batch key if not provided
        if batch_key is None:
            batch_key = self._determine_batch_key(jobs[0])

        async with self._batch_lock:
            batch = self._get_or_create_batch(batch_key, jobs[0])
            batch.jobs.extend(jobs)

            # Process the batch immediately since it was explicitly batched
            await self._process_batch(batch_key)

    async def get_queue_status(self) -> Dict[str, any]:
        """Get current queue status and metrics."""
        pending_count = self._job_queue.pending_count()

        # Update queue depth metrics
        if self._metrics:
            # This is a simplified approach - in practice you'd track per-priority counts
            self._metrics.queue_depth_by_priority[JobPriority.HIGH] = pending_count

        status = {
            "running": self._running,
            "pending_jobs": pending_count,
            "pending_batches": len(self._pending_batches),
            "backpressure_active": self._backpressure_active,
            "batching_strategy": self._batching_strategy.value,
        }

        if self._metrics:
            status["metrics"] = {
                "total_jobs_processed": self._metrics.total_jobs_processed,
                "total_batches_processed": self._metrics.total_batches_processed,
                "average_batch_size": self._metrics.average_batch_size,
                "backpressure_events": self._metrics.backpressure_events,
                "last_update": self._metrics.last_update.isoformat(),
            }

        return status

    async def trigger_batch_processing(self) -> None:
        """Manually trigger processing of all pending batches."""
        async with self._batch_lock:
            await self._flush_all_pending_batches()

    def _determine_batch_key(self, job: IngestionJob) -> str:
        """Determine the batch key for a job based on batching strategy."""
        if self._batching_strategy == BatchingStrategy.NONE:
            return f"single_{job.job_id}"

        payload = job.payload if isinstance(job.payload, dict) else {}

        if self._batching_strategy == BatchingStrategy.BY_VAULT:
            vault_id = payload.get('vault_id', 'unknown')
            return f"vault_{vault_id}_{job.priority.value}"

        elif self._batching_strategy == BatchingStrategy.BY_PRIORITY:
            return f"priority_{job.priority.value}"

        elif self._batching_strategy == BatchingStrategy.BY_TYPE:
            job_type = payload.get('job_type', job.job_type.value)
            return f"type_{job_type}_{job.priority.value}"

        elif self._batching_strategy == BatchingStrategy.ADAPTIVE:
            # Adaptive strategy: batch by vault for normal/low priority,
            # individual processing for high priority
            if job.priority == JobPriority.HIGH:
                return f"high_priority_{job.job_id}"
            else:
                vault_id = payload.get('vault_id', 'unknown')
                return f"adaptive_{vault_id}_{job.priority.value}"

        return f"default_{job.priority.value}"

    def _get_or_create_batch(self, batch_key: str, reference_job: IngestionJob) -> JobBatch:
        """Get existing batch or create a new one."""
        if batch_key not in self._pending_batches:
            # Determine batch type from reference job
            payload = reference_job.payload if isinstance(reference_job.payload, dict) else {}
            batch_type = payload.get('job_type', reference_job.job_type.value)

            self._pending_batches[batch_key] = JobBatch(
                batch_id=batch_key,
                jobs=[],
                priority=reference_job.priority,
                batch_type=batch_type,
                metadata={"strategy": self._batching_strategy.value}
            )

        return self._pending_batches[batch_key]

    async def _process_batch(self, batch_key: str) -> None:
        """Process a pending batch by enqueuing jobs to the underlying queue."""
        batch = self._pending_batches.pop(batch_key, None)
        if not batch or not batch.jobs:
            return

        start_time = time.time()

        try:
            # Sort jobs within batch by priority
            batch.jobs.sort(key=lambda j: self._priority_order.get(j.priority, 0), reverse=True)

            # Enqueue all jobs in the batch
            for job in batch.jobs:
                self._job_queue.enqueue(job)

            self._last_batch_time[batch_key] = time.time()

            processing_time = time.time() - start_time

            logger.debug(
                f"Processed batch {batch.batch_id}: {len(batch.jobs)} jobs "
                f"(priority: {batch.priority.name}, time: {processing_time:.3f}s)"
            )

            # Update metrics
            if self._metrics:
                self._metrics.update_batch_processed(batch, processing_time)

        except Exception as e:
            logger.error(f"Failed to process batch {batch.batch_id}: {e}", exc_info=True)

            # Re-add jobs individually as fallback
            for job in batch.jobs:
                try:
                    self._job_queue.enqueue(job)
                except Exception as job_error:
                    logger.error(f"Failed to enqueue job {job.job_id}: {job_error}")

    async def _flush_all_pending_batches(self) -> None:
        """Flush all pending batches."""
        batch_keys = list(self._pending_batches.keys())
        for batch_key in batch_keys:
            await self._process_batch(batch_key)

    async def _check_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        current_time = time.time()

        # Only check periodically to avoid excessive overhead
        if current_time - self._last_backpressure_check < 5.0:
            return self._backpressure_active

        self._last_backpressure_check = current_time

        # Check queue depth
        pending_count = self._job_queue.pending_count()
        depth_ratio = pending_count / self._max_queue_depth

        # Activate backpressure if queue is too full
        should_activate = depth_ratio >= self._backpressure_threshold

        if should_activate != self._backpressure_active:
            self._backpressure_active = should_activate
            if should_activate:
                logger.warning(f"Backpressure activated: queue depth {pending_count}/{self._max_queue_depth}")
            else:
                logger.info("Backpressure deactivated")

        return self._backpressure_active

    def get_metrics(self) -> Optional[QueueMetrics]:
        """Get current metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        if self._metrics:
            self._metrics = QueueMetrics()


def create_priority_scheduler(
    job_queue: JobQueue,
    **config
) -> PriorityScheduler:
    """Factory function to create a priority scheduler with sensible defaults."""
    return PriorityScheduler(job_queue, **config)
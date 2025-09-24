"""Async ingestion orchestrator leveraging APScheduler."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Dict, Iterable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from croniter import croniter

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..ingestion.local.connector import ElementSink, LocalFilesConnector
from ..ingestion.local.config import LocalIngestionSource
from ..ingestion.local.state import StateStore
from ..privacy.consent import ConsentRegistry
from .models import IngestionJob, JobPriority, JobType
from .audit import AuditEvent, AuditLogger
from .metrics import TelemetryRecorder
from .queue import JobQueue

logger = logging.getLogger(__name__)


@dataclass
class SourceRegistration:
    source: LocalIngestionSource
    schedule: str
    priority: JobPriority = JobPriority.NORMAL


class IngestionOrchestrator:
    """Coordinates ingestion jobs using a persistent queue and async workers."""

    def __init__(
        self,
        *,
        job_queue: JobQueue,
        state_store: StateStore,
        workspace_dir: str,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        element_sink: ElementSink | None = None,
        telemetry: TelemetryRecorder | None = None,
    ) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._job_queue = job_queue
        self._loop = loop or asyncio.get_event_loop()
        self._scheduler = AsyncIOScheduler(event_loop=self._loop)
        self._sources: Dict[str, SourceRegistration] = {}
        self._running = False
        self._telemetry = telemetry or TelemetryRecorder(self._workspace_dir / "telemetry")
        self._audit_logger = AuditLogger(self._workspace_dir / "audit")
        consent_dir = self._workspace_dir / "privacy"
        self._consent_registry = ConsentRegistry(consent_dir)
        self._connector = LocalFilesConnector(
            workspace_dir=self._workspace_dir,
            state_store=state_store,
            element_sink=element_sink,
            audit_logger=self._audit_logger,
            consent_registry=self._consent_registry,
        )
        self._max_retries = 3
        self._retry_backoff_seconds = 60
        hardware_cpu_count = os.cpu_count() or 4
        self._hardware_worker_cap = max(1, min(hardware_cpu_count, 8))
        self._configured_workers = self._hardware_worker_cap
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_jobs = 0
        self._debounce_windows: Dict[str, float] = {}
        self._observer: Optional[Observer] = None

    def register_source(self, registration: SourceRegistration) -> None:
        if registration.schedule != "@manual" and not croniter.is_valid(registration.schedule):
            raise ValueError(f"Invalid cron schedule: {registration.schedule}")
        self._sources[registration.source.name] = registration
        if registration.source.max_workers:
            requested = registration.source.max_workers
            capped = max(1, min(requested, self._hardware_worker_cap))
            self._configured_workers = min(self._configured_workers, capped)
        if registration.schedule != "@manual":
            trigger = CronTrigger.from_crontab(registration.schedule)
            self._scheduler.add_job(
                self._enqueue_job,
                trigger=trigger,
                args=[registration.source.name],
                id=f"schedule-{registration.source.name}",
                replace_existing=True,
            )
        logger.info("Registered source %s", registration.source.name)
        self._register_file_watch(registration.source)
        if registration.source.scan_interval_seconds:
            seconds = int(registration.source.scan_interval_seconds)
            if seconds > 0:
                self._scheduler.add_job(
                    self._enqueue_job,
                    trigger=IntervalTrigger(seconds=seconds),
                    args=[registration.source.name],
                    id=f"fallback-{registration.source.name}",
                    replace_existing=True,
                )

    def register_interval_source(
        self, registration: SourceRegistration, *, seconds: int
    ) -> None:
        self._sources[registration.source.name] = registration
        self._scheduler.add_job(
            self._enqueue_job,
            trigger=IntervalTrigger(seconds=seconds),
            args=[registration.source.name],
            id=f"interval-{registration.source.name}",
            replace_existing=True,
        )

    def start(self) -> None:
        if self._running:
            return
        self._scheduler.start()
        self._loop.create_task(self._worker_loop())
        self._start_observer()
        self._running = True
        logger.info("Ingestion orchestrator started")

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._scheduler.shutdown(wait=False)
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        logger.info("Ingestion orchestrator stopped")

    def run_manual_job(self, source_name: str) -> None:
        if source_name not in self._sources:
            raise KeyError(f"Unknown source {source_name}")
        self._enqueue_job(source_name)

    def _enqueue_job(self, source_name: str) -> None:
        registration = self._sources[source_name]
        job_id = str(uuid.uuid4())
        now = time.perf_counter()
        debounce = registration.source.watcher_debounce_seconds or 0.0
        last_event = self._debounce_windows.get(source_name)
        if debounce and last_event and (now - last_event) < debounce:
            return
        self._debounce_windows[source_name] = now
        job = IngestionJob(
            job_id=job_id,
            job_type=JobType.LOCAL_FILES,
            payload={
                "source_name": source_name,
            },
            priority=registration.priority,
            scheduled_for=datetime.utcnow(),
        )
        self._job_queue.enqueue(job)
        logger.debug("Enqueued job %s for source %s", job_id, source_name)

    async def _worker_loop(self) -> None:
        max_workers = self._configured_workers or 1
        self._semaphore = asyncio.Semaphore(max_workers)
        while self._running:
            async for job in self._fetch_jobs():
                await self._semaphore.acquire()
                self._active_jobs += 1
                self._loop.create_task(self._run_job(job))
            await asyncio.sleep(0.1)

    async def _fetch_jobs(self):
        # Convert iterator to async generator
        for job in self._job_queue.fetch_pending():
            yield job

    async def _run_job(self, job: IngestionJob) -> None:
        try:
            await self._process_job(job)
        finally:
            self._active_jobs -= 1
            if self._semaphore:
                self._semaphore.release()

    async def _process_job(self, job: IngestionJob) -> None:
        logger.info("Processing job %s", job.job_id)
        self._job_queue.mark_running(job.job_id)
        start = time.perf_counter()
        files_processed = 0
        bytes_processed = 0
        try:
            files_processed, bytes_processed = await self._execute_job(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job.job_id)
            self._job_queue.mark_failed(job.job_id)
            if self._telemetry:
                duration = time.perf_counter() - start
                self._telemetry.record(
                    job.job_id,
                    duration,
                    "failed",
                    files_processed=files_processed,
                    bytes_processed=bytes_processed,
                    active_workers=self._active_jobs,
                    configured_workers=self._configured_workers,
                    queue_depth=self._job_queue.pending_count(),
                    effective_throughput=(bytes_processed / duration) if duration else None,
                )
            job.payload.update(
                {
                    "files_processed": files_processed,
                    "bytes_processed": bytes_processed,
                    "error": str(exc),
                }
            )
            self._record_audit_event(job, "failed")
            await self._maybe_retry(job)
        else:
            self._job_queue.mark_completed(job.job_id)
            duration = time.perf_counter() - start
            logger.info("Job %s completed in %.2fs", job.job_id, duration)
            if self._telemetry:
                queue_depth = self._job_queue.pending_count()
                throughput = bytes_processed / duration if duration else None
                self._telemetry.record(
                    job.job_id,
                    duration,
                    "succeeded",
                    files_processed=files_processed,
                    bytes_processed=bytes_processed,
                    active_workers=self._active_jobs,
                    configured_workers=self._configured_workers,
                    queue_depth=queue_depth,
                    effective_throughput=throughput,
                )
            job.payload.update(
                {
                    "files_processed": files_processed,
                    "bytes_processed": bytes_processed,
                }
            )
            self._record_audit_event(job, "succeeded")

    async def _execute_job(self, job: IngestionJob) -> None:
        if job.job_type == JobType.LOCAL_FILES:
            return await self._ingest_local(job)
        else:
            raise ValueError(f"Unsupported job type {job.job_type}")

    async def _ingest_local(self, job: IngestionJob) -> None:
        source_name = job.payload["source_name"]
        registration = self._sources[source_name]
        files_processed = 0
        bytes_processed = 0
        for element in self._connector.ingest(registration.source, job_id=job.job_id):
            await self._handle_ingested_element(element)
            files_processed += 1
            bytes_processed += element.get("size_bytes", 0)
        return files_processed, bytes_processed

    async def _handle_ingested_element(self, element: dict) -> None:
        logger.debug("Ingested element from %s", element.get("path"))

    # Removed helper; ingestion happens synchronously within event loop execution

    def _record_audit_event(self, job: IngestionJob, status: str) -> None:
        logger.debug("Audit event job=%s status=%s", job.job_id, status)
        detail = {
            "files_processed": job.payload.get("files_processed"),
            "bytes_processed": job.payload.get("bytes_processed"),
        }
        if status == "failed":
            detail["error"] = job.payload.get("error")
        metadata = {k: v for k, v in detail.items() if v is not None}
        event = AuditEvent(
            job_id=job.job_id,
            source=job.payload["source_name"],
            action="job",
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
        self._audit_logger.record(event)

    async def _maybe_retry(self, job: IngestionJob) -> None:
        record = DataRetryRecord(job_id=job.job_id, attempts=job.payload.get("attempts", 0))
        if record.attempts >= self._max_retries:
            logger.error("Job %s exceeded retry limit", job.job_id)
            return

        record.attempts += 1
        job.payload["attempts"] = record.attempts
        logger.info(
            "Rescheduling job %s (attempt %s/%s)", job.job_id, record.attempts, self._max_retries
        )
        self._job_queue.reschedule(job.job_id, self._retry_backoff_seconds)
        await asyncio.sleep(self._retry_backoff_seconds)

    def _start_observer(self) -> None:
        if self._observer is not None:
            return
        observer = Observer()
        for registration in self._sources.values():
            handler = _SourceEventHandler(self, registration.source)
            observer.schedule(handler, str(registration.source.root_path), recursive=True)
        observer.start()
        self._observer = observer

    def _register_file_watch(self, source: LocalIngestionSource) -> None:
        if self._observer is None:
            return
        handler = _SourceEventHandler(self, source)
        self._observer.schedule(handler, str(source.root_path), recursive=True)


class _SourceEventHandler(FileSystemEventHandler):
    def __init__(self, orchestrator: IngestionOrchestrator, source: LocalIngestionSource) -> None:
        self._orchestrator = orchestrator
        self._source = source

    def on_any_event(self, event):  # pragma: no cover - requires filesystem events
        if event.is_directory:
            return
        self._orchestrator._enqueue_job(self._source.name)


@dataclass
class DataRetryRecord:
    job_id: str
    attempts: int = 0



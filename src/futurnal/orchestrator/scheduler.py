"""Async ingestion orchestrator leveraging APScheduler."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from croniter import croniter

from ..ingestion.local.connector import ElementSink, LocalFilesConnector
from ..ingestion.local.config import LocalIngestionSource
from ..ingestion.local.state import StateStore
from .models import IngestionJob, JobPriority, JobType
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
        self._job_queue = job_queue
        self._connector = LocalFilesConnector(
            workspace_dir=workspace_dir, state_store=state_store, element_sink=element_sink
        )
        self._loop = loop or asyncio.get_event_loop()
        self._scheduler = AsyncIOScheduler(event_loop=self._loop)
        self._sources: Dict[str, SourceRegistration] = {}
        self._running = False
        self._telemetry = telemetry

    def register_source(self, registration: SourceRegistration) -> None:
        if registration.schedule != "@manual" and not croniter.is_valid(registration.schedule):
            raise ValueError(f"Invalid cron schedule: {registration.schedule}")
        self._sources[registration.source.name] = registration
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
        self._running = True
        logger.info("Ingestion orchestrator started")

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("Ingestion orchestrator stopped")

    def run_manual_job(self, source_name: str) -> None:
        if source_name not in self._sources:
            raise KeyError(f"Unknown source {source_name}")
        self._enqueue_job(source_name)

    def _enqueue_job(self, source_name: str) -> None:
        registration = self._sources[source_name]
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            job_id=job_id,
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": source_name},
            priority=registration.priority,
            scheduled_for=datetime.utcnow(),
        )
        self._job_queue.enqueue(job)
        logger.debug("Enqueued job %s for source %s", job_id, source_name)

    async def _worker_loop(self) -> None:
        while self._running:
            async for job in self._fetch_jobs():
                await self._process_job(job)
            await asyncio.sleep(1)

    async def _fetch_jobs(self):
        # Convert iterator to async generator
        for job in self._job_queue.fetch_pending():
            yield job

    async def _process_job(self, job: IngestionJob) -> None:
        logger.info("Processing job %s", job.job_id)
        self._job_queue.mark_running(job.job_id)
        start = time.perf_counter()
        try:
            await self._execute_job(job)
        except Exception:  # noqa: BLE001
            logger.exception("Job %s failed", job.job_id)
            self._job_queue.mark_failed(job.job_id)
            # TODO: emit audit log entry when privacy module is available
            if self._telemetry:
                self._telemetry.record(job.job_id, time.perf_counter() - start, "failed")
        else:
            self._job_queue.mark_completed(job.job_id)
            duration = time.perf_counter() - start
            logger.info("Job %s completed in %.2fs", job.job_id, duration)
            if self._telemetry:
                self._telemetry.record(job.job_id, duration, "succeeded")

    async def _execute_job(self, job: IngestionJob) -> None:
        if job.job_type == JobType.LOCAL_FILES:
            await self._ingest_local(job)
        else:
            raise ValueError(f"Unsupported job type {job.job_type}")

    async def _ingest_local(self, job: IngestionJob) -> None:
        source_name = job.payload["source_name"]
        registration = self._sources[source_name]
        for element in self._connector.ingest(registration.source):
            await self._handle_ingested_element(element)

    async def _handle_ingested_element(self, element: dict) -> None:
        # Placeholder for downstream normalization and storage
        logger.debug("Ingested element from %s", element.get("path"))



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
from typing import Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
try:
    from croniter import is_valid
except ImportError:
    # Fallback for older croniter versions
    from croniter import croniter
    def is_valid(cron_expression):
        try:
            croniter(cron_expression)
            return True
        except:
            return False

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
    interval_seconds: Optional[int] = None
    priority: JobPriority = JobPriority.NORMAL
    paused: bool = False


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
        self._local_connector = LocalFilesConnector(
            workspace_dir=self._workspace_dir,
            state_store=state_store,
            element_sink=element_sink,
            audit_logger=self._audit_logger,
            consent_registry=self._consent_registry,
        )
        
        # Initialize Obsidian connector
        from ..ingestion.obsidian.connector import ObsidianVaultConnector
        from ..ingestion.obsidian.descriptor import VaultRegistry

        self._vault_registry = VaultRegistry()
        self._obsidian_connector = ObsidianVaultConnector(
            workspace_dir=self._workspace_dir,
            state_store=state_store,
            vault_registry=self._vault_registry,
            element_sink=element_sink,
            audit_logger=self._audit_logger,
            consent_registry=self._consent_registry,
        )

        # Initialize IMAP connector
        from ..ingestion.imap.connector import ImapEmailConnector
        from ..ingestion.imap.descriptor import MailboxRegistry
        from ..ingestion.imap.sync_state import ImapSyncStateStore

        self._mailbox_registry = MailboxRegistry(
            registry_root=self._workspace_dir / "sources" / "imap",
            audit_logger=self._audit_logger,
        )
        imap_state_db = self._workspace_dir / "imap" / "sync_state.db"
        self._imap_state_store = ImapSyncStateStore(path=imap_state_db)
        self._imap_connector = ImapEmailConnector(
            workspace_dir=self._workspace_dir,
            state_store=self._imap_state_store,
            mailbox_registry=self._mailbox_registry,
            element_sink=element_sink,
            audit_logger=self._audit_logger,
            consent_registry=self._consent_registry,
        )

        # Initialize GitHub connector
        from ..ingestion.github.api_client_manager import GitHubAPIClientManager
        from ..ingestion.github.credential_manager import GitHubCredentialManager
        from ..ingestion.github.orchestrator_integration import GitHubConnectorManager

        github_workspace = self._workspace_dir / "github"
        github_workspace.mkdir(parents=True, exist_ok=True)

        self._github_credential_manager = GitHubCredentialManager()
        self._github_api_client_manager = GitHubAPIClientManager(
            credential_manager=self._github_credential_manager
        )
        self._github_connector = GitHubConnectorManager(
            workspace_dir=self._workspace_dir,
            credential_manager=self._github_credential_manager,
            api_client_manager=self._github_api_client_manager,
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
        if registration.schedule not in {"@manual", "@interval"} and not is_valid(
            registration.schedule
        ):
            raise ValueError(f"Invalid cron schedule: {registration.schedule}")
        if registration.schedule == "@interval":
            if not registration.interval_seconds:
                raise ValueError("interval_seconds must be provided for '@interval' schedules")
            if registration.interval_seconds <= 0:
                raise ValueError("interval_seconds must be greater than zero")
        self._sources[registration.source.name] = registration
        if registration.source.max_workers:
            requested = registration.source.max_workers
            capped = max(1, min(requested, self._hardware_worker_cap))
            self._configured_workers = min(self._configured_workers, capped)
        if registration.schedule == "@interval" and registration.interval_seconds:
            self._scheduler.add_job(
                self._enqueue_job,
                trigger=IntervalTrigger(seconds=registration.interval_seconds),
                args=[registration.source.name],
                kwargs={"trigger": "interval"},
                id=f"interval-{registration.source.name}",
                replace_existing=True,
            )
        elif registration.schedule != "@manual":
            trigger = CronTrigger.from_crontab(registration.schedule)
            self._scheduler.add_job(
                self._enqueue_job,
                trigger=trigger,
                args=[registration.source.name],
                kwargs={"trigger": "schedule"},
                id=f"schedule-{registration.source.name}",
                replace_existing=True,
            )
        logger.info(
            "Registered source",
            extra={
                "ingestion_source": registration.source.name,
                "ingestion_schedule": registration.schedule,
            },
        )
        self._register_file_watch(registration.source)
        if registration.source.scan_interval_seconds:
            seconds = int(registration.source.scan_interval_seconds)
            if seconds > 0:
                self._scheduler.add_job(
                    self._enqueue_job,
                    trigger=IntervalTrigger(seconds=seconds),
                    args=[registration.source.name],
                    kwargs={"trigger": "fallback"},
                    id=f"fallback-{registration.source.name}",
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

    def run_manual_job(self, source_name: str, *, force: bool = False) -> None:
        if source_name not in self._sources:
            raise KeyError(f"Unknown source {source_name}")
        self._enqueue_job(source_name, force=force, trigger="manual")

    def _enqueue_job(self, source_name: str, *, force: bool = False, trigger: str = "schedule") -> None:
        registration = self._sources[source_name]
        if registration.paused and not force:
            logger.debug(
                "Skipping enqueue because source is paused", extra={"ingestion_source": source_name, "ingestion_event": "pause_skip"}
            )
            return
        job_id = str(uuid.uuid4())
        now = time.perf_counter()
        debounce = registration.source.watcher_debounce_seconds or 0.0
        last_event = self._debounce_windows.get(source_name)
        if debounce and last_event and (now - last_event) < debounce:
            return
        self._debounce_windows[source_name] = now

        # Determine job type from source name
        if source_name.startswith("imap-"):
            job_type = JobType.IMAP_MAILBOX
            # Extract mailbox_id from source name (format: "imap-{mailbox_id[:8]}")
            mailbox_id = registration.source.root_path.name  # Gets mailbox ID from workspace path
            payload = {
                "mailbox_id": mailbox_id,
                "trigger": trigger,
            }
        else:
            job_type = JobType.LOCAL_FILES
            payload = {
                "source_name": source_name,
                "trigger": trigger,
            }

        job = IngestionJob(
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            priority=registration.priority,
            scheduled_for=datetime.utcnow(),
        )
        self._job_queue.enqueue(job)
        logger.debug(
            "Enqueued job",
            extra={
                "ingestion_job_id": job_id,
                "ingestion_source": source_name,
                "ingestion_trigger": trigger,
                "job_type": job_type.value,
            },
        )

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
        logger.info(
            "Processing job",
            extra={
                "ingestion_job_id": job.job_id,
                "ingestion_source": job.payload.get("source_name"),
            },
        )
        self._job_queue.mark_running(job.job_id)
        start = time.perf_counter()
        files_processed = 0
        bytes_processed = 0
        try:
            files_processed, bytes_processed = await self._execute_job(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Job failed",
                extra={
                    "ingestion_job_id": job.job_id,
                    "ingestion_source": job.payload.get("source_name"),
                },
            )
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
            logger.info(
                "Job completed",
                extra={
                    "ingestion_job_id": job.job_id,
                    "ingestion_source": job.payload.get("source_name"),
                    "ingestion_duration_s": round(duration, 2),
                },
            )
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
        elif job.job_type == JobType.OBSIDIAN_VAULT:
            return await self._ingest_obsidian(job)
        elif job.job_type == JobType.IMAP_MAILBOX:
            return await self._ingest_imap(job)
        elif job.job_type == JobType.GITHUB_REPOSITORY:
            return await self._ingest_github(job)
        else:
            raise ValueError(f"Unsupported job type {job.job_type}")

    async def _ingest_local(self, job: IngestionJob) -> None:
        source_name = job.payload["source_name"]
        registration = self._sources[source_name]
        files_processed = 0
        bytes_processed = 0
        for element in self._local_connector.ingest(registration.source, job_id=job.job_id):
            await self._handle_ingested_element(element)
            files_processed += 1
            bytes_processed += element.get("size_bytes", 0)
        return files_processed, bytes_processed

    async def _ingest_obsidian(self, job: IngestionJob) -> tuple[int, int]:
        source_name = job.payload["source_name"]
        registration = self._sources[source_name]
        files_processed = 0
        bytes_processed = 0
        for element in self._obsidian_connector.ingest(registration.source, job_id=job.job_id):
            await self._handle_ingested_element(element)
            files_processed += 1
            bytes_processed += element.get("size_bytes", 0)
        return files_processed, bytes_processed

    async def _ingest_imap(self, job: IngestionJob) -> tuple[int, int]:
        """Process IMAP mailbox sync job.

        Args:
            job: Ingestion job with mailbox_id in payload

        Returns:
            Tuple of (messages_processed, bytes_processed)
        """
        mailbox_id = job.payload["mailbox_id"]
        messages_processed = 0
        bytes_processed = 0

        # Ingest mailbox (runs async sync internally)
        for element in self._imap_connector.ingest(mailbox_id, job_id=job.job_id):
            await self._handle_ingested_element(element)
            messages_processed += element.get("new_messages", 0)
            # Estimate bytes (actual message size tracking would require more instrumentation)
            bytes_processed += messages_processed * 5000  # Rough estimate: 5KB per message

        return messages_processed, bytes_processed

    async def _ingest_github(self, job: IngestionJob) -> tuple[int, int]:
        """Process GitHub repository sync job.

        Args:
            job: Ingestion job with repo_id in payload

        Returns:
            Tuple of (files_processed, bytes_processed)
        """
        repo_id = job.payload["repo_id"]
        trigger = job.payload.get("trigger", "schedule")

        logger.info(
            f"Processing GitHub repository sync: {repo_id} (trigger: {trigger})",
            extra={
                "ingestion_job_id": job.job_id,
                "github_repo_id": repo_id,
                "github_trigger": trigger,
            },
        )

        # Perform sync via GitHub connector manager
        result = await self._github_connector.sync_repository(
            repo_id=repo_id,
            job_id=job.job_id,
        )

        files_processed = result.files_synced
        bytes_processed = result.bytes_synced

        # Log sync result
        if result.is_success():
            logger.info(
                f"GitHub sync completed: {files_processed} files, {bytes_processed} bytes",
                extra={
                    "ingestion_job_id": job.job_id,
                    "github_repo_id": repo_id,
                    "github_files_synced": files_processed,
                    "github_bytes_synced": bytes_processed,
                    "github_commits_processed": result.commits_processed,
                },
            )
        else:
            logger.error(
                f"GitHub sync failed: {result.error_message}",
                extra={
                    "ingestion_job_id": job.job_id,
                    "github_repo_id": repo_id,
                    "github_error": result.error_message,
                },
            )
            # Raise exception to trigger retry logic
            raise RuntimeError(f"GitHub sync failed: {result.error_message}")

        return files_processed, bytes_processed

    async def _handle_ingested_element(self, element: dict) -> None:
        logger.debug(
            "Ingested element", extra={"ingestion_source": element.get("source")}
        )

    # Removed helper; ingestion happens synchronously within event loop execution

    def _record_audit_event(self, job: IngestionJob, status: str) -> None:
        logger.debug(
            "Audit event emitted",
            extra={
                "ingestion_job_id": job.job_id,
                "ingestion_source": job.payload.get("source_name"),
                "ingestion_status": status,
            },
        )
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
            logger.error(
                "Job exceeded retry limit",
                extra={
                    "ingestion_job_id": job.job_id,
                    "ingestion_source": job.payload.get("source_name"),
                },
            )
            return

        record.attempts += 1
        job.payload["attempts"] = record.attempts
        logger.info(
            "Rescheduling job",
            extra={
                "ingestion_job_id": job.job_id,
                "ingestion_source": job.payload.get("source_name"),
                "ingestion_attempt": record.attempts,
                "ingestion_max_retries": self._max_retries,
            },
        )
        job.payload["trigger"] = "retry"
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
        self._orchestrator._enqueue_job(self._source.name, trigger="watcher")


@dataclass
class DataRetryRecord:
    job_id: str
    attempts: int = 0



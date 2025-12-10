"""Async ingestion orchestrator leveraging APScheduler."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
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
from .quarantine import QuarantineStore, classify_failure, quarantine_reason_to_failure_type
from .retry_policy import RetryBudget, RetryPolicyRegistry
from .resource_monitor import ResourceMonitor
from .resource_registry import ResourceProfileRegistry
from .source_control import PausedSourcesRegistry
from .crash_recovery import CrashRecoveryManager

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
        quarantine_store: QuarantineStore | None = None,
        retry_policy_registry: Optional[RetryPolicyRegistry] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        resource_profiles: Optional[ResourceProfileRegistry] = None,
        crash_recovery: Optional[CrashRecoveryManager] = None,
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

        # Crash recovery system
        self._crash_recovery = crash_recovery or CrashRecoveryManager(
            job_queue=job_queue,
            workspace_dir=self._workspace_dir,
            audit_logger=self._audit_logger,
        )

        # Check if recovering from crash on startup
        if self._crash_recovery._recovery_tracker.is_recovering_from_crash():
            logger.warning("Crash detected - initiating recovery")
            recovery_report = self._crash_recovery.recover_from_crash()

            if not recovery_report.was_successful():
                logger.error(
                    "Crash recovery completed with errors",
                    extra={"errors": recovery_report.errors},
                )
            else:
                logger.info(
                    "Crash recovery completed successfully",
                    extra={
                        "jobs_reset": recovery_report.jobs_reset_to_pending,
                        "duration_seconds": recovery_report.recovery_duration_seconds,
                    },
                )
        self._quarantine = quarantine_store or QuarantineStore(
            self._workspace_dir / "quarantine" / "quarantine.db"
        )

        # Retry policy system
        self._retry_policies = retry_policy_registry or RetryPolicyRegistry()
        self._retry_budgets: Dict[str, RetryBudget] = {}

        # Resource profiling system
        self._resource_monitor = resource_monitor or ResourceMonitor(telemetry=self._telemetry)
        self._resource_profiles = resource_profiles or ResourceProfileRegistry()
        self._per_connector_semaphores: Dict[JobType, asyncio.Semaphore] = {}
        self._sampling_task: Optional[asyncio.Task] = None

        # Source pause/resume registry
        self._paused_sources = PausedSourcesRegistry(
            self._workspace_dir / "orchestrator" / "paused_sources.json"
        )

        # Store element sink and state store for lazy connector initialization
        self._element_sink = element_sink
        self._state_store = state_store

        # Initialize LOCAL_FILES connector eagerly (always available)
        self._local_connector = LocalFilesConnector(
            workspace_dir=self._workspace_dir,
            state_store=state_store,
            element_sink=element_sink,
            audit_logger=self._audit_logger,
            consent_registry=self._consent_registry,
        )

        # Lazy-initialized connectors (only created when needed)
        self._obsidian_connector = None
        self._vault_registry = None
        self._imap_connector = None
        self._mailbox_registry = None
        self._imap_state_store = None
        self._github_connector = None
        self._github_credential_manager = None
        self._github_api_client_manager = None

        # Temporary storage for resource metrics (populated in _run_job, consumed in _process_job)
        self._job_resource_metrics: Dict[str, any] = {}

        self._max_retries = 3
        self._retry_backoff_seconds = 60
        hardware_cpu_count = os.cpu_count() or 4
        self._hardware_worker_cap = max(1, min(hardware_cpu_count, 8))
        self._configured_workers = self._hardware_worker_cap
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_jobs = 0
        self._active_jobs_map: Dict[str, Dict] = {}  # Track active job IDs for invariant checking
        self._debounce_windows: Dict[str, float] = {}
        self._observer: Optional[Observer] = None

        # Deadlock detection and invariant checking
        from .deadlock import DeadlockDetector
        self._deadlock_detector = DeadlockDetector(
            queue=self._job_queue,
            timeout_seconds=600,  # 10 minutes default
        )
        self._deadlock_detection_task: Optional[asyncio.Task] = None
        self._invariant_check_task: Optional[asyncio.Task] = None

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
                next_run_time=datetime.now(),  # Run immediately on registration
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

        # Set recovery marker (cleared on graceful shutdown)
        self._crash_recovery._recovery_tracker.mark_crash()

        # Initialize per-connector semaphores based on resource profiles
        self._initialize_connector_semaphores()
        self._scheduler.start()
        self._loop.create_task(self._worker_loop())
        self._start_observer()
        # Start periodic resource sampling
        self._sampling_task = self._loop.create_task(self._resource_sampling_loop())
        # Start periodic deadlock detection
        self._deadlock_detection_task = self._loop.create_task(self._deadlock_detection_loop())
        # Start periodic invariant checking
        self._invariant_check_task = self._loop.create_task(self._invariant_check_loop())
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
        # Cancel resource sampling task
        if self._sampling_task:
            self._sampling_task.cancel()
            try:
                await self._sampling_task
            except asyncio.CancelledError:
                pass
            self._sampling_task = None
        # Cancel deadlock detection task
        if self._deadlock_detection_task:
            self._deadlock_detection_task.cancel()
            try:
                await self._deadlock_detection_task
            except asyncio.CancelledError:
                pass
            self._deadlock_detection_task = None
        # Cancel invariant check task
        if self._invariant_check_task:
            self._invariant_check_task.cancel()
            try:
                await self._invariant_check_task
            except asyncio.CancelledError:
                pass
            self._invariant_check_task = None

        # Clear recovery marker (graceful shutdown)
        self._crash_recovery._recovery_tracker.clear_recovery_marker()

        self._running = False
        logger.info("Ingestion orchestrator stopped")

    def run_manual_job(self, source_name: str, *, force: bool = False) -> None:
        if source_name not in self._sources:
            raise KeyError(f"Unknown source {source_name}")
        self._enqueue_job(source_name, force=force, trigger="manual")

    def _enqueue_job(self, source_name: str, *, force: bool = False, trigger: str = "schedule") -> None:
        registration = self._sources[source_name]

        # Check both in-memory paused flag and global pause registry
        is_paused = registration.paused or self._paused_sources.is_paused(source_name)

        if is_paused and not force:
            logger.debug(
                "Skipping enqueue because source is paused",
                extra={"ingestion_source": source_name, "ingestion_event": "pause_skip"}
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
        # Track active job for invariant checking
        self._active_jobs_map[job.job_id] = {
            "job_type": job.job_type.value,
            "started_at": datetime.utcnow(),
        }

        # Acquire connector-specific semaphore
        connector_semaphore = self._per_connector_semaphores.get(job.job_type)
        if connector_semaphore:
            await connector_semaphore.acquire()

        try:
            # Start resource monitoring
            self._resource_monitor.start_monitoring(job.job_id, job.job_type)

            # Execute job
            await self._process_job(job)

        finally:
            # Stop resource monitoring and store metrics for telemetry
            metrics = self._resource_monitor.stop_monitoring(job.job_id)
            if metrics:
                self._job_resource_metrics[job.job_id] = metrics

            # Release connector-specific semaphore
            if connector_semaphore:
                connector_semaphore.release()

            # Adaptive concurrency adjustment
            if metrics and self._should_adjust_concurrency():
                await self._adjust_concurrency(job.job_type, metrics)

            # Remove from active jobs map
            if job.job_id in self._active_jobs_map:
                del self._active_jobs_map[job.job_id]

            # Also release global semaphore for backward compatibility
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
            # Capture full traceback for debugging
            error_traceback = traceback.format_exc()
            job.payload.update(
                {
                    "files_processed": files_processed,
                    "bytes_processed": bytes_processed,
                    "error": str(exc),
                    "error_traceback": error_traceback,
                }
            )
            # Check if this is a quarantine retry that failed
            if "from_quarantine" in job.payload:
                quarantine_job_id = job.payload["from_quarantine"]
                self._quarantine.mark_retry_attempted(
                    quarantine_job_id,
                    success=False,
                    error_message=str(exc),
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

                # Include resource metrics if available
                metadata = {}
                resource_metrics = self._job_resource_metrics.get(job.job_id)
                if resource_metrics:
                    metadata.update({
                        "cpu_percent_avg": resource_metrics.cpu_percent_avg,
                        "cpu_percent_peak": resource_metrics.cpu_percent_peak,
                        "memory_mb_avg": resource_metrics.memory_mb_avg,
                        "memory_mb_peak": resource_metrics.memory_mb_peak,
                        "bytes_read": resource_metrics.bytes_read,
                        "bytes_written": resource_metrics.bytes_written,
                    })
                    # Clean up metrics after use
                    del self._job_resource_metrics[job.job_id]

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
                    metadata=metadata if metadata else None,
                )
            job.payload.update(
                {
                    "files_processed": files_processed,
                    "bytes_processed": bytes_processed,
                }
            )
            # Check if this is a quarantine retry that succeeded
            if "from_quarantine" in job.payload:
                quarantine_job_id = job.payload["from_quarantine"]
                self._quarantine.mark_retry_attempted(
                    quarantine_job_id,
                    success=True,
                )
                logger.info(
                    "Quarantine retry succeeded",
                    extra={
                        "ingestion_job_id": job.job_id,
                        "quarantine_job_id": quarantine_job_id,
                    },
                )

            # Clean up retry budget on success
            if job.job_id in self._retry_budgets:
                del self._retry_budgets[job.job_id]

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

    def _ensure_obsidian_connector(self):
        """Lazy-initialize Obsidian connector if needed."""
        if self._obsidian_connector is None:
            try:
                from ..ingestion.obsidian.connector import ObsidianVaultConnector
                from ..ingestion.obsidian.descriptor import VaultRegistry

                self._vault_registry = VaultRegistry()
                self._obsidian_connector = ObsidianVaultConnector(
                    workspace_dir=self._workspace_dir,
                    state_store=self._state_store,
                    vault_registry=self._vault_registry,
                    element_sink=self._element_sink,
                    audit_logger=self._audit_logger,
                    consent_registry=self._consent_registry,
                )
                logger.info("Initialized Obsidian connector")
            except (ImportError, AttributeError) as exc:
                logger.error(f"Failed to initialize Obsidian connector: {exc}")
                raise RuntimeError("Obsidian connector not available") from exc

    def _ensure_imap_connector(self):
        """Lazy-initialize IMAP connector if needed."""
        if self._imap_connector is None:
            try:
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
                    element_sink=self._element_sink,
                    audit_logger=self._audit_logger,
                    consent_registry=self._consent_registry,
                )
                logger.info("Initialized IMAP connector")
            except (ImportError, AttributeError) as exc:
                logger.error(f"Failed to initialize IMAP connector: {exc}")
                raise RuntimeError("IMAP connector not available") from exc

    def _ensure_github_connector(self):
        """Lazy-initialize GitHub connector if needed."""
        if self._github_connector is None:
            try:
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
                    element_sink=self._element_sink,
                    audit_logger=self._audit_logger,
                    consent_registry=self._consent_registry,
                )
                logger.info("Initialized GitHub connector")
            except (ImportError, AttributeError) as exc:
                logger.error(f"Failed to initialize GitHub connector: {exc}")
                raise RuntimeError("GitHub connector not available") from exc

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
        self._ensure_obsidian_connector()
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
        self._ensure_imap_connector()
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
        self._ensure_github_connector()
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
        """Enhanced retry with per-connector policies and failure classification.

        Args:
            job: Failed ingestion job to potentially retry
        """
        # Get or create retry budget
        budget = self._retry_budgets.get(job.job_id)
        if not budget:
            budget = RetryBudget(
                job_id=job.job_id,
                job_type=job.job_type,
                first_attempt_at=datetime.utcnow(),
            )
            self._retry_budgets[job.job_id] = budget

        # Classify failure type for retry strategy selection
        error_message = job.payload.get("error", "")
        error_exception = job.payload.get("error_exception")  # Optional exception instance

        # Use enhanced classification
        quarantine_reason = classify_failure(
            error_message,
            exception=error_exception if error_exception else None,
        )
        budget.failure_type = quarantine_reason_to_failure_type(quarantine_reason)

        # Get connector-specific policy
        policy = self._retry_policies.get_policy(job.job_type)

        # Check if retry allowed
        if not budget.can_retry(policy):
            logger.info(
                "Retry budget exhausted, quarantining job",
                extra={
                    "ingestion_job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "attempts": budget.attempts,
                    "max_attempts": policy.max_attempts,
                    "failure_type": budget.failure_type.value,
                },
            )
            await self._quarantine_job(job)
            # Clean up budget after quarantine
            del self._retry_budgets[job.job_id]
            return

        # Calculate delay with jitter
        delay_seconds = budget.next_delay(policy)
        budget.attempts += 1
        budget.last_attempt_at = datetime.utcnow()
        budget.total_delay_seconds += delay_seconds

        logger.info(
            "Scheduling retry with policy",
            extra={
                "ingestion_job_id": job.job_id,
                "job_type": job.job_type.value,
                "attempt": budget.attempts,
                "max_attempts": policy.max_attempts,
                "delay_seconds": delay_seconds,
                "failure_type": budget.failure_type.value,
                "retry_strategy": policy.strategy.value,
            },
        )

        # Update job payload with retry metadata
        job.payload["attempts"] = budget.attempts
        job.payload["failure_type"] = budget.failure_type.value
        job.payload["trigger"] = "retry"
        job.payload["retry_delay_seconds"] = delay_seconds

        # Reschedule with calculated delay
        self._job_queue.reschedule(job.job_id, delay_seconds)

        # Record telemetry
        if self._telemetry:
            self._telemetry.record(
                job_id=job.job_id,
                duration=0.0,
                status="retry_scheduled",
                metadata={
                    "job_type": job.job_type.value,
                    "attempt": budget.attempts,
                    "delay_seconds": delay_seconds,
                    "failure_type": budget.failure_type.value,
                    "retry_strategy": policy.strategy.value,
                    "connector_type": job.job_type.value,
                },
            )

    async def _quarantine_job(self, job: IngestionJob) -> None:
        """Move failed job to quarantine with classification.

        Args:
            job: The failed ingestion job
        """
        error_message = job.payload.get("error", "Unknown error")
        error_traceback = job.payload.get("error_traceback")

        # Classify the failure
        reason = classify_failure(error_message)

        # Mark job as QUARANTINED in the main queue
        self._job_queue.mark_quarantined(job.job_id)

        # Also add to quarantine store for detailed tracking
        quarantined = self._quarantine.quarantine(
            job=job,
            reason=reason,
            error_message=error_message,
            error_traceback=error_traceback,
            metadata={
                "source_name": job.payload.get("source_name"),
                "attempts": job.payload.get("attempts", 0),
                "job_type": job.job_type.value,
            },
        )

        logger.error(
            "Job quarantined",
            extra={
                "ingestion_job_id": job.job_id,
                "quarantine_reason": reason.value,
                "ingestion_attempts": job.payload.get("attempts", 0),
            },
        )

        # Record audit event
        self._audit_logger.record(
            AuditEvent(
                job_id=job.job_id,
                source=job.payload.get("source_name", "unknown"),
                action="quarantine",
                status="quarantined",
                timestamp=datetime.utcnow(),
                metadata={
                    "reason": reason.value,
                    "attempts": job.payload.get("attempts", 0),
                },
            )
        )

        # Record telemetry
        if self._telemetry:
            self._telemetry.record(
                job_id=job.job_id,
                duration=0.0,
                status="quarantined",
                metadata={"reason": reason.value},
            )

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

    def _initialize_connector_semaphores(self) -> None:
        """Create per-connector semaphores based on resource profiles."""
        system_resources = self._resource_monitor.get_system_resources()
        available_cpu = system_resources["available_cpu_cores"]
        available_memory_mb = system_resources["available_memory_mb"]
        current_system_load = system_resources["cpu_percent"] / 100.0

        for job_type in JobType:
            optimal = self._resource_profiles.calculate_optimal_concurrency(
                job_type=job_type,
                available_cpu_cores=available_cpu,
                available_memory_mb=available_memory_mb,
                current_system_load=current_system_load,
            )
            self._per_connector_semaphores[job_type] = asyncio.Semaphore(optimal)

            logger.info(
                "Initialized connector semaphore",
                extra={
                    "job_type": job_type.value,
                    "concurrency": optimal,
                },
            )

    def _should_adjust_concurrency(self) -> bool:
        """Check if concurrency should be adjusted.

        Adjusts periodically to avoid thrashing. Currently adjusts every 10th job.

        Returns:
            True if concurrency should be adjusted
        """
        # Simple heuristic: adjust every 10 jobs
        return (self._active_jobs % 10) == 0

    async def _adjust_concurrency(
        self,
        job_type: JobType,
        metrics,
    ) -> None:
        """Dynamically adjust concurrency based on observed resource usage.

        Args:
            job_type: Type of connector job
            metrics: Job resource metrics
        """
        system_resources = self._resource_monitor.get_system_resources()

        # Check if system is under pressure
        system_overloaded = (
            system_resources["cpu_percent"] > 80
            or system_resources["memory_percent"] > 85
        )

        connector_semaphore = self._per_connector_semaphores.get(job_type)
        if not connector_semaphore:
            return

        # Get current limit (approximation - semaphores don't expose this cleanly)
        # We track it indirectly through the profile
        profile = self._resource_profiles.get_profile(job_type)
        current_limit = profile.max_concurrent_jobs or 2

        if system_overloaded and current_limit > 1:
            # Log that we detected overload (actual adjustment requires semaphore recreation)
            logger.info(
                "System overloaded, would reduce connector concurrency",
                extra={
                    "job_type": job_type.value,
                    "cpu_percent": system_resources["cpu_percent"],
                    "memory_percent": system_resources["memory_percent"],
                    "current_limit": current_limit,
                },
            )
            # Note: Actual semaphore limit adjustment requires recreation
            # This is logged for monitoring purposes
        elif not system_overloaded and current_limit < 8:
            # Could increase concurrency
            if profile.adaptive_concurrency:
                logger.info(
                    "System resources available, could increase connector concurrency",
                    extra={
                        "job_type": job_type.value,
                        "cpu_percent": system_resources["cpu_percent"],
                        "memory_percent": system_resources["memory_percent"],
                        "current_limit": current_limit,
                    },
                )

    async def _resource_sampling_loop(self) -> None:
        """Periodically sample resource usage for active jobs.

        Runs in the background while the orchestrator is active.
        """
        try:
            while self._running:
                self._resource_monitor.sample_active_jobs()
                await asyncio.sleep(1.0)  # Sample every second
        except asyncio.CancelledError:
            logger.debug("Resource sampling loop cancelled")
            raise

    async def _deadlock_detection_loop(self) -> None:
        """Periodically detect and recover stalled jobs.

        Runs in the background while the orchestrator is active,
        checking for jobs stuck in RUNNING state beyond the timeout.
        """
        try:
            while self._running:
                # Check for stalled jobs
                stalled_jobs = self._deadlock_detector.detect_stalled_jobs()

                # Recover each stalled job
                for job_id in stalled_jobs:
                    try:
                        self._deadlock_detector.recover_stalled_job(job_id)
                        logger.info(
                            "Recovered stalled job",
                            extra={"job_id": job_id},
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to recover stalled job",
                            extra={"job_id": job_id, "error": str(exc)},
                        )

                # Check every 60 seconds
                await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            logger.debug("Deadlock detection loop cancelled")
            raise

    async def _invariant_check_loop(self) -> None:
        """Periodically check state machine invariants.

        Runs in the background while the orchestrator is active,
        verifying that state machine consistency is maintained.
        """
        try:
            while self._running:
                # Check invariants via the job queue's validator
                violations = self._job_queue._validator.check_invariants(self)

                # Log any violations
                if violations:
                    logger.error(
                        "State machine invariant violations detected",
                        extra={
                            "violation_count": len(violations),
                            "violations": violations,
                        },
                    )
                else:
                    logger.debug("State machine invariants check passed")

                # Check every 5 minutes
                await asyncio.sleep(300.0)
        except asyncio.CancelledError:
            logger.debug("Invariant check loop cancelled")
            raise

    def check_invariants(self) -> List[str]:
        """Manually trigger state machine invariant checks.

        Returns:
            List of invariant violation descriptions (empty if all pass)
        """
        return self._job_queue._validator.check_invariants(self)


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



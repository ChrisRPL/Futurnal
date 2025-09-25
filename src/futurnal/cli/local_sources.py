def _load_state_store(workspace: Path) -> StateStore:
    state_dir = workspace / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return StateStore(state_dir / "state.db")


def _load_connector(workspace: Path, state_store: StateStore) -> LocalFilesConnector:
    return LocalFilesConnector(workspace_dir=workspace, state_store=state_store)

def _resolve_workspace(path: Optional[Path]) -> Path:
    return path or DEFAULT_WORKSPACE_PATH


def _load_quarantine_dir(workspace: Path) -> Path:
    return workspace / QUARANTINE_DIR_NAME


def _load_archive_dir(workspace: Path) -> Path:
    return workspace / QUARANTINE_ARCHIVE_DIR_NAME


def _resolve_source(config: dict, name: str) -> Dict:
    for source in config.get("sources", []):
        if source.get("name") == name:
            return source
    raise KeyError(f"Source '{name}' not found in configuration")


def _load_source(config_path: Path, source_name: str) -> LocalIngestionSource:
    config = _load_config(config_path)
    load_config_from_dict(config)
    source_dict = _resolve_source(config, source_name)
    return LocalIngestionSource(**source_dict)


def _quarantine_summary_path(workspace: Path) -> Path:
    return workspace / TELEMETRY_DIR_NAME / QUARANTINE_SUMMARY_FILENAME


def _audit_logger(workspace: Path) -> AuditLogger:
    return AuditLogger(workspace / AUDIT_DIR_NAME)


def _load_consent_registry(workspace: Path) -> ConsentRegistry:
    return ConsentRegistry(workspace / CONSENT_DIR_NAME)


def _load_job_queue(workspace: Path) -> JobQueue:
    queue_dir = workspace / QUEUE_DIR_NAME
    queue_dir.mkdir(parents=True, exist_ok=True)
    return JobQueue(queue_dir / QUEUE_DB_FILE)


def _load_sources(config_path: Path) -> Dict[str, LocalIngestionSource]:
    config = _load_config(config_path)
    valid = load_config_from_dict(config)
    return {source.name: source for source in valid.root}


def _save_sources(config_path: Path, sources: Iterable[LocalIngestionSource]) -> None:
    serialized = []
    for source in sorted(sources, key=lambda item: item.name):
        payload = source.model_dump(mode="json")
        payload["root_path"] = str(source.root_path)
        if payload.get("ignore_file") is None:
            payload["ignore_file"] = None
        serialized.append(payload)
    _save_config(config_path, {"sources": serialized})


def _update_source(
    config_path: Path,
    name: str,
    *,
    mutate: Callable[[LocalIngestionSource], LocalIngestionSource],
) -> LocalIngestionSource:
    sources = _load_sources(config_path)
    try:
        existing = sources[name]
    except KeyError as exc:
        raise typer.BadParameter(f"Source '{name}' not found") from exc
    updated = mutate(existing)
    sources[name] = updated
    _save_sources(config_path, sources.values())
    return updated
"""CLI utilities for managing local ingestion sources."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import typer
from croniter import croniter

from ..configuration.settings import (
    DEFAULT_CONFIG_PATH as SETTINGS_CONFIG_PATH,
    bootstrap_settings,
    load_settings,
)
from ..ingestion.local.config import LocalIngestionSource, load_config_from_dict
from ..ingestion.local.connector import LocalFilesConnector
from ..ingestion.local.quarantine import (
    QUARANTINE_SUMMARY_FILENAME,
    MAX_RETRY_ATTEMPTS,
    QuarantineEntry,
    archive_entry,
    append_note,
    iter_entries,
    remove_entry,
    update_entry,
    write_summary,
)
from ..ingestion.local.state import StateStore
from ..orchestrator.models import IngestionJob, JobPriority, JobType
from ..orchestrator.queue import JobQueue, JobStatus
from ..privacy import AuditLogger, ConsentRegistry, ConsentRequiredError, build_policy, redact_path
from ..privacy.audit import AuditEvent

app = typer.Typer(help="Manage Futurnal local data sources")

DEFAULT_CONFIG_PATH = Path.home() / ".futurnal" / "sources.json"
DEFAULT_WORKSPACE_PATH = Path.home() / ".futurnal" / "workspace"
TELEMETRY_DIR_NAME = "telemetry"
TELEMETRY_LOG_FILE = "telemetry.log"
TELEMETRY_SUMMARY_FILE = "telemetry_summary.json"
AUDIT_DIR_NAME = "audit"
AUDIT_LOG_FILE = "audit.log"
QUARANTINE_DIR_NAME = "quarantine"
QUARANTINE_ARCHIVE_DIR_NAME = "quarantine_archive"
QUEUE_DIR_NAME = "queue"
QUEUE_DB_FILE = "queue.db"
MAX_RETRIES = MAX_RETRY_ATTEMPTS
DISMISS_NOTE_KEY = "dismissed_by"
CONSENT_DIR_NAME = "privacy"
CONSENT_SCOPE_LOCAL_EXTERNAL = "local.external_processing"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {"sources": []}
    return json.loads(path.read_text())


def _save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


@app.command("register")
def register_local_source(
    name: str = typer.Option(..., help="Unique name for the source"),
    root: Path = typer.Option(..., exists=True, file_okay=False, help="Root directory to ingest"),
    include: Optional[str] = typer.Option(None, help="Comma-separated glob include patterns"),
    exclude: Optional[str] = typer.Option(None, help="Comma-separated glob exclude patterns"),
    ignore_file: Optional[Path] = typer.Option(None, help="Custom ignore file path"),
    follow_symlinks: bool = typer.Option(False, help="Follow symlinks when scanning"),
    max_workers: Optional[int] = typer.Option(
        None,
        min=1,
        max=32,
        help="Upper bound on concurrent ingestion workers for this source",
    ),
    max_files_per_batch: Optional[int] = typer.Option(
        None,
        min=1,
        max=1000,
        help="Maximum number of files an ingestion worker processes per batch",
    ),
    scan_interval_seconds: Optional[float] = typer.Option(
        None,
        min=0.1,
        max=3600.0,
        help="Fallback scan interval in seconds when watcher is unavailable",
    ),
    watcher_debounce_seconds: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=120.0,
        help="Debounce interval in seconds between watcher-triggered job enqueues",
    ),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Register or update a local directory source."""

    config = _load_config(config_path)
    include_list = [pattern.strip() for pattern in include.split(",") if pattern.strip()] if include else []
    exclude_list = [pattern.strip() for pattern in exclude.split(",") if pattern.strip()] if exclude else []

    source_dict = {
        "name": name,
        "root_path": str(root),
        "include": include_list,
        "exclude": exclude_list,
        "follow_symlinks": follow_symlinks,
        "ignore_file": str(ignore_file) if ignore_file else None,
        "max_workers": max_workers,
        "max_files_per_batch": max_files_per_batch,
        "scan_interval_seconds": scan_interval_seconds,
        "watcher_debounce_seconds": watcher_debounce_seconds,
    }

    new_source = LocalIngestionSource(**source_dict)
    sources = _load_sources(config_path)
    sources[name] = new_source
    _save_sources(config_path, sources.values())
    typer.echo(f"Registered source '{name}' at {root}")


@app.command("list")
def list_sources(config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config")) -> None:
    """List configured local sources."""

    sources = _load_sources(config_path)
    if not sources:
        typer.echo("No sources configured")
        return
    for source in sources.values():
        typer.echo(f"- {source.name}: {source.root_path}")


@app.command("telemetry")
def show_telemetry(
    workspace_path: Optional[Path] = typer.Option(
        None,
        help="Path to ingestion workspace",
    ),
    raw: bool = typer.Option(False, "--raw", help="Emit raw JSON instead of formatted output"),
) -> None:
    """Display ingestion telemetry summary."""

    workspace = workspace_path or DEFAULT_WORKSPACE_PATH
    summary_path = workspace / TELEMETRY_DIR_NAME / TELEMETRY_SUMMARY_FILE
    if not summary_path.exists():
        typer.echo(f"Telemetry summary not found at {summary_path}")
        raise typer.Exit(code=1)

    summary = json.loads(summary_path.read_text())
    if raw:
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.echo("Telemetry Summary")
    typer.echo(f"Total jobs: {summary.get('total', 0)}")
    typer.echo(f"Succeeded: {summary.get('succeeded', 0)}")
    typer.echo(f"Failed: {summary.get('failed', 0)}")
    avg = summary.get("avg_duration", {})
    for status, duration in avg.items():
        typer.echo(f"Average duration ({status}): {duration:.2f}s")


schedule_app = typer.Typer(help="Manage ingestion schedules")
app.add_typer(schedule_app, name="schedule")


def _describe_schedule(source: LocalIngestionSource) -> str:
    if source.schedule == "@manual":
        return "manual"
    if source.schedule == "@interval" and source.interval_seconds:
        return f"every {int(source.interval_seconds)}s"
    return source.schedule


def _next_run(source: LocalIngestionSource) -> Optional[datetime]:
    now = datetime.utcnow()
    if source.paused:
        return None
    if source.schedule == "@manual":
        return None
    if source.schedule == "@interval" and source.interval_seconds:
        return now + timedelta(seconds=source.interval_seconds)
    if not source.schedule:
        return None
    try:
        iterator = croniter(source.schedule, now)
        return iterator.get_next(datetime)
    except (ValueError, TypeError):
        return None


@schedule_app.command("list")
def schedule_list(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    show_paused: bool = typer.Option(True, help="Include paused sources"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text"),
) -> None:
    sources = _load_sources(config_path)
    output = []
    for source in sources.values():
        if not show_paused and source.paused:
            continue
        entry: Dict[str, object] = {
            "name": source.name,
            "schedule": _describe_schedule(source),
            "priority": source.priority,
            "paused": source.paused,
        }
        next_run = _next_run(source)
        if next_run:
            entry["next_run"] = next_run.isoformat()
        output.append(entry)

    if json_output:
        typer.echo(json.dumps(output, indent=2))
        return

    if not output:
        typer.echo("No schedules configured")
        return

    typer.echo("Source schedules")
    for entry in output:
        line = f"- {entry['name']}: {entry['schedule']} (priority={entry['priority']})"
        if entry.get("next_run"):
            line += f" next={entry['next_run']}"
        if entry["paused"]:
            line += " [paused]"
        typer.echo(line)


def _validate_schedule_args(cron: Optional[str], interval: Optional[float], manual: bool) -> None:
    if manual:
        return
    if interval is not None:
        if interval <= 0:
            raise typer.BadParameter("Interval must be positive")
        LocalIngestionSource(
            name="__interval__",
            root_path=Path.cwd(),
            schedule="@interval",
            interval_seconds=interval,
        )
        return
    if cron:
        if not croniter.is_valid(cron):
            raise typer.BadParameter("Invalid cron expression")
        return
    raise typer.BadParameter("Provide --cron, --interval, or --manual")


@schedule_app.command("update")
def schedule_update(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    cron: Optional[str] = typer.Option(None, help="Cron expression"),
    interval: Optional[float] = typer.Option(None, help="Interval in seconds"),
    manual: bool = typer.Option(False, help="Switch to manual-only"),
) -> None:
    _validate_schedule_args(cron, interval, manual)

    def mutate(source: LocalIngestionSource) -> LocalIngestionSource:
        payload = source.model_copy()
        if manual:
            payload.schedule = "@manual"
            payload.interval_seconds = None
        elif interval is not None:
            payload.schedule = "@interval"
            payload.interval_seconds = interval
        elif cron:
            payload.schedule = cron
            payload.interval_seconds = None
        return payload

    updated = _update_source(config_path, source_name, mutate=mutate)
    next_run = _next_run(updated)
    next_repr = next_run.isoformat() if next_run else "manual"
    typer.echo(f"Updated schedule for {updated.name}; next run {next_repr}")


@schedule_app.command("remove")
def schedule_remove(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    updated = _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"schedule": "@manual", "interval_seconds": None}),
    )
    typer.echo(f"Removed schedule for {updated.name}; source is now manual-only")


@app.command("priority")
def set_priority(
    source_name: str = typer.Argument(..., help="Name of the source"),
    level: str = typer.Option("normal", help="Priority level: low|normal|high"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    allowed = {"low", "normal", "high"}
    normalized = level.lower()
    if normalized not in allowed:
        raise typer.BadParameter("Priority must be one of: low, normal, high")

    _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"priority": normalized}),
    )
    typer.echo(f"Set priority for {source_name} to {normalized}")


def _record_operator_event(
    workspace: Path,
    *,
    source: str,
    action: str,
    status: str,
    operator: Optional[str],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    logger = _audit_logger(workspace)
    event = AuditEvent(
        job_id=str(uuid.uuid4()),
        source=source,
        action=action,
        status=status,
        timestamp=datetime.utcnow(),
        operator_action=operator,
        metadata=metadata or {},
    )
    logger.record(event)


@app.command("pause")
def pause_source(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Workspace path for audit logging"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    updated = _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"paused": True}),
    )
    workspace = _resolve_workspace(workspace_path)
    _record_operator_event(
        workspace,
        source=updated.name,
        action="scheduler",
        status="paused",
        operator=operator,
        metadata={"schedule": _describe_schedule(updated)},
    )
    typer.echo(f"Paused automatic ingestion for {updated.name}")


@app.command("resume")
def resume_source(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Workspace path for audit logging"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    updated = _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"paused": False}),
    )
    workspace = _resolve_workspace(workspace_path)
    _record_operator_event(
        workspace,
        source=updated.name,
        action="scheduler",
        status="resumed",
        operator=operator,
        metadata={"schedule": _describe_schedule(updated)},
    )
    typer.echo(f"Resumed automatic ingestion for {updated.name}")


@app.command("run")
def trigger_manual_run(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    force: bool = typer.Option(False, help="Allow manual run even when paused"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    sources = _load_sources(config_path)
    if source_name not in sources:
        raise typer.BadParameter(f"Source '{source_name}' not found")

    source = sources[source_name]
    if source.paused and not force:
        typer.echo("Source is paused; use --force to run anyway")
        raise typer.Exit(code=1)

    queue = _load_job_queue(workspace)
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
    }
    job = IngestionJob(
        job_id=str(uuid.uuid4()),
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": source.name, "trigger": "manual"},
        priority=priority_map[source.priority],
        scheduled_for=datetime.utcnow(),
    )
    queue.enqueue(job)
    typer.echo(f"Enqueued manual ingestion job {job.job_id} for {source.name}")


queue_app = typer.Typer(help="Inspect the job queue")
app.add_typer(queue_app, name="queue")


@queue_app.command("status")
def queue_status(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    limit: int = typer.Option(50, help="Maximum entries to display"),
    status: Optional[JobStatus] = typer.Option(None, help="Filter by job status"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    queue = _load_job_queue(workspace)
    entries = queue.snapshot(status=status, limit=limit)

    if json_output:
        typer.echo(json.dumps(entries, indent=2, default=str))
        return

    if not entries:
        typer.echo("Queue is empty")
        return

    typer.echo("Job queue")
    for entry in entries:
        payload = entry.get("payload", {})
        line = (
            f"- {entry['job_id']} :: {entry['status']} :: source={payload.get('source_name')}"
            f" attempts={entry['attempts']} priority={entry['priority']}"
        )
        if entry.get("scheduled_for"):
            line += f" scheduled_for={entry['scheduled_for']}"
        if entry.get("updated_at"):
            line += f" updated_at={entry['updated_at']}"
        if payload.get("error"):
            line += f" error={payload['error']}"
        typer.echo(line)


@app.command("audit")
def show_audit(
    workspace_path: Optional[Path] = typer.Option(
        None,
        help="Path to ingestion workspace",
    ),
    tail: int = typer.Option(20, min=1, help="Number of recent audit events to display"),
    verify: bool = typer.Option(False, help="Verify log chain integrity before printing"),
) -> None:
    """Print recent audit log entries."""

    workspace = workspace_path or DEFAULT_WORKSPACE_PATH
    audit_path = workspace / AUDIT_DIR_NAME / AUDIT_LOG_FILE
    if not audit_path.exists():
        typer.echo(f"Audit log not found at {audit_path}")
        raise typer.Exit(code=1)

    logger = _audit_logger(workspace)
    if verify and not logger.verify(path=audit_path):
        typer.echo("Audit log integrity check failed")
        raise typer.Exit(code=2)

    lines = audit_path.read_text().splitlines()
    for line in lines[-tail:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            typer.echo(line)
            continue
        typer.echo(
            f"[{payload.get('timestamp')}] {payload.get('source')} {payload.get('action')} {payload.get('status')}"
        )
        redacted_path = payload.get("redacted_path")
        if redacted_path:
            typer.echo(f"  path: {redacted_path} (hash={payload.get('path_hash')})")
        if payload.get("metadata"):
            typer.echo(f"  metadata: {json.dumps(payload['metadata'])}")


@app.command("remove")
def remove_source(
    name: str = typer.Argument(..., help="Name of the source to remove"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Remove a configured local source."""

    sources = _load_sources(config_path)
    if name not in sources:
        typer.echo(f"Source '{name}' not found")
        raise typer.Exit(code=1)
    sources.pop(name)
    _save_sources(config_path, sources.values())
    typer.echo(f"Removed source '{name}'")


quarantine_app = typer.Typer(help="Inspect and manage quarantined files")
app.add_typer(quarantine_app, name="quarantine")


@quarantine_app.command("list")
def quarantine_list(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    reason: Optional[str] = typer.Option(None, help="Filter by reason"),
    source: Optional[str] = typer.Option(None, help="Filter by source name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of entries"),
    summary: bool = typer.Option(False, help="Print aggregated summary"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    entries = list(iter_entries(_load_quarantine_dir(workspace)))

    if reason:
        entries = [entry for entry in entries if entry.reason == reason]
    if source:
        entries = [entry for entry in entries if entry.source == source]
    if limit is not None:
        entries = entries[:limit]

    if summary:
        counts = Counter(entry.reason for entry in entries)
        typer.echo("Quarantine Summary")
        typer.echo(f"Total entries: {len(entries)}")
        for reason_name, count in counts.items():
            typer.echo(f"  {reason_name}: {count}")
        return

    if not entries:
        typer.echo("No quarantine entries found")
        return

    for entry in entries:
        timestamp = entry.timestamp.isoformat() if entry.timestamp else "unknown"
        typer.echo(
            f"- {entry.identifier}: {entry.reason}"
            f" (source={entry.source}, retries={entry.retry_count}, timestamp={timestamp})"
        )
        if entry.redacted_path:
            typer.echo(f"    path: {entry.redacted_path} (hash={entry.path_hash})")


@quarantine_app.command("inspect")
def quarantine_inspect(
    entry_id: str,
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    raw: bool = typer.Option(False, help="Print raw JSON payload"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    entry_path = (_load_quarantine_dir(workspace) / f"{entry_id}.json")
    if not entry_path.exists():
        typer.echo(f"Quarantine entry {entry_id} not found")
        raise typer.Exit(code=1)

    entry = QuarantineEntry.from_file(entry_path)
    if raw:
        typer.echo(json.dumps(entry.to_dict(), indent=2))
        return

    typer.echo(f"id: {entry.identifier}")
    typer.echo(f"source: {entry.source}")
    typer.echo(f"reason: {entry.reason}")
    typer.echo(f"detail: {entry.detail}")
    typer.echo(f"timestamp: {entry.timestamp}")
    typer.echo(f"retries: {entry.retry_count}")
    typer.echo(f"redacted_path: {entry.redacted_path}")
    typer.echo(f"path_hash: {entry.path_hash}")


@quarantine_app.command("retry")
def quarantine_retry(
    entry_id: str,
    source_name: str = typer.Option(..., help="Name of the registered source"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    dry_run: bool = typer.Option(False, help="Preview retry without executing"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    quarantine_dir = _load_quarantine_dir(workspace)
    entry_path = quarantine_dir / f"{entry_id}.json"

    if not entry_path.exists():
        typer.echo(f"Quarantine entry {entry_id} not found")
        raise typer.Exit(code=1)

    entry = update_entry(entry_path)
    if entry.retry_count >= MAX_RETRIES:
        typer.echo(f"Entry {entry_id} has reached retry limit ({MAX_RETRIES})")
        raise typer.Exit(code=1)

    source = _load_source(config_path, source_name)

    if dry_run:
        redacted = entry.redacted_path or redact_path(entry.path).redacted
        typer.echo(f"Dry run: would retry {redacted} for source {source.name}")
        return

    state_store = _load_state_store(workspace)
    try:
        connector = _load_connector(workspace, state_store)
        records = list(connector.ingest(source))
    finally:
        state_store.close()

    update_entry(
        entry_path,
        retry_count=entry.retry_count + 1,
        last_retry_at=datetime.utcnow().isoformat(),
    )

    if records:
        redacted = entry.redacted_path or redact_path(entry.path).redacted
        typer.echo(f"Retried {redacted}; {len(records)} elements ingested and entry cleared")
    else:
        redacted = entry.redacted_path or redact_path(entry.path).redacted
        typer.echo(f"Retried {redacted}; no elements produced. Entry remains for further action")


@quarantine_app.command("dismiss")
def quarantine_dismiss(
    entry_id: str,
    note: Optional[str] = typer.Option(None, help="Reason for dismissal"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    quarantine_dir = _load_quarantine_dir(workspace)
    entry_path = quarantine_dir / f"{entry_id}.json"

    if not entry_path.exists():
        typer.echo(f"Quarantine entry {entry_id} not found")
        raise typer.Exit(code=1)

    note_parts = [note for note in [note, operator] if note]
    if note_parts:
        append_note(entry_path, "; ".join(note_parts))

    archive_entry(entry_path, _load_archive_dir(workspace))
    typer.echo(f"Dismissed and archived entry {entry_id}")


@quarantine_app.command("summary")
def quarantine_summary(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    summary = write_summary(_load_quarantine_dir(workspace), workspace / TELEMETRY_DIR_NAME)
    typer.echo(json.dumps(summary, indent=2))


consent_app = typer.Typer(help="Manage connector consent decisions")
app.add_typer(consent_app, name="consent")


@consent_app.command("grant")
def consent_grant(
    source_name: str = typer.Argument(..., help="Name of the registered source"),
    scope: str = typer.Option(CONSENT_SCOPE_LOCAL_EXTERNAL, help="Consent scope"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    duration_hours: Optional[int] = typer.Option(None, help="Optional duration for consent"),
    token: Optional[str] = typer.Option(None, help="Optional consent token"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    registry = _load_consent_registry(workspace)
    record = registry.grant(
        source=source_name,
        scope=scope,
        duration_hours=duration_hours,
        token=token,
        operator=operator,
    )
    audit = _audit_logger(workspace)
    audit.record_consent_event(
        job_id=str(uuid.uuid4()),
        source=source_name,
        scope=scope,
        granted=True,
        operator=operator,
        token_hash=record.token_hash,
    )
    typer.echo(f"Granted consent for {source_name}:{scope}")


@consent_app.command("revoke")
def consent_revoke(
    source_name: str = typer.Argument(..., help="Name of the registered source"),
    scope: str = typer.Option(CONSENT_SCOPE_LOCAL_EXTERNAL, help="Consent scope"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    registry = _load_consent_registry(workspace)
    record = registry.revoke(source=source_name, scope=scope, operator=operator)
    audit = _audit_logger(workspace)
    audit.record_consent_event(
        job_id=str(uuid.uuid4()),
        source=source_name,
        scope=scope,
        granted=False,
        operator=operator,
        token_hash=record.token_hash,
    )
    typer.echo(f"Revoked consent for {source_name}:{scope}")


@consent_app.command("status")
def consent_status(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    active_only: bool = typer.Option(True, help="Show only active consents"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    registry = _load_consent_registry(workspace)
    records = list(registry.iter_active()) if active_only else list(registry.snapshot())
    if not records:
        typer.echo("No consent records found")
        return
    for record in records:
        typer.echo(
            f"- {record.source}:{record.scope} :: granted={record.granted}"
            f" expires={record.expires_at} operator={record.operator}"
        )


if __name__ == "__main__":
    app()



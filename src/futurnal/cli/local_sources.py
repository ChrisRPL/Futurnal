def _load_state_store(workspace: Path) -> StateStore:
    state_path = workspace / "state" / "state.db"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    return StateStore(state_path)


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
"""CLI utilities for managing local ingestion sources."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import typer

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
MAX_RETRIES = MAX_RETRY_ATTEMPTS
DISMISS_NOTE_KEY = "dismissed_by"


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
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Register or update a local directory source."""

    config = _load_config(config_path)
    include_list = include.split(",") if include else []
    exclude_list = exclude.split(",") if exclude else []

    source_dict = {
        "name": name,
        "root_path": str(root),
        "include": include_list,
        "exclude": exclude_list,
        "follow_symlinks": follow_symlinks,
        "ignore_file": str(ignore_file) if ignore_file else None,
    }

    # Validate source configuration
    LocalIngestionSource(**source_dict)

    sources = [src for src in config.get("sources", []) if src.get("name") != name]
    sources.append(source_dict)
    _save_config(config_path, {"sources": sources})
    typer.echo(f"Registered source '{name}' at {root}")


@app.command("list")
def list_sources(config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config")) -> None:
    """List configured local sources."""

    config = _load_config(config_path)
    load_config_from_dict(config)  # Validate schema
    for source in config.get("sources", []):
        typer.echo(f"- {source['name']}: {source['root_path']}")


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


@app.command("audit")
def show_audit(
    workspace_path: Optional[Path] = typer.Option(
        None,
        help="Path to ingestion workspace",
    ),
    tail: int = typer.Option(20, min=1, help="Number of recent audit events to display"),
) -> None:
    """Print recent audit log entries."""

    workspace = workspace_path or DEFAULT_WORKSPACE_PATH
    audit_path = workspace / AUDIT_DIR_NAME / AUDIT_LOG_FILE
    if not audit_path.exists():
        typer.echo(f"Audit log not found at {audit_path}")
        raise typer.Exit(code=1)

    lines = audit_path.read_text().splitlines()
    for line in lines[-tail:]:
        typer.echo(line)


@app.command("remove")
def remove_source(
    name: str = typer.Argument(..., help="Name of the source to remove"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Remove a configured local source."""

    config = _load_config(config_path)
    sources = [src for src in config.get("sources", []) if src.get("name") != name]
    if len(sources) == len(config.get("sources", [])):
        typer.echo(f"Source '{name}' not found")
        raise typer.Exit(code=1)
    _save_config(config_path, {"sources": sources})
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
            f"- {entry.identifier}: {entry.reason} (source={entry.source}, retries={entry.retry_count}, timestamp={timestamp})"
        )


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

    entry = json.loads(entry_path.read_text())
    if raw:
        typer.echo(json.dumps(entry, indent=2))
        return

    for key, value in entry.items():
        typer.echo(f"{key}: {value}")


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
        typer.echo(f"Dry run: would retry {entry.path} for source {source.name}")
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
        remove_entry(entry_path)
        typer.echo(f"Retried {entry.path}; {len(records)} elements ingested and entry cleared")
    else:
        typer.echo(f"Retried {entry.path}; no elements produced. Entry remains for further action")


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


if __name__ == "__main__":
    app()



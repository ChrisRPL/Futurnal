"""Tests for local source CLI utilities."""

from pathlib import Path

import json
import pytest
from typer.testing import CliRunner

from futurnal.cli.local_sources import (
    AUDIT_DIR_NAME,
    AUDIT_LOG_FILE,
    QUARANTINE_ARCHIVE_DIR_NAME,
    QUARANTINE_DIR_NAME,
    TELEMETRY_DIR_NAME,
    TELEMETRY_SUMMARY_FILE,
    app,
)
from futurnal.orchestrator.queue import JobQueue
from tests.fixtures.local_connector import QuarantinePayloadBuilder

typer = pytest.importorskip("typer")

runner = CliRunner()


def read_config(path: Path) -> dict:
    return json.loads(path.read_text())


def find_source(config: dict, name: str) -> dict:
    for source in config.get("sources", []):
        if source.get("name") == name:
            return source
    raise AssertionError(f"Source {name} not found")


def test_register_and_list(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "sources.json"
    root = tmp_path / "vault"
    root.mkdir()

    result = runner.invoke(
        app,
        [
            "register",
            "--name",
            "docs",
            "--root",
            str(root),
            "--include",
            "**/*.md",
            "--max-workers",
            "3",
            "--max-files-per-batch",
            "25",
            "--scan-interval-seconds",
            "30",
            "--watcher-debounce-seconds",
            "0.5",
            "--config-path",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    config = read_config(config_path)
    source = find_source(config, "docs")
    assert source["name"] == "docs"
    assert source["max_workers"] == 3
    assert source["max_files_per_batch"] == 25
    assert source["scan_interval_seconds"] == 30.0
    assert source["watcher_debounce_seconds"] == 0.5
    assert source["schedule"] == "@manual"
    assert source["priority"] == "normal"
    assert source["paused"] is False

    list_result = runner.invoke(app, ["list", "--config-path", str(config_path)])
    assert "docs" in list_result.output


def test_remove_source(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.json"
    root = tmp_path / "data"
    root.mkdir()

    runner.invoke(
        app,
        [
            "register",
            "--name",
            "data",
            "--root",
            str(root),
            "--config-path",
            str(config_path),
        ],
    )
    result = runner.invoke(app, ["remove", "data", "--config-path", str(config_path)])
    assert result.exit_code == 0
    config = read_config(config_path)
    assert config["sources"] == []


def test_schedule_update_and_list(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.json"
    root = tmp_path / "data"
    root.mkdir()

    runner.invoke(
        app,
        [
            "register",
            "--name",
            "data",
            "--root",
            str(root),
            "--config-path",
            str(config_path),
        ],
    )

    update_result = runner.invoke(
        app,
        [
            "schedule",
            "update",
            "data",
            "--cron",
            "*/10 * * * *",
            "--config-path",
            str(config_path),
        ],
    )
    assert update_result.exit_code == 0

    config = read_config(config_path)
    source = find_source(config, "data")
    assert source["schedule"] == "*/10 * * * *"
    assert source["interval_seconds"] is None

    list_result = runner.invoke(
        app,
        [
            "schedule",
            "list",
            "--json",
            "--config-path",
            str(config_path),
        ],
    )
    assert list_result.exit_code == 0
    payload = json.loads(list_result.output)
    assert payload[0]["schedule"] == "*/10 * * * *"
    assert payload[0]["priority"] == "normal"

    interval_result = runner.invoke(
        app,
        [
            "schedule",
            "update",
            "data",
            "--interval",
            "600",
            "--config-path",
            str(config_path),
        ],
    )
    assert interval_result.exit_code == 0
    config = read_config(config_path)
    source = find_source(config, "data")
    assert source["schedule"] == "@interval"
    assert source["interval_seconds"] == 600.0

    remove_result = runner.invoke(
        app,
        [
            "schedule",
            "remove",
            "data",
            "--config-path",
            str(config_path),
        ],
    )
    assert remove_result.exit_code == 0
    config = read_config(config_path)
    source = find_source(config, "data")
    assert source["schedule"] == "@manual"


def test_priority_pause_resume_and_run(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.json"
    workspace = tmp_path / "workspace"
    root = tmp_path / "docs"
    root.mkdir()

    runner.invoke(
        app,
        [
            "register",
            "--name",
            "docs",
            "--root",
            str(root),
            "--config-path",
            str(config_path),
        ],
    )

    priority_result = runner.invoke(
        app,
        [
            "priority",
            "docs",
            "--level",
            "high",
            "--config-path",
            str(config_path),
        ],
    )
    assert priority_result.exit_code == 0
    source = find_source(read_config(config_path), "docs")
    assert source["priority"] == "high"

    pause_result = runner.invoke(
        app,
        [
            "pause",
            "docs",
            "--config-path",
            str(config_path),
            "--workspace-path",
            str(workspace),
            "--operator",
            "ops",
        ],
    )
    assert pause_result.exit_code == 0
    source = find_source(read_config(config_path), "docs")
    assert source["paused"] is True
    audit_path = workspace / "audit" / "audit.log"
    assert "paused" in audit_path.read_text()

    resume_result = runner.invoke(
        app,
        [
            "resume",
            "docs",
            "--config-path",
            str(config_path),
            "--workspace-path",
            str(workspace),
            "--operator",
            "ops",
        ],
    )
    assert resume_result.exit_code == 0
    source = find_source(read_config(config_path), "docs")
    assert source["paused"] is False
    assert "resumed" in audit_path.read_text()

    run_result = runner.invoke(
        app,
        [
            "run",
            "docs",
            "--config-path",
            str(config_path),
            "--workspace-path",
            str(workspace),
        ],
    )
    assert run_result.exit_code == 1

    run_force_result = runner.invoke(
        app,
        [
            "run",
            "docs",
            "--config-path",
            str(config_path),
            "--workspace-path",
            str(workspace),
            "--force",
        ],
    )
    assert run_force_result.exit_code == 0
    queue = JobQueue(workspace / "queue" / "queue.db")
    entries = queue.snapshot()
    assert entries
    assert entries[0]["payload"]["trigger"] == "manual"
    queue_result = runner.invoke(
        app,
        [
            "queue",
            "status",
            "--json",
            "--workspace-path",
            str(workspace),
        ],
    )
    assert queue_result.exit_code == 0
    queue_payload = json.loads(queue_result.output)
    assert queue_payload[0]["job_id"] == entries[0]["job_id"]


def test_telemetry_command(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    telemetry_dir = workspace / TELEMETRY_DIR_NAME
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / TELEMETRY_SUMMARY_FILE).write_text(
        json.dumps({"total": 3, "succeeded": 2, "failed": 1, "avg_duration": {"succeeded": 1.2, "failed": 2.5}})
    )

    monkeypatch.setattr("futurnal.cli.local_sources.DEFAULT_WORKSPACE_PATH", workspace)

    result = runner.invoke(app, ["telemetry"])

    assert result.exit_code == 0
    assert "Total jobs" in result.output


def test_audit_command(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    audit_dir = workspace / AUDIT_DIR_NAME
    audit_dir.mkdir(parents=True)
    (audit_dir / AUDIT_LOG_FILE).write_text('{"job_id":"1","status":"succeeded"}\n')

    monkeypatch.setattr("futurnal.cli.local_sources.DEFAULT_WORKSPACE_PATH", workspace)

    result = runner.invoke(app, ["audit", "--tail", "1"])

    assert result.exit_code == 0
    assert "succeeded" in result.output


def _write_quarantine_entry(directory: Path, identifier: str, content: dict) -> Path:
    builder = QuarantinePayloadBuilder(directory)
    return builder.write(
        identifier,
        path=content["path"],
        reason=content["reason"],
        detail=content["detail"],
        source=content.get("source"),
        timestamp=content.get("timestamp", "2024-01-01T00:00:00"),
        retry_count=content.get("retry_count", 0),
        last_retry_at=content.get("last_retry_at"),
        notes=content.get("notes", []),
    )


def test_quarantine_list_and_summary(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    q_dir = workspace / QUARANTINE_DIR_NAME

    _write_quarantine_entry(
        q_dir,
        "entry1",
        {
            "path": "/tmp/data.txt",
            "reason": "partition_error",
            "detail": "boom",
            "timestamp": "2024-01-01T00:00:00",
            "retry_count": 1,
            "last_retry_at": None,
            "notes": [],
            "source": "docs",
        },
    )
    _write_quarantine_entry(
        q_dir,
        "entry2",
        {
            "path": "/tmp/locked.txt",
            "reason": "hash_error",
            "detail": "permission denied",
            "timestamp": "2024-01-02T00:00:00",
            "retry_count": 0,
            "last_retry_at": None,
            "notes": [],
            "source": "docs",
        },
    )

    monkeypatch.setattr("futurnal.cli.local_sources.DEFAULT_WORKSPACE_PATH", workspace)

    result = runner.invoke(app, ["quarantine", "list"])
    assert result.exit_code == 0
    assert "partition_error" in result.output
    assert "hash_error" in result.output
    assert "path:" in result.output

    summary_result = runner.invoke(app, ["quarantine", "list", "--summary"])
    assert summary_result.exit_code == 0
    assert "Total entries" in summary_result.output


def test_quarantine_inspect(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    q_dir = workspace / QUARANTINE_DIR_NAME
    _write_quarantine_entry(
        q_dir,
        "inspect",
        {
            "path": "/tmp/file.txt",
            "reason": "partition_error",
            "detail": "stacktrace",
            "timestamp": "2024-01-01T00:00:00",
            "retry_count": 0,
            "last_retry_at": None,
            "notes": [],
            "source": "docs",
        },
    )

    monkeypatch.setattr("futurnal.cli.local_sources.DEFAULT_WORKSPACE_PATH", workspace)

    result = runner.invoke(app, ["quarantine", "inspect", "inspect"])
    assert result.exit_code == 0
    assert "stacktrace" in result.output
    assert "redacted_path" in result.output


def test_quarantine_dismiss(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    q_dir = workspace / QUARANTINE_DIR_NAME
    archive_dir = workspace / QUARANTINE_ARCHIVE_DIR_NAME
    _write_quarantine_entry(
        q_dir,
        "dismiss",
        {
            "path": "/tmp/file.txt",
            "reason": "partition_error",
            "detail": "stacktrace",
            "timestamp": "2024-01-01T00:00:00",
            "retry_count": 0,
            "last_retry_at": None,
            "notes": [],
            "source": "docs",
        },
    )

    monkeypatch.setattr("futurnal.cli.local_sources.DEFAULT_WORKSPACE_PATH", workspace)

    result = runner.invoke(app, ["quarantine", "dismiss", "dismiss", "--note", "resolved", "--operator", "ops"])
    assert result.exit_code == 0
    assert (archive_dir / "dismiss.json").exists()


def test_quarantine_summary(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    q_dir = workspace / QUARANTINE_DIR_NAME
    _write_quarantine_entry(
        q_dir,
        "summary",
        {
            "path": "/tmp/file.txt",
            "reason": "partition_error",
            "detail": "stacktrace",
            "timestamp": "2024-01-01T00:00:00",
            "retry_count": 0,
            "last_retry_at": None,
            "notes": [],
            "source": "docs",
        },
    )

    monkeypatch.setattr("futurnal.cli.local_sources.DEFAULT_WORKSPACE_PATH", workspace)

    result = runner.invoke(app, ["quarantine", "summary"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 1



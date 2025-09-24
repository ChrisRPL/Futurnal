"""Tests for local source CLI utilities."""

from pathlib import Path

import json
import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner

from tests.fixtures.local_connector import QuarantinePayloadBuilder
from futurnal.cli.local_sources import (
    AUDIT_DIR_NAME,
    AUDIT_LOG_FILE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_WORKSPACE_PATH,
    QUARANTINE_ARCHIVE_DIR_NAME,
    QUARANTINE_DIR_NAME,
    TELEMETRY_DIR_NAME,
    TELEMETRY_LOG_FILE,
    TELEMETRY_SUMMARY_FILE,
    app,
)

runner = CliRunner()


def read_config(path: Path) -> dict:
    return json.loads(path.read_text())


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
            "--config-path",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    config = read_config(config_path)
    assert config["sources"][0]["name"] == "docs"

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



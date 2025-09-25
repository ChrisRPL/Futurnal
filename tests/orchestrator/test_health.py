"""Tests for health checks."""

from __future__ import annotations

from pathlib import Path

from futurnal.configuration.settings import Settings
from futurnal.orchestrator.health import collect_health_report


def _settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "workspace": {
                "workspace_path": str(tmp_path / "workspace"),
                "storage": {
                    "neo4j_uri": "bolt://localhost:7687",
                    "neo4j_username": "neo4j",
                    "neo4j_password": "secret",
                    "chroma_path": str(tmp_path / "workspace" / "vector"),
                },
            }
        }
    )


def test_collect_health_report(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    state_dir = workspace / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "state.db").write_text("ok")
    settings = _settings(tmp_path)

    report = collect_health_report(settings=settings, workspace_path=workspace)
    assert report["status"] in {"ok", "warning"}
    assert report["checks"]


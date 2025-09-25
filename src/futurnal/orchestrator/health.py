"""Health checks for Futurnal workspace services."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from futurnal.configuration.settings import Settings
from futurnal.pipeline.graph import Neo4jPKGWriter
from futurnal.pipeline.vector import ChromaVectorWriter


@dataclass
class HealthCheckResult:
    name: str
    status: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def collect_health_report(
    *,
    settings: Settings,
    workspace_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect health checks for core subsystems."""

    results: List[HealthCheckResult] = []
    workspace = workspace_path or settings.workspace.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)

    results.append(_check_disk_space(workspace))
    results.append(_check_state_store(workspace))
    results.append(_check_neo4j(settings))
    results.append(_check_chroma(settings))

    status = "ok" if all(result.status == "ok" for result in results) else "warning"
    return {
        "status": status,
        "checks": [result.to_dict() for result in results],
    }


def _check_disk_space(workspace: Path) -> HealthCheckResult:
    usage = shutil.disk_usage(workspace)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 5:
        return HealthCheckResult(
            name="disk",
            status="warning",
            detail=f"Low disk space: {free_gb:.2f} GB available",
        )
    return HealthCheckResult(name="disk", status="ok", detail=f"{free_gb:.2f} GB free")


def _check_state_store(workspace: Path) -> HealthCheckResult:
    state_db = workspace / "state" / "state.db"
    if not state_db.exists():
        return HealthCheckResult(
            name="state_store",
            status="warning",
            detail="State database not initialized",
        )
    try:
        conn = sqlite3.connect(state_db)
        conn.execute("SELECT 1")
        conn.close()
        return HealthCheckResult(name="state_store", status="ok", detail="SQLite reachable")
    except sqlite3.Error as exc:
        return HealthCheckResult(name="state_store", status="warning", detail=str(exc))


def _check_neo4j(settings: Settings) -> HealthCheckResult:
    storage = settings.workspace.storage
    try:
        writer = Neo4jPKGWriter(
            uri=storage.neo4j_uri,
            username=storage.neo4j_username,
            password=storage.neo4j_password.get_secret_value(),
            encrypted=storage.neo4j_encrypted,
        )
        writer.close()
        return HealthCheckResult(name="neo4j", status="ok", detail="Connected successfully")
    except Exception as exc:  # noqa: BLE001
        return HealthCheckResult(name="neo4j", status="warning", detail=str(exc))


def _check_chroma(settings: Settings) -> HealthCheckResult:
    storage = settings.workspace.storage
    try:
        writer = ChromaVectorWriter(persist_directory=storage.chroma_path)
        writer.close()
        return HealthCheckResult(name="chroma", status="ok", detail="Collection reachable")
    except Exception as exc:  # noqa: BLE001
        return HealthCheckResult(name="chroma", status="warning", detail=str(exc))


"""Tests for orchestrator scheduling workflow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.models import JobPriority
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.scheduler import IngestionOrchestrator, SourceRegistration
from futurnal.pipeline import NormalizationSink


class MemoryStateStore(StateStore):
    def __init__(self) -> None:
        super().__init__(Path(":memory:"))


@pytest.mark.asyncio()
async def test_manual_job_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = JobQueue(tmp_path / "queue.db")
    state_store = MemoryStateStore()

    class FakePartition:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            self.calls.append(filename)
            return [{"text": "element"}]

    fake_partition = FakePartition()
    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        fake_partition,
    )

    class StubPKGWriter:
        def __init__(self) -> None:
            self.documents = []

        def write_document(self, payload: dict) -> None:
            self.documents.append(payload)

        def remove_document(self, sha256: str) -> None:  # pragma: no cover - unused
            self.documents = [doc for doc in self.documents if doc["sha256"] != sha256]

    class StubVectorWriter:
        def __init__(self) -> None:
            self.embeddings = []

        def write_embedding(self, payload: dict) -> None:
            self.embeddings.append(payload)

        def remove_embedding(self, sha256: str) -> None:  # pragma: no cover - unused
            self.embeddings = [emb for emb in self.embeddings if emb["sha256"] != sha256]

    pkg_writer = StubPKGWriter()
    vector_writer = StubVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)
    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(tmp_path / "workspace"),
        element_sink=sink,
    )

    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_text("hello")
    source = LocalIngestionSource(
        name="notes",
        root_path=root,
        max_workers=2,
        watcher_debounce_seconds=0.2,
        scan_interval_seconds=0.5,
    )

    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )
    orchestrator.run_manual_job("notes", force=True)
    orchestrator.start()

    await asyncio.sleep(0.5)
    await orchestrator.shutdown()

    # Ensure parsed file exists
    assert fake_partition.calls
    assert pkg_writer.documents
    assert vector_writer.embeddings
    telemetry_dir = tmp_path / "workspace" / "telemetry"
    assert telemetry_dir.exists()
    log_file = telemetry_dir / "telemetry.log"
    summary_file = telemetry_dir / "telemetry_summary.json"
    assert log_file.exists()
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text())
    assert summary["overall"]["jobs"] == 1
    assert summary["statuses"]["succeeded"]["files"] >= 1
    audit_dir = tmp_path / "workspace" / "audit"
    audit_file = audit_dir / "audit.log"
    assert audit_file.exists()
    content = audit_file.read_text()
    assert "notes" in content
    assert "succeeded" in content


def test_paused_source_blocks_automatic_jobs(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.db")
    state_store = MemoryStateStore()
    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(tmp_path / "workspace"),
    )

    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_text("hello")
    source = LocalIngestionSource(name="notes", root_path=root)
    registration = SourceRegistration(
        source=source,
        schedule="*/5 * * * *",
        priority=JobPriority.NORMAL,
        paused=True,
    )

    orchestrator.register_source(registration)
    orchestrator._enqueue_job("notes")  # type: ignore[attr-defined]
    assert queue.pending_count() == 0

    orchestrator.run_manual_job("notes")
    assert queue.pending_count() == 0

    orchestrator.run_manual_job("notes", force=True)
    assert queue.pending_count() == 1


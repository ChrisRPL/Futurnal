"""Tests for orchestrator scheduling workflow."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.models import JobPriority
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.scheduler import IngestionOrchestrator, SourceRegistration


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

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(tmp_path / "workspace"),
    )

    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_text("hello")
    source = LocalIngestionSource(name="notes", root_path=root)

    orchestrator.register_source(SourceRegistration(source=source, schedule="@manual"))
    orchestrator.run_manual_job("notes")
    orchestrator.start()

    await asyncio.sleep(0.5)
    await orchestrator.shutdown()

    # Ensure parsed file exists
    parsed_dir = tmp_path / "workspace" / "parsed"
    assert fake_partition.calls



"""Tests for the LocalFilesConnector."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.connector import LocalFilesConnector
from futurnal.ingestion.local.state import StateStore


class DummyStateStore(StateStore):
    """State store using temporary in-memory SQLite for testing."""

    def __init__(self) -> None:
        super().__init__(Path(":memory:"))


class FakePartition:
    def __init__(self, return_value: Iterable[object]) -> None:
        self.return_value = list(return_value)
        self.calls: list[str] = []

    def __call__(self, *, filename: str, strategy: str, include_metadata: bool) -> Iterable[object]:
        self.calls.append(filename)
        return self.return_value


@pytest.fixture()
def connector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LocalFilesConnector:
    state_store = DummyStateStore()
    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        FakePartition(return_value=[{"text": "element"}]),
    )
    return LocalFilesConnector(workspace_dir=tmp_path, state_store=state_store)


def test_crawl_source_detects_new_file(tmp_path: Path, connector: LocalFilesConnector) -> None:
    root = tmp_path / "source"
    root.mkdir()
    file_path = root / "note.md"
    file_path.write_text("hello")

    source = LocalIngestionSource(name="notes", root_path=root)
    records = connector.crawl_source(source)

    assert len(records) == 1
    assert records[0].path == file_path


def test_ingest_yields_partitioned_elements(tmp_path: Path, connector: LocalFilesConnector) -> None:
    root = tmp_path / "source"
    root.mkdir()
    file_path = root / "note.md"
    file_path.write_text("hello")

    source = LocalIngestionSource(name="notes", root_path=root)
    entries = list(connector.ingest(source))

    assert entries
    assert entries[0]["source"] == "notes"
    assert entries[0]["path"] == str(file_path)


def test_crawl_source_skips_unchanged(tmp_path: Path, connector: LocalFilesConnector) -> None:
    root = tmp_path / "source"
    root.mkdir()
    file_path = root / "note.md"
    file_path.write_text("hello")

    source = LocalIngestionSource(name="notes", root_path=root)
    first_records = connector.crawl_source(source)
    assert len(first_records) == 1

    second_records = connector.crawl_source(source)
    assert not second_records



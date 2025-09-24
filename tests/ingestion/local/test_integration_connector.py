"""Integration tests exercising LocalFilesConnector end-to-end behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.connector import LocalFilesConnector
from futurnal.ingestion.local.state import StateStore
from futurnal.pipeline.stubs import NormalizationSink
from tests.fixtures.local_connector import (
    create_concurrent_modification_fixture,
    create_nested_fixture,
    create_permission_locked_fixture,
    create_sparse_large_file_fixture,
    create_symlink_fixture,
)


class RecordingPKGWriter:
    def __init__(self) -> None:
        self.documents_by_sha: Dict[str, dict] = {}
        self.documents_by_path: Dict[str, dict] = {}
        self.removed: list[str] = []

    def write_document(self, payload: dict) -> None:
        self.documents_by_sha[payload["sha256"]] = payload
        self.documents_by_path[payload.get("path", "")] = payload

    def remove_document(self, sha256: str) -> None:
        self.removed.append(sha256)
        self.documents_by_sha.pop(sha256, None)
        for path, payload in list(self.documents_by_path.items()):
            if payload.get("sha256") == sha256:
                self.documents_by_path.pop(path, None)


class RecordingVectorWriter:
    def __init__(self) -> None:
        self.embeddings_by_sha: Dict[str, dict] = {}
        self.removed: list[str] = []

    def write_embedding(self, payload: dict) -> None:
        self.embeddings_by_sha[payload["sha256"]] = payload

    def remove_embedding(self, sha256: str) -> None:
        self.removed.append(sha256)
        self.embeddings_by_sha.pop(sha256, None)


@pytest.fixture(autouse=True)
def stub_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path as _Path

    def _fake_partition(*, filename: str, strategy: str, include_metadata: bool):
        path = _Path(filename)
        return [
            {
                "text": f"stub:{path.name}",
                "metadata": {
                    "filename": path.name,
                    "relative_parent": str(path.parent),
                },
            }
        ]

    monkeypatch.setattr("futurnal.ingestion.local.connector.partition", _fake_partition)


def _make_connector(tmp_path: Path) -> tuple[LocalFilesConnector, RecordingPKGWriter, RecordingVectorWriter, StateStore, Path]:
    workspace = tmp_path / "workspace"
    pkg_writer = RecordingPKGWriter()
    vector_writer = RecordingVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)
    state_store_path = workspace / "state.db"
    state_store = StateStore(state_store_path)
    connector = LocalFilesConnector(
        workspace_dir=workspace,
        state_store=state_store,
        element_sink=sink,
    )
    return connector, pkg_writer, vector_writer, state_store, workspace


def test_nested_fixture_ingestion_synchronizes_pkg_and_vector(tmp_path: Path) -> None:
    fixture = create_nested_fixture(tmp_path)
    connector, pkg_writer, vector_writer, state_store, _workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(name="nested", root_path=fixture.root)
        records = list(connector.ingest(source))

        assert len(records) == len(fixture.files)
        hashes = {record["sha256"] for record in records}
        assert hashes == set(pkg_writer.documents_by_sha.keys())
        assert hashes == set(vector_writer.embeddings_by_sha.keys())
    finally:
        state_store.close()


def test_deletion_propagates_to_pkg_and_vector(tmp_path: Path) -> None:
    fixture = create_nested_fixture(tmp_path)
    connector, pkg_writer, vector_writer, state_store, workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(name="nested", root_path=fixture.root)
        initial_records = list(connector.ingest(source))
        removed_path = initial_records[0]["path"]
        removed_sha = initial_records[0]["sha256"]
        Path(removed_path).unlink()

        rerun = list(connector.ingest(source))
        assert rerun == []
        assert removed_sha in pkg_writer.removed
        assert removed_sha in vector_writer.removed

        quarantine_files = list((workspace / "quarantine").glob("*.json"))
        # No new quarantine entries expected for simple deletion.
        for file in quarantine_files:
            payload = json.loads(file.read_text())
            assert payload["reason"] != "hash_error"
    finally:
        state_store.close()


def test_permission_locked_files_are_quarantined(tmp_path: Path) -> None:
    fixture = create_permission_locked_fixture(tmp_path)
    locked_path = fixture.metadata["locked_file"]
    reset_mode = fixture.metadata["reset_mode"]
    connector, pkg_writer, vector_writer, state_store, workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(name="permissions", root_path=fixture.root)
        records = list(connector.ingest(source))
        # Readable file ingested, locked file skipped.
        assert len(records) == 1
        hashes = {record["sha256"] for record in records}
        assert hashes == set(pkg_writer.documents_by_sha.keys())
        assert hashes == set(vector_writer.embeddings_by_sha.keys())

        quarantine_files = list((workspace / "quarantine").glob("*.json"))
        assert quarantine_files, "Locked file should be quarantined"
        reasons = {json.loads(file.read_text())["reason"] for file in quarantine_files}
        assert "hash_error" in reasons
    finally:
        locked_path.chmod(reset_mode)
        state_store.close()


@pytest.mark.parametrize("follow_symlinks", [False, True])
def test_symlink_fixture_handles_broken_links(tmp_path: Path, follow_symlinks: bool) -> None:
    fixture = create_symlink_fixture(tmp_path)
    connector, pkg_writer, vector_writer, state_store, _workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(
            name="symlinks",
            root_path=fixture.root,
            follow_symlinks=follow_symlinks,
        )
        records = list(connector.ingest(source))

        assert records, "At least the symlink target should be ingested"
        hashes = {record["sha256"] for record in records}
        assert hashes == set(pkg_writer.documents_by_sha.keys())
        assert hashes == set(vector_writer.embeddings_by_sha.keys())
    finally:
        state_store.close()


def test_concurrent_modification_updates_state(tmp_path: Path) -> None:
    fixture = create_concurrent_modification_fixture(tmp_path)
    mutable_path = fixture.metadata["mutable_file"]
    connector, pkg_writer, vector_writer, state_store, _workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(name="concurrency", root_path=fixture.root)
        first_records = list(connector.ingest(source))
        first_sha = first_records[0]["sha256"]

        mutable_path.write_text("Updated content", encoding="utf-8")
        second_records = list(connector.ingest(source))

        assert second_records, "File update should trigger re-ingestion"
        second_sha = second_records[0]["sha256"]
        assert first_sha != second_sha
        assert pkg_writer.documents_by_path[str(mutable_path)]["sha256"] == second_sha
        assert vector_writer.embeddings_by_sha[second_sha]["sha256"] == second_sha
    finally:
        state_store.close()


@pytest.mark.performance
def test_sparse_large_file_reporting(tmp_path: Path) -> None:
    fixture = create_sparse_large_file_fixture(tmp_path)
    connector, pkg_writer, vector_writer, state_store, workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(name="large", root_path=fixture.root)
        records = list(connector.ingest(source))

        assert records, "Sparse large file should be processed"
        telemetry_dir = workspace / "parsed"
        assert telemetry_dir.exists()
        assert set(pkg_writer.documents_by_sha) == {records[0]["sha256"]}
        assert records[0]["size_bytes"] == fixture.metadata["size_bytes"]
        # The connector currently emits parsed payload for each element; ensure file persisted.
        assert Path(records[0]["element_path"]).exists()
    finally:
        state_store.close()


def test_max_files_per_batch_limits_results(tmp_path: Path) -> None:
    fixture = create_nested_fixture(tmp_path)
    connector, pkg_writer, vector_writer, state_store, _workspace = _make_connector(tmp_path)
    try:
        source = LocalIngestionSource(name="nested", root_path=fixture.root, max_files_per_batch=1)
        records = list(connector.ingest(source))

        assert len(records) == 1
        assert len(pkg_writer.documents_by_sha) == 1
        assert len(vector_writer.embeddings_by_sha) == 1
    finally:
        state_store.close()



"""Performance benchmarking tests for LocalFilesConnector throughput."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.connector import LocalFilesConnector
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.metrics import TelemetryRecorder
from futurnal.pipeline.stubs import NormalizationSink
from tests.fixtures.local_connector import create_sparse_large_file_fixture


class NullPKGWriter:
    def write_document(self, payload: dict) -> None:  # pragma: no cover - no-op
        pass

    def remove_document(self, sha256: str) -> None:  # pragma: no cover - no-op
        pass


class NullVectorWriter:
    def write_embedding(self, payload: dict) -> None:  # pragma: no cover - no-op
        pass

    def remove_embedding(self, sha256: str) -> None:  # pragma: no cover - no-op
        pass


@pytest.mark.performance
def test_performance_marker_logs_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = create_sparse_large_file_fixture(tmp_path, size_bytes=5 * 1024 * 1024)
    workspace = tmp_path / "workspace"
    telemetry_dir = workspace / "telemetry"
    telemetry = TelemetryRecorder(output_dir=telemetry_dir, metrics_file="perf.log", summary_file="perf.json")

    def _fake_partition(*, filename: str, strategy: str, include_metadata: bool):
        path = Path(filename)
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

    state_store = StateStore(workspace / "state.db")
    connector = LocalFilesConnector(
        workspace_dir=workspace,
        state_store=state_store,
        element_sink=NormalizationSink(pkg_writer=NullPKGWriter(), vector_writer=NullVectorWriter()),
    )

    source = LocalIngestionSource(name="perf", root_path=fixture.root)

    start = time.perf_counter()
    records = list(connector.ingest(source))
    duration = time.perf_counter() - start
    bytes_processed = sum(record["size_bytes"] for record in records)
    throughput = bytes_processed / duration if duration else 0.0

    telemetry.record(
        job_id="benchmark-local-files",
        duration=duration,
        status="succeeded",
        files_processed=len(records),
        bytes_processed=bytes_processed,
        metadata={
            "fixture": "sparse_large",
            "hardware": "local",
            "threshold_bytes_per_second": 5 * 1024 * 1024,
        },
    )

    summary_path = telemetry_dir / "perf.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())

    overall = summary["overall"]
    assert overall["bytes"] >= bytes_processed
    assert overall["throughput_bytes_per_second"] >= throughput * 0.9
    assert (
        throughput >= 5 * 1024 * 1024
    ), f"Throughput should meet ≥5 MB/s target (observed {throughput:.2f} B/s)"



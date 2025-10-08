"""Tests for orchestrator status collection."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.orchestrator.status import (
    collect_status_report,
    _collect_queue_metrics,
    _collect_system_metrics,
    _collect_throughput_metrics,
    _collect_source_metrics,
)
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.quarantine import QuarantineStore
from futurnal.orchestrator.source_control import PausedSourcesRegistry


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create temporary workspace directory for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


@pytest.fixture
def mock_queue(tmp_path: Path) -> JobQueue:
    """Create a mock job queue for testing."""
    queue_db = tmp_path / "queue.db"
    return JobQueue(queue_db)


@pytest.fixture
def mock_quarantine(tmp_path: Path) -> QuarantineStore:
    """Create a mock quarantine store for testing."""
    quarantine_db = tmp_path / "quarantine.db"
    return QuarantineStore(quarantine_db)


@pytest.fixture
def mock_paused_sources(tmp_path: Path) -> PausedSourcesRegistry:
    """Create a mock paused sources registry for testing."""
    paused_file = tmp_path / "paused_sources.json"
    return PausedSourcesRegistry(paused_file)


def test_collect_status_report_basic(temp_workspace: Path):
    """Test basic status report collection."""
    status = collect_status_report(workspace_path=temp_workspace)

    assert "queue" in status
    assert "workers" in status
    assert "system" in status
    assert "throughput" in status
    assert "sources" in status


def test_collect_queue_metrics_with_empty_queue(mock_queue: JobQueue):
    """Test queue metrics collection with empty queue."""
    metrics = _collect_queue_metrics(mock_queue, None)

    assert metrics["pending"] == 0
    assert metrics["running"] == 0
    assert metrics["completed_24h"] == 0
    assert metrics["failed_24h"] == 0
    assert metrics["quarantined"] == 0


def test_collect_queue_metrics_with_quarantine(
    mock_queue: JobQueue,
    mock_quarantine: QuarantineStore,
):
    """Test queue metrics include quarantine count."""
    from futurnal.orchestrator.models import IngestionJob, JobType, JobPriority
    from futurnal.orchestrator.quarantine import QuarantineReason

    # Add a job to quarantine
    job = IngestionJob(
        job_id="test_job",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
    )
    mock_quarantine.quarantine(
        job=job,
        reason=QuarantineReason.PERMISSION_DENIED,
        error_message="Test error",
    )

    metrics = _collect_queue_metrics(mock_queue, mock_quarantine)
    assert metrics["quarantined"] == 1


def test_collect_queue_metrics_with_none_inputs():
    """Test queue metrics with None inputs returns zeros."""
    metrics = _collect_queue_metrics(None, None)

    assert metrics["pending"] == 0
    assert metrics["running"] == 0
    assert metrics["completed_24h"] == 0
    assert metrics["failed_24h"] == 0
    assert metrics["quarantined"] == 0


@patch("futurnal.orchestrator.status.psutil")
def test_collect_system_metrics(mock_psutil, temp_workspace: Path):
    """Test system metrics collection."""
    # Mock psutil responses
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8

    mock_memory = Mock()
    mock_memory.used = 8 * (1024 ** 3)  # 8 GB
    mock_memory.total = 16 * (1024 ** 3)  # 16 GB
    mock_memory.percent = 50.0
    mock_psutil.virtual_memory.return_value = mock_memory

    mock_disk = Mock()
    mock_disk.free = 100 * (1024 ** 3)  # 100 GB
    mock_disk.total = 500 * (1024 ** 3)  # 500 GB
    mock_disk.percent = 80.0
    mock_psutil.disk_usage.return_value = mock_disk

    metrics = _collect_system_metrics(temp_workspace)

    assert metrics["cpu_percent"] == 45.5
    assert metrics["cpu_count"] == 8
    assert 7.9 < metrics["memory_used_gb"] < 8.1
    assert 15.9 < metrics["memory_total_gb"] < 16.1
    assert metrics["memory_percent"] == 50.0
    assert 99 < metrics["disk_free_gb"] < 101
    assert 499 < metrics["disk_total_gb"] < 501
    assert metrics["disk_percent"] == 80.0


def test_collect_throughput_metrics_no_telemetry(temp_workspace: Path):
    """Test throughput metrics when no telemetry exists."""
    metrics = _collect_throughput_metrics(temp_workspace)

    assert metrics["files_last_hour"] == 0
    assert metrics["bytes_last_hour"] == 0
    assert metrics["rate_bytes_per_second"] == 0.0


def test_collect_throughput_metrics_with_telemetry(temp_workspace: Path):
    """Test throughput metrics from telemetry log."""
    telemetry_dir = temp_workspace / "telemetry"
    telemetry_dir.mkdir(parents=True)
    telemetry_file = telemetry_dir / "telemetry.log"

    # Create sample telemetry entries
    now = datetime.utcnow()
    entries = [
        {
            "job_id": "job1",
            "status": "succeeded",
            "duration": 10.0,
            "files_processed": 100,
            "bytes_processed": 1000000,
            "timestamp": now.isoformat(),
        },
        {
            "job_id": "job2",
            "status": "succeeded",
            "duration": 15.0,
            "files_processed": 50,
            "bytes_processed": 500000,
            "timestamp": now.isoformat(),
        },
        {
            "job_id": "job3",
            "status": "failed",  # Should be ignored
            "duration": 5.0,
            "files_processed": 10,
            "bytes_processed": 100000,
            "timestamp": now.isoformat(),
        },
        {
            "job_id": "job4",
            "status": "succeeded",
            "duration": 20.0,
            "files_processed": 75,
            "bytes_processed": 750000,
            "timestamp": (now - timedelta(hours=2)).isoformat(),  # Too old, should be ignored
        },
    ]

    with telemetry_file.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    metrics = _collect_throughput_metrics(temp_workspace)

    # Should only count successful jobs from last hour (job1 and job2)
    assert metrics["files_last_hour"] == 150
    assert metrics["bytes_last_hour"] == 1500000
    # Rate = total_bytes / total_duration = 1500000 / 25 = 60000
    assert metrics["rate_bytes_per_second"] == pytest.approx(60000.0, rel=0.01)


def test_collect_source_metrics_empty_workspace(
    temp_workspace: Path,
    mock_paused_sources: PausedSourcesRegistry,
):
    """Test source metrics with no configured sources."""
    sources = _collect_source_metrics(temp_workspace, mock_paused_sources)
    assert sources == []


def test_collect_source_metrics_with_local_sources(
    temp_workspace: Path,
    mock_paused_sources: PausedSourcesRegistry,
):
    """Test source discovery for local file sources."""
    # Create local source directories
    local_dir = temp_workspace / "sources" / "local"
    local_dir.mkdir(parents=True)
    (local_dir / "source1").mkdir()
    (local_dir / "source2").mkdir()

    sources = _collect_source_metrics(temp_workspace, mock_paused_sources)

    assert len(sources) == 2
    source_names = {s["name"] for s in sources}
    assert source_names == {"source1", "source2"}
    assert all(s["type"] == "local_files" for s in sources)
    assert all(s["status"] == "active" for s in sources)


def test_collect_source_metrics_with_paused_sources(
    temp_workspace: Path,
    mock_paused_sources: PausedSourcesRegistry,
):
    """Test source metrics with paused sources."""
    # Create local source directories
    local_dir = temp_workspace / "sources" / "local"
    local_dir.mkdir(parents=True)
    (local_dir / "source1").mkdir()
    (local_dir / "source2").mkdir()

    # Pause one source
    mock_paused_sources.pause("source1")

    sources = _collect_source_metrics(temp_workspace, mock_paused_sources)

    assert len(sources) == 2
    source1 = next(s for s in sources if s["name"] == "source1")
    source2 = next(s for s in sources if s["name"] == "source2")

    assert source1["status"] == "paused"
    assert source2["status"] == "active"


def test_collect_source_metrics_with_imap_sources(
    temp_workspace: Path,
    mock_paused_sources: PausedSourcesRegistry,
):
    """Test source discovery for IMAP mailboxes."""
    # Create IMAP source directories with descriptors
    imap_dir = temp_workspace / "sources" / "imap"
    mailbox1_dir = imap_dir / "mailbox1"
    mailbox1_dir.mkdir(parents=True)

    descriptor = {
        "id": "mailbox1",
        "email_address": "test@example.com",
        "server": "imap.example.com",
        "folders": ["INBOX"],
    }
    (mailbox1_dir / "descriptor.json").write_text(json.dumps(descriptor))

    sources = _collect_source_metrics(temp_workspace, mock_paused_sources)

    assert len(sources) == 1
    assert sources[0]["name"] == "test@example.com"
    assert sources[0]["type"] == "imap_mailbox"
    assert sources[0]["status"] == "active"


def test_collect_source_metrics_with_github_sources(
    temp_workspace: Path,
    mock_paused_sources: PausedSourcesRegistry,
):
    """Test source discovery for GitHub repositories."""
    # Create GitHub source directories with descriptors
    github_dir = temp_workspace / "sources" / "github"
    repo1_dir = github_dir / "repo1"
    repo1_dir.mkdir(parents=True)

    descriptor = {
        "id": "repo1",
        "full_name": "owner/repository",
        "url": "https://github.com/owner/repository",
    }
    (repo1_dir / "descriptor.json").write_text(json.dumps(descriptor))

    sources = _collect_source_metrics(temp_workspace, mock_paused_sources)

    assert len(sources) == 1
    assert sources[0]["name"] == "owner/repository"
    assert sources[0]["type"] == "github_repository"
    assert sources[0]["status"] == "active"


def test_collect_status_report_integration(temp_workspace: Path):
    """Test full status report collection integration."""
    # Create some source directories
    local_dir = temp_workspace / "sources" / "local"
    local_dir.mkdir(parents=True)
    (local_dir / "test_source").mkdir()

    # Create telemetry
    telemetry_dir = temp_workspace / "telemetry"
    telemetry_dir.mkdir(parents=True)
    telemetry_file = telemetry_dir / "telemetry.log"
    telemetry_file.write_text(
        json.dumps({
            "job_id": "job1",
            "status": "succeeded",
            "duration": 10.0,
            "files_processed": 100,
            "bytes_processed": 1000000,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "active_workers": 2,
                "configured_workers": 4,
            },
        }) + "\n"
    )

    status = collect_status_report(workspace_path=temp_workspace)

    # Verify all sections present
    assert "queue" in status
    assert "workers" in status
    assert "system" in status
    assert "throughput" in status
    assert "sources" in status

    # Verify workers from telemetry
    assert status["workers"]["active"] == 2
    assert status["workers"]["max"] == 4
    assert status["workers"]["utilization"] == 50.0

    # Verify sources discovered
    assert len(status["sources"]) == 1
    assert status["sources"][0]["name"] == "test_source"


def test_collect_status_report_with_missing_queue(temp_workspace: Path):
    """Test status collection when queue DB doesn't exist."""
    status = collect_status_report(workspace_path=temp_workspace)

    # Should still work with zeros for queue metrics
    assert status["queue"]["pending"] == 0
    assert status["queue"]["running"] == 0

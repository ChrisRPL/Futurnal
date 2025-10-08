"""Shared fixtures for runbook tests."""

import tempfile
from pathlib import Path

import pytest

from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.quarantine import QuarantineStore
from futurnal.orchestrator.source_control import PausedSourcesRegistry
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.daemon import OrchestratorDaemon
from futurnal.orchestrator.db_utils import DatabaseManager
from futurnal.orchestrator.telemetry_analysis import TelemetryAnalyzer
from datetime import datetime


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory structure."""
    workspace = tmp_path / "futurnal_test"
    workspace.mkdir()

    # Create subdirectories
    (workspace / "queue").mkdir()
    (workspace / "quarantine").mkdir()
    (workspace / "orchestrator").mkdir()
    (workspace / "telemetry").mkdir()
    (workspace / "audit").mkdir()
    (workspace / "sources" / "local").mkdir(parents=True)
    (workspace / "sources" / "imap").mkdir(parents=True)
    (workspace / "sources" / "github").mkdir(parents=True)

    return workspace


@pytest.fixture
def job_queue(temp_workspace):
    """Create a JobQueue instance with test database."""
    db_path = temp_workspace / "queue" / "jobs.db"
    return JobQueue(db_path)


@pytest.fixture
def quarantine_store(temp_workspace):
    """Create a QuarantineStore instance."""
    db_path = temp_workspace / "quarantine" / "quarantine.db"
    return QuarantineStore(db_path)


@pytest.fixture
def paused_sources_registry(temp_workspace):
    """Create a PausedSourcesRegistry instance."""
    registry_path = temp_workspace / "orchestrator" / "paused_sources.json"
    return PausedSourcesRegistry(registry_path)


@pytest.fixture
def orchestrator_daemon(temp_workspace):
    """Create an OrchestratorDaemon instance."""
    return OrchestratorDaemon(temp_workspace)


@pytest.fixture
def database_manager(temp_workspace):
    """Create a DatabaseManager instance."""
    db_path = temp_workspace / "queue" / "jobs.db"
    # Create empty database
    JobQueue(db_path)
    return DatabaseManager(db_path)


@pytest.fixture
def telemetry_analyzer(temp_workspace):
    """Create a TelemetryAnalyzer instance with sample data."""
    telemetry_dir = temp_workspace / "telemetry"
    telemetry_file = telemetry_dir / "telemetry.log"

    # Create sample telemetry entries
    sample_entries = [
        {
            "job_id": "test-job-1",
            "duration": 10.5,
            "status": "succeeded",
            "files_processed": 5,
            "bytes_processed": 1024000,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"connector_type": "local_files"},
        },
        {
            "job_id": "test-job-2",
            "duration": 5.2,
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"connector_type": "imap_mailbox", "failure_reason": "connection_error"},
        },
    ]

    import json
    with telemetry_file.open("w") as f:
        for entry in sample_entries:
            f.write(json.dumps(entry) + "\n")

    return TelemetryAnalyzer(telemetry_dir)


@pytest.fixture
def sample_jobs(job_queue):
    """Create sample jobs in the queue."""
    jobs = [
        IngestionJob(
            job_id="job-1",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "test-source-1"},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        ),
        IngestionJob(
            job_id="job-2",
            job_type=JobType.OBSIDIAN_VAULT,
            payload={"source_name": "test-source-2"},
            priority=JobPriority.HIGH,
            scheduled_for=datetime.utcnow(),
        ),
    ]

    for job in jobs:
        job_queue.enqueue(job)

    return jobs

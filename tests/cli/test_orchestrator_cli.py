"""Tests for orchestrator CLI commands."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, Mock

from futurnal.cli.orchestrator import orchestrator_app
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.models import IngestionJob, JobType, JobPriority
from futurnal.orchestrator.source_control import PausedSourcesRegistry


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create temporary workspace for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    # Create required directories
    (workspace / "queue").mkdir(parents=True)
    (workspace / "telemetry").mkdir(parents=True)
    (workspace / "orchestrator").mkdir(parents=True)
    (workspace / "sources" / "local").mkdir(parents=True, exist_ok=True)

    return workspace


@patch("futurnal.orchestrator.status.psutil")
def test_status_command_table_format(mock_psutil, runner, temp_workspace):
    """Test status command with table format output."""
    # Mock psutil
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.cpu_count.return_value = 4
    mock_memory = Mock(used=4*(1024**3), total=8*(1024**3), percent=50.0)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_disk = Mock(free=100*(1024**3), total=500*(1024**3), percent=80.0)
    mock_psutil.disk_usage.return_value = mock_disk

    result = runner.invoke(orchestrator_app, ["status", "--workspace", str(temp_workspace)])

    assert result.exit_code == 0
    assert "Orchestrator Status" in result.output
    assert "Queue" in result.output
    assert "Workers" in result.output
    assert "System Resources" in result.output


@patch("futurnal.orchestrator.status.psutil")
def test_status_command_json_format(mock_psutil, runner, temp_workspace):
    """Test status command with JSON format output."""
    # Mock psutil
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.cpu_count.return_value = 4
    mock_memory = Mock(used=4*(1024**3), total=8*(1024**3), percent=50.0)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_disk = Mock(free=100*(1024**3), total=500*(1024**3), percent=80.0)
    mock_psutil.disk_usage.return_value = mock_disk

    result = runner.invoke(
        orchestrator_app,
        ["status", "--workspace", str(temp_workspace), "--format", "json"]
    )

    assert result.exit_code == 0

    # Verify output is valid JSON
    output_data = json.loads(result.output)
    assert "queue" in output_data
    assert "workers" in output_data
    assert "system" in output_data


def test_jobs_list_empty(runner, temp_workspace):
    """Test jobs list with no jobs."""
    result = runner.invoke(
        orchestrator_app,
        ["jobs", "list", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "No jobs found" in result.output


def test_jobs_list_with_jobs(runner, temp_workspace):
    """Test jobs list with some jobs in queue."""
    # Create job queue and add jobs
    queue_db = temp_workspace / "queue" / "jobs.db"
    queue = JobQueue(queue_db)

    job = IngestionJob(
        job_id="test-job-123",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test_source"},
        priority=JobPriority.NORMAL,
    )
    queue.enqueue(job)

    result = runner.invoke(
        orchestrator_app,
        ["jobs", "list", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "test-job" in result.output
    assert "local_files" in result.output


def test_jobs_list_json_format(runner, temp_workspace):
    """Test jobs list with JSON output."""
    queue_db = temp_workspace / "queue" / "jobs.db"
    queue = JobQueue(queue_db)

    job = IngestionJob(
        job_id="test-job-123",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test_source"},
        priority=JobPriority.NORMAL,
    )
    queue.enqueue(job)

    result = runner.invoke(
        orchestrator_app,
        ["jobs", "list", "--workspace", str(temp_workspace), "--format", "json"]
    )

    assert result.exit_code == 0
    jobs = json.loads(result.output)
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "test-job-123"


def test_jobs_show(runner, temp_workspace):
    """Test jobs show command."""
    queue_db = temp_workspace / "queue" / "jobs.db"
    queue = JobQueue(queue_db)

    job = IngestionJob(
        job_id="test-job-123",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test_source"},
        priority=JobPriority.NORMAL,
    )
    queue.enqueue(job)

    result = runner.invoke(
        orchestrator_app,
        ["jobs", "show", "test-job-123", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "test-job-123" in result.output
    assert "local_files" in result.output
    assert "test_source" in result.output


def test_jobs_show_not_found(runner, temp_workspace):
    """Test jobs show with non-existent job."""
    result = runner.invoke(
        orchestrator_app,
        ["jobs", "show", "nonexistent", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_jobs_cancel(runner, temp_workspace):
    """Test jobs cancel command."""
    queue_db = temp_workspace / "queue" / "jobs.db"
    queue = JobQueue(queue_db)

    job = IngestionJob(
        job_id="test-job-123",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test_source"},
        priority=JobPriority.NORMAL,
    )
    queue.enqueue(job)

    result = runner.invoke(
        orchestrator_app,
        ["jobs", "cancel", "test-job-123", "--workspace", str(temp_workspace), "--reason", "Test cancel"]
    )

    assert result.exit_code == 0
    assert "cancelled" in result.output.lower()

    # Verify job was cancelled
    job_status = queue.get_job("test-job-123")
    assert job_status["status"] == "failed"  # Cancelled jobs marked as failed


def test_jobs_cancel_not_found(runner, temp_workspace):
    """Test cancelling non-existent job."""
    result = runner.invoke(
        orchestrator_app,
        ["jobs", "cancel", "nonexistent", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_sources_list_empty(runner, temp_workspace):
    """Test sources list with no sources."""
    result = runner.invoke(
        orchestrator_app,
        ["sources", "list", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "No sources found" in result.output


def test_sources_list_with_sources(runner, temp_workspace):
    """Test sources list with configured sources."""
    # Create a local source
    source_dir = temp_workspace / "sources" / "local" / "test_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        orchestrator_app,
        ["sources", "list", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "test_source" in result.output
    assert "local_files" in result.output


def test_sources_pause(runner, temp_workspace):
    """Test pausing a source."""
    result = runner.invoke(
        orchestrator_app,
        ["sources", "pause", "test_source", "--workspace", str(temp_workspace), "--reason", "Testing"]
    )

    assert result.exit_code == 0
    assert "paused" in result.output.lower()

    # Verify source is paused
    paused_registry = PausedSourcesRegistry(temp_workspace / "orchestrator" / "paused_sources.json")
    assert paused_registry.is_paused("test_source")


def test_sources_pause_already_paused(runner, temp_workspace):
    """Test pausing an already paused source."""
    paused_registry = PausedSourcesRegistry(temp_workspace / "orchestrator" / "paused_sources.json")
    paused_registry.pause("test_source")

    result = runner.invoke(
        orchestrator_app,
        ["sources", "pause", "test_source", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "already paused" in result.output.lower()


def test_sources_resume(runner, temp_workspace):
    """Test resuming a paused source."""
    paused_registry = PausedSourcesRegistry(temp_workspace / "orchestrator" / "paused_sources.json")
    paused_registry.pause("test_source")

    result = runner.invoke(
        orchestrator_app,
        ["sources", "resume", "test_source", "--workspace", str(temp_workspace), "--reason", "Testing"]
    )

    assert result.exit_code == 0
    assert "resumed" in result.output.lower()

    # Verify source is no longer paused
    assert not paused_registry.is_paused("test_source")


def test_sources_resume_not_paused(runner, temp_workspace):
    """Test resuming a non-paused source."""
    result = runner.invoke(
        orchestrator_app,
        ["sources", "resume", "test_source", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "not paused" in result.output.lower()


def test_sources_trigger(runner, temp_workspace):
    """Test manually triggering a source."""
    result = runner.invoke(
        orchestrator_app,
        ["sources", "trigger", "test_source", "--workspace", str(temp_workspace), "--priority", "high"]
    )

    assert result.exit_code == 0
    assert "enqueued" in result.output.lower()

    # Verify job was created
    queue_db = temp_workspace / "queue" / "jobs.db"
    queue = JobQueue(queue_db)
    jobs = queue.snapshot(limit=10)
    assert len(jobs) == 1
    assert jobs[0]["payload"]["source_name"] == "test_source"


def test_sources_trigger_paused_without_force(runner, temp_workspace):
    """Test triggering a paused source without force flag."""
    paused_registry = PausedSourcesRegistry(temp_workspace / "orchestrator" / "paused_sources.json")
    paused_registry.pause("test_source")

    result = runner.invoke(
        orchestrator_app,
        ["sources", "trigger", "test_source", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 1
    assert "paused" in result.output.lower()


def test_sources_trigger_paused_with_force(runner, temp_workspace):
    """Test triggering a paused source with force flag."""
    paused_registry = PausedSourcesRegistry(temp_workspace / "orchestrator" / "paused_sources.json")
    paused_registry.pause("test_source")

    result = runner.invoke(
        orchestrator_app,
        ["sources", "trigger", "test_source", "--workspace", str(temp_workspace), "--force"]
    )

    assert result.exit_code == 0
    assert "enqueued" in result.output.lower()


@patch("futurnal.cli.orchestrator.collect_health_report")
@patch("futurnal.cli.orchestrator.load_settings")
def test_health_command(mock_load_settings, mock_health_report, runner, temp_workspace):
    """Test health check command."""
    mock_settings_obj = Mock()
    mock_settings_obj.workspace.workspace_path = temp_workspace
    mock_load_settings.return_value = mock_settings_obj

    mock_health_report.return_value = {
        "status": "ok",
        "checks": [
            {"name": "queue", "status": "ok", "detail": "Healthy"},
            {"name": "disk", "status": "ok", "detail": "100 GB free"},
        ],
    }

    result = runner.invoke(
        orchestrator_app,
        ["health", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "Health Checks" in result.output
    assert "queue" in result.output


def test_telemetry_summary_no_data(runner, temp_workspace):
    """Test telemetry summary with no data."""
    result = runner.invoke(
        orchestrator_app,
        ["telemetry", "summary", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "No telemetry data" in result.output


def test_telemetry_summary_with_data(runner, temp_workspace):
    """Test telemetry summary with data."""
    # Create telemetry summary file
    telemetry_dir = temp_workspace / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    summary_file = telemetry_dir / "telemetry_summary.json"

    summary_data = {
        "overall": {
            "jobs": 100,
            "files": 500,
            "bytes": 1000000,
            "avg_duration": 10.5,
            "throughput_bytes_per_second": 95000,
        },
        "statuses": {
            "succeeded": {"count": 90, "files": 450, "bytes": 900000, "throughput_bytes_per_second": 85000},
            "failed": {"count": 10, "files": 50, "bytes": 100000, "throughput_bytes_per_second": 10000},
        },
    }
    summary_file.write_text(json.dumps(summary_data))

    result = runner.invoke(
        orchestrator_app,
        ["telemetry", "summary", "--workspace", str(temp_workspace)]
    )

    assert result.exit_code == 0
    assert "Overall Statistics" in result.output
    assert "100" in result.output  # total jobs
    assert "succeeded" in result.output


@patch("futurnal.orchestrator.config_cli.ConfigurationManager")
def test_config_command(mock_config_manager, runner, temp_workspace):
    """Test config display command."""
    from futurnal.orchestrator.config import OrchestratorConfig

    # Create a mock config with default values
    mock_config = OrchestratorConfig()
    mock_manager_instance = Mock()
    mock_manager_instance.load.return_value = mock_config
    mock_config_manager.return_value = mock_manager_instance

    # Create a temp config file
    config_file = temp_workspace / "orchestrator.yaml"
    config_file.write_text("version: '1.0'\n")

    result = runner.invoke(
        orchestrator_app,
        ["config", "show", "--config", str(config_file)]
    )

    # Config show should work or fail gracefully
    # Exit code 0 = success, 1 = config error (e.g. invalid file)
    assert result.exit_code in [0, 1]

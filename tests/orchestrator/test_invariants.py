"""Tests for state machine invariant checking."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue, JobStatus
from futurnal.orchestrator.scheduler import IngestionOrchestrator
from futurnal.orchestrator.state_machine import StateMachineInvariants


def make_job(job_id: str) -> IngestionJob:
    """Helper to create a test job."""
    return IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
    )


@pytest.fixture
def orchestrator(tmp_path: Path) -> IngestionOrchestrator:
    """Create test orchestrator."""
    queue = JobQueue(tmp_path / "queue.db")
    state_store = StateStore(tmp_path / "state")

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(tmp_path),
        loop=asyncio.new_event_loop(),
        element_sink=MagicMock(),
    )

    return orchestrator


def test_running_jobs_have_workers_passes_when_all_tracked(orchestrator: IngestionOrchestrator):
    """Test invariant passes when all RUNNING jobs have active workers."""
    # Add job to queue as RUNNING
    job = make_job("job-1")
    orchestrator._job_queue.enqueue(job)
    orchestrator._job_queue.mark_running("job-1")

    # Add job to active jobs map
    orchestrator._active_jobs_map["job-1"] = {
        "job_type": JobType.LOCAL_FILES.value,
        "started_at": datetime.utcnow(),
    }

    # Check invariant
    result = StateMachineInvariants.check_running_jobs_have_workers(orchestrator)

    assert result is True


def test_running_jobs_have_workers_fails_when_not_tracked(orchestrator: IngestionOrchestrator):
    """Test invariant fails when RUNNING job has no active worker."""
    # Add job to queue as RUNNING
    job = make_job("job-1")
    orchestrator._job_queue.enqueue(job)
    orchestrator._job_queue.mark_running("job-1")

    # Do NOT add job to active jobs map (simulates missing worker)

    # Check invariant
    result = StateMachineInvariants.check_running_jobs_have_workers(orchestrator)

    assert result is False


def test_running_jobs_have_workers_passes_with_no_running_jobs(orchestrator: IngestionOrchestrator):
    """Test invariant passes when there are no RUNNING jobs."""
    # Don't add any jobs

    # Check invariant
    result = StateMachineInvariants.check_running_jobs_have_workers(orchestrator)

    assert result is True


def test_running_jobs_have_workers_passes_with_multiple_jobs(orchestrator: IngestionOrchestrator):
    """Test invariant with multiple RUNNING jobs all tracked."""
    # Add multiple RUNNING jobs
    for i in range(3):
        job = make_job(f"job-{i}")
        orchestrator._job_queue.enqueue(job)
        orchestrator._job_queue.mark_running(f"job-{i}")

        # Track in active jobs map
        orchestrator._active_jobs_map[f"job-{i}"] = {
            "job_type": JobType.LOCAL_FILES.value,
            "started_at": datetime.utcnow(),
        }

    # Check invariant
    result = StateMachineInvariants.check_running_jobs_have_workers(orchestrator)

    assert result is True


def test_succeeded_jobs_immutable_placeholder(orchestrator: IngestionOrchestrator):
    """Test succeeded jobs immutability check (placeholder implementation)."""
    # This is a placeholder test since the implementation returns True
    result = StateMachineInvariants.check_succeeded_jobs_immutable(
        orchestrator._job_queue
    )

    assert result is True


def test_attempts_monotonic_placeholder(orchestrator: IngestionOrchestrator):
    """Test attempts monotonicity check (placeholder implementation)."""
    # This is a placeholder test since the implementation returns True
    result = StateMachineInvariants.check_attempts_monotonic(
        orchestrator._job_queue
    )

    assert result is True


def test_validator_check_invariants_returns_empty_on_success(orchestrator: IngestionOrchestrator):
    """Test that validator returns empty list when all invariants pass."""
    # Create a valid state with tracked jobs
    job = make_job("job-1")
    orchestrator._job_queue.enqueue(job)
    orchestrator._job_queue.mark_running("job-1")
    orchestrator._active_jobs_map["job-1"] = {
        "job_type": JobType.LOCAL_FILES.value,
        "started_at": datetime.utcnow(),
    }

    # Check invariants via validator
    violations = orchestrator._job_queue._validator.check_invariants(orchestrator)

    assert violations == []


def test_validator_check_invariants_returns_violations(orchestrator: IngestionOrchestrator):
    """Test that validator returns violations when invariants fail."""
    # Create an invalid state with untracked RUNNING job
    job = make_job("job-1")
    orchestrator._job_queue.enqueue(job)
    orchestrator._job_queue.mark_running("job-1")
    # Do NOT add to active jobs map

    # Check invariants via validator
    violations = orchestrator._job_queue._validator.check_invariants(orchestrator)

    assert len(violations) > 0
    assert "RUNNING jobs without active workers detected" in violations[0]

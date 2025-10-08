"""Simplified integration tests for quarantine system.

These tests validate quarantine system integration without requiring
full orchestrator instantiation (to avoid missing connector dependencies).
"""

from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.quarantine import (
    QuarantineReason,
    QuarantineStore,
    classify_failure,
)
from futurnal.orchestrator.queue import JobQueue


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


@pytest.fixture
def job_queue(workspace: Path) -> JobQueue:
    """Create a test job queue."""
    return JobQueue(workspace / "queue" / "jobs.db")


@pytest.fixture
def quarantine_store(workspace: Path) -> QuarantineStore:
    """Create a test quarantine store."""
    return QuarantineStore(workspace / "quarantine" / "quarantine.db")


def test_job_queue_and_quarantine_isolation(
    job_queue: JobQueue,
    quarantine_store: QuarantineStore,
) -> None:
    """Test that quarantined jobs are separate from job queue."""

    # Create and enqueue a job
    job = IngestionJob(
        job_id="test_job",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test", "attempts": 3},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )
    job_queue.enqueue(job)

    # Mark as failed
    job_queue.mark_failed("test_job")

    # Quarantine the job
    quarantine_store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Test failure",
    )

    # Verify job is in quarantine
    quarantined = quarantine_store.get("test_job")
    assert quarantined is not None
    assert quarantined.job_id == "test_job"

    # Verify job queue and quarantine are independent
    # (quarantining doesn't remove from queue - that's orchestrator's job)
    queue_snapshot = job_queue.snapshot()
    assert any(j["job_id"] == "test_job" for j in queue_snapshot)


def test_quarantine_retry_workflow(
    job_queue: JobQueue,
    quarantine_store: QuarantineStore,
) -> None:
    """Test the complete quarantine and retry workflow."""

    # Step 1: Job fails and gets quarantined
    job = IngestionJob(
        job_id="retry_job",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test", "attempts": 3},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )

    quarantine_store.quarantine(
        job=job,
        reason=QuarantineReason.TIMEOUT,
        error_message="Operation timed out",
    )

    # Step 2: Verify job is in quarantine
    quarantined = quarantine_store.get("retry_job")
    assert quarantined is not None
    assert quarantined.retry_count == 0

    # Step 3: Simulate CLI retry - create new job with quarantine markers
    retry_job = IngestionJob(
        job_id="retry_job",
        job_type=JobType.LOCAL_FILES,
        payload={
            "source_name": "test",
            "from_quarantine": "retry_job",
            "quarantine_retry": True,
            "attempts": 0,
        },
        priority=JobPriority.HIGH,
        scheduled_for=datetime.utcnow(),
    )
    job_queue.enqueue(retry_job)

    # Step 4: Simulate orchestrator processing success
    quarantine_store.mark_retry_attempted("retry_job", success=True)

    # Step 5: Verify job removed from quarantine
    assert quarantine_store.get("retry_job") is None


def test_quarantine_failure_classification_integration(
    quarantine_store: QuarantineStore,
) -> None:
    """Test that failure classification integrates with quarantine storage."""

    error_cases = [
        ("Permission denied: access restricted", QuarantineReason.PERMISSION_DENIED),
        ("Parse error in document", QuarantineReason.PARSE_ERROR),
        ("Connection timeout exceeded", QuarantineReason.TIMEOUT),
        ("Neo4j database unavailable", QuarantineReason.DEPENDENCY_FAILURE),
    ]

    for idx, (error_msg, expected_reason) in enumerate(error_cases):
        # Classify the failure
        classified_reason = classify_failure(error_msg)
        assert classified_reason == expected_reason

        # Quarantine with classified reason
        job = IngestionJob(
            job_id=f"classify_job_{idx}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "test"},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )

        quarantine_store.quarantine(
            job=job,
            reason=classified_reason,
            error_message=error_msg,
        )

        # Verify stored with correct reason
        quarantined = quarantine_store.get(f"classify_job_{idx}")
        assert quarantined is not None
        assert quarantined.reason == expected_reason


def test_quarantine_statistics_integration(
    quarantine_store: QuarantineStore,
) -> None:
    """Test quarantine statistics with multiple jobs."""

    # Create several quarantined jobs
    for i in range(3):
        job = IngestionJob(
            job_id=f"job_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": f"source_{i}"},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        quarantine_store.quarantine(
            job=job,
            reason=QuarantineReason.PARSE_ERROR,
            error_message=f"Error {i}",
        )

    # Create job with different reason
    job4 = IngestionJob(
        job_id="job_4",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "source_4"},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )
    quarantine_store.quarantine(
        job=job4,
        reason=QuarantineReason.TIMEOUT,
        error_message="Timeout",
    )

    # Get statistics
    stats = quarantine_store.statistics()

    assert stats["total_quarantined"] == 4
    assert stats["by_reason"]["parse_error"] == 3
    assert stats["by_reason"]["timeout"] == 1
    assert stats["recent_quarantines_24h"] == 4


def test_quarantine_retry_failure_tracking(
    quarantine_store: QuarantineStore,
) -> None:
    """Test that failed retry attempts are tracked."""

    job = IngestionJob(
        job_id="retry_fail_job",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )

    quarantine_store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Initial error",
    )

    # First retry fails
    quarantine_store.mark_retry_attempted(
        "retry_fail_job",
        success=False,
        error_message="Still failing",
    )

    quarantined = quarantine_store.get("retry_fail_job")
    assert quarantined is not None
    assert quarantined.retry_count == 1
    assert quarantined.retry_failure_count == 1
    assert quarantined.last_retry_at is not None

    # Second retry fails
    quarantine_store.mark_retry_attempted(
        "retry_fail_job",
        success=False,
        error_message="Still failing",
    )

    quarantined = quarantine_store.get("retry_fail_job")
    assert quarantined is not None
    assert quarantined.retry_count == 2
    assert quarantined.retry_failure_count == 2


def test_quarantine_metadata_preservation(
    quarantine_store: QuarantineStore,
) -> None:
    """Test that job metadata is preserved for recovery."""

    original_payload = {
        "source_name": "critical_source",
        "path": "/important/data",
        "custom_field": "value",
        "attempts": 3,
    }

    job = IngestionJob(
        job_id="metadata_job",
        job_type=JobType.OBSIDIAN_VAULT,
        payload=original_payload,
        priority=JobPriority.HIGH,
        scheduled_for=datetime.utcnow(),
    )

    quarantine_store.quarantine(
        job=job,
        reason=QuarantineReason.CONNECTOR_ERROR,
        error_message="Connector failed",
        metadata={"additional": "info"},
    )

    # Verify all data is preserved
    quarantined = quarantine_store.get("metadata_job")
    assert quarantined is not None
    assert quarantined.job_type == JobType.OBSIDIAN_VAULT.value
    assert quarantined.original_payload == original_payload
    assert quarantined.metadata["additional"] == "info"


def test_quarantine_purge_workflow(
    quarantine_store: QuarantineStore,
) -> None:
    """Test purging old quarantined jobs."""
    from datetime import timedelta

    # Create old job
    job1 = IngestionJob(
        job_id="old_job",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )
    quarantine_store.quarantine(
        job=job1,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Old error",
    )

    # Manually age the job
    old_time = (datetime.utcnow() - timedelta(days=60)).isoformat()
    with quarantine_store._lock:
        with quarantine_store._conn:
            quarantine_store._conn.execute(
                "UPDATE quarantined_jobs SET quarantined_at = ? WHERE job_id = ?",
                (old_time, "old_job"),
            )

    # Create new job
    job2 = IngestionJob(
        job_id="new_job",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )
    quarantine_store.quarantine(
        job=job2,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="New error",
    )

    # Purge old jobs
    purged = quarantine_store.purge_old(days=30)
    assert purged == 1

    # Verify only new job remains
    assert quarantine_store.get("old_job") is None
    assert quarantine_store.get("new_job") is not None

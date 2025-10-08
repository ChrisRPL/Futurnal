"""Tests for enhanced queue operations with state validation."""

from pathlib import Path

import pytest

from futurnal.orchestrator.exceptions import (
    InvalidStateTransitionError,
    StateTransitionRaceError,
    JobNotFoundError,
)
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue, JobStatus


def make_job(job_id: str) -> IngestionJob:
    """Helper to create a test job."""
    return IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
    )


def test_mark_running_validates_transition(tmp_path: Path):
    """Test that mark_running validates state transition."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)

    # Valid transition PENDING → RUNNING should succeed
    queue.mark_running("test-job")

    # Verify job is RUNNING
    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.RUNNING.value


def test_mark_running_idempotent_allows_same_state(tmp_path: Path):
    """Test that mark_running allows idempotent RUNNING → RUNNING."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")

    # Idempotent transition RUNNING → RUNNING should succeed
    queue.mark_running("test-job", idempotent=True)

    # Verify job is still RUNNING
    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.RUNNING.value


def test_mark_running_raises_on_invalid_transition_strict_mode(tmp_path: Path):
    """Test that mark_running raises on invalid transition when idempotent=False."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")
    queue.mark_completed("test-job")

    # Invalid transition SUCCEEDED → RUNNING should raise with idempotent=False
    with pytest.raises(InvalidStateTransitionError):
        queue.mark_running("test-job", idempotent=False)


def test_mark_completed_only_from_running(tmp_path: Path):
    """Test that mark_completed only succeeds from RUNNING state."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")

    # Valid transition RUNNING → SUCCEEDED
    queue.mark_completed("test-job")

    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.SUCCEEDED.value


def test_mark_completed_idempotent_allows_same_state(tmp_path: Path):
    """Test that mark_completed allows idempotent SUCCEEDED → SUCCEEDED."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")
    queue.mark_completed("test-job")

    # Idempotent transition SUCCEEDED → SUCCEEDED should succeed
    queue.mark_completed("test-job", idempotent=True)

    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.SUCCEEDED.value


def test_mark_failed_validates_transition(tmp_path: Path):
    """Test that mark_failed validates state transition."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")

    # Valid transition RUNNING → FAILED
    queue.mark_failed("test-job")

    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.FAILED.value


def test_mark_failed_idempotent_allows_same_state(tmp_path: Path):
    """Test that mark_failed allows idempotent FAILED → FAILED."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")
    queue.mark_failed("test-job")

    # Idempotent transition FAILED → FAILED should succeed
    queue.mark_failed("test-job", idempotent=True)

    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.FAILED.value


def test_mark_quarantined_validates_transition(tmp_path: Path):
    """Test that mark_quarantined validates state transition."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")
    queue.mark_failed("test-job")

    # Valid transition FAILED → QUARANTINED
    queue.mark_quarantined("test-job")

    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.QUARANTINED.value


def test_mark_quarantined_idempotent_allows_same_state(tmp_path: Path):
    """Test that mark_quarantined allows idempotent QUARANTINED → QUARANTINED."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)
    queue.mark_running("test-job")
    queue.mark_failed("test-job")
    queue.mark_quarantined("test-job")

    # Idempotent transition QUARANTINED → QUARANTINED should succeed
    queue.mark_quarantined("test-job", idempotent=True)

    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.QUARANTINED.value


def test_get_status_raises_on_nonexistent_job(tmp_path: Path):
    """Test that _get_status raises JobNotFoundError for nonexistent job."""
    queue = JobQueue(tmp_path / "queue.db")

    with pytest.raises(JobNotFoundError):
        queue._get_status("nonexistent-job")


def test_atomic_updates_with_transaction(tmp_path: Path):
    """Test that state updates use SQLite transactions."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("test-job")
    queue.enqueue(job)

    # Mark running should be atomic (uses with self._conn)
    queue.mark_running("test-job")

    # Verify the update persisted
    job_data = queue.get_job("test-job")
    assert job_data["status"] == JobStatus.RUNNING.value
    assert job_data["attempts"] == 1


def test_concurrent_transitions_thread_safe(tmp_path: Path):
    """Test that concurrent transitions are thread-safe via lock."""
    import threading

    queue = JobQueue(tmp_path / "queue.db")

    # Create multiple jobs
    for i in range(5):
        job = make_job(f"job-{i}")
        queue.enqueue(job)

    # Concurrent marking as running
    threads = []
    for i in range(5):
        thread = threading.Thread(
            target=lambda job_id: queue.mark_running(job_id),
            args=(f"job-{i}",),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Verify all jobs are RUNNING
    for i in range(5):
        job_data = queue.get_job(f"job-{i}")
        assert job_data["status"] == JobStatus.RUNNING.value


def test_full_lifecycle_with_validation(tmp_path: Path):
    """Test full job lifecycle with validation at each step."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("lifecycle-job")
    queue.enqueue(job)

    # PENDING → RUNNING
    queue.mark_running("lifecycle-job")
    assert queue.get_job("lifecycle-job")["status"] == JobStatus.RUNNING.value

    # RUNNING → FAILED
    queue.mark_failed("lifecycle-job")
    assert queue.get_job("lifecycle-job")["status"] == JobStatus.FAILED.value

    # FAILED → PENDING (retry)
    queue.reschedule("lifecycle-job", retry_delay_seconds=60)
    assert queue.get_job("lifecycle-job")["status"] == JobStatus.PENDING.value

    # PENDING → RUNNING (second attempt)
    queue.mark_running("lifecycle-job")
    assert queue.get_job("lifecycle-job")["status"] == JobStatus.RUNNING.value

    # RUNNING → SUCCEEDED
    queue.mark_completed("lifecycle-job")
    assert queue.get_job("lifecycle-job")["status"] == JobStatus.SUCCEEDED.value


def test_quarantine_to_pending_manual_retry(tmp_path: Path):
    """Test manual retry from QUARANTINED to PENDING."""
    queue = JobQueue(tmp_path / "queue.db")

    job = make_job("quarantined-job")
    queue.enqueue(job)
    queue.mark_running("quarantined-job")
    queue.mark_failed("quarantined-job")
    queue.mark_quarantined("quarantined-job")

    # Manual retry: QUARANTINED → PENDING
    queue.reschedule("quarantined-job", retry_delay_seconds=0)

    job_data = queue.get_job("quarantined-job")
    assert job_data["status"] == JobStatus.PENDING.value

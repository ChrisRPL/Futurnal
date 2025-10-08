"""Tests for deadlock detection and recovery."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from futurnal.orchestrator.deadlock import DeadlockDetector
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


def test_detect_stalled_jobs_finds_old_running_jobs(tmp_path: Path):
    """Test that detector identifies jobs stuck in RUNNING state."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=60)

    # Enqueue and mark running
    job = make_job("stalled-job")
    queue.enqueue(job)
    queue.mark_running("stalled-job")

    # Manually set updated_at to old timestamp to simulate stalled job
    old_time = (datetime.utcnow() - timedelta(seconds=120)).isoformat()
    with queue._lock:
        queue._conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
            (old_time, "stalled-job"),
        )
        queue._conn.commit()

    # Detect stalled jobs
    stalled = detector.detect_stalled_jobs()

    assert len(stalled) == 1
    assert "stalled-job" in stalled


def test_detect_stalled_jobs_ignores_recent_running_jobs(tmp_path: Path):
    """Test that detector doesn't flag recently updated RUNNING jobs."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=600)

    # Enqueue and mark running
    job = make_job("active-job")
    queue.enqueue(job)
    queue.mark_running("active-job")

    # Detect stalled jobs (should find none since job is recent)
    stalled = detector.detect_stalled_jobs()

    assert len(stalled) == 0


def test_detect_stalled_jobs_ignores_non_running_jobs(tmp_path: Path):
    """Test that detector only checks RUNNING jobs."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=60)

    # Create old PENDING job
    job = make_job("pending-job")
    queue.enqueue(job)

    # Manually set updated_at to old timestamp
    old_time = (datetime.utcnow() - timedelta(seconds=120)).isoformat()
    with queue._lock:
        queue._conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
            (old_time, "pending-job"),
        )
        queue._conn.commit()

    # Detect stalled jobs (should find none since job is not RUNNING)
    stalled = detector.detect_stalled_jobs()

    assert len(stalled) == 0


def test_recover_stalled_job_resets_to_pending(tmp_path: Path):
    """Test that recovery resets stalled job to PENDING."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=60)

    # Create stalled job
    job = make_job("stalled-job")
    queue.enqueue(job)
    queue.mark_running("stalled-job")

    # Recover the job
    detector.recover_stalled_job("stalled-job")

    # Verify job is now PENDING
    job_data = queue.get_job("stalled-job")
    assert job_data["status"] == JobStatus.PENDING.value


def test_recover_stalled_job_with_retry_delay(tmp_path: Path):
    """Test that recovered jobs are scheduled with retry delay."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=60)

    # Create stalled job
    job = make_job("stalled-job")
    queue.enqueue(job)
    queue.mark_running("stalled-job")

    # Recover the job
    detector.recover_stalled_job("stalled-job")

    # Verify job has scheduled_for set (indicating delay)
    job_data = queue.get_job("stalled-job")
    assert job_data["scheduled_for"] is not None


def test_deadlock_timeout_configurable(tmp_path: Path):
    """Test that deadlock timeout is configurable."""
    queue = JobQueue(tmp_path / "queue.db")

    # Create detector with custom timeout
    detector_short = DeadlockDetector(queue, timeout_seconds=30)
    assert detector_short._timeout_seconds == 30

    detector_long = DeadlockDetector(queue, timeout_seconds=1200)
    assert detector_long._timeout_seconds == 1200


def test_detect_multiple_stalled_jobs(tmp_path: Path):
    """Test detection of multiple stalled jobs."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=60)

    # Create multiple stalled jobs
    for i in range(3):
        job = make_job(f"stalled-job-{i}")
        queue.enqueue(job)
        queue.mark_running(f"stalled-job-{i}")

        # Set old timestamp
        old_time = (datetime.utcnow() - timedelta(seconds=120)).isoformat()
        with queue._lock:
            queue._conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                (old_time, f"stalled-job-{i}"),
            )
            queue._conn.commit()

    # Detect all stalled jobs
    stalled = detector.detect_stalled_jobs()

    assert len(stalled) == 3
    assert all(f"stalled-job-{i}" in stalled for i in range(3))


def test_no_false_positives_on_empty_queue(tmp_path: Path):
    """Test that detector doesn't report false positives on empty queue."""
    queue = JobQueue(tmp_path / "queue.db")
    detector = DeadlockDetector(queue, timeout_seconds=60)

    # Detect stalled jobs on empty queue
    stalled = detector.detect_stalled_jobs()

    assert len(stalled) == 0

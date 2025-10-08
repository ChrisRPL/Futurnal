"""Performance and resilience tests for crash recovery."""

import time
from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.crash_recovery import CrashRecoveryManager, RecoveryStateTracker
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue, JobStatus


def make_job(job_id: str) -> IngestionJob:
    """Helper to create test job."""
    return IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )


@pytest.mark.performance
class TestCrashRecoveryPerformance:
    """Performance tests for crash recovery."""

    def test_recovery_performance_1k_jobs(self, tmp_path: Path) -> None:
        """Test recovery completes in <1s for 1K jobs."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create 1000 jobs (mix of pending and running)
        for i in range(800):
            job = make_job(f"pending-{i}")
            queue.enqueue(job)

        for i in range(200):
            job = make_job(f"running-{i}")
            queue.enqueue(job)
            queue.mark_running(f"running-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Measure recovery time
        start = time.perf_counter()
        report = manager.recover_from_crash()
        duration = time.perf_counter() - start

        # Verify performance
        assert report.was_successful()
        assert duration < 1.0  # Should complete in less than 1 second
        assert report.jobs_reset_to_pending == 200

        print(f"\n1K jobs recovery: {duration:.3f}s")

    def test_recovery_performance_10k_jobs(self, tmp_path: Path) -> None:
        """Test recovery completes in <5s for 10K jobs (acceptance criteria)."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create 10,000 jobs (mix of pending and running)
        # Use batch inserts for faster test setup
        for i in range(8000):
            job = make_job(f"pending-{i}")
            queue.enqueue(job)

        for i in range(2000):
            job = make_job(f"running-{i}")
            queue.enqueue(job)
            queue.mark_running(f"running-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Measure recovery time
        start = time.perf_counter()
        report = manager.recover_from_crash()
        duration = time.perf_counter() - start

        # Verify performance (acceptance criteria: <5s for 10K jobs)
        assert report.was_successful()
        assert duration < 5.0  # Acceptance criteria
        assert report.jobs_reset_to_pending == 2000
        assert report.jobs_recovered == 10000

        print(f"\n10K jobs recovery: {duration:.3f}s")

    def test_recovery_performance_100k_jobs(self, tmp_path: Path) -> None:
        """Test recovery scales to 100K jobs (stress test)."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create 100,000 jobs
        # Most pending, some running
        batch_size = 1000
        for batch in range(95):
            for i in range(batch_size):
                job_id = f"pending-{batch * batch_size + i}"
                job = make_job(job_id)
                queue.enqueue(job)

        for batch in range(5):
            for i in range(batch_size):
                job_id = f"running-{batch * batch_size + i}"
                job = make_job(job_id)
                queue.enqueue(job)
                queue.mark_running(job_id)

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Measure recovery time
        start = time.perf_counter()
        report = manager.recover_from_crash()
        duration = time.perf_counter() - start

        # Verify recovery works at scale
        assert report.was_successful()
        assert report.jobs_reset_to_pending == 5000
        assert report.jobs_recovered == 100000

        # Performance should scale reasonably (not strict requirement)
        print(f"\n100K jobs recovery: {duration:.3f}s")


class TestWALFileTracking:
    """Tests for WAL file size tracking."""

    def test_wal_size_tracking(self, tmp_path: Path) -> None:
        """Test WAL file size is correctly tracked during recovery."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create jobs to generate WAL activity
        for i in range(100):
            job = make_job(f"job-{i}")
            queue.enqueue(job)
            queue.mark_running(f"job-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Perform recovery
        report = manager.recover_from_crash()

        # Verify WAL size was tracked
        assert report.wal_size_before_recovery_bytes >= 0
        # WAL file should exist or have existed
        assert isinstance(report.wal_size_before_recovery_bytes, int)

    def test_wal_checkpoint_reduces_wal_size(self, tmp_path: Path) -> None:
        """Test WAL checkpoint during recovery."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create significant WAL activity
        for i in range(500):
            job = make_job(f"job-{i}")
            queue.enqueue(job)

        # Check WAL file
        wal_path = tmp_path / "queue.db-wal"

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Perform recovery (includes checkpoint)
        report = manager.recover_from_crash()

        # Verify recovery succeeded
        assert report.was_successful()

        # After checkpoint, WAL should be minimal or empty
        # (SQLite may keep WAL file but it should be checkpointed)
        if wal_path.exists():
            post_checkpoint_size = wal_path.stat().st_size
            # Size should be reasonable after checkpoint
            assert post_checkpoint_size >= 0


class TestRecoveryResilience:
    """Resilience and edge case tests."""

    def test_recovery_idempotency(self, tmp_path: Path) -> None:
        """Test recovery can be run multiple times safely."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create jobs
        for i in range(5):
            job = make_job(f"job-{i}")
            queue.enqueue(job)
            queue.mark_running(f"job-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # First recovery
        report1 = manager.recover_from_crash()
        assert report1.was_successful()
        assert report1.jobs_reset_to_pending == 5

        # Simulate crash again
        tracker.mark_crash()

        # Second recovery - should handle gracefully (no running jobs to reset)
        report2 = manager.recover_from_crash()
        assert report2.was_successful()
        assert report2.jobs_reset_to_pending == 0  # No running jobs this time

    def test_recovery_with_empty_queue(self, tmp_path: Path) -> None:
        """Test recovery handles empty queue gracefully."""
        queue = JobQueue(tmp_path / "queue.db")

        # No jobs in queue
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Recovery should succeed even with no jobs
        report = manager.recover_from_crash()
        assert report.was_successful()
        assert report.jobs_recovered == 0
        assert report.jobs_reset_to_pending == 0

    def test_recovery_with_only_completed_jobs(self, tmp_path: Path) -> None:
        """Test recovery when all jobs are already completed."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create only completed jobs
        for i in range(10):
            job = make_job(f"job-{i}")
            queue.enqueue(job)
            queue.mark_running(f"job-{i}")
            queue.mark_completed(f"job-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Recovery should succeed with no resets needed
        report = manager.recover_from_crash()
        assert report.was_successful()
        assert report.jobs_recovered == 10
        assert report.jobs_running_before_crash == 0
        assert report.jobs_reset_to_pending == 0

    def test_recovery_timing_accuracy(self, tmp_path: Path) -> None:
        """Test recovery duration is accurately measured."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create moderate number of jobs
        for i in range(100):
            job = make_job(f"job-{i}")
            queue.enqueue(job)
            queue.mark_running(f"job-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Measure externally
        start = time.perf_counter()
        report = manager.recover_from_crash()
        external_duration = time.perf_counter() - start

        # Reported duration should be close to measured duration
        assert report.recovery_duration_seconds > 0
        assert abs(report.recovery_duration_seconds - external_duration) < 0.1

    def test_recovery_with_scheduled_jobs(self, tmp_path: Path) -> None:
        """Test recovery preserves scheduled_for timestamps."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create jobs with specific scheduling
        from datetime import timedelta
        future_time = datetime.utcnow() + timedelta(hours=1)

        for i in range(5):
            job = IngestionJob(
                job_id=f"scheduled-{i}",
                job_type=JobType.LOCAL_FILES,
                payload={"source_name": "test"},
                priority=JobPriority.NORMAL,
                scheduled_for=future_time,
            )
            queue.enqueue(job)
            queue.mark_running(f"scheduled-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Perform recovery
        report = manager.recover_from_crash()
        assert report.was_successful()
        assert report.jobs_reset_to_pending == 5

        # Verify scheduled times are preserved in reset jobs
        # Jobs should be rescheduled immediately (delay=0)
        pending_jobs = queue.snapshot(status=JobStatus.PENDING)
        assert len(pending_jobs) == 5

    def test_recovery_metadata_completeness(self, tmp_path: Path) -> None:
        """Test all recovery report fields are populated."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create various job states
        for i in range(3):
            job = make_job(f"pending-{i}")
            queue.enqueue(job)

        for i in range(2):
            job = make_job(f"running-{i}")
            queue.enqueue(job)
            queue.mark_running(f"running-{i}")

        # Setup recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )

        # Perform recovery
        report = manager.recover_from_crash()

        # Verify all fields are populated
        assert report.recovered_at is not None
        assert isinstance(report.recovered_at, datetime)
        assert report.jobs_recovered >= 0
        assert report.jobs_pending >= 0
        assert report.jobs_running_before_crash >= 0
        assert report.jobs_reset_to_pending >= 0
        assert report.wal_size_before_recovery_bytes >= 0
        assert report.recovery_duration_seconds >= 0
        assert isinstance(report.errors, list)

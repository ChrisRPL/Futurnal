"""Unit tests for crash recovery components."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.crash_recovery import (
    CrashRecoveryManager,
    CrashRecoveryReport,
    RecoveryStateTracker,
)
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue, JobStatus
from futurnal.privacy.audit import AuditLogger


def make_job(job_id: str) -> IngestionJob:
    """Helper to create test job."""
    return IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "test"},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )


class TestRecoveryStateTracker:
    """Test suite for RecoveryStateTracker."""

    def test_recovery_tracker_marker_lifecycle(self, tmp_path: Path) -> None:
        """Test creating, checking, and clearing recovery marker."""
        tracker = RecoveryStateTracker(tmp_path)

        # Initially no marker
        assert not tracker.is_recovering_from_crash()
        assert tracker.get_crash_info() is None

        # Create marker
        tracker.mark_crash()
        assert tracker.is_recovering_from_crash()

        # Clear marker
        tracker.clear_recovery_marker()
        assert not tracker.is_recovering_from_crash()

    def test_recovery_tracker_crash_info(self, tmp_path: Path) -> None:
        """Test crash metadata storage and retrieval."""
        tracker = RecoveryStateTracker(tmp_path)

        # Mark crash
        tracker.mark_crash()

        # Get crash info
        info = tracker.get_crash_info()
        assert info is not None
        assert "crashed_at" in info
        assert "pid" in info
        assert info["pid"] == os.getpid()

        # Verify timestamp format
        datetime.fromisoformat(info["crashed_at"])

    def test_recovery_tracker_idempotent_clear(self, tmp_path: Path) -> None:
        """Test clearing marker multiple times is safe."""
        tracker = RecoveryStateTracker(tmp_path)

        # Clear when no marker exists - should not error
        tracker.clear_recovery_marker()
        assert not tracker.is_recovering_from_crash()

        # Create and clear
        tracker.mark_crash()
        tracker.clear_recovery_marker()

        # Clear again - should not error
        tracker.clear_recovery_marker()
        assert not tracker.is_recovering_from_crash()

    def test_recovery_tracker_corrupted_marker(self, tmp_path: Path) -> None:
        """Test handling of corrupted marker file."""
        tracker = RecoveryStateTracker(tmp_path)

        # Create corrupted marker
        marker_path = tmp_path / ".orchestrator_recovery"
        marker_path.write_text("invalid json {{{")

        # Should detect marker exists
        assert tracker.is_recovering_from_crash()

        # Should handle corruption gracefully
        info = tracker.get_crash_info()
        assert info is None  # Returns None on JSON error


class TestCrashRecoveryReport:
    """Test suite for CrashRecoveryReport."""

    def test_recovery_report_success_check(self) -> None:
        """Test report success validation."""
        # Successful recovery
        report = CrashRecoveryReport(
            recovered_at=datetime.utcnow(),
            jobs_recovered=10,
            jobs_pending=5,
            jobs_running_before_crash=3,
            jobs_reset_to_pending=3,
            wal_size_before_recovery_bytes=1024,
            recovery_duration_seconds=0.5,
            errors=[],
        )
        assert report.was_successful()

        # Failed recovery
        report_with_errors = CrashRecoveryReport(
            recovered_at=datetime.utcnow(),
            jobs_recovered=10,
            jobs_pending=5,
            jobs_running_before_crash=3,
            jobs_reset_to_pending=3,
            wal_size_before_recovery_bytes=1024,
            recovery_duration_seconds=0.5,
            errors=["Database corruption detected"],
        )
        assert not report_with_errors.was_successful()


class TestCrashRecoveryManager:
    """Test suite for CrashRecoveryManager."""

    def test_crash_recovery_no_running_jobs(self, tmp_path: Path) -> None:
        """Test recovery when no jobs were running (clean shutdown scenario)."""
        queue = JobQueue(tmp_path / "queue.db")
        audit_logger = AuditLogger(tmp_path / "audit")

        # Create some completed and pending jobs
        job1 = make_job("completed-1")
        job2 = make_job("pending-1")

        queue.enqueue(job1)
        queue.mark_running("completed-1")
        queue.mark_completed("completed-1")

        queue.enqueue(job2)

        # Simulate crash detection
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        # Perform recovery
        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
            audit_logger=audit_logger,
        )
        report = manager.recover_from_crash()

        # Verify recovery report
        assert report.was_successful()
        assert report.jobs_recovered == 2
        assert report.jobs_running_before_crash == 0
        assert report.jobs_reset_to_pending == 0
        assert report.recovery_duration_seconds > 0

        # Marker should be cleared
        assert not tracker.is_recovering_from_crash()

    def test_crash_recovery_resets_running_jobs(self, tmp_path: Path) -> None:
        """Test core recovery logic: RUNNING → PENDING transitions."""
        queue = JobQueue(tmp_path / "queue.db")
        audit_logger = AuditLogger(tmp_path / "audit")

        # Create jobs in various states
        jobs = [
            ("pending-1", JobStatus.PENDING),
            ("running-1", JobStatus.RUNNING),
            ("running-2", JobStatus.RUNNING),
            ("succeeded-1", JobStatus.SUCCEEDED),
        ]

        for job_id, status in jobs:
            job = make_job(job_id)
            queue.enqueue(job)

            if status == JobStatus.RUNNING:
                queue.mark_running(job_id)
            elif status == JobStatus.SUCCEEDED:
                queue.mark_running(job_id)
                queue.mark_completed(job_id)

        # Simulate crash
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        # Perform recovery
        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
            audit_logger=audit_logger,
        )
        report = manager.recover_from_crash()

        # Verify recovery
        assert report.was_successful()
        assert report.jobs_running_before_crash == 2
        assert report.jobs_reset_to_pending == 2

        # Verify all running jobs are now pending
        running_jobs = queue.snapshot(status=JobStatus.RUNNING)
        assert len(running_jobs) == 0

        # Verify pending count increased
        pending_jobs = queue.snapshot(status=JobStatus.PENDING)
        assert len(pending_jobs) == 3  # 1 original + 2 reset

    def test_crash_recovery_wal_checkpoint(self, tmp_path: Path) -> None:
        """Test WAL checkpoint execution during recovery."""
        queue = JobQueue(tmp_path / "queue.db")

        # Add some jobs to create WAL activity
        for i in range(10):
            job = make_job(f"job-{i}")
            queue.enqueue(job)

        # Check WAL file exists
        wal_path = tmp_path / "queue.db-wal"
        if wal_path.exists():
            initial_wal_size = wal_path.stat().st_size
        else:
            initial_wal_size = 0

        # Perform recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )
        report = manager.recover_from_crash()

        # Verify checkpoint was called
        assert report.was_successful()
        assert report.wal_size_before_recovery_bytes >= 0

    def test_crash_recovery_integrity_validation(self, tmp_path: Path) -> None:
        """Test database integrity verification during recovery."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create normal jobs
        job = make_job("test-job")
        queue.enqueue(job)

        # Perform recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )
        report = manager.recover_from_crash()

        # Integrity check should pass
        assert report.was_successful()
        assert len(report.errors) == 0

    def test_crash_recovery_audit_logging(self, tmp_path: Path) -> None:
        """Test audit event recording during recovery."""
        queue = JobQueue(tmp_path / "queue.db")
        audit_logger = AuditLogger(tmp_path / "audit")

        # Create and reset some jobs
        job = make_job("running-job")
        queue.enqueue(job)
        queue.mark_running("running-job")

        # Perform recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
            audit_logger=audit_logger,
        )
        report = manager.recover_from_crash()

        # Verify audit event was logged
        audit_path = audit_logger.output_dir / audit_logger.filename
        assert audit_path.exists()

        # Read and parse audit log
        content = audit_path.read_text()
        event = json.loads(content.strip())

        # Verify event details
        assert event["source"] == "crash_recovery"
        assert event["action"] == "recover_from_crash"
        assert event["status"] == "succeeded"
        assert "jobs_recovered" in event["metadata"]
        assert "jobs_reset" in event["metadata"]
        assert event["metadata"]["jobs_reset"] == 1

    def test_crash_recovery_report_metrics(self, tmp_path: Path) -> None:
        """Test all recovery metrics are populated correctly."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create mixed job states
        for i in range(5):
            job = make_job(f"pending-{i}")
            queue.enqueue(job)

        for i in range(3):
            job = make_job(f"running-{i}")
            queue.enqueue(job)
            queue.mark_running(f"running-{i}")

        # Perform recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )
        report = manager.recover_from_crash()

        # Verify all metrics
        assert report.recovered_at is not None
        assert isinstance(report.recovered_at, datetime)
        assert report.jobs_recovered == 8
        assert report.jobs_pending == 8  # 5 original + 3 reset
        assert report.jobs_running_before_crash == 3
        assert report.jobs_reset_to_pending == 3
        assert report.wal_size_before_recovery_bytes >= 0
        assert report.recovery_duration_seconds > 0
        assert isinstance(report.errors, list)

    def test_crash_recovery_preserves_job_data(self, tmp_path: Path) -> None:
        """Test no data loss during recovery."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create jobs with specific payloads
        job_data = {
            "job-1": {"source_name": "test1", "custom_field": "value1"},
            "job-2": {"source_name": "test2", "custom_field": "value2"},
        }

        for job_id, payload in job_data.items():
            job = IngestionJob(
                job_id=job_id,
                job_type=JobType.LOCAL_FILES,
                payload=payload,
                priority=JobPriority.NORMAL,
                scheduled_for=datetime.utcnow(),
            )
            queue.enqueue(job)
            queue.mark_running(job_id)

        # Perform recovery
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )
        report = manager.recover_from_crash()

        # Verify all jobs still exist with same data
        assert report.jobs_recovered == 2

        for job_id, expected_payload in job_data.items():
            job_dict = queue.get_job(job_id)
            assert job_dict is not None
            assert job_dict["payload"] == expected_payload

    def test_crash_recovery_without_audit_logger(self, tmp_path: Path) -> None:
        """Test recovery works without audit logger."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create job
        job = make_job("test-job")
        queue.enqueue(job)

        # Perform recovery without audit logger
        tracker = RecoveryStateTracker(tmp_path)
        tracker.mark_crash()

        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
            audit_logger=None,  # No audit logger
        )
        report = manager.recover_from_crash()

        # Should still succeed
        assert report.was_successful()

    def test_reset_interrupted_jobs_count(self, tmp_path: Path) -> None:
        """Test accurate counting of reset jobs."""
        queue = JobQueue(tmp_path / "queue.db")

        # Create 5 running jobs
        for i in range(5):
            job = make_job(f"running-{i}")
            queue.enqueue(job)
            queue.mark_running(f"running-{i}")

        # Create manager and reset
        manager = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=tmp_path,
        )
        count = manager._reset_interrupted_jobs()

        # Verify count
        assert count == 5

        # Verify no running jobs remain
        running_jobs = queue.snapshot(status=JobStatus.RUNNING)
        assert len(running_jobs) == 0

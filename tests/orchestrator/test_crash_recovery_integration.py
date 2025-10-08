"""Integration tests for orchestrator crash recovery."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.crash_recovery import CrashRecoveryManager
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue, JobStatus
from futurnal.orchestrator.scheduler import IngestionOrchestrator
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


class TestOrchestratorCrashRecovery:
    """Integration tests for orchestrator crash recovery lifecycle."""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_orchestrator_startup_with_recovery_marker(self, tmp_path: Path, event_loop) -> None:
        """Test orchestrator automatically recovers on startup when marker exists."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")

        # Create jobs in running state (simulating crash)
        for i in range(3):
            job = make_job(f"crashed-job-{i}")
            queue.enqueue(job)
            queue.mark_running(f"crashed-job-{i}")

        # Create recovery marker to simulate crash
        recovery_tracker_workspace = tmp_path / "workspace"
        recovery_tracker_workspace.mkdir(parents=True, exist_ok=True)
        marker_file = recovery_tracker_workspace / ".orchestrator_recovery"
        marker_file.write_text('{"crashed_at": "2024-01-01T00:00:00", "pid": 12345}')

        # Initialize orchestrator - should trigger recovery
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(recovery_tracker_workspace),
            loop=event_loop,
        )

        # Verify recovery occurred
        # 1. All running jobs should be reset to pending
        running_jobs = queue.snapshot(status=JobStatus.RUNNING)
        assert len(running_jobs) == 0

        # 2. Pending jobs should include the reset ones
        pending_jobs = queue.snapshot(status=JobStatus.PENDING)
        assert len(pending_jobs) == 3

        # 3. Recovery marker should be cleared
        assert not marker_file.exists()

    def test_orchestrator_startup_without_marker(self, tmp_path: Path, event_loop) -> None:
        """Test normal startup when no crash marker exists."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")

        # Create some normal pending jobs
        for i in range(2):
            job = make_job(f"normal-job-{i}")
            queue.enqueue(job)

        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Initialize orchestrator - should NOT trigger recovery
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify no recovery occurred
        pending_jobs = queue.snapshot(status=JobStatus.PENDING)
        assert len(pending_jobs) == 2

    def test_orchestrator_sets_marker_on_start(self, tmp_path: Path, event_loop) -> None:
        """Test orchestrator creates recovery marker on start."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Marker should not exist initially
        marker_file = workspace / ".orchestrator_recovery"
        assert not marker_file.exists()

        # Start orchestrator
        orchestrator.start()

        # Marker should now exist
        assert marker_file.exists()

        # Cleanup
        event_loop.run_until_complete(orchestrator.shutdown())

    def test_orchestrator_clears_marker_on_shutdown(self, tmp_path: Path, event_loop) -> None:
        """Test orchestrator clears recovery marker on graceful shutdown."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Start and verify marker exists
        orchestrator.start()
        marker_file = workspace / ".orchestrator_recovery"
        assert marker_file.exists()

        # Shutdown and verify marker cleared
        event_loop.run_until_complete(orchestrator.shutdown())
        assert not marker_file.exists()

    def test_recovery_with_mixed_job_states(self, tmp_path: Path, event_loop) -> None:
        """Test recovery correctly handles mixed job states."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")

        # Create jobs in various states
        # Pending jobs
        for i in range(2):
            job = make_job(f"pending-{i}")
            queue.enqueue(job)

        # Running jobs (to be reset)
        for i in range(3):
            job = make_job(f"running-{i}")
            queue.enqueue(job)
            queue.mark_running(f"running-{i}")

        # Succeeded jobs
        for i in range(1):
            job = make_job(f"succeeded-{i}")
            queue.enqueue(job)
            queue.mark_running(f"succeeded-{i}")
            queue.mark_completed(f"succeeded-{i}")

        # Failed jobs
        for i in range(1):
            job = make_job(f"failed-{i}")
            queue.enqueue(job)
            queue.mark_running(f"failed-{i}")
            queue.mark_failed(f"failed-{i}")

        # Simulate crash
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        marker_file = workspace / ".orchestrator_recovery"
        marker_file.write_text('{"crashed_at": "2024-01-01T00:00:00", "pid": 12345}')

        # Initialize orchestrator - triggers recovery
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify state after recovery
        assert len(queue.snapshot(status=JobStatus.RUNNING)) == 0
        assert len(queue.snapshot(status=JobStatus.PENDING)) == 5  # 2 original + 3 reset
        assert len(queue.snapshot(status=JobStatus.SUCCEEDED)) == 1
        assert len(queue.snapshot(status=JobStatus.FAILED)) == 1

    def test_recovery_preserves_job_data(self, tmp_path: Path, event_loop) -> None:
        """Test recovery doesn't lose or corrupt job data."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")

        # Create jobs with specific payloads
        test_jobs = [
            ("job-1", {"source_name": "source1", "custom": "data1"}),
            ("job-2", {"source_name": "source2", "custom": "data2"}),
            ("job-3", {"source_name": "source3", "custom": "data3"}),
        ]

        for job_id, payload in test_jobs:
            job = IngestionJob(
                job_id=job_id,
                job_type=JobType.LOCAL_FILES,
                payload=payload,
                priority=JobPriority.NORMAL,
                scheduled_for=datetime.utcnow(),
            )
            queue.enqueue(job)
            queue.mark_running(job_id)

        # Simulate crash
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        marker_file = workspace / ".orchestrator_recovery"
        marker_file.write_text('{"crashed_at": "2024-01-01T00:00:00", "pid": 12345}')

        # Initialize orchestrator - triggers recovery
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify all job data preserved
        for job_id, expected_payload in test_jobs:
            job_dict = queue.get_job(job_id)
            assert job_dict is not None
            assert job_dict["payload"] == expected_payload

    def test_multiple_recovery_cycles(self, tmp_path: Path, event_loop) -> None:
        """Test successive crashes and recoveries are handled correctly."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # First crash cycle
        job1 = make_job("job-1")
        queue.enqueue(job1)
        queue.mark_running("job-1")

        # Simulate first crash
        marker_file = workspace / ".orchestrator_recovery"
        marker_file.write_text('{"crashed_at": "2024-01-01T00:00:00", "pid": 11111}')

        # First recovery
        orchestrator1 = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify first recovery
        assert len(queue.snapshot(status=JobStatus.PENDING)) == 1
        assert not marker_file.exists()

        # Second crash cycle - add another job and mark running
        job2 = make_job("job-2")
        queue.enqueue(job2)
        queue.mark_running("job-2")

        # Simulate second crash
        marker_file.write_text('{"crashed_at": "2024-01-01T01:00:00", "pid": 22222}')

        # Second recovery
        orchestrator2 = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify second recovery
        assert len(queue.snapshot(status=JobStatus.PENDING)) == 2
        assert len(queue.snapshot(status=JobStatus.RUNNING)) == 0
        assert not marker_file.exists()

    def test_recovery_audit_logging(self, tmp_path: Path, event_loop) -> None:
        """Test recovery events are properly audit logged."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create running jobs to trigger meaningful recovery
        for i in range(2):
            job = make_job(f"job-{i}")
            queue.enqueue(job)
            queue.mark_running(f"job-{i}")

        # Simulate crash
        marker_file = workspace / ".orchestrator_recovery"
        marker_file.write_text('{"crashed_at": "2024-01-01T00:00:00", "pid": 12345}')

        # Initialize orchestrator - triggers recovery with audit logging
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify audit events were logged
        audit_path = workspace / "audit" / "audit.log"
        assert audit_path.exists()

        # Read and parse audit log
        content = audit_path.read_text()
        event = json.loads(content.strip())

        # Verify event details
        assert event["source"] == "crash_recovery"
        assert event["action"] == "recover_from_crash"
        assert event["status"] == "succeeded"
        assert "jobs_recovered" in event["metadata"]
        assert event["metadata"]["jobs_reset"] == 2

    def test_orchestrator_lifecycle_without_crash(self, tmp_path: Path, event_loop) -> None:
        """Test normal orchestrator lifecycle without any crashes."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create normal jobs
        for i in range(3):
            job = make_job(f"job-{i}")
            queue.enqueue(job)

        # Initialize orchestrator
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Start
        orchestrator.start()
        marker_file = workspace / ".orchestrator_recovery"
        assert marker_file.exists()

        # Shutdown gracefully
        event_loop.run_until_complete(orchestrator.shutdown())
        assert not marker_file.exists()

        # Restart - no recovery should happen
        orchestrator2 = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # All jobs should still be pending
        pending_jobs = queue.snapshot(status=JobStatus.PENDING)
        assert len(pending_jobs) == 3

    def test_recovery_with_custom_crash_recovery_manager(self, tmp_path: Path, event_loop) -> None:
        """Test orchestrator accepts custom crash recovery manager (dependency injection)."""
        # Setup
        queue = JobQueue(tmp_path / "queue.db")
        state_store = StateStore(tmp_path / "state.db")
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(workspace / "audit")

        # Create custom crash recovery manager
        custom_recovery = CrashRecoveryManager(
            job_queue=queue,
            workspace_dir=workspace,
            audit_logger=audit_logger,
        )

        # Create job and simulate crash
        job = make_job("test-job")
        queue.enqueue(job)
        queue.mark_running("test-job")

        marker_file = workspace / ".orchestrator_recovery"
        marker_file.write_text('{"crashed_at": "2024-01-01T00:00:00", "pid": 12345}')

        # Initialize orchestrator with custom manager
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
            crash_recovery=custom_recovery,
        )

        # Verify recovery occurred using custom manager
        assert len(queue.snapshot(status=JobStatus.RUNNING)) == 0
        assert len(queue.snapshot(status=JobStatus.PENDING)) == 1
        assert not marker_file.exists()

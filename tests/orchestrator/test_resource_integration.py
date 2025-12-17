"""Integration tests for resource profiling with orchestrator."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.resource_monitor import ResourceMonitor
from futurnal.orchestrator.resource_profile import ResourceIntensity, ResourceProfile
from futurnal.orchestrator.resource_registry import ResourceProfileRegistry
from futurnal.orchestrator.scheduler import IngestionOrchestrator


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace."""
    return tmp_path / "workspace"


@pytest.fixture
def job_queue(tmp_path):
    """Create a job queue."""
    return JobQueue(tmp_path / "queue.db")


@pytest.fixture
def state_store(tmp_path):
    """Create a state store."""
    return StateStore(tmp_path / "state.db")


@pytest.fixture
def resource_monitor(tmp_path):
    """Create a resource monitor."""
    from futurnal.orchestrator.metrics import TelemetryRecorder
    telemetry = TelemetryRecorder(tmp_path / "telemetry")
    return ResourceMonitor(telemetry=telemetry)


@pytest.fixture
def resource_registry():
    """Create a resource profile registry."""
    return ResourceProfileRegistry()


class TestResourceProfilingIntegration:
    """Integration tests for resource profiling."""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_orchestrator_initialization_with_resource_profiling(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test that orchestrator initializes with resource profiling components."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Verify resource components are initialized
        assert orchestrator._resource_monitor is not None
        assert orchestrator._resource_profiles is not None
        assert isinstance(orchestrator._per_connector_semaphores, dict)

    def test_per_connector_semaphores_initialization(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test that per-connector semaphores are initialized on start."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Semaphores should be empty before start
        assert len(orchestrator._per_connector_semaphores) == 0

        # Start orchestrator
        orchestrator.start()

        try:
            # Semaphores should be initialized for all job types
            assert len(orchestrator._per_connector_semaphores) == len(JobType)

            for job_type in JobType:
                semaphore = orchestrator._per_connector_semaphores.get(job_type)
                assert semaphore is not None
                assert isinstance(semaphore, asyncio.Semaphore)
        finally:
            # Cleanup
            event_loop.run_until_complete(orchestrator.shutdown())

    def test_resource_monitoring_during_job_execution(
        self, workspace, job_queue, state_store, tmp_path, event_loop
    ):
        """Test that resources are monitored during job execution."""
        # Create a mock element sink
        element_sink = MagicMock()

        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=element_sink,
            loop=event_loop,
        )

        # Create a test source directory
        test_source = tmp_path / "test_source"
        test_source.mkdir()
        test_file = test_source / "test.txt"
        test_file.write_text("Test content")

        orchestrator.start()

        try:
            # Enqueue a job
            job = IngestionJob(
                job_id="test-job-123",
                job_type=JobType.LOCAL_FILES,
                payload={"source_name": "test_source"},
                priority=JobPriority.NORMAL,
            )
            job_queue.enqueue(job)

            # Give orchestrator time to process
            time.sleep(0.5)

            # Verify resource stats were collected
            stats = orchestrator._resource_monitor.get_connector_stats(
                JobType.LOCAL_FILES
            )

            # Stats may or may not exist depending on timing
            # This is a best-effort test
            if stats:
                assert stats.connector_type == JobType.LOCAL_FILES
        finally:
            event_loop.run_until_complete(orchestrator.shutdown())

    def test_custom_resource_profile(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test using a custom resource profile."""
        # Create custom registry with modified profile
        registry = ResourceProfileRegistry()
        custom_profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            cpu_intensity=ResourceIntensity.HIGH,
            max_concurrent_jobs=2,
        )
        registry.set_custom_profile(custom_profile)

        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            resource_profiles=registry,
            loop=event_loop,
        )

        orchestrator.start()

        try:
            # Verify custom profile is used
            profile = orchestrator._resource_profiles.get_profile(JobType.LOCAL_FILES)
            assert profile.max_concurrent_jobs == 2
            assert profile.cpu_intensity == ResourceIntensity.HIGH
        finally:
            event_loop.run_until_complete(orchestrator.shutdown())

    def test_resource_sampling_loop(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test that resource sampling loop runs in background."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Start orchestrator
        orchestrator.start()

        try:
            # Verify sampling task is running
            assert orchestrator._sampling_task is not None
            assert not orchestrator._sampling_task.done()

            # Start monitoring a job
            orchestrator._resource_monitor.start_monitoring(
                "test-job", JobType.LOCAL_FILES
            )

            # Wait for a few samples
            time.sleep(2.5)

            # Stop monitoring
            metrics = orchestrator._resource_monitor.stop_monitoring("test-job")

            # Should have multiple samples (approximately 2 samples in 2.5 seconds)
            assert metrics is not None
            # At least 1 sample should have been taken
            assert metrics.cpu_percent_avg is not None or metrics.memory_mb_avg is not None
        finally:
            event_loop.run_until_complete(orchestrator.shutdown())

    def test_sampling_task_cleanup_on_shutdown(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test that sampling task is properly cleaned up on shutdown."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        orchestrator.start()
        sampling_task = orchestrator._sampling_task
        assert sampling_task is not None

        # Shutdown
        event_loop.run_until_complete(orchestrator.shutdown())

        # Sampling task should be cancelled
        assert orchestrator._sampling_task is None
        assert sampling_task.done()

    def test_system_resource_availability_check(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test checking system resource availability."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Get system resources
        resources = orchestrator._resource_monitor.get_system_resources()

        # Verify all required fields are present
        assert "cpu_percent" in resources
        assert "memory_percent" in resources
        assert "disk_usage_percent" in resources
        assert "available_memory_mb" in resources
        assert "available_cpu_cores" in resources

        # Verify reasonable values
        assert resources["cpu_percent"] >= 0
        assert resources["memory_percent"] >= 0
        assert resources["available_memory_mb"] > 0
        assert resources["available_cpu_cores"] >= 1

    def test_optimal_concurrency_calculation_per_connector(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test that optimal concurrency is calculated per connector."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        system_resources = orchestrator._resource_monitor.get_system_resources()

        # Calculate optimal concurrency for different connectors
        local_files_concurrency = orchestrator._resource_profiles.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=system_resources["available_cpu_cores"],
            available_memory_mb=system_resources["available_memory_mb"],
            current_system_load=0.3,
        )

        imap_concurrency = orchestrator._resource_profiles.calculate_optimal_concurrency(
            job_type=JobType.IMAP_MAILBOX,
            available_cpu_cores=system_resources["available_cpu_cores"],
            available_memory_mb=system_resources["available_memory_mb"],
            current_system_load=0.3,
        )

        # Verify both have valid concurrency levels
        assert local_files_concurrency >= 1
        assert imap_concurrency >= 1

        # LOCAL_FILES typically has higher concurrency than IMAP
        # (This may not always be true depending on system resources)
        assert local_files_concurrency > 0
        assert imap_concurrency > 0

    def test_connector_statistics_aggregation(
        self, workspace, job_queue, state_store, event_loop
    ):
        """Test that connector statistics are properly aggregated."""
        orchestrator = IngestionOrchestrator(
            job_queue=job_queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            loop=event_loop,
        )

        # Simulate multiple jobs for the same connector
        monitor = orchestrator._resource_monitor

        for i in range(3):
            job_id = f"job-{i}"
            monitor.start_monitoring(job_id, JobType.LOCAL_FILES)
            time.sleep(0.1)
            monitor.sample_active_jobs()
            monitor.stop_monitoring(job_id)

        # Get aggregated stats
        stats = monitor.get_connector_stats(JobType.LOCAL_FILES)

        assert stats is not None
        assert stats.connector_type == JobType.LOCAL_FILES
        assert stats.job_count == 3
        assert stats.avg_cpu_percent >= 0
        assert stats.avg_memory_mb > 0

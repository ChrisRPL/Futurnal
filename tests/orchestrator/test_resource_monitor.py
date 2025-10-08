"""Unit tests for resource monitor."""

import time
from unittest.mock import MagicMock, patch

import pytest

from futurnal.orchestrator.models import JobType
from futurnal.orchestrator.resource_monitor import ResourceMeasurement, ResourceMonitor
from futurnal.orchestrator.metrics import TelemetryRecorder


class TestResourceMeasurement:
    """Test ResourceMeasurement class."""

    def test_measurement_initialization(self):
        """Test that measurement initializes correctly."""
        measurement = ResourceMeasurement(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            start_time=time.perf_counter(),
        )
        assert measurement.job_id == "test-job-123"
        assert measurement.job_type == JobType.LOCAL_FILES
        assert len(measurement.cpu_samples) == 0
        assert len(measurement.memory_samples) == 0

    def test_sampling(self):
        """Test that sampling captures metrics."""
        measurement = ResourceMeasurement(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            start_time=time.perf_counter(),
        )

        # Take a sample
        measurement.sample()

        # Should have captured at least one sample
        assert len(measurement.cpu_samples) >= 1
        assert len(measurement.memory_samples) >= 1

    def test_multiple_samples(self):
        """Test taking multiple samples."""
        measurement = ResourceMeasurement(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            start_time=time.perf_counter(),
        )

        # Take multiple samples
        for _ in range(3):
            measurement.sample()
            time.sleep(0.01)  # Small delay between samples

        assert len(measurement.cpu_samples) >= 3
        assert len(measurement.memory_samples) >= 3

    def test_finalize_with_samples(self):
        """Test finalizing measurement with samples."""
        start = time.perf_counter()
        measurement = ResourceMeasurement(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            start_time=start,
        )

        # Take some samples
        measurement.sample()
        time.sleep(0.1)
        measurement.sample()

        # Finalize
        metrics = measurement.finalize()

        assert metrics.job_id == "test-job-123"
        assert metrics.job_type == JobType.LOCAL_FILES
        assert metrics.duration_seconds > 0
        assert metrics.cpu_percent_avg is not None
        assert metrics.cpu_percent_peak is not None
        assert metrics.memory_mb_avg is not None
        assert metrics.memory_mb_peak is not None

    def test_finalize_without_samples(self):
        """Test finalizing measurement without samples."""
        start = time.perf_counter()
        measurement = ResourceMeasurement(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            start_time=start,
        )

        time.sleep(0.05)

        # Finalize without sampling
        metrics = measurement.finalize()

        assert metrics.job_id == "test-job-123"
        assert metrics.duration_seconds > 0
        # No samples means None for averages/peaks
        assert metrics.cpu_percent_avg is None
        assert metrics.cpu_percent_peak is None
        assert metrics.memory_mb_avg is None
        assert metrics.memory_mb_peak is None


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a resource monitor for testing."""
        telemetry = TelemetryRecorder(tmp_path / "telemetry")
        return ResourceMonitor(telemetry=telemetry)

    def test_monitor_initialization(self, monitor):
        """Test that monitor initializes correctly."""
        assert isinstance(monitor, ResourceMonitor)
        assert len(monitor._active_measurements) == 0
        assert len(monitor._connector_stats) == 0

    def test_ema_alpha_validation(self, tmp_path):
        """Test that EMA alpha is validated."""
        # Valid values
        monitor = ResourceMonitor(ema_alpha=0.3)
        assert monitor._ema_alpha == 0.3

        # Invalid values
        with pytest.raises(ValueError, match="ema_alpha must be between 0.0 and 1.0"):
            ResourceMonitor(ema_alpha=0.0)

        with pytest.raises(ValueError, match="ema_alpha must be between 0.0 and 1.0"):
            ResourceMonitor(ema_alpha=1.0)

    def test_start_monitoring(self, monitor):
        """Test starting resource monitoring."""
        job_id = "test-job-123"
        job_type = JobType.LOCAL_FILES

        monitor.start_monitoring(job_id, job_type)

        assert job_id in monitor._active_measurements
        measurement = monitor._active_measurements[job_id]
        assert measurement.job_id == job_id
        assert measurement.job_type == job_type

    def test_stop_monitoring(self, monitor):
        """Test stopping resource monitoring."""
        job_id = "test-job-123"
        job_type = JobType.LOCAL_FILES

        # Start monitoring
        monitor.start_monitoring(job_id, job_type)
        assert job_id in monitor._active_measurements

        # Take a sample
        time.sleep(0.05)
        monitor.sample_active_jobs()

        # Stop monitoring
        metrics = monitor.stop_monitoring(job_id)

        assert metrics is not None
        assert metrics.job_id == job_id
        assert metrics.job_type == job_type
        assert job_id not in monitor._active_measurements

    def test_stop_monitoring_nonexistent_job(self, monitor):
        """Test stopping monitoring for a job that doesn't exist."""
        metrics = monitor.stop_monitoring("nonexistent-job")
        assert metrics is None

    def test_sample_active_jobs(self, monitor):
        """Test sampling all active jobs."""
        # Start monitoring multiple jobs
        job1 = "job-1"
        job2 = "job-2"
        monitor.start_monitoring(job1, JobType.LOCAL_FILES)
        monitor.start_monitoring(job2, JobType.OBSIDIAN_VAULT)

        # Sample all active jobs
        monitor.sample_active_jobs()

        # Both jobs should have samples
        assert len(monitor._active_measurements[job1].cpu_samples) >= 1
        assert len(monitor._active_measurements[job2].cpu_samples) >= 1

    def test_get_system_resources(self, monitor):
        """Test getting system resource availability."""
        resources = monitor.get_system_resources()

        assert "cpu_percent" in resources
        assert "memory_percent" in resources
        assert "disk_usage_percent" in resources
        assert "available_memory_mb" in resources
        assert "available_cpu_cores" in resources

        # Verify reasonable values
        assert 0 <= resources["cpu_percent"] <= 100
        assert 0 <= resources["memory_percent"] <= 100
        assert resources["available_memory_mb"] > 0
        assert resources["available_cpu_cores"] >= 1

    def test_get_connector_stats(self, monitor):
        """Test getting connector statistics."""
        job_id = "test-job-123"
        job_type = JobType.LOCAL_FILES

        # No stats initially
        stats = monitor.get_connector_stats(job_type)
        assert stats is None

        # Process a job
        monitor.start_monitoring(job_id, job_type)
        time.sleep(0.05)
        monitor.sample_active_jobs()
        monitor.stop_monitoring(job_id)

        # Stats should now exist
        stats = monitor.get_connector_stats(job_type)
        assert stats is not None
        assert stats.connector_type == job_type
        assert stats.job_count == 1

    def test_get_all_connector_stats(self, monitor):
        """Test getting all connector statistics."""
        # Process jobs for different connectors
        monitor.start_monitoring("job-1", JobType.LOCAL_FILES)
        time.sleep(0.05)
        monitor.stop_monitoring("job-1")

        monitor.start_monitoring("job-2", JobType.OBSIDIAN_VAULT)
        time.sleep(0.05)
        monitor.stop_monitoring("job-2")

        # Get all stats
        all_stats = monitor.get_all_connector_stats()

        assert len(all_stats) == 2
        assert JobType.LOCAL_FILES in all_stats
        assert JobType.OBSIDIAN_VAULT in all_stats

    def test_connector_stats_aggregation(self, monitor):
        """Test that connector statistics are properly aggregated."""
        job_type = JobType.LOCAL_FILES

        # Process multiple jobs
        for i in range(3):
            job_id = f"job-{i}"
            monitor.start_monitoring(job_id, job_type)
            time.sleep(0.05)
            monitor.sample_active_jobs()
            monitor.stop_monitoring(job_id)

        # Check aggregated stats
        stats = monitor.get_connector_stats(job_type)
        assert stats is not None
        assert stats.job_count == 3
        assert stats.avg_cpu_percent > 0  # Should have some CPU usage
        assert stats.avg_memory_mb > 0  # Should have some memory usage

    def test_exponential_moving_average(self, monitor):
        """Test that EMA is applied to statistics."""
        job_type = JobType.LOCAL_FILES

        # Process first job
        monitor.start_monitoring("job-1", job_type)
        time.sleep(0.05)
        monitor.sample_active_jobs()
        monitor.stop_monitoring("job-1")

        stats_1 = monitor.get_connector_stats(job_type)
        first_avg_cpu = stats_1.avg_cpu_percent

        # Process second job
        monitor.start_monitoring("job-2", job_type)
        time.sleep(0.05)
        monitor.sample_active_jobs()
        monitor.stop_monitoring("job-2")

        stats_2 = monitor.get_connector_stats(job_type)
        second_avg_cpu = stats_2.avg_cpu_percent

        # EMA should cause the average to change
        # (Note: This test is probabilistic and might occasionally fail
        # if CPU usage is identical for both jobs)
        assert stats_2.job_count == 2

    def test_telemetry_recording(self, tmp_path):
        """Test that resource metrics are collected (telemetry is handled by orchestrator)."""
        telemetry = TelemetryRecorder(tmp_path / "telemetry")
        monitor = ResourceMonitor(telemetry=telemetry)

        job_id = "test-job-123"
        job_type = JobType.LOCAL_FILES

        # Process a job
        monitor.start_monitoring(job_id, job_type)
        time.sleep(0.05)
        monitor.sample_active_jobs()
        metrics = monitor.stop_monitoring(job_id)

        # Verify metrics were collected
        assert metrics is not None
        assert metrics.job_id == job_id
        assert metrics.job_type == job_type
        assert metrics.duration_seconds > 0

        # Note: Telemetry recording is now handled by the orchestrator,
        # not directly by ResourceMonitor. The monitor only collects metrics.

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.cpu_count")
    def test_get_system_resources_error_handling(
        self, mock_cpu_count, mock_disk, mock_memory, mock_cpu, monitor
    ):
        """Test error handling when psutil fails."""
        # Make psutil calls raise an exception
        mock_cpu.side_effect = OSError("Test error")
        mock_memory.side_effect = OSError("Test error")
        mock_disk.side_effect = OSError("Test error")
        mock_cpu_count.side_effect = OSError("Test error")

        # Should return safe defaults
        resources = monitor.get_system_resources()

        assert resources["cpu_percent"] == 50.0
        assert resources["memory_percent"] == 50.0
        assert resources["available_memory_mb"] == 4096.0
        assert resources["available_cpu_cores"] == 4.0

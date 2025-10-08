"""Unit tests for resource profile models."""

import pytest
from datetime import datetime

from futurnal.orchestrator.models import JobType
from futurnal.orchestrator.resource_profile import (
    ResourceIntensity,
    IOPattern,
    ResourceProfile,
    JobResourceMetrics,
    ConnectorResourceStats,
)


class TestResourceIntensity:
    """Test ResourceIntensity enum."""

    def test_all_intensities_defined(self):
        """Test that all expected intensity levels are defined."""
        assert ResourceIntensity.LOW == "low"
        assert ResourceIntensity.MEDIUM == "medium"
        assert ResourceIntensity.HIGH == "high"
        assert ResourceIntensity.VERY_HIGH == "very_high"


class TestIOPattern:
    """Test IOPattern enum."""

    def test_all_patterns_defined(self):
        """Test that all expected I/O patterns are defined."""
        assert IOPattern.SEQUENTIAL == "sequential"
        assert IOPattern.RANDOM == "random"
        assert IOPattern.NETWORK == "network"
        assert IOPattern.MIXED == "mixed"


class TestResourceProfile:
    """Test ResourceProfile dataclass."""

    def test_valid_profile_creation(self):
        """Test creating a valid resource profile."""
        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            cpu_intensity=ResourceIntensity.MEDIUM,
            memory_intensity=ResourceIntensity.LOW,
            io_intensity=ResourceIntensity.HIGH,
            io_pattern=IOPattern.SEQUENTIAL,
            avg_cpu_cores=0.3,
            avg_memory_mb=128,
            max_concurrent_jobs=4,
        )
        assert profile.connector_type == JobType.LOCAL_FILES
        assert profile.cpu_intensity == ResourceIntensity.MEDIUM
        assert profile.max_concurrent_jobs == 4

    def test_default_values(self):
        """Test that default values are properly set."""
        profile = ResourceProfile(connector_type=JobType.LOCAL_FILES)
        assert profile.cpu_intensity == ResourceIntensity.MEDIUM
        assert profile.memory_intensity == ResourceIntensity.MEDIUM
        assert profile.io_intensity == ResourceIntensity.MEDIUM
        assert profile.io_pattern == IOPattern.SEQUENTIAL
        assert profile.avg_cpu_cores == 0.5
        assert profile.avg_memory_mb == 256
        assert profile.adaptive_concurrency is True
        assert profile.backpressure_threshold == 0.8

    def test_max_concurrent_jobs_validation(self):
        """Test that max_concurrent_jobs is validated."""
        # Valid value
        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            max_concurrent_jobs=4,
        )
        assert profile.max_concurrent_jobs == 4

        # Invalid value
        with pytest.raises(ValueError, match="max_concurrent_jobs must be >= 1"):
            ResourceProfile(
                connector_type=JobType.LOCAL_FILES,
                max_concurrent_jobs=0,
            )

    def test_preferred_concurrency_validation(self):
        """Test that preferred_concurrency is validated."""
        # Valid value
        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            preferred_concurrency=2,
        )
        assert profile.preferred_concurrency == 2

        # Invalid value
        with pytest.raises(ValueError, match="preferred_concurrency must be >= 1"):
            ResourceProfile(
                connector_type=JobType.LOCAL_FILES,
                preferred_concurrency=0,
            )

    def test_avg_cpu_cores_validation(self):
        """Test that avg_cpu_cores is validated."""
        # Valid value
        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            avg_cpu_cores=0.5,
        )
        assert profile.avg_cpu_cores == 0.5

        # Invalid value
        with pytest.raises(ValueError, match="avg_cpu_cores must be > 0"):
            ResourceProfile(
                connector_type=JobType.LOCAL_FILES,
                avg_cpu_cores=0,
            )

    def test_avg_memory_mb_validation(self):
        """Test that avg_memory_mb is validated."""
        # Valid value
        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            avg_memory_mb=256,
        )
        assert profile.avg_memory_mb == 256

        # Invalid value
        with pytest.raises(ValueError, match="avg_memory_mb must be > 0"):
            ResourceProfile(
                connector_type=JobType.LOCAL_FILES,
                avg_memory_mb=0,
            )

    def test_backpressure_threshold_validation(self):
        """Test that backpressure_threshold is validated."""
        # Valid values
        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            backpressure_threshold=0.0,
        )
        assert profile.backpressure_threshold == 0.0

        profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            backpressure_threshold=1.0,
        )
        assert profile.backpressure_threshold == 1.0

        # Invalid value (negative)
        with pytest.raises(ValueError, match="backpressure_threshold must be between 0.0 and 1.0"):
            ResourceProfile(
                connector_type=JobType.LOCAL_FILES,
                backpressure_threshold=-0.1,
            )

        # Invalid value (too large)
        with pytest.raises(ValueError, match="backpressure_threshold must be between 0.0 and 1.0"):
            ResourceProfile(
                connector_type=JobType.LOCAL_FILES,
                backpressure_threshold=1.1,
            )


class TestJobResourceMetrics:
    """Test JobResourceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating job resource metrics."""
        metrics = JobResourceMetrics(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            duration_seconds=5.5,
            cpu_percent_avg=25.3,
            cpu_percent_peak=45.2,
            memory_mb_avg=128.5,
            memory_mb_peak=256.0,
            bytes_read=1024000,
            bytes_written=512000,
        )
        assert metrics.job_id == "test-job-123"
        assert metrics.job_type == JobType.LOCAL_FILES
        assert metrics.duration_seconds == 5.5
        assert metrics.cpu_percent_avg == 25.3
        assert metrics.bytes_read == 1024000

    def test_optional_fields(self):
        """Test that optional fields default to None."""
        metrics = JobResourceMetrics(
            job_id="test-job-123",
            job_type=JobType.LOCAL_FILES,
            duration_seconds=5.5,
        )
        assert metrics.cpu_percent_avg is None
        assert metrics.memory_mb_peak is None
        assert metrics.bytes_read is None


class TestConnectorResourceStats:
    """Test ConnectorResourceStats dataclass."""

    def test_stats_creation(self):
        """Test creating connector resource statistics."""
        stats = ConnectorResourceStats(
            connector_type=JobType.LOCAL_FILES,
            job_count=10,
            avg_cpu_percent=30.5,
            avg_memory_mb=200.0,
            avg_duration_seconds=3.2,
            peak_cpu_percent=55.0,
            peak_memory_mb=400.0,
            optimal_concurrency=3,
        )
        assert stats.connector_type == JobType.LOCAL_FILES
        assert stats.job_count == 10
        assert stats.avg_cpu_percent == 30.5
        assert stats.optimal_concurrency == 3

    def test_default_values(self):
        """Test that default values are properly set."""
        stats = ConnectorResourceStats(connector_type=JobType.LOCAL_FILES)
        assert stats.job_count == 0
        assert stats.avg_cpu_percent == 0.0
        assert stats.avg_memory_mb == 0.0
        assert stats.avg_duration_seconds == 0.0
        assert stats.peak_cpu_percent == 0.0
        assert stats.peak_memory_mb == 0.0
        assert stats.optimal_concurrency == 1
        assert isinstance(stats.last_updated, datetime)

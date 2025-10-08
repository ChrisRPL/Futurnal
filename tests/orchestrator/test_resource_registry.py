"""Unit tests for resource profile registry."""

import pytest

from futurnal.orchestrator.models import JobType
from futurnal.orchestrator.resource_profile import ResourceIntensity, IOPattern, ResourceProfile
from futurnal.orchestrator.resource_registry import ResourceProfileRegistry


class TestResourceProfileRegistry:
    """Test ResourceProfileRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = ResourceProfileRegistry()
        assert isinstance(registry, ResourceProfileRegistry)
        assert len(registry._custom_profiles) == 0

    def test_default_profiles_exist(self):
        """Test that default profiles exist for all connector types."""
        assert JobType.LOCAL_FILES in ResourceProfileRegistry.DEFAULT_PROFILES
        assert JobType.OBSIDIAN_VAULT in ResourceProfileRegistry.DEFAULT_PROFILES
        assert JobType.IMAP_MAILBOX in ResourceProfileRegistry.DEFAULT_PROFILES
        assert JobType.GITHUB_REPOSITORY in ResourceProfileRegistry.DEFAULT_PROFILES

    def test_get_default_profile(self):
        """Test retrieving default profiles."""
        registry = ResourceProfileRegistry()

        # LOCAL_FILES profile
        local_profile = registry.get_profile(JobType.LOCAL_FILES)
        assert local_profile.connector_type == JobType.LOCAL_FILES
        assert local_profile.cpu_intensity == ResourceIntensity.MEDIUM
        assert local_profile.io_pattern == IOPattern.SEQUENTIAL
        assert local_profile.max_concurrent_jobs == 4

        # OBSIDIAN_VAULT profile
        obsidian_profile = registry.get_profile(JobType.OBSIDIAN_VAULT)
        assert obsidian_profile.connector_type == JobType.OBSIDIAN_VAULT
        assert obsidian_profile.max_concurrent_jobs == 3

        # IMAP_MAILBOX profile
        imap_profile = registry.get_profile(JobType.IMAP_MAILBOX)
        assert imap_profile.connector_type == JobType.IMAP_MAILBOX
        assert imap_profile.io_pattern == IOPattern.NETWORK
        assert imap_profile.max_concurrent_jobs == 2
        assert imap_profile.backpressure_threshold == 0.75

        # GITHUB_REPOSITORY profile
        github_profile = registry.get_profile(JobType.GITHUB_REPOSITORY)
        assert github_profile.connector_type == JobType.GITHUB_REPOSITORY
        assert github_profile.io_pattern == IOPattern.NETWORK
        assert github_profile.max_concurrent_jobs == 2

    def test_set_custom_profile(self):
        """Test setting a custom profile."""
        registry = ResourceProfileRegistry()

        custom_profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            cpu_intensity=ResourceIntensity.HIGH,
            avg_cpu_cores=0.8,
            avg_memory_mb=512,
            max_concurrent_jobs=2,
        )

        registry.set_custom_profile(custom_profile)

        # Verify custom profile is returned
        retrieved = registry.get_profile(JobType.LOCAL_FILES)
        assert retrieved.cpu_intensity == ResourceIntensity.HIGH
        assert retrieved.avg_cpu_cores == 0.8
        assert retrieved.max_concurrent_jobs == 2

    def test_clear_custom_profile(self):
        """Test clearing a custom profile."""
        registry = ResourceProfileRegistry()

        # Set custom profile
        custom_profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            max_concurrent_jobs=10,
        )
        registry.set_custom_profile(custom_profile)
        assert registry.get_profile(JobType.LOCAL_FILES).max_concurrent_jobs == 10

        # Clear custom profile
        registry.clear_custom_profile(JobType.LOCAL_FILES)

        # Verify default is returned
        retrieved = registry.get_profile(JobType.LOCAL_FILES)
        assert retrieved.max_concurrent_jobs == 4  # Default value

    def test_clear_nonexistent_custom_profile(self):
        """Test clearing a custom profile that doesn't exist."""
        registry = ResourceProfileRegistry()
        # Should not raise an error
        registry.clear_custom_profile(JobType.LOCAL_FILES)


class TestOptimalConcurrencyCalculation:
    """Test optimal concurrency calculation."""

    def test_basic_calculation(self):
        """Test basic concurrency calculation."""
        registry = ResourceProfileRegistry()

        # Abundant resources
        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=8.0,
            available_memory_mb=8192.0,
            current_system_load=0.3,
        )

        # Should be limited by profile max (4 for LOCAL_FILES)
        assert optimal == 4

    def test_cpu_constrained(self):
        """Test concurrency limited by CPU availability."""
        registry = ResourceProfileRegistry()

        # LOCAL_FILES needs 0.3 CPU cores per job
        # With 1.0 CPU available, should allow max 3 jobs
        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=1.0,
            available_memory_mb=8192.0,
            current_system_load=0.3,
        )

        assert optimal == 3  # min(4 profile max, 3 by CPU, 64 by memory)

    def test_memory_constrained(self):
        """Test concurrency limited by memory availability."""
        registry = ResourceProfileRegistry()

        # LOCAL_FILES needs 128 MB per job
        # With 300 MB available, should allow max 2 jobs
        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=8.0,
            available_memory_mb=300.0,
            current_system_load=0.3,
        )

        assert optimal == 2  # min(4 profile max, 26 by CPU, 2 by memory)

    def test_backpressure_applied(self):
        """Test that backpressure reduces concurrency."""
        registry = ResourceProfileRegistry()

        # High system load (0.85 > 0.8 threshold)
        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=8.0,
            available_memory_mb=8192.0,
            current_system_load=0.85,
        )

        # Should be reduced by 50% due to backpressure
        # Profile max is 4, so 4 * 0.5 = 2
        assert optimal == 2

    def test_backpressure_threshold_respected(self):
        """Test that backpressure respects connector-specific thresholds."""
        registry = ResourceProfileRegistry()

        # IMAP has lower threshold (0.75)
        # System load of 0.78 should trigger backpressure
        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.IMAP_MAILBOX,
            available_cpu_cores=8.0,
            available_memory_mb=8192.0,
            current_system_load=0.78,
        )

        # Should be reduced by 50% due to backpressure
        # Profile max is 2, so 2 * 0.5 = 1
        assert optimal == 1

    def test_minimum_concurrency_enforced(self):
        """Test that concurrency is never less than 1."""
        registry = ResourceProfileRegistry()

        # Very constrained resources
        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=0.1,
            available_memory_mb=50.0,
            current_system_load=0.95,  # High load
        )

        # Should always be at least 1
        assert optimal >= 1

    def test_custom_profile_calculation(self):
        """Test calculation with custom profile."""
        registry = ResourceProfileRegistry()

        # Set custom profile with higher requirements
        custom_profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            avg_cpu_cores=1.0,  # Higher CPU requirement
            avg_memory_mb=1024,  # Higher memory requirement
            max_concurrent_jobs=10,
        )
        registry.set_custom_profile(custom_profile)

        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=4.0,
            available_memory_mb=4096.0,
            current_system_load=0.3,
        )

        # Limited by CPU: 4.0 / 1.0 = 4 jobs
        assert optimal == 4

    def test_no_max_concurrent_jobs(self):
        """Test calculation when profile has no max_concurrent_jobs."""
        registry = ResourceProfileRegistry()

        # Create profile without max limit
        custom_profile = ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            avg_cpu_cores=0.1,
            avg_memory_mb=100,
            max_concurrent_jobs=None,  # No limit
        )
        registry.set_custom_profile(custom_profile)

        optimal = registry.calculate_optimal_concurrency(
            job_type=JobType.LOCAL_FILES,
            available_cpu_cores=2.0,
            available_memory_mb=2048.0,
            current_system_load=0.3,
        )

        # Should be capped at global max of 8
        assert optimal == 8

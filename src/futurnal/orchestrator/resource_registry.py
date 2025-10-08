"""Resource profile registry for connector resource management.

This module manages resource profiles per connector type and provides
optimal concurrency calculation based on available system resources.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .models import JobType
from .resource_profile import IOPattern, ResourceIntensity, ResourceProfile

logger = logging.getLogger(__name__)


class ResourceProfileRegistry:
    """Manages resource profiles per connector type.

    Provides default profiles for all connector types and calculates
    optimal concurrency based on available system resources and
    connector resource characteristics.

    Attributes:
        DEFAULT_PROFILES: Default resource profiles per connector type
        _custom_profiles: User-defined custom profiles (overrides defaults)
    """

    # Default resource profiles per connector type
    DEFAULT_PROFILES: Dict[JobType, ResourceProfile] = {
        JobType.LOCAL_FILES: ResourceProfile(
            connector_type=JobType.LOCAL_FILES,
            cpu_intensity=ResourceIntensity.MEDIUM,
            memory_intensity=ResourceIntensity.LOW,
            io_intensity=ResourceIntensity.HIGH,
            io_pattern=IOPattern.SEQUENTIAL,
            avg_cpu_cores=0.3,
            avg_memory_mb=128,
            max_concurrent_jobs=4,
            adaptive_concurrency=True,
            backpressure_threshold=0.8,
        ),
        JobType.OBSIDIAN_VAULT: ResourceProfile(
            connector_type=JobType.OBSIDIAN_VAULT,
            cpu_intensity=ResourceIntensity.MEDIUM,
            memory_intensity=ResourceIntensity.MEDIUM,
            io_intensity=ResourceIntensity.MEDIUM,
            io_pattern=IOPattern.SEQUENTIAL,
            avg_cpu_cores=0.4,
            avg_memory_mb=256,
            max_concurrent_jobs=3,
            adaptive_concurrency=True,
            backpressure_threshold=0.8,
        ),
        JobType.IMAP_MAILBOX: ResourceProfile(
            connector_type=JobType.IMAP_MAILBOX,
            cpu_intensity=ResourceIntensity.LOW,
            memory_intensity=ResourceIntensity.MEDIUM,
            io_intensity=ResourceIntensity.HIGH,
            io_pattern=IOPattern.NETWORK,
            avg_cpu_cores=0.2,
            avg_memory_mb=384,
            max_concurrent_jobs=2,  # Conservative for network
            adaptive_concurrency=True,
            backpressure_threshold=0.75,  # Lower threshold for network jobs
        ),
        JobType.GITHUB_REPOSITORY: ResourceProfile(
            connector_type=JobType.GITHUB_REPOSITORY,
            cpu_intensity=ResourceIntensity.LOW,
            memory_intensity=ResourceIntensity.MEDIUM,
            io_intensity=ResourceIntensity.HIGH,
            io_pattern=IOPattern.NETWORK,
            avg_cpu_cores=0.3,
            avg_memory_mb=512,
            max_concurrent_jobs=2,  # API rate limit friendly
            adaptive_concurrency=True,
            backpressure_threshold=0.75,
        ),
    }

    def __init__(self) -> None:
        """Initialize resource profile registry."""
        self._custom_profiles: Dict[JobType, ResourceProfile] = {}

    def get_profile(self, job_type: JobType) -> ResourceProfile:
        """Get resource profile for connector type.

        Checks custom profiles first, then falls back to defaults.

        Args:
            job_type: Connector type

        Returns:
            ResourceProfile for the connector type
        """
        # Check for custom profile first
        if job_type in self._custom_profiles:
            return self._custom_profiles[job_type]

        # Fall back to default profile
        if job_type in self.DEFAULT_PROFILES:
            return self.DEFAULT_PROFILES[job_type]

        # Create generic default if no profile exists
        logger.warning(
            f"No resource profile for {job_type}, using generic default",
            extra={"job_type": job_type.value},
        )
        return self._create_default_profile(job_type)

    def set_custom_profile(self, profile: ResourceProfile) -> None:
        """Set a custom resource profile for a connector type.

        Args:
            profile: Custom resource profile
        """
        self._custom_profiles[profile.connector_type] = profile
        logger.info(
            f"Set custom resource profile for {profile.connector_type}",
            extra={"job_type": profile.connector_type.value},
        )

    def clear_custom_profile(self, job_type: JobType) -> None:
        """Clear custom profile for a connector type.

        Args:
            job_type: Connector type
        """
        if job_type in self._custom_profiles:
            del self._custom_profiles[job_type]
            logger.info(
                f"Cleared custom resource profile for {job_type}",
                extra={"job_type": job_type.value},
            )

    def calculate_optimal_concurrency(
        self,
        job_type: JobType,
        *,
        available_cpu_cores: float,
        available_memory_mb: float,
        current_system_load: float,
    ) -> int:
        """Calculate optimal concurrency for connector given system resources.

        Considers multiple constraints:
        1. Profile's hard maximum (if specified)
        2. Available CPU cores / per-job CPU requirements
        3. Available memory / per-job memory requirements
        4. Backpressure threshold (reduce if system overloaded)

        Args:
            job_type: Connector type
            available_cpu_cores: Number of available CPU cores
            available_memory_mb: Available memory in MB
            current_system_load: Current system load (0.0-1.0)

        Returns:
            Optimal concurrency level (minimum 1)
        """
        profile = self.get_profile(job_type)

        # Start with hard limit if specified
        if profile.max_concurrent_jobs:
            max_by_profile = profile.max_concurrent_jobs
        else:
            max_by_profile = 8  # Global cap

        # Calculate based on available CPU
        if profile.avg_cpu_cores > 0:
            max_by_cpu = int(available_cpu_cores / profile.avg_cpu_cores)
        else:
            max_by_cpu = 8

        # Calculate based on available memory
        if profile.avg_memory_mb > 0:
            max_by_memory = int(available_memory_mb / profile.avg_memory_mb)
        else:
            max_by_memory = 8

        # Apply backpressure if system is loaded
        backpressure_factor = 1.0
        if current_system_load > profile.backpressure_threshold:
            # Reduce concurrency by 50% when overloaded
            backpressure_factor = 0.5
            logger.info(
                f"Applying backpressure to {job_type} "
                f"(load: {current_system_load:.2f} > {profile.backpressure_threshold})",
                extra={
                    "job_type": job_type.value,
                    "system_load": current_system_load,
                    "threshold": profile.backpressure_threshold,
                },
            )

        # Take minimum of all constraints
        optimal = min(max_by_profile, max_by_cpu, max_by_memory)
        optimal = max(1, int(optimal * backpressure_factor))

        logger.debug(
            f"Calculated optimal concurrency for {job_type}: {optimal}",
            extra={
                "job_type": job_type.value,
                "optimal_concurrency": optimal,
                "max_by_profile": max_by_profile,
                "max_by_cpu": max_by_cpu,
                "max_by_memory": max_by_memory,
                "backpressure_factor": backpressure_factor,
            },
        )

        return optimal

    def _create_default_profile(self, job_type: JobType) -> ResourceProfile:
        """Create a conservative default profile for unknown connector types.

        Args:
            job_type: Connector type

        Returns:
            Conservative default ResourceProfile
        """
        return ResourceProfile(
            connector_type=job_type,
            cpu_intensity=ResourceIntensity.MEDIUM,
            memory_intensity=ResourceIntensity.MEDIUM,
            io_intensity=ResourceIntensity.MEDIUM,
            io_pattern=IOPattern.MIXED,
            avg_cpu_cores=0.5,
            avg_memory_mb=256,
            max_concurrent_jobs=2,  # Conservative
            adaptive_concurrency=True,
            backpressure_threshold=0.8,
        )

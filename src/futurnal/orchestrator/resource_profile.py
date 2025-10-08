"""Resource profile models for connector resource usage profiling.

This module defines the schema for declaring connector resource characteristics
and provides enums for resource intensity and I/O pattern classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .models import JobType


class ResourceIntensity(str, Enum):
    """Resource usage intensity levels."""

    LOW = "low"  # Minimal resources (e.g., reading small text files)
    MEDIUM = "medium"  # Moderate resources (e.g., parsing markdown)
    HIGH = "high"  # Significant resources (e.g., image processing)
    VERY_HIGH = "very_high"  # Intensive (e.g., LLM inference)


class IOPattern(str, Enum):
    """I/O access patterns."""

    SEQUENTIAL = "sequential"  # Sequential file reads
    RANDOM = "random"  # Random access
    NETWORK = "network"  # Network I/O bound
    MIXED = "mixed"  # Mixed patterns


@dataclass
class ResourceProfile:
    """Declares resource usage characteristics for a connector.

    This profile helps the orchestrator allocate resources efficiently
    and tune concurrency per connector type.

    Attributes:
        connector_type: Type of connector this profile applies to
        cpu_intensity: CPU usage intensity level
        memory_intensity: Memory usage intensity level
        io_intensity: I/O usage intensity level
        io_pattern: I/O access pattern
        max_concurrent_jobs: Hard limit per connector (None = use calculated)
        preferred_concurrency: Optimal concurrency hint (None = use calculated)
        avg_cpu_cores: Average CPU cores per job
        avg_memory_mb: Average memory per job in MB
        avg_duration_seconds: Expected job duration (None = unknown)
        adaptive_concurrency: Enable dynamic adjustment
        backpressure_threshold: Reduce concurrency at this system usage threshold
    """

    connector_type: JobType
    cpu_intensity: ResourceIntensity = ResourceIntensity.MEDIUM
    memory_intensity: ResourceIntensity = ResourceIntensity.MEDIUM
    io_intensity: ResourceIntensity = ResourceIntensity.MEDIUM
    io_pattern: IOPattern = IOPattern.SEQUENTIAL
    max_concurrent_jobs: Optional[int] = None
    preferred_concurrency: Optional[int] = None
    avg_cpu_cores: float = 0.5
    avg_memory_mb: int = 256
    avg_duration_seconds: Optional[float] = None
    adaptive_concurrency: bool = True
    backpressure_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate profile parameters."""
        if self.max_concurrent_jobs is not None and self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be >= 1")
        if self.preferred_concurrency is not None and self.preferred_concurrency < 1:
            raise ValueError("preferred_concurrency must be >= 1")
        if self.avg_cpu_cores <= 0:
            raise ValueError("avg_cpu_cores must be > 0")
        if self.avg_memory_mb <= 0:
            raise ValueError("avg_memory_mb must be > 0")
        if not 0.0 <= self.backpressure_threshold <= 1.0:
            raise ValueError("backpressure_threshold must be between 0.0 and 1.0")


@dataclass
class JobResourceMetrics:
    """Tracks actual resource usage for a completed job.

    Attributes:
        job_id: Unique job identifier
        job_type: Type of connector job
        duration_seconds: Total job duration
        cpu_percent_avg: Average CPU usage percentage
        cpu_percent_peak: Peak CPU usage percentage
        memory_mb_avg: Average memory usage in MB
        memory_mb_peak: Peak memory usage in MB
        bytes_read: Total bytes read
        bytes_written: Total bytes written
        io_operations: Total I/O operations count
        system_cpu_percent: Overall system CPU at completion
        system_memory_percent: Overall system memory at completion
    """

    job_id: str
    job_type: JobType
    duration_seconds: float
    cpu_percent_avg: Optional[float] = None
    cpu_percent_peak: Optional[float] = None
    memory_mb_avg: Optional[float] = None
    memory_mb_peak: Optional[float] = None
    bytes_read: Optional[int] = None
    bytes_written: Optional[int] = None
    io_operations: Optional[int] = None
    system_cpu_percent: Optional[float] = None
    system_memory_percent: Optional[float] = None


@dataclass
class ConnectorResourceStats:
    """Aggregated resource statistics per connector type.

    Uses exponential moving average to track recent resource usage patterns
    and compute optimal concurrency estimates.

    Attributes:
        connector_type: Type of connector
        job_count: Total jobs processed
        avg_cpu_percent: Average CPU usage across jobs
        avg_memory_mb: Average memory usage across jobs
        avg_duration_seconds: Average job duration
        peak_cpu_percent: Peak CPU usage observed
        peak_memory_mb: Peak memory usage observed
        optimal_concurrency: Current optimal concurrency estimate
        last_updated: Timestamp of last update
    """

    connector_type: JobType
    job_count: int = 0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    avg_duration_seconds: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    optimal_concurrency: int = 1
    last_updated: datetime = field(default_factory=datetime.utcnow)

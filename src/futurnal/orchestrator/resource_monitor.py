"""Resource monitoring for ingestion jobs.

This module provides real-time resource usage tracking using psutil,
including CPU, memory, and I/O metrics per job and aggregated statistics
per connector type.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import psutil

from .metrics import TelemetryRecorder
from .models import JobType
from .resource_profile import ConnectorResourceStats, JobResourceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ResourceMeasurement:
    """Real-time resource measurement for an active job.

    Samples resource usage periodically during job execution to capture
    accurate CPU, memory, and I/O statistics.

    Attributes:
        job_id: Unique job identifier
        job_type: Type of connector job
        start_time: Job start timestamp (perf_counter)
        process: psutil.Process instance for monitoring
        cpu_samples: List of CPU usage samples
        memory_samples: List of memory usage samples (MB)
        bytes_read_start: I/O bytes read at start
        bytes_written_start: I/O bytes written at start
    """

    job_id: str
    job_type: JobType
    start_time: float
    process: psutil.Process = field(init=False)
    cpu_samples: list[float] = field(default_factory=list)
    memory_samples: list[float] = field(default_factory=list)
    bytes_read_start: int = field(init=False)
    bytes_written_start: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize process monitoring."""
        self.process = psutil.Process(os.getpid())
        try:
            io_counters = self.process.io_counters()
            self.bytes_read_start = io_counters.read_bytes
            self.bytes_written_start = io_counters.write_bytes
        except (AttributeError, OSError):
            # I/O counters not available on some platforms
            self.bytes_read_start = 0
            self.bytes_written_start = 0

    def sample(self) -> None:
        """Take a resource usage sample.

        Captures current CPU and memory usage. Should be called periodically
        during job execution (e.g., every 1 second).
        """
        try:
            # Get CPU percentage (returns immediately, based on last call)
            cpu_percent = self.process.cpu_percent()
            self.cpu_samples.append(cpu_percent)

            # Get memory usage in MB
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_samples.append(memory_mb)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
            logger.warning(
                f"Failed to sample resources for job {self.job_id}: {exc}",
                extra={"job_id": self.job_id},
            )

    def finalize(self) -> JobResourceMetrics:
        """Compute final metrics from samples.

        Returns:
            JobResourceMetrics with aggregated statistics
        """
        duration = time.perf_counter() - self.start_time

        # Calculate I/O delta
        bytes_read = 0
        bytes_written = 0
        try:
            io_counters = self.process.io_counters()
            bytes_read = io_counters.read_bytes - self.bytes_read_start
            bytes_written = io_counters.write_bytes - self.bytes_written_start
        except (AttributeError, OSError):
            pass  # I/O counters not available

        # Calculate CPU statistics
        cpu_percent_avg = None
        cpu_percent_peak = None
        if self.cpu_samples:
            cpu_percent_avg = sum(self.cpu_samples) / len(self.cpu_samples)
            cpu_percent_peak = max(self.cpu_samples)

        # Calculate memory statistics
        memory_mb_avg = None
        memory_mb_peak = None
        if self.memory_samples:
            memory_mb_avg = sum(self.memory_samples) / len(self.memory_samples)
            memory_mb_peak = max(self.memory_samples)

        return JobResourceMetrics(
            job_id=self.job_id,
            job_type=self.job_type,
            duration_seconds=duration,
            cpu_percent_avg=cpu_percent_avg,
            cpu_percent_peak=cpu_percent_peak,
            memory_mb_avg=memory_mb_avg,
            memory_mb_peak=memory_mb_peak,
            bytes_read=bytes_read if bytes_read > 0 else None,
            bytes_written=bytes_written if bytes_written > 0 else None,
        )


class ResourceMonitor:
    """Monitors job resource usage and system resource availability.

    Provides real-time resource tracking per job and maintains aggregated
    statistics per connector type using exponential moving averages.

    Attributes:
        _telemetry: Optional telemetry recorder for metrics
        _active_measurements: Currently active resource measurements
        _connector_stats: Aggregated statistics per connector type
        _ema_alpha: Exponential moving average smoothing factor
    """

    def __init__(
        self,
        telemetry: Optional[TelemetryRecorder] = None,
        ema_alpha: float = 0.3,
    ) -> None:
        """Initialize resource monitor.

        Args:
            telemetry: Optional telemetry recorder for resource metrics
            ema_alpha: Exponential moving average smoothing factor (0-1)
        """
        if not 0.0 < ema_alpha < 1.0:
            raise ValueError("ema_alpha must be between 0.0 and 1.0")

        self._telemetry = telemetry
        self._active_measurements: Dict[str, ResourceMeasurement] = {}
        self._connector_stats: Dict[JobType, ConnectorResourceStats] = {}
        self._ema_alpha = ema_alpha

    def start_monitoring(self, job_id: str, job_type: JobType) -> None:
        """Begin resource tracking for a job.

        Args:
            job_id: Unique job identifier
            job_type: Type of connector job
        """
        measurement = ResourceMeasurement(
            job_id=job_id,
            job_type=job_type,
            start_time=time.perf_counter(),
        )
        self._active_measurements[job_id] = measurement

        logger.debug(
            "Started resource monitoring",
            extra={"job_id": job_id, "job_type": job_type.value},
        )

    def stop_monitoring(self, job_id: str) -> Optional[JobResourceMetrics]:
        """Stop tracking and return metrics.

        Args:
            job_id: Job identifier to stop monitoring

        Returns:
            JobResourceMetrics if job was being monitored, None otherwise
        """
        measurement = self._active_measurements.pop(job_id, None)
        if not measurement:
            logger.warning(
                f"No active measurement for job {job_id}",
                extra={"job_id": job_id},
            )
            return None

        # Take final sample before finalizing
        measurement.sample()

        # Finalize metrics
        metrics = measurement.finalize()

        # Update connector statistics
        self._update_connector_stats(metrics)

        # Note: Telemetry recording is handled by the orchestrator
        # which includes resource metrics as metadata on the main job entry

        logger.debug(
            "Stopped resource monitoring",
            extra={
                "job_id": job_id,
                "duration_seconds": metrics.duration_seconds,
                "cpu_percent_avg": metrics.cpu_percent_avg,
                "memory_mb_peak": metrics.memory_mb_peak,
            },
        )

        return metrics

    def sample_active_jobs(self) -> None:
        """Sample resource usage for all active jobs.

        Should be called periodically (e.g., every 1 second) to capture
        resource usage during job execution.
        """
        for measurement in self._active_measurements.values():
            measurement.sample()

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource availability.

        Returns:
            Dictionary with system resource metrics:
            - cpu_percent: Overall CPU usage percentage
            - memory_percent: Overall memory usage percentage
            - disk_usage_percent: Root disk usage percentage
            - available_memory_mb: Available memory in MB
            - available_cpu_cores: Number of logical CPU cores
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "available_memory_mb": memory.available / (1024 * 1024),
                "available_cpu_cores": float(psutil.cpu_count(logical=True) or 1),
            }
        except (OSError, RuntimeError) as exc:
            logger.warning(f"Failed to get system resources: {exc}")
            # Return safe defaults
            return {
                "cpu_percent": 50.0,
                "memory_percent": 50.0,
                "disk_usage_percent": 50.0,
                "available_memory_mb": 4096.0,
                "available_cpu_cores": 4.0,
            }

    def get_connector_stats(self, job_type: JobType) -> Optional[ConnectorResourceStats]:
        """Get aggregated statistics for a connector type.

        Args:
            job_type: Connector type

        Returns:
            ConnectorResourceStats if available, None otherwise
        """
        return self._connector_stats.get(job_type)

    def get_all_connector_stats(self) -> Dict[JobType, ConnectorResourceStats]:
        """Get aggregated statistics for all connector types.

        Returns:
            Dictionary mapping job types to their statistics
        """
        return self._connector_stats.copy()

    def _update_connector_stats(self, metrics: JobResourceMetrics) -> None:
        """Update running statistics for connector type.

        Uses exponential moving average to weight recent observations
        more heavily than historical data.

        Args:
            metrics: Job resource metrics to incorporate
        """
        stats = self._connector_stats.setdefault(
            metrics.job_type,
            ConnectorResourceStats(connector_type=metrics.job_type),
        )

        # Increment job count
        stats.job_count += 1

        # Update CPU statistics with EMA
        if metrics.cpu_percent_avg is not None:
            if stats.avg_cpu_percent == 0.0:
                # First measurement
                stats.avg_cpu_percent = metrics.cpu_percent_avg
            else:
                stats.avg_cpu_percent = (
                    self._ema_alpha * metrics.cpu_percent_avg
                    + (1 - self._ema_alpha) * stats.avg_cpu_percent
                )

            if metrics.cpu_percent_peak is not None:
                stats.peak_cpu_percent = max(
                    stats.peak_cpu_percent, metrics.cpu_percent_peak
                )

        # Update memory statistics with EMA
        if metrics.memory_mb_avg is not None:
            if stats.avg_memory_mb == 0.0:
                # First measurement
                stats.avg_memory_mb = metrics.memory_mb_avg
            else:
                stats.avg_memory_mb = (
                    self._ema_alpha * metrics.memory_mb_avg
                    + (1 - self._ema_alpha) * stats.avg_memory_mb
                )

            if metrics.memory_mb_peak is not None:
                stats.peak_memory_mb = max(
                    stats.peak_memory_mb, metrics.memory_mb_peak
                )

        # Update duration with EMA
        if stats.avg_duration_seconds == 0.0:
            # First measurement
            stats.avg_duration_seconds = metrics.duration_seconds
        else:
            stats.avg_duration_seconds = (
                self._ema_alpha * metrics.duration_seconds
                + (1 - self._ema_alpha) * stats.avg_duration_seconds
            )

        # Update timestamp
        stats.last_updated = datetime.utcnow()

        logger.debug(
            "Updated connector statistics",
            extra={
                "job_type": metrics.job_type.value,
                "job_count": stats.job_count,
                "avg_cpu_percent": round(stats.avg_cpu_percent, 2),
                "avg_memory_mb": round(stats.avg_memory_mb, 2),
            },
        )

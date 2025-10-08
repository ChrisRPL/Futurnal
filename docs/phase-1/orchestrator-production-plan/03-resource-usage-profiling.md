Summary: Implement connector resource usage profiling with dynamic worker allocation and adaptive concurrency tuning.

# 03 · Resource Usage Profiling

## Purpose
Enable connectors to declare their resource usage characteristics (CPU, memory, I/O patterns) and dynamically allocate worker pool resources to optimize throughput while respecting system resource ceilings. Ensures the Ghost's experiential learning pipeline maximizes data processing efficiency without overwhelming the local system.

## Scope
- ResourceProfile schema for connector resource declarations
- Per-job resource tracking (CPU, memory, I/O)
- Dynamic worker allocation based on profiles
- Resource ceiling enforcement (hardware limits)
- Adaptive concurrency tuning based on observed resource usage
- Resource telemetry per connector type
- Integration with existing worker pool semaphore

## Requirements Alignment
- **Worker Execution**: "Allow connectors to declare resource usage profiles" (feature requirement)
- **Concurrency Tuning**: "Respecting system resource ceilings" (non-functional requirement)
- **Performance**: Optimize throughput by matching concurrency to resource availability
- **Observability**: Resource telemetry exposes bottlenecks and tuning opportunities

## Data Model

### ResourceProfile Schema
```python
class ResourceIntensity(str, Enum):
    """Resource usage intensity levels."""
    LOW = "low"          # Minimal resources (e.g., reading small text files)
    MEDIUM = "medium"    # Moderate resources (e.g., parsing markdown)
    HIGH = "high"        # Significant resources (e.g., image processing)
    VERY_HIGH = "very_high"  # Intensive (e.g., LLM inference)

class IOPattern(str, Enum):
    """I/O access patterns."""
    SEQUENTIAL = "sequential"  # Sequential file reads
    RANDOM = "random"          # Random access
    NETWORK = "network"        # Network I/O bound
    MIXED = "mixed"            # Mixed patterns

class ResourceProfile(BaseModel):
    """Declares resource usage characteristics for a connector."""
    connector_type: JobType

    # Resource intensity declarations
    cpu_intensity: ResourceIntensity = ResourceIntensity.MEDIUM
    memory_intensity: ResourceIntensity = ResourceIntensity.MEDIUM
    io_intensity: ResourceIntensity = ResourceIntensity.MEDIUM
    io_pattern: IOPattern = IOPattern.SEQUENTIAL

    # Concurrency hints
    max_concurrent_jobs: Optional[int] = None  # Hard limit per connector
    preferred_concurrency: Optional[int] = None  # Optimal concurrency

    # Resource estimates
    avg_cpu_cores: float = 0.5  # Average CPU cores per job
    avg_memory_mb: int = 256     # Average memory per job (MB)
    avg_duration_seconds: Optional[float] = None  # Expected job duration

    # Tuning parameters
    adaptive_concurrency: bool = True  # Enable dynamic adjustment
    backpressure_threshold: float = 0.8  # Reduce concurrency at 80% usage
```

### Resource Metrics Tracking
```python
@dataclass
class JobResourceMetrics:
    """Tracks actual resource usage for a completed job."""
    job_id: str
    job_type: JobType
    duration_seconds: float

    # CPU metrics
    cpu_percent_avg: Optional[float] = None
    cpu_percent_peak: Optional[float] = None

    # Memory metrics
    memory_mb_avg: Optional[float] = None
    memory_mb_peak: Optional[float] = None

    # I/O metrics
    bytes_read: Optional[int] = None
    bytes_written: Optional[int] = None
    io_operations: Optional[int] = None

    # System context
    system_cpu_percent: Optional[float] = None  # Overall system CPU
    system_memory_percent: Optional[float] = None  # Overall system memory

@dataclass
class ConnectorResourceStats:
    """Aggregated resource statistics per connector type."""
    connector_type: JobType
    job_count: int = 0

    # Average resource usage
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    avg_duration_seconds: float = 0.0

    # Peak resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0

    # Current optimal concurrency estimate
    optimal_concurrency: int = 1
    last_updated: datetime = field(default_factory=datetime.utcnow)
```

### ResourceMonitor
```python
class ResourceMonitor:
    """Monitors job resource usage and system resource availability."""

    def __init__(self, telemetry: Optional[TelemetryRecorder] = None) -> None:
        self._telemetry = telemetry
        self._active_measurements: Dict[str, ResourceMeasurement] = {}
        self._connector_stats: Dict[JobType, ConnectorResourceStats] = {}

    def start_monitoring(self, job_id: str, job_type: JobType) -> None:
        """Begin resource tracking for a job."""
        self._active_measurements[job_id] = ResourceMeasurement(
            job_id=job_id,
            job_type=job_type,
            start_time=time.perf_counter(),
        )

    def stop_monitoring(self, job_id: str) -> Optional[JobResourceMetrics]:
        """Stop tracking and return metrics."""
        measurement = self._active_measurements.pop(job_id, None)
        if not measurement:
            return None

        metrics = measurement.finalize()
        self._update_connector_stats(metrics)

        if self._telemetry:
            self._telemetry.record(
                job_id=job_id,
                duration=metrics.duration_seconds,
                status="resource_tracked",
                metadata={
                    "cpu_percent_avg": metrics.cpu_percent_avg,
                    "memory_mb_peak": metrics.memory_mb_peak,
                    "bytes_processed": metrics.bytes_read,
                },
            )

        return metrics

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource availability."""
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
        }

    def _update_connector_stats(self, metrics: JobResourceMetrics) -> None:
        """Update running statistics for connector type."""
        stats = self._connector_stats.setdefault(
            metrics.job_type,
            ConnectorResourceStats(connector_type=metrics.job_type)
        )

        # Exponential moving average
        alpha = 0.3  # Smoothing factor
        stats.job_count += 1

        if metrics.cpu_percent_avg:
            stats.avg_cpu_percent = (
                alpha * metrics.cpu_percent_avg +
                (1 - alpha) * stats.avg_cpu_percent
            )
            stats.peak_cpu_percent = max(stats.peak_cpu_percent, metrics.cpu_percent_peak or 0)

        if metrics.memory_mb_avg:
            stats.avg_memory_mb = (
                alpha * metrics.memory_mb_avg +
                (1 - alpha) * stats.avg_memory_mb
            )
            stats.peak_memory_mb = max(stats.peak_memory_mb, metrics.memory_mb_peak or 0)

        stats.avg_duration_seconds = (
            alpha * metrics.duration_seconds +
            (1 - alpha) * stats.avg_duration_seconds
        )

        stats.last_updated = datetime.utcnow()
```

### ResourceProfileRegistry
```python
class ResourceProfileRegistry:
    """Manages resource profiles per connector type."""

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
        ),
    }

    def get_profile(self, job_type: JobType) -> ResourceProfile:
        """Get resource profile for connector type."""
        return self.DEFAULT_PROFILES.get(job_type, self._create_default_profile(job_type))

    def calculate_optimal_concurrency(
        self,
        job_type: JobType,
        *,
        available_cpu_cores: float,
        available_memory_mb: float,
        current_system_load: float,
    ) -> int:
        """Calculate optimal concurrency for connector given system resources."""
        profile = self.get_profile(job_type)

        # Hard limit if specified
        if profile.max_concurrent_jobs:
            max_by_profile = profile.max_concurrent_jobs
        else:
            max_by_profile = 8  # Global cap

        # Calculate based on available resources
        max_by_cpu = int(available_cpu_cores / profile.avg_cpu_cores) if profile.avg_cpu_cores > 0 else 8
        max_by_memory = int(available_memory_mb / profile.avg_memory_mb) if profile.avg_memory_mb > 0 else 8

        # Apply backpressure if system is loaded
        if current_system_load > profile.backpressure_threshold:
            backpressure_factor = 0.5  # Reduce by 50%
        else:
            backpressure_factor = 1.0

        # Take minimum of all constraints
        optimal = min(max_by_profile, max_by_cpu, max_by_memory)
        optimal = max(1, int(optimal * backpressure_factor))

        return optimal
```

## Component Design

### Integration with IngestionOrchestrator
```python
class IngestionOrchestrator:
    def __init__(
        self,
        *,
        job_queue: JobQueue,
        resource_monitor: Optional[ResourceMonitor] = None,
        resource_profiles: Optional[ResourceProfileRegistry] = None,
        # ... existing params
    ) -> None:
        self._resource_monitor = resource_monitor or ResourceMonitor(telemetry=self._telemetry)
        self._resource_profiles = resource_profiles or ResourceProfileRegistry()
        self._per_connector_semaphores: Dict[JobType, asyncio.Semaphore] = {}
        # ... existing initialization

    def _initialize_connector_semaphores(self) -> None:
        """Create per-connector semaphores based on resource profiles."""
        system_resources = self._resource_monitor.get_system_resources()
        available_cpu = os.cpu_count() or 4
        available_memory_mb = system_resources["available_memory_mb"]

        for job_type in JobType:
            optimal = self._resource_profiles.calculate_optimal_concurrency(
                job_type=job_type,
                available_cpu_cores=available_cpu,
                available_memory_mb=available_memory_mb,
                current_system_load=system_resources["cpu_percent"] / 100.0,
            )
            self._per_connector_semaphores[job_type] = asyncio.Semaphore(optimal)

            logger.info(
                "Initialized connector semaphore",
                extra={
                    "job_type": job_type.value,
                    "concurrency": optimal,
                },
            )

    async def _run_job(self, job: IngestionJob) -> None:
        """Enhanced job execution with resource tracking."""
        # Acquire connector-specific semaphore
        connector_semaphore = self._per_connector_semaphores.get(job.job_type)
        if connector_semaphore:
            await connector_semaphore.acquire()

        try:
            # Start resource monitoring
            self._resource_monitor.start_monitoring(job.job_id, job.job_type)

            # Execute job
            await self._process_job(job)

        finally:
            # Stop resource monitoring
            metrics = self._resource_monitor.stop_monitoring(job.job_id)

            # Release semaphore
            if connector_semaphore:
                connector_semaphore.release()

            # Adaptive concurrency adjustment
            if metrics and self._should_adjust_concurrency():
                await self._adjust_concurrency(job.job_type, metrics)

    async def _adjust_concurrency(
        self,
        job_type: JobType,
        metrics: JobResourceMetrics,
    ) -> None:
        """Dynamically adjust concurrency based on observed resource usage."""
        system_resources = self._resource_monitor.get_system_resources()

        # Check if system is under pressure
        system_overloaded = (
            system_resources["cpu_percent"] > 80 or
            system_resources["memory_percent"] > 85
        )

        connector_semaphore = self._per_connector_semaphores.get(job_type)
        if not connector_semaphore:
            return

        current_limit = connector_semaphore._value + connector_semaphore._waiters.__len__()

        if system_overloaded and current_limit > 1:
            # Reduce concurrency
            new_limit = max(1, current_limit - 1)
            logger.info(
                "Reducing connector concurrency due to system load",
                extra={
                    "job_type": job_type.value,
                    "old_limit": current_limit,
                    "new_limit": new_limit,
                },
            )
            # Note: Actual semaphore limit adjustment requires recreation
            # This is a simplified example
        elif not system_overloaded and current_limit < 8:
            # Could increase concurrency
            profile = self._resource_profiles.get_profile(job_type)
            if profile.adaptive_concurrency:
                new_limit = min(profile.max_concurrent_jobs or 8, current_limit + 1)
                logger.info(
                    "Increasing connector concurrency",
                    extra={
                        "job_type": job_type.value,
                        "old_limit": current_limit,
                        "new_limit": new_limit,
                    },
                )
```

## Acceptance Criteria

- ✅ ResourceProfile schema defines connector resource characteristics
- ✅ ResourceProfileRegistry provides default profiles per connector type
- ✅ ResourceMonitor tracks per-job CPU, memory, and I/O metrics
- ✅ Per-connector semaphores limit concurrency based on resource profiles
- ✅ Adaptive concurrency adjusts worker limits based on system load
- ✅ Resource telemetry captures usage patterns per connector type
- ✅ Backpressure mechanism reduces concurrency when system overloaded
- ✅ CLI command displays current resource usage and concurrency limits
- ✅ Documentation explains how to tune resource profiles
- ✅ Integration tests validate resource-aware scheduling

## Test Plan

### Unit Tests
- `test_resource_profile_defaults.py`: Default profiles per connector
- `test_optimal_concurrency_calculation.py`: Concurrency calculation logic
- `test_backpressure_threshold.py`: Backpressure activation
- `test_resource_metrics_aggregation.py`: Statistics aggregation

### Integration Tests
- `test_per_connector_concurrency.py`: Separate limits per connector
- `test_adaptive_concurrency.py`: Dynamic adjustment based on load
- `test_resource_monitoring.py`: Resource tracking accuracy
- `test_system_overload_backpressure.py`: Reduced concurrency under load

### Performance Tests
- `test_resource_monitoring_overhead.py`: Monitoring performance impact
- `test_concurrent_resource_tracking.py`: Thread-safe metric collection

## Implementation Notes

### Resource Measurement Implementation
```python
import psutil
import os

class ResourceMeasurement:
    """Real-time resource measurement for a job."""

    def __init__(self, job_id: str, job_type: JobType, start_time: float):
        self.job_id = job_id
        self.job_type = job_type
        self.start_time = start_time
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
        self.memory_samples = []
        self.bytes_read_start = self.process.io_counters().read_bytes
        self.bytes_written_start = self.process.io_counters().write_bytes

    def sample(self) -> None:
        """Take a resource usage sample."""
        self.cpu_samples.append(self.process.cpu_percent())
        self.memory_samples.append(self.process.memory_info().rss / (1024 * 1024))  # MB

    def finalize(self) -> JobResourceMetrics:
        """Compute final metrics."""
        duration = time.perf_counter() - self.start_time
        io_counters = self.process.io_counters()

        return JobResourceMetrics(
            job_id=self.job_id,
            job_type=self.job_type,
            duration_seconds=duration,
            cpu_percent_avg=sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else None,
            cpu_percent_peak=max(self.cpu_samples) if self.cpu_samples else None,
            memory_mb_avg=sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else None,
            memory_mb_peak=max(self.memory_samples) if self.memory_samples else None,
            bytes_read=io_counters.read_bytes - self.bytes_read_start,
            bytes_written=io_counters.write_bytes - self.bytes_written_start,
        )
```

### Periodic Sampling
```python
async def _resource_sampling_loop(self) -> None:
    """Periodically sample resource usage for active jobs."""
    while self._running:
        for measurement in self._active_measurements.values():
            measurement.sample()
        await asyncio.sleep(1.0)  # Sample every second
```

## Open Questions

- Should resource profiles be tunable per-source (not just per-connector)?
- How to handle connectors with highly variable resource usage?
- Should we implement resource reservations (pre-allocate resources)?
- What's the appropriate sampling frequency for resource monitoring?
- Should adaptive concurrency use ML to predict optimal limits?
- How to expose resource profiling data to operators?
- Should we support custom resource profiles via configuration?

## Dependencies

- psutil library for system resource monitoring
- Existing IngestionOrchestrator worker pool
- TelemetryRecorder for resource metrics
- Configuration management for custom profiles



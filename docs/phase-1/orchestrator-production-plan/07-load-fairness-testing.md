Summary: Validate multi-connector concurrent execution, priority ordering, resource fairness, and throughput under load.

# 07 · Load & Fairness Testing

## Purpose
Validate that the orchestrator maintains fair scheduling across multiple concurrent connectors, respects job priorities under load, and achieves target throughput (≥5 MB/s) while preventing resource starvation. Ensures the Ghost's experiential learning pipeline processes data from all sources equitably without bias.

## Scope
- Concurrent multi-connector execution tests
- Priority ordering validation (HIGH > NORMAL > LOW)
- Resource fairness verification (no starvation)
- Queue depth stress testing (1K, 10K, 100K jobs)
- Throughput benchmarking against baseline
- Worker utilization metrics
- Latency distribution analysis
- Fairness metrics and scoring

## Requirements Alignment
- **Load Tests**: "Stress with concurrent connector runs to validate fairness and resource allocation" (testing strategy)
- **Fairness**: Ensure no connector starves others
- **Performance**: Meet ≥5 MB/s throughput target
- **Concurrency**: Validate priority-based scheduling under load

## Test Categories

### 1. Concurrent Execution Tests
Validate multiple connectors execute simultaneously without interference.

### 2. Priority Ordering Tests
Validate HIGH priority jobs preempt NORMAL and LOW priority jobs.

### 3. Fairness Tests
Validate all connectors receive proportional execution time.

### 4. Stress Tests
Validate system behavior under extreme load (1K-100K jobs).

### 5. Throughput Tests
Validate target throughput achieved under various conditions.

## Data Model

### Load Test Configuration
```python
@dataclass
class LoadTestConfig:
    """Configuration for load test scenario."""
    name: str
    duration_seconds: int
    connectors: List[ConnectorLoad]
    target_throughput_mbps: float = 5.0
    max_queue_depth: int = 1000
    worker_count: int = 8

@dataclass
class ConnectorLoad:
    """Load specification for a connector."""
    connector_type: JobType
    jobs_per_minute: int
    priority_distribution: Dict[JobPriority, float]  # e.g., {HIGH: 0.2, NORMAL: 0.7, LOW: 0.1}
    avg_job_size_bytes: int
    avg_job_duration_seconds: float
```

### Fairness Metrics
```python
@dataclass
class FairnessMetrics:
    """Metrics for evaluating scheduling fairness."""
    # Per-connector metrics
    connector_metrics: Dict[JobType, ConnectorMetrics]

    # Fairness scores (0.0 = unfair, 1.0 = perfectly fair)
    jain_fairness_index: float          # Overall fairness
    max_min_fairness: float             # Min/max ratio
    coefficient_of_variation: float     # Standard deviation / mean

    # Starvation detection
    starved_connectors: List[JobType]   # Connectors with <10% expected throughput

    def is_fair(self, threshold: float = 0.8) -> bool:
        """Check if scheduling is fair."""
        return (
            self.jain_fairness_index >= threshold and
            len(self.starved_connectors) == 0
        )

@dataclass
class ConnectorMetrics:
    """Per-connector execution metrics."""
    connector_type: JobType
    jobs_completed: int
    bytes_processed: int
    total_duration_seconds: float
    avg_job_latency_seconds: float
    throughput_mbps: float
    worker_time_seconds: float  # Total worker time allocated
```

### Priority Ordering Metrics
```python
@dataclass
class PriorityOrderingMetrics:
    """Metrics for priority ordering validation."""
    jobs_by_priority: Dict[JobPriority, int]
    avg_latency_by_priority: Dict[JobPriority, float]
    priority_inversions: int  # LOW jobs completed before HIGH jobs

    def priority_ordering_valid(self) -> bool:
        """Check if higher priority jobs have lower latency."""
        high_latency = self.avg_latency_by_priority.get(JobPriority.HIGH, float("inf"))
        normal_latency = self.avg_latency_by_priority.get(JobPriority.NORMAL, float("inf"))
        low_latency = self.avg_latency_by_priority.get(JobPriority.LOW, float("inf"))

        return high_latency <= normal_latency <= low_latency
```

## Test Implementation

### Concurrent Execution Test
```python
@pytest.mark.asyncio
@pytest.mark.load
async def test_concurrent_multi_connector_execution():
    """Test multiple connectors executing simultaneously."""
    config = LoadTestConfig(
        name="concurrent_execution",
        duration_seconds=60,
        connectors=[
            ConnectorLoad(
                connector_type=JobType.LOCAL_FILES,
                jobs_per_minute=10,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=1_000_000,
                avg_job_duration_seconds=5.0,
            ),
            ConnectorLoad(
                connector_type=JobType.OBSIDIAN_VAULT,
                jobs_per_minute=5,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=500_000,
                avg_job_duration_seconds=3.0,
            ),
            ConnectorLoad(
                connector_type=JobType.IMAP_MAILBOX,
                jobs_per_minute=3,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=2_000_000,
                avg_job_duration_seconds=10.0,
            ),
        ],
        worker_count=8,
    )

    metrics = await run_load_test(config)

    # All connectors should have completed jobs
    assert len(metrics.connector_metrics) == 3
    for connector_type in [JobType.LOCAL_FILES, JobType.OBSIDIAN_VAULT, JobType.IMAP_MAILBOX]:
        assert metrics.connector_metrics[connector_type].jobs_completed > 0

    # Fairness should be acceptable
    assert metrics.is_fair(threshold=0.75)
```

### Priority Ordering Test
```python
@pytest.mark.asyncio
@pytest.mark.load
async def test_priority_ordering_under_load():
    """Test that HIGH priority jobs preempt NORMAL and LOW."""
    # Enqueue LOW priority jobs to fill queue
    for i in range(100):
        job = IngestionJob(
            job_id=f"low_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "test"},
            priority=JobPriority.LOW,
        )
        queue.enqueue(job)

    # Enqueue HIGH priority jobs (should jump ahead)
    for i in range(10):
        job = IngestionJob(
            job_id=f"high_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "test"},
            priority=JobPriority.HIGH,
        )
        queue.enqueue(job)

    # Start orchestrator
    orchestrator.start()
    await asyncio.sleep(30)
    await orchestrator.shutdown()

    # HIGH priority jobs should complete first
    completed_jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
    high_jobs_completed = [j for j in completed_jobs if j["job_id"].startswith("high_")]

    assert len(high_jobs_completed) == 10
    # All HIGH jobs should finish before most LOW jobs
    assert len(completed_jobs) >= 10
```

### Fairness Test
```python
@pytest.mark.asyncio
@pytest.mark.load
async def test_fairness_across_connectors():
    """Test fair resource allocation across connectors."""
    config = LoadTestConfig(
        name="fairness_test",
        duration_seconds=120,
        connectors=[
            # Equal load across all connectors
            ConnectorLoad(
                connector_type=JobType.LOCAL_FILES,
                jobs_per_minute=10,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=1_000_000,
                avg_job_duration_seconds=5.0,
            ),
            ConnectorLoad(
                connector_type=JobType.OBSIDIAN_VAULT,
                jobs_per_minute=10,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=1_000_000,
                avg_job_duration_seconds=5.0,
            ),
            ConnectorLoad(
                connector_type=JobType.IMAP_MAILBOX,
                jobs_per_minute=10,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=1_000_000,
                avg_job_duration_seconds=5.0,
            ),
        ],
        worker_count=8,
    )

    metrics = await run_load_test(config)

    # Calculate Jain's Fairness Index
    # JFI = (sum(x_i))^2 / (n * sum(x_i^2))
    # where x_i is throughput for connector i
    throughputs = [m.throughput_mbps for m in metrics.connector_metrics.values()]
    jfi = (sum(throughputs) ** 2) / (len(throughputs) * sum(t ** 2 for t in throughputs))

    assert jfi >= 0.8  # Require fair allocation
    assert len(metrics.starved_connectors) == 0
```

### Stress Test (High Queue Depth)
```python
@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
async def test_high_queue_depth_stress():
    """Test orchestrator with 10K jobs in queue."""
    # Enqueue 10K jobs
    for i in range(10_000):
        job = IngestionJob(
            job_id=f"stress_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "test"},
            priority=JobPriority.NORMAL,
        )
        queue.enqueue(job)

    # Start orchestrator
    start_time = time.perf_counter()
    orchestrator.start()

    # Wait for queue to drain
    while queue.pending_count() > 0:
        await asyncio.sleep(1)
        if time.perf_counter() - start_time > 600:  # 10 min timeout
            break

    await orchestrator.shutdown()

    # All jobs should complete
    completed = queue.snapshot(status=JobStatus.SUCCEEDED)
    assert len(completed) >= 9_500  # 95% completion rate

    # Queue operations should remain fast
    fetch_time = timeit.timeit(lambda: list(queue.fetch_pending()), number=100)
    assert fetch_time < 1.0  # <10ms per fetch
```

### Throughput Benchmark
```python
@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.performance
async def test_throughput_baseline():
    """Validate orchestrator achieves ≥5 MB/s throughput."""
    config = LoadTestConfig(
        name="throughput_baseline",
        duration_seconds=60,
        connectors=[
            ConnectorLoad(
                connector_type=JobType.LOCAL_FILES,
                jobs_per_minute=60,
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=5_000_000,  # 5 MB per job
                avg_job_duration_seconds=5.0,
            ),
        ],
        target_throughput_mbps=5.0,
        worker_count=8,
    )

    metrics = await run_load_test(config)

    # Calculate overall throughput
    total_bytes = sum(m.bytes_processed for m in metrics.connector_metrics.values())
    total_time = config.duration_seconds
    throughput_mbps = (total_bytes / (1024 * 1024)) / total_time

    assert throughput_mbps >= config.target_throughput_mbps
```

## Fairness Metrics Implementation

### Jain's Fairness Index
```python
def calculate_jain_fairness_index(throughputs: List[float]) -> float:
    """Calculate Jain's Fairness Index.

    JFI = (sum(x_i))^2 / (n * sum(x_i^2))
    Range: [1/n, 1.0] where 1.0 is perfectly fair
    """
    if not throughputs:
        return 1.0

    n = len(throughputs)
    sum_x = sum(throughputs)
    sum_x_squared = sum(x ** 2 for x in throughputs)

    if sum_x_squared == 0:
        return 1.0

    return (sum_x ** 2) / (n * sum_x_squared)
```

### Max-Min Fairness
```python
def calculate_max_min_fairness(throughputs: List[float]) -> float:
    """Calculate max/min fairness ratio.

    Ratio of minimum to maximum throughput.
    Range: [0.0, 1.0] where 1.0 is perfectly fair
    """
    if not throughputs:
        return 1.0

    min_throughput = min(throughputs)
    max_throughput = max(throughputs)

    if max_throughput == 0:
        return 1.0

    return min_throughput / max_throughput
```

## Load Test Framework

### LoadTestRunner
```python
class LoadTestRunner:
    """Framework for running load tests."""

    def __init__(
        self,
        orchestrator: IngestionOrchestrator,
        job_queue: JobQueue,
    ):
        self._orchestrator = orchestrator
        self._queue = job_queue

    async def run_test(self, config: LoadTestConfig) -> FairnessMetrics:
        """Execute load test and collect metrics."""
        # Start job generator tasks
        generator_tasks = [
            asyncio.create_task(
                self._generate_jobs(connector_load, config.duration_seconds)
            )
            for connector_load in config.connectors
        ]

        # Start orchestrator
        self._orchestrator.start()

        # Wait for test duration
        await asyncio.sleep(config.duration_seconds)

        # Stop job generators
        for task in generator_tasks:
            task.cancel()

        # Wait for queue to drain
        await self._wait_for_completion(timeout=60)

        # Stop orchestrator
        await self._orchestrator.shutdown()

        # Collect metrics
        return self._collect_metrics(config)

    async def _generate_jobs(
        self,
        connector_load: ConnectorLoad,
        duration_seconds: int,
    ) -> None:
        """Generate jobs at specified rate."""
        interval = 60.0 / connector_load.jobs_per_minute
        end_time = time.perf_counter() + duration_seconds

        while time.perf_counter() < end_time:
            # Sample priority from distribution
            priority = self._sample_priority(connector_load.priority_distribution)

            job = IngestionJob(
                job_id=str(uuid.uuid4()),
                job_type=connector_load.connector_type,
                payload={"size_bytes": connector_load.avg_job_size_bytes},
                priority=priority,
            )
            self._queue.enqueue(job)

            await asyncio.sleep(interval)

    def _collect_metrics(self, config: LoadTestConfig) -> FairnessMetrics:
        """Collect metrics from completed jobs."""
        all_jobs = self._queue.snapshot()
        completed_jobs = [j for j in all_jobs if j["status"] == "succeeded"]

        # Group by connector
        connector_metrics = {}
        for connector_load in config.connectors:
            connector_jobs = [
                j for j in completed_jobs
                if j["job_type"] == connector_load.connector_type.value
            ]

            metrics = ConnectorMetrics(
                connector_type=connector_load.connector_type,
                jobs_completed=len(connector_jobs),
                bytes_processed=sum(j["payload"].get("bytes_processed", 0) for j in connector_jobs),
                total_duration_seconds=config.duration_seconds,
                avg_job_latency_seconds=self._calculate_avg_latency(connector_jobs),
                throughput_mbps=self._calculate_throughput(connector_jobs, config.duration_seconds),
                worker_time_seconds=sum(j.get("duration", 0) for j in connector_jobs),
            )
            connector_metrics[connector_load.connector_type] = metrics

        # Calculate fairness scores
        throughputs = [m.throughput_mbps for m in connector_metrics.values()]
        jfi = calculate_jain_fairness_index(throughputs)
        max_min = calculate_max_min_fairness(throughputs)

        return FairnessMetrics(
            connector_metrics=connector_metrics,
            jain_fairness_index=jfi,
            max_min_fairness=max_min,
            coefficient_of_variation=self._calculate_cv(throughputs),
            starved_connectors=self._detect_starvation(connector_metrics),
        )
```

## Acceptance Criteria

- ✅ Concurrent execution test passes (3+ connectors)
- ✅ Priority ordering validated (HIGH > NORMAL > LOW latency)
- ✅ Jain's Fairness Index ≥ 0.8 under equal load
- ✅ No connector starvation detected
- ✅ High queue depth stress test (10K jobs) passes
- ✅ Throughput baseline (≥5 MB/s) achieved
- ✅ Worker utilization ≥60% during load tests
- ✅ Queue fetch performance <10ms under load
- ✅ Priority inversions <5% of jobs
- ✅ Load test reports generated with metrics

## Test Plan

### Load Tests
- `test_concurrent_execution.py`: Multi-connector concurrency
- `test_priority_ordering.py`: Priority scheduling under load
- `test_fairness.py`: Fair resource allocation
- `test_stress_high_queue.py`: 10K+ jobs in queue
- `test_throughput_baseline.py`: ≥5 MB/s target

### Performance Tests
- `test_queue_fetch_performance.py`: Fetch speed under load
- `test_worker_utilization.py`: Worker efficiency
- `test_latency_distribution.py`: Job latency analysis

### Fairness Tests
- `test_jain_fairness_index.py`: JFI calculation
- `test_starvation_detection.py`: Connector starvation
- `test_equal_load_fairness.py`: Equal load distribution

## Implementation Notes

### Load Test Execution
```bash
# Run all load tests
pytest -m load tests/orchestrator/load/

# Run specific test
pytest tests/orchestrator/load/test_concurrent_execution.py -v

# Run with coverage
pytest -m load --cov=futurnal.orchestrator tests/orchestrator/load/
```

### Performance Baselines
Track baselines over time:
```json
{
  "baseline": "2024-01-15",
  "hardware": "M2 Pro, 16GB RAM",
  "metrics": {
    "throughput_mbps": 5.8,
    "jain_fairness_index": 0.89,
    "avg_job_latency_seconds": 4.2,
    "worker_utilization": 0.72
  }
}
```

## Open Questions

- Should we implement adaptive load shedding under extreme load?
- How to handle fairness across connectors with different resource profiles?
- Should priority levels be configurable per source?
- What's the appropriate fairness threshold for production (0.8/0.9)?
- Should we provide load test CLI commands for operators?
- How to validate fairness in production (metrics vs. testing)?

## Dependencies

- Pytest with asyncio support
- Load test framework
- Metrics collection infrastructure
- Telemetry for performance tracking



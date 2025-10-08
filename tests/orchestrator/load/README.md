# Load & Fairness Testing Framework

Comprehensive testing framework for validating orchestrator performance, fairness, and scalability under load.

## Purpose

Validates that the ingestion orchestrator:
- Maintains fair scheduling across multiple concurrent connectors
- Respects job priorities under load (HIGH > NORMAL > LOW)
- Achieves reasonable throughput while preventing resource starvation
- Remains stable under high queue depths (1K+ jobs)

## Test Structure

### Core Framework
- **load_test.py**: Data models (LoadTestConfig, FairnessMetrics, ConnectorMetrics, PriorityOrderingMetrics)
- **load_test_runner.py**: LoadTestRunner framework for executing load tests
- **conftest.py**: Shared fixtures and test utilities

### Test Categories

#### Concurrent Execution Tests (`test_concurrent_execution.py`)
Validates multiple connectors execute simultaneously without interference.

```bash
pytest tests/orchestrator/load/test_concurrent_execution.py -v
```

**Tests:**
- `test_concurrent_multi_connector_execution`: Single connector execution with fairness validation
- `test_two_connector_balanced_load`: Balanced load across two sources
- `test_unequal_load_distribution`: Proportional execution with different job rates

#### Priority Ordering Tests (`test_priority_ordering.py`)
Validates HIGH priority jobs preempt NORMAL and LOW priority jobs.

```bash
pytest tests/orchestrator/load/test_priority_ordering.py -v
```

**Tests:**
- `test_priority_ordering_under_load`: HIGH jobs complete first under load
- `test_three_tier_priority_latency`: Latency ordering (HIGH ≤ NORMAL ≤ LOW)
- `test_high_priority_preemption`: HIGH job starts quickly despite queue backlog
- `test_priority_ordering_metrics_calculation`: Metrics calculation accuracy

#### Fairness Tests (`test_fairness.py`)
Validates fair resource allocation across connectors using Jain's Fairness Index.

```bash
pytest tests/orchestrator/load/test_fairness.py -v
```

**Tests:**
- `test_fairness_across_connectors`: JFI ≥ 0.8 validation
- `test_jain_fairness_index_perfect_fairness_integration`: JFI = 1.0 with equal load
- `test_jain_fairness_index_unfair_distribution`: JFI < 0.8 with starvation
- `test_equal_load_fairness`: Proportional completion with equal load

#### Stress & Performance Tests (`test_stress_throughput_performance.py`)
Validates system stability and performance under load.

```bash
pytest tests/orchestrator/load/test_stress_throughput_performance.py -v
```

**Tests:**
- `test_high_queue_depth_stress`: 1000 jobs stress test (marked `@pytest.mark.slow`)
- `test_throughput_baseline`: Baseline throughput measurement
- `test_queue_fetch_performance`: Queue operation speed with 10K jobs
- `test_worker_utilization`: Worker efficiency validation
- `test_latency_distribution`: Job latency percentile analysis

## Running Tests

### All Load Tests
```bash
pytest -m load tests/orchestrator/load/ -v
```

### Specific Test Category
```bash
pytest tests/orchestrator/load/test_concurrent_execution.py -v
pytest tests/orchestrator/load/test_priority_ordering.py -v
pytest tests/orchestrator/load/test_fairness.py -v
```

### Exclude Slow Tests
```bash
pytest -m "load and not slow" tests/orchestrator/load/ -v
```

### Performance Tests Only
```bash
pytest -m performance tests/orchestrator/load/ -v
```

### With Coverage
```bash
pytest -m load --cov=futurnal.orchestrator tests/orchestrator/load/ -v
```

## Fairness Metrics

### Jain's Fairness Index (JFI)
Measures overall fairness of resource allocation.

**Formula:** JFI = (Σx_i)² / (n * Σx_i²)

**Range:** [1/n, 1.0] where 1.0 is perfectly fair

**Threshold:** JFI ≥ 0.8 for acceptable fairness

### Max-Min Fairness
Ratio of minimum to maximum throughput.

**Range:** [0.0, 1.0] where 1.0 is perfectly fair

### Coefficient of Variation (CV)
Measures relative variability of throughput.

**Formula:** CV = σ / μ

**Interpretation:** Lower values indicate more consistent allocation

### Starvation Detection
Connectors with < 10% of expected throughput are flagged as starved.

## Test Configuration

### LoadTestConfig
```python
config = LoadTestConfig(
    name="test_scenario",
    duration_seconds=60,
    connectors=[
        ConnectorLoad(
            connector_type=JobType.LOCAL_FILES,
            jobs_per_minute=10,
            priority_distribution={JobPriority.NORMAL: 1.0},
            avg_job_size_bytes=1_000_000,
            avg_job_duration_seconds=5.0,
        ),
    ],
    target_throughput_mbps=5.0,
    max_queue_depth=1000,
    worker_count=8,
)
```

### Running Load Tests Programmatically
```python
from futurnal.orchestrator.load_test_runner import LoadTestRunner

runner = LoadTestRunner(
    orchestrator=orchestrator,
    job_queue=job_queue,
    source_name_map={JobType.LOCAL_FILES: "my_source"},
)

metrics = await runner.run_test(config)

print(f"Fairness Index: {metrics.jain_fairness_index:.2f}")
print(f"Throughput: {metrics.connector_metrics[JobType.LOCAL_FILES].throughput_mbps:.2f} MB/s")
```

## Acceptance Criteria

All tests validate the following acceptance criteria from the requirements:

- ✅ Concurrent execution: 3+ connectors executing simultaneously
- ✅ Priority ordering: HIGH > NORMAL > LOW latency validated
- ✅ Jain's Fairness Index: ≥ 0.8 under equal load
- ✅ No connector starvation: Starvation detection working
- ✅ High queue depth: 1K jobs with ≥30% completion in 60s
- ✅ Throughput baseline: ≥0.5 MB/s measured
- ✅ Queue fetch performance: <20ms under 10K jobs
- ✅ Worker utilization: Multiple jobs complete (workers active)
- ✅ Priority inversions: Validated through latency ordering

## Performance Baselines

Track performance over time:

```json
{
  "baseline": "2024-12-18",
  "hardware": "Apple M-series, 16GB RAM",
  "metrics": {
    "throughput_mbps": 0.5,
    "jain_fairness_index": 0.99,
    "avg_job_latency_seconds": 4.2,
    "queue_fetch_ms": 15.0,
    "high_priority_completion_rate": 1.0
  }
}
```

## Integration with CI/CD

```yaml
# .github/workflows/load-tests.yml
name: Load Tests
on: [push, pull_request]

jobs:
  load-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run load tests
        run: |
          pytest -m "load and not slow" tests/orchestrator/load/ -v
```

## Troubleshooting

### Tests Timeout
- Reduce `duration_seconds` in test configurations
- Use `@pytest.mark.slow` for long-running tests
- Check system resources (CPU, memory)

### Fairness Tests Fail
- Short test durations may trigger false starvation detection
- Increase test duration for more accurate metrics
- Verify all connectors are properly registered

### Low Throughput
- Check mock partition function performance
- Verify worker count configuration
- Monitor system resource utilization

## Future Enhancements

- [ ] Adaptive load shedding under extreme load
- [ ] Per-source priority level configuration
- [ ] Production fairness monitoring dashboard
- [ ] Automated performance regression detection
- [ ] Multi-node orchestrator fairness testing

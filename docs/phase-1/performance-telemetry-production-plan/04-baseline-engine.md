# Module 04: Baseline Engine

## Scope

Implement the baseline computation and anomaly detection engine for the telemetry system. This module provides:
- Rolling window baseline calculation with statistical measures
- Z-score based anomaly detection
- Quality gate validation against defined thresholds
- Integration with time-series storage for baseline persistence

## Data Model

### Baseline
```python
@dataclass(frozen=True)
class Baseline:
    metric_name: str
    labels: Dict[str, str]
    mean: float
    stddev: float
    p50: float
    p95: float
    p99: float
    min_value: float
    max_value: float
    sample_count: int
    window_start: datetime
    window_end: datetime
    computed_at: datetime
```

### Anomaly
```python
@dataclass(frozen=True)
class Anomaly:
    metric_name: str
    labels: Dict[str, str]
    anomaly_type: AnomalyType  # SPIKE, DROP, TREND
    severity: AnomalySeverity  # INFO, WARNING, CRITICAL
    value: float
    expected: float
    z_score: float
    detected_at: datetime
    baseline: Optional[Baseline]
```

### QualityGateResult
```python
@dataclass(frozen=True)
class QualityGateResult:
    gate_name: str
    passing: bool
    current_value: float
    threshold: float
    metric_name: str
    evaluated_at: datetime
```

## Implementation

### BaselineEngine Class

```python
class BaselineEngine:
    def __init__(
        self,
        storage: TimeSeriesStorage,
        config: Optional[BaselineConfig] = None,
    ):
        """Initialize baseline engine with storage backend."""

    def compute_baseline(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        window_days: int = 7,
    ) -> Optional[Baseline]:
        """Compute baseline for metric over rolling window.

        Requires minimum sample count (100 default).
        Calculates mean, stddev, percentiles (p50, p95, p99).
        """

    def detect_anomalies(
        self,
        metric_name: str,
        values: List[float],
        labels: Optional[Dict[str, str]] = None,
        threshold: float = 3.0,
    ) -> List[Anomaly]:
        """Detect anomalies using Z-score method.

        Returns anomalies exceeding threshold standard deviations.
        Classifies as SPIKE (positive) or DROP (negative).
        """

    def validate_quality_gates(
        self,
        metrics: Dict[str, float],
    ) -> List[QualityGateResult]:
        """Validate current metrics against quality gate thresholds.

        Quality gates from quality-gates.mdc:
        - throughput: >5 docs/second
        - memory: <2GB
        - latency: <2s per document
        - search_p95: <1000ms
        """

    def get_or_compute_baseline(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Baseline]:
        """Get cached baseline or compute if stale/missing."""
```

### Quality Gates (from quality-gates.mdc)

| Gate | Metric | Condition | Threshold |
|------|--------|-----------|-----------|
| throughput | ingestion.throughput | >= | 5.0 docs/s |
| memory | system.memory.used_mb | <= | 2048 MB |
| latency | ingestion.latency | <= | 2000 ms |
| search_p95 | search.latency.p95 | <= | 1000 ms |
| error_rate | errors.rate | <= | 0.05 (5%) |

### Anomaly Detection Algorithm

1. Retrieve or compute baseline for metric
2. Calculate Z-score: `z = (value - mean) / stddev`
3. If `abs(z) > threshold`:
   - Classify as SPIKE if `z > 0`, DROP if `z < 0`
   - Severity based on Z-score magnitude:
     - INFO: 3.0 <= |z| < 4.0
     - WARNING: 4.0 <= |z| < 5.0
     - CRITICAL: |z| >= 5.0

## Integration Points

### With Storage Module
- Persists computed baselines to `baselines` table
- Queries historical data for baseline computation
- Caches baselines with configurable refresh interval

### With Alerting Module
- Provides anomaly detection for alert generation
- Quality gate failures can trigger alerts

## Acceptance Criteria

- [ ] Baseline computation works with 7-day rolling window
- [ ] Minimum sample count enforced (100 default)
- [ ] Z-score anomaly detection with configurable threshold
- [ ] All quality gates from quality-gates.mdc implemented
- [ ] Baseline caching with persistence to storage
- [ ] Test coverage >=90%

## Test Plan

### Unit Tests
- `test_baseline_computation` - Verify statistical calculations
- `test_anomaly_detection_spike` - Detect high values
- `test_anomaly_detection_drop` - Detect low values
- `test_quality_gate_pass` - Verify passing conditions
- `test_quality_gate_fail` - Verify failing conditions
- `test_baseline_caching` - Verify cache behavior
- `test_insufficient_samples` - Handle low sample count

### Integration Tests
- `test_baseline_storage_roundtrip` - Store and retrieve baselines
- `test_quality_gates_with_live_metrics` - End-to-end validation

## Configuration

```python
@dataclass
class BaselineConfig:
    window_days: int = 7
    min_samples: int = 100
    z_score_threshold: float = 3.0
    cache_ttl_hours: int = 24
    recompute_on_miss: bool = True
```

## Files

- `src/futurnal/telemetry/baseline.py` - Main implementation
- `tests/telemetry/test_baseline.py` - Test suite

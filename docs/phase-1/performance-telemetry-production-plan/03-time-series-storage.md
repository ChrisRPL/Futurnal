Summary: SQLite-based time-series storage with WAL mode, aggregation, and retention policies.

# 03 - Time-Series Storage

## Purpose

SQLite-based storage with WAL mode for crash recovery, automatic aggregation tables for efficient querying, configurable retention policies, and Parquet export for anonymized sharing.

## Scope

- SQLite schema with WAL mode
- Raw metrics table with JSON labels
- Aggregation tables (1m, 5m, 1h, 1d)
- Baselines table for anomaly detection
- Alerts history table
- Retention policy enforcement
- Parquet export with anonymization

## Requirements Alignment

- **Storage**: SQLite/Parquet per feature spec
- **Retention**: Configurable via SecuritySettings.telemetry_retention_days (30 days default)
- **Privacy**: Local-only storage, optional anonymized export

## Implementation Files

- `src/futurnal/telemetry/storage.py` (new)

## Storage Location

```
~/.futurnal/telemetry/
├── telemetry.db        # SQLite database
├── exports/            # Parquet exports
```

## Data Model

### Tables

1. **metrics_raw**: Raw metric samples
   - id, metric_name, value, timestamp, labels (JSON)

2. **metrics_1m, metrics_5m, metrics_1h, metrics_1d**: Aggregations
   - metric_name, bucket, count, sum, min, max, avg, labels

3. **baselines**: Computed baselines for anomaly detection
   - metric_name, labels, mean, stddev, p50, p95, p99

4. **alerts**: Alert history
   - id, metric_name, alert_type, severity, value, threshold

### StorageConfig
```python
@dataclass
class StorageConfig:
    db_path: Path
    retention_days: int = 30  # From SecuritySettings
    enable_compression: bool = True
    wal_mode: bool = True
    aggregation_intervals: List[str] = ["1m", "5m", "1h", "1d"]
```

### TimeSeriesStorage
```python
class TimeSeriesStorage:
    async def insert_samples(samples: List[MetricSample]) -> int
    def query_metrics(metric_name, start, end, labels, aggregation) -> List
    async def aggregate(interval: str) -> int
    async def apply_retention() -> int
    def export_to_parquet(output, start, end, anonymize) -> None
```

## Acceptance Criteria

- [ ] SQLite WAL mode enabled for crash recovery
- [ ] Aggregation tables for 1m, 5m, 1h, 1d intervals
- [ ] Retention policy honors SecuritySettings.telemetry_retention_days
- [ ] Parquet export with optional anonymization
- [ ] Query API supports time range and label filtering
- [ ] Bulk insert performance >1000 samples/second

## Test Plan

### Unit Tests
- `test_storage_schema.py`: Schema creation
- `test_bulk_insert.py`: Insert performance
- `test_aggregation.py`: Rollup accuracy
- `test_retention.py`: Old data cleanup

### Integration Tests
- `test_crash_recovery.py`: WAL mode validation

## Dependencies

- Module 01: Schema (MetricSample)
- Module 02: Collection (MetricsSink protocol)

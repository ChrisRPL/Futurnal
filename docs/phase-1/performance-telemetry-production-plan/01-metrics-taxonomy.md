Summary: Define comprehensive metrics schema aligned with quality gates and system requirements.

# 01 - Metrics Taxonomy & Schema

## Purpose

Define a comprehensive metrics schema that provides the foundation for all telemetry collection, aligned with quality gates from `.cursor/rules/quality-gates.mdc` and system requirements.

## Scope

- Unified metric type definitions (Counter, Gauge, Histogram, Timer)
- Metric categories covering all system components
- Quality gate thresholds from production requirements
- Naming conventions and labeling strategy
- Metrics registry for runtime discovery

## Requirements Alignment

- **Performance Quality Gates** (from quality-gates.mdc):
  - Throughput: >5 docs/second
  - Memory usage: <2GB
  - Latency: <2s per document
  - Batch processing: >100 docs/minute
- **System Requirements** (from system-requirements.md):
  - Sub-second search responses
  - Parallel processing throughput
  - Privacy-preserving (no raw content)

## Data Model

### Metric Types

Following OpenTelemetry conventions:

| Type | Description | Example |
|------|-------------|---------|
| COUNTER | Cumulative, monotonically increasing | `documents.processed` |
| GAUGE | Point-in-time value | `cpu.percent` |
| HISTOGRAM | Distribution of values | `latency` with percentiles |
| TIMER | Duration measurements | `job.duration` |

### Metric Categories

| Category | Description | Components |
|----------|-------------|------------|
| INGESTION | Document processing pipeline | Connectors, normalizer |
| EXTRACTION | Entity/relationship extraction | Triple extractor |
| SEARCH | Hybrid search operations | Query router, cache |
| ORCHESTRATOR | Job scheduling and execution | Queue, workers |
| SYSTEM | System-level resources | CPU, memory, disk |
| PRIVACY | Privacy operations | Consent, audit |

### Label Strategy

Labels enable filtering and grouping:

- `connector_type`: LOCAL_FILES, OBSIDIAN_VAULT, IMAP_MAILBOX, GITHUB_REPOSITORY
- `status`: succeeded, failed, quarantined, retry_scheduled
- `component`: orchestrator, search, normalization, extraction
- `phase`: scan, detect, process, queue (for ingestion phases)
- `intent_type`: temporal, causal, code, general (for search queries)

### Metrics Registry

Complete registry defined in `src/futurnal/telemetry/schema.py`:

**Ingestion Metrics:**
- `ingestion.documents.processed` (counter) - Total documents processed
- `ingestion.bytes.processed` (counter) - Total bytes processed
- `ingestion.throughput` (gauge) - Current docs/second
- `ingestion.latency` (histogram) - Per-document processing time

**Search Metrics:**
- `search.queries.total` (counter) - Total queries executed
- `search.latency` (histogram) - Query latency with percentiles
- `search.results.count` (histogram) - Results per query
- `search.cache.hit_rate` (gauge) - Cache hit percentage

**Orchestrator Metrics:**
- `orchestrator.jobs.queued` (gauge) - Jobs awaiting processing
- `orchestrator.jobs.active` (gauge) - Currently executing jobs
- `orchestrator.jobs.completed` (counter) - Completed jobs by status
- `orchestrator.job.duration` (histogram) - Job execution time

**System Metrics:**
- `system.cpu.percent` (gauge) - CPU usage
- `system.memory.used_mb` (gauge) - Memory usage in MB
- `system.disk.io_bytes` (counter) - Disk I/O bytes

**Error Metrics:**
- `errors.total` (counter) - Total errors by component/type
- `errors.rate` (gauge) - Errors per minute

## Implementation Files

- `src/futurnal/telemetry/__init__.py`
- `src/futurnal/telemetry/schema.py`

## Acceptance Criteria

- [ ] MetricCategory enum covers all system components
- [ ] MetricType enum follows OpenTelemetry conventions
- [ ] MetricDefinition includes name, type, category, description, unit, labels
- [ ] QualityGateThresholds aligns with quality-gates.mdc values
- [ ] METRICS_REGISTRY provides complete metric definitions
- [ ] Unit definitions follow industry conventions (ms, bytes, percent)
- [ ] Labels support all required filtering dimensions

## Test Plan

### Unit Tests
- `test_metric_definition_validation.py`: Validate schema requirements
- `test_metrics_registry_completeness.py`: All categories have metrics
- `test_quality_gate_thresholds.py`: Threshold values correct

## Implementation Notes

```python
@dataclass(frozen=True)
class MetricDefinition:
    """Immutable metric definition."""
    name: str
    metric_type: MetricType
    category: MetricCategory
    description: str
    unit: str
    labels: Tuple[str, ...] = ()  # Frozen for hashability
```

Quality gate thresholds derived from existing rules:
- `.cursor/rules/quality-gates.mdc` - Performance targets
- `src/futurnal/search/hybrid/performance/profiler.py` - Search latency targets
- `src/futurnal/orchestrator/metrics.py` - Throughput patterns

## Open Questions

- Should we support custom metrics from plugins/extensions?
- Should metric names be hierarchical (dots) or flat (underscores)?
- How to handle metrics versioning for schema changes?

## Dependencies

- None (foundation module)

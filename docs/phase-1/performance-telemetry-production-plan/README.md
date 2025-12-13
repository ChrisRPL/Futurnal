Summary: Production plan for Feature 15 Performance Telemetry Baseline - lightweight, privacy-preserving telemetry system for local monitoring.

# Performance Telemetry Baseline - Production Plan

This folder tracks the work required to ship Feature 15 (Performance Telemetry Baseline) with production-quality metrics collection, storage, visualization, and alerting. The telemetry system enables monitoring of system health without compromising privacy, aligned with the experiential learning architecture in [system-architecture.md](../../architecture/system-architecture.md).

## Task Index
- [01-metrics-taxonomy.md](01-metrics-taxonomy.md) - Metrics schema and definitions
- [02-collection-framework.md](02-collection-framework.md) - Instrumentation and collectors
- [03-time-series-storage.md](03-time-series-storage.md) - SQLite storage and aggregation
- [04-baseline-engine.md](04-baseline-engine.md) - Anomaly detection and quality gates
- [05-desktop-dashboard.md](05-desktop-dashboard.md) - Visualization and UI integration
- [06-alerting-export.md](06-alerting-export.md) - Alerting rules and CLI commands

## Technical Foundation

### Metrics Collection
**Unified MetricsCollector** - Low-overhead instrumentation
- Thread-safe buffered collection with async flush
- `@timed` decorator with <1ms overhead
- Component-specific collectors wrapping existing infrastructure
- Sampling rate configuration for high-frequency metrics

### Time-Series Storage
**SQLite with WAL mode** - Local-first, crash-recoverable storage
- Raw metrics table with JSON labels
- Aggregation tables (1m, 5m, 1h, 1d) for efficient querying
- Configurable retention policy (default: 30 days)
- Parquet export for anonymized sharing

### Baseline & Anomaly Detection
**Statistical baseline engine**
- 7-day rolling window baseline computation
- Z-score anomaly detection (threshold: 3.0 standard deviations)
- Quality gate validation against production targets
- Percentile tracking (P50, P95, P99)

### Desktop Dashboard
**Real-time visualization**
- Zustand store for telemetry settings (following existing patterns)
- React Query hooks with 2s polling for real-time updates
- Quality gate panel with pass/fail indicators
- Metric charts with historical trends

### Alerting
**Threshold-based alerting**
- Default rules: CPU, memory, search latency, throughput, error rate
- Cooldown mechanism to prevent alert storms
- Desktop notification integration via Tauri
- CLI commands for operator visibility

## Architectural Patterns

Following established patterns from other production plans:

1. **Privacy-First Design**
   - No raw content in metrics (paths redacted)
   - Labels anonymized in exports
   - Retention policy honors SecuritySettings
   - Local-only storage by default

2. **Integration with Existing Infrastructure**
   - Wraps TelemetryRecorder (src/futurnal/orchestrator/metrics.py)
   - Wraps ResourceMonitor (src/futurnal/orchestrator/resource_monitor.py)
   - Wraps PerformanceProfiler (src/futurnal/search/hybrid/performance/profiler.py)
   - Follows SyncMetricsCollector patterns (src/futurnal/ingestion/obsidian/sync_metrics.py)

3. **Quality Gates**
   - Throughput: >5 documents/second
   - Memory: <2GB for extraction pipeline
   - Latency: <2s per document
   - Search P95: <1000ms
   - Collection overhead: <5%

4. **Observability**
   - CLI `futurnal telemetry status` for quick health check
   - Dashboard widget with real-time metrics
   - Alert history with acknowledgment tracking
   - Export functionality for troubleshooting

## Current Implementation Status

### Complete
- (none yet)

### In Progress
- Module 01: Metrics Taxonomy & Schema

### Pending
- Module 02: Collection Framework
- Module 03: Time-Series Storage
- Module 04: Baseline Engine
- Module 05: Desktop Dashboard
- Module 06: Alerting & Export

## Dependencies

- **Feature 5**: Ingestion Orchestrator (Complete)
- **Feature 6**: Document Normalization Pipeline (Complete)
- **Feature 10**: Hybrid Search API (Complete)
- **Feature 13**: Privacy & Audit Logging (Complete)

## AI Learning Focus

Transform telemetry into experiential learning enabler:

- **Performance Visibility**: Monitor AI learning pipeline throughput and resource usage
- **Quality Tracking**: Validate extraction precision, temporal accuracy, and schema alignment
- **Anomaly Detection**: Surface performance degradation before it impacts learning quality
- **Operator Empowerment**: Provide actionable metrics for tuning and optimization

The telemetry system provides the instrumentation needed to ensure the Ghost's experiential learning pipeline operates at peak efficiency.

## References

- **Feature Spec**: [docs/phase-1/feature-performance-telemetry.md](../feature-performance-telemetry.md)
- **Quality Gates**: [.cursor/rules/quality-gates.mdc](../../../.cursor/rules/quality-gates.mdc)
- **System Requirements**: [requirements/system-requirements.md](../../requirements/system-requirements.md)
- **Architecture**: [architecture/system-architecture.md](../../architecture/system-architecture.md)

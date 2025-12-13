Summary: Implement unified metrics collection framework with minimal overhead (<5%) collectors.

# 02 - Collection Framework

## Purpose

Instrument existing components (orchestrator, normalization, search API) with minimal overhead collectors that feed into the unified telemetry system.

## Scope

- Unified BaseCollector abstract class
- MetricsBuffer for thread-safe async collection
- `@timed` decorator for automatic timing
- Component-specific collectors wrapping existing infrastructure
- <5% performance overhead guarantee

## Requirements Alignment

- **Minimal overhead**: <5% performance impact (from feature spec)
- **Privacy-preserving**: No raw content captured
- **Integration**: Build on existing TelemetryRecorder, ResourceMonitor, PerformanceProfiler

## Implementation Files

- `src/futurnal/telemetry/collector.py` (new)
- `src/futurnal/telemetry/collectors/orchestrator_collector.py` (new)
- `src/futurnal/telemetry/collectors/search_collector.py` (new)
- `src/futurnal/telemetry/collectors/normalization_collector.py` (new)

## Integration Points (existing files)

- `src/futurnal/orchestrator/metrics.py` - TelemetryRecorder
- `src/futurnal/orchestrator/resource_monitor.py` - ResourceMonitor
- `src/futurnal/search/hybrid/performance/profiler.py` - PerformanceProfiler
- `src/futurnal/pipeline/normalization/performance.py` - PerformanceMonitor

## Data Model

### CollectorConfig
```python
@dataclass
class CollectorConfig:
    buffer_size: int = 10000
    flush_interval_seconds: float = 5.0
    sampling_rate: float = 1.0  # 1.0 = all, 0.1 = 10%
    enabled: bool = True
    privacy_mode: bool = True
```

### BaseCollector
```python
class BaseCollector(ABC):
    def __init__(self, config: CollectorConfig = None)
    @abstractmethod
    def component_name(self) -> str
    async def record(metric_name, value, labels) -> None
    def increment(metric_name, value=1.0) -> None  # Sync counter
    def set_gauge(metric_name, value) -> None  # Sync gauge
    @contextmanager
    def timer(metric_name, labels) -> Generator
```

### UnifiedMetricsCollector
```python
class UnifiedMetricsCollector(BaseCollector):
    def register_component(name: str, collector: BaseCollector) -> None
    def register_sink(sink: MetricsSink) -> None
    async def flush_all() -> int
```

### @timed decorator
```python
@timed("search.latency", labels={"strategy": "hybrid"})
async def execute_search(query: str):
    ...
```

## Component Design

### OrchestratorCollector
Wraps existing TelemetryRecorder and ResourceMonitor:
- Records job completion via TelemetryRecorder (backward compatible)
- Also records to unified MetricsSample system
- Captures resource snapshots from ResourceMonitor

### SearchCollector
Wraps existing PerformanceProfiler:
- Records query metrics via profiler (backward compatible)
- Also records to unified system
- Tracks cache hits/misses

### NormalizationCollector
Wraps existing PerformanceMonitor:
- Records document processing metrics
- Tracks per-format statistics

## Acceptance Criteria

- [ ] BaseCollector provides record(), increment(), set_gauge(), timer()
- [ ] MetricsBuffer is thread-safe with async flush
- [ ] UnifiedMetricsCollector aggregates all components
- [ ] OrchestratorCollector wraps TelemetryRecorder
- [ ] SearchCollector wraps PerformanceProfiler
- [ ] @timed decorator adds <1ms overhead
- [ ] Collection overhead <5% in benchmark tests
- [ ] Privacy mode redacts paths/content

## Test Plan

### Unit Tests
- `test_collector_buffer.py`: Buffer operations, thread safety
- `test_timer_decorator.py`: Timing accuracy, overhead
- `test_collector_registration.py`: Component registration

### Integration Tests
- `test_orchestrator_collector_integration.py`: TelemetryRecorder bridging
- `test_search_collector_integration.py`: PerformanceProfiler bridging

### Performance Tests
- `test_collection_overhead.py`: <5% overhead validation

## Dependencies

- Module 01: Metrics Taxonomy & Schema
- Existing: TelemetryRecorder, ResourceMonitor, PerformanceProfiler

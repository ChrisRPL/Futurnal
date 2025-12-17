# Monitoring Guide

**Futurnal v1.0.0 - Phase 1 (Archivist)**

This guide covers monitoring and telemetry for production Futurnal deployments.

---

## Health Checks

### CLI Health Command

Run comprehensive health checks:

```bash
futurnal health check
```

Output:
```
System Health: ✓ HEALTHY
----------------------------------------
  ✓ ollama: Connected, 3 models available (45ms)
  ✓ neo4j: Connected and responsive (120ms)
  ✓ chromadb: Initialized, 5 collections (15ms)
  ✓ disk_space: 45.2GB free of 500.0GB
  ✓ memory: 8.5GB available of 16.0GB
  ✓ config: Configuration valid
----------------------------------------
Checked at: 2024-12-17 10:30:00
```

### Component Checks

Check individual components:

```bash
# Check Ollama
futurnal health check --component ollama

# Check Neo4j
futurnal health check --component neo4j

# Check disk space
futurnal health check --component disk_space
```

### Health Status Levels

| Status | Meaning | Action |
|--------|---------|--------|
| HEALTHY | All systems normal | None |
| DEGRADED | Some issues detected | Review warnings |
| UNHEALTHY | Critical issues | Immediate attention |
| UNKNOWN | Check failed | Investigate |

---

## Telemetry System

### Privacy Guarantees

Futurnal telemetry is designed with privacy as the core principle:

1. **Disabled by default** - Requires explicit opt-in
2. **No PII collection** - Never collects personal data
3. **No content logging** - No queries, no file paths
4. **Local storage** - All data stays on your machine
5. **User control** - Export or delete anytime

### What Is Collected (When Enabled)

| Metric | Example | NOT Collected |
|--------|---------|---------------|
| Search latency | `250ms` | Query text |
| Chat response time | `2.1s` | Message content |
| Ingestion throughput | `8 docs/sec` | File paths |
| Error types | `timeout` | Error messages |
| Cache hit rate | `85%` | Cache contents |

### Enabling Telemetry

**Via Settings UI:**
1. Open Settings → Telemetry
2. Toggle "Enable Telemetry"
3. Select which categories to collect

**Via CLI:**
```bash
futurnal config set telemetry.enabled true
```

**Via Configuration:**
```yaml
# ~/.futurnal/config.yaml
telemetry:
  enabled: true
  retention_days: 7
  categories:
    performance: true
    errors: true
    usage: true
```

---

## Performance Metrics

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| search_latency_p50 | <500ms | 50th percentile search time |
| search_latency_p95 | <1000ms | 95th percentile search time |
| chat_latency_p50 | <2000ms | 50th percentile chat response |
| chat_latency_p95 | <3000ms | 95th percentile chat response |
| ingestion_throughput | >5 docs/s | Documents processed per second |
| cache_hit_rate | >70% | Percentage of cache hits |
| memory_usage | <2GB | Peak memory usage |

### Viewing Metrics

**CLI Summary:**
```bash
futurnal telemetry summary
```

Output:
```
Telemetry Summary
-----------------
Session Duration: 3h 45m
Operations:
  search: 127
  chat: 43
  ingestion: 892

Latency (ms):
  search: avg=320, p95=780
  chat: avg=1850, p95=2900

Errors: 3 total
  timeout: 2
  connection: 1
```

**Export to JSON:**
```bash
futurnal telemetry export --format json --output metrics.json
```

---

## System Health Indicators

### Memory Usage

Monitor memory to prevent OOM issues:

```bash
futurnal health memory
```

**Thresholds:**
- HEALTHY: >2GB available
- DEGRADED: 1-2GB available
- UNHEALTHY: <1GB available

**Memory Optimization:**
```yaml
# ~/.futurnal/config.yaml
performance:
  cache:
    max_entries: 500  # Reduce if memory limited
    ttl_seconds: 300
  batch_size: 10  # Smaller batches use less memory
```

### Disk Space

Monitor disk space for data storage:

```bash
futurnal health disk
```

**Thresholds:**
- HEALTHY: >5GB free
- DEGRADED: 1-5GB free
- UNHEALTHY: <1GB free

**Disk Cleanup:**
```bash
# Clear old cache
futurnal cache clear --older-than 7d

# Compact graph database
futurnal admin compact-graph
```

### Service Health

Check external service connectivity:

```bash
# Ollama (required)
futurnal health ollama

# Neo4j (required)
futurnal health neo4j

# ChromaDB (required)
futurnal health chromadb
```

---

## Alerting (Future)

### Alert Configuration

When alerting is enabled:

```yaml
# ~/.futurnal/config.yaml
alerting:
  enabled: false  # Coming in Phase 2
  rules:
    - metric: search_latency_p95
      threshold: 2000
      severity: warning
    - metric: memory_usage
      threshold: 3000  # MB
      severity: critical
```

### Alert Channels

Planned for Phase 2:
- Desktop notifications
- Log file alerts
- Optional webhook integration

---

## Baseline & Anomaly Detection

### Automatic Baselines

The system learns normal performance patterns:

```bash
futurnal telemetry baseline show
```

Output:
```
Baselines (7-day learning period)
---------------------------------
search_latency:
  baseline: 280ms ± 85ms
  current: 320ms (normal)

chat_latency:
  baseline: 1800ms ± 400ms
  current: 2100ms (normal)
```

### Anomaly Detection

Anomalies are flagged when metrics exceed baselines:

```bash
futurnal telemetry anomalies
```

Output:
```
Recent Anomalies
----------------
[2024-12-17 09:15] search_latency spike: 1250ms (4.5σ)
  Possible cause: Large result set

[2024-12-17 08:30] ingestion_throughput drop: 2.1 docs/s
  Possible cause: Network latency
```

---

## Quality Gates Integration

### Monitoring Quality Gates

Quality gates are monitored continuously:

```bash
futurnal quality status
```

Output:
```
Quality Gates Status
--------------------
✓ temporal_accuracy: 88% (target: >85%)
✓ schema_alignment: 94% (target: >90%)
✓ extraction_precision: 0.85 (target: >=0.8)
✓ ghost_frozen: true (required)
✓ causal_ordering: 100% valid
```

### Quality Metrics in Telemetry

Quality metrics are included in telemetry (when enabled):

```json
{
  "category": "quality",
  "event_type": "gate_check",
  "metrics": {
    "gate_name": "temporal_accuracy",
    "value": 0.88,
    "passed": true
  }
}
```

---

## Troubleshooting

### Common Issues

**High Search Latency:**
1. Check cache hit rate: `futurnal telemetry summary`
2. Warm up cache: `futurnal cache warmup`
3. Check Neo4j health: `futurnal health neo4j`

**Memory Issues:**
1. Check memory usage: `futurnal health memory`
2. Clear cache: `futurnal cache clear`
3. Reduce batch size in config

**Service Connection Failures:**
1. Check Ollama: `curl http://localhost:11434/api/tags`
2. Check Neo4j: `futurnal health neo4j`
3. Review logs: `futurnal logs tail`

### Diagnostic Commands

```bash
# Full system diagnostic
futurnal health check --verbose

# Export diagnostic report
futurnal health report --output diagnostic.json

# View recent logs
futurnal logs tail --lines 100

# Check configuration
futurnal config validate
```

---

## Best Practices

### Regular Monitoring

1. Run `futurnal health check` daily
2. Review telemetry summary weekly
3. Monitor disk space before large ingestions
4. Check quality gates after schema changes

### Performance Optimization

1. Keep cache warm for frequently used queries
2. Run ingestion during off-peak hours
3. Monitor memory during large operations
4. Compact databases periodically

### Privacy Compliance

1. Review telemetry data before enabling
2. Set appropriate retention periods
3. Export and review data periodically
4. Clear telemetry when no longer needed

---

## Reference

### Health Check API

```python
from futurnal.telemetry import HealthChecker, format_health_report

checker = HealthChecker()
health = await checker.check_all()

print(format_health_report(health, verbose=True))
```

### Metrics API

```python
from futurnal.telemetry import MetricsCollector

collector = MetricsCollector(enabled=True)

# Record search
collector.record_search(
    search_type="hybrid",
    result_count=10,
    latency_ms=250,
    cache_hit=True,
)

# Get summary
summary = collector.get_summary()
```

### Timed Operations

```python
from futurnal.telemetry import timed_operation, MetricCategory

@timed_operation("search", MetricCategory.SEARCH, collector)
async def search(query: str) -> List[Result]:
    # Latency automatically recorded
    ...
```

---

*Part of Step 10: Production Readiness*
*Phase 1 (Archivist) - December 2024*

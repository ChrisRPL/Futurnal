# Module 06: Alerting & Export

## Scope

Implement threshold-based alerting and data export capabilities for the telemetry system. This module provides:
- Alert rule definitions with configurable thresholds
- AlertManager for rule evaluation and notification
- Cooldown mechanism to prevent alert storms
- Data export to various formats (Parquet, CSV, JSON)
- CLI commands for telemetry management

## Data Model

### AlertRule
```python
@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: Literal['>', '<', '>=', '<=', '==']
    threshold: float
    severity: Literal['info', 'warning', 'critical']
    cooldown_seconds: int = 300  # 5 minutes
    labels: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    enabled: bool = True
```

### Alert
```python
@dataclass
class Alert:
    id: int
    rule_name: str
    metric_name: str
    alert_type: str
    severity: Literal['info', 'warning', 'critical']
    value: float
    threshold: float
    message: str
    labels: Dict[str, str]
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
```

## Implementation

### Default Alert Rules

| Rule Name | Metric | Condition | Threshold | Severity |
|-----------|--------|-----------|-----------|----------|
| high_cpu | system.cpu.percent | > | 80% | warning |
| critical_cpu | system.cpu.percent | > | 95% | critical |
| high_memory | system.memory.used_mb | > | 2048 MB | warning |
| critical_memory | system.memory.used_mb | > | 2560 MB | critical |
| slow_search | search.latency.p95 | > | 1000 ms | warning |
| very_slow_search | search.latency.p95 | > | 2000 ms | critical |
| low_throughput | ingestion.throughput | < | 5 docs/s | warning |
| very_low_throughput | ingestion.throughput | < | 1 docs/s | critical |
| high_error_rate | errors.rate | > | 0.05 | warning |
| critical_error_rate | errors.rate | > | 0.10 | critical |

### AlertManager Class

```python
class AlertManager:
    def __init__(
        self,
        storage: TimeSeriesStorage,
        config: Optional[AlertConfig] = None,
    ):
        """Initialize alert manager with storage backend."""

    def add_rule(self, rule: AlertRule) -> None:
        """Add custom alert rule."""

    def remove_rule(self, rule_name: str) -> bool:
        """Remove alert rule."""

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate all rules against current metrics.

        Returns list of triggered alerts, respecting cooldowns.
        """

    def acknowledge(self, alert_id: int) -> bool:
        """Acknowledge an alert."""

    def get_active_alerts(self) -> List[Alert]:
        """Get unacknowledged alerts."""

    def set_notification_callback(
        self,
        callback: Callable[[Alert], None]
    ) -> None:
        """Set callback for alert notifications."""
```

### Export Functions

```python
async def export_telemetry(
    storage: TimeSeriesStorage,
    output_path: Path,
    format: Literal['parquet', 'csv', 'json'],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    metrics: Optional[List[str]] = None,
    anonymize: bool = True,
) -> ExportResult:
    """Export telemetry data to file.

    Args:
        storage: Storage backend to export from.
        output_path: Path for output file.
        format: Export format.
        start_time: Start of time range.
        end_time: End of time range.
        metrics: Specific metrics to export (all if None).
        anonymize: Whether to anonymize labels.

    Returns:
        ExportResult with file path and record count.
    """
```

## CLI Commands

### telemetry status
```bash
futurnal telemetry status
```
Display current metrics and quality gate status.

### telemetry alerts
```bash
futurnal telemetry alerts [--severity SEVERITY] [--ack ID]
```
List recent alerts or acknowledge an alert.

### telemetry export
```bash
futurnal telemetry export [--format FORMAT] [--output PATH] [--anonymize]
```
Export telemetry data to file.

### telemetry retention
```bash
futurnal telemetry retention [--days DAYS]
```
View or set retention policy.

### telemetry baseline
```bash
futurnal telemetry baseline METRIC [--days DAYS]
```
Compute baseline for a metric.

## Integration Points

### With Storage Module
- Stores and retrieves alerts from `alerts` table
- Exports data from raw and aggregated tables

### With Baseline Engine
- Can generate anomaly-based alerts
- Uses quality gate violations for alerts

### With Desktop Dashboard
- Notification callback for desktop alerts
- Real-time alert status updates

## Acceptance Criteria

- [ ] All default alert rules implemented
- [ ] Cooldown mechanism prevents alert storms
- [ ] Export supports Parquet, CSV, JSON formats
- [ ] Anonymization removes sensitive label values
- [ ] CLI commands work correctly
- [ ] Test coverage >=90%

## Test Plan

### Unit Tests
- `test_alert_rule_evaluation` - Rule conditions work
- `test_alert_cooldown` - Cooldown prevents duplicates
- `test_export_formats` - All formats produce valid output
- `test_anonymization` - Labels are properly anonymized

### Integration Tests
- `test_alert_persistence` - Alerts stored and retrieved
- `test_export_roundtrip` - Exported data can be read back
- `test_cli_commands` - CLI commands execute correctly

## Configuration

```python
@dataclass
class AlertConfig:
    default_cooldown_seconds: int = 300
    max_active_alerts: int = 100
    enable_notifications: bool = True
    notification_severity_threshold: str = 'warning'
```

## Files

- `src/futurnal/telemetry/alerting.py` - Alert manager and rules
- `src/futurnal/telemetry/export.py` - Export functionality
- `src/futurnal/cli/telemetry_commands.py` - CLI commands
- `tests/telemetry/test_alerting.py` - Test suite

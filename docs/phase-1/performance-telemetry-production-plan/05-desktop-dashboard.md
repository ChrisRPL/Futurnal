# Module 05: Desktop Dashboard Integration

## Scope

Extend the desktop shell with telemetry visualization components. This module provides:
- TypeScript types matching Python telemetry schema
- Tauri IPC API layer for telemetry data
- Zustand store for telemetry settings and state
- React Query hooks for data fetching
- Dashboard components for visualization

## Data Model

### TypeScript Types (matching Python schema)

```typescript
// Metric types
type MetricCategory = 'ingestion' | 'extraction' | 'search' | 'orchestrator' | 'system' | 'privacy';
type MetricType = 'counter' | 'gauge' | 'histogram' | 'timer';

interface MetricSample {
  metric_name: string;
  value: number;
  timestamp: string;
  labels: Record<string, string>;
}

// Baseline types
type AnomalyType = 'spike' | 'drop' | 'trend';
type AnomalySeverity = 'info' | 'warning' | 'critical';

interface Baseline {
  metric_name: string;
  labels: Record<string, string>;
  mean: number;
  stddev: number;
  p50: number;
  p95: number;
  p99: number;
}

interface Anomaly {
  metric_name: string;
  anomaly_type: AnomalyType;
  severity: AnomalySeverity;
  value: number;
  expected: number;
  z_score: number;
  detected_at: string;
}

// Quality gates
interface QualityGateResult {
  gate_name: string;
  passing: boolean;
  current_value: number;
  threshold: number;
  metric_name: string;
}

// Alerts
interface TelemetryAlert {
  id: number;
  metric_name: string;
  alert_type: string;
  severity: 'info' | 'warning' | 'critical';
  value: number;
  threshold: number;
  message: string;
  triggered_at: string;
  acknowledged_at?: string;
}

// Snapshot (current state)
interface TelemetrySnapshot {
  latest_values: Record<string, number>;
  quality_gates: QualityGateResult[];
  active_alerts: TelemetryAlert[];
  storage_stats: StorageStats;
}
```

## Implementation

### API Layer (`lib/telemetryApi.ts`)

```typescript
export const telemetryApi = {
  // Get current telemetry snapshot
  async getSnapshot(): Promise<TelemetrySnapshot>;

  // Get historical metrics
  async getMetrics(params: MetricsQuery): Promise<MetricSample[]>;

  // Get quality gate status
  async getQualityGates(): Promise<QualityGateResult[]>;

  // Get alerts
  async getAlerts(params?: AlertsQuery): Promise<TelemetryAlert[]>;

  // Acknowledge an alert
  async acknowledgeAlert(alertId: number): Promise<boolean>;

  // Export telemetry data
  async exportTelemetry(params: ExportParams): Promise<string>;
};
```

### Zustand Store (`stores/telemetryStore.ts`)

```typescript
interface TelemetryState {
  // Settings
  enabled: boolean;
  retentionDays: number;
  showDashboardWidget: boolean;
  alertNotifications: boolean;

  // Custom thresholds (override defaults)
  thresholds: {
    throughputMin: number;
    memoryMax: number;
    latencyMax: number;
    searchP95Max: number;
    errorRateMax: number;
  };

  // Actions
  setSetting: <K>(key: K, value: TelemetryState[K]) => void;
  setThreshold: (name: string, value: number) => void;
  resetSettings: () => void;
}
```

### React Query Hooks (`hooks/useTelemetry.ts`)

```typescript
// Get telemetry snapshot (polls every 2s)
export function useTelemetrySnapshot();

// Get historical metrics
export function useTelemetryMetrics(query: MetricsQuery);

// Get quality gate status
export function useQualityGates();

// Get alerts
export function useTelemetryAlerts(query?: AlertsQuery);

// Acknowledge alert mutation
export function useAcknowledgeAlert();

// Export telemetry mutation
export function useExportTelemetry();
```

## File Structure

```
desktop/src/
├── types/
│   └── telemetry.ts           # TypeScript interfaces
├── lib/
│   └── telemetryApi.ts        # Tauri IPC wrappers
├── stores/
│   └── telemetryStore.ts      # Zustand store
└── hooks/
    └── useTelemetry.ts        # React Query hooks
```

## IPC Commands Required

The following Tauri IPC commands need to be implemented in Rust:

| Command | Parameters | Returns |
|---------|------------|---------|
| `get_telemetry_snapshot` | none | `TelemetrySnapshot` |
| `get_telemetry_metrics` | `MetricsQuery` | `MetricSample[]` |
| `get_quality_gates` | none | `QualityGateResult[]` |
| `get_telemetry_alerts` | `AlertsQuery?` | `TelemetryAlert[]` |
| `acknowledge_alert` | `alertId: number` | `boolean` |
| `export_telemetry` | `ExportParams` | `string` (file path) |

## Acceptance Criteria

- [ ] TypeScript types match Python schema
- [ ] API layer handles errors gracefully with fallbacks
- [ ] Store persists settings to localStorage
- [ ] Hooks provide real-time data with polling
- [ ] Dashboard renders in <100ms
- [ ] Alert notifications integrate with system

## Test Plan

### Unit Tests
- `test_telemetry_api_error_handling` - Graceful error fallbacks
- `test_telemetry_store_persistence` - Settings persist correctly
- `test_telemetry_hooks_polling` - Data refreshes at interval

### Integration Tests
- `test_snapshot_to_ui` - End-to-end data flow
- `test_alert_acknowledgement` - Alert workflow

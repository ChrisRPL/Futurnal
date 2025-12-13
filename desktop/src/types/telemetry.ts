/**
 * Futurnal Desktop Shell - Telemetry Types
 *
 * TypeScript interfaces matching the Python telemetry schema.
 * These types ensure type safety between the frontend and backend.
 */

// ============================================================================
// Metric Types
// ============================================================================

/**
 * Metric category for classification.
 */
export type MetricCategory =
  | 'ingestion'
  | 'extraction'
  | 'search'
  | 'orchestrator'
  | 'system'
  | 'privacy';

/**
 * Type of metric (how values should be interpreted).
 */
export type MetricType = 'counter' | 'gauge' | 'histogram' | 'timer';

/**
 * A single metric sample from the telemetry system.
 */
export interface MetricSample {
  metric_name: string;
  value: number;
  timestamp: string;
  labels: Record<string, string>;
}

/**
 * Definition of a metric in the registry.
 */
export interface MetricDefinition {
  name: string;
  metric_type: MetricType;
  category: MetricCategory;
  description: string;
  unit: string;
  labels: string[];
}

// ============================================================================
// Baseline & Anomaly Types
// ============================================================================

/**
 * Type of detected anomaly.
 */
export type AnomalyType = 'spike' | 'drop' | 'trend';

/**
 * Severity level of anomaly or alert.
 */
export type AnomalySeverity = 'info' | 'warning' | 'critical';

/**
 * Statistical baseline for a metric.
 */
export interface Baseline {
  metric_name: string;
  labels: Record<string, string>;
  mean: number;
  stddev: number;
  p50: number;
  p95: number;
  p99: number;
  min_value: number;
  max_value: number;
  sample_count: number;
  computed_at: string;
}

/**
 * Detected anomaly in metric values.
 */
export interface Anomaly {
  metric_name: string;
  labels: Record<string, string>;
  anomaly_type: AnomalyType;
  severity: AnomalySeverity;
  value: number;
  expected: number;
  z_score: number;
  detected_at: string;
}

// ============================================================================
// Quality Gate Types
// ============================================================================

/**
 * Result of quality gate evaluation.
 */
export interface QualityGateResult {
  gate_name: string;
  passing: boolean;
  current_value: number;
  threshold: number;
  metric_name: string;
  evaluated_at: string;
}

/**
 * Quality gate definition.
 */
export interface QualityGateDefinition {
  name: string;
  metric: string;
  condition: '>=' | '<=' | '>' | '<' | '==';
  threshold: number;
  description: string;
}

/**
 * Default quality gates from quality-gates.mdc.
 */
export const QUALITY_GATES: Record<string, QualityGateDefinition> = {
  throughput: {
    name: 'throughput',
    metric: 'ingestion.throughput',
    condition: '>=',
    threshold: 5.0,
    description: 'Ingestion throughput >= 5 docs/second',
  },
  memory: {
    name: 'memory',
    metric: 'system.memory.used_mb',
    condition: '<=',
    threshold: 2048.0,
    description: 'Memory usage <= 2GB',
  },
  latency: {
    name: 'latency',
    metric: 'ingestion.latency',
    condition: '<=',
    threshold: 2000.0,
    description: 'Document processing latency <= 2s',
  },
  search_p95: {
    name: 'search_p95',
    metric: 'search.latency.p95',
    condition: '<=',
    threshold: 1000.0,
    description: 'Search P95 latency <= 1000ms',
  },
  error_rate: {
    name: 'error_rate',
    metric: 'errors.rate',
    condition: '<=',
    threshold: 0.05,
    description: 'Error rate <= 5%',
  },
};

// ============================================================================
// Alert Types
// ============================================================================

/**
 * Telemetry alert from the alerting system.
 */
export interface TelemetryAlert {
  id: number;
  metric_name: string;
  alert_type: string;
  severity: AnomalySeverity;
  value: number;
  threshold: number;
  message: string;
  labels: Record<string, string>;
  triggered_at: string;
  acknowledged_at?: string;
}

// ============================================================================
// Snapshot & Query Types
// ============================================================================

/**
 * Current telemetry snapshot with latest values and status.
 */
export interface TelemetrySnapshot {
  /** Latest value for each metric */
  latest_values: Record<string, number>;
  /** Quality gate evaluation results */
  quality_gates: QualityGateResult[];
  /** Currently active (unacknowledged) alerts */
  active_alerts: TelemetryAlert[];
  /** Storage statistics */
  storage_stats: StorageStats;
  /** Timestamp of snapshot */
  timestamp: string;
}

/**
 * Storage statistics.
 */
export interface StorageStats {
  metrics_raw_count: number;
  metrics_1h_count: number;
  alerts_count: number;
  distinct_metrics: number;
  db_path: string;
  retention_days: number;
}

/**
 * Query parameters for historical metrics.
 */
export interface MetricsQuery {
  metric_name: string;
  start_time?: string;
  end_time?: string;
  labels?: Record<string, string>;
  aggregation?: '1m' | '5m' | '1h' | '1d';
  limit?: number;
}

/**
 * Query parameters for alerts.
 */
export interface AlertsQuery {
  severity?: AnomalySeverity;
  acknowledged?: boolean;
  limit?: number;
  start_time?: string;
  end_time?: string;
}

/**
 * Parameters for telemetry export.
 */
export interface ExportParams {
  start_time?: string;
  end_time?: string;
  format: 'parquet' | 'csv' | 'json';
  anonymize?: boolean;
  metrics?: string[];
}

// ============================================================================
// Dashboard Display Types
// ============================================================================

/**
 * Time range presets for dashboard.
 */
export type TimeRangePreset = '1h' | '6h' | '24h' | '7d' | '30d';

/**
 * Time range for metrics display.
 */
export interface TimeRange {
  start: string;
  end: string;
  preset?: TimeRangePreset;
}

/**
 * Chart data point for visualization.
 */
export interface ChartDataPoint {
  timestamp: string;
  value: number;
}

/**
 * Chart series for multi-line charts.
 */
export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
}

/**
 * Get preset time range.
 */
export function getTimeRangeFromPreset(preset: TimeRangePreset): TimeRange {
  const now = new Date();
  const end = now.toISOString();
  let start: Date;

  switch (preset) {
    case '1h':
      start = new Date(now.getTime() - 60 * 60 * 1000);
      break;
    case '6h':
      start = new Date(now.getTime() - 6 * 60 * 60 * 1000);
      break;
    case '24h':
      start = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      break;
    case '7d':
      start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      break;
    case '30d':
      start = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      break;
  }

  return {
    start: start.toISOString(),
    end,
    preset,
  };
}

/**
 * Format duration in milliseconds to human-readable string.
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

/**
 * Format bytes to human-readable string.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

/**
 * Get severity color for UI.
 */
export function getSeverityColor(severity: AnomalySeverity): string {
  switch (severity) {
    case 'info':
      return 'blue';
    case 'warning':
      return 'yellow';
    case 'critical':
      return 'red';
    default:
      return 'gray';
  }
}

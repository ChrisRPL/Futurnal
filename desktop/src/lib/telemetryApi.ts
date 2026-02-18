/**
 * Futurnal Desktop Shell - Telemetry API Client
 *
 * This module provides typed wrappers around Tauri's invoke function
 * for calling telemetry-related IPC commands.
 */

import { invoke } from '@tauri-apps/api/core';
import type {
  TelemetrySnapshot,
  MetricSample,
  MetricsQuery,
  QualityGateResult,
  TelemetryAlert,
  AlertsQuery,
  ExportParams,
  StorageStats,
} from '@/types/telemetry';

/**
 * Default timeout for telemetry API calls in milliseconds.
 */
const DEFAULT_TIMEOUT_MS = 10000;

/**
 * Wrapper around invoke with timeout handling.
 */
async function invokeWithTimeout<T>(
  command: string,
  args?: Record<string, unknown>,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`Request timeout: ${command}`)), timeoutMs);
  });

  try {
    const result = await Promise.race([invoke<T>(command, args), timeout]);
    return result;
  } catch (error) {
    throw error instanceof Error ? error : new Error(String(error));
  }
}

/**
 * Get mock snapshot for UI testing when backend is unavailable.
 */
function getMockSnapshot(): TelemetrySnapshot {
  const now = new Date().toISOString();
  return {
    latest_values: {
      'ingestion.throughput': 7.5,
      'ingestion.documents.processed': 1250,
      'ingestion.bytes.processed': 52428800,
      'search.latency.p95': 450,
      'search.queries.total': 89,
      'search.cache.hit_rate': 0.72,
      'system.cpu.percent': 35.5,
      'system.memory.used_mb': 1024,
      'errors.total': 3,
    },
    quality_gates: [
      {
        gate_name: 'throughput',
        passing: true,
        current_value: 7.5,
        threshold: 5.0,
        metric_name: 'ingestion.throughput',
        evaluated_at: now,
      },
      {
        gate_name: 'memory',
        passing: true,
        current_value: 1024,
        threshold: 2048,
        metric_name: 'system.memory.used_mb',
        evaluated_at: now,
      },
      {
        gate_name: 'latency',
        passing: true,
        current_value: 850,
        threshold: 2000,
        metric_name: 'ingestion.latency',
        evaluated_at: now,
      },
      {
        gate_name: 'search_p95',
        passing: true,
        current_value: 450,
        threshold: 1000,
        metric_name: 'search.latency.p95',
        evaluated_at: now,
      },
      {
        gate_name: 'error_rate',
        passing: true,
        current_value: 0.02,
        threshold: 0.05,
        metric_name: 'errors.rate',
        evaluated_at: now,
      },
    ],
    active_alerts: [],
    storage_stats: {
      metrics_raw_count: 15000,
      metrics_1h_count: 720,
      alerts_count: 12,
      distinct_metrics: 25,
      db_path: '~/.futurnal/telemetry/telemetry.db',
      retention_days: 30,
    },
    timestamp: now,
  };
}

/**
 * Get mock metrics for UI testing when backend is unavailable.
 */
function getMockMetrics(query: MetricsQuery): MetricSample[] {
  const samples: MetricSample[] = [];
  const now = Date.now();

  // Generate 60 samples over the last hour
  for (let i = 59; i >= 0; i--) {
    const timestamp = new Date(now - i * 60 * 1000);
    let value: number;

    // Generate realistic-looking data based on metric name
    if (query.metric_name.includes('throughput')) {
      value = 5 + Math.random() * 5 + Math.sin(i / 10) * 2;
    } else if (query.metric_name.includes('latency')) {
      value = 300 + Math.random() * 300 + Math.sin(i / 15) * 100;
    } else if (query.metric_name.includes('memory')) {
      value = 800 + Math.random() * 400 + i * 2;
    } else if (query.metric_name.includes('cpu')) {
      value = 20 + Math.random() * 40 + Math.sin(i / 8) * 15;
    } else {
      value = Math.random() * 100;
    }

    samples.push({
      metric_name: query.metric_name,
      value,
      timestamp: timestamp.toISOString(),
      labels: query.labels || {},
    });
  }

  return samples;
}

/**
 * Telemetry API methods.
 */
export const telemetryApi = {
  /**
   * Get current telemetry snapshot with latest values and status.
   * Falls back to mock data if backend is unavailable.
   */
  async getSnapshot(): Promise<TelemetrySnapshot> {
    try {
      return await invokeWithTimeout<TelemetrySnapshot>('get_telemetry_snapshot');
    } catch {
      console.info('[Telemetry API] Using mock data - backend unavailable');
      return getMockSnapshot();
    }
  },

  /**
   * Get historical metrics for a given query.
   * Falls back to mock data if backend is unavailable.
   */
  async getMetrics(query: MetricsQuery): Promise<MetricSample[]> {
    try {
      return await invokeWithTimeout<MetricSample[]>('get_telemetry_metrics', { query });
    } catch {
      console.info('[Telemetry API] Using mock metrics - backend unavailable');
      return getMockMetrics(query);
    }
  },

  /**
   * Get quality gate evaluation results.
   * Falls back to snapshot quality gates if direct call fails.
   */
  async getQualityGates(): Promise<QualityGateResult[]> {
    try {
      return await invokeWithTimeout<QualityGateResult[]>('get_quality_gates');
    } catch {
      // Fall back to getting from snapshot
      const snapshot = await this.getSnapshot();
      return snapshot.quality_gates;
    }
  },

  /**
   * Get telemetry alerts.
   */
  async getAlerts(query?: AlertsQuery): Promise<TelemetryAlert[]> {
    try {
      return await invokeWithTimeout<TelemetryAlert[]>('get_telemetry_alerts', { query });
    } catch {
      console.info('[Telemetry API] Using empty alerts - backend unavailable');
      return [];
    }
  },

  /**
   * Acknowledge an alert.
   */
  async acknowledgeAlert(alertId: number): Promise<boolean> {
    try {
      return await invokeWithTimeout<boolean>('acknowledge_alert', { alertId });
    } catch (error) {
      console.error('[Telemetry API] Failed to acknowledge alert:', error);
      return false;
    }
  },

  /**
   * Export telemetry data.
   * Returns the path to the exported file.
   */
  async exportTelemetry(params: ExportParams): Promise<string> {
    return invokeWithTimeout<string>(
      'export_telemetry',
      { params },
      60000 // 1 minute timeout for export
    );
  },

  /**
   * Get storage statistics.
   */
  async getStorageStats(): Promise<StorageStats> {
    try {
      return await invokeWithTimeout<StorageStats>('get_telemetry_storage_stats');
    } catch {
      const snapshot = await this.getSnapshot();
      return snapshot.storage_stats;
    }
  },

  /**
   * Clear old telemetry data (apply retention).
   * Returns number of records deleted.
   */
  async applyRetention(): Promise<number> {
    return invokeWithTimeout<number>('apply_telemetry_retention');
  },

  /**
   * Compute baseline for a metric.
   */
  async computeBaseline(metricName: string, windowDays?: number): Promise<boolean> {
    return invokeWithTimeout<boolean>('compute_telemetry_baseline', {
      metricName,
      windowDays,
    });
  },
};

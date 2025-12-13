/**
 * React Query hooks for Futurnal Telemetry API
 *
 * These hooks provide a convenient way to interact with the telemetry system
 * with automatic caching, refetching, and error handling.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { telemetryApi } from '@/lib/telemetryApi';
import { useTelemetryStore } from '@/stores/telemetryStore';
import type {
  MetricsQuery,
  AlertsQuery,
  ExportParams,
  TelemetrySnapshot,
  MetricSample,
  QualityGateResult,
  TelemetryAlert,
  TimeRange,
  getTimeRangeFromPreset,
} from '@/types/telemetry';

// ============================================================================
// Query Keys
// ============================================================================

export const telemetryQueryKeys = {
  all: ['telemetry'] as const,
  snapshot: () => [...telemetryQueryKeys.all, 'snapshot'] as const,
  metrics: (query: MetricsQuery) => [...telemetryQueryKeys.all, 'metrics', query] as const,
  qualityGates: () => [...telemetryQueryKeys.all, 'quality-gates'] as const,
  alerts: (query?: AlertsQuery) => [...telemetryQueryKeys.all, 'alerts', query] as const,
  storageStats: () => [...telemetryQueryKeys.all, 'storage-stats'] as const,
};

// ============================================================================
// Snapshot Hook (Real-time polling)
// ============================================================================

/**
 * Hook to get telemetry snapshot with real-time polling.
 * Updates every `pollingInterval` milliseconds (default 2s).
 */
export function useTelemetrySnapshot(options?: { enabled?: boolean }) {
  const pollingInterval = useTelemetryStore((state) => state.pollingInterval);

  return useQuery({
    queryKey: telemetryQueryKeys.snapshot(),
    queryFn: () => telemetryApi.getSnapshot(),
    refetchInterval: pollingInterval,
    enabled: options?.enabled !== false,
    staleTime: pollingInterval / 2, // Consider stale at half the polling interval
  });
}

// ============================================================================
// Historical Metrics Hook
// ============================================================================

/**
 * Hook to get historical metrics for a specific query.
 */
export function useTelemetryMetrics(query: MetricsQuery, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: telemetryQueryKeys.metrics(query),
    queryFn: () => telemetryApi.getMetrics(query),
    enabled: options?.enabled !== false && !!query.metric_name,
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to get metrics for a specific metric name with default time range.
 * Uses the store's default time range preset.
 */
export function useMetricHistory(
  metricName: string,
  options?: {
    enabled?: boolean;
    aggregation?: MetricsQuery['aggregation'];
    labels?: Record<string, string>;
  }
) {
  const defaultTimeRange = useTelemetryStore((state) => state.defaultTimeRange);

  // Import and use the helper
  const { getTimeRangeFromPreset } = require('@/types/telemetry');
  const timeRange = getTimeRangeFromPreset(defaultTimeRange);

  const query: MetricsQuery = {
    metric_name: metricName,
    start_time: timeRange.start,
    end_time: timeRange.end,
    aggregation: options?.aggregation,
    labels: options?.labels,
  };

  return useTelemetryMetrics(query, { enabled: options?.enabled });
}

// ============================================================================
// Quality Gates Hook
// ============================================================================

/**
 * Hook to get quality gate evaluation results.
 */
export function useQualityGates(options?: { enabled?: boolean }) {
  const pollingInterval = useTelemetryStore((state) => state.pollingInterval);

  return useQuery({
    queryKey: telemetryQueryKeys.qualityGates(),
    queryFn: () => telemetryApi.getQualityGates(),
    refetchInterval: pollingInterval,
    enabled: options?.enabled !== false,
    staleTime: pollingInterval / 2,
  });
}

/**
 * Hook to check if all quality gates are passing.
 */
export function useQualityGatesStatus() {
  const { data: gates, isLoading, error } = useQualityGates();

  const allPassing = gates?.every((gate) => gate.passing) ?? true;
  const failingGates = gates?.filter((gate) => !gate.passing) ?? [];

  return {
    allPassing,
    failingGates,
    totalGates: gates?.length ?? 0,
    passingCount: gates?.filter((gate) => gate.passing).length ?? 0,
    isLoading,
    error,
  };
}

// ============================================================================
// Alerts Hooks
// ============================================================================

/**
 * Hook to get telemetry alerts.
 */
export function useTelemetryAlerts(query?: AlertsQuery, options?: { enabled?: boolean }) {
  const pollingInterval = useTelemetryStore((state) => state.pollingInterval);

  return useQuery({
    queryKey: telemetryQueryKeys.alerts(query),
    queryFn: () => telemetryApi.getAlerts(query),
    refetchInterval: pollingInterval,
    enabled: options?.enabled !== false,
    staleTime: pollingInterval / 2,
  });
}

/**
 * Hook to get active (unacknowledged) alerts.
 */
export function useActiveAlerts() {
  return useTelemetryAlerts({ acknowledged: false });
}

/**
 * Hook to get alerts count by severity.
 */
export function useAlertCounts() {
  const { data: alerts, isLoading } = useActiveAlerts();

  const counts = {
    critical: alerts?.filter((a) => a.severity === 'critical').length ?? 0,
    warning: alerts?.filter((a) => a.severity === 'warning').length ?? 0,
    info: alerts?.filter((a) => a.severity === 'info').length ?? 0,
    total: alerts?.length ?? 0,
  };

  return { counts, isLoading };
}

// ============================================================================
// Mutations
// ============================================================================

/**
 * Hook to acknowledge an alert.
 */
export function useAcknowledgeAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: number) => telemetryApi.acknowledgeAlert(alertId),
    onSuccess: () => {
      // Invalidate alerts and snapshot queries
      queryClient.invalidateQueries({ queryKey: telemetryQueryKeys.alerts() });
      queryClient.invalidateQueries({ queryKey: telemetryQueryKeys.snapshot() });
    },
  });
}

/**
 * Hook to export telemetry data.
 */
export function useExportTelemetry() {
  return useMutation({
    mutationFn: (params: ExportParams) => telemetryApi.exportTelemetry(params),
  });
}

/**
 * Hook to apply retention policy (cleanup old data).
 */
export function useApplyRetention() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => telemetryApi.applyRetention(),
    onSuccess: () => {
      // Invalidate storage stats
      queryClient.invalidateQueries({ queryKey: telemetryQueryKeys.storageStats() });
    },
  });
}

/**
 * Hook to compute baseline for a metric.
 */
export function useComputeBaseline() {
  return useMutation({
    mutationFn: ({ metricName, windowDays }: { metricName: string; windowDays?: number }) =>
      telemetryApi.computeBaseline(metricName, windowDays),
  });
}

// ============================================================================
// Storage Stats Hook
// ============================================================================

/**
 * Hook to get storage statistics.
 */
export function useStorageStats(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: telemetryQueryKeys.storageStats(),
    queryFn: () => telemetryApi.getStorageStats(),
    staleTime: 60000, // 1 minute
    enabled: options?.enabled !== false,
  });
}

// ============================================================================
// Composite Hooks
// ============================================================================

/**
 * Hook to get overall telemetry health status.
 * Combines snapshot, quality gates, and alerts into a single status.
 */
export function useTelemetryHealth() {
  const { data: snapshot, isLoading: snapshotLoading } = useTelemetrySnapshot();
  const { allPassing, failingGates } = useQualityGatesStatus();
  const { counts: alertCounts } = useAlertCounts();

  // Determine overall health status
  let status: 'healthy' | 'degraded' | 'critical' = 'healthy';
  if (alertCounts.critical > 0 || failingGates.length > 2) {
    status = 'critical';
  } else if (alertCounts.warning > 0 || failingGates.length > 0) {
    status = 'degraded';
  }

  return {
    status,
    allGatesPassing: allPassing,
    failingGatesCount: failingGates.length,
    activeAlertsCount: alertCounts.total,
    criticalAlertsCount: alertCounts.critical,
    latestValues: snapshot?.latest_values ?? {},
    isLoading: snapshotLoading,
  };
}

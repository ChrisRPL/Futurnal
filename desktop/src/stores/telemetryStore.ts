/**
 * Telemetry Store - Zustand state management for telemetry settings
 *
 * Manages telemetry display preferences, custom thresholds, and dashboard state.
 * Uses persist middleware for localStorage persistence.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { QUALITY_GATES } from '@/types/telemetry';
import type { TimeRangePreset, AnomalySeverity } from '@/types/telemetry';

/**
 * Custom threshold overrides for quality gates.
 */
interface ThresholdOverrides {
  /** Minimum throughput (docs/second) */
  throughputMin: number;
  /** Maximum memory (MB) */
  memoryMax: number;
  /** Maximum latency (ms) */
  latencyMax: number;
  /** Maximum search P95 (ms) */
  searchP95Max: number;
  /** Maximum error rate (0-1) */
  errorRateMax: number;
}

/**
 * Telemetry store state interface.
 */
interface TelemetryState {
  // Display settings
  /** Show telemetry widget on dashboard */
  showDashboardWidget: boolean;
  /** Show quality gate status bar */
  showQualityGates: boolean;
  /** Default time range preset */
  defaultTimeRange: TimeRangePreset;
  /** Polling interval in milliseconds */
  pollingInterval: number;

  // Notification settings
  /** Enable alert notifications */
  alertNotifications: boolean;
  /** Minimum severity for notifications */
  notificationSeverity: AnomalySeverity;
  /** Play sound on alerts */
  alertSounds: boolean;

  // Custom thresholds (override defaults)
  thresholds: ThresholdOverrides;

  // View state (not persisted)
  /** Currently selected metric for detail view */
  selectedMetric: string | null;
  /** Expanded dashboard panels */
  expandedPanels: string[];

  // Actions
  /** Set a single setting value */
  setSetting: <K extends keyof Omit<TelemetryState, 'setSetting' | 'setThreshold' | 'resetSettings' | 'setSelectedMetric' | 'togglePanel'>>(
    key: K,
    value: TelemetryState[K]
  ) => void;
  /** Set a custom threshold */
  setThreshold: <K extends keyof ThresholdOverrides>(
    key: K,
    value: number
  ) => void;
  /** Reset all settings to defaults */
  resetSettings: () => void;
  /** Set selected metric for detail view */
  setSelectedMetric: (metric: string | null) => void;
  /** Toggle expanded state of a panel */
  togglePanel: (panelId: string) => void;
}

/**
 * Default settings values.
 */
const defaultSettings = {
  // Display - show by default for visibility
  showDashboardWidget: true,
  showQualityGates: true,
  defaultTimeRange: '1h' as TimeRangePreset,
  pollingInterval: 2000, // 2 seconds

  // Notifications - conservative defaults
  alertNotifications: true,
  notificationSeverity: 'warning' as AnomalySeverity,
  alertSounds: false,

  // Use quality gate defaults
  thresholds: {
    throughputMin: QUALITY_GATES.throughput.threshold,
    memoryMax: QUALITY_GATES.memory.threshold,
    latencyMax: QUALITY_GATES.latency.threshold,
    searchP95Max: QUALITY_GATES.search_p95.threshold,
    errorRateMax: QUALITY_GATES.error_rate.threshold,
  },

  // View state
  selectedMetric: null as string | null,
  expandedPanels: ['overview', 'gates'] as string[],
};

/**
 * Telemetry store with persistence.
 */
export const useTelemetryStore = create<TelemetryState>()(
  persist(
    (set) => ({
      ...defaultSettings,

      setSetting: (key, value) => set({ [key]: value }),

      setThreshold: (key, value) =>
        set((state) => ({
          thresholds: {
            ...state.thresholds,
            [key]: value,
          },
        })),

      resetSettings: () => set(defaultSettings),

      setSelectedMetric: (metric) => set({ selectedMetric: metric }),

      togglePanel: (panelId) =>
        set((state) => ({
          expandedPanels: state.expandedPanels.includes(panelId)
            ? state.expandedPanels.filter((id) => id !== panelId)
            : [...state.expandedPanels, panelId],
        })),
    }),
    {
      name: 'futurnal-telemetry',
      // Only persist settings, not view state
      partialize: (state) => ({
        showDashboardWidget: state.showDashboardWidget,
        showQualityGates: state.showQualityGates,
        defaultTimeRange: state.defaultTimeRange,
        pollingInterval: state.pollingInterval,
        alertNotifications: state.alertNotifications,
        notificationSeverity: state.notificationSeverity,
        alertSounds: state.alertSounds,
        thresholds: state.thresholds,
      }),
    }
  )
);

// ============================================================================
// Convenience Selectors
// ============================================================================

/**
 * Get display settings.
 */
export const useTelemetryDisplaySettings = () =>
  useTelemetryStore((state) => ({
    showDashboardWidget: state.showDashboardWidget,
    showQualityGates: state.showQualityGates,
    defaultTimeRange: state.defaultTimeRange,
    pollingInterval: state.pollingInterval,
  }));

/**
 * Get notification settings.
 */
export const useTelemetryNotificationSettings = () =>
  useTelemetryStore((state) => ({
    alertNotifications: state.alertNotifications,
    notificationSeverity: state.notificationSeverity,
    alertSounds: state.alertSounds,
  }));

/**
 * Get custom thresholds.
 */
export const useTelemetryThresholds = () =>
  useTelemetryStore((state) => state.thresholds);

/**
 * Get selected metric.
 */
export const useSelectedMetric = () =>
  useTelemetryStore((state) => state.selectedMetric);

/**
 * Get expanded panels.
 */
export const useExpandedPanels = () =>
  useTelemetryStore((state) => state.expandedPanels);

/**
 * Check if a panel is expanded.
 */
export const useIsPanelExpanded = (panelId: string) =>
  useTelemetryStore((state) => state.expandedPanels.includes(panelId));

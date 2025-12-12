/**
 * Settings Store - Zustand state management for application settings
 *
 * Manages appearance, search preferences, privacy settings, connector
 * settings, and graph settings. Uses persist middleware for localStorage persistence.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsState {
  // Appearance
  /** Theme preference */
  theme: 'dark' | 'light' | 'system';
  /** Font size for UI */
  fontSize: 'small' | 'medium' | 'large';

  // Search preferences
  /** Default search mode */
  defaultSearchMode: 'hybrid' | 'semantic' | 'keyword';
  /** Number of results to show per page */
  resultsPerPage: number;
  /** Show confidence scores on results */
  showConfidenceScores: boolean;
  /** Show provenance panel by default */
  showProvenance: boolean;

  // Privacy settings
  /** Enable anonymous telemetry */
  telemetryEnabled: boolean;
  /** Enable crash reports */
  crashReportsEnabled: boolean;
  /** Enable analytics */
  analyticsEnabled: boolean;

  // Connector settings
  /** Enable automatic sync */
  autoSyncEnabled: boolean;
  /** Sync interval in minutes */
  syncIntervalMinutes: number;
  /** Notify when sync completes */
  notifyOnSyncComplete: boolean;

  // Graph settings
  /** Maximum nodes to display in graph */
  graphNodeLimit: number;
  /** Enable graph animations */
  graphAnimations: boolean;

  // Actions
  /** Set a single setting value */
  setSetting: <K extends keyof Omit<SettingsState, 'setSetting' | 'resetSettings'>>(
    key: K,
    value: SettingsState[K]
  ) => void;
  /** Reset all settings to defaults */
  resetSettings: () => void;
}

const defaultSettings = {
  // Appearance - dark mode first per design system
  theme: 'dark' as const,
  fontSize: 'medium' as const,

  // Search - sensible defaults
  defaultSearchMode: 'hybrid' as const,
  resultsPerPage: 20,
  showConfidenceScores: true,
  showProvenance: false,

  // Privacy - privacy-first defaults
  telemetryEnabled: false,
  crashReportsEnabled: true,
  analyticsEnabled: false,

  // Connectors
  autoSyncEnabled: true,
  syncIntervalMinutes: 60,
  notifyOnSyncComplete: true,

  // Graph
  graphNodeLimit: 1000,
  graphAnimations: true,
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...defaultSettings,

      setSetting: (key, value) => set({ [key]: value }),

      resetSettings: () => set(defaultSettings),
    }),
    {
      name: 'futurnal-settings',
      // Persist all settings
    }
  )
);

// Convenience selectors
export const useTheme = () => useSettingsStore((state) => state.theme);
export const useFontSize = () => useSettingsStore((state) => state.fontSize);

export const useSearchPreferences = () =>
  useSettingsStore((state) => ({
    defaultSearchMode: state.defaultSearchMode,
    resultsPerPage: state.resultsPerPage,
    showConfidenceScores: state.showConfidenceScores,
    showProvenance: state.showProvenance,
  }));

export const usePrivacySettings = () =>
  useSettingsStore((state) => ({
    telemetryEnabled: state.telemetryEnabled,
    crashReportsEnabled: state.crashReportsEnabled,
    analyticsEnabled: state.analyticsEnabled,
  }));

export const useConnectorSettings = () =>
  useSettingsStore((state) => ({
    autoSyncEnabled: state.autoSyncEnabled,
    syncIntervalMinutes: state.syncIntervalMinutes,
    notifyOnSyncComplete: state.notifyOnSyncComplete,
  }));

export const useGraphSettings = () =>
  useSettingsStore((state) => ({
    graphNodeLimit: state.graphNodeLimit,
    graphAnimations: state.graphAnimations,
  }));

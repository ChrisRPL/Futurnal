/**
 * Telemetry Settings Component
 *
 * Provides user control over privacy-respecting telemetry.
 * Telemetry is opt-in only and never collects PII.
 *
 * Privacy Guarantee:
 * - Disabled by default
 * - No query content collected
 * - No file paths collected
 * - Only aggregate metrics
 */

import { useState, useEffect } from 'react';
import {
  BarChart3,
  Shield,
  Activity,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Download,
  Trash2
} from 'lucide-react';

interface TelemetryConfig {
  enabled: boolean;
  collectPerformance: boolean;
  collectErrors: boolean;
  collectUsage: boolean;
  retentionDays: number;
}

interface TelemetrySummary {
  session: {
    duration_seconds: number;
    start_time: string;
  };
  operations: Record<string, number>;
  latency: Record<string, {
    count: number;
    avg_ms: number;
    min_ms: number;
    max_ms: number;
  }>;
  errors: Record<string, number>;
  telemetry_enabled: boolean;
}

const DEFAULT_CONFIG: TelemetryConfig = {
  enabled: false,
  collectPerformance: true,
  collectErrors: true,
  collectUsage: true,
  retentionDays: 7,
};

export function TelemetrySettings() {
  const [config, setConfig] = useState<TelemetryConfig>(DEFAULT_CONFIG);
  const [summary, setSummary] = useState<TelemetrySummary | null>(null);
  const [loading, setLoading] = useState(false);

  // Load telemetry config on mount
  useEffect(() => {
    loadConfig();
    if (config.enabled) {
      loadSummary();
    }
  }, []);

  const loadConfig = async () => {
    try {
      // In production, this would call the backend
      // const response = await invoke('get_telemetry_config');
      // setConfig(response as TelemetryConfig);
    } catch (error) {
      console.error('Failed to load telemetry config:', error);
    }
  };

  const loadSummary = async () => {
    setLoading(true);
    try {
      // In production, this would call the backend
      // const response = await invoke('get_telemetry_summary');
      // setSummary(response as TelemetrySummary);
    } catch (error) {
      console.error('Failed to load telemetry summary:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleTelemetry = async () => {
    const newConfig = { ...config, enabled: !config.enabled };
    setConfig(newConfig);

    try {
      // In production, this would call the backend
      // await invoke('set_telemetry_config', { config: newConfig });
      if (newConfig.enabled) {
        loadSummary();
      }
    } catch (error) {
      console.error('Failed to update telemetry config:', error);
      // Rollback on error
      setConfig(config);
    }
  };

  const handleExportData = async () => {
    try {
      // In production, this would call the backend
      // await invoke('export_telemetry_data');
      alert('Telemetry data exported successfully');
    } catch (error) {
      console.error('Failed to export telemetry:', error);
    }
  };

  const handleClearData = async () => {
    if (!confirm('Are you sure you want to clear all telemetry data? This cannot be undone.')) {
      return;
    }

    try {
      // In production, this would call the backend
      // await invoke('clear_telemetry_data');
      setSummary(null);
    } catch (error) {
      console.error('Failed to clear telemetry:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">
          Telemetry & Analytics
        </h2>
        <p className="mt-1 text-sm text-[var(--color-text-tertiary)]">
          Help improve Futurnal by sharing anonymous usage data
        </p>
      </div>

      {/* Privacy Notice */}
      <div className="p-4 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
        <div className="flex items-start gap-3">
          <Shield className="h-5 w-5 text-[var(--color-accent)] mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-[var(--color-text-primary)]">
              Privacy Guarantee
            </h3>
            <ul className="mt-2 text-sm text-[var(--color-text-tertiary)] space-y-1">
              <li>• Telemetry is disabled by default</li>
              <li>• No search queries or file paths are ever collected</li>
              <li>• No personal data or document content</li>
              <li>• Only aggregate performance metrics</li>
              <li>• All data stays local unless you explicitly export</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Master Toggle */}
      <div className="flex items-center justify-between p-4 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
        <div className="flex items-center gap-3">
          <BarChart3 className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <div>
            <div className="text-sm font-medium text-[var(--color-text-primary)]">
              Enable Telemetry
            </div>
            <div className="text-xs text-[var(--color-text-tertiary)]">
              Collect anonymous performance metrics
            </div>
          </div>
        </div>
        <button
          onClick={handleToggleTelemetry}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            config.enabled
              ? 'bg-[var(--color-accent)]'
              : 'bg-[var(--color-surface-hover)]'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              config.enabled ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Data Categories */}
      {config.enabled && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
            Data Categories
          </h3>

          {/* Performance Metrics */}
          <div className="flex items-center justify-between p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
            <div className="flex items-center gap-3">
              <Activity className="h-4 w-4 text-[var(--color-text-tertiary)]" />
              <div>
                <div className="text-sm text-[var(--color-text-primary)]">
                  Performance Metrics
                </div>
                <div className="text-xs text-[var(--color-text-tertiary)]">
                  Search latency, response times, throughput
                </div>
              </div>
            </div>
            <button
              onClick={() => setConfig({ ...config, collectPerformance: !config.collectPerformance })}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                config.collectPerformance
                  ? 'bg-[var(--color-accent)]'
                  : 'bg-[var(--color-surface-hover)]'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  config.collectPerformance ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          {/* Error Tracking */}
          <div className="flex items-center justify-between p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
            <div className="flex items-center gap-3">
              <AlertTriangle className="h-4 w-4 text-[var(--color-text-tertiary)]" />
              <div>
                <div className="text-sm text-[var(--color-text-primary)]">
                  Error Tracking
                </div>
                <div className="text-xs text-[var(--color-text-tertiary)]">
                  Error types and counts (no messages)
                </div>
              </div>
            </div>
            <button
              onClick={() => setConfig({ ...config, collectErrors: !config.collectErrors })}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                config.collectErrors
                  ? 'bg-[var(--color-accent)]'
                  : 'bg-[var(--color-surface-hover)]'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  config.collectErrors ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          {/* Usage Patterns */}
          <div className="flex items-center justify-between p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-4 w-4 text-[var(--color-text-tertiary)]" />
              <div>
                <div className="text-sm text-[var(--color-text-primary)]">
                  Usage Patterns
                </div>
                <div className="text-xs text-[var(--color-text-tertiary)]">
                  Feature usage counts (no content)
                </div>
              </div>
            </div>
            <button
              onClick={() => setConfig({ ...config, collectUsage: !config.collectUsage })}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                config.collectUsage
                  ? 'bg-[var(--color-accent)]'
                  : 'bg-[var(--color-surface-hover)]'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  config.collectUsage ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      )}

      {/* Statistics Summary */}
      {config.enabled && summary && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
              Current Session
            </h3>
            <button
              onClick={loadSummary}
              disabled={loading}
              className="p-1 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
              <div className="text-xs text-[var(--color-text-tertiary)]">Operations</div>
              <div className="text-lg font-semibold text-[var(--color-text-primary)]">
                {Object.values(summary.operations).reduce((a, b) => a + b, 0)}
              </div>
            </div>
            <div className="p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
              <div className="text-xs text-[var(--color-text-tertiary)]">Errors</div>
              <div className="text-lg font-semibold text-[var(--color-text-primary)]">
                {Object.values(summary.errors).reduce((a, b) => a + b, 0)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Data Management */}
      {config.enabled && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
            Data Management
          </h3>

          <div className="flex gap-3">
            <button
              onClick={handleExportData}
              className="flex items-center gap-2 px-4 py-2 text-sm bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors"
            >
              <Download className="h-4 w-4" />
              Export Data
            </button>
            <button
              onClick={handleClearData}
              className="flex items-center gap-2 px-4 py-2 text-sm text-red-500 bg-[var(--color-surface)] border border-red-500/30 rounded-lg hover:bg-red-500/10 transition-colors"
            >
              <Trash2 className="h-4 w-4" />
              Clear All Data
            </button>
          </div>
        </div>
      )}

      {/* Retention Settings */}
      {config.enabled && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
            Data Retention
          </h3>

          <div className="p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-[var(--color-text-primary)]">
                  Retention Period
                </div>
                <div className="text-xs text-[var(--color-text-tertiary)]">
                  Data older than this will be automatically deleted
                </div>
              </div>
              <select
                value={config.retentionDays}
                onChange={(e) => setConfig({ ...config, retentionDays: parseInt(e.target.value) })}
                className="px-3 py-1 text-sm bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded text-[var(--color-text-primary)]"
              >
                <option value={1}>1 day</option>
                <option value={7}>7 days</option>
                <option value={30}>30 days</option>
                <option value={90}>90 days</option>
              </select>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

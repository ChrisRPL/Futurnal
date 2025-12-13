/**
 * Privacy Section
 *
 * Data collection controls, consent management, and audit log access.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { useState, useEffect } from 'react';
import { Eye, Shield, FileText, Clock, Cloud, AlertTriangle, CheckCircle, X, RefreshCw, Loader2 } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ConsentManager } from './ConsentManager';
import { AuditLogViewer } from './AuditLogViewer';
import { CloudSyncConsentModal } from './CloudSyncConsentModal';
import { useSettingsStore } from '@/stores/settingsStore';
import { useTierLimits } from '@/hooks/useTierLimits';
import { useCloudSyncStore, useCloudSyncConsent, useCloudSyncStatus } from '@/stores/cloudSyncStore';

interface PrivacyAlert {
  alert_id: string;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  anomaly_type?: string;
  created_at: string;
  acknowledged: boolean;
}

/**
 * Privacy Alert Banner Component
 * Shows active privacy alerts from the anomaly detection system.
 */
function PrivacyAlertBanner({
  alerts,
  onDismiss,
  onViewAll,
}: {
  alerts: PrivacyAlert[];
  onDismiss: (alertId: string) => void;
  onViewAll: () => void;
}) {
  if (alerts.length === 0) return null;

  const criticalAlerts = alerts.filter((a) => a.severity === 'critical');
  const warningAlerts = alerts.filter((a) => a.severity === 'warning');
  const hasUrgent = criticalAlerts.length > 0;

  const displayAlert = criticalAlerts[0] || warningAlerts[0] || alerts[0];

  const severityStyles = {
    critical: 'bg-red-950/50 border-red-800/50 text-red-200',
    warning: 'bg-amber-950/50 border-amber-800/50 text-amber-200',
    info: 'bg-blue-950/50 border-blue-800/50 text-blue-200',
  };

  const iconStyles = {
    critical: 'text-red-400',
    warning: 'text-amber-400',
    info: 'text-blue-400',
  };

  return (
    <div
      className={`p-4 border rounded-lg ${severityStyles[displayAlert.severity]} mb-6`}
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className={`h-5 w-5 mt-0.5 flex-shrink-0 ${iconStyles[displayAlert.severity]}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-sm">{displayAlert.title}</span>
            {alerts.length > 1 && (
              <Badge
                variant="secondary"
                className="text-xs bg-white/10"
              >
                +{alerts.length - 1} more
              </Badge>
            )}
          </div>
          <p className="text-sm opacity-90">{displayAlert.message}</p>
          <div className="flex items-center gap-2 mt-3">
            <Button
              size="sm"
              variant="secondary"
              onClick={onViewAll}
              className="h-7 text-xs"
            >
              View All Alerts
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => onDismiss(displayAlert.alert_id)}
              className="h-7 text-xs opacity-70 hover:opacity-100"
            >
              Dismiss
            </Button>
          </div>
        </div>
        <button
          onClick={() => onDismiss(displayAlert.alert_id)}
          className="text-current opacity-60 hover:opacity-100 transition-opacity"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export function PrivacySection() {
  const [showAuditLogs, setShowAuditLogs] = useState(false);
  const [showCloudSyncModal, setShowCloudSyncModal] = useState(false);
  const [privacyAlerts, setPrivacyAlerts] = useState<PrivacyAlert[]>([]);
  const { isPro } = useTierLimits();
  const {
    telemetryEnabled,
    crashReportsEnabled,
    analyticsEnabled,
    setSetting,
  } = useSettingsStore();

  // Cloud sync state
  const { hasConsent, isLoading: isLoadingConsent } = useCloudSyncConsent();
  const { syncEnabled, syncInProgress, lastSyncAt, lastSyncError, nodesSynced } = useCloudSyncStatus();
  const {
    loadConsentStatus,
    enableSync,
    disableSync,
    syncNow,
    revokeConsent,
  } = useCloudSyncStore();

  // Load cloud sync consent on mount
  useEffect(() => {
    loadConsentStatus();
  }, [loadConsentStatus]);

  // Load privacy alerts from IPC file
  useEffect(() => {
    async function loadAlerts() {
      try {
        // In production, this would use IPC to load from ~/.futurnal/alerts/ipc_alerts.json
        // For now, we'll use a placeholder that can be populated by the backend
        const response = await window.electron?.invoke('privacy:getAlerts') ?? [];
        const unacknowledged = response.filter((a: PrivacyAlert) => !a.acknowledged);
        setPrivacyAlerts(unacknowledged);
      } catch {
        // Alerts not available, that's fine
        setPrivacyAlerts([]);
      }
    }

    loadAlerts();

    // Poll for new alerts every 30 seconds
    const interval = setInterval(loadAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleDismissAlert = async (alertId: string) => {
    try {
      await window.electron?.invoke('privacy:acknowledgeAlert', alertId);
      setPrivacyAlerts((prev) => prev.filter((a) => a.alert_id !== alertId));
    } catch {
      // Dismiss locally even if IPC fails
      setPrivacyAlerts((prev) => prev.filter((a) => a.alert_id !== alertId));
    }
  };

  const handleViewAllAlerts = () => {
    setShowAuditLogs(true);
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">Privacy</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Control how your data is collected, stored, and used.
        </p>
      </div>

      {/* Privacy Alert Banner */}
      <PrivacyAlertBanner
        alerts={privacyAlerts}
        onDismiss={handleDismissAlert}
        onViewAll={handleViewAllAlerts}
      />

      {/* Data Collection */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Eye className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Data Collection</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Choose what information Futurnal can collect to improve the product.
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">Anonymous Telemetry</div>
              <div className="text-xs text-[var(--color-text-muted)]">
                Help improve Futurnal by sending anonymous usage statistics
              </div>
            </div>
            <Switch
              checked={telemetryEnabled}
              onCheckedChange={(checked) => setSetting('telemetryEnabled', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">Crash Reports</div>
              <div className="text-xs text-[var(--color-text-muted)]">
                Automatically send crash reports to help fix bugs
              </div>
            </div>
            <Switch
              checked={crashReportsEnabled}
              onCheckedChange={(checked) => setSetting('crashReportsEnabled', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">Feature Analytics</div>
              <div className="text-xs text-[var(--color-text-muted)]">
                Share feature usage to help prioritize development
              </div>
            </div>
            <Switch
              checked={analyticsEnabled}
              onCheckedChange={(checked) => setSetting('analyticsEnabled', checked)}
            />
          </div>
        </div>
      </div>

      {/* Source Consent */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Shield className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Data Source Consent</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Manage permissions for each connected data source.
        </p>
        <ConsentManager />
      </div>

      {/* Audit Log */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <FileText className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Activity Audit Log</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          View a complete history of data access and operations.
        </p>
        <Button
          variant="outline"
          onClick={() => setShowAuditLogs(true)}
          className="w-full"
        >
          View Audit Log
        </Button>
      </div>

      {/* Data Retention */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Clock className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Data Retention</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Control how long your data is stored locally.
        </p>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">Search History</div>
              <div className="text-xs text-[var(--color-text-muted)]">Keep last 50 searches</div>
            </div>
            <Badge variant="secondary">50 items</Badge>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">Audit Logs</div>
              <div className="text-xs text-[var(--color-text-muted)]">Retained for 90 days</div>
            </div>
            <Badge variant="secondary">90 days</Badge>
          </div>
        </div>
      </div>

      {/* Cloud Sync */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Cloud className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Cloud Sync
            {!isPro && (
              <Badge variant="outline" className="ml-2 text-xs">
                Pro
              </Badge>
            )}
          </h3>
          {hasConsent && (
            <Badge variant="secondary" className="ml-auto text-xs bg-green-900/30 text-green-400 border-green-800/50">
              <CheckCircle className="h-3 w-3 mr-1" />
              Enabled
            </Badge>
          )}
        </div>

        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          {isPro
            ? hasConsent
              ? 'Your knowledge graph structure is synced to the cloud. Document content stays on your device.'
              : 'Sync your knowledge graph metadata across devices. Your documents never leave your device.'
            : 'Upgrade to Pro to sync your knowledge graph across devices.'}
        </p>

        {isPro ? (
          hasConsent ? (
            <div className="space-y-4">
              {/* Sync Enable/Disable Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-[var(--color-text-primary)]">Automatic Sync</div>
                  <div className="text-xs text-[var(--color-text-muted)]">
                    Sync every 15 minutes when app is open
                  </div>
                </div>
                <Switch
                  checked={syncEnabled}
                  onCheckedChange={(checked) => {
                    if (checked) {
                      enableSync();
                    } else {
                      disableSync();
                    }
                  }}
                  disabled={isLoadingConsent}
                />
              </div>

              <Separator />

              {/* Sync Status */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--color-text-secondary)]">Last synced</span>
                  <span className="text-[var(--color-text-primary)]">
                    {lastSyncAt
                      ? new Date(lastSyncAt).toLocaleString()
                      : 'Never'}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--color-text-secondary)]">Nodes synced</span>
                  <span className="text-[var(--color-text-primary)]">{nodesSynced}</span>
                </div>
                {lastSyncError && (
                  <div className="p-2 rounded bg-red-950/30 border border-red-800/50 text-red-400 text-xs">
                    {lastSyncError}
                  </div>
                )}
              </div>

              <Separator />

              {/* Sync Actions */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={syncNow}
                  disabled={syncInProgress || !syncEnabled}
                  className="flex-1"
                >
                  {syncInProgress ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Syncing...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Sync Now
                    </>
                  )}
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    if (confirm('Are you sure you want to revoke cloud sync? This will DELETE all your cloud data.')) {
                      revokeConsent();
                    }
                  }}
                  className="text-red-400 hover:text-red-300 hover:bg-red-950/30"
                >
                  Revoke
                </Button>
              </div>
            </div>
          ) : (
            <Button
              variant="outline"
              onClick={() => setShowCloudSyncModal(true)}
              disabled={isLoadingConsent}
              className="w-full"
            >
              {isLoadingConsent ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <Cloud className="h-4 w-4 mr-2" />
                  Enable Cloud Sync
                </>
              )}
            </Button>
          )
        ) : (
          <Button variant="outline" className="w-full">
            Upgrade to Pro
          </Button>
        )}
      </div>

      {/* Cloud Sync Consent Modal */}
      <CloudSyncConsentModal
        open={showCloudSyncModal}
        onOpenChange={setShowCloudSyncModal}
      />

      {/* Audit Log Modal */}
      <AuditLogViewer open={showAuditLogs} onOpenChange={setShowAuditLogs} />
    </div>
  );
}

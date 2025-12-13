/**
 * Privacy Section
 *
 * Data collection controls, consent management, and audit log access.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { useState } from 'react';
import { Eye, Shield, FileText, Clock, Lock } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ConsentManager } from './ConsentManager';
import { AuditLogViewer } from './AuditLogViewer';
import { useSettingsStore } from '@/stores/settingsStore';
import { useTierLimits } from '@/hooks/useTierLimits';

export function PrivacySection() {
  const [showAuditLogs, setShowAuditLogs] = useState(false);
  const { isPro } = useTierLimits();
  const {
    telemetryEnabled,
    crashReportsEnabled,
    analyticsEnabled,
    setSetting,
  } = useSettingsStore();

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">Privacy</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Control how your data is collected, stored, and used.
        </p>
      </div>

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

      {/* Cloud Backup (Pro) */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Lock className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Encrypted Cloud Backup
            {!isPro && (
              <Badge variant="outline" className="ml-2 text-xs">
                Pro
              </Badge>
            )}
          </h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          {isPro
            ? 'Securely backup your data with end-to-end encryption.'
            : 'Upgrade to Pro for encrypted cloud backup.'}
        </p>

        {isPro ? (
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">Enable Cloud Backup</div>
              <div className="text-xs text-[var(--color-text-muted)]">All data encrypted before upload</div>
            </div>
            <Switch />
          </div>
        ) : (
          <Button variant="outline" className="w-full">
            Upgrade to Pro
          </Button>
        )}
      </div>

      {/* Audit Log Modal */}
      <AuditLogViewer open={showAuditLogs} onOpenChange={setShowAuditLogs} />
    </div>
  );
}

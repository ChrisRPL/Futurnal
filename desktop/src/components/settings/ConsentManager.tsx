/**
 * Consent Manager
 *
 * Manages consent permissions for each connected data source.
 * Part of the Privacy Section - Sovereignty Control Center.
 *
 * Features:
 * - Per-source consent toggles
 * - Consent expiration display
 * - Retention policy selector
 * - Bulk consent grant/revoke
 * - Data purge with confirmation
 * - Consent history timeline
 */

import { useState, useMemo } from 'react';
import {
  Folder,
  FileText,
  Github,
  Mail,
  CheckCircle,
  AlertCircle,
  Loader2,
  Clock,
  Trash2,
  History,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Calendar,
} from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { useConnectors, useConsent, useGrantConsent, useRevokeConsent } from '@/hooks/useApi';
import { formatDistanceToNow, format, addDays, isPast, differenceInDays } from 'date-fns';

const SOURCE_ICONS: Record<string, typeof Folder> = {
  local_folder: Folder,
  obsidian: FileText,
  github: Github,
  imap: Mail,
};

const CONSENT_TYPES = [
  {
    type: 'local.external_processing',
    label: 'Read Access',
    description: 'Allow your Ghost to access and index this source',
  },
  {
    type: 'local.ai_processing',
    label: 'AI Processing',
    description: 'Allow your Ghost to extract entities and relationships',
  },
  {
    type: 'local.memory_storage',
    label: 'Memory Storage',
    description: 'Allow your Ghost to remember content in its knowledge graph',
  },
];

const RETENTION_OPTIONS = [
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
  { value: '365', label: '1 year' },
  { value: 'unlimited', label: 'Unlimited' },
];

interface ConsentHistoryEntry {
  timestamp: string;
  action: 'grant' | 'revoke';
  consent_type: string;
  source_name: string;
}

interface PurgeDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceName: string;
  onConfirm: () => void;
  isPending: boolean;
}

function PurgeConfirmDialog({
  open,
  onOpenChange,
  sourceName,
  onConfirm,
  isPending,
}: PurgeDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="h-5 w-5" />
            Purge Data for {sourceName}
          </DialogTitle>
          <DialogDescription className="space-y-2">
            <p>This will permanently delete all data associated with this source:</p>
            <ul className="list-disc list-inside text-sm space-y-1 ml-2">
              <li>All indexed content from this source</li>
              <li>Extracted entities and relationships</li>
              <li>Search history related to this source</li>
              <li>All consent records for this source</li>
            </ul>
            <p className="text-red-400 font-medium mt-4">
              This action cannot be undone.
            </p>
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            disabled={isPending}
          >
            {isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Purging...
              </>
            ) : (
              <>
                <Trash2 className="h-4 w-4 mr-2" />
                Purge All Data
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ConsentHistoryTimeline({ history }: { history: ConsentHistoryEntry[] }) {
  if (history.length === 0) {
    return (
      <div className="text-sm text-[var(--color-text-muted)] py-2">
        No consent history available.
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-48 overflow-y-auto">
      {history.map((entry, index) => (
        <div
          key={`${entry.timestamp}-${index}`}
          className="flex items-start gap-3 text-sm"
        >
          <div
            className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${
              entry.action === 'grant' ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <div className="flex-1 min-w-0">
            <div className="text-[var(--color-text-secondary)]">
              {entry.action === 'grant' ? 'Granted' : 'Revoked'}{' '}
              <span className="font-medium">
                {CONSENT_TYPES.find((t) => t.type === entry.consent_type)?.label ||
                  entry.consent_type}
              </span>
            </div>
            <div className="text-xs text-[var(--color-text-muted)]">
              {format(new Date(entry.timestamp), 'MMM d, yyyy h:mm a')}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ExpirationBadge({ expiresAt }: { expiresAt: string | null }) {
  if (!expiresAt) {
    return (
      <Badge variant="secondary" className="text-xs">
        <Clock className="h-3 w-3 mr-1" />
        No expiration
      </Badge>
    );
  }

  const expirationDate = new Date(expiresAt);
  const isExpired = isPast(expirationDate);
  const daysUntilExpiry = differenceInDays(expirationDate, new Date());

  if (isExpired) {
    return (
      <Badge variant="destructive" className="text-xs">
        <AlertCircle className="h-3 w-3 mr-1" />
        Expired
      </Badge>
    );
  }

  if (daysUntilExpiry <= 7) {
    return (
      <Badge variant="outline" className="text-xs border-amber-600 text-amber-400">
        <Clock className="h-3 w-3 mr-1" />
        Expires in {daysUntilExpiry} day{daysUntilExpiry !== 1 ? 's' : ''}
      </Badge>
    );
  }

  return (
    <Badge variant="secondary" className="text-xs">
      <Calendar className="h-3 w-3 mr-1" />
      Expires {format(expirationDate, 'MMM d, yyyy')}
    </Badge>
  );
}

export function ConsentManager() {
  const { data: connectors, isLoading: loadingConnectors } = useConnectors();
  const { data: consents, isLoading: loadingConsents, refetch: refetchConsents } = useConsent();
  const grantConsentMutation = useGrantConsent();
  const revokeConsentMutation = useRevokeConsent();
  const [pendingChanges, setPendingChanges] = useState<Set<string>>(new Set());
  const [expandedHistory, setExpandedHistory] = useState<Set<string>>(new Set());
  const [purgeDialogSource, setPurgeDialogSource] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [isPurging, setIsPurging] = useState(false);
  const [bulkActionPending, setBulkActionPending] = useState(false);

  // Mock consent history - in production, this would come from the API
  const consentHistory = useMemo<Record<string, ConsentHistoryEntry[]>>(() => {
    const history: Record<string, ConsentHistoryEntry[]> = {};
    connectors?.forEach((connector) => {
      const sourceConsents = consents?.filter(
        (c) => c.source_id === connector.id || c.source_name === connector.name
      ) || [];

      history[connector.id] = sourceConsents
        .filter((c) => c.granted_at)
        .map((c) => ({
          timestamp: c.granted_at!,
          action: 'grant' as const,
          consent_type: c.consent_type,
          source_name: connector.name,
        }))
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    });
    return history;
  }, [connectors, consents]);

  if (loadingConnectors || loadingConsents) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 bg-[var(--color-surface)] animate-pulse" />
        ))}
      </div>
    );
  }

  if (!connectors?.length) {
    return (
      <div className="text-center py-8">
        <p className="text-[var(--color-text-tertiary)]">No data sources configured.</p>
        <p className="text-sm text-[var(--color-text-muted)] mt-1">
          Add a source to manage its consent.
        </p>
      </div>
    );
  }

  const handleConsentChange = async (
    sourceId: string,
    consentType: string,
    granted: boolean,
    retentionDays?: number
  ) => {
    const changeKey = `${sourceId}:${consentType}`;
    setPendingChanges((prev) => new Set(prev).add(changeKey));

    try {
      if (granted) {
        await grantConsentMutation.mutateAsync({
          source_id: sourceId,
          consent_type: consentType,
          retention_days: retentionDays,
        });
      } else {
        await revokeConsentMutation.mutateAsync({
          sourceId,
          consentType,
        });
      }
    } catch (error) {
      console.error('Failed to update consent:', error);
    } finally {
      setPendingChanges((prev) => {
        const next = new Set(prev);
        next.delete(changeKey);
        return next;
      });
    }
  };

  const handleBulkGrant = async (sourceId: string) => {
    setBulkActionPending(true);
    try {
      for (const consentType of CONSENT_TYPES) {
        await grantConsentMutation.mutateAsync({
          source_id: sourceId,
          consent_type: consentType.type,
        });
      }
      await refetchConsents();
    } catch (error) {
      console.error('Failed to grant all consents:', error);
    } finally {
      setBulkActionPending(false);
    }
  };

  const handleBulkRevoke = async (sourceId: string) => {
    setBulkActionPending(true);
    try {
      for (const consentType of CONSENT_TYPES) {
        await revokeConsentMutation.mutateAsync({
          sourceId,
          consentType: consentType.type,
        });
      }
      await refetchConsents();
    } catch (error) {
      console.error('Failed to revoke all consents:', error);
    } finally {
      setBulkActionPending(false);
    }
  };

  const handleRetentionChange = async (
    sourceId: string,
    consentType: string,
    retentionValue: string
  ) => {
    const retentionDays = retentionValue === 'unlimited' ? undefined : parseInt(retentionValue, 10);
    await handleConsentChange(sourceId, consentType, true, retentionDays);
  };

  const handlePurgeSource = async () => {
    if (!purgeDialogSource) return;

    setIsPurging(true);
    try {
      // Call IPC to purge source data
      await window.electron?.invoke('privacy:purgeSource', purgeDialogSource.id);
      // Revoke all consents for this source
      await handleBulkRevoke(purgeDialogSource.id);
      setPurgeDialogSource(null);
    } catch (error) {
      console.error('Failed to purge source:', error);
    } finally {
      setIsPurging(false);
    }
  };

  const toggleHistory = (sourceId: string) => {
    setExpandedHistory((prev) => {
      const next = new Set(prev);
      if (next.has(sourceId)) {
        next.delete(sourceId);
      } else {
        next.add(sourceId);
      }
      return next;
    });
  };

  return (
    <div className="space-y-4">
      {connectors.map((connector) => {
        const Icon = SOURCE_ICONS[connector.connector_type] || FileText;
        const sourceConsents =
          consents?.filter(
            (c) => c.source_id === connector.id || c.source_name === connector.name
          ) || [];

        const allGranted = CONSENT_TYPES.every((ct) =>
          sourceConsents.some((sc) => sc.consent_type === ct.type && sc.granted)
        );
        const noneGranted = CONSENT_TYPES.every(
          (ct) => !sourceConsents.some((sc) => sc.consent_type === ct.type && sc.granted)
        );

        const historyExpanded = expandedHistory.has(connector.id);

        return (
          <div
            key={connector.id}
            className="border border-[var(--color-border)] bg-[var(--color-surface)]"
          >
            {/* Header */}
            <div className="p-4 border-b border-[var(--color-border)]">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-[var(--color-bg-secondary)]">
                    <Icon className="h-4 w-4 text-[var(--color-text-secondary)]" />
                  </div>
                  <div>
                    <div className="font-medium text-[var(--color-text-primary)]">
                      {connector.name}
                    </div>
                    <div className="text-xs text-[var(--color-text-muted)] capitalize">
                      {connector.connector_type.replace('_', ' ')}
                    </div>
                  </div>
                </div>

                {/* Bulk Actions */}
                <div className="flex items-center gap-2">
                  {bulkActionPending ? (
                    <Loader2 className="h-4 w-4 animate-spin text-[var(--color-text-tertiary)]" />
                  ) : (
                    <>
                      {!allGranted && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleBulkGrant(connector.id)}
                          className="h-7 text-xs"
                        >
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Grant All
                        </Button>
                      )}
                      {!noneGranted && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleBulkRevoke(connector.id)}
                          className="h-7 text-xs text-[var(--color-text-muted)]"
                        >
                          Revoke All
                        </Button>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Consent Types */}
            <div className="p-4 space-y-4">
              {CONSENT_TYPES.map((consentType) => {
                const consent = sourceConsents.find(
                  (c) => c.consent_type === consentType.type
                );
                const isGranted = consent?.granted ?? false;
                const changeKey = `${connector.id}:${consentType.type}`;
                const isPending = pendingChanges.has(changeKey);
                const expiresAt = consent?.expires_at ?? null;
                const currentRetention = consent?.retention_days?.toString() || 'unlimited';

                return (
                  <div key={consentType.type} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="text-sm text-[var(--color-text-primary)] flex items-center gap-2">
                          {consentType.label}
                          {isPending ? (
                            <Loader2 className="h-3 w-3 animate-spin text-[var(--color-text-tertiary)]" />
                          ) : isGranted ? (
                            <CheckCircle className="h-3 w-3 text-green-500" />
                          ) : (
                            <AlertCircle className="h-3 w-3 text-[var(--color-text-muted)]" />
                          )}
                        </div>
                        <div className="text-xs text-[var(--color-text-muted)]">
                          {consentType.description}
                        </div>
                      </div>
                      <Switch
                        checked={isGranted}
                        disabled={isPending}
                        onCheckedChange={(checked) =>
                          handleConsentChange(connector.id, consentType.type, checked)
                        }
                      />
                    </div>

                    {/* Expiration and Retention (only show when granted) */}
                    {isGranted && (
                      <div className="flex items-center gap-4 ml-0 pl-0">
                        <ExpirationBadge expiresAt={expiresAt} />

                        <div className="flex items-center gap-2">
                          <span className="text-xs text-[var(--color-text-muted)]">
                            Retention:
                          </span>
                          <Select
                            value={currentRetention}
                            onValueChange={(value) =>
                              handleRetentionChange(connector.id, consentType.type, value)
                            }
                          >
                            <SelectTrigger className="h-6 w-24 text-xs">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {RETENTION_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {consent?.granted_at && (
                          <span className="text-xs text-[var(--color-text-muted)]">
                            Granted{' '}
                            {formatDistanceToNow(new Date(consent.granted_at), {
                              addSuffix: true,
                            })}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Footer: History & Purge */}
            <div className="border-t border-[var(--color-border)] p-4">
              <div className="flex items-center justify-between">
                <Collapsible open={historyExpanded} onOpenChange={() => toggleHistory(connector.id)}>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" size="sm" className="h-7 text-xs gap-1">
                      <History className="h-3 w-3" />
                      Consent History
                      {historyExpanded ? (
                        <ChevronUp className="h-3 w-3" />
                      ) : (
                        <ChevronDown className="h-3 w-3" />
                      )}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-3">
                    <ConsentHistoryTimeline history={consentHistory[connector.id] || []} />
                  </CollapsibleContent>
                </Collapsible>

                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs text-red-400 hover:text-red-300 hover:bg-red-950/30"
                  onClick={() =>
                    setPurgeDialogSource({ id: connector.id, name: connector.name })
                  }
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Purge Data
                </Button>
              </div>
            </div>
          </div>
        );
      })}

      {/* Purge Confirmation Dialog */}
      <PurgeConfirmDialog
        open={!!purgeDialogSource}
        onOpenChange={(open) => !open && setPurgeDialogSource(null)}
        sourceName={purgeDialogSource?.name || ''}
        onConfirm={handlePurgeSource}
        isPending={isPurging}
      />
    </div>
  );
}

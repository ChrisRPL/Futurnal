/**
 * Consent Manager
 *
 * Manages consent permissions for each connected data source.
 * Part of the Privacy Section - Sovereignty Control Center.
 */

import { useState } from 'react';
import { Folder, FileText, Github, Mail, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { useConnectors, useConsent, useGrantConsent, useRevokeConsent } from '@/hooks/useApi';
import { formatDistanceToNow } from 'date-fns';

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

export function ConsentManager() {
  const { data: connectors, isLoading: loadingConnectors } = useConnectors();
  const { data: consents, isLoading: loadingConsents } = useConsent();
  const grantConsentMutation = useGrantConsent();
  const revokeConsentMutation = useRevokeConsent();
  const [pendingChanges, setPendingChanges] = useState<Set<string>>(new Set());

  if (loadingConnectors || loadingConsents) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 bg-white/5 animate-pulse" />
        ))}
      </div>
    );
  }

  if (!connectors?.length) {
    return (
      <div className="text-center py-8">
        <p className="text-white/50">No data sources configured.</p>
        <p className="text-sm text-white/30 mt-1">
          Add a source to manage its consent.
        </p>
      </div>
    );
  }

  const handleConsentChange = async (
    sourceId: string,
    consentType: string,
    granted: boolean
  ) => {
    const changeKey = `${sourceId}:${consentType}`;
    setPendingChanges((prev) => new Set(prev).add(changeKey));

    try {
      if (granted) {
        await grantConsentMutation.mutateAsync({
          source_id: sourceId,
          consent_type: consentType,
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

  return (
    <div className="space-y-4">
      {connectors.map((connector) => {
        const Icon = SOURCE_ICONS[connector.connector_type] || FileText;
        const sourceConsents = consents?.filter(
          (c) => c.source_id === connector.id || c.source_name === connector.name
        ) || [];

        return (
          <div
            key={connector.id}
            className="border border-white/10 p-4"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-white/5">
                <Icon className="h-4 w-4 text-white/60" />
              </div>
              <div>
                <div className="font-medium text-white">{connector.name}</div>
                <div className="text-xs text-white/40 capitalize">
                  {connector.connector_type.replace('_', ' ')}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              {CONSENT_TYPES.map((consentType) => {
                const consent = sourceConsents.find(
                  (c) => c.consent_type === consentType.type
                );
                const isGranted = consent?.granted ?? false;
                const changeKey = `${connector.id}:${consentType.type}`;
                const isPending = pendingChanges.has(changeKey);

                return (
                  <div
                    key={consentType.type}
                    className="flex items-center justify-between"
                  >
                    <div>
                      <div className="text-sm text-white flex items-center gap-2">
                        {consentType.label}
                        {isPending ? (
                          <Loader2 className="h-3 w-3 animate-spin text-white/50" />
                        ) : isGranted ? (
                          <CheckCircle className="h-3 w-3 text-green-500" />
                        ) : (
                          <AlertCircle className="h-3 w-3 text-white/30" />
                        )}
                      </div>
                      <div className="text-xs text-white/40">
                        {consentType.description}
                      </div>
                      {consent?.granted_at && (
                        <div className="text-xs text-white/30 mt-1">
                          Granted {formatDistanceToNow(new Date(consent.granted_at), { addSuffix: true })}
                        </div>
                      )}
                    </div>
                    <Switch
                      checked={isGranted}
                      disabled={isPending}
                      onCheckedChange={(checked) =>
                        handleConsentChange(connector.id, consentType.type, checked)
                      }
                    />
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

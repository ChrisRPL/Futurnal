/**
 * ConnectorSettings Component
 *
 * Modal dialog for viewing connector settings (read-only).
 * Note: Since no update API exists in the CLI, settings cannot be changed.
 */

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Folder, FileText, Github, Mail, AlertCircle } from 'lucide-react';
import type { Connector, ConnectorType } from '@/types/api';

interface ConnectorSettingsProps {
  connector: Connector | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const TYPE_ICONS: Record<ConnectorType, React.ComponentType<{ className?: string }>> = {
  local_folder: Folder,
  obsidian: FileText,
  github: Github,
  imap: Mail,
};

const TYPE_LABELS: Record<ConnectorType, string> = {
  local_folder: 'Local Folder',
  obsidian: 'Obsidian Vault',
  github: 'GitHub Repository',
  imap: 'Email (IMAP)',
};

export function ConnectorSettings({ connector, open, onOpenChange }: ConnectorSettingsProps) {
  if (!connector) return null;

  const Icon = TYPE_ICONS[connector.connector_type] || Folder;
  const hasBeenSynced = connector.last_sync || connector.stats.files_processed > 0;

  const handleClose = () => {
    onOpenChange(false);
  };

  // Get type-specific config display
  const getConfigDetails = () => {
    const config = connector.config as Record<string, unknown>;

    switch (connector.connector_type) {
      case 'local_folder':
      case 'obsidian':
        return config?.path ? (
          <div>
            <Label className="text-white/50">Path</Label>
            <p className="text-sm text-white/80 mt-1 font-mono break-all">
              {String(config.path)}
            </p>
          </div>
        ) : null;

      case 'github':
        return config?.repo ? (
          <div>
            <Label className="text-white/50">Repository</Label>
            <p className="text-sm text-white/80 mt-1 font-mono">
              {String(config.repo)}
            </p>
          </div>
        ) : null;

      case 'imap':
        return (
          <div className="space-y-3">
            {typeof config?.server === 'string' && config.server && (
              <div>
                <Label className="text-white/50">Server</Label>
                <p className="text-sm text-white/80 mt-1 font-mono">
                  {config.server}
                </p>
              </div>
            )}
            {typeof config?.email === 'string' && config.email && (
              <div>
                <Label className="text-white/50">Email</Label>
                <p className="text-sm text-white/80 mt-1">
                  {config.email}
                </p>
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md bg-black border-white/10">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-white">
            <div className="p-2 bg-white/10">
              <Icon className="h-5 w-5 text-white/60" />
            </div>
            <div>
              <div>{connector.name}</div>
              <div className="text-xs font-normal text-white/50">
                {TYPE_LABELS[connector.connector_type]}
              </div>
            </div>
          </DialogTitle>
          <DialogDescription className="sr-only">
            Settings for {connector.name}
          </DialogDescription>
        </DialogHeader>

        <div className="py-4 space-y-6">
          {/* Status */}
          <div className="flex items-center justify-between">
            <Label className="text-white/50">Status</Label>
            <Badge
              variant={
                connector.status === 'active'
                  ? 'success'
                  : connector.status === 'error'
                    ? 'destructive'
                    : connector.status === 'paused'
                      ? 'warning'
                      : 'secondary'
              }
            >
              {connector.status}
            </Badge>
          </div>

          {/* Type-specific config (read-only) */}
          {getConfigDetails()}

          {/* Divider */}
          <div className="border-t border-white/10" />

          {/* Notice about changes */}
          <div className="flex items-start gap-2 p-3 bg-white/5 border border-white/10 text-xs text-white/50">
            <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <span>
              To change settings, remove this source and add it again with the new configuration.
            </span>
          </div>

          {/* Stats summary */}
          {hasBeenSynced ? (
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div>
                <div className="text-xs text-white/50">Files Processed</div>
                <div className="text-lg font-medium text-white">
                  {connector.stats.files_processed.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-xs text-white/50">Entities Extracted</div>
                <div className="text-lg font-medium text-white">
                  {connector.stats.entities_extracted.toLocaleString()}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-white/50 pt-2">
              Not yet synced. The orchestrator will process this source when running.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end">
          <Button
            variant="outline"
            onClick={handleClose}
            className="border-white/20 text-white/70 hover:bg-white/10"
          >
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default ConnectorSettings;

/**
 * ConnectorCard Component
 *
 * Displays a single data source connector with status, controls, and expandable details.
 * Follows Futurnal monochrome design system.
 */

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import {
  Folder,
  FileText,
  Github,
  Mail,
  RefreshCw,
  Settings,
  Trash2,
  AlertCircle,
  CheckCircle,
  Clock,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  usePauseConnector,
  useResumeConnector,
  useDeleteConnector,
  useRetryConnector,
} from '@/hooks/useApi';
import { formatDistanceToNow } from '@/lib/utils';
import type { Connector, ConnectorType, ConnectorStatus } from '@/types/api';

interface ConnectorCardProps {
  connector: Connector;
  onSettingsClick?: (connector: Connector) => void;
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

const STATUS_VARIANTS: Record<ConnectorStatus, 'success' | 'warning' | 'destructive' | 'default' | 'secondary'> = {
  active: 'success',
  paused: 'warning',
  error: 'destructive',
  syncing: 'default',
  disabled: 'secondary',
};

export function ConnectorCard({ connector, onSettingsClick }: ConnectorCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const pauseMutation = usePauseConnector();
  const resumeMutation = useResumeConnector();
  const deleteMutation = useDeleteConnector();
  const retryMutation = useRetryConnector();

  const Icon = TYPE_ICONS[connector.connector_type] || Folder;
  const isActive = connector.status === 'active' || connector.status === 'syncing';
  const isMutating = pauseMutation.isPending || resumeMutation.isPending;

  const handleToggle = () => {
    if (isMutating) return;

    if (isActive) {
      pauseMutation.mutate(connector.id);
    } else if (connector.status === 'paused') {
      resumeMutation.mutate(connector.id);
    }
  };

  const handleDelete = () => {
    if (confirm('Are you sure you want to remove this data source?')) {
      deleteMutation.mutate(connector.id);
    }
  };

  const handleRetry = () => {
    retryMutation.mutate(connector.id);
  };

  return (
    <Card
      className={cn(
        'transition-all duration-150 bg-white/5 border-white/10',
        connector.status === 'error' && 'border-red-500/50'
      )}
    >
      <CardContent className="p-4">
        {/* Main Row */}
        <div className="flex items-center gap-4">
          {/* Icon */}
          <div
            className={cn(
              'p-2.5',
              isActive ? 'bg-white/20' : 'bg-white/5'
            )}
          >
            <Icon
              className={cn(
                'h-5 w-5',
                isActive ? 'text-white' : 'text-white/40'
              )}
            />
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-medium text-white truncate">
                {connector.name}
              </h3>
              <Badge
                variant={STATUS_VARIANTS[connector.status]}
                icon={
                  connector.status === 'syncing' ? (
                    <RefreshCw className="h-3 w-3 animate-spin" />
                  ) : connector.status === 'error' ? (
                    <AlertCircle className="h-3 w-3" />
                  ) : connector.status === 'active' ? (
                    <CheckCircle className="h-3 w-3" />
                  ) : undefined
                }
              >
                {connector.status}
              </Badge>
            </div>
            <div className="flex items-center gap-3 mt-1 text-xs text-white/50">
              <span>{TYPE_LABELS[connector.connector_type]}</span>
              {connector.last_sync && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatDistanceToNow(connector.last_sync)}
                </span>
              )}
            </div>
          </div>

          {/* Toggle & Actions */}
          <div className="flex items-center gap-3">
            {connector.status !== 'disabled' && (
              <Switch
                checked={isActive}
                onCheckedChange={handleToggle}
                disabled={connector.status === 'syncing' || isMutating}
              />
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-white/60 hover:text-white"
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Progress Bar (during sync) */}
        {connector.status === 'syncing' && connector.progress && (
          <div className="mt-4">
            <Progress
              value={(connector.progress.current / connector.progress.total) * 100}
              label={connector.progress.phase}
              secondaryLabel={`${connector.progress.current} / ${connector.progress.total}`}
            />
          </div>
        )}

        {/* Error Message */}
        {connector.status === 'error' && connector.error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm text-red-400">{connector.error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2 border-red-500/30 text-red-400 hover:bg-red-500/10"
                  onClick={handleRetry}
                  disabled={retryMutation.isPending}
                >
                  <RefreshCw className={cn('h-3 w-3 mr-1', retryMutation.isPending && 'animate-spin')} />
                  Retry
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Expanded Details */}
        {isExpanded && (
          <div className="mt-4 pt-4 border-t border-white/10">
            {/* Stats */}
            {!connector.last_sync && connector.stats.files_processed === 0 ? (
              <div className="text-sm text-white/50 mb-4">
                Not yet synced. Sync will run automatically based on schedule.
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4 mb-4">
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
                <div>
                  <div className="text-xs text-white/50">Last Sync Duration</div>
                  <div className="text-lg font-medium text-white">
                    {connector.stats.last_duration
                      ? `${Math.round(connector.stats.last_duration / 1000)}s`
                      : '—'}
                  </div>
                </div>
              </div>
            )}

            {/* Next Sync */}
            {connector.next_sync && (
              <div className="text-xs text-white/50 mb-4">
                Next sync: {new Date(connector.next_sync).toLocaleString()}
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-2">
              {onSettingsClick && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onSettingsClick(connector)}
                  className="border-white/20 text-white/70 hover:bg-white/10"
                >
                  <Settings className="h-3 w-3 mr-1" />
                  Settings
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                className="border-red-500/30 text-red-400 hover:bg-red-500/10"
                onClick={handleDelete}
                disabled={deleteMutation.isPending}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Remove
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default ConnectorCard;

Summary: Implement data source connector management dashboard with status monitoring and tier enforcement.

# 08 · Connector Dashboard

## Purpose

Create the **Experiential Connector Dashboard**—the control center where users manage the data sources that feed their Ghost's stream of experience. Every connector represents a new dimension of the user's personal universe that their Ghost can learn from.

> **"Feed your Ghost's memories"** — Each connector extends your Ghost's understanding of your personal universe. The more experiential data you connect, the deeper your Ghost's comprehension grows.

This dashboard includes status indicators, sync progress, error handling, and tier-based source limits (Free: 3 sources, Pro: unlimited).

**Criticality**: HIGH - Core data management functionality

## Scope

- Data source list with real-time status indicators
- Enable/disable toggles per connector
- Last sync time and next scheduled sync display
- Ingestion progress bars with file counts
- Error alerts with retry actions
- "Add New Connector" flow with type selection
- Tier enforcement (3-source limit for Free tier)
- Source types: Local Folder, Obsidian, GitHub, IMAP
- Connector configuration editing
- Bulk actions (pause all, resume all)

## Requirements Alignment

- **Feature Requirement**: "Connector dashboard shows sources, statuses, and controls"
- **Tier Enforcement**: Free tier limited to 3 data sources
- **Orchestrator Integration**: Uses `collect_status_report()` for status

## Component Design

### Connectors Page

```tsx
// src/pages/Connectors.tsx
import { useState } from 'react';
import { useConnectorsStore } from '@/stores/connectorsStore';
import { useSubscription } from '@/hooks/useSubscription';
import { ConnectorCard } from '@/components/connectors/ConnectorCard';
import { AddConnectorModal } from '@/components/connectors/AddConnectorModal';
import { UpgradePrompt } from '@/components/pricing/UpgradePrompt';
import { Button } from '@/components/ui/button';
import { Plus, Play, Pause, RefreshCw } from 'lucide-react';

export function ConnectorsPage() {
  const { connectors, pauseAll, resumeAll, refreshStatus } = useConnectorsStore();
  const { tier, limits } = useSubscription();
  const [showAddModal, setShowAddModal] = useState(false);
  const [showUpgradePrompt, setShowUpgradePrompt] = useState(false);

  const activeCount = connectors.filter((c) => c.status !== 'disabled').length;
  const canAddMore = tier === 'pro' || activeCount < limits.maxSources;

  const handleAddConnector = () => {
    if (canAddMore) {
      setShowAddModal(true);
    } else {
      setShowUpgradePrompt(true);
    }
  };

  const pausedCount = connectors.filter((c) => c.status === 'paused').length;
  const activeRunning = connectors.filter((c) => c.status === 'active').length;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">
            Data Sources
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {activeCount} of {tier === 'pro' ? '∞' : limits.maxSources} sources configured
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Bulk Actions */}
          {connectors.length > 0 && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={refreshStatus}
                className="gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
              {activeRunning > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={pauseAll}
                  className="gap-2"
                >
                  <Pause className="h-4 w-4" />
                  Pause All
                </Button>
              )}
              {pausedCount > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resumeAll}
                  className="gap-2"
                >
                  <Play className="h-4 w-4" />
                  Resume All
                </Button>
              )}
            </>
          )}

          {/* Add Connector */}
          <Button onClick={handleAddConnector} className="gap-2">
            <Plus className="h-4 w-4" />
            Add Source
          </Button>
        </div>
      </div>

      {/* Tier Warning for Free Users */}
      {tier === 'free' && activeCount >= limits.maxSources && (
        <div className="mb-4 p-4 rounded-lg bg-warning/10 border border-warning/20">
          <p className="text-sm text-warning">
            You've reached the {limits.maxSources}-source limit on the Free tier.{' '}
            <button
              onClick={() => setShowUpgradePrompt(true)}
              className="underline hover:no-underline"
            >
              Upgrade to Pro
            </button>{' '}
            for unlimited sources.
          </p>
        </div>
      )}

      {/* Connector List */}
      {connectors.length === 0 ? (
        <EmptyState onAdd={handleAddConnector} />
      ) : (
        <div className="space-y-3">
          {connectors.map((connector) => (
            <ConnectorCard key={connector.id} connector={connector} />
          ))}
        </div>
      )}

      {/* Add Connector Modal */}
      <AddConnectorModal
        open={showAddModal}
        onOpenChange={setShowAddModal}
      />

      {/* Upgrade Prompt */}
      <UpgradePrompt
        open={showUpgradePrompt}
        onOpenChange={setShowUpgradePrompt}
        reason="source_limit"
      />
    </div>
  );
}

function EmptyState({ onAdd }: { onAdd: () => void }) {
  return (
    <div className="text-center py-16 border border-dashed border-border rounded-lg">
      <div className="w-12 h-12 rounded-full bg-background-elevated mx-auto mb-4 flex items-center justify-center">
        <Plus className="h-6 w-6 text-text-tertiary" />
      </div>
      <h3 className="text-lg font-medium text-text-primary mb-2">
        Your Ghost awaits its first memories
      </h3>
      <p className="text-text-secondary mb-4 max-w-sm mx-auto">
        Connect your first experiential data source to begin grounding your Ghost in your personal universe.
      </p>
      <Button onClick={onAdd}>
        Feed Your Ghost
      </Button>
    </div>
  );
}
```

### Connector Card Component

```tsx
// src/components/connectors/ConnectorCard.tsx
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
  Play,
  Pause,
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
import { useConnectorsStore } from '@/stores/connectorsStore';
import { ConnectorSettings } from './ConnectorSettings';
import { formatDistanceToNow } from 'date-fns';

interface Connector {
  id: string;
  name: string;
  type: 'local_folder' | 'obsidian' | 'github' | 'imap';
  status: 'active' | 'paused' | 'error' | 'syncing' | 'disabled';
  lastSync?: string;
  nextSync?: string;
  progress?: {
    current: number;
    total: number;
    phase: string;
  };
  error?: string;
  config: Record<string, unknown>;
  stats: {
    filesProcessed: number;
    entitiesExtracted: number;
    lastDuration?: number;
  };
}

interface ConnectorCardProps {
  connector: Connector;
}

const TYPE_ICONS = {
  local_folder: Folder,
  obsidian: FileText,
  github: Github,
  imap: Mail,
};

const TYPE_LABELS = {
  local_folder: 'Local Folder',
  obsidian: 'Obsidian Vault',
  github: 'GitHub Repository',
  imap: 'Email (IMAP)',
};

const STATUS_COLORS = {
  active: 'success',
  paused: 'warning',
  error: 'destructive',
  syncing: 'default',
  disabled: 'secondary',
};

export function ConnectorCard({ connector }: ConnectorCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const { pauseConnector, resumeConnector, retryConnector, deleteConnector } =
    useConnectorsStore();

  const Icon = TYPE_ICONS[connector.type];
  const isActive = connector.status === 'active' || connector.status === 'syncing';

  const handleToggle = () => {
    if (isActive) {
      pauseConnector(connector.id);
    } else if (connector.status === 'paused') {
      resumeConnector(connector.id);
    }
  };

  return (
    <>
      <Card
        className={cn(
          'transition-all duration-150',
          connector.status === 'error' && 'border-destructive/50'
        )}
      >
        <CardContent className="p-4">
          {/* Main Row */}
          <div className="flex items-center gap-4">
            {/* Icon */}
            <div
              className={cn(
                'p-2.5 rounded-lg',
                isActive ? 'bg-primary/20' : 'bg-background-elevated'
              )}
            >
              <Icon
                className={cn(
                  'h-5 w-5',
                  isActive ? 'text-primary' : 'text-text-tertiary'
                )}
              />
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-medium text-text-primary truncate">
                  {connector.name}
                </h3>
                <Badge variant={STATUS_COLORS[connector.status] as any}>
                  {connector.status === 'syncing' ? (
                    <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                  ) : connector.status === 'error' ? (
                    <AlertCircle className="h-3 w-3 mr-1" />
                  ) : connector.status === 'active' ? (
                    <CheckCircle className="h-3 w-3 mr-1" />
                  ) : null}
                  {connector.status}
                </Badge>
              </div>
              <div className="flex items-center gap-3 mt-1 text-xs text-text-tertiary">
                <span>{TYPE_LABELS[connector.type]}</span>
                {connector.lastSync && (
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatDistanceToNow(new Date(connector.lastSync), {
                      addSuffix: true,
                    })}
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
                  disabled={connector.status === 'syncing'}
                />
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsExpanded(!isExpanded)}
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
              <div className="flex justify-between text-xs text-text-secondary mb-1">
                <span>{connector.progress.phase}</span>
                <span>
                  {connector.progress.current} / {connector.progress.total}
                </span>
              </div>
              <Progress
                value={(connector.progress.current / connector.progress.total) * 100}
              />
            </div>
          )}

          {/* Error Message */}
          {connector.status === 'error' && connector.error && (
            <div className="mt-4 p-3 rounded-md bg-destructive/10 border border-destructive/20">
              <div className="flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm text-destructive">{connector.error}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-2"
                    onClick={() => retryConnector(connector.id)}
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Retry
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Expanded Details */}
          {isExpanded && (
            <div className="mt-4 pt-4 border-t border-border">
              {/* Stats */}
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <div className="text-xs text-text-tertiary">Files Processed</div>
                  <div className="text-lg font-medium text-text-primary">
                    {connector.stats.filesProcessed.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-text-tertiary">Entities Extracted</div>
                  <div className="text-lg font-medium text-text-primary">
                    {connector.stats.entitiesExtracted.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-text-tertiary">Last Sync Duration</div>
                  <div className="text-lg font-medium text-text-primary">
                    {connector.stats.lastDuration
                      ? `${Math.round(connector.stats.lastDuration / 1000)}s`
                      : '—'}
                  </div>
                </div>
              </div>

              {/* Next Sync */}
              {connector.nextSync && (
                <div className="text-xs text-text-tertiary mb-4">
                  Next sync: {new Date(connector.nextSync).toLocaleString()}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSettings(true)}
                >
                  <Settings className="h-3 w-3 mr-1" />
                  Settings
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-destructive hover:bg-destructive/10"
                  onClick={() => {
                    if (confirm('Are you sure you want to remove this source?')) {
                      deleteConnector(connector.id);
                    }
                  }}
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Remove
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Settings Modal */}
      <ConnectorSettings
        connector={connector}
        open={showSettings}
        onOpenChange={setShowSettings}
      />
    </>
  );
}
```

### Add Connector Modal

```tsx
// src/components/connectors/AddConnectorModal.tsx
import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Folder, FileText, Github, Mail, ArrowLeft } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useConnectorsStore } from '@/stores/connectorsStore';
import { open } from '@tauri-apps/plugin-dialog';

interface AddConnectorModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type ConnectorType = 'local_folder' | 'obsidian' | 'github' | 'imap';

const CONNECTOR_TYPES = [
  {
    type: 'local_folder' as const,
    label: 'Local Folder',
    description: 'Ground your Ghost in documents, images, and local files',
    icon: Folder,
  },
  {
    type: 'obsidian' as const,
    label: 'Obsidian Vault',
    description: 'Let your Ghost learn from your thought network and knowledge patterns',
    icon: FileText,
  },
  {
    type: 'github' as const,
    label: 'GitHub Repository',
    description: 'Expand your Ghost's understanding of your code and projects',
    icon: Github,
  },
  {
    type: 'imap' as const,
    label: 'Email (IMAP)',
    description: 'Give your Ghost access to your communication patterns',
    icon: Mail,
  },
];

export function AddConnectorModal({ open, onOpenChange }: AddConnectorModalProps) {
  const [step, setStep] = useState<'select' | 'configure'>('select');
  const [selectedType, setSelectedType] = useState<ConnectorType | null>(null);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const { addConnector } = useConnectorsStore();

  const handleSelectType = (type: ConnectorType) => {
    setSelectedType(type);
    setStep('configure');
    setConfig({});
  };

  const handleBack = () => {
    setStep('select');
    setSelectedType(null);
    setConfig({});
  };

  const handleClose = () => {
    onOpenChange(false);
    setTimeout(() => {
      setStep('select');
      setSelectedType(null);
      setConfig({});
    }, 200);
  };

  const handleBrowseFolder = async () => {
    const selected = await open({
      directory: true,
      multiple: false,
      title: 'Select Folder',
    });
    if (selected) {
      setConfig((prev) => ({ ...prev, path: selected as string }));
    }
  };

  const handleSubmit = async () => {
    if (!selectedType) return;

    setIsLoading(true);
    try {
      await addConnector({
        type: selectedType,
        name: config.name || `New ${selectedType}`,
        config,
      });
      handleClose();
    } catch (error) {
      console.error('Failed to add connector:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {step === 'configure' && (
              <button onClick={handleBack} className="hover:bg-background-elevated rounded p-1">
                <ArrowLeft className="h-4 w-4" />
              </button>
            )}
            {step === 'select' ? 'Add Data Source' : `Configure ${selectedType}`}
          </DialogTitle>
        </DialogHeader>

        {step === 'select' ? (
          <div className="grid grid-cols-2 gap-3 py-4">
            {CONNECTOR_TYPES.map((type) => (
              <button
                key={type.type}
                onClick={() => handleSelectType(type.type)}
                className={cn(
                  'flex flex-col items-start p-4 rounded-lg border border-border',
                  'hover:border-primary hover:bg-primary/5 transition-colors',
                  'text-left'
                )}
              >
                <type.icon className="h-6 w-6 text-primary mb-2" />
                <div className="font-medium text-text-primary">{type.label}</div>
                <div className="text-xs text-text-tertiary mt-1">
                  {type.description}
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="py-4 space-y-4">
            {/* Name */}
            <div>
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={config.name || ''}
                onChange={(e) => setConfig((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="My Data Source"
              />
            </div>

            {/* Type-specific fields */}
            {(selectedType === 'local_folder' || selectedType === 'obsidian') && (
              <div>
                <Label htmlFor="path">
                  {selectedType === 'obsidian' ? 'Vault Path' : 'Folder Path'}
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="path"
                    value={config.path || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, path: e.target.value }))}
                    placeholder="/path/to/folder"
                    className="flex-1"
                  />
                  <Button variant="outline" onClick={handleBrowseFolder}>
                    Browse
                  </Button>
                </div>
              </div>
            )}

            {selectedType === 'github' && (
              <>
                <div>
                  <Label htmlFor="repo">Repository URL</Label>
                  <Input
                    id="repo"
                    value={config.repo || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, repo: e.target.value }))}
                    placeholder="https://github.com/owner/repo"
                  />
                </div>
                <div>
                  <Label htmlFor="token">Personal Access Token</Label>
                  <Input
                    id="token"
                    type="password"
                    value={config.token || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, token: e.target.value }))}
                    placeholder="ghp_..."
                  />
                  <p className="text-xs text-text-tertiary mt-1">
                    Required for private repos. Create at GitHub Settings → Developer Settings.
                  </p>
                </div>
              </>
            )}

            {selectedType === 'imap' && (
              <>
                <div>
                  <Label htmlFor="server">IMAP Server</Label>
                  <Input
                    id="server"
                    value={config.server || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, server: e.target.value }))}
                    placeholder="imap.gmail.com"
                  />
                </div>
                <div>
                  <Label htmlFor="email">Email Address</Label>
                  <Input
                    id="email"
                    type="email"
                    value={config.email || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, email: e.target.value }))}
                    placeholder="you@example.com"
                  />
                </div>
                <div>
                  <Label htmlFor="password">Password / App Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={config.password || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, password: e.target.value }))}
                  />
                </div>
              </>
            )}

            {/* Submit */}
            <div className="flex justify-end gap-2 pt-4">
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button onClick={handleSubmit} disabled={isLoading}>
                {isLoading ? 'Adding...' : 'Add Source'}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
```

### Connector Settings Modal

```tsx
// src/components/connectors/ConnectorSettings.tsx
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { useState } from 'react';
import { useConnectorsStore } from '@/stores/connectorsStore';

interface ConnectorSettingsProps {
  connector: {
    id: string;
    name: string;
    type: string;
    config: Record<string, unknown>;
  };
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ConnectorSettings({
  connector,
  open,
  onOpenChange,
}: ConnectorSettingsProps) {
  const [name, setName] = useState(connector.name);
  const [config, setConfig] = useState(connector.config);
  const { updateConnector } = useConnectorsStore();

  const handleSave = async () => {
    await updateConnector(connector.id, { name, config });
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Connector Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div>
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label>Auto-sync</Label>
              <p className="text-xs text-text-tertiary">
                Automatically sync on schedule
              </p>
            </div>
            <Switch
              checked={(config.autoSync as boolean) ?? true}
              onCheckedChange={(checked) =>
                setConfig((prev) => ({ ...prev, autoSync: checked }))
              }
            />
          </div>

          <div>
            <Label htmlFor="interval">Sync Interval (minutes)</Label>
            <Input
              id="interval"
              type="number"
              min={5}
              value={(config.syncInterval as number) ?? 60}
              onChange={(e) =>
                setConfig((prev) => ({
                  ...prev,
                  syncInterval: parseInt(e.target.value),
                }))
              }
            />
          </div>
        </div>

        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save Changes</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

### Connectors Store

```tsx
// src/stores/connectorsStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';

interface Connector {
  id: string;
  name: string;
  type: 'local_folder' | 'obsidian' | 'github' | 'imap';
  status: 'active' | 'paused' | 'error' | 'syncing' | 'disabled';
  lastSync?: string;
  nextSync?: string;
  progress?: {
    current: number;
    total: number;
    phase: string;
  };
  error?: string;
  config: Record<string, unknown>;
  stats: {
    filesProcessed: number;
    entitiesExtracted: number;
    lastDuration?: number;
  };
}

interface ConnectorsState {
  connectors: Connector[];
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchConnectors: () => Promise<void>;
  addConnector: (data: {
    type: Connector['type'];
    name: string;
    config: Record<string, unknown>;
  }) => Promise<void>;
  updateConnector: (
    id: string,
    updates: Partial<Pick<Connector, 'name' | 'config'>>
  ) => Promise<void>;
  deleteConnector: (id: string) => Promise<void>;
  pauseConnector: (id: string) => Promise<void>;
  resumeConnector: (id: string) => Promise<void>;
  retryConnector: (id: string) => Promise<void>;
  pauseAll: () => Promise<void>;
  resumeAll: () => Promise<void>;
  refreshStatus: () => Promise<void>;
}

export const useConnectorsStore = create<ConnectorsState>()(
  persist(
    (set, get) => ({
      connectors: [],
      isLoading: false,
      error: null,

      fetchConnectors: async () => {
        set({ isLoading: true, error: null });
        try {
          const connectors = await invoke<Connector[]>('list_sources');
          set({ connectors, isLoading: false });
        } catch (error) {
          set({ error: String(error), isLoading: false });
        }
      },

      addConnector: async (data) => {
        try {
          const connector = await invoke<Connector>('add_source', data);
          set((state) => ({
            connectors: [...state.connectors, connector],
          }));
        } catch (error) {
          throw error;
        }
      },

      updateConnector: async (id, updates) => {
        try {
          await invoke('update_source', { id, ...updates });
          set((state) => ({
            connectors: state.connectors.map((c) =>
              c.id === id ? { ...c, ...updates } : c
            ),
          }));
        } catch (error) {
          throw error;
        }
      },

      deleteConnector: async (id) => {
        try {
          await invoke('delete_source', { id });
          set((state) => ({
            connectors: state.connectors.filter((c) => c.id !== id),
          }));
        } catch (error) {
          throw error;
        }
      },

      pauseConnector: async (id) => {
        try {
          await invoke('pause_source', { id });
          set((state) => ({
            connectors: state.connectors.map((c) =>
              c.id === id ? { ...c, status: 'paused' } : c
            ),
          }));
        } catch (error) {
          throw error;
        }
      },

      resumeConnector: async (id) => {
        try {
          await invoke('resume_source', { id });
          set((state) => ({
            connectors: state.connectors.map((c) =>
              c.id === id ? { ...c, status: 'active' } : c
            ),
          }));
        } catch (error) {
          throw error;
        }
      },

      retryConnector: async (id) => {
        try {
          await invoke('retry_source', { id });
          set((state) => ({
            connectors: state.connectors.map((c) =>
              c.id === id ? { ...c, status: 'syncing', error: undefined } : c
            ),
          }));
        } catch (error) {
          throw error;
        }
      },

      pauseAll: async () => {
        try {
          await invoke('pause_all_sources');
          set((state) => ({
            connectors: state.connectors.map((c) =>
              c.status === 'active' ? { ...c, status: 'paused' } : c
            ),
          }));
        } catch (error) {
          throw error;
        }
      },

      resumeAll: async () => {
        try {
          await invoke('resume_all_sources');
          set((state) => ({
            connectors: state.connectors.map((c) =>
              c.status === 'paused' ? { ...c, status: 'active' } : c
            ),
          }));
        } catch (error) {
          throw error;
        }
      },

      refreshStatus: async () => {
        await get().fetchConnectors();
      },
    }),
    {
      name: 'futurnal-connectors',
      partialize: (state) => ({ connectors: state.connectors }),
    }
  )
);
```

## Acceptance Criteria

- [ ] Connector list displays all configured sources
- [ ] Status badges show correct colors (green/yellow/red)
- [ ] Enable/disable toggles work correctly
- [ ] Progress bar displays during sync
- [ ] Error messages display with retry option
- [ ] Add connector modal works for all types
- [ ] Tier enforcement blocks adding beyond limit
- [ ] Upgrade prompt displays for Free tier at limit
- [ ] Bulk pause/resume works
- [ ] Connector settings can be edited
- [ ] Delete confirmation and removal works
- [ ] Stats display accurately

## Test Plan

### Unit Tests
```typescript
describe('ConnectorCard', () => {
  it('should display correct status badge', () => {
    render(<ConnectorCard connector={{ status: 'active', ...mockConnector }} />);
    expect(screen.getByText('active')).toHaveClass('bg-success');
  });

  it('should show progress during sync', () => {
    const connector = {
      ...mockConnector,
      status: 'syncing',
      progress: { current: 50, total: 100, phase: 'Processing' },
    };
    render(<ConnectorCard connector={connector} />);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '50');
  });
});
```

### E2E Tests
```typescript
test('add connector flow', async ({ page }) => {
  await page.goto('/connectors');
  await page.click('text=Add Source');
  await page.click('text=Obsidian Vault');
  await page.fill('input[name="name"]', 'My Vault');
  await page.fill('input[name="path"]', '/path/to/vault');
  await page.click('text=Add Source');

  await expect(page.locator('text=My Vault')).toBeVisible();
});
```

## Dependencies

- @tauri-apps/plugin-dialog (for folder picker)
- zustand with persist middleware
- date-fns for time formatting

## Next Steps

After connector dashboard complete:
1. Add connector health monitoring
2. Implement sync scheduling UI
3. Add file exclusion patterns
4. Create sync history view

---

## UI Copy Reference

| Element | Copy | Purpose |
|---------|------|---------|
| Page title | "Data Sources" | Simple, clear |
| Empty state headline | "Your Ghost awaits its first memories" | Personalizes the Ghost |
| Empty state CTA | "Feed Your Ghost" | Action-oriented, experiential |
| Local Folder desc | "Ground your Ghost in documents, images, and local files" | Experiential framing |
| Obsidian desc | "Let your Ghost learn from your thought network and knowledge patterns" | Emphasizes learning |
| GitHub desc | "Expand your Ghost's understanding of your code and projects" | Emphasizes understanding |
| Email desc | "Give your Ghost access to your communication patterns" | Pattern recognition focus |
| Upgrade prompt | "Unlock unlimited experiential sources with Pro" | Ties to evolution |

---

**This connector dashboard is where users build their Ghost's experiential foundation—every connected source deepens the Ghost's understanding of their personal universe.**

Summary: Implement privacy controls UI with consent management, audit log viewer, and data export options.

# 11 · Privacy & Settings Panel

## Purpose

Create the **Sovereignty Control Center**—the comprehensive settings panel where users exercise absolute control over their data and their Ghost's access permissions. This module embodies Futurnal's core value of **Sovereignty**: *"The user is in absolute control. Their data is theirs, period."*

> **"Your data remains yours. Always."** — Privacy isn't a feature—it's the foundation of the Ghost→Animal evolution. Your Ghost learns from your experiential data *only* with your explicit, revocable consent.

This module implements privacy controls, consent management, audit log viewer, and user preferences with complete transparency.

**Criticality**: HIGH - Core privacy commitment and user sovereignty

## Scope

- Settings page with organized sections
- Profile & Account management
- Privacy Controls:
  - Consent management per source
  - Telemetry opt-in/out
  - Data retention settings
- Audit log viewer with filtering
- Data export functionality
- Cloud backup consent (Pro tier)
- Search history management
- Appearance settings (future light mode)
- Keyboard shortcuts reference

## Requirements Alignment

- **Privacy-First**: "Explicit consent, comprehensive audit logging"
- **User Control**: Full transparency over data handling
- **GDPR/CCPA Ready**: Data export and deletion capabilities

## Component Design

### Settings Page Layout

```tsx
// src/pages/Settings.tsx
import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ProfileSection } from '@/components/settings/ProfileSection';
import { PrivacySection } from '@/components/settings/PrivacySection';
import { AppearanceSection } from '@/components/settings/AppearanceSection';
import { ConnectorsSection } from '@/components/settings/ConnectorsSection';
import { DataSection } from '@/components/settings/DataSection';
import { AboutSection } from '@/components/settings/AboutSection';
import {
  User,
  Shield,
  Palette,
  Database,
  HardDrive,
  Info,
} from 'lucide-react';

const SECTIONS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'privacy', label: 'Privacy', icon: Shield },
  { id: 'appearance', label: 'Appearance', icon: Palette },
  { id: 'connectors', label: 'Data Sources', icon: Database },
  { id: 'data', label: 'Data Management', icon: HardDrive },
  { id: 'about', label: 'About', icon: Info },
];

export function SettingsPage() {
  const [activeSection, setActiveSection] = useState('profile');

  return (
    <div className="flex h-full">
      {/* Sidebar Navigation */}
      <nav className="w-56 border-r border-border bg-background-surface p-4">
        <h1 className="text-lg font-semibold text-text-primary mb-6">Settings</h1>
        <ul className="space-y-1">
          {SECTIONS.map((section) => (
            <li key={section.id}>
              <button
                onClick={() => setActiveSection(section.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                  activeSection === section.id
                    ? 'bg-primary/10 text-primary'
                    : 'text-text-secondary hover:bg-background-elevated hover:text-text-primary'
                }`}
              >
                <section.icon className="h-4 w-4" />
                {section.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Content Area */}
      <main className="flex-1 overflow-y-auto p-8">
        <div className="max-w-2xl">
          {activeSection === 'profile' && <ProfileSection />}
          {activeSection === 'privacy' && <PrivacySection />}
          {activeSection === 'appearance' && <AppearanceSection />}
          {activeSection === 'connectors' && <ConnectorsSection />}
          {activeSection === 'data' && <DataSection />}
          {activeSection === 'about' && <AboutSection />}
        </div>
      </main>
    </div>
  );
}
```

### Privacy Section

```tsx
// src/components/settings/PrivacySection.tsx
import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ConsentManager } from './ConsentManager';
import { AuditLogViewer } from './AuditLogViewer';
import { useSettingsStore } from '@/stores/settingsStore';
import { useIsPro } from '@/stores/userStore';
import { Shield, Eye, FileText, Clock, Lock } from 'lucide-react';

export function PrivacySection() {
  const [showAuditLogs, setShowAuditLogs] = useState(false);
  const isPro = useIsPro();
  const {
    telemetryEnabled,
    crashReportsEnabled,
    analyticsEnabled,
    setSetting,
  } = useSettingsStore();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Privacy</h2>
        <p className="text-sm text-text-secondary mt-1">
          Control how your data is collected, stored, and used.
        </p>
      </div>

      {/* Data Collection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Data Collection
          </CardTitle>
          <CardDescription>
            Choose what information Futurnal can collect to improve the product.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-text-primary text-sm">
                Anonymous Telemetry
              </div>
              <div className="text-xs text-text-tertiary">
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
              <div className="font-medium text-text-primary text-sm">
                Crash Reports
              </div>
              <div className="text-xs text-text-tertiary">
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
              <div className="font-medium text-text-primary text-sm">
                Feature Analytics
              </div>
              <div className="text-xs text-text-tertiary">
                Share feature usage to help prioritize development
              </div>
            </div>
            <Switch
              checked={analyticsEnabled}
              onCheckedChange={(checked) => setSetting('analyticsEnabled', checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Source Consent */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Data Source Consent
          </CardTitle>
          <CardDescription>
            Manage permissions for each connected data source.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ConsentManager />
        </CardContent>
      </Card>

      {/* Audit Log */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Activity Audit Log
          </CardTitle>
          <CardDescription>
            View a complete history of data access and operations.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            variant="outline"
            onClick={() => setShowAuditLogs(true)}
            className="w-full"
          >
            View Audit Log
          </Button>
        </CardContent>
      </Card>

      {/* Data Retention */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Data Retention
          </CardTitle>
          <CardDescription>
            Control how long your data is stored locally.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-text-primary text-sm">
                Search History
              </div>
              <div className="text-xs text-text-tertiary">
                Keep last 50 searches
              </div>
            </div>
            <Badge variant="secondary">50 items</Badge>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-text-primary text-sm">
                Audit Logs
              </div>
              <div className="text-xs text-text-tertiary">
                Retained for 90 days
              </div>
            </div>
            <Badge variant="secondary">90 days</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Cloud Backup (Pro) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Lock className="h-4 w-4" />
            Encrypted Cloud Backup
            {!isPro && (
              <Badge variant="accent" className="ml-2">
                Pro
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            {isPro
              ? 'Securely backup your data with end-to-end encryption.'
              : 'Upgrade to Pro for encrypted cloud backup.'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isPro ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-text-primary text-sm">
                    Enable Cloud Backup
                  </div>
                  <div className="text-xs text-text-tertiary">
                    All data encrypted before upload
                  </div>
                </div>
                <Switch />
              </div>
            </div>
          ) : (
            <Button variant="outline" className="w-full">
              Upgrade to Pro
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Audit Log Modal */}
      <AuditLogViewer open={showAuditLogs} onOpenChange={setShowAuditLogs} />
    </div>
  );
}
```

### Consent Manager

```tsx
// src/components/settings/ConsentManager.tsx
import { useState } from 'react';
import { useConsent, useGrantConsent } from '@/hooks/useApi';
import { useConnectors } from '@/hooks/useApi';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Folder,
  FileText,
  Github,
  Mail,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

const SOURCE_ICONS = {
  local_folder: Folder,
  obsidian: FileText,
  github: Github,
  imap: Mail,
};

const CONSENT_TYPES = [
  { type: 'read', label: 'Read Access', description: 'Allow your Ghost to access and index this source' },
  { type: 'process', label: 'AI Processing', description: 'Allow your Ghost to extract entities and relationships' },
  { type: 'store', label: 'Memory Storage', description: 'Allow your Ghost to remember content in its knowledge graph' },
];

export function ConsentManager() {
  const { data: connectors, isLoading: loadingConnectors } = useConnectors();
  const { data: consents, isLoading: loadingConsents } = useConsent();
  const grantConsentMutation = useGrantConsent();

  if (loadingConnectors || loadingConsents) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
    );
  }

  if (!connectors?.length) {
    return (
      <div className="text-center py-8 text-text-tertiary">
        <p>No data sources configured.</p>
        <p className="text-sm mt-1">Add a source to manage its consent.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {connectors.map((connector) => {
        const Icon = SOURCE_ICONS[connector.type] || FileText;
        const sourceConsents = consents?.filter(
          (c) => c.source_id === connector.id
        ) || [];

        return (
          <div
            key={connector.id}
            className="border border-border rounded-lg p-4"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-background-elevated">
                <Icon className="h-4 w-4 text-text-secondary" />
              </div>
              <div>
                <div className="font-medium text-text-primary">
                  {connector.name}
                </div>
                <div className="text-xs text-text-tertiary">
                  {connector.type}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              {CONSENT_TYPES.map((consentType) => {
                const consent = sourceConsents.find(
                  (c) => c.consent_type === consentType.type
                );
                const isGranted = consent?.granted ?? false;

                return (
                  <div
                    key={consentType.type}
                    className="flex items-center justify-between"
                  >
                    <div>
                      <div className="text-sm text-text-primary flex items-center gap-2">
                        {consentType.label}
                        {isGranted ? (
                          <CheckCircle className="h-3 w-3 text-secondary" />
                        ) : (
                          <AlertCircle className="h-3 w-3 text-text-tertiary" />
                        )}
                      </div>
                      <div className="text-xs text-text-tertiary">
                        {consentType.description}
                      </div>
                      {consent?.granted_at && (
                        <div className="text-xs text-text-tertiary mt-1">
                          Granted {formatDistanceToNow(new Date(consent.granted_at), { addSuffix: true })}
                        </div>
                      )}
                    </div>
                    <Switch
                      checked={isGranted}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          grantConsentMutation.mutate({
                            source_id: connector.id,
                            consent_type: consentType.type,
                          });
                        } else {
                          // Revoke consent
                        }
                      }}
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
```

### Audit Log Viewer

```tsx
// src/components/settings/AuditLogViewer.tsx
import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useAuditLogs } from '@/hooks/useApi';
import { Search, Download, RefreshCw } from 'lucide-react';
import { format } from 'date-fns';

interface AuditLogViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ACTION_COLORS: Record<string, string> = {
  search: 'default',
  read: 'secondary',
  write: 'warning',
  delete: 'destructive',
  consent_grant: 'success',
  consent_revoke: 'destructive',
  sync: 'default',
  export: 'accent',
};

export function AuditLogViewer({ open, onOpenChange }: AuditLogViewerProps) {
  const [filter, setFilter] = useState('');
  const [actionFilter, setActionFilter] = useState<string | undefined>();
  const [limit, setLimit] = useState(50);

  const { data: logs, isLoading, refetch } = useAuditLogs({
    limit,
    action_filter: actionFilter,
  });

  const filteredLogs = logs?.filter((log) =>
    filter
      ? log.action.toLowerCase().includes(filter.toLowerCase()) ||
        log.resource_type.toLowerCase().includes(filter.toLowerCase())
      : true
  );

  const handleExport = () => {
    if (!logs) return;

    const csv = [
      ['Timestamp', 'Action', 'Resource Type', 'Resource ID', 'Details'],
      ...logs.map((log) => [
        log.timestamp,
        log.action,
        log.resource_type,
        log.resource_id || '',
        JSON.stringify(log.details || {}),
      ]),
    ]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `futurnal-audit-log-${format(new Date(), 'yyyy-MM-dd')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Activity Audit Log</DialogTitle>
        </DialogHeader>

        {/* Filters */}
        <div className="flex items-center gap-3 py-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-tertiary" />
            <Input
              placeholder="Search logs..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="pl-9"
            />
          </div>
          <Select value={actionFilter} onValueChange={setActionFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="All actions" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">All actions</SelectItem>
              <SelectItem value="search">Search</SelectItem>
              <SelectItem value="read">Read</SelectItem>
              <SelectItem value="write">Write</SelectItem>
              <SelectItem value="sync">Sync</SelectItem>
              <SelectItem value="consent">Consent</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleExport}>
            <Download className="h-4 w-4" />
          </Button>
        </div>

        {/* Log List */}
        <ScrollArea className="h-[500px] pr-4">
          {isLoading ? (
            <div className="text-center py-8 text-text-tertiary">
              Loading logs...
            </div>
          ) : filteredLogs?.length === 0 ? (
            <div className="text-center py-8 text-text-tertiary">
              No logs found
            </div>
          ) : (
            <div className="space-y-2">
              {filteredLogs?.map((log) => (
                <div
                  key={log.id}
                  className="flex items-start gap-4 p-3 rounded-lg bg-background-elevated"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <Badge variant={ACTION_COLORS[log.action] as any || 'secondary'}>
                        {log.action}
                      </Badge>
                      <span className="text-sm text-text-secondary">
                        {log.resource_type}
                      </span>
                    </div>
                    {log.resource_id && (
                      <div className="text-xs text-text-tertiary mt-1 truncate">
                        {log.resource_id}
                      </div>
                    )}
                    {log.details && (
                      <div className="text-xs text-text-tertiary mt-1">
                        {JSON.stringify(log.details)}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-text-tertiary whitespace-nowrap">
                    {format(new Date(log.timestamp), 'MMM d, HH:mm:ss')}
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>

        {/* Load More */}
        {logs && logs.length >= limit && (
          <div className="text-center pt-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setLimit((l) => l + 50)}
            >
              Load more
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
```

### Data Management Section

```tsx
// src/components/settings/DataSection.tsx
import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useSearchStore } from '@/stores/searchStore';
import {
  Download,
  Trash2,
  Database,
  History,
  AlertTriangle,
} from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

export function DataSection() {
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const { history, clearHistory } = useSearchStore();

  const handleExportData = async () => {
    setExportProgress(0);

    // Simulate export progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise((r) => setTimeout(r, 200));
      setExportProgress(i);
    }

    // Trigger download
    // In production, this would call the Tauri command
    setExportProgress(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Data Management</h2>
        <p className="text-sm text-text-secondary mt-1">
          Export, manage, and clear your data.
        </p>
      </div>

      {/* Storage Usage */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Database className="h-4 w-4" />
            Local Storage
          </CardTitle>
          <CardDescription>
            Your data is stored locally on this device.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-text-secondary">Knowledge Graph</span>
                <span className="text-text-primary">245 MB</span>
              </div>
              <Progress value={24.5} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-text-secondary">Vector Embeddings</span>
                <span className="text-text-primary">128 MB</span>
              </div>
              <Progress value={12.8} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-text-secondary">Audit Logs</span>
                <span className="text-text-primary">12 MB</span>
              </div>
              <Progress value={1.2} />
            </div>
            <div className="pt-2 border-t border-border">
              <div className="flex justify-between text-sm">
                <span className="text-text-primary font-medium">Total</span>
                <span className="text-text-primary font-medium">385 MB</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Export Data */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export Your Data
          </CardTitle>
          <CardDescription>
            Download a complete copy of all your data in portable formats.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {exportProgress !== null ? (
            <div className="space-y-2">
              <Progress value={exportProgress} />
              <p className="text-xs text-text-tertiary text-center">
                Exporting... {exportProgress}%
              </p>
            </div>
          ) : (
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleExportData}>
                Export as JSON
              </Button>
              <Button variant="outline" onClick={handleExportData}>
                Export as CSV
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Search History */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <History className="h-4 w-4" />
            Search History
          </CardTitle>
          <CardDescription>
            You have {history.length} searches in your history.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            variant="outline"
            onClick={() => setShowClearConfirm(true)}
            disabled={history.length === 0}
          >
            Clear Search History
          </Button>
        </CardContent>
      </Card>

      {/* Danger Zone */}
      <Card className="border-destructive/50">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-4 w-4" />
            Danger Zone
          </CardTitle>
          <CardDescription>
            Irreversible actions that permanently delete your data.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-text-primary text-sm">
                Delete All Data
              </div>
              <div className="text-xs text-text-tertiary">
                Remove all knowledge graph data, embeddings, and logs.
              </div>
            </div>
            <Button variant="destructive" size="sm">
              <Trash2 className="h-4 w-4 mr-1" />
              Delete All
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Clear History Confirmation */}
      <AlertDialog open={showClearConfirm} onOpenChange={setShowClearConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear search history?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete {history.length} search queries from your history.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                clearHistory();
                setShowClearConfirm(false);
              }}
            >
              Clear History
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
```

### Profile Section

```tsx
// src/components/settings/ProfileSection.tsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useUserStore } from '@/stores/userStore';
import { useAuth } from '@/hooks/useAuth';
import { Crown, LogOut, Mail, Calendar } from 'lucide-react';
import { format } from 'date-fns';

export function ProfileSection() {
  const { user, tier, subscriptionEndsAt } = useUserStore();
  const { logout } = useAuth();

  const initials = user?.displayName
    ?.split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase() || user?.email?.[0].toUpperCase() || '?';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Profile</h2>
        <p className="text-sm text-text-secondary mt-1">
          Manage your account and subscription.
        </p>
      </div>

      {/* User Info */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src={user?.photoURL || undefined} />
              <AvatarFallback className="text-lg">{initials}</AvatarFallback>
            </Avatar>
            <div>
              <div className="font-medium text-text-primary text-lg">
                {user?.displayName || 'Futurnal User'}
              </div>
              <div className="flex items-center gap-1 text-sm text-text-secondary">
                <Mail className="h-3 w-3" />
                {user?.email}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Subscription */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Crown className="h-4 w-4" />
            Subscription
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium text-text-primary">
                  {tier === 'pro' ? 'Futurnal Pro' : 'The Archivist'}
                </span>
                <Badge variant={tier === 'pro' ? 'default' : 'secondary'}>
                  {tier === 'pro' ? 'Pro' : 'Free'}
                </Badge>
              </div>
              {tier === 'pro' && subscriptionEndsAt && (
                <div className="flex items-center gap-1 text-xs text-text-tertiary mt-1">
                  <Calendar className="h-3 w-3" />
                  Renews {format(new Date(subscriptionEndsAt), 'MMM d, yyyy')}
                </div>
              )}
              {tier === 'free' && (
                <div className="text-xs text-text-tertiary mt-1">
                  3 data sources included
                </div>
              )}
            </div>
            {tier === 'free' ? (
              <Button>Upgrade to Pro</Button>
            ) : (
              <Button variant="outline">Manage Subscription</Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Sign Out */}
      <Card>
        <CardContent className="pt-6">
          <Button variant="outline" onClick={logout} className="w-full gap-2">
            <LogOut className="h-4 w-4" />
            Sign Out
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
```

### Appearance Section

```tsx
// src/components/settings/AppearanceSection.tsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Switch } from '@/components/ui/switch';
import { useSettingsStore } from '@/stores/settingsStore';
import { Palette, Type, Sparkles } from 'lucide-react';

export function AppearanceSection() {
  const { fontSize, graphAnimations, setSetting } = useSettingsStore();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Appearance</h2>
        <p className="text-sm text-text-secondary mt-1">
          Customize how Futurnal looks and feels.
        </p>
      </div>

      {/* Theme */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Theme
          </CardTitle>
          <CardDescription>
            Futurnal is designed for dark mode. Light mode coming soon.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <div className="flex-1 p-4 rounded-lg bg-background-deep border-2 border-primary">
              <div className="text-sm font-medium text-text-primary mb-1">Dark</div>
              <div className="text-xs text-text-tertiary">Current theme</div>
            </div>
            <div className="flex-1 p-4 rounded-lg bg-gray-100 border border-gray-200 opacity-50 cursor-not-allowed">
              <div className="text-sm font-medium text-gray-800 mb-1">Light</div>
              <div className="text-xs text-gray-500">Coming soon</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Font Size */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Type className="h-4 w-4" />
            Font Size
          </CardTitle>
        </CardHeader>
        <CardContent>
          <RadioGroup
            value={fontSize}
            onValueChange={(v) => setSetting('fontSize', v as any)}
            className="flex gap-4"
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="small" id="small" />
              <Label htmlFor="small" className="text-sm">Small</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="medium" id="medium" />
              <Label htmlFor="medium">Medium</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="large" id="large" />
              <Label htmlFor="large" className="text-lg">Large</Label>
            </div>
          </RadioGroup>
        </CardContent>
      </Card>

      {/* Animations */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            Animations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-text-primary text-sm">
                Graph Animations
              </div>
              <div className="text-xs text-text-tertiary">
                Enable breathing and transition animations in the knowledge graph
              </div>
            </div>
            <Switch
              checked={graphAnimations}
              onCheckedChange={(checked) => setSetting('graphAnimations', checked)}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

## Acceptance Criteria

- [ ] Settings page loads with all sections
- [ ] Privacy toggles persist correctly
- [ ] Consent manager shows all sources
- [ ] Consent grant/revoke works
- [ ] Audit log displays with filtering
- [ ] Audit log export works (CSV)
- [ ] Search history can be cleared
- [ ] Data export initiates correctly
- [ ] Profile shows user info and tier
- [ ] Logout works correctly
- [ ] Appearance settings apply

## Test Plan

### Unit Tests
```typescript
describe('ConsentManager', () => {
  it('should display all configured sources', async () => {
    render(<ConsentManager />);
    await waitFor(() => {
      expect(screen.getByText('My Vault')).toBeInTheDocument();
    });
  });

  it('should toggle consent correctly', async () => {
    render(<ConsentManager />);
    const toggle = screen.getByRole('switch', { name: /read access/i });
    await userEvent.click(toggle);
    expect(mockGrantConsent).toHaveBeenCalled();
  });
});
```

### E2E Tests
```typescript
test('privacy settings flow', async ({ page }) => {
  await page.goto('/settings');
  await page.click('text=Privacy');

  // Toggle telemetry
  await page.click('text=Anonymous Telemetry');

  // View audit log
  await page.click('text=View Audit Log');
  await expect(page.locator('[role="dialog"]')).toBeVisible();

  // Export audit log
  await page.click('[title="Download"]');
});
```

## Dependencies

- @/components/ui/* (shadcn/ui components)
- date-fns
- zustand stores
- React Query hooks

## Next Steps

After privacy panel complete:
1. Add GDPR data deletion workflow
2. Implement data portability export
3. Add privacy policy viewer
4. Create consent expiry reminders

---

## Vision Alignment

This panel embodies the core Futurnal value of **Sovereignty**:

| Principle | Implementation |
|-----------|----------------|
| **Absolute Control** | Every data source requires explicit consent before Ghost can access |
| **Revocable Permission** | Consent can be revoked at any time, immediately |
| **Complete Transparency** | Audit log shows every operation on user data |
| **Data Portability** | Full export in standard formats (JSON, CSV) |
| **Local-First** | All data stored on user's device by default |
| **Explicit Escalation** | Cloud backup requires additional, explicit consent |

### UI Copy Philosophy

All copy in this panel emphasizes:
- User as the **owner** of their data
- Ghost as a **servant** that requires permission
- Consent as **revocable** and **explicit**
- No passive data collection—everything is opt-in

---

**This sovereignty control center ensures users have absolute control over their Ghost's access to their personal universe—because privacy isn't a feature, it's a fundamental right.**

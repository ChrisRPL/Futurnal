/**
 * Data Section
 *
 * Storage usage, data export, and danger zone (deletion).
 * Part of the Settings page - Sovereignty Control Center.
 */

import { useState } from 'react';
import { Download, Trash2, Database, History, AlertTriangle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
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
import { useSearchStore } from '@/stores/searchStore';
import { format } from 'date-fns';

export function DataSection() {
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const { searchHistory, clearSearchHistory } = useSearchStore();

  // Note: Storage info is mock data until backend storage-info command is implemented
  const storageInfo = {
    knowledgeGraph: { bytes: 245000000, label: 'Knowledge Graph' },
    embeddings: { bytes: 128000000, label: 'Vector Embeddings' },
    auditLogs: { bytes: 12000000, label: 'Audit Logs' },
  };

  const totalBytes = Object.values(storageInfo).reduce((acc, item) => acc + item.bytes, 0);
  const maxBytes = 1000000000; // 1GB for progress display

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleExportData = async (fileFormat: 'json' | 'csv') => {
    setExportProgress(0);

    // Simulate export progress
    // In production, this would call the Tauri command and report real progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise((r) => setTimeout(r, 200));
      setExportProgress(i);
    }

    // Create and download a placeholder export
    const exportData = {
      exported_at: new Date().toISOString(),
      format_version: '1.0',
      search_history: searchHistory,
    };

    const content = fileFormat === 'json'
      ? JSON.stringify(exportData, null, 2)
      : 'timestamp,query,resultCount\n' +
        searchHistory.map((h) => `"${h.timestamp}","${h.query}",${h.resultCount}`).join('\n');

    const blob = new Blob([content], {
      type: fileFormat === 'json' ? 'application/json' : 'text/csv',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `futurnal-export-${format(new Date(), 'yyyy-MM-dd')}.${fileFormat}`;
    a.click();
    URL.revokeObjectURL(url);

    setExportProgress(null);
  };

  const handleDeleteAllData = () => {
    // In production, this would call a Tauri command to delete all local data
    console.log('Delete all data requested');
    setShowDeleteConfirm(false);
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">Data Management</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Export, manage, and clear your data.
        </p>
      </div>

      {/* Storage Usage */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Database className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Local Storage</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Your data is stored locally on this device.
        </p>

        <div className="space-y-4">
          {Object.entries(storageInfo).map(([key, { bytes, label }]) => (
            <div key={key}>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-[var(--color-text-secondary)]">{label}</span>
                <span className="text-[var(--color-text-primary)]">{formatBytes(bytes)}</span>
              </div>
              <Progress value={(bytes / maxBytes) * 100} />
            </div>
          ))}

          <div className="pt-2 border-t border-[var(--color-border)]">
            <div className="flex justify-between text-sm">
              <span className="text-[var(--color-text-primary)] font-medium">Total</span>
              <span className="text-[var(--color-text-primary)] font-medium">{formatBytes(totalBytes)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Export Data */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Download className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Export Your Data</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Download a complete copy of all your data in portable formats.
        </p>

        {exportProgress !== null ? (
          <div className="space-y-2">
            <Progress value={exportProgress} />
            <p className="text-xs text-[var(--color-text-tertiary)] text-center flex items-center justify-center gap-2">
              <Loader2 className="h-3 w-3 animate-spin" />
              Exporting... {exportProgress}%
            </p>
          </div>
        ) : (
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => handleExportData('json')}>
              Export as JSON
            </Button>
            <Button variant="outline" onClick={() => handleExportData('csv')}>
              Export as CSV
            </Button>
          </div>
        )}
      </div>

      {/* Search History */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <History className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Search History</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          You have {searchHistory.length} searches in your history.
        </p>
        <Button
          variant="outline"
          onClick={() => setShowClearConfirm(true)}
          disabled={searchHistory.length === 0}
        >
          Clear Search History
        </Button>
      </div>

      {/* Danger Zone */}
      <div className="p-6 border border-red-500/30 bg-red-500/5">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="h-5 w-5 text-red-500" />
          <h3 className="text-base font-medium text-red-500">Danger Zone</h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Irreversible actions that permanently delete your data.
        </p>

        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--color-text-primary)]">Delete All Data</div>
            <div className="text-xs text-[var(--color-text-muted)]">
              Remove all knowledge graph data, embeddings, and logs.
            </div>
          </div>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => setShowDeleteConfirm(true)}
          >
            <Trash2 className="h-4 w-4 mr-1" />
            Delete All
          </Button>
        </div>
      </div>

      {/* Clear History Confirmation */}
      <AlertDialog open={showClearConfirm} onOpenChange={setShowClearConfirm}>
        <AlertDialogContent className="bg-[var(--color-bg-primary)] border-[var(--color-border)]">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-[var(--color-text-primary)]">Clear search history?</AlertDialogTitle>
            <AlertDialogDescription className="text-[var(--color-text-secondary)]">
              This will permanently delete {searchHistory.length} search queries from your history.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                clearSearchHistory();
                setShowClearConfirm(false);
              }}
            >
              Clear History
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete All Confirmation */}
      <AlertDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <AlertDialogContent className="bg-[var(--color-bg-primary)] border-[var(--color-border)]">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-[var(--color-text-primary)]">Delete all data?</AlertDialogTitle>
            <AlertDialogDescription className="text-[var(--color-text-secondary)]">
              This will permanently delete all your knowledge graph data, vector embeddings,
              and audit logs. This action cannot be undone and you will need to re-sync
              all your data sources.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteAllData}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete Everything
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

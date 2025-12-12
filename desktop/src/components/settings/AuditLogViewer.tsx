/**
 * Audit Log Viewer
 *
 * Displays activity audit logs with filtering and export capabilities.
 * Part of the Privacy Section - Sovereignty Control Center.
 */

import { useState, useMemo } from 'react';
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
import { Search, Download, RefreshCw, Loader2 } from 'lucide-react';
import { format } from 'date-fns';

interface AuditLogViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ACTION_VARIANTS: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
  search: 'secondary',
  read: 'secondary',
  write: 'default',
  delete: 'destructive',
  consent_grant: 'default',
  consent_revoke: 'destructive',
  sync: 'secondary',
  export: 'outline',
};

export function AuditLogViewer({ open, onOpenChange }: AuditLogViewerProps) {
  const [filter, setFilter] = useState('');
  const [actionFilter, setActionFilter] = useState<string>('all');
  const [limit, setLimit] = useState(50);

  const { data: logs, isLoading, refetch } = useAuditLogs({
    limit,
    action_filter: actionFilter !== 'all' ? actionFilter : undefined,
  });

  const filteredLogs = useMemo(() => {
    if (!logs) return [];
    if (!filter) return logs;

    const lowerFilter = filter.toLowerCase();
    return logs.filter(
      (log) =>
        log.action.toLowerCase().includes(lowerFilter) ||
        log.resource_type.toLowerCase().includes(lowerFilter) ||
        (log.resource_id?.toLowerCase().includes(lowerFilter) ?? false)
    );
  }, [logs, filter]);

  const handleExport = () => {
    if (!logs?.length) return;

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
      .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
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
      <DialogContent className="max-w-3xl max-h-[80vh] bg-black border-white/20">
        <DialogHeader>
          <DialogTitle className="text-white">Activity Audit Log</DialogTitle>
        </DialogHeader>

        {/* Filters */}
        <div className="flex items-center gap-3 py-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/40" />
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
              <SelectItem value="all">All actions</SelectItem>
              <SelectItem value="search">Search</SelectItem>
              <SelectItem value="read">Read</SelectItem>
              <SelectItem value="write">Write</SelectItem>
              <SelectItem value="sync">Sync</SelectItem>
              <SelectItem value="consent">Consent</SelectItem>
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={handleExport}
            disabled={!logs?.length}
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>

        {/* Log List */}
        <ScrollArea className="h-[500px] pr-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8 text-white/50">
              <Loader2 className="h-5 w-5 animate-spin mr-2" />
              Loading logs...
            </div>
          ) : filteredLogs.length === 0 ? (
            <div className="text-center py-8 text-white/50">
              No logs found
            </div>
          ) : (
            <div className="space-y-2">
              {filteredLogs.map((log) => (
                <div
                  key={log.id}
                  className="flex items-start gap-4 p-3 bg-white/5"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <Badge variant={ACTION_VARIANTS[log.action] || 'secondary'}>
                        {log.action}
                      </Badge>
                      <span className="text-sm text-white/60">
                        {log.resource_type}
                      </span>
                    </div>
                    {log.resource_id && (
                      <div className="text-xs text-white/40 mt-1 truncate">
                        {log.resource_id}
                      </div>
                    )}
                    {log.details && Object.keys(log.details).length > 0 && (
                      <div className="text-xs text-white/30 mt-1 font-mono">
                        {JSON.stringify(log.details)}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-white/40 whitespace-nowrap">
                    {formatTimestamp(log.timestamp)}
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

function formatTimestamp(timestamp: string): string {
  try {
    return format(new Date(timestamp), 'MMM d, HH:mm:ss');
  } catch {
    return timestamp;
  }
}

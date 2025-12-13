/**
 * Audit Log Viewer
 *
 * Displays activity audit logs with filtering and export capabilities.
 * Part of the Privacy Section - Sovereignty Control Center.
 *
 * Enhanced features:
 * - JSON and CSV export
 * - Date range filtering
 * - Job ID correlation filter
 * - Source filter dropdown
 * - Tamper verification status
 * - Improved pagination with total count
 */

import { useState, useMemo, useEffect } from 'react';
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
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Calendar } from '@/components/ui/calendar';
import { useAuditLogs } from '@/hooks/useApi';
import {
  Search,
  Download,
  RefreshCw,
  Loader2,
  Calendar as CalendarIcon,
  ShieldCheck,
  ShieldAlert,
  Link,
  ChevronLeft,
  ChevronRight,
  FileJson,
  FileSpreadsheet,
} from 'lucide-react';
import { format, subDays, startOfDay, endOfDay, isWithinInterval } from 'date-fns';
import type { DateRange } from 'react-day-picker';

interface AuditLogViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface AuditLog {
  id: string;
  timestamp: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  job_id?: string;
  source?: string;
  status?: string;
  details?: Record<string, unknown>;
  chain_hash?: string;
  chain_prev?: string;
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
  purge: 'destructive',
  anomaly: 'destructive',
};

const PAGE_SIZE = 50;

export function AuditLogViewer({ open, onOpenChange }: AuditLogViewerProps) {
  const [searchFilter, setSearchFilter] = useState('');
  const [actionFilter, setActionFilter] = useState<string>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [jobIdFilter, setJobIdFilter] = useState<string>('');
  const [dateRange, setDateRange] = useState<DateRange | undefined>(undefined);
  const [page, setPage] = useState(0);
  const [verificationStatus, setVerificationStatus] = useState<'idle' | 'verifying' | 'valid' | 'invalid'>('idle');

  const { data: logs, isLoading, refetch, totalCount } = useAuditLogs({
    limit: 1000, // Load more for client-side filtering
    action_filter: actionFilter !== 'all' ? actionFilter : undefined,
  });

  // Extract unique sources from logs
  const uniqueSources = useMemo(() => {
    if (!logs) return [];
    const sources = new Set<string>();
    logs.forEach((log) => {
      if (log.source) sources.add(log.source);
    });
    return Array.from(sources).sort();
  }, [logs]);

  // Extract unique job IDs from logs
  const uniqueJobIds = useMemo(() => {
    if (!logs) return [];
    const jobIds = new Set<string>();
    logs.forEach((log) => {
      if (log.job_id) jobIds.add(log.job_id);
    });
    return Array.from(jobIds).slice(0, 50); // Limit to 50 most recent
  }, [logs]);

  // Filtered logs
  const filteredLogs = useMemo(() => {
    if (!logs) return [];

    let result = [...logs];

    // Text search filter
    if (searchFilter) {
      const lowerFilter = searchFilter.toLowerCase();
      result = result.filter(
        (log) =>
          log.action.toLowerCase().includes(lowerFilter) ||
          log.resource_type.toLowerCase().includes(lowerFilter) ||
          (log.resource_id?.toLowerCase().includes(lowerFilter) ?? false) ||
          (log.job_id?.toLowerCase().includes(lowerFilter) ?? false) ||
          (log.source?.toLowerCase().includes(lowerFilter) ?? false)
      );
    }

    // Source filter
    if (sourceFilter !== 'all') {
      result = result.filter((log) => log.source === sourceFilter);
    }

    // Job ID filter
    if (jobIdFilter) {
      result = result.filter((log) => log.job_id === jobIdFilter);
    }

    // Date range filter
    if (dateRange?.from) {
      const start = startOfDay(dateRange.from);
      const end = dateRange.to ? endOfDay(dateRange.to) : endOfDay(dateRange.from);
      result = result.filter((log) => {
        const logDate = new Date(log.timestamp);
        return isWithinInterval(logDate, { start, end });
      });
    }

    return result;
  }, [logs, searchFilter, sourceFilter, jobIdFilter, dateRange]);

  // Paginated logs
  const paginatedLogs = useMemo(() => {
    const start = page * PAGE_SIZE;
    return filteredLogs.slice(start, start + PAGE_SIZE);
  }, [filteredLogs, page]);

  const totalPages = Math.ceil(filteredLogs.length / PAGE_SIZE);

  // Reset page when filters change
  useEffect(() => {
    setPage(0);
  }, [searchFilter, actionFilter, sourceFilter, jobIdFilter, dateRange]);

  // Verify chain integrity
  const handleVerifyIntegrity = async () => {
    setVerificationStatus('verifying');
    try {
      const result = await window.electron?.invoke('audit:verifyIntegrity');
      setVerificationStatus(result ? 'valid' : 'invalid');
    } catch {
      // Fallback: verify locally if IPC not available
      // Check that chain_prev matches previous chain_hash
      let valid = true;
      let prevHash: string | null = null;

      for (const log of logs || []) {
        if (log.chain_prev !== prevHash) {
          valid = false;
          break;
        }
        prevHash = log.chain_hash || null;
      }

      setVerificationStatus(valid ? 'valid' : 'invalid');
    }

    // Reset after 5 seconds
    setTimeout(() => setVerificationStatus('idle'), 5000);
  };

  // Export as CSV
  const handleExportCSV = () => {
    if (!filteredLogs.length) return;

    const csv = [
      ['Timestamp', 'Action', 'Source', 'Job ID', 'Resource Type', 'Resource ID', 'Status', 'Details'],
      ...filteredLogs.map((log) => [
        log.timestamp,
        log.action,
        log.source || '',
        log.job_id || '',
        log.resource_type,
        log.resource_id || '',
        log.status || '',
        JSON.stringify(log.details || {}),
      ]),
    ]
      .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
      .join('\n');

    downloadFile(csv, 'text/csv', `futurnal-audit-log-${format(new Date(), 'yyyy-MM-dd')}.csv`);
  };

  // Export as JSON
  const handleExportJSON = () => {
    if (!filteredLogs.length) return;

    const json = JSON.stringify(
      {
        exported_at: new Date().toISOString(),
        filter: {
          action: actionFilter !== 'all' ? actionFilter : null,
          source: sourceFilter !== 'all' ? sourceFilter : null,
          job_id: jobIdFilter || null,
          date_range: dateRange
            ? {
                from: dateRange.from?.toISOString(),
                to: dateRange.to?.toISOString(),
              }
            : null,
        },
        total_count: filteredLogs.length,
        logs: filteredLogs,
      },
      null,
      2
    );

    downloadFile(json, 'application/json', `futurnal-audit-log-${format(new Date(), 'yyyy-MM-dd')}.json`);
  };

  // Quick date range presets
  const handleDatePreset = (preset: 'today' | '7days' | '30days' | 'all') => {
    if (preset === 'all') {
      setDateRange(undefined);
    } else if (preset === 'today') {
      const today = new Date();
      setDateRange({ from: today, to: today });
    } else if (preset === '7days') {
      setDateRange({ from: subDays(new Date(), 7), to: new Date() });
    } else if (preset === '30days') {
      setDateRange({ from: subDays(new Date(), 30), to: new Date() });
    }
  };

  // Filter by job ID when clicking on a log entry
  const handleFilterByJobId = (jobId: string) => {
    setJobIdFilter(jobId);
  };

  // Clear job ID filter
  const handleClearJobIdFilter = () => {
    setJobIdFilter('');
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] bg-[var(--color-bg-primary)] border-[var(--color-border)]">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="text-[var(--color-text-primary)]">Activity Audit Log</DialogTitle>
            <div className="flex items-center gap-2">
              {/* Verification Status */}
              <Button
                variant="ghost"
                size="sm"
                onClick={handleVerifyIntegrity}
                disabled={verificationStatus === 'verifying' || !logs?.length}
                className="h-8"
              >
                {verificationStatus === 'verifying' ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                ) : verificationStatus === 'valid' ? (
                  <ShieldCheck className="h-4 w-4 mr-1 text-green-500" />
                ) : verificationStatus === 'invalid' ? (
                  <ShieldAlert className="h-4 w-4 mr-1 text-red-500" />
                ) : (
                  <ShieldCheck className="h-4 w-4 mr-1" />
                )}
                {verificationStatus === 'verifying'
                  ? 'Verifying...'
                  : verificationStatus === 'valid'
                  ? 'Chain Valid'
                  : verificationStatus === 'invalid'
                  ? 'Chain Invalid!'
                  : 'Verify Integrity'}
              </Button>
            </div>
          </div>
        </DialogHeader>

        {/* Filters Row 1 */}
        <div className="flex items-center gap-2 py-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--color-text-muted)]" />
            <Input
              placeholder="Search logs..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Action Filter */}
          <Select value={actionFilter} onValueChange={setActionFilter}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Action" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All actions</SelectItem>
              <SelectItem value="search">Search</SelectItem>
              <SelectItem value="read">Read</SelectItem>
              <SelectItem value="write">Write</SelectItem>
              <SelectItem value="sync">Sync</SelectItem>
              <SelectItem value="consent">Consent</SelectItem>
              <SelectItem value="purge">Purge</SelectItem>
              <SelectItem value="anomaly">Anomaly</SelectItem>
            </SelectContent>
          </Select>

          {/* Source Filter */}
          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Source" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sources</SelectItem>
              {uniqueSources.map((source) => (
                <SelectItem key={source} value={source}>
                  {source}
                </SelectItem>
              ))}
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
        </div>

        {/* Filters Row 2 */}
        <div className="flex items-center gap-2 pb-2">
          {/* Date Range Picker */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="w-56 justify-start text-left font-normal">
                <CalendarIcon className="mr-2 h-4 w-4" />
                {dateRange?.from ? (
                  dateRange.to ? (
                    <>
                      {format(dateRange.from, 'MMM d')} - {format(dateRange.to, 'MMM d')}
                    </>
                  ) : (
                    format(dateRange.from, 'MMM d, yyyy')
                  )
                ) : (
                  <span className="text-[var(--color-text-muted)]">All dates</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <div className="flex gap-1 p-2 border-b">
                <Button size="sm" variant="ghost" onClick={() => handleDatePreset('today')}>
                  Today
                </Button>
                <Button size="sm" variant="ghost" onClick={() => handleDatePreset('7days')}>
                  7 days
                </Button>
                <Button size="sm" variant="ghost" onClick={() => handleDatePreset('30days')}>
                  30 days
                </Button>
                <Button size="sm" variant="ghost" onClick={() => handleDatePreset('all')}>
                  All
                </Button>
              </div>
              <Calendar
                mode="range"
                selected={dateRange}
                onSelect={setDateRange}
                numberOfMonths={1}
              />
            </PopoverContent>
          </Popover>

          {/* Job ID Filter */}
          {jobIdFilter ? (
            <div className="flex items-center gap-1 px-3 py-1.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md">
              <Link className="h-3.5 w-3.5 text-[var(--color-text-muted)]" />
              <span className="text-sm font-mono truncate max-w-32">{jobIdFilter.slice(0, 8)}...</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-5 w-5 p-0 ml-1"
                onClick={handleClearJobIdFilter}
              >
                ×
              </Button>
            </div>
          ) : (
            uniqueJobIds.length > 0 && (
              <Select value="" onValueChange={handleFilterByJobId}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Filter by Job ID" />
                </SelectTrigger>
                <SelectContent>
                  {uniqueJobIds.map((jobId) => (
                    <SelectItem key={jobId} value={jobId}>
                      <span className="font-mono text-xs">{jobId.slice(0, 12)}...</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )
          )}

          <div className="flex-1" />

          {/* Export Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" disabled={!filteredLogs.length}>
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={handleExportCSV}>
                <FileSpreadsheet className="h-4 w-4 mr-2" />
                Export as CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportJSON}>
                <FileJson className="h-4 w-4 mr-2" />
                Export as JSON
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Results Summary */}
        <div className="flex items-center justify-between text-xs text-[var(--color-text-muted)] pb-2">
          <span>
            Showing {paginatedLogs.length} of {filteredLogs.length} logs
            {totalCount && totalCount > filteredLogs.length && ` (${totalCount} total)`}
          </span>
          {totalPages > 1 && (
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span>
                Page {page + 1} of {totalPages}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {/* Log List */}
        <ScrollArea className="h-[400px] pr-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8 text-[var(--color-text-tertiary)]">
              <Loader2 className="h-5 w-5 animate-spin mr-2" />
              Loading logs...
            </div>
          ) : paginatedLogs.length === 0 ? (
            <div className="text-center py-8 text-[var(--color-text-tertiary)]">
              No logs found matching your filters
            </div>
          ) : (
            <div className="space-y-2">
              {paginatedLogs.map((log) => (
                <div
                  key={log.id}
                  className="flex items-start gap-4 p-3 bg-[var(--color-surface)] rounded-md hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge variant={ACTION_VARIANTS[log.action] || 'secondary'}>
                        {log.action}
                      </Badge>
                      <span className="text-sm text-[var(--color-text-secondary)]">
                        {log.resource_type}
                      </span>
                      {log.source && (
                        <Badge variant="outline" className="text-xs">
                          {log.source}
                        </Badge>
                      )}
                      {log.status && log.status !== 'success' && (
                        <Badge
                          variant={log.status === 'failed' ? 'destructive' : 'secondary'}
                          className="text-xs"
                        >
                          {log.status}
                        </Badge>
                      )}
                    </div>
                    {log.resource_id && (
                      <div className="text-xs text-[var(--color-text-muted)] mt-1 truncate font-mono">
                        {log.resource_id}
                      </div>
                    )}
                    {log.job_id && (
                      <button
                        className="text-xs text-blue-400 hover:text-blue-300 mt-1 flex items-center gap-1"
                        onClick={() => handleFilterByJobId(log.job_id!)}
                      >
                        <Link className="h-3 w-3" />
                        <span className="font-mono">{log.job_id.slice(0, 12)}...</span>
                      </button>
                    )}
                    {log.details && Object.keys(log.details).length > 0 && (
                      <details className="mt-2">
                        <summary className="text-xs text-[var(--color-text-muted)] cursor-pointer hover:text-[var(--color-text-secondary)]">
                          Details
                        </summary>
                        <pre className="text-xs text-[var(--color-text-muted)] mt-1 p-2 bg-[var(--color-bg-primary)] rounded overflow-x-auto">
                          {JSON.stringify(log.details, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                  <div className="text-xs text-[var(--color-text-muted)] whitespace-nowrap">
                    {formatTimestamp(log.timestamp)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
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

function downloadFile(content: string, mimeType: string, filename: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

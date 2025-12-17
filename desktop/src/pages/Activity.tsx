/**
 * Activity Page - Full activity stream with filters
 *
 * Step 08: Frontend Intelligence Integration - Phase 3
 *
 * Research Foundation:
 * - AgentFlow: Activity tracking patterns
 * - RLHI: User interaction history
 *
 * Features:
 * - Full history with infinite scroll
 * - Filter by event type
 * - Date range picker
 * - Search within activity
 * - Export activity log
 */

import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Clock,
  Filter,
  Download,
  Loader2,
  RefreshCw,
  X,
  Search,
  FileText,
  MessageSquare,
  Lightbulb,
  Database,
  Circle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  useActivityStore,
  groupActivitiesByTime,
  type ActivityEventType,
} from '@/stores/activityStore';
import { ActivityItem } from '@/components/activity';

/** Filter chip component */
function FilterChip({
  label,
  icon: Icon,
  active,
  onClick,
}: {
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-1.5 px-3 py-1.5 text-xs transition-colors',
        active
          ? 'bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)]'
          : 'bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)]'
      )}
    >
      {Icon && <Icon className="w-3 h-3" />}
      {label}
    </button>
  );
}

/** Event type filter options */
const EVENT_TYPES: { type: ActivityEventType; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { type: 'search', label: 'Search', icon: Search },
  { type: 'document', label: 'Document', icon: FileText },
  { type: 'chat', label: 'Chat', icon: MessageSquare },
  { type: 'insight', label: 'Insight', icon: Lightbulb },
  { type: 'schema', label: 'Schema', icon: Database },
  { type: 'entity', label: 'Entity', icon: Circle },
];

export default function ActivityPage() {
  const navigate = useNavigate();
  const [showFilters, setShowFilters] = useState(false);

  const {
    events,
    filters,
    isLoading,
    hasMore,
    total,
    error,
    fetchActivities,
    loadMore,
    setTypeFilter,
    clearFilters,
    clearError,
  } = useActivityStore();

  // Fetch on mount
  useEffect(() => {
    fetchActivities(true);
  }, [fetchActivities]);

  // Refetch when filters change
  useEffect(() => {
    fetchActivities(true);
  }, [filters.types, filters.dateRange, fetchActivities]);

  // Toggle type filter
  const toggleTypeFilter = useCallback(
    (type: ActivityEventType) => {
      const currentTypes = filters.types;
      if (currentTypes.includes(type)) {
        setTypeFilter(currentTypes.filter((t) => t !== type));
      } else {
        setTypeFilter([...currentTypes, type]);
      }
    },
    [filters.types, setTypeFilter]
  );

  // Group events by time
  const groupedEvents = groupActivitiesByTime(events);

  // Handler for entity click
  const handleEntityClick = (entityId: string) => {
    navigate(`/graph?select=${encodeURIComponent(entityId)}`);
  };

  // Handler for causal exploration
  const handleExploreCauses = (eventId: string) => {
    navigate(`/graph?causal=${encodeURIComponent(eventId)}`);
  };

  // Export activity log
  const handleExport = useCallback(() => {
    const data = JSON.stringify(events, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `futurnal-activity-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [events]);

  // Scroll handler for infinite scroll
  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
      if (scrollHeight - scrollTop <= clientHeight * 1.5 && hasMore && !isLoading) {
        loadMore();
      }
    },
    [hasMore, isLoading, loadMore]
  );

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-[var(--color-border)]">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="p-2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-hover)] transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-lg font-semibold text-[var(--color-text-primary)] flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Activity
              </h1>
              <p className="text-xs text-[var(--color-text-muted)]">
                {total} events
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => fetchActivities(true)}
              disabled={isLoading}
              className="p-2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw className={cn('h-4 w-4', isLoading && 'animate-spin')} />
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={cn(
                'p-2 transition-colors',
                showFilters
                  ? 'text-[var(--color-text-primary)] bg-[var(--color-surface)]'
                  : 'text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)]'
              )}
              title="Filters"
            >
              <Filter className="h-4 w-4" />
            </button>
            <button
              onClick={handleExport}
              disabled={events.length === 0}
              className="p-2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors disabled:opacity-50"
              title="Export"
            >
              <Download className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Filters panel */}
        {showFilters && (
          <div className="px-6 py-3 border-t border-[var(--color-border)] bg-[var(--color-surface)]">
            <div className="flex items-center gap-4">
              <span className="text-xs text-[var(--color-text-muted)]">Type:</span>
              <div className="flex flex-wrap gap-2">
                {EVENT_TYPES.map(({ type, label, icon }) => (
                  <FilterChip
                    key={type}
                    label={label}
                    icon={icon}
                    active={filters.types.includes(type)}
                    onClick={() => toggleTypeFilter(type)}
                  />
                ))}
              </div>
              {filters.types.length > 0 && (
                <button
                  onClick={clearFilters}
                  className="flex items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
                >
                  <X className="w-3 h-3" />
                  Clear
                </button>
              )}
            </div>
          </div>
        )}
      </header>

      {/* Error */}
      {error && (
        <div className="mx-6 mt-4 px-4 py-3 bg-red-500/10 border border-red-500/20 text-sm text-red-400 flex items-center justify-between">
          <span>{error}</span>
          <button onClick={clearError} className="text-red-300 hover:text-red-200">
            Dismiss
          </button>
        </div>
      )}

      {/* Content */}
      <main
        className="flex-1 overflow-y-auto"
        onScroll={handleScroll}
      >
        {/* Loading initial */}
        {isLoading && events.length === 0 && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-8 h-8 text-[var(--color-text-muted)] animate-spin" />
          </div>
        )}

        {/* Empty state */}
        {!isLoading && events.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16">
            <Clock className="w-12 h-12 text-[var(--color-text-faint)] mb-4" />
            <p className="text-[var(--color-text-secondary)] mb-2">No activity found</p>
            <p className="text-sm text-[var(--color-text-muted)]">
              {filters.types.length > 0
                ? 'Try adjusting your filters'
                : 'Activities will appear as you use Futurnal'}
            </p>
          </div>
        )}

        {/* Activity list grouped by time */}
        <div className="max-w-4xl mx-auto">
          {Array.from(groupedEvents.entries()).map(([groupName, groupEvents]) => (
            <div key={groupName} className="mb-4">
              {/* Group header */}
              <div className="sticky top-0 z-10 px-6 py-2 bg-[var(--color-bg-primary)] border-b border-[var(--color-border)]">
                <span className="text-sm font-medium text-[var(--color-text-secondary)]">
                  {groupName}
                </span>
                <span className="ml-2 text-xs text-[var(--color-text-muted)]">
                  {groupEvents.length} events
                </span>
              </div>

              {/* Events */}
              <div className="px-3">
                {groupEvents.map((event) => (
                  <ActivityItem
                    key={event.id}
                    event={event}
                    onEntityClick={handleEntityClick}
                    onExploreCauses={handleExploreCauses}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Loading more */}
        {isLoading && events.length > 0 && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="w-5 h-5 text-[var(--color-text-muted)] animate-spin" />
          </div>
        )}

        {/* End of list */}
        {!hasMore && events.length > 0 && (
          <div className="text-center py-6 text-sm text-[var(--color-text-faint)]">
            End of activity log
          </div>
        )}
      </main>
    </div>
  );
}

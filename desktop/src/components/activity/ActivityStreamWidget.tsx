/**
 * ActivityStreamWidget - Compact activity stream for dashboard
 *
 * Step 08: Frontend Intelligence Integration - Phase 3
 *
 * Research Foundation:
 * - AgentFlow: Activity tracking patterns
 * - RLHI: User interaction history
 *
 * Features:
 * - Last 2 days grouped with counts
 * - Compact event display
 * - "Open full view →" link to /activity page
 */

import { useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock, Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useActivityStore, groupActivitiesByTime } from '@/stores/activityStore';
import { ActivityItem } from './ActivityItem';

interface ActivityStreamWidgetProps {
  /** Maximum events to show */
  maxEvents?: number;
  /** Additional class names */
  className?: string;
}

export function ActivityStreamWidget({
  maxEvents = 10,
  className,
}: ActivityStreamWidgetProps) {
  const navigate = useNavigate();
  const {
    events,
    isLoading,
    error,
    fetchActivities,
    clearError,
  } = useActivityStore();

  // Fetch activities on mount
  useEffect(() => {
    fetchActivities(true);
  }, [fetchActivities]);

  // Group events by time
  const groupedEvents = useMemo(() => {
    const limited = events.slice(0, maxEvents);
    return groupActivitiesByTime(limited);
  }, [events, maxEvents]);

  // Handler for entity click
  const handleEntityClick = (entityId: string) => {
    navigate(`/graph?select=${encodeURIComponent(entityId)}`);
  };

  // Handler for causal exploration
  const handleExploreCauses = (eventId: string) => {
    navigate(`/graph?causal=${encodeURIComponent(eventId)}`);
  };

  // Get total count per group
  const getGroupSummary = (groupName: string, groupEvents: typeof events) => {
    const counts = new Map<string, number>();
    groupEvents.forEach((e) => {
      counts.set(e.type, (counts.get(e.type) || 0) + 1);
    });

    const parts: string[] = [];
    counts.forEach((count, type) => {
      parts.push(`${count} ${type}`);
    });

    return parts.join(', ');
  };

  if (error) {
    return (
      <div className={cn('p-4', className)}>
        <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">{error}</span>
          <button
            onClick={clearError}
            className="text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('', className)}>
      {/* Loading */}
      {isLoading && events.length === 0 && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-5 h-5 text-[var(--color-text-muted)] animate-spin" />
        </div>
      )}

      {/* Empty state */}
      {!isLoading && events.length === 0 && (
        <div className="py-6 text-center">
          <Clock className="w-8 h-8 text-[var(--color-text-faint)] mx-auto mb-2" />
          <p className="text-sm text-[var(--color-text-muted)]">No recent activity</p>
          <p className="text-xs text-[var(--color-text-faint)] mt-1">
            Activities will appear as you use Futurnal
          </p>
        </div>
      )}

      {/* Activity groups */}
      {Array.from(groupedEvents.entries()).map(([groupName, groupEvents]) => (
        <div key={groupName}>
          {/* Group header */}
          <div className="flex items-center justify-between px-3 py-2 bg-[var(--color-surface)] border-b border-[var(--color-border)]">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">
              {groupName}
            </span>
            <span className="text-[10px] text-[var(--color-text-muted)]">
              {getGroupSummary(groupName, groupEvents)}
            </span>
          </div>

          {/* Group events */}
          <div>
            {groupEvents.map((event) => (
              <ActivityItem
                key={event.id}
                event={event}
                compact
                onEntityClick={handleEntityClick}
                onExploreCauses={handleExploreCauses}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Refresh indicator */}
      {isLoading && events.length > 0 && (
        <div className="flex items-center justify-center py-2 border-t border-[var(--color-border)]">
          <RefreshCw className="w-3 h-3 text-[var(--color-text-muted)] animate-spin" />
        </div>
      )}
    </div>
  );
}

export default ActivityStreamWidget;

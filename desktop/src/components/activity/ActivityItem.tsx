/**
 * ActivityItem - Individual activity event display
 *
 * Step 08: Frontend Intelligence Integration - Phase 3
 *
 * Research Foundation:
 * - AgentFlow: Activity tracking patterns
 * - RLHI: User interaction history
 *
 * Features:
 * - Icon by type
 * - Relative timestamp
 * - Related entity links
 * - "What led to this?" causal button
 */

import { useMemo } from 'react';
import {
  Search,
  FileText,
  MessageSquare,
  Lightbulb,
  Database,
  Circle,
  Download,
  Activity,
  ExternalLink,
  GitBranch,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { type ActivityEvent, type ActivityEventType, getCategoryColor } from '@/stores/activityStore';

interface ActivityItemProps {
  event: ActivityEvent;
  onEntityClick?: (entityId: string) => void;
  onExploreCauses?: (eventId: string) => void;
  compact?: boolean;
  className?: string;
}

/** Get icon component for activity type */
function getActivityIcon(type: ActivityEventType): React.ComponentType<{ className?: string }> {
  switch (type) {
    case 'search':
      return Search;
    case 'document':
      return FileText;
    case 'chat':
      return MessageSquare;
    case 'insight':
      return Lightbulb;
    case 'schema':
      return Database;
    case 'entity':
      return Circle;
    case 'ingestion':
      return Download;
    default:
      return Activity;
  }
}

/** Format relative time */
function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const date = new Date(timestamp);
  const diff = now.getTime() - date.getTime();

  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;

  return date.toLocaleDateString();
}

export function ActivityItem({
  event,
  onEntityClick,
  onExploreCauses,
  compact = false,
  className,
}: ActivityItemProps) {
  const Icon = useMemo(() => getActivityIcon(event.type), [event.type]);
  const categoryColor = useMemo(() => getCategoryColor(event.category), [event.category]);
  const relativeTime = useMemo(() => formatRelativeTime(event.timestamp), [event.timestamp]);

  if (compact) {
    return (
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5',
          'hover:bg-[var(--color-surface)] transition-colors cursor-pointer',
          className
        )}
        onClick={() => event.relatedEntityIds[0] && onEntityClick?.(event.relatedEntityIds[0])}
      >
        <Icon className={cn('w-3.5 h-3.5 flex-shrink-0', categoryColor)} />
        <span className="text-xs text-[var(--color-text-secondary)] truncate flex-1">
          {event.title}
        </span>
        <span className="text-[10px] text-[var(--color-text-muted)] flex-shrink-0">
          {relativeTime}
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'group px-3 py-2.5 border-b border-[var(--color-border)]',
        'hover:bg-[var(--color-surface)] transition-colors',
        className
      )}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={cn('p-1.5 bg-[var(--color-surface)]', categoryColor)}>
          <Icon className="w-4 h-4" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-wide">
              {event.type}
            </span>
            <span className="text-[10px] text-[var(--color-text-faint)]">
              {relativeTime}
            </span>
          </div>

          {/* Title */}
          <p className="text-sm text-[var(--color-text-primary)] truncate">
            {event.title}
          </p>

          {/* Description */}
          {event.description && (
            <p className="text-xs text-[var(--color-text-tertiary)] mt-0.5 line-clamp-2">
              {event.description}
            </p>
          )}

          {/* Related entities */}
          {event.relatedEntityIds.length > 0 && (
            <div className="flex items-center gap-2 mt-1.5">
              {event.relatedEntityIds.slice(0, 3).map((entityId) => (
                <button
                  key={entityId}
                  onClick={() => onEntityClick?.(entityId)}
                  className="flex items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
                >
                  <ExternalLink className="w-3 h-3" />
                  <span className="truncate max-w-[100px]">{entityId}</span>
                </button>
              ))}
              {event.relatedEntityIds.length > 3 && (
                <span className="text-xs text-[var(--color-text-faint)]">
                  +{event.relatedEntityIds.length - 3} more
                </span>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {onExploreCauses && (
            <button
              onClick={() => onExploreCauses(event.id)}
              className="p-1 text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors"
              title="What led to this?"
            >
              <GitBranch className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default ActivityItem;

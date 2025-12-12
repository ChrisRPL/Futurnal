/**
 * RecentSearches - Display recent search queries with metadata
 *
 * Shows a list of recently executed queries with timestamps and result counts.
 * Uses monochrome styling with click-to-select functionality.
 */

import { Clock } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { SearchHistoryItem } from '@/stores/searchStore';

interface RecentSearchesProps {
  /** List of recent search history items with metadata */
  searches: SearchHistoryItem[];
  /** Handler when a recent search is selected */
  onSelect: (query: string) => void;
  /** Currently selected index for keyboard navigation */
  selectedIndex?: number;
  /** Additional class names */
  className?: string;
}

/**
 * Format a timestamp into a relative time string
 */
function formatRelativeTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export function RecentSearches({
  searches,
  onSelect,
  selectedIndex = -1,
  className,
}: RecentSearchesProps) {
  if (searches.length === 0) {
    return null;
  }

  return (
    <div data-slot="recent-searches" className={cn('py-2', className)}>
      <div className="px-3 py-2 text-xs text-white/40 font-medium">
        Recent Searches
      </div>
      <div className="space-y-0.5">
        {searches.map((item, index) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.query)}
            data-selected={index === selectedIndex}
            className={cn(
              'w-full flex items-center gap-3 px-3 py-2 text-left',
              'text-white/70 hover:text-white',
              'hover:bg-white/5 transition-colors',
              index === selectedIndex && 'bg-white/10 text-white border-l-2 border-white/50'
            )}
          >
            <Clock className="h-4 w-4 text-white/30 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <span className="text-sm truncate block">{item.query}</span>
              <span className="text-xs text-white/30">
                {item.resultCount} results · {formatRelativeTime(item.timestamp)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

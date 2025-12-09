/**
 * RecentSearches - Display recent search queries
 *
 * Shows a list of recently executed queries with click-to-select
 * functionality. Uses monochrome styling.
 */

import { Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

interface RecentSearchesProps {
  /** List of recent search queries */
  searches: string[];
  /** Handler when a recent search is selected */
  onSelect: (query: string) => void;
  /** Currently selected index for keyboard navigation */
  selectedIndex?: number;
  /** Additional class names */
  className?: string;
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
        {searches.map((query, index) => (
          <button
            key={`${query}-${index}`}
            onClick={() => onSelect(query)}
            data-selected={index === selectedIndex}
            className={cn(
              'w-full flex items-center gap-3 px-3 py-2 text-left',
              'text-white/70 hover:text-white',
              'hover:bg-white/5 transition-colors',
              index === selectedIndex && 'bg-white/10 text-white border-l-2 border-white/50'
            )}
          >
            <Clock className="h-4 w-4 text-white/30 flex-shrink-0" />
            <span className="text-sm truncate">{query}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

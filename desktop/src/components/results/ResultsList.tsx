/**
 * ResultsList - Container component for search results
 *
 * Displays a list of ResultCard components with:
 * - Result count header
 * - Selection state management
 * - Keyboard navigation support
 */

import { cn } from '@/lib/utils';
import { ResultCard } from './ResultCard';
import type { SearchResult } from '@/types/api';

interface ResultsListProps {
  /** Array of search results to display */
  results: SearchResult[];
  /** Search query for term highlighting */
  query: string;
  /** Currently selected result ID */
  selectedId?: string | null;
  /** Handler for result selection */
  onSelect?: (result: SearchResult) => void;
  /** Currently selected index for keyboard navigation */
  selectedIndex?: number;
  /** Additional class names */
  className?: string;
}

export function ResultsList({
  results,
  query,
  selectedId,
  onSelect,
  selectedIndex = -1,
  className,
}: ResultsListProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <div
      data-slot="results-list"
      data-testid="results-list"
      className={cn('py-2', className)}
    >
      {/* Results count header */}
      <div className="px-3 py-2 text-xs text-white/40 font-medium">
        {results.length} result{results.length !== 1 ? 's' : ''}
      </div>

      {/* Results */}
      <div className="space-y-1">
        {results.map((result, index) => (
          <ResultCard
            key={result.id}
            result={result}
            query={query}
            isSelected={selectedId === result.id || index === selectedIndex}
            onSelect={() => onSelect?.(result)}
          />
        ))}
      </div>
    </div>
  );
}

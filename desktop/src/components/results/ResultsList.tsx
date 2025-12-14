/**
 * ResultsList - Container component for search results
 *
 * Displays a list of ResultCard components with:
 * - Result count header
 * - Selection state management
 * - Keyboard navigation support
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { ResultCard } from './ResultCard';
import type { SearchResult } from '@/types/api';

/**
 * Error boundary to catch rendering errors in individual result cards
 */
class ResultErrorBoundary extends Component<
  { children: ReactNode; resultId: string },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: ReactNode; resultId: string }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ResultsList] Error rendering result:', this.props.resultId, error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-900/20 border border-red-500/30 text-red-400 text-sm rounded">
          <p className="font-medium">Error rendering result</p>
          <p className="text-xs mt-1 text-red-400/70">
            ID: {this.props.resultId}
          </p>
          <p className="text-xs mt-1 text-red-400/70">
            {this.state.error?.message}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

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
          <ResultErrorBoundary key={result.id} resultId={result.id}>
            <ResultCard
              result={result}
              query={query}
              isSelected={selectedId === result.id || index === selectedIndex}
              onSelect={() => onSelect?.(result)}
            />
          </ResultErrorBoundary>
        ))}
      </div>
    </div>
  );
}

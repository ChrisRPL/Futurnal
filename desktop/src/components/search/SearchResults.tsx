/**
 * SearchResults - Display search results with keyboard navigation
 *
 * Shows search results with score, entity type, and metadata.
 * Supports arrow key navigation with visual selection state.
 */

import { FileText, Code, User, Calendar, Lightbulb } from 'lucide-react';
import { cn } from '@/lib/utils';
import { truncate, formatRelativeTime } from '@/lib/utils';
import type { SearchResult, EntityType } from '@/types/api';

interface SearchResultsProps {
  /** Search results to display */
  results: SearchResult[];
  /** Handler when a result is selected */
  onSelect: (result: SearchResult) => void;
  /** Currently selected index for keyboard navigation */
  selectedIndex?: number;
  /** Additional class names */
  className?: string;
}

/**
 * Icon mapping for entity types
 */
const ENTITY_ICONS: Record<EntityType, typeof FileText> = {
  Event: Calendar,
  Document: FileText,
  Code: Code,
  Person: User,
  Concept: Lightbulb,
};

/**
 * Format confidence score for display
 */
function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

export function SearchResults({
  results,
  onSelect,
  selectedIndex = -1,
  className,
}: SearchResultsProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <div
      data-slot="search-results"
      data-testid="search-results"
      className={cn('py-2', className)}
    >
      <div className="px-3 py-2 text-xs text-white/40 font-medium">
        Results ({results.length})
      </div>
      <div className="space-y-1">
        {results.map((result, index) => {
          const Icon = result.entity_type
            ? ENTITY_ICONS[result.entity_type]
            : FileText;

          return (
            <button
              key={result.id}
              onClick={() => onSelect(result)}
              data-selected={index === selectedIndex}
              className={cn(
                'w-full text-left px-3 py-3',
                'bg-white/5 border border-white/10',
                'hover:border-white/20 transition-colors',
                index === selectedIndex && 'bg-white/10 border-l-2 border-l-white/50'
              )}
            >
              <div className="flex items-start gap-3">
                {/* Icon */}
                <div className="flex-shrink-0 mt-0.5">
                  <Icon className="h-4 w-4 text-white/40" />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {/* Entity type badge */}
                    {result.entity_type && (
                      <span className="text-xs px-1.5 py-0.5 bg-white/10 text-white/60 rounded">
                        {result.entity_type}
                      </span>
                    )}
                    {/* Score badge */}
                    <span className="text-xs px-1.5 py-0.5 bg-white/5 text-white/40 rounded">
                      {formatScore(result.score)}
                    </span>
                  </div>

                  {/* Content preview */}
                  <p className="text-sm text-white/80 line-clamp-2">
                    {truncate(result.content, 150)}
                  </p>

                  {/* Metadata row */}
                  <div className="flex items-center gap-3 mt-2 text-xs text-white/40">
                    {result.source_type && (
                      <span className="capitalize">{result.source_type}</span>
                    )}
                    {result.timestamp && (
                      <span>{formatRelativeTime(result.timestamp)}</span>
                    )}
                    {result.confidence && (
                      <span>Confidence: {formatScore(result.confidence)}</span>
                    )}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

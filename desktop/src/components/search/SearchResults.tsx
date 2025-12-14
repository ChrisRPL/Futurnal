/**
 * SearchResults - Display search results with keyboard navigation
 *
 * Shows search results with score, entity type, and metadata.
 * Supports arrow key navigation with visual selection state.
 * Includes "Show in Graph" action to visualize results in the knowledge graph.
 */

import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  Code,
  User,
  Calendar,
  Lightbulb,
  Mail,
  Inbox,
  Database,
  Building,
  Network,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { truncate, formatRelativeTime } from '@/lib/utils';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { useUIStore } from '@/stores/uiStore';
import type { SearchResult, EntityType, GraphContext } from '@/types/api';

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
  Email: Mail,
  Mailbox: Inbox,
  Source: Database,
  Organization: Building,
  Entity: Database,  // Generic entity from knowledge graph
};

/**
 * Format confidence score for display
 */
function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/**
 * GraphContextBadge - Shows graph traversal context for GraphRAG results
 *
 * Per GFM-RAG paper: Shows "why" a result is relevant via graph connections
 */
function GraphContextBadge({ context }: { context: GraphContext }) {
  const relatedCount = context.relatedEntities?.length || 0;
  const hasPath = context.pathToQuery && context.pathToQuery.length > 0;

  if (relatedCount === 0 && !hasPath) return null;

  // Format path as readable trail
  const pathDisplay = hasPath
    ? context.pathToQuery.slice(0, 3).map((p) => {
        // Extract just the entity name from path
        const parts = p.split(':');
        return parts.length > 1 ? parts[1] : p.slice(0, 12);
      }).join(' \u2192 ')
    : '';

  return (
    <div className="mt-2 flex items-center gap-2 text-xs text-white/40">
      <Network className="h-3 w-3" />
      {relatedCount > 0 && (
        <span>{relatedCount} related</span>
      )}
      {hasPath && (
        <span className="text-white/30 truncate max-w-[200px]">
          via {pathDisplay}{context.pathToQuery.length > 3 ? '\u2026' : ''}
        </span>
      )}
      {context.hopCount > 0 && (
        <span className="text-white/30">
          ({context.hopCount} {context.hopCount === 1 ? 'hop' : 'hops'})
        </span>
      )}
    </div>
  );
}

export function SearchResults({
  results,
  onSelect,
  selectedIndex = -1,
  className,
}: SearchResultsProps) {
  const navigate = useNavigate();
  const closeCommandPalette = useUIStore((state) => state.closeCommandPalette);

  // Build highlight ID for a search result (same logic as ResultCard)
  const getHighlightId = useCallback((result: SearchResult): string => {
    const entityId = result.metadata?.entityId as string | undefined;
    const graphNodeId = result.metadata?.graphNodeId as string | undefined;
    const parentId = result.metadata?.parent_id as string | undefined;
    const sourceDocId = result.metadata?.source_document_id as string | undefined;

    let highlightId: string;
    if (graphNodeId) {
      highlightId = graphNodeId;
    } else if (entityId) {
      highlightId = entityId;
    } else if (parentId) {
      highlightId = `doc:${parentId}`;
    } else if (sourceDocId) {
      highlightId = `doc:${sourceDocId}`;
    } else {
      highlightId = `${result.id},doc:${result.id}`;
    }

    // Extract filename from path
    const pathField = result.metadata?.path as string | undefined;
    const sourceField = result.metadata?.source as string | undefined;
    const filePath = pathField || (sourceField?.includes('/') ? sourceField : undefined);
    const filename = filePath?.split('/').pop()?.replace(/\.\w+$/, '');

    if (filename && filename.length >= 4 && !highlightId.includes(filename)) {
      highlightId = `${highlightId},${filename}`;
    }

    return highlightId;
  }, []);

  // Handle show in graph action - selects the node (opens detail panel)
  const handleShowInGraph = useCallback(
    (e: React.MouseEvent, result: SearchResult) => {
      e.stopPropagation(); // Prevent triggering onSelect

      const highlightId = getHighlightId(result);
      closeCommandPalette();
      // Use ?select= to select the node and open detail panel
      navigate(`/graph?select=${encodeURIComponent(highlightId)}`);
    },
    [navigate, closeCommandPalette, getHighlightId]
  );

  // Handle show all in graph - highlights multiple (doesn't select)
  const handleShowAllInGraph = useCallback(() => {
    // Get all highlight IDs from results
    const highlightIds = results.map(getHighlightId);

    closeCommandPalette();
    // Use ?highlight= for multiple nodes (can't select multiple)
    navigate(`/graph?highlight=${highlightIds.map(encodeURIComponent).join(',')}`);
  }, [results, navigate, closeCommandPalette, getHighlightId]);

  if (results.length === 0) {
    return null;
  }

  return (
    <TooltipProvider delayDuration={300}>
      <div
        data-slot="search-results"
        data-testid="search-results"
        className={cn('py-2', className)}
      >
        <div className="flex items-center justify-between px-3 py-2">
          <span className="text-xs text-white/40 font-medium">
            Results ({results.length})
          </span>
          {results.length > 1 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={handleShowAllInGraph}
                  className="flex items-center gap-1 text-xs text-white/40 hover:text-white/70 transition-colors"
                >
                  <Network className="h-3 w-3" />
                  Show all in graph
                </button>
              </TooltipTrigger>
              <TooltipContent>View all results in knowledge graph</TooltipContent>
            </Tooltip>
          )}
        </div>
        <div className="space-y-1">
          {results.map((result, index) => {
            // Defensive icon lookup with fallback
            const Icon = (() => {
              if (!result.entity_type) return FileText;
              const icon = ENTITY_ICONS[result.entity_type];
              if (!icon) {
                console.warn('[SearchResults] Unknown entity_type:', result.entity_type);
                return FileText;
              }
              return icon;
            })();

            return (
              <div
                key={result.id}
                data-selected={index === selectedIndex}
                className={cn(
                  'relative group px-3 py-3',
                  'bg-white/5 border border-white/10',
                  'hover:border-white/20 transition-colors',
                  index === selectedIndex && 'bg-white/10 border-l-2 border-l-white/50'
                )}
              >
                <button
                  onClick={() => onSelect(result)}
                  className="w-full text-left"
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
                        {/* GraphRAG scores */}
                        {result.vector_score !== undefined && (
                          <span className="text-white/30">
                            Vector: {formatScore(result.vector_score)}
                          </span>
                        )}
                        {result.graph_score !== undefined && result.graph_score > 0 && (
                          <span className="text-white/30">
                            Graph: {formatScore(result.graph_score)}
                          </span>
                        )}
                      </div>

                      {/* Graph context from GraphRAG */}
                      {result.graph_context && (
                        <GraphContextBadge context={result.graph_context} />
                      )}
                    </div>
                  </div>
                </button>

                {/* Show in Graph action - always visible */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={(e) => handleShowInGraph(e, result)}
                      className={cn(
                        'absolute right-2 top-2 p-1.5 rounded',
                        'text-white/40 hover:text-white hover:bg-white/10',
                        'border border-white/10 hover:border-white/30',
                        'transition-colors'
                      )}
                      aria-label="Show in knowledge graph"
                    >
                      <Network className="h-4 w-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>Show in knowledge graph</TooltipContent>
                </Tooltip>
              </div>
            );
          })}
        </div>
      </div>
    </TooltipProvider>
  );
}

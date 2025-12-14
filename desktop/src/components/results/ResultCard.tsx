/**
 * ResultCard - Main search result display component
 *
 * Displays a search result with:
 * - Entity type and source type badges with icons
 * - Timestamp in relative format
 * - Score and confidence indicators
 * - Highlighted content snippet with expand/collapse
 * - Causal chain preview (when present)
 * - Quick actions (Open, Copy, Save, Share)
 * - Collapsible provenance panel
 */

import { useState, useCallback } from 'react';
import {
  Calendar,
  FileText,
  Code,
  User,
  Lightbulb,
  Image,
  Mic,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Copy,
  Bookmark,
  Share2,
  Sparkles,
  Mail,
  Inbox,
  Database,
  Building,
  Network,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { cn, highlightTerms, formatTimestampRelative } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScoreIndicator } from './ScoreIndicator';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import { CausalChainPreview } from './CausalChainPreview';
import { ProvenancePanel } from './ProvenancePanel';
import { useUIStore } from '@/stores/uiStore';
import type { SearchResult, EntityType, SourceType } from '@/types/api';

interface ResultCardProps {
  /** Search result data */
  result: SearchResult;
  /** Search query for term highlighting */
  query: string;
  /** Whether this card is selected */
  isSelected?: boolean;
  /** Handler for card selection */
  onSelect?: () => void;
  /** Additional class names */
  className?: string;
}

/** Icon mapping for entity types */
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

/** Icon mapping for source types */
const SOURCE_ICONS: Record<SourceType, typeof FileText> = {
  text: FileText,
  ocr: Image,
  audio: Mic,
  code: Code,
};

/** Maximum content length before showing expand button */
const MAX_CONTENT_LENGTH = 200;

export function ResultCard({
  result,
  query,
  isSelected = false,
  onSelect,
  className,
}: ResultCardProps) {
  const navigate = useNavigate();
  const closeCommandPalette = useUIStore((state) => state.closeCommandPalette);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showProvenance, setShowProvenance] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

  // Handle show in knowledge graph
  // Graph nodes use IDs like "doc:{parent_id}" or "email:{source}:{uid}"
  // Search results may have different ID schemes, so we try multiple approaches
  const handleShowInGraph = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();

    // Priority order for finding the graph node ID:
    // 1. entityId in metadata (if backend sets it)
    // 2. graphNodeId in metadata (explicit graph ID)
    // 3. parent_id in metadata with doc: prefix
    // 4. source_document_id in metadata with doc: prefix
    // 5. result.id with doc: prefix (assume it's a document hash)
    // 6. Raw result.id as fallback

    const entityId = result.metadata?.entityId as string | undefined;
    const graphNodeId = result.metadata?.graphNodeId as string | undefined;
    const parentId = result.metadata?.parent_id as string | undefined;
    const sourceDocId = result.metadata?.source_document_id as string | undefined;

    let highlightId: string;

    if (graphNodeId) {
      // Explicit graph node ID - use as-is
      highlightId = graphNodeId;
    } else if (entityId) {
      // Entity ID from metadata
      highlightId = entityId;
    } else if (parentId) {
      // Parent ID - prefix with doc:
      highlightId = `doc:${parentId}`;
    } else if (sourceDocId) {
      // Source document ID - prefix with doc:
      highlightId = `doc:${sourceDocId}`;
    } else {
      // Fallback: try both raw ID and doc: prefixed version
      // Pass comma-separated to try multiple matches
      highlightId = `${result.id},doc:${result.id}`;
    }

    // Extract filename from the actual file path (not the source name)
    // Priority: path field > source field (if it looks like a path)
    const pathField = result.metadata?.path as string | undefined;
    const sourceField = result.metadata?.source as string | undefined;

    // Use path if available, otherwise use source only if it contains a slash (is a path)
    const filePath = pathField || (sourceField?.includes('/') ? sourceField : undefined);
    const filename = filePath?.split('/').pop()?.replace(/\.\w+$/, ''); // Remove extension

    console.log('[ResultCard] Show in graph:', {
      resultId: result.id,
      entityId,
      graphNodeId,
      parentId,
      sourceDocId,
      pathField,
      sourceField,
      filename,
      highlightId,
      allMetadata: JSON.stringify(result.metadata, null, 2)
    });

    // If we extracted a filename (at least 4 chars to avoid false matches), add it
    let finalHighlightId = highlightId;
    if (filename && filename.length >= 4 && !highlightId.includes(filename)) {
      finalHighlightId = `${highlightId},${filename}`;
    }

    // Close the command palette before navigating
    closeCommandPalette();

    // Use ?select= to select the node and open detail panel (not just highlight)
    navigate(`/graph?select=${encodeURIComponent(finalHighlightId)}`);
  }, [result.id, result.metadata, navigate, closeCommandPalette]);

  // Defensive icon lookup with fallback - logs unknown types for debugging
  const EntityIcon = (() => {
    if (!result.entity_type) return FileText;
    const icon = ENTITY_ICONS[result.entity_type];
    if (!icon) {
      console.warn('[ResultCard] Unknown entity_type:', result.entity_type, 'for result:', result.id);
      return FileText;
    }
    return icon;
  })();

  const SourceIcon = (() => {
    if (!result.source_type) return FileText;
    const icon = SOURCE_ICONS[result.source_type];
    if (!icon) {
      console.warn('[ResultCard] Unknown source_type:', result.source_type, 'for result:', result.id);
      return FileText;
    }
    return icon;
  })();

  // Highlight query terms in content
  const highlightedContent = highlightTerms(result.content, query);

  // Check if content is long enough to warrant expand/collapse
  const isLongContent = result.content.length > MAX_CONTENT_LENGTH;

  // Handle copy action
  const handleCopy = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(result.content);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [result.content]);

  // Handle open source action - use path field for actual file location
  const handleOpenSource = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    const filePath = (result.metadata?.path || result.metadata?.source) as string | undefined;

    console.log('[ResultCard] Opening file:', filePath);

    // Only open if it looks like a valid file path
    if (filePath && filePath.startsWith('/')) {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('open_file', { path: filePath });
        console.log('[ResultCard] File opened successfully');
      } catch (err) {
        console.error('[ResultCard] Failed to open source:', err);
      }
    } else {
      console.warn('[ResultCard] No valid file path found. filePath:', filePath);
    }
  }, [result.metadata]);

  // Toggle expand/collapse
  const handleToggleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded((prev) => !prev);
  }, []);

  // Toggle provenance panel
  const handleToggleProvenance = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setShowProvenance((prev) => !prev);
  }, []);

  return (
    <div
      data-slot="result-card"
      data-testid="result-card"
      data-selected={isSelected}
      onClick={onSelect}
      className={cn(
        'group p-4 transition-all duration-150 cursor-pointer',
        'bg-white/5 border border-white/10',
        'hover:border-white/20 hover:bg-white/[0.07]',
        isSelected && 'bg-white/10 border-l-2 border-l-white/50',
        className
      )}
    >
      {/* Header Row */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Entity Type Badge */}
          {result.entity_type && (
            <span className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 bg-white/10 text-white/70 rounded">
              <EntityIcon className="h-3 w-3" />
              {result.entity_type}
            </span>
          )}

          {/* Graph Enhanced Badge - shown for GraphRAG results */}
          {result.metadata?.graph_enhanced && (
            <span className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-300 rounded border border-blue-500/30">
              <Network className="h-3 w-3" />
              Graph Enhanced
            </span>
          )}

          {/* Source Type Badge (only show for non-text sources) */}
          {result.source_type && result.source_type !== 'text' && (
            <span className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 bg-white/5 text-white/50 rounded border border-white/10">
              <SourceIcon className="h-3 w-3" />
              {result.source_type.toUpperCase()}
              {result.source_confidence && (
                <span className="text-white/40">
                  {Math.round(result.source_confidence * 100)}%
                </span>
              )}
            </span>
          )}

          {/* Timestamp */}
          {result.timestamp && (
            <span className="inline-flex items-center gap-1 text-xs text-white/40">
              <Calendar className="h-3 w-3" />
              {formatTimestampRelative(result.timestamp)}
            </span>
          )}
        </div>

        {/* Score and Confidence Indicators */}
        <div className="flex items-center gap-2">
          <ScoreIndicator score={result.score} />
          <ConfidenceIndicator confidence={result.confidence} />
        </div>
      </div>

      {/* Content Snippet */}
      <div
        className={cn(
          'text-sm text-white/80 leading-relaxed',
          !isExpanded && isLongContent && 'line-clamp-3'
        )}
        dangerouslySetInnerHTML={{ __html: highlightedContent }}
      />

      {/* Expand/Collapse Button */}
      {isLongContent && (
        <button
          onClick={handleToggleExpand}
          className="mt-2 text-xs text-white/50 hover:text-white/70 flex items-center gap-1 transition-colors"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="h-3 w-3" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3" />
              Show more
            </>
          )}
        </button>
      )}

      {/* Causal Chain Preview */}
      {result.causal_chain && (
        <CausalChainPreview chain={result.causal_chain} className="mt-3" />
      )}

      {/* Footer - Provenance Toggle and Quick Actions */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/10">
        {/* Provenance Toggle */}
        <button
          onClick={handleToggleProvenance}
          className="text-xs text-white/40 hover:text-white/60 flex items-center gap-1 transition-colors"
        >
          <Sparkles className="h-3 w-3" />
          {showProvenance ? 'Hide' : 'Show'} provenance
        </button>

        {/* Quick Actions - always visible */}
        <div className="flex items-center gap-1">
          {/* Show in Graph - always visible */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-blue-400/70 hover:text-blue-400 hover:bg-blue-500/10 border border-blue-400/30 hover:border-blue-400/50"
            onClick={handleShowInGraph}
            title="Show in knowledge graph"
          >
            <Network className="h-3.5 w-3.5" />
          </Button>

          {/* Open Source */}
          {!!(result.metadata?.path || result.metadata?.source) && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/10"
              onClick={handleOpenSource}
              title="Open source file"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </Button>
          )}

          {/* Copy */}
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              'h-7 w-7 hover:bg-white/10 transition-colors',
              copySuccess ? 'text-white' : 'text-white/40 hover:text-white/70'
            )}
            onClick={handleCopy}
            title={copySuccess ? 'Copied!' : 'Copy content'}
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>

          {/* Save (placeholder) */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/10"
            onClick={(e) => e.stopPropagation()}
            title="Save (coming soon)"
          >
            <Bookmark className="h-3.5 w-3.5" />
          </Button>

          {/* Share (placeholder) */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/10"
            onClick={(e) => e.stopPropagation()}
            title="Share (coming soon)"
          >
            <Share2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Provenance Panel (collapsible) */}
      {showProvenance && (
        <ProvenancePanel metadata={result.metadata} className="mt-3" />
      )}
    </div>
  );
}

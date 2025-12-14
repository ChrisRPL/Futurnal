/**
 * ResultDetailPanel - Slide-in side panel for detailed result view
 *
 * Displays full result information:
 * - Complete content (no truncation)
 * - Full provenance information
 * - All quick actions prominently displayed
 * - Causal chain if present
 */

import { useCallback, useState } from 'react';
import {
  X,
  Calendar,
  FileText,
  Code,
  User,
  Lightbulb,
  Image,
  Mic,
  ExternalLink,
  Copy,
  Bookmark,
  Share2,
  Mail,
  Inbox,
  Database,
  Building,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { cn, highlightTerms, formatTimestampRelative } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ScoreIndicator } from './ScoreIndicator';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import { CausalChainPreview } from './CausalChainPreview';
import { ProvenancePanel } from './ProvenancePanel';
import type { SearchResult, EntityType, SourceType } from '@/types/api';

interface ResultDetailPanelProps {
  /** Search result to display */
  result: SearchResult;
  /** Search query for term highlighting */
  query: string;
  /** Handler for closing the panel */
  onClose: () => void;
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

export function ResultDetailPanel({
  result,
  query,
  onClose,
  className,
}: ResultDetailPanelProps) {
  const [copySuccess, setCopySuccess] = useState(false);

  // Defensive icon lookup with fallback
  const EntityIcon = (() => {
    if (!result.entity_type) return FileText;
    const icon = ENTITY_ICONS[result.entity_type];
    if (!icon) {
      console.warn('[ResultDetailPanel] Unknown entity_type:', result.entity_type);
      return FileText;
    }
    return icon;
  })();

  const SourceIcon = (() => {
    if (!result.source_type) return FileText;
    const icon = SOURCE_ICONS[result.source_type];
    if (!icon) {
      console.warn('[ResultDetailPanel] Unknown source_type:', result.source_type);
      return FileText;
    }
    return icon;
  })();

  // Highlight query terms in content
  const highlightedContent = highlightTerms(result.content, query);

  // Handle copy action
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(result.content);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [result.content]);

  // Handle open source action - try multiple locations for path
  const handleOpenSource = useCallback(async () => {
    const sourcePath = (
      result.metadata?.path ||
      result.metadata?.source
    ) as string | undefined;

    console.log('[ResultDetailPanel] Opening file:', sourcePath);
    console.log('[ResultDetailPanel] Metadata:', result.metadata);

    // Only open if it looks like a valid file path
    if (sourcePath && sourcePath.startsWith('/')) {
      try {
        await invoke('open_file', { path: sourcePath });
        console.log('[ResultDetailPanel] File opened successfully');
      } catch (err) {
        console.error('[ResultDetailPanel] Failed to open source:', err);
      }
    } else {
      console.warn('[ResultDetailPanel] No valid file path found. sourcePath:', sourcePath);
    }
  }, [result.metadata]);

  return (
    <div
      data-slot="result-detail-panel"
      data-testid="result-detail-panel"
      className={cn(
        'flex flex-col h-full bg-black border-l border-white/10',
        'animate-slide-in-right',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between p-4 border-b border-white/10">
        <div className="flex-1 min-w-0">
          {/* Badges Row */}
          <div className="flex items-center gap-2 flex-wrap mb-2">
            {/* Entity Type Badge */}
            {result.entity_type && (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-white/10 text-white/80 rounded">
                <EntityIcon className="h-3.5 w-3.5" />
                {result.entity_type}
              </span>
            )}

            {/* Graph Enhanced Badge - shown for GraphRAG results */}
            {result.metadata?.graph_enhanced && (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-blue-500/20 text-blue-300 rounded border border-blue-500/30">
                <Database className="h-3.5 w-3.5" />
                Graph Enhanced
              </span>
            )}

            {/* Source Type Badge */}
            {result.source_type && result.source_type !== 'text' && (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-white/5 text-white/60 rounded border border-white/10">
                <SourceIcon className="h-3.5 w-3.5" />
                {result.source_type.toUpperCase()}
                {result.source_confidence && (
                  <span className="text-white/40 ml-1">
                    {Math.round(result.source_confidence * 100)}%
                  </span>
                )}
              </span>
            )}
          </div>

          {/* Timestamp and Scores */}
          <div className="flex items-center gap-4 text-xs text-white/50">
            {result.timestamp && (
              <span className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {formatTimestampRelative(result.timestamp)}
              </span>
            )}
            <div className="flex items-center gap-2">
              <span>Score:</span>
              <ScoreIndicator score={result.score} />
            </div>
            <div className="flex items-center gap-2">
              <span>Confidence:</span>
              <ConfidenceIndicator confidence={result.confidence} />
            </div>
          </div>
        </div>

        {/* Close Button */}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-white/40 hover:text-white/70 hover:bg-white/10 flex-shrink-0"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </Button>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Full Content */}
          <div>
            <div className="text-xs font-medium text-white/50 mb-2">Content</div>
            <div
              className="text-sm text-white/80 leading-relaxed whitespace-pre-wrap"
              dangerouslySetInnerHTML={{ __html: highlightedContent }}
            />
          </div>

          {/* Causal Chain */}
          {result.causal_chain && (
            <CausalChainPreview chain={result.causal_chain} />
          )}

          {/* Provenance */}
          <div>
            <div className="text-xs font-medium text-white/50 mb-2">Provenance</div>
            <ProvenancePanel metadata={result.metadata} />
          </div>

          {/* Additional Metadata - filtered to user-relevant fields */}
          {(() => {
            // Technical fields to hide from users
            const hiddenFields = new Set([
              'source', 'extractionTimestamp', 'schemaVersion', 'entityId',
              'graph_score', 'graph_enhanced', 'graphrag', 'vector_score',
              'indexed_at', 'ingested_at', 'needs_reembedding', 'schema_version',
              'content', 'document', 'source_type', 'entity_type',
            ]);
            const visibleEntries = Object.entries(result.metadata)
              .filter(([key]) => !hiddenFields.has(key));

            if (visibleEntries.length === 0) return null;

            return (
              <div>
                <div className="text-xs font-medium text-white/50 mb-2">
                  Details
                </div>
                <div className="p-3 rounded bg-white/[0.03] border border-white/10 space-y-1">
                  {visibleEntries.map(([key, value]) => (
                    <div key={key} className="flex items-start gap-2 text-xs">
                      <span className="text-white/40 flex-shrink-0 capitalize">{key.replace(/_/g, ' ')}:</span>
                      <span className="text-white/60 break-all">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      </ScrollArea>

      {/* Actions Footer */}
      <div className="p-4 border-t border-white/10 space-y-2">
        {/* Primary Actions */}
        <div className="flex gap-2">
          {!!(result.metadata?.path || result.metadata?.source) && (
            <Button
              variant="outline"
              size="sm"
              className="flex-1 gap-2 border-white/20 text-white/70 hover:text-white hover:bg-white/10"
              onClick={handleOpenSource}
            >
              <ExternalLink className="h-4 w-4" />
              Open File
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            className={cn(
              'flex-1 gap-2 border-white/20 hover:bg-white/10',
              copySuccess ? 'text-white' : 'text-white/70 hover:text-white'
            )}
            onClick={handleCopy}
          >
            <Copy className="h-4 w-4" />
            {copySuccess ? 'Copied!' : 'Copy'}
          </Button>
        </div>

        {/* Secondary Actions */}
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="flex-1 gap-2 text-white/50 hover:text-white/70 hover:bg-white/10"
            disabled
            title="Coming soon"
          >
            <Bookmark className="h-4 w-4" />
            Save
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="flex-1 gap-2 text-white/50 hover:text-white/70 hover:bg-white/10"
            disabled
            title="Coming soon"
          >
            <Share2 className="h-4 w-4" />
            Share
          </Button>
        </div>
      </div>
    </div>
  );
}

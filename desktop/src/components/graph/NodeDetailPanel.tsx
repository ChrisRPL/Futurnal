/**
 * NodeDetailPanel - Slide-in side panel for node details
 *
 * Displays detailed information about a selected graph node:
 * - Node type and label
 * - Timestamp if available
 * - Metadata properties
 * - Connected nodes list
 * - Quick actions (bookmark, copy ID, focus, open source)
 *
 * Follows the ResultDetailPanel pattern from search results.
 */

import { useCallback, useState, useEffect } from 'react';
import {
  X,
  Calendar,
  FileText,
  Code,
  User,
  Lightbulb,
  ExternalLink,
  Link2,
  ArrowRight,
  Mail,
  Inbox,
  Database,
  Building,
  Star,
  Copy,
  Focus,
  Check,
  MessageSquare,
  GitBranch,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { cn, formatTimestampRelative } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { useGraphStore } from '@/stores/graphStore';
import { useCausalStore } from '@/stores/causalStore';
import { graphApi } from '@/lib/api';
import type { GraphNode, GraphLink, EntityType } from '@/types/api';

interface NodeDetailPanelProps {
  /** Selected graph node */
  node: GraphNode;
  /** All links in the graph (to find connections) */
  links: GraphLink[];
  /** All nodes in the graph (to resolve labels) */
  nodes: GraphNode[];
  /** Close handler */
  onClose: () => void;
  /** Navigate to a connected node */
  onNavigateToNode?: (nodeId: string) => void;
  /** Focus on node in graph (center and zoom) */
  onFocusNode?: (nodeId: string) => void;
  /** Ask about this node - opens chat with context */
  onAskAbout?: (nodeId: string, nodeLabel: string) => void;
  /** Additional CSS classes */
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
};

/** Opacity mapping for entity type badges */
const TYPE_OPACITIES: Record<EntityType, string> = {
  Event: 'bg-white/20',
  Person: 'bg-white/16',
  Document: 'bg-white/12',
  Code: 'bg-white/10',
  Concept: 'bg-white/8',
  Email: 'bg-amber-500/15',
  Mailbox: 'bg-blue-500/15',
  Source: 'bg-purple-500/15',
  Organization: 'bg-white/15',
};

export function NodeDetailPanel({
  node,
  links,
  nodes,
  onClose,
  onNavigateToNode,
  onFocusNode,
  onAskAbout,
  className,
}: NodeDetailPanelProps) {
  const nodeType = node.node_type || 'Document';
  const Icon = ENTITY_ICONS[nodeType] || FileText;

  // Bookmark state from store
  const { bookmarkedNodeIds, toggleBookmark, setBookmarkedNodes } = useGraphStore();
  const isBookmarked = bookmarkedNodeIds.includes(node.id);

  // Feedback state for copy action
  const [copiedFeedback, setCopiedFeedback] = useState(false);
  const [bookmarkLoading, setBookmarkLoading] = useState(false);
  const [causalExpanded, setCausalExpanded] = useState(false);

  // Causal store for exploring causes and effects
  const {
    causes,
    effects,
    isLoading: causalLoading,
    error: causalError,
    anchorEventId,
    findCauses,
    findEffects,
  } = useCausalStore();

  // Load bookmarks from backend on mount
  useEffect(() => {
    graphApi.getBookmarks().then(setBookmarkedNodes).catch(console.warn);
  }, [setBookmarkedNodes]);

  // Find connected nodes
  const connections = links.filter((link) => {
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id;
    const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id;
    return sourceId === node.id || targetId === node.id;
  });

  // Resolve connected node info
  const connectedNodes = connections.map((link) => {
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id;
    const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id;
    const connectedId = sourceId === node.id ? targetId : sourceId;
    const connectedNode = nodes.find((n) => n.id === connectedId);
    const isOutgoing = sourceId === node.id;

    return {
      id: connectedId,
      label: connectedNode?.label ?? connectedId,
      type: connectedNode?.node_type ?? 'Document',
      relationship: link.relationship,
      isOutgoing,
    };
  });

  // Handle bookmark toggle
  const handleToggleBookmark = useCallback(async () => {
    setBookmarkLoading(true);
    try {
      if (isBookmarked) {
        await graphApi.unbookmarkNode(node.id);
      } else {
        await graphApi.bookmarkNode(node.id);
      }
      toggleBookmark(node.id);
    } catch (err) {
      console.error('Failed to toggle bookmark:', err);
    } finally {
      setBookmarkLoading(false);
    }
  }, [node.id, isBookmarked, toggleBookmark]);

  // Handle copy node ID
  const handleCopyId = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(node.id);
      setCopiedFeedback(true);
      setTimeout(() => setCopiedFeedback(false), 2000);
    } catch (err) {
      console.error('Failed to copy node ID:', err);
    }
  }, [node.id]);

  // Handle focus in graph
  const handleFocusInGraph = useCallback(() => {
    onFocusNode?.(node.id);
  }, [node.id, onFocusNode]);

  // Handle ask about this node - opens chat with entity context
  const handleAskAbout = useCallback(() => {
    onAskAbout?.(node.id, node.label);
  }, [node.id, node.label, onAskAbout]);

  // Handle causal exploration
  const handleExploreCausal = useCallback(async () => {
    setCausalExpanded(true);
    await Promise.all([
      findCauses(node.id),
      findEffects(node.id),
    ]);
  }, [node.id, findCauses, findEffects]);

  // Check if this node is the current causal anchor
  const isCausalAnchor = anchorEventId === node.id;

  // Handle open source action - try multiple locations for path
  const handleOpenSource = useCallback(async () => {
    const details = node.metadata?.details as Record<string, unknown> | undefined;
    const sourcePath = (
      details?.path ||
      node.metadata?.path ||
      node.metadata?.source
    ) as string | undefined;

    // Only open if it looks like a valid file path
    if (sourcePath && sourcePath.startsWith('/')) {
      try {
        await invoke('open_file', { path: sourcePath });
      } catch (err) {
        console.error('Failed to open source:', err);
      }
    }
  }, [node.metadata]);

  const details = node.metadata?.details as Record<string, unknown> | undefined;
  const hasPath = details?.path || node.metadata?.path;

  return (
    <TooltipProvider delayDuration={300}>
      <div
        data-slot="node-detail-panel"
        data-testid="node-detail-panel"
        role="complementary"
        aria-label={`Details for ${node.label}, ${nodeType}`}
        className={cn(
          'flex flex-col h-full bg-black border-l border-white/10',
          'animate-slide-in-right',
          className
        )}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-4 border-b border-white/10">
          <div className="flex-1 min-w-0">
            {/* Entity Type Badge */}
            <div className="flex items-center gap-2 mb-2">
              <span
                className={cn(
                  'inline-flex items-center gap-1.5 text-xs px-2 py-1 text-white/80',
                  TYPE_OPACITIES[nodeType]
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {nodeType}
              </span>
              {isBookmarked && (
                <Star className="h-3.5 w-3.5 text-amber-400 fill-amber-400" />
              )}
            </div>

            {/* Node Label */}
            <h3 className="text-base font-medium text-white/90 leading-tight mb-2">
              {node.label}
            </h3>

            {/* Timestamp */}
            {node.timestamp && (
              <div className="flex items-center gap-1 text-xs text-white/50">
                <Calendar className="h-3 w-3" />
                {formatTimestampRelative(node.timestamp)}
              </div>
            )}
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

        {/* Quick Actions Bar */}
        <div className="flex items-center gap-1 px-4 py-2 border-b border-white/5 bg-white/[0.02]">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className={cn(
                  'h-7 px-2 gap-1.5 text-xs',
                  isBookmarked
                    ? 'text-amber-400 hover:text-amber-300'
                    : 'text-white/50 hover:text-white/80'
                )}
                onClick={handleToggleBookmark}
                disabled={bookmarkLoading}
              >
                <Star className={cn('h-3.5 w-3.5', isBookmarked && 'fill-current')} />
                {isBookmarked ? 'Saved' : 'Save'}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {isBookmarked ? 'Remove from bookmarks' : 'Add to bookmarks'}
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 gap-1.5 text-xs text-white/50 hover:text-white/80"
                onClick={handleCopyId}
              >
                {copiedFeedback ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-green-400" />
                    <span className="text-green-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    Copy ID
                  </>
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Copy node ID to clipboard</TooltipContent>
          </Tooltip>

          {onFocusNode && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 gap-1.5 text-xs text-white/50 hover:text-white/80"
                  onClick={handleFocusInGraph}
                >
                  <Focus className="h-3.5 w-3.5" />
                  Focus
                </Button>
              </TooltipTrigger>
              <TooltipContent>Center graph on this node</TooltipContent>
            </Tooltip>
          )}

          {onAskAbout && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 gap-1.5 text-xs text-white/50 hover:text-white/80"
                  onClick={handleAskAbout}
                >
                  <MessageSquare className="h-3.5 w-3.5" />
                  Ask
                </Button>
              </TooltipTrigger>
              <TooltipContent>Ask about this node</TooltipContent>
            </Tooltip>
          )}
        </div>

        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-4">
            {/* Connections */}
            {connectedNodes.length > 0 && (
              <div>
                <div className="text-xs font-medium text-white/50 mb-2 flex items-center gap-1.5">
                  <Link2 className="h-3 w-3" />
                  Connections ({connectedNodes.length})
                </div>
                <div className="space-y-1">
                  {connectedNodes.slice(0, 10).map((conn, index) => {
                    const ConnIcon = ENTITY_ICONS[conn.type as EntityType] || FileText;
                    return (
                      <button
                        key={`${conn.id}-${index}`}
                        onClick={() => onNavigateToNode?.(conn.id)}
                        className={cn(
                          'w-full flex items-center gap-2 p-2 text-left',
                          'bg-white/[0.03] hover:bg-white/[0.06] transition-colors',
                          'border border-white/5 hover:border-white/10'
                        )}
                      >
                        <ConnIcon className="h-3.5 w-3.5 text-white/40 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs text-white/70 truncate">
                            {conn.label}
                          </div>
                          <div className="text-[10px] text-white/40 flex items-center gap-1">
                            {conn.isOutgoing ? (
                              <>
                                <ArrowRight className="h-2.5 w-2.5" />
                                {conn.relationship}
                              </>
                            ) : (
                              <>
                                {conn.relationship}
                                <ArrowRight className="h-2.5 w-2.5 rotate-180" />
                              </>
                            )}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                  {connectedNodes.length > 10 && (
                    <div className="text-[10px] text-white/30 text-center py-1">
                      +{connectedNodes.length - 10} more connections
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Causal Exploration */}
            <div>
              <button
                onClick={() => causalExpanded ? setCausalExpanded(false) : handleExploreCausal()}
                className="w-full flex items-center justify-between text-xs font-medium text-white/50 mb-2 hover:text-white/70 transition-colors"
              >
                <div className="flex items-center gap-1.5">
                  <GitBranch className="h-3 w-3" />
                  Causal Exploration
                </div>
                {causalExpanded ? (
                  <ChevronUp className="h-3 w-3" />
                ) : (
                  <ChevronDown className="h-3 w-3" />
                )}
              </button>

              {causalExpanded && (
                <div className="p-3 bg-white/[0.03] border border-white/5 space-y-3">
                  {causalLoading ? (
                    <div className="flex items-center justify-center py-4">
                      <div className="h-4 w-4 border-2 border-white/20 border-t-white/60 rounded-full animate-spin" />
                    </div>
                  ) : causalError ? (
                    <div className="text-xs text-red-400 py-2">{causalError}</div>
                  ) : isCausalAnchor ? (
                    <>
                      {/* Causes */}
                      <div>
                        <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1.5">
                          What led to this ({causes.length})
                        </div>
                        {causes.length === 0 ? (
                          <div className="text-xs text-white/30">No causes found</div>
                        ) : (
                          <div className="space-y-1">
                            {causes.slice(0, 5).map((cause) => (
                              <button
                                key={cause.causeId}
                                onClick={() => onNavigateToNode?.(cause.causeId)}
                                className="w-full flex items-center gap-2 p-1.5 text-left bg-white/[0.02] hover:bg-white/[0.05] transition-colors text-xs"
                              >
                                <ArrowRight className="h-3 w-3 text-white/30 rotate-180" />
                                <span className="text-white/70 truncate flex-1">{cause.causeName}</span>
                                <span className="text-white/30 text-[10px]">
                                  {Math.round(cause.aggregateConfidence * 100)}%
                                </span>
                              </button>
                            ))}
                            {causes.length > 5 && (
                              <div className="text-[10px] text-white/30 text-center">
                                +{causes.length - 5} more
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Effects */}
                      <div>
                        <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1.5">
                          What resulted ({effects.length})
                        </div>
                        {effects.length === 0 ? (
                          <div className="text-xs text-white/30">No effects found</div>
                        ) : (
                          <div className="space-y-1">
                            {effects.slice(0, 5).map((effect) => (
                              <button
                                key={effect.effectId}
                                onClick={() => onNavigateToNode?.(effect.effectId)}
                                className="w-full flex items-center gap-2 p-1.5 text-left bg-white/[0.02] hover:bg-white/[0.05] transition-colors text-xs"
                              >
                                <ArrowRight className="h-3 w-3 text-white/30" />
                                <span className="text-white/70 truncate flex-1">{effect.effectName}</span>
                                <span className="text-white/30 text-[10px]">
                                  {Math.round(effect.aggregateConfidence * 100)}%
                                </span>
                              </button>
                            ))}
                            {effects.length > 5 && (
                              <div className="text-[10px] text-white/30 text-center">
                                +{effects.length - 5} more
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-4">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleExploreCausal}
                        className="text-xs border-white/20 text-white/60 hover:text-white hover:bg-white/10"
                      >
                        <GitBranch className="h-3 w-3 mr-1.5" />
                        Explore Causes & Effects
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Metadata */}
            {node.metadata && Object.keys(node.metadata).length > 0 && (
              <div>
                <div className="text-xs font-medium text-white/50 mb-2">
                  Properties
                </div>
                <div className="p-3 bg-white/[0.03] border border-white/5 space-y-2">
                  {(() => {
                    // Format and flatten metadata for better display
                    const formatKey = (key: string) =>
                      key.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
                        .split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');

                    const formatValue = (value: unknown): string => {
                      if (value === null || value === undefined) return '—';
                      if (typeof value === 'boolean') return value ? 'Yes' : 'No';
                      if (typeof value === 'number') return value.toLocaleString();
                      if (Array.isArray(value)) return value.join(', ');
                      if (typeof value === 'object') {
                        // Flatten simple objects
                        return Object.entries(value)
                          .map(([k, v]) => `${formatKey(k)}: ${v}`)
                          .join(', ');
                      }
                      return String(value);
                    };

                    // Filter out internal/redundant keys
                    const hiddenKeys = ['source', 'sha256', 'parent_id', 'element_id'];

                    return Object.entries(node.metadata)
                      .filter(([key]) => !hiddenKeys.includes(key))
                      .map(([key, value]) => (
                        <div key={key} className="flex items-start gap-2 text-xs">
                          <span className="text-white/50 flex-shrink-0 min-w-[80px]">
                            {formatKey(key)}
                          </span>
                          <span className="text-white/70 break-words">
                            {formatValue(value)}
                          </span>
                        </div>
                      ));
                  })()}
                </div>
              </div>
            )}

            {/* Node ID (for debugging/reference) */}
            <div>
              <div className="text-xs font-medium text-white/50 mb-2">
                Node ID
              </div>
              <div className="text-xs text-white/40 font-mono bg-white/[0.03] p-2 border border-white/5">
                {node.id}
              </div>
            </div>
          </div>
        </ScrollArea>

        {/* Actions Footer */}
        <div className="p-4 border-t border-white/10">
          {hasPath ? (
            <Button
              variant="outline"
              size="sm"
              className="w-full gap-2 border-white/20 text-white/70 hover:text-white hover:bg-white/10"
              onClick={handleOpenSource}
            >
              <ExternalLink className="h-4 w-4" />
              Open Source
            </Button>
          ) : (
            <div className="text-xs text-white/30 text-center">
              No source file available
            </div>
          )}
        </div>
      </div>
    </TooltipProvider>
  );
}

export default NodeDetailPanel;

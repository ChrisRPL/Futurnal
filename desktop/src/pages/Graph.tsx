/**
 * Graph Page - Full-screen knowledge graph exploration
 *
 * Dedicated route (/graph) for immersive graph exploration.
 * Provides full-screen visualization with controls and detail panel.
 * Supports highlight params from search results (?highlight=id1,id2).
 */

import { useRef, useCallback, useMemo, useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Network, Filter, Eye, EyeOff, Mail, Palette, X } from 'lucide-react';
import { KnowledgeGraph, type KnowledgeGraphRef } from '@/components/graph/KnowledgeGraph';
import { GraphControls } from '@/components/graph/GraphControls';
import { NodeDetailPanel } from '@/components/graph/NodeDetailPanel';
import { ChatInterface } from '@/components/chat';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useKnowledgeGraph } from '@/hooks/useApi';
import { useGraphStore, ALL_ENTITY_TYPES, isNodeTypeVisible } from '@/stores/graphStore';
import { cn } from '@/lib/utils';

export function GraphPage() {
  const graphRef = useRef<KnowledgeGraphRef>(null);
  const [searchParams, setSearchParams] = useSearchParams();

  // Chat panel state for "Ask about this" feature
  // Research: ProPerSim - contextual entity discussion
  const [chatPanelOpen, setChatPanelOpen] = useState(false);
  const [chatContextEntity, setChatContextEntity] = useState<{ id: string; label: string } | null>(null);

  // Get graph data
  const { data } = useKnowledgeGraph();

  // Store state
  const {
    selectedNodeId,
    visibleNodeTypes,
    breathingEnabled,
    colorMode,
    highlightedNodeIds,
    setSelectedNode,
    toggleNodeType,
    setBreathingEnabled,
    toggleColorMode,
    showAllNodeTypes,
    setHighlightedNodes,
    clearHighlights,
  } = useGraphStore();

  // Track if we've already zoomed to highlighted node (to avoid repeating)
  const hasZoomedToHighlightRef = useRef(false);

  // Parse highlight/select params from URL on mount
  // ?select=xxx will select the node (open detail panel)
  // ?highlight=xxx will just highlight without selecting
  useEffect(() => {
    const selectParam = searchParams.get('select');
    const highlightParam = searchParams.get('highlight');

    if (selectParam) {
      // Decode first, then split by comma for multiple match attempts
      const decodedParam = decodeURIComponent(selectParam);
      const nodeIds = decodedParam.split(',').filter(id => id.trim().length > 0);
      console.log('[Graph] Setting nodes to select:', nodeIds);
      // Store as highlights temporarily - will be converted to selection when node is found
      setHighlightedNodes(nodeIds);
      hasZoomedToHighlightRef.current = false;
    } else if (highlightParam) {
      const decodedParam = decodeURIComponent(highlightParam);
      const nodeIds = decodedParam.split(',').filter(id => id.trim().length > 0);
      console.log('[Graph] Setting highlighted nodes:', nodeIds);
      setHighlightedNodes(nodeIds);
      hasZoomedToHighlightRef.current = false;
    }
  }, [searchParams, setHighlightedNodes]);

  // Check if we should select (not just highlight) the node
  const shouldSelectNode = searchParams.get('select') !== null;

  // Handle when graph is ready with highlighted node coordinates
  const handleHighlightReady = useCallback(
    (nodeId: string, x: number, y: number) => {
      // Only zoom once per highlight
      if (hasZoomedToHighlightRef.current) return;
      hasZoomedToHighlightRef.current = true;

      // If using ?select param, select the node (opens detail panel)
      if (shouldSelectNode) {
        setSelectedNode(nodeId);
        // Clear highlights since we're selecting instead
        clearHighlights();
        // Remove the select param from URL
        setSearchParams((params) => {
          params.delete('select');
          return params;
        });
      } else {
        // Just highlighting - update store to only contain matched node
        setHighlightedNodes([nodeId]);
      }

      if (graphRef.current) {
        graphRef.current.centerAt(x, y, 500);
        graphRef.current.zoom(1.5, 500);
      }
    },
    [shouldSelectNode, setSelectedNode, clearHighlights, setHighlightedNodes, setSearchParams]
  );

  // Clear highlights handler
  const handleClearHighlights = useCallback(() => {
    clearHighlights();
    // Remove highlight param from URL
    setSearchParams((params) => {
      params.delete('highlight');
      return params;
    });
  }, [clearHighlights, setSearchParams]);

  // Find selected node
  const selectedNode = data?.nodes.find((n) => n.id === selectedNodeId);

  // Check if emails are visible
  const emailsVisible = isNodeTypeVisible('Email', visibleNodeTypes);

  // Count emails in data for the badge
  const emailCount = useMemo(() => {
    return data?.nodes.filter((n) => n.node_type === 'Email').length ?? 0;
  }, [data?.nodes]);

  // Handle reset
  const handleReset = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  // Handle navigate to node (selects and focuses)
  const handleNavigateToNode = useCallback(
    (nodeId: string) => {
      setSelectedNode(nodeId);
      if (graphRef.current) {
        const node = data?.nodes.find((n) => n.id === nodeId);
        if (node && node.x !== undefined && node.y !== undefined) {
          graphRef.current.centerAt(node.x, node.y, 500);
          graphRef.current.zoom(2, 500);
        }
      }
    },
    [data?.nodes, setSelectedNode]
  );

  // Handle focus on node (centers view without changing selection)
  const handleFocusNode = useCallback(
    (nodeId: string) => {
      if (graphRef.current) {
        const node = data?.nodes.find((n) => n.id === nodeId);
        if (node && node.x !== undefined && node.y !== undefined) {
          graphRef.current.centerAt(node.x, node.y, 500);
          graphRef.current.zoom(2, 500);
        }
      }
    },
    [data?.nodes]
  );

  // Handle "Ask about this" - opens chat panel with entity context
  // Research: ProPerSim - proactive contextual AI conversation
  const handleAskAbout = useCallback((nodeId: string, nodeLabel: string) => {
    setChatContextEntity({ id: nodeId, label: nodeLabel });
    setChatPanelOpen(true);
  }, []);

  // Close chat panel
  const handleCloseChatPanel = useCallback(() => {
    setChatPanelOpen(false);
    // Keep context entity for potential re-open, clear after animation
    setTimeout(() => {
      if (!chatPanelOpen) setChatContextEntity(null);
    }, 300);
  }, [chatPanelOpen]);

  return (
    <div className="h-screen bg-[var(--color-bg-primary)] flex flex-col overflow-hidden">
      {/* Header - fixed, no scroll */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-[var(--color-border)]">
        {/* Left: Back + Title */}
        <div className="flex items-center gap-4">
          <Link to="/dashboard">
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-hover)]"
            >
              <ArrowLeft className="h-5 w-5" />
              <span className="sr-only">Back to Dashboard</span>
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <Network className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h1 className="text-lg font-brand tracking-wide text-[var(--color-text-primary)]">
              Knowledge Graph
            </h1>
          </div>
        </div>

        {/* Right: Filters */}
        <div className="flex items-center gap-4">
          {/* Highlighted nodes indicator */}
          {highlightedNodeIds.length > 0 && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 border border-blue-500/30 rounded">
              <span className="text-xs text-blue-400">
                {highlightedNodeIds.length} highlighted
              </span>
              <button
                onClick={handleClearHighlights}
                className="text-blue-400/60 hover:text-blue-400 transition-colors"
                title="Clear highlights"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          )}

          {/* Quick email toggle - prominent button */}
          {emailCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => toggleNodeType('Email')}
              className={cn(
                'h-8 gap-2 transition-colors',
                emailsVisible
                  ? 'border-amber-500/50 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20'
                  : 'border-[var(--color-border-hover)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]'
              )}
            >
              <Mail className="h-4 w-4" />
              {emailsVisible ? (
                <>
                  <span>Hide {emailCount} Emails</span>
                  <EyeOff className="h-3 w-3" />
                </>
              ) : (
                <>
                  <span>Show Emails</span>
                  <Eye className="h-3 w-3" />
                </>
              )}
            </Button>
          )}

          {/* Show all button */}
          {visibleNodeTypes.length > 0 && (
            <button
              onClick={showAllNodeTypes}
              className="px-2 py-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
            >
              Show all
            </button>
          )}

          {/* Node type filters */}
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-[var(--color-text-muted)]" />
            {ALL_ENTITY_TYPES.filter((t) => t !== 'Email').map((type) => {
              const isActive =
                visibleNodeTypes.length === 0 || visibleNodeTypes.includes(type);
              return (
                <button
                  key={type}
                  onClick={() => toggleNodeType(type)}
                  className={cn(
                    'px-2 py-1 text-xs border transition-colors',
                    isActive
                      ? 'border-[var(--color-border-active)] text-[var(--color-text-secondary)] bg-[var(--color-surface-hover)]'
                      : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-border-hover)]'
                  )}
                >
                  {type}
                </button>
              );
            })}
          </div>

          {/* Color mode toggle */}
          <button
            onClick={toggleColorMode}
            className={cn(
              'px-3 py-1 text-xs border transition-colors flex items-center gap-1.5',
              colorMode === 'colored'
                ? 'border-purple-500/50 text-purple-400 bg-purple-500/10'
                : 'border-[var(--color-border)] text-[var(--color-text-muted)]'
            )}
            title={colorMode === 'colored' ? 'Switch to monochrome' : 'Switch to colored'}
          >
            <Palette className="h-3 w-3" />
            {colorMode === 'colored' ? 'Colored' : 'Mono'}
          </button>

          {/* Breathing toggle */}
          <button
            onClick={() => setBreathingEnabled(!breathingEnabled)}
            className={cn(
              'px-3 py-1 text-xs border transition-colors',
              breathingEnabled
                ? 'border-[var(--color-border-active)] text-[var(--color-text-secondary)] bg-[var(--color-surface-hover)]'
                : 'border-[var(--color-border)] text-[var(--color-text-muted)]'
            )}
            title={breathingEnabled ? 'Disable breathing animation' : 'Enable breathing animation'}
          >
            {breathingEnabled ? 'Breathing: On' : 'Breathing: Off'}
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 relative flex">
        {/* Graph area */}
        <div className={cn('flex-1 relative', selectedNode && 'pr-80')}>
          <KnowledgeGraph
            ref={graphRef}
            className="h-full w-full"
            breathing={breathingEnabled}
            onHighlightReady={handleHighlightReady}
          />

          {/* Graph controls */}
          <GraphControls
            graphRef={graphRef}
            onReset={handleReset}
            graphData={data}
            className="absolute top-4 right-4"
          />

          {/* Stats */}
          <div className="absolute bottom-4 left-4 flex items-center gap-3">
            {data && (
              <>
                <Badge variant="outline" className="text-[var(--color-text-tertiary)] border-[var(--color-border-hover)]">
                  {data.nodes.length} nodes
                </Badge>
                <Badge variant="outline" className="text-[var(--color-text-tertiary)] border-[var(--color-border-hover)]">
                  {data.links.length} connections
                </Badge>
              </>
            )}
          </div>
        </div>

        {/* Node detail panel */}
        {selectedNode && data && (
          <NodeDetailPanel
            node={selectedNode}
            links={data.links}
            nodes={data.nodes}
            onClose={() => setSelectedNode(null)}
            onNavigateToNode={handleNavigateToNode}
            onFocusNode={handleFocusNode}
            onAskAbout={handleAskAbout}
            className="absolute top-0 right-0 h-full w-80"
          />
        )}

        {/* Chat panel for "Ask about this" feature
            Research: ProPerSim - contextual entity discussion
            Research: Causal-Copilot - natural language exploration */}
        {chatPanelOpen && (
          <div
            className={cn(
              'absolute top-0 right-0 h-full w-96',
              'bg-black border-l border-white/10',
              'animate-slide-in-right z-20'
            )}
          >
            <ChatInterface
              sessionId={`graph-entity-${chatContextEntity?.id ?? 'default'}`}
              contextEntityId={chatContextEntity?.id}
              onEntityClick={(entityId) => {
                // Navigate to clicked entity in graph
                handleNavigateToNode(entityId);
                handleCloseChatPanel();
              }}
              onClose={handleCloseChatPanel}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default GraphPage;

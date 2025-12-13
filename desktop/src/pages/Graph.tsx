/**
 * Graph Page - Full-screen knowledge graph exploration
 *
 * Dedicated route (/graph) for immersive graph exploration.
 * Provides full-screen visualization with controls and detail panel.
 */

import { useRef, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Network, Filter, Eye, EyeOff, Mail, Palette } from 'lucide-react';
import { KnowledgeGraph, type KnowledgeGraphRef } from '@/components/graph/KnowledgeGraph';
import { GraphControls } from '@/components/graph/GraphControls';
import { NodeDetailPanel } from '@/components/graph/NodeDetailPanel';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useKnowledgeGraph } from '@/hooks/useApi';
import { useGraphStore, ALL_ENTITY_TYPES, isNodeTypeVisible } from '@/stores/graphStore';
import { cn } from '@/lib/utils';

export function GraphPage() {
  const graphRef = useRef<KnowledgeGraphRef>(null);

  // Get graph data
  const { data } = useKnowledgeGraph();

  // Store state
  const {
    selectedNodeId,
    visibleNodeTypes,
    breathingEnabled,
    colorMode,
    setSelectedNode,
    toggleNodeType,
    setBreathingEnabled,
    toggleColorMode,
    showAllNodeTypes,
  } = useGraphStore();

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

  // Handle navigate to node
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
          />

          {/* Graph controls */}
          <GraphControls
            graphRef={graphRef}
            onReset={handleReset}
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
            className="absolute top-0 right-0 h-full w-80"
          />
        )}
      </div>
    </div>
  );
}

export default GraphPage;

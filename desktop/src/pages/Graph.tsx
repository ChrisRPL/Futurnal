/**
 * Graph Page - Full-screen knowledge graph exploration
 *
 * Dedicated route (/graph) for immersive graph exploration.
 * Provides full-screen visualization with controls and detail panel.
 */

import { useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Network, Filter } from 'lucide-react';
import { KnowledgeGraph, type KnowledgeGraphRef } from '@/components/graph/KnowledgeGraph';
import { GraphControls } from '@/components/graph/GraphControls';
import { NodeDetailPanel } from '@/components/graph/NodeDetailPanel';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useKnowledgeGraph } from '@/hooks/useApi';
import { useGraphStore, ALL_ENTITY_TYPES } from '@/stores/graphStore';
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
    setSelectedNode,
    toggleNodeType,
    setBreathingEnabled,
  } = useGraphStore();

  // Find selected node
  const selectedNode = data?.nodes.find((n) => n.id === selectedNodeId);

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
    <div className="h-screen bg-black flex flex-col overflow-hidden">
      {/* Header - fixed, no scroll */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-white/10">
        {/* Left: Back + Title */}
        <div className="flex items-center gap-4">
          <Link to="/dashboard">
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 text-white/60 hover:text-white hover:bg-white/10"
            >
              <ArrowLeft className="h-5 w-5" />
              <span className="sr-only">Back to Dashboard</span>
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <Network className="h-5 w-5 text-white/60" />
            <h1 className="text-lg font-brand tracking-wide text-white">
              Knowledge Graph
            </h1>
          </div>
        </div>

        {/* Right: Filters */}
        <div className="flex items-center gap-3">
          {/* Node type filters */}
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-white/40" />
            {ALL_ENTITY_TYPES.map((type) => {
              const isActive =
                visibleNodeTypes.length === 0 || visibleNodeTypes.includes(type);
              return (
                <button
                  key={type}
                  onClick={() => toggleNodeType(type)}
                  className={cn(
                    'px-2 py-1 text-xs border transition-colors',
                    isActive
                      ? 'border-white/30 text-white/80 bg-white/10'
                      : 'border-white/10 text-white/40 hover:border-white/20'
                  )}
                >
                  {type}
                </button>
              );
            })}
          </div>

          {/* Breathing toggle */}
          <button
            onClick={() => setBreathingEnabled(!breathingEnabled)}
            className={cn(
              'px-3 py-1 text-xs border transition-colors',
              breathingEnabled
                ? 'border-white/30 text-white/80 bg-white/10'
                : 'border-white/10 text-white/40'
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
                <Badge variant="outline" className="text-white/60 border-white/20">
                  {data.nodes.length} nodes
                </Badge>
                <Badge variant="outline" className="text-white/60 border-white/20">
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

/**
 * GraphFullscreen - Full-screen dialog wrapper for Knowledge Graph
 *
 * Provides a full-screen exploration mode with:
 * - Complete graph visualization
 * - Graph controls (zoom, pan, reset)
 * - Node detail panel on selection
 */

import { useRef, useCallback } from 'react';
import { Network } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { KnowledgeGraph } from './KnowledgeGraph';
import { GraphControls } from './GraphControls';
import { NodeDetailPanel } from './NodeDetailPanel';
import { useKnowledgeGraph } from '@/hooks/useApi';
import { useGraphStore } from '@/stores/graphStore';
import { cn } from '@/lib/utils';

interface GraphFullscreenProps {
  /** Controlled open state */
  open: boolean;
  /** Open state change handler */
  onOpenChange: (open: boolean) => void;
}

export function GraphFullscreen({ open, onOpenChange }: GraphFullscreenProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(undefined);

  // Get graph data for the detail panel
  const { data } = useKnowledgeGraph();

  // Store state
  const { selectedNodeId, setSelectedNode } = useGraphStore();

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

  // Handle close - also clear selection
  const handleOpenChange = useCallback(
    (newOpen: boolean) => {
      if (!newOpen) {
        setSelectedNode(null);
      }
      onOpenChange(newOpen);
    },
    [onOpenChange, setSelectedNode]
  );

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className={cn(
          'max-w-[95vw] max-h-[95vh] w-full h-[90vh]',
          'p-0 bg-black border-white/10 rounded-none',
          // Override default close button positioning
          '[&>[data-slot=dialog-close]]:top-5 [&>[data-slot=dialog-close]]:right-5'
        )}
      >
        {/* Header */}
        <DialogHeader className="absolute top-0 left-0 right-0 z-10 p-4 bg-gradient-to-b from-black to-transparent pointer-events-none">
          <DialogTitle className="flex items-center gap-2 text-white/90 pointer-events-auto">
            <Network className="h-5 w-5 text-white/60" />
            <span className="font-brand tracking-wide">Knowledge Graph</span>
          </DialogTitle>
        </DialogHeader>

        {/* Main content area */}
        <div className="relative h-full flex">
          {/* Graph area */}
          <div className={cn('flex-1 relative', selectedNode && 'pr-80')}>
            <KnowledgeGraph className="h-full w-full" breathing />

            {/* Graph controls */}
            <GraphControls
              graphRef={graphRef}
              onReset={handleReset}
              className="absolute top-16 right-4"
            />
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
      </DialogContent>
    </Dialog>
  );
}

export default GraphFullscreen;

/**
 * KnowledgeGraph - Main force-directed graph visualization component
 *
 * Renders an interactive Personal Knowledge Graph using react-force-graph-2d
 * with monochrome styling (white with opacity variants per node type).
 *
 * Features:
 * - Force-directed layout with physics simulation
 * - Monochrome node styling (no accent colors)
 * - Zoom, pan, and click-to-focus interactions
 * - "Breathing" animation for ambient liveness
 * - Level-of-detail based on zoom level
 */

import { useCallback, useRef, useEffect, useState, useMemo, forwardRef, useImperativeHandle } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceCollide } from 'd3-force';
import { useKnowledgeGraph } from '@/hooks/useApi';
import { useGraphStore, isNodeTypeVisible } from '@/stores/graphStore';
import { cn } from '@/lib/utils';
import type { GraphData, GraphNode, GraphLink, EntityType } from '@/types/api';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ForceGraphRef = any;

/** Methods exposed via ref */
export interface KnowledgeGraphRef {
  zoom: (k?: number, ms?: number) => number | undefined;
  zoomToFit: (ms?: number, padding?: number) => void;
  centerAt: (x?: number, y?: number, ms?: number) => void;
}

interface KnowledgeGraphProps {
  /** Additional CSS classes */
  className?: string;
  /** Mini mode for dashboard preview (disables most interactions) */
  miniMode?: boolean;
  /** Callback when user wants to expand (mini mode only) */
  onExpand?: () => void;
  /** Override graph data (for testing/stories) */
  data?: GraphData;
  /** Enable breathing animation (default: true, disabled in mini mode) */
  breathing?: boolean;
}

/**
 * Node opacity based on entity type - MONOCHROME ONLY
 * Brighter nodes = more important entity types
 */
const NODE_OPACITIES: Record<EntityType, number> = {
  Event: 1.0,      // white/100 - brightest (most important)
  Person: 0.8,     // white/80
  Document: 0.6,   // white/60
  Code: 0.5,       // white/50
  Concept: 0.4,    // white/40 - dimmest
};

/**
 * Node sizes based on entity type
 */
const NODE_SIZES: Record<EntityType, number> = {
  Event: 8,
  Person: 10,
  Document: 6,
  Code: 7,
  Concept: 9,
};

/**
 * Get level of detail based on zoom scale
 */
function getLevelOfDetail(scale: number): 'minimal' | 'medium' | 'full' {
  if (scale < 0.5) return 'minimal';
  if (scale > 1.5) return 'full';
  return 'medium';
}

export const KnowledgeGraph = forwardRef<KnowledgeGraphRef, KnowledgeGraphProps>(function KnowledgeGraph({
  className,
  miniMode = false,
  onExpand,
  data: overrideData,
  breathing: _breathing = true, // Reserved for future breathing animation
}, ref) {
  const graphRef = useRef<ForceGraphRef>(undefined);

  // Expose graph methods via ref
  useImperativeHandle(ref, () => ({
    zoom: (k?: number, ms?: number) => graphRef.current?.zoom(k, ms),
    zoomToFit: (ms?: number, padding?: number) => graphRef.current?.zoomToFit(ms, padding),
    centerAt: (x?: number, y?: number, ms?: number) => graphRef.current?.centerAt(x, y, ms),
  }), []);
  const containerRef = useRef<HTMLDivElement>(null);
  // Start with null dimensions - only render graph when we have real measurements
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  // Store state
  const {
    selectedNodeId,
    hoveredNodeId,
    visibleNodeTypes,
    setSelectedNode,
    setHoveredNode,
  } = useGraphStore();

  // Fetch graph data
  const { data: fetchedData, isLoading, error } = useKnowledgeGraph(
    miniMode ? 25 : undefined
  );

  // Use override data if provided, otherwise use fetched data
  const rawData = overrideData ?? fetchedData;

  // Filter data based on visible node types
  const filteredData = useMemo(() => {
    if (!rawData) return undefined;

    const nodes = rawData.nodes.filter((node) =>
      isNodeTypeVisible(node.node_type, visibleNodeTypes)
    );

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = rawData.links.filter(
      (link) =>
        nodeIds.has(typeof link.source === 'string' ? link.source : (link.source as GraphNode).id) &&
        nodeIds.has(typeof link.target === 'string' ? link.target : (link.target as GraphNode).id)
    );

    return { nodes, links };
  }, [rawData, visibleNodeTypes]);

  // Track if initial fit has been done - use state for proper reactivity
  const [hasInitialFit, setHasInitialFit] = useState(false);

  // Simplified dimension observer - let ResizeObserver do its job
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions((prev) => {
            // Reset fit state if dimensions changed significantly
            if (prev && (Math.abs(prev.width - width) > 50 || Math.abs(prev.height - height) > 50)) {
              setHasInitialFit(false);
            }
            return { width, height };
          });
        }
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Reset fit state when filtered data changes
  useEffect(() => {
    setHasInitialFit(false);
  }, [filteredData]);

  // Fallback timer - if onEngineStop never fires, force fit after 2 seconds
  useEffect(() => {
    if (!graphRef.current || !dimensions || hasInitialFit) return;

    const timeoutId = setTimeout(() => {
      if (!hasInitialFit && graphRef.current) {
        graphRef.current.zoomToFit(400, 40);
        setHasInitialFit(true);
      }
    }, 2000);

    return () => clearTimeout(timeoutId);
  }, [dimensions, hasInitialFit]);

  // Configure forces after mount
  useEffect(() => {
    if (!graphRef.current) return;

    // Strengthen repulsion to prevent clumping
    graphRef.current.d3Force('charge')?.strength(-200);

    // Add collision detection to prevent node overlaps
    graphRef.current.d3Force(
      'collide',
      forceCollide<GraphNode>((node) => (NODE_SIZES[node.node_type || 'Document'] || 6) + 5).iterations(2)
    );
  }, [filteredData]);

  // Handle engine stop - this is called when the force simulation stabilizes
  const handleEngineStop = useCallback(() => {
    if (!graphRef.current || hasInitialFit) return;

    // Fit graph to view - this automatically centers and zooms
    graphRef.current.zoomToFit(400, 40);
    setHasInitialFit(true);
  }, [hasInitialFit]);

  // Node click handler
  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      if (miniMode) {
        onExpand?.();
        return;
      }

      setSelectedNode(node.id);

      // Zoom to node
      if (graphRef.current) {
        graphRef.current.centerAt(node.x, node.y, 500);
        graphRef.current.zoom(2, 500);
      }
    },
    [miniMode, onExpand, setSelectedNode]
  );

  // Node hover handler
  const handleNodeHover = useCallback(
    (node: GraphNode | null) => {
      if (miniMode) return;
      setHoveredNode(node?.id ?? null);
    },
    [miniMode, setHoveredNode]
  );

  // Custom node rendering with monochrome styling
  const nodeCanvasObject = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const nodeType = node.node_type || 'Document';
      const size = NODE_SIZES[nodeType] || 6;
      const opacity = NODE_OPACITIES[nodeType] || 0.5;
      const isSelected = selectedNodeId === node.id;
      const isHovered = hoveredNodeId === node.id;
      const lod = getLevelOfDetail(globalScale);

      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Glow effect for selected/hovered nodes
      if ((isSelected || isHovered) && lod !== 'minimal') {
        ctx.beginPath();
        ctx.arc(x, y, size + 4, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.3})`;
        ctx.fill();
      }

      // Main node circle
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
      ctx.fill();

      // Border for selected nodes
      if (isSelected) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label (only when zoomed in enough or hovering)
      if (lod === 'full' || isHovered) {
        const label =
          node.label.length > 20 ? node.label.slice(0, 17) + '...' : node.label;

        ctx.font = `${11 / globalScale}px Inter, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
        ctx.fillText(label, x, y + size + 4);
      }
    },
    [selectedNodeId, hoveredNodeId]
  );

  // CRITICAL: nodePointerAreaPaint MUST draw the exact same shape as nodeCanvasObject
  // Using different sizes breaks click detection in react-force-graph-2d
  const nodePointerAreaPaint = useCallback(
    (node: GraphNode, color: string, ctx: CanvasRenderingContext2D) => {
      const nodeType = node.node_type || 'Document';
      const size = NODE_SIZES[nodeType] || 6;
      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Draw EXACT same circle as nodeCanvasObject - same radius
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI); // MUST match visual size
      ctx.fillStyle = color; // MUST use this color parameter for hit detection
      ctx.fill();
    },
    []
  );

  // Custom link rendering
  const linkCanvasObject = useCallback(
    (link: GraphLink, ctx: CanvasRenderingContext2D, globalScale: number) => {
      // After simulation, source/target become node objects (not strings)
      const source = link.source as unknown as GraphNode;
      const target = link.target as unknown as GraphNode;

      if (!source.x || !source.y || !target.x || !target.y) return;

      // Draw link line - improved visibility
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)'; // Increased from 0.1
      ctx.lineWidth = Math.max(1, (link.weight || 1) / globalScale);
      ctx.stroke();

      // Draw relationship label when zoomed in
      if (globalScale > 1.5 && link.relationship) {
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;

        ctx.font = `${10 / globalScale}px Inter, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'; // Increased from 0.3
        ctx.fillText(link.relationship, midX, midY);
      }
    },
    []
  );

  // Memoized link color to prevent re-initialization on every render
  const linkColor = useCallback(() => 'rgba(255, 255, 255, 0.25)', []);

  // Loading skeleton
  if (isLoading) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative flex items-center justify-center bg-black overflow-hidden',
          'border border-white/10',
          className
        )}
      >
        <GraphSkeleton />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative flex items-center justify-center bg-black',
          'border border-white/10',
          className
        )}
      >
        <div className="text-white/50 text-sm">Failed to load graph</div>
      </div>
    );
  }

  // Empty state
  if (!filteredData || filteredData.nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative flex items-center justify-center bg-black',
          'border border-white/10',
          className
        )}
      >
        <div className="text-center">
          <div className="text-white/50 text-sm">No data available</div>
          <div className="text-white/30 text-xs mt-1">
            Connect a data source to build your knowledge graph
          </div>
        </div>
      </div>
    );
  }

  // Waiting for dimensions - render container but not graph yet
  if (!dimensions) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'relative flex items-center justify-center bg-black overflow-hidden',
          'border border-white/10',
          className
        )}
      >
        <GraphSkeleton />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      data-slot="knowledge-graph"
      data-testid="knowledge-graph"
      className={cn(
        'relative bg-black overflow-hidden',
        'border border-white/10',
        miniMode && 'cursor-pointer',
        className
      )}
      onClick={miniMode ? onExpand : undefined}
    >
      {/* Graph Canvas - only rendered when we have valid dimensions */}
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={filteredData}
        nodeCanvasObject={nodeCanvasObject}
        nodePointerAreaPaint={nodePointerAreaPaint}
        linkCanvasObject={linkCanvasObject}
        nodeRelSize={6}
        linkWidth={1}
        linkColor={linkColor}
        backgroundColor="#000000"
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onEngineStop={handleEngineStop}
        warmupTicks={miniMode ? 50 : 100}
        cooldownTicks={miniMode ? 150 : 200}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        enableZoomInteraction={!miniMode}
        enablePanInteraction={!miniMode}
        enableNodeDrag={!miniMode}
      />

      {/* Mini mode expand hint */}
      {miniMode && (
        <div className="absolute bottom-2 right-2 text-xs text-white/40 bg-black/80 px-2 py-1">
          Click to expand
        </div>
      )}

      {/* Stats footer */}
      <div className="absolute bottom-2 left-2 text-xs text-white/40">
        {filteredData.nodes.length} memories · {filteredData.links.length} connections
      </div>
    </div>
  );
});

/**
 * Loading skeleton with animated placeholder nodes
 */
function GraphSkeleton() {
  return (
    <div className="relative w-full h-full">
      {/* Animated placeholder nodes */}
      {Array.from({ length: 12 }).map((_, i) => (
        <div
          key={i}
          className="absolute rounded-full bg-white/10 animate-pulse"
          style={{
            width: Math.random() * 12 + 6,
            height: Math.random() * 12 + 6,
            left: `${Math.random() * 80 + 10}%`,
            top: `${Math.random() * 80 + 10}%`,
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-white/40 text-sm">Loading knowledge graph...</span>
      </div>
    </div>
  );
}

export default KnowledgeGraph;

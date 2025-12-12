/**
 * KnowledgeGraph - Main force-directed graph visualization component
 *
 * Renders an interactive Personal Knowledge Graph using react-force-graph-2d
 * with semantic color coding by node type category.
 *
 * Features:
 * - Force-directed layout with physics simulation
 * - Semantic color scheme (origins=purple, content=blue, actors=teal, abstract=amber)
 * - Link styling by relationship type (contains=dashed, mentions=solid)
 * - Zoom, pan, and click-to-focus interactions
 * - "Breathing" animation for ambient liveness
 * - Level-of-detail based on zoom level
 * - Interactive legend
 */

import { useCallback, useRef, useEffect, useState, useMemo, forwardRef, useImperativeHandle } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceCollide, forceManyBody, forceX, forceY } from 'd3-force';
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
  /** Show legend (default: true in full mode) */
  showLegend?: boolean;
}

/**
 * Semantic color scheme for node types (dark-mode optimized)
 * Grouped by meaning:
 * - Data Origins (purple): Source, Mailbox - containers that hold content
 * - Content (blue): Document, Email - main knowledge artifacts
 * - Actors (teal/green): Person, Organization - human/entity elements
 * - Abstract (amber/orange): Concept, Event - ideas and happenings
 * - Technical (cyan): Code - stands out as technical
 */
const NODE_COLORS: Record<EntityType, string> = {
  // Sources - Purple (anchor points)
  Source: '#A855F7',      // Purple-500
  Mailbox: '#A855F7',     // Purple-500 (same as Source)

  // Content - Blue (main content)
  Document: '#3B82F6',    // Blue-500
  Email: '#3B82F6',       // Blue-500 (same as Document)

  // Actors - Teal (human/entity elements)
  Person: '#14B8A6',      // Teal-500
  Organization: '#14B8A6', // Teal-500 (same as Person)

  // Abstract - Amber (ideas and happenings)
  Concept: '#F59E0B',     // Amber-500
  Event: '#F59E0B',       // Amber-500 (same as Concept)

  // Code - Cyan (technical)
  Code: '#22D3EE',        // Cyan-400
};

/**
 * Node sizes by semantic role
 * Sources are largest (anchor points), entities are smallest (many of them)
 */
const NODE_SIZES: Record<EntityType, number> = {
  // Sources - Largest (22px anchor points)
  Source: 22,
  Mailbox: 22,

  // Content - Medium (14px main content)
  Document: 14,
  Email: 14,

  // Actors - Small (9-11px extracted knowledge)
  Person: 10,
  Organization: 11,

  // Abstract - Small (9-11px extracted knowledge)
  Concept: 9,
  Event: 10,

  // Code - Medium (content type)
  Code: 14,
};

/**
 * Node opacity for monochrome mode
 */
const NODE_OPACITIES: Record<EntityType, number> = {
  Source: 0.95,
  Mailbox: 0.95,
  Document: 0.7,
  Email: 0.65,
  Person: 0.8,
  Organization: 0.75,
  Concept: 0.6,
  Event: 0.7,
  Code: 0.65,
};

/**
 * Link colors by relationship type
 */
const LINK_STYLES: Record<string, { color: string; width: number; dashed: boolean }> = {
  contains: { color: 'rgba(168, 85, 247, 0.35)', width: 1, dashed: true },   // Purple, structural
  mentions: { color: 'rgba(59, 130, 246, 0.5)', width: 2, dashed: false },   // Blue, semantic
  default: { color: 'rgba(255, 255, 255, 0.25)', width: 1.5, dashed: false },
};

/**
 * Legend categories for display
 */
const LEGEND_CATEGORIES = [
  { label: 'Sources', types: ['Source', 'Mailbox'] as EntityType[], color: '#A855F7' },
  { label: 'Content', types: ['Document', 'Email'] as EntityType[], color: '#3B82F6' },
  { label: 'Actors', types: ['Person', 'Organization'] as EntityType[], color: '#14B8A6' },
  { label: 'Abstract', types: ['Concept', 'Event'] as EntityType[], color: '#F59E0B' },
  { label: 'Code', types: ['Code'] as EntityType[], color: '#22D3EE' },
];

/**
 * Get level of detail based on zoom scale
 * Thresholds lowered to show more detail at lower zoom levels
 */
function getLevelOfDetail(scale: number): 'minimal' | 'medium' | 'full' {
  if (scale < 0.2) return 'minimal';  // Only show dots when very zoomed out
  if (scale > 0.8) return 'full';     // Show labels earlier
  return 'medium';
}

export const KnowledgeGraph = forwardRef<KnowledgeGraphRef, KnowledgeGraphProps>(function KnowledgeGraph({
  className,
  miniMode = false,
  onExpand,
  data: overrideData,
  breathing = true,
  showLegend = true,
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
    colorMode,
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
  // IMPORTANT: Initialize nodes with RANDOM positions spread across canvas
  // Without this, d3-force places all nodes near center causing clustering
  const filteredData = useMemo(() => {
    if (!rawData) return undefined;

    // Use canvas dimensions or a default spread area
    const spreadWidth = dimensions?.width || 800;
    const spreadHeight = dimensions?.height || 600;

    // Create fresh node copies WITH random initial positions
    // This is critical - nodes must start spread apart for proper force layout
    const nodes = rawData.nodes
      .filter((node) => isNodeTypeVisible(node.node_type, visibleNodeTypes))
      .map((node, index) => {
        // Use deterministic but spread positions based on index
        // Distribute in a spiral pattern for even coverage
        const angle = index * 2.4; // Golden angle for even distribution
        const radius = Math.sqrt(index) * 30; // Spiral outward
        return {
          id: node.id,
          label: node.label,
          node_type: node.node_type,
          timestamp: node.timestamp,
          metadata: node.metadata,
          // Initialize with spread positions - CRUCIAL for force layout
          x: spreadWidth / 2 + radius * Math.cos(angle),
          y: spreadHeight / 2 + radius * Math.sin(angle),
        };
      });

    const nodeIds = new Set(nodes.map((n) => n.id));
    // Create fresh link copies as well
    const links = rawData.links
      .filter(
        (link) =>
          nodeIds.has(typeof link.source === 'string' ? link.source : (link.source as GraphNode).id) &&
          nodeIds.has(typeof link.target === 'string' ? link.target : (link.target as GraphNode).id)
      )
      .map((link) => ({
        source: typeof link.source === 'string' ? link.source : (link.source as GraphNode).id,
        target: typeof link.target === 'string' ? link.target : (link.target as GraphNode).id,
        relationship: link.relationship,
        weight: link.weight,
      }));

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

  // Reset fit state and reheat simulation when filtered data changes
  useEffect(() => {
    setHasInitialFit(false);

    // Reheat the simulation to restart force layout when data changes
    if (graphRef.current && filteredData && filteredData.nodes.length > 0) {
      // Small delay to ensure new data is applied first
      const timeoutId = setTimeout(() => {
        if (graphRef.current) {
          // Reheat simulation with high alpha to restart force calculations
          graphRef.current.d3ReheatSimulation();
        }
      }, 100);
      return () => clearTimeout(timeoutId);
    }
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

  // Configure forces after mount - balanced settings for spread without excessive shaking
  useEffect(() => {
    if (!graphRef.current) return;

    const nodeCount = filteredData?.nodes.length ?? 0;

    // Moderate repulsion - enough to spread but not shake violently
    const chargeStrength = nodeCount > 500 ? -800 : nodeCount > 100 ? -500 : -300;

    // Replace default charge force with tuned repulsion
    graphRef.current.d3Force('charge', forceManyBody()
      .strength(chargeStrength)
      .distanceMin(30)    // Larger min distance for smoother transitions
      .distanceMax(300)   // Reduced max to limit long-range effects
    );

    // REMOVE default center force - it causes clustering!
    graphRef.current.d3Force('center', null);

    // Very gentle centering forces
    const centerX = (dimensions?.width || 800) / 2;
    const centerY = (dimensions?.height || 600) / 2;
    graphRef.current.d3Force('x', forceX(centerX).strength(0.01));
    graphRef.current.d3Force('y', forceY(centerY).strength(0.01));

    // Add collision detection with moderate padding
    graphRef.current.d3Force(
      'collide',
      forceCollide<GraphNode>((node) => (NODE_SIZES[node.node_type || 'Document'] || 10) + 12).iterations(2)
    );

    // Moderate link distance for connected nodes
    const linkDist = nodeCount > 500 ? 120 : nodeCount > 100 ? 100 : 80;
    graphRef.current.d3Force('link')?.distance(linkDist);

    // Don't reheat - let warmup ticks handle initial positioning
    // This prevents double-animation on load
  }, [filteredData, dimensions]);

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

  // Custom node rendering with monochrome or colored styling
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

      // Determine node color based on colorMode
      const isColored = colorMode === 'colored';
      const nodeColor = isColored
        ? NODE_COLORS[nodeType] || '#6B7280'
        : `rgba(255, 255, 255, ${opacity})`;

      // Glow effect for selected/hovered nodes
      if ((isSelected || isHovered) && lod !== 'minimal') {
        ctx.beginPath();
        ctx.arc(x, y, size + 4, 0, 2 * Math.PI);
        if (isColored) {
          ctx.fillStyle = `${NODE_COLORS[nodeType] || '#6B7280'}40`; // 25% opacity
        } else {
          ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.3})`;
        }
        ctx.fill();
      }

      // Main node circle
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fillStyle = nodeColor;
      ctx.fill();

      // Border for selected nodes
      if (isSelected) {
        ctx.strokeStyle = isColored ? '#FFFFFF' : '#FFFFFF';
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
        ctx.fillStyle = isColored
          ? 'rgba(255, 255, 255, 0.9)'
          : `rgba(255, 255, 255, ${opacity})`;
        ctx.fillText(label, x, y + size + 4);
      }
    },
    [selectedNodeId, hoveredNodeId, colorMode]
  );

  // CRITICAL: nodePointerAreaPaint defines the clickable area
  // Use a larger area than visual size for easier clicking, especially when zoomed out
  const nodePointerAreaPaint = useCallback(
    (node: GraphNode, color: string, ctx: CanvasRenderingContext2D) => {
      const nodeType = node.node_type || 'Document';
      const visualSize = NODE_SIZES[nodeType] || 6;
      // Make hit area larger for easier clicking (minimum 12px radius)
      const hitSize = Math.max(visualSize, 12);
      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Draw larger hit area for better click detection
      ctx.beginPath();
      ctx.arc(x, y, hitSize, 0, 2 * Math.PI);
      ctx.fillStyle = color; // MUST use this color parameter for hit detection
      ctx.fill();
    },
    []
  );

  // Custom link rendering with relationship-based styling
  const linkCanvasObject = useCallback(
    (link: GraphLink, ctx: CanvasRenderingContext2D, globalScale: number) => {
      // After simulation, source/target become node objects (not strings)
      const source = link.source as unknown as GraphNode;
      const target = link.target as unknown as GraphNode;

      if (!source.x || !source.y || !target.x || !target.y) return;

      // Get link style based on relationship type
      const style = LINK_STYLES[link.relationship] || LINK_STYLES.default;
      const isColored = colorMode === 'colored';

      // Calculate line width (scale with zoom but maintain minimum visibility)
      const baseWidth = style.width * (link.weight || 1);
      const lineWidth = Math.max(0.5, baseWidth / Math.sqrt(globalScale));

      ctx.beginPath();

      // Draw dashed or solid line based on relationship type
      if (style.dashed) {
        ctx.setLineDash([4 / globalScale, 4 / globalScale]);
      } else {
        ctx.setLineDash([]);
      }

      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);

      // Use colored styles only in colored mode, otherwise use white with opacity
      ctx.strokeStyle = isColored ? style.color : 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = lineWidth;
      ctx.stroke();

      // Reset line dash
      ctx.setLineDash([]);

      // Draw relationship label when zoomed in
      if (globalScale > 1.5 && link.relationship) {
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;

        ctx.font = `${10 / globalScale}px Inter, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = isColored ? style.color.replace(/[\d.]+\)$/, '0.8)') : 'rgba(255, 255, 255, 0.5)';
        ctx.fillText(link.relationship, midX, midY);
      }
    },
    [colorMode]
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
        breathing && !miniMode && 'graph-breathing',
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
        warmupTicks={miniMode ? 100 : 300}
        cooldownTicks={miniMode ? 200 : 500}
        d3AlphaDecay={0.03}
        d3VelocityDecay={0.5}
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

      {/* Legend - only in colored mode and full view */}
      {showLegend && !miniMode && colorMode === 'colored' && (
        <GraphLegend />
      )}
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

/**
 * Graph legend showing node type categories and link types
 */
function GraphLegend() {
  return (
    <div className="absolute top-3 right-3 bg-black/80 backdrop-blur-sm border border-white/10 p-3 rounded-lg">
      {/* Node types */}
      <div className="text-[10px] text-white/50 uppercase tracking-wider mb-2">
        Node Types
      </div>
      <div className="space-y-1.5 mb-3">
        {LEGEND_CATEGORIES.map((category) => (
          <div key={category.label} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: category.color }}
            />
            <span className="text-xs text-white/70">{category.label}</span>
          </div>
        ))}
      </div>

      {/* Link types */}
      <div className="text-[10px] text-white/50 uppercase tracking-wider mb-2 pt-2 border-t border-white/10">
        Connections
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <div className="w-6 border-t-2 border-dashed" style={{ borderColor: 'rgba(168, 85, 247, 0.6)' }} />
          <span className="text-xs text-white/70">contains</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 border-t-2" style={{ borderColor: 'rgba(59, 130, 246, 0.7)' }} />
          <span className="text-xs text-white/70">mentions</span>
        </div>
      </div>
    </div>
  );
}

export default KnowledgeGraph;

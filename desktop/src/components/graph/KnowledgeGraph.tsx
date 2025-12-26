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
import { useKnowledgeGraph, useGraphStats } from '@/hooks/useApi';
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
  /** Callback when graph is ready to show highlighted nodes (with actual coordinates) */
  onHighlightReady?: (nodeId: string, x: number, y: number) => void;
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
  related_to: { color: 'rgba(16, 185, 129, 0.6)', width: 2, dashed: false }, // Green, causal (document-to-document)
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
  onHighlightReady,
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

  // Accessibility: track keyboard focus state and announcements
  const [keyboardFocusIndex, setKeyboardFocusIndex] = useState<number>(-1);
  const [announcement, setAnnouncement] = useState<string>('');
  const announcementTimeoutRef = useRef<number | undefined>(undefined);

  // Store state
  const {
    selectedNodeId,
    hoveredNodeId,
    visibleNodeTypes,
    colorMode,
    sourceFilter,
    confidenceRange,
    timeRange,
    highlightedNodeIds,
    bookmarkedNodeIds,
    setSelectedNode,
    setHoveredNode,
  } = useGraphStore();

  // Fetch graph data
  const { data: fetchedData, isLoading, error } = useKnowledgeGraph(
    miniMode ? 25 : undefined
  );

  // Fetch total stats for mini mode footer (shows total count, not just what's displayed)
  const { data: graphStats } = useGraphStats();

  // Use override data if provided, otherwise use fetched data
  const rawData = overrideData ?? fetchedData;

  // Filter data based on visible node types and advanced filters
  // IMPORTANT: Initialize nodes with RANDOM positions spread across canvas
  // Without this, d3-force places all nodes near center causing clustering
  // NOTE: We use fixed spread dimensions (not reactive to container size) to prevent
  // graphData from changing when container resizes (e.g., when detail panel opens).
  // This avoids resetting hasInitialFit and causing unwanted zoom-out.
  const filteredData = useMemo(() => {
    if (!rawData) return undefined;

    // Use fixed spread area for initial positions - force simulation will handle actual layout
    // This must NOT depend on dimensions to avoid re-creating graphData on resize
    const spreadWidth = 800;
    const spreadHeight = 600;

    // Parse time range dates once
    const startDate = timeRange.start ? new Date(timeRange.start) : null;
    const endDate = timeRange.end ? new Date(timeRange.end) : null;

    // Create fresh node copies WITH random initial positions
    // This is critical - nodes must start spread apart for proper force layout
    const nodes = rawData.nodes
      .filter((node) => {
        // Node type visibility filter
        if (!isNodeTypeVisible(node.node_type, visibleNodeTypes)) {
          return false;
        }

        // Source filter (if sources selected, only show nodes from those sources)
        if (sourceFilter.length > 0) {
          const nodeSource = node.metadata?.source as string | undefined;
          if (!nodeSource || !sourceFilter.some(s => nodeSource.includes(s))) {
            return false;
          }
        }

        // Time range filter
        if (node.timestamp) {
          const nodeDate = new Date(node.timestamp);
          if (startDate && nodeDate < startDate) return false;
          if (endDate && nodeDate > endDate) return false;
        } else if (startDate || endDate) {
          // If time filter is set but node has no timestamp, exclude it
          // (unless we want to show undated nodes - could add a toggle for this)
          return false;
        }

        return true;
      })
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

    // Create fresh link copies and filter by confidence
    const links = rawData.links
      .filter((link) => {
        // Ensure both source and target nodes exist
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id;
        const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id;
        if (!nodeIds.has(sourceId) || !nodeIds.has(targetId)) {
          return false;
        }

        // Confidence filter (for links that have confidence)
        if (link.confidence !== undefined) {
          if (link.confidence < confidenceRange[0] || link.confidence > confidenceRange[1]) {
            return false;
          }
        }

        return true;
      })
      .map((link) => ({
        source: typeof link.source === 'string' ? link.source : (link.source as GraphNode).id,
        target: typeof link.target === 'string' ? link.target : (link.target as GraphNode).id,
        relationship: link.relationship,
        weight: link.weight,
        confidence: link.confidence,
      }));

    return { nodes, links };
  }, [rawData, visibleNodeTypes, sourceFilter, confidenceRange, timeRange]);

  // Prepare graph data - use filteredData directly for force layout
  const graphData = useMemo<GraphData | undefined>(() => {
    if (!filteredData) {
      return undefined;
    }
    return filteredData;
  }, [filteredData]);

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
          setDimensions({ width, height });
        }
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Reset fit state and reheat simulation when data changes
  useEffect(() => {
    setHasInitialFit(false);

    if (graphRef.current && graphData && graphData.nodes.length > 0) {
      // Small delay to ensure new data is applied first
      const timeoutId = setTimeout(() => {
        if (graphRef.current) {
          // Reheat simulation with high alpha to restart force calculations
          graphRef.current.d3ReheatSimulation();
        }
      }, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [graphData]);

  // Fallback timer - if onEngineStop never fires, force fit after 2 seconds
  useEffect(() => {
    if (!graphRef.current || !dimensions || hasInitialFit) return;

    const timeoutId = setTimeout(() => {
      if (!hasInitialFit && graphRef.current) {
        const nodes = graphData?.nodes;
        if (nodes && nodes.length > 0) {
          // Always use zoomToFit to calculate appropriate zoom for current layout
          graphRef.current.zoomToFit(400, 50);
          setHasInitialFit(true);
        }
      }
    }, 2000);

    return () => clearTimeout(timeoutId);
  }, [dimensions, hasInitialFit, graphData]);

  // Configure forces after mount - balanced settings for spread without excessive shaking
  useEffect(() => {
    if (!graphRef.current) return;

    // Force layout configuration - increased distances for better readability
    const nodeCount = graphData?.nodes.length ?? 0;

    // Strong repulsion - spread nodes apart significantly
    const chargeStrength = nodeCount > 500 ? -1500 : nodeCount > 100 ? -1000 : -600;

    // Replace default charge force with strong repulsion for more spacing
    graphRef.current.d3Force('charge', forceManyBody()
      .strength(chargeStrength)
      .distanceMin(50)    // Increased min distance
      .distanceMax(500)   // Extended range for better spread
    );

    // REMOVE default center force - it causes clustering!
    graphRef.current.d3Force('center', null);

    // Very gentle centering forces
    const centerX = (dimensions?.width || 800) / 2;
    const centerY = (dimensions?.height || 600) / 2;
    graphRef.current.d3Force('x', forceX(centerX).strength(0.008));
    graphRef.current.d3Force('y', forceY(centerY).strength(0.008));

    // Add collision detection with larger padding for more spacing
    graphRef.current.d3Force(
      'collide',
      forceCollide<GraphNode>((node) => (NODE_SIZES[node.node_type || 'Document'] || 10) + 25).iterations(3)
    );

    // Increased link distance for more spread between connected nodes
    const linkDist = nodeCount > 500 ? 200 : nodeCount > 100 ? 160 : 120;
    graphRef.current.d3Force('link')?.distance(linkDist);
  }, [graphData, dimensions]);

  // Handle engine stop - this is called when the force simulation stabilizes
  const handleEngineStop = useCallback(() => {
    if (!graphRef.current) return;

    const graphNodes = graphData?.nodes;

    // Handle highlighted nodes - notify parent with actual coordinates
    // Use flexible ID matching to handle format differences (raw hash vs prefixed IDs)
    if (highlightedNodeIds.length > 0 && onHighlightReady && graphNodes) {
      const firstHighlightedNode = graphNodes.find(n => {
        if (n.x === undefined || n.y === undefined) return false;
        if (highlightedNodeIds.includes(n.id)) return true;
        const nodeIdParts = n.id.split(':');
        if (nodeIdParts.length === 2 && highlightedNodeIds.includes(nodeIdParts[1])) return true;
        return highlightedNodeIds.some(hid => n.id.includes(hid) || hid.includes(n.id));
      });
      if (firstHighlightedNode && firstHighlightedNode.x !== undefined && firstHighlightedNode.y !== undefined) {
        onHighlightReady(firstHighlightedNode.id, firstHighlightedNode.x, firstHighlightedNode.y);
        // Skip zoomToFit when we have highlights - let the highlight handler control zoom
        setHasInitialFit(true);
        return;
      }
    }

    // Fit graph to view on initial load (only if no highlights)
    if (!hasInitialFit && graphNodes && graphNodes.length > 0) {
      graphRef.current.zoomToFit(400, 50);
      setHasInitialFit(true);
    }
  }, [graphData, hasInitialFit, highlightedNodeIds, onHighlightReady]);

  // Handle highlighted nodes changes - zoom to highlighted node when IDs change
  // This is separate from handleEngineStop because we need to react to highlight changes
  // even after the simulation has already stopped
  useEffect(() => {
    if (!graphRef.current || !graphData?.nodes || highlightedNodeIds.length === 0 || !onHighlightReady) {
      return;
    }

    // Debug: Log what we're looking for vs what's available
    console.log('[KnowledgeGraph] Highlighting - Looking for IDs:', highlightedNodeIds);
    console.log('[KnowledgeGraph] Available graph nodes (first 5):', graphData.nodes.slice(0, 5).map(n => ({
      id: n.id,
      label: n.label,
      metadata: n.metadata
    })));

    // Find the first highlighted node with valid coordinates
    // Handle ID format mismatch: search results use raw hashes, graph uses prefixed IDs (doc:xxx, source:xxx)
    const highlightedNode = graphData.nodes.find(n => {
      if (n.x === undefined || n.y === undefined) return false;

      // Check exact match first
      if (highlightedNodeIds.includes(n.id)) {
        console.log('[KnowledgeGraph] Found exact match:', n.id);
        return true;
      }

      // Check if any highlighted ID matches the hash part of the node ID (after the prefix)
      // Graph IDs are like "doc:abc123" or "source:xyz789"
      const nodeIdParts = n.id.split(':');
      if (nodeIdParts.length === 2) {
        const nodeHash = nodeIdParts[1];
        if (highlightedNodeIds.includes(nodeHash)) {
          console.log('[KnowledgeGraph] Found hash match:', n.id, 'matches', nodeHash);
          return true;
        }
      }

      // Check if highlighted ID matches the node label (partial match for filenames)
      // Require minimum 4 chars to avoid false matches with short names
      // Also skip Source/Mailbox nodes for label matching - prefer content nodes
      if (n.node_type !== 'Source' && n.node_type !== 'Mailbox') {
        const labelMatch = highlightedNodeIds.some(hid => {
          if (hid.length < 4) return false; // Skip short search terms
          const label = n.label.toLowerCase();
          const searchTerm = hid.toLowerCase();
          return label.includes(searchTerm) || searchTerm.includes(label);
        });
        if (labelMatch) {
          console.log('[KnowledgeGraph] Found label match:', n.id, n.label);
          return true;
        }
      }

      // Check if highlighted ID matches filename in metadata
      const nodeFilename = (n.metadata?.details as Record<string, unknown>)?.filename as string | undefined
        || (n.metadata?.filename as string | undefined);
      if (nodeFilename) {
        const filenameMatch = highlightedNodeIds.some(hid => {
          const fn = nodeFilename.toLowerCase();
          const searchTerm = hid.toLowerCase();
          return fn.includes(searchTerm) || searchTerm.includes(fn);
        });
        if (filenameMatch) {
          console.log('[KnowledgeGraph] Found filename match:', n.id, nodeFilename);
          return true;
        }
      }

      return false;
    });

    if (highlightedNode) {
      console.log('[KnowledgeGraph] Will highlight node:', highlightedNode.id, highlightedNode.label);
      // Small delay to ensure graph is rendered
      const timeoutId = setTimeout(() => {
        if (graphRef.current && highlightedNode.x !== undefined && highlightedNode.y !== undefined) {
          onHighlightReady(highlightedNode.id, highlightedNode.x, highlightedNode.y);
        }
      }, 300);

      return () => clearTimeout(timeoutId);
    } else {
      console.log('[KnowledgeGraph] No matching node found for highlights');
    }
  }, [highlightedNodeIds, graphData?.nodes, onHighlightReady]);

  // Node click handler
  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      if (miniMode) {
        onExpand?.();
        return;
      }

      setSelectedNode(node.id);

      // Zoom to node - only if coordinates are valid
      if (graphRef.current &&
          node.x !== undefined && Number.isFinite(node.x) &&
          node.y !== undefined && Number.isFinite(node.y)) {
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

  // Announce to screen readers
  const announce = useCallback((message: string) => {
    if (announcementTimeoutRef.current) {
      clearTimeout(announcementTimeoutRef.current);
    }
    setAnnouncement(message);
    announcementTimeoutRef.current = window.setTimeout(() => {
      setAnnouncement('');
    }, 1000);
  }, []);

  // Keyboard navigation handler
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (miniMode || !graphData || graphData.nodes.length === 0) return;

      const nodes = graphData.nodes;
      let newIndex = keyboardFocusIndex;

      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          e.preventDefault();
          newIndex = keyboardFocusIndex < 0 ? 0 : Math.min(keyboardFocusIndex + 1, nodes.length - 1);
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          e.preventDefault();
          newIndex = keyboardFocusIndex < 0 ? nodes.length - 1 : Math.max(keyboardFocusIndex - 1, 0);
          break;
        case 'Home':
          e.preventDefault();
          newIndex = 0;
          break;
        case 'End':
          e.preventDefault();
          newIndex = nodes.length - 1;
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (keyboardFocusIndex >= 0 && keyboardFocusIndex < nodes.length) {
            const node = nodes[keyboardFocusIndex];
            setSelectedNode(node.id);
            if (graphRef.current && node.x !== undefined && node.y !== undefined) {
              graphRef.current.centerAt(node.x, node.y, 500);
              graphRef.current.zoom(2, 500);
            }
            announce(`Selected ${node.label}, ${node.node_type}`);
          }
          return;
        case 'Escape':
          e.preventDefault();
          setSelectedNode(null);
          setKeyboardFocusIndex(-1);
          announce('Selection cleared');
          return;
        default:
          return;
      }

      if (newIndex !== keyboardFocusIndex && newIndex >= 0 && newIndex < nodes.length) {
        setKeyboardFocusIndex(newIndex);
        const node = nodes[newIndex];

        // Center view on keyboard-focused node
        if (graphRef.current && node.x !== undefined && node.y !== undefined) {
          graphRef.current.centerAt(node.x, node.y, 300);
        }

        // Announce to screen reader
        announce(`${node.label}, ${node.node_type}, ${newIndex + 1} of ${nodes.length}`);
      }
    },
    [miniMode, graphData, keyboardFocusIndex, setSelectedNode, announce]
  );

  // Reset keyboard focus when data changes
  useEffect(() => {
    setKeyboardFocusIndex(-1);
  }, [graphData]);

  // Get keyboard-focused node
  const keyboardFocusedNodeId = useMemo(() => {
    if (keyboardFocusIndex < 0 || !graphData) return null;
    return graphData.nodes[keyboardFocusIndex]?.id ?? null;
  }, [keyboardFocusIndex, graphData]);

  // Custom node rendering with monochrome or colored styling
  const nodeCanvasObject = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const nodeType = node.node_type || 'Document';
      const size = NODE_SIZES[nodeType] || 6;
      const opacity = NODE_OPACITIES[nodeType] || 0.5;
      const isSelected = selectedNodeId === node.id;
      const isHovered = hoveredNodeId === node.id;
      const isKeyboardFocused = keyboardFocusedNodeId === node.id;
      const isBookmarked = bookmarkedNodeIds.includes(node.id);
      // Flexible ID matching for highlighting - handle format differences (raw hash vs prefixed IDs)
      // Also match by label and filename for better cross-system matching
      const isHighlighted = highlightedNodeIds.length > 0 && (
        highlightedNodeIds.includes(node.id) ||
        (node.id.includes(':') && highlightedNodeIds.includes(node.id.split(':')[1])) ||
        highlightedNodeIds.some(hid => node.id.includes(hid) || hid.includes(node.id)) ||
        // Match by label (case-insensitive partial match) - skip Source nodes, require min length
        (nodeType !== 'Source' && nodeType !== 'Mailbox' && highlightedNodeIds.some(hid => {
          if (hid.length < 4) return false;
          const label = node.label.toLowerCase();
          const searchTerm = hid.toLowerCase();
          return label.includes(searchTerm) || searchTerm.includes(label);
        }))
      );
      const lod = getLevelOfDetail(globalScale);

      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Determine node color based on colorMode
      const isColored = colorMode === 'colored';
      const nodeColor = isColored
        ? NODE_COLORS[nodeType] || '#6B7280'
        : `rgba(255, 255, 255, ${opacity})`;

      // Keyboard focus ring (accessibility)
      if (isKeyboardFocused && lod !== 'minimal') {
        ctx.beginPath();
        ctx.arc(x, y, size + 6, 0, 2 * Math.PI);
        ctx.strokeStyle = '#FFFFFF'; // White focus ring
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 2]); // Dashed for visibility
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Highlighted glow effect (from search results)
      if (isHighlighted && lod !== 'minimal') {
        ctx.beginPath();
        ctx.arc(x, y, size + 8, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'; // Blue glow
        ctx.fill();
      }

      // Glow effect for selected/hovered nodes
      if ((isSelected || isHovered || isKeyboardFocused) && lod !== 'minimal') {
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

      // Bookmark indicator (small star at top-right of node)
      if (isBookmarked && lod !== 'minimal') {
        const starX = x + size * 0.7;
        const starY = y - size * 0.7;
        const starSize = Math.max(4, size * 0.4);

        // Draw star shape
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
          const angle = (i * 4 * Math.PI) / 5 - Math.PI / 2;
          const r = i % 2 === 0 ? starSize : starSize * 0.5;
          const px = starX + r * Math.cos(angle);
          const py = starY + r * Math.sin(angle);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fillStyle = '#FBBF24'; // Amber-400
        ctx.fill();
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
    [selectedNodeId, hoveredNodeId, keyboardFocusedNodeId, colorMode, bookmarkedNodeIds, highlightedNodeIds]
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
      // But during early simulation ticks, they might still be string IDs
      const source = link.source as unknown as GraphNode;
      const target = link.target as unknown as GraphNode;

      // More robust coordinate check - handle both string IDs and node objects
      const sourceX = typeof source === 'object' && source !== null ? source.x : undefined;
      const sourceY = typeof source === 'object' && source !== null ? source.y : undefined;
      const targetX = typeof target === 'object' && target !== null ? target.x : undefined;
      const targetY = typeof target === 'object' && target !== null ? target.y : undefined;

      if (sourceX === undefined || sourceY === undefined || targetX === undefined || targetY === undefined) return;

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

      ctx.moveTo(sourceX, sourceY);
      ctx.lineTo(targetX, targetY);

      // Use colored styles only in colored mode, otherwise use white with opacity
      ctx.strokeStyle = isColored ? style.color : 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = lineWidth;
      ctx.stroke();

      // Reset line dash
      ctx.setLineDash([]);

      // Draw relationship label when zoomed in
      if (globalScale > 1.5 && link.relationship) {
        const midX = (sourceX + targetX) / 2;
        const midY = (sourceY + targetY) / 2;

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
      role="application"
      aria-label={`Knowledge graph with ${graphData?.nodes.length ?? 0} nodes and ${graphData?.links.length ?? 0} connections. Use arrow keys to navigate nodes, Enter to select, Escape to clear selection.`}
      aria-roledescription="interactive knowledge graph"
      tabIndex={miniMode ? undefined : 0}
      className={cn(
        'relative bg-black overflow-hidden',
        'border border-white/10',
        miniMode && 'cursor-pointer',
        breathing && !miniMode && 'graph-breathing',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-inset',
        className
      )}
      onClick={miniMode ? onExpand : undefined}
      onKeyDown={!miniMode ? handleKeyDown : undefined}
    >
      {/* Screen reader live region for announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcement}
      </div>
      {/* Graph Canvas - force-directed layout */}
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData ?? { nodes: [], links: [] }}
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

      {/* Stats footer - show total stats in mini mode, displayed stats in full mode */}
      <div className="absolute bottom-2 left-2 text-xs text-white/40">
        {miniMode
          ? `${graphStats?.total_nodes ?? 0} memories · ${graphStats?.total_links ?? 0} connections`
          : `${graphData?.nodes.length ?? 0} memories · ${graphData?.links.length ?? 0} connections`
        }
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
    <div className="absolute top-3 right-14 bg-black/80 backdrop-blur-sm border border-white/10 p-3 rounded-lg">
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
        <div className="flex items-center gap-2">
          <div className="w-6 border-t-2" style={{ borderColor: 'rgba(16, 185, 129, 0.8)' }} />
          <span className="text-xs text-white/70">related to</span>
        </div>
      </div>
    </div>
  );
}

export default KnowledgeGraph;

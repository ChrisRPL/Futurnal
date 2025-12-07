Summary: Implement interactive Personal Knowledge Graph visualization with force-directed layout and node clustering.

# 07 · Knowledge Graph Visualization

## Purpose

Create an interactive, visually stunning Personal Knowledge Graph (PKG) visualization—the visual manifestation of your Ghost's evolving memory. This component serves as the centerpiece of the Futurnal dashboard, allowing users to see their personal universe as their Ghost perceives it: a living network of entities, relationships, and experiential connections.

> **"Your experiential network, made visible"** — The PKG visualization transforms abstract data relationships into tangible, explorable insights. This is where users witness their Ghost's understanding take shape.

This component enables users to explore knowledge connections, discover relationships between entities, and develop intuition about patterns in their stream of experience.

**Criticality**: CRITICAL - Core differentiating feature and visual showcase

## Scope

- Force-directed graph layout with react-force-graph-2d
- Node types with distinct visual styles (Events, Documents, People, Code)
- Edge visualization for relationships with semantic labels
- Zoom, pan, and click-to-focus interactions
- "Breathing" animation for liveness indication
- Node clustering for large graphs (1000+ nodes)
- Detail panel on node selection
- Performance optimization for 60fps rendering
- Mini-view on dashboard with expand capability
- Full-screen graph exploration mode

## Requirements Alignment

- **Feature Requirement**: "Knowledge Graph mini-view on Dashboard"
- **Design Philosophy**: "Breathtaking, special, perfect" visualization
- **Performance Target**: 60fps with 1000+ nodes
- **User Request**: "Graph visualization priority - deliver it right now"

## Component Design

### Graph Container Component

```tsx
// src/components/graph/KnowledgeGraph.tsx
import { useCallback, useRef, useEffect, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods } from 'react-force-graph-2d';
import { useGraphData } from '@/hooks/useGraphData';
import { GraphControls } from './GraphControls';
import { NodeDetailPanel } from './NodeDetailPanel';
import { cn } from '@/lib/utils';

interface GraphNode {
  id: string;
  label: string;
  type: 'Event' | 'Document' | 'Person' | 'Code' | 'Concept';
  timestamp?: string;
  metadata?: Record<string, unknown>;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface GraphLink {
  source: string;
  target: string;
  relationship: string;
  weight?: number;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

interface KnowledgeGraphProps {
  className?: string;
  miniMode?: boolean;
  onNodeClick?: (node: GraphNode) => void;
  onExpand?: () => void;
}

// Node type colors from FRONTEND_DESIGN.md
const NODE_COLORS: Record<string, string> = {
  Event: '#3B82F6',      // Primary Blue (Ghost)
  Document: '#A0A0A0',   // Secondary Gray
  Person: '#10B981',     // Emerald Green (Animal)
  Code: '#8B5CF6',       // Violet (Insights)
  Concept: '#F59E0B',    // Warning Yellow
};

const NODE_SIZES: Record<string, number> = {
  Event: 8,
  Document: 6,
  Person: 10,
  Code: 7,
  Concept: 9,
};

export function KnowledgeGraph({
  className,
  miniMode = false,
  onNodeClick,
  onExpand,
}: KnowledgeGraphProps) {
  const graphRef = useRef<ForceGraphMethods>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  const { data, isLoading, error } = useGraphData();

  // Responsive dimensions
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Breathing animation for ambient liveness
  useEffect(() => {
    if (!graphRef.current || miniMode) return;

    const breathe = () => {
      graphRef.current?.d3Force('charge')?.strength((node: any) => {
        const baseStrength = -100;
        const breathFactor = 1 + Math.sin(Date.now() / 2000) * 0.05;
        return baseStrength * breathFactor;
      });
      graphRef.current?.d3ReheatSimulation();
    };

    const interval = setInterval(breathe, 100);
    return () => clearInterval(interval);
  }, [miniMode]);

  // Node click handler
  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      setSelectedNode(node);
      onNodeClick?.(node);

      // Zoom to node
      if (graphRef.current && !miniMode) {
        graphRef.current.centerAt(node.x, node.y, 500);
        graphRef.current.zoom(2, 500);
      }
    },
    [onNodeClick, miniMode]
  );

  // Custom node rendering
  const nodeCanvasObject = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const size = NODE_SIZES[node.type] || 6;
      const color = NODE_COLORS[node.type] || '#666666';
      const isSelected = selectedNode?.id === node.id;
      const isHovered = hoveredNode?.id === node.id;

      // Glow effect for selected/hovered
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, size + 4, 0, 2 * Math.PI);
        ctx.fillStyle = `${color}40`;
        ctx.fill();
      }

      // Main node circle
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, size, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // Border
      ctx.strokeStyle = isSelected ? '#FFFFFF' : `${color}80`;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      // Label (only when zoomed in or hovered)
      if (globalScale > 1.5 || isHovered) {
        const label = node.label.length > 20
          ? node.label.slice(0, 17) + '...'
          : node.label;

        ctx.font = `${11 / globalScale}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = '#EDEDED';
        ctx.fillText(label, node.x!, node.y! + size + 4);
      }
    },
    [selectedNode, hoveredNode]
  );

  // Custom link rendering
  const linkCanvasObject = useCallback(
    (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const start = link.source;
      const end = link.target;

      // Draw link line
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.strokeStyle = '#33333380';
      ctx.lineWidth = Math.max(0.5, (link.weight || 1) / globalScale);
      ctx.stroke();

      // Draw relationship label when zoomed in
      if (globalScale > 2 && link.relationship) {
        const midX = (start.x + end.x) / 2;
        const midY = (start.y + end.y) / 2;

        ctx.font = `${9 / globalScale}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#666666';
        ctx.fillText(link.relationship, midX, midY);
      }
    },
    []
  );

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center bg-background-deep', className)}>
        <GraphSkeleton />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex items-center justify-center bg-background-deep', className)}>
        <div className="text-text-tertiary text-sm">Failed to load graph</div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative bg-background-deep rounded-lg overflow-hidden',
        'border border-border',
        miniMode && 'cursor-pointer',
        className
      )}
      onClick={miniMode ? onExpand : undefined}
    >
      {/* Graph Canvas */}
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={data}
        nodeCanvasObject={nodeCanvasObject}
        linkCanvasObject={linkCanvasObject}
        nodeRelSize={6}
        linkWidth={1}
        linkColor={() => '#333333'}
        backgroundColor="#0A0A0A"
        onNodeClick={handleNodeClick}
        onNodeHover={setHoveredNode}
        cooldownTicks={miniMode ? 50 : 100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        enableZoomInteraction={!miniMode}
        enablePanInteraction={!miniMode}
        enableNodeDrag={!miniMode}
      />

      {/* Controls (non-mini mode only) */}
      {!miniMode && (
        <GraphControls
          graphRef={graphRef}
          onReset={() => {
            graphRef.current?.zoomToFit(400, 50);
            setSelectedNode(null);
          }}
          className="absolute top-4 right-4"
        />
      )}

      {/* Mini mode expand hint */}
      {miniMode && (
        <div className="absolute bottom-2 right-2 text-xs text-text-tertiary bg-background-surface/80 px-2 py-1 rounded">
          Click to expand
        </div>
      )}

      {/* Experiential network stats */}
      <div className="absolute bottom-2 left-2 text-xs text-text-tertiary">
        {data?.nodes.length || 0} memories · {data?.links.length || 0} connections
      </div>

      {/* Node Detail Panel (slide-in) */}
      {selectedNode && !miniMode && (
        <NodeDetailPanel
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
          className="absolute top-0 right-0 h-full w-80"
        />
      )}
    </div>
  );
}

// Loading skeleton
function GraphSkeleton() {
  return (
    <div className="relative w-full h-full">
      {/* Animated placeholder nodes */}
      {Array.from({ length: 12 }).map((_, i) => (
        <div
          key={i}
          className="absolute rounded-full bg-border animate-pulse"
          style={{
            width: Math.random() * 20 + 10,
            height: Math.random() * 20 + 10,
            left: `${Math.random() * 80 + 10}%`,
            top: `${Math.random() * 80 + 10}%`,
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-text-tertiary text-sm">Loading your Ghost's memory...</span>
      </div>
    </div>
  );
}
```

### Graph Controls Component

```tsx
// src/components/graph/GraphControls.tsx
import { Button } from '@/components/ui/button';
import { ZoomIn, ZoomOut, Maximize, RotateCcw, Focus } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ForceGraphMethods } from 'react-force-graph-2d';

interface GraphControlsProps {
  graphRef: React.RefObject<ForceGraphMethods>;
  onReset?: () => void;
  className?: string;
}

export function GraphControls({ graphRef, onReset, className }: GraphControlsProps) {
  const handleZoomIn = () => {
    const currentZoom = graphRef.current?.zoom() || 1;
    graphRef.current?.zoom(currentZoom * 1.5, 300);
  };

  const handleZoomOut = () => {
    const currentZoom = graphRef.current?.zoom() || 1;
    graphRef.current?.zoom(currentZoom / 1.5, 300);
  };

  const handleFit = () => {
    graphRef.current?.zoomToFit(400, 50);
  };

  return (
    <div
      className={cn(
        'flex flex-col gap-1 p-1 rounded-lg bg-background-surface/90 backdrop-blur-sm border border-border',
        className
      )}
    >
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={handleZoomIn}
        title="Zoom in"
      >
        <ZoomIn className="h-4 w-4" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={handleZoomOut}
        title="Zoom out"
      >
        <ZoomOut className="h-4 w-4" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={handleFit}
        title="Fit to view"
      >
        <Maximize className="h-4 w-4" />
      </Button>
      <div className="w-full h-px bg-border my-1" />
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={onReset}
        title="Reset view"
      >
        <RotateCcw className="h-4 w-4" />
      </Button>
    </div>
  );
}
```

### Node Detail Panel Component

```tsx
// src/components/graph/NodeDetailPanel.tsx
import { X, Calendar, FileText, User, Code, Link2, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';

interface GraphNode {
  id: string;
  label: string;
  type: 'Event' | 'Document' | 'Person' | 'Code' | 'Concept';
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

interface NodeDetailPanelProps {
  node: GraphNode;
  onClose: () => void;
  className?: string;
}

const TYPE_ICONS = {
  Event: Calendar,
  Document: FileText,
  Person: User,
  Code: Code,
  Concept: Link2,
};

const TYPE_COLORS = {
  Event: 'default',
  Document: 'secondary',
  Person: 'success',
  Code: 'accent',
  Concept: 'warning',
};

export function NodeDetailPanel({ node, onClose, className }: NodeDetailPanelProps) {
  const Icon = TYPE_ICONS[node.type] || FileText;

  return (
    <div
      className={cn(
        'bg-background-surface border-l border-border',
        'animate-slide-in-right',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between p-4 border-b border-border">
        <div className="flex items-start gap-3">
          <div
            className={cn(
              'p-2 rounded-lg',
              node.type === 'Event' && 'bg-primary/20',
              node.type === 'Document' && 'bg-muted',
              node.type === 'Person' && 'bg-secondary/20',
              node.type === 'Code' && 'bg-accent/20',
              node.type === 'Concept' && 'bg-warning/20'
            )}
          >
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-medium text-text-primary">{node.label}</h3>
            <Badge variant={TYPE_COLORS[node.type] as any} className="mt-1">
              {node.type}
            </Badge>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <ScrollArea className="h-[calc(100%-5rem)]">
        <div className="p-4 space-y-4">
          {/* Timestamp */}
          {node.timestamp && (
            <div>
              <div className="text-xs text-text-tertiary mb-1">Timestamp</div>
              <div className="text-sm text-text-secondary">
                {new Date(node.timestamp).toLocaleString()}
              </div>
            </div>
          )}

          {/* Metadata */}
          {node.metadata && Object.keys(node.metadata).length > 0 && (
            <div>
              <div className="text-xs text-text-tertiary mb-2">Properties</div>
              <div className="space-y-2">
                {Object.entries(node.metadata).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-text-tertiary">{key}</span>
                    <span className="text-text-secondary truncate ml-2">
                      {String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connections section placeholder */}
          <div>
            <div className="text-xs text-text-tertiary mb-2">Connections</div>
            <div className="text-sm text-text-secondary">
              View related nodes and relationships
            </div>
          </div>

          {/* Actions */}
          <div className="pt-4 border-t border-border">
            <Button variant="outline" size="sm" className="w-full gap-2">
              <ExternalLink className="h-4 w-4" />
              Open Source
            </Button>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
```

### Graph Data Hook

```tsx
// src/hooks/useGraphData.ts
import { useQuery } from '@tanstack/react-query';
import { invoke } from '@tauri-apps/api/core';

interface GraphNode {
  id: string;
  label: string;
  type: 'Event' | 'Document' | 'Person' | 'Code' | 'Concept';
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

interface GraphLink {
  source: string;
  target: string;
  relationship: string;
  weight?: number;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

async function fetchGraphData(): Promise<GraphData> {
  try {
    const result = await invoke<GraphData>('get_knowledge_graph');
    return result;
  } catch (error) {
    console.error('Failed to fetch graph data:', error);
    throw error;
  }
}

export function useGraphData(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['knowledgeGraph'],
    queryFn: fetchGraphData,
    staleTime: 1000 * 60 * 5, // 5 minutes
    refetchOnWindowFocus: false,
    enabled: options?.enabled !== false,
  });
}

// Hook for filtered graph data
export function useFilteredGraphData(filters: {
  nodeTypes?: string[];
  dateRange?: { start: Date; end: Date };
  searchQuery?: string;
}) {
  const { data, ...rest } = useGraphData();

  const filteredData = data ? filterGraphData(data, filters) : undefined;

  return { data: filteredData, ...rest };
}

function filterGraphData(
  data: GraphData,
  filters: {
    nodeTypes?: string[];
    dateRange?: { start: Date; end: Date };
    searchQuery?: string;
  }
): GraphData {
  let nodes = [...data.nodes];

  // Filter by node types
  if (filters.nodeTypes?.length) {
    nodes = nodes.filter((n) => filters.nodeTypes!.includes(n.type));
  }

  // Filter by date range
  if (filters.dateRange) {
    nodes = nodes.filter((n) => {
      if (!n.timestamp) return true;
      const date = new Date(n.timestamp);
      return date >= filters.dateRange!.start && date <= filters.dateRange!.end;
    });
  }

  // Filter by search query
  if (filters.searchQuery) {
    const query = filters.searchQuery.toLowerCase();
    nodes = nodes.filter((n) => n.label.toLowerCase().includes(query));
  }

  // Filter links to only include connected nodes
  const nodeIds = new Set(nodes.map((n) => n.id));
  const links = data.links.filter(
    (l) => nodeIds.has(l.source) && nodeIds.has(l.target)
  );

  return { nodes, links };
}
```

### Dashboard Mini-View Integration

```tsx
// src/components/dashboard/GraphMiniView.tsx
import { useState } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { KnowledgeGraph } from '@/components/graph/KnowledgeGraph';
import { Maximize2 } from 'lucide-react';

export function GraphMiniView() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <>
      {/* Mini view card */}
      <div className="relative group">
        <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => setIsExpanded(true)}
            className="p-1.5 rounded-md bg-background-surface/90 border border-border hover:bg-background-elevated transition-colors"
          >
            <Maximize2 className="h-4 w-4 text-text-secondary" />
          </button>
        </div>
        <KnowledgeGraph
          miniMode
          onExpand={() => setIsExpanded(true)}
          className="h-64 w-full"
        />
      </div>

      {/* Expanded full-screen dialog */}
      <Dialog open={isExpanded} onOpenChange={setIsExpanded}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] w-full h-full p-0">
          <KnowledgeGraph className="w-full h-[90vh]" />
        </DialogContent>
      </Dialog>
    </>
  );
}
```

### Clustering for Large Graphs

```tsx
// src/components/graph/useGraphClustering.ts
import { useMemo } from 'react';

interface GraphNode {
  id: string;
  type: string;
  cluster?: number;
}

interface GraphData {
  nodes: GraphNode[];
  links: { source: string; target: string }[];
}

// Simple clustering based on node type and connectivity
export function useGraphClustering(
  data: GraphData | undefined,
  options: { threshold?: number; enabled?: boolean } = {}
) {
  const { threshold = 500, enabled = true } = options;

  return useMemo(() => {
    if (!data || !enabled || data.nodes.length < threshold) {
      return data;
    }

    // Group nodes by type for cluster representatives
    const typeGroups = new Map<string, GraphNode[]>();
    data.nodes.forEach((node) => {
      const existing = typeGroups.get(node.type) || [];
      typeGroups.set(node.type, [...existing, node]);
    });

    // Create cluster nodes for large groups
    const clusterNodes: GraphNode[] = [];
    const clusterLinks: { source: string; target: string }[] = [];
    const nodeToCluster = new Map<string, string>();

    typeGroups.forEach((nodes, type) => {
      if (nodes.length > 50) {
        // Create cluster representative
        const clusterId = `cluster-${type}`;
        clusterNodes.push({
          id: clusterId,
          type: type,
          cluster: nodes.length,
        } as any);

        // Map original nodes to cluster
        nodes.forEach((n) => nodeToCluster.set(n.id, clusterId));
      } else {
        clusterNodes.push(...nodes);
      }
    });

    // Rebuild links with cluster references
    data.links.forEach((link) => {
      const source = nodeToCluster.get(link.source) || link.source;
      const target = nodeToCluster.get(link.target) || link.target;

      if (source !== target) {
        const exists = clusterLinks.some(
          (l) => l.source === source && l.target === target
        );
        if (!exists) {
          clusterLinks.push({ source, target });
        }
      }
    });

    return { nodes: clusterNodes, links: clusterLinks };
  }, [data, threshold, enabled]);
}
```

### Animation Styles

```css
/* src/styles/graph-animations.css */

@keyframes slide-in-right {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.animate-slide-in-right {
  animation: slide-in-right 0.2s ease-out;
}

/* Node pulse animation for highlighted nodes */
@keyframes node-pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.2);
    opacity: 0.8;
  }
}

/* Gentle breathing effect for the whole graph */
@keyframes graph-breathe {
  0%, 100% {
    filter: brightness(1);
  }
  50% {
    filter: brightness(1.02);
  }
}

.graph-container {
  animation: graph-breathe 4s ease-in-out infinite;
}
```

## Performance Optimizations

### Canvas Rendering Optimizations

```typescript
// src/components/graph/graphOptimizations.ts

// Reduce node detail based on zoom level
export function getNodeDetail(globalScale: number): 'full' | 'medium' | 'minimal' {
  if (globalScale > 2) return 'full';
  if (globalScale > 0.5) return 'medium';
  return 'minimal';
}

// Viewport culling - only render visible nodes
export function isNodeVisible(
  node: { x: number; y: number },
  viewport: { x: number; y: number; width: number; height: number },
  padding = 50
): boolean {
  return (
    node.x >= viewport.x - padding &&
    node.x <= viewport.x + viewport.width + padding &&
    node.y >= viewport.y - padding &&
    node.y <= viewport.y + viewport.height + padding
  );
}

// Throttle simulation updates for better performance
export function createThrottledUpdate(callback: () => void, limit = 16) {
  let lastCall = 0;
  return () => {
    const now = Date.now();
    if (now - lastCall >= limit) {
      lastCall = now;
      callback();
    }
  };
}

// Web Worker for physics simulation (optional)
export function createPhysicsWorker() {
  const workerCode = `
    self.onmessage = function(e) {
      const { nodes, links, iterations } = e.data;
      // Simple force simulation
      for (let i = 0; i < iterations; i++) {
        // Apply forces...
      }
      self.postMessage({ nodes });
    };
  `;

  const blob = new Blob([workerCode], { type: 'application/javascript' });
  return new Worker(URL.createObjectURL(blob));
}
```

## Acceptance Criteria

- [ ] Force-directed graph renders with proper physics
- [ ] Node types display distinct colors and sizes
- [ ] Zoom, pan, and drag interactions work smoothly
- [ ] Click on node shows detail panel
- [ ] Hover highlights node and shows label
- [ ] Mini-view displays on dashboard
- [ ] Expand to full-screen works
- [ ] Performance: 60fps with 1000 nodes
- [ ] Clustering activates for large graphs (500+ nodes)
- [ ] Breathing animation creates ambient liveness
- [ ] Graph controls (zoom, fit, reset) work
- [ ] Loading skeleton displays while fetching

## Test Plan

### Unit Tests
```typescript
describe('KnowledgeGraph', () => {
  it('should render nodes with correct colors', () => {
    const data = {
      nodes: [{ id: '1', label: 'Test', type: 'Event' }],
      links: [],
    };
    // Verify Event node renders with primary blue
  });

  it('should filter nodes by type', () => {
    const result = filterGraphData(mockData, { nodeTypes: ['Person'] });
    expect(result.nodes.every(n => n.type === 'Person')).toBe(true);
  });

  it('should cluster large graphs', () => {
    const largeData = generateMockNodes(1000);
    const clustered = useGraphClustering(largeData, { threshold: 500 });
    expect(clustered.nodes.length).toBeLessThan(largeData.nodes.length);
  });
});
```

### Performance Tests
```typescript
describe('Graph Performance', () => {
  it('should render 1000 nodes at 60fps', async () => {
    const data = generateMockNodes(1000);
    const startTime = performance.now();

    render(<KnowledgeGraph data={data} />);

    // Measure frame rate over 1 second
    const frames = await measureFrameRate(1000);
    expect(frames).toBeGreaterThanOrEqual(55);
  });
});
```

### E2E Tests
```typescript
test('graph interactions', async ({ page }) => {
  await page.goto('/dashboard');

  // Wait for graph to load
  const graphCanvas = page.locator('canvas');
  await expect(graphCanvas).toBeVisible();

  // Test zoom
  await page.mouse.wheel(0, -100);

  // Click on node (mock coordinates)
  await graphCanvas.click({ position: { x: 400, y: 300 } });

  // Verify detail panel appears
  await expect(page.locator('[data-testid="node-detail-panel"]')).toBeVisible();
});
```

## Dependencies

- react-force-graph-2d (or react-force-graph-3d for optional 3D mode)
- d3-force (included with react-force-graph)
- @tanstack/react-query
- @tauri-apps/api

## Next Steps

After graph visualization complete:
1. Integrate with search results (highlight related nodes)
2. Add temporal filtering (date range slider)
3. Implement node clustering expansion
4. Add graph export (PNG, SVG)
5. Consider WebGL renderer for 10,000+ nodes

---

## Vision Alignment

This visualization represents the core Futurnal experience: **seeing your Ghost's understanding of your personal universe**.

Key conceptual alignments:
- **Node colors** reflect Ghost/Animal evolution (Blue = Ghost intelligence, Emerald = Animal evolution)
- **Breathing animation** creates ambient "liveness"—the Ghost is always processing, always learning
- **Connections** reveal how your Ghost perceives relationships between experiential elements
- **Click-to-explore** enables users to dive deeper into any memory

As Phase 2 (The Analyst) is implemented, this visualization will show:
- Emergent Insight clusters (Phase 2)
- Curiosity Engine gaps (Phase 2)
- Causal relationship chains (Phase 3)

---

**This knowledge graph visualization is the window into your Ghost's mind—your personal universe made visible, explorable, and alive.**

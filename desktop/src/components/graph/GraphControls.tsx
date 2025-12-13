/**
 * GraphControls - Control panel for graph interactions
 *
 * Provides buttons for:
 * - Zoom in/out
 * - Fit to view
 * - Reset view
 * - Export graph data as JSON
 */

import { ZoomIn, ZoomOut, Maximize, RotateCcw, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import type { KnowledgeGraphRef } from './KnowledgeGraph';
import type { GraphData } from '@/types/api';

interface GraphControlsProps {
  /** Ref to KnowledgeGraph for imperative control */
  graphRef: React.RefObject<KnowledgeGraphRef | null>;
  /** Reset callback (clears selection, fits to view) */
  onReset?: () => void;
  /** Graph data for export (optional - export button hidden if not provided) */
  graphData?: GraphData;
  /** Additional CSS classes */
  className?: string;
}

export function GraphControls({ graphRef, onReset, graphData, className }: GraphControlsProps) {
  const handleZoomIn = () => {
    const current = graphRef.current;
    if (!current) return;
    const currentZoom = current.zoom() ?? 1;
    current.zoom(currentZoom * 1.5, 300);
  };

  const handleZoomOut = () => {
    const current = graphRef.current;
    if (!current) return;
    const currentZoom = current.zoom() ?? 1;
    current.zoom(currentZoom / 1.5, 300);
  };

  const handleFit = () => {
    graphRef.current?.zoomToFit(400, 50);
  };

  const handleReset = () => {
    graphRef.current?.zoomToFit(400, 50);
    onReset?.();
  };

  const handleExportJSON = () => {
    if (!graphData) return;

    // Create export data with cleaned node properties (remove force-graph internal props)
    const exportData = {
      nodes: graphData.nodes.map((node) => ({
        id: node.id,
        label: node.label,
        node_type: node.node_type,
        timestamp: node.timestamp,
        metadata: node.metadata,
      })),
      links: graphData.links.map((link) => ({
        source: typeof link.source === 'object' ? (link.source as { id: string }).id : link.source,
        target: typeof link.target === 'object' ? (link.target as { id: string }).id : link.target,
        relationship: link.relationship,
        weight: link.weight,
        confidence: link.confidence,
      })),
      exportedAt: new Date().toISOString(),
      nodeCount: graphData.nodes.length,
      linkCount: graphData.links.length,
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `futurnal-graph-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <TooltipProvider delayDuration={300}>
      <div
        data-slot="graph-controls"
        data-testid="graph-controls"
        role="toolbar"
        aria-label="Graph controls"
        aria-orientation="vertical"
        className={cn(
          'flex flex-col gap-1 p-1',
          'bg-black/90 backdrop-blur-sm border border-white/10',
          className
        )}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
              onClick={handleZoomIn}
            >
              <ZoomIn className="h-4 w-4" />
              <span className="sr-only">Zoom in</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="left" className="bg-black border-white/10 text-white/80">
            Zoom in
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
              onClick={handleZoomOut}
            >
              <ZoomOut className="h-4 w-4" />
              <span className="sr-only">Zoom out</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="left" className="bg-black border-white/10 text-white/80">
            Zoom out
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
              onClick={handleFit}
            >
              <Maximize className="h-4 w-4" />
              <span className="sr-only">Fit to view</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="left" className="bg-black border-white/10 text-white/80">
            Fit to view
          </TooltipContent>
        </Tooltip>

        <div className="w-full h-px bg-white/10 my-1" />

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
              onClick={handleReset}
            >
              <RotateCcw className="h-4 w-4" />
              <span className="sr-only">Reset view</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="left" className="bg-black border-white/10 text-white/80">
            Reset view
          </TooltipContent>
        </Tooltip>

        {graphData && (
          <>
            <div className="w-full h-px bg-white/10 my-1" />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
                  onClick={handleExportJSON}
                >
                  <Download className="h-4 w-4" />
                  <span className="sr-only">Export as JSON</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left" className="bg-black border-white/10 text-white/80">
                Export as JSON
              </TooltipContent>
            </Tooltip>
          </>
        )}
      </div>
    </TooltipProvider>
  );
}

export default GraphControls;

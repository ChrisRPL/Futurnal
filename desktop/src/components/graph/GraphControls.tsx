/**
 * GraphControls - Control panel for graph interactions
 *
 * Provides buttons for:
 * - Zoom in/out
 * - Fit to view
 * - Reset view
 */

import { ZoomIn, ZoomOut, Maximize, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import type { KnowledgeGraphRef } from './KnowledgeGraph';

interface GraphControlsProps {
  /** Ref to KnowledgeGraph for imperative control */
  graphRef: React.RefObject<KnowledgeGraphRef | null>;
  /** Reset callback (clears selection, fits to view) */
  onReset?: () => void;
  /** Additional CSS classes */
  className?: string;
}

export function GraphControls({ graphRef, onReset, className }: GraphControlsProps) {
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

  return (
    <TooltipProvider delayDuration={300}>
      <div
        data-slot="graph-controls"
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
      </div>
    </TooltipProvider>
  );
}

export default GraphControls;

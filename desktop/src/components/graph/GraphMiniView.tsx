/**
 * GraphMiniView - Dashboard mini-view wrapper for Knowledge Graph
 *
 * Displays a compact preview of the knowledge graph on the dashboard.
 * Clicking navigates to the full /graph page for exploration.
 */

import { useNavigate } from 'react-router-dom';
import { Maximize2 } from 'lucide-react';
import { KnowledgeGraph } from './KnowledgeGraph';
import { cn } from '@/lib/utils';

interface GraphMiniViewProps {
  /** Additional CSS classes */
  className?: string;
}

export function GraphMiniView({ className }: GraphMiniViewProps) {
  const navigate = useNavigate();

  const handleExpand = () => {
    navigate('/graph');
  };

  return (
    <div
      data-slot="graph-mini-view"
      data-testid="graph-mini-view"
      className={cn('relative group', className)}
    >
      {/* Expand button overlay */}
      <div className="absolute top-3 right-3 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={handleExpand}
          className={cn(
            'p-1.5 bg-black/90 border border-white/10',
            'hover:bg-white/10 hover:border-white/20 transition-colors'
          )}
          title="Expand graph"
        >
          <Maximize2 className="h-4 w-4 text-white/60" />
        </button>
      </div>

      {/* Mini graph */}
      <KnowledgeGraph
        miniMode
        onExpand={handleExpand}
        className="h-full w-full"
      />
    </div>
  );
}

export default GraphMiniView;

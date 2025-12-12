/**
 * GraphFilterPanel - Filter panel for toggling node type visibility
 *
 * Provides toggleable chips/buttons for each entity type to show/hide
 * nodes in the knowledge graph. Essential for managing large graphs.
 */

import { Filter, Palette } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { cn } from '@/lib/utils';
import { useGraphStore, ALL_ENTITY_TYPES, isNodeTypeVisible } from '@/stores/graphStore';
import type { EntityType } from '@/types/api';

interface GraphFilterPanelProps {
  /** Additional CSS classes */
  className?: string;
}

/** Icons/labels for each entity type */
const ENTITY_TYPE_INFO: Record<EntityType, { label: string; shortLabel: string }> = {
  Event: { label: 'Events', shortLabel: 'EVT' },
  Person: { label: 'People', shortLabel: 'PER' },
  Document: { label: 'Documents', shortLabel: 'DOC' },
  Code: { label: 'Code', shortLabel: 'CODE' },
  Concept: { label: 'Concepts', shortLabel: 'CON' },
  Email: { label: 'Emails', shortLabel: 'EMAIL' },
  Mailbox: { label: 'Mailboxes', shortLabel: 'MBOX' },
  Source: { label: 'Sources', shortLabel: 'SRC' },
  Organization: { label: 'Organizations', shortLabel: 'ORG' },
};

export function GraphFilterPanel({ className }: GraphFilterPanelProps) {
  const { visibleNodeTypes, colorMode, toggleNodeType, showAllNodeTypes, toggleColorMode } = useGraphStore();

  // Count how many types are hidden
  const hiddenCount = visibleNodeTypes.length === 0
    ? 0  // Empty means all visible
    : ALL_ENTITY_TYPES.length - visibleNodeTypes.length;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            'h-8 px-2 text-white/60 hover:text-white hover:bg-white/10 gap-1.5',
            hiddenCount > 0 && 'text-amber-400/80',
            className
          )}
        >
          <Filter className="h-4 w-4" />
          <span className="text-xs">
            {hiddenCount > 0 ? `${hiddenCount} hidden` : 'Filter'}
          </span>
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        align="end"
        className="w-64 p-3 bg-black/95 border-white/10 backdrop-blur-sm"
      >
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/60 font-medium">Node Types</span>
            <button
              onClick={showAllNodeTypes}
              className="text-xs text-white/40 hover:text-white/80 transition-colors"
            >
              Show all
            </button>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {ALL_ENTITY_TYPES.map((type) => {
              const isVisible = isNodeTypeVisible(type, visibleNodeTypes);
              const info = ENTITY_TYPE_INFO[type];

              return (
                <button
                  key={type}
                  onClick={() => toggleNodeType(type)}
                  className={cn(
                    'px-2 py-1 text-xs rounded transition-all',
                    'border',
                    isVisible
                      ? 'bg-white/10 border-white/20 text-white/90 hover:bg-white/20'
                      : 'bg-transparent border-white/10 text-white/30 hover:text-white/50 hover:border-white/20'
                  )}
                >
                  {info.label}
                </button>
              );
            })}
          </div>

          {/* Color Mode Toggle */}
          <div className="pt-2 border-t border-white/10">
            <div className="flex items-center justify-between">
              <span className="text-xs text-white/60 font-medium">Color Mode</span>
              <button
                onClick={toggleColorMode}
                className={cn(
                  'flex items-center gap-1.5 px-2 py-1 text-xs rounded transition-all border',
                  colorMode === 'colored'
                    ? 'bg-white/10 border-white/20 text-white/90'
                    : 'bg-transparent border-white/10 text-white/40 hover:text-white/60'
                )}
              >
                <Palette className="h-3 w-3" />
                {colorMode === 'colored' ? 'Colored' : 'Monochrome'}
              </button>
            </div>
          </div>

          <div className="pt-2 border-t border-white/10">
            <p className="text-xs text-white/40">
              Click to toggle visibility. Hide emails to see structure.
            </p>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export default GraphFilterPanel;

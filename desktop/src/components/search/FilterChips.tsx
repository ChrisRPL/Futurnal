/**
 * FilterChips - Filter management for search
 *
 * Displays active filters as removable chips and provides
 * quick filter buttons for common filter types.
 */

import { X, Clock, GitBranch, FileText, Mic, Image, Code } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { SearchFilter } from '@/stores/searchStore';

interface FilterChipsProps {
  /** Active filters */
  filters: SearchFilter[];
  /** Handler to update filters */
  onChange: (filters: SearchFilter[]) => void;
  /** Additional class names */
  className?: string;
}

/**
 * Quick filter options for intent types
 */
const INTENT_OPTIONS = [
  { value: 'temporal', label: 'Temporal', icon: Clock },
  { value: 'causal', label: 'Causal', icon: GitBranch },
];

/**
 * Quick filter options for entity types
 */
const ENTITY_TYPE_OPTIONS = [
  { value: 'Event', label: 'Events' },
  { value: 'Document', label: 'Documents' },
  { value: 'Person', label: 'People' },
  { value: 'Code', label: 'Code' },
];

/**
 * Quick filter options for source types
 */
const SOURCE_TYPE_OPTIONS = [
  { value: 'text', label: 'Text', icon: FileText },
  { value: 'ocr', label: 'OCR', icon: Image },
  { value: 'audio', label: 'Audio', icon: Mic },
  { value: 'code', label: 'Code', icon: Code },
];

export function FilterChips({ filters, onChange, className }: FilterChipsProps) {
  const addFilter = (filter: SearchFilter) => {
    // Prevent duplicates
    const exists = filters.some(
      (f) => f.type === filter.type && f.value === filter.value
    );
    if (!exists) {
      onChange([...filters, filter]);
    }
  };

  const removeFilter = (filter: SearchFilter) => {
    onChange(
      filters.filter((f) => !(f.type === filter.type && f.value === filter.value))
    );
  };

  return (
    <div
      data-slot="filter-chips"
      className={cn('flex flex-wrap items-center gap-2', className)}
    >
      {/* Active filters */}
      {filters.map((filter) => (
        <button
          key={`${filter.type}-${filter.value}`}
          onClick={() => removeFilter(filter)}
          className={cn(
            'inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs',
            'bg-white/20 text-white border border-white/30',
            'hover:bg-white/25 transition-colors cursor-pointer'
          )}
        >
          {filter.label}
          <X className="h-3 w-3" />
        </button>
      ))}

      {/* Quick filter buttons - show when no filters active */}
      {filters.length === 0 && (
        <div className="flex items-center gap-1.5 text-xs text-white/50">
          <span>Quick filters:</span>
          {INTENT_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() =>
                addFilter({ type: 'intent', value: opt.value, label: opt.label })
              }
              className={cn(
                'px-2 py-0.5 rounded',
                'bg-white/5 text-white/50',
                'hover:bg-white/10 hover:text-white/70',
                'transition-colors'
              )}
            >
              {opt.label}
            </button>
          ))}
          {ENTITY_TYPE_OPTIONS.slice(0, 2).map((opt) => (
            <button
              key={opt.value}
              onClick={() =>
                addFilter({ type: 'entityType', value: opt.value, label: opt.label })
              }
              className={cn(
                'px-2 py-0.5 rounded',
                'bg-white/5 text-white/50',
                'hover:bg-white/10 hover:text-white/70',
                'transition-colors'
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Export filter options for use in other components
export { INTENT_OPTIONS, ENTITY_TYPE_OPTIONS, SOURCE_TYPE_OPTIONS };

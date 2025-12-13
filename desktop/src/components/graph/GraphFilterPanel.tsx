/**
 * GraphFilterPanel - Advanced filter panel for knowledge graph
 *
 * Provides filters for:
 * - Node type visibility toggles
 * - Source multi-select
 * - Confidence range slider
 * - Time range picker
 * - Color mode toggle
 */

import { useState, useEffect, useCallback } from 'react';
import { Filter, Palette, Calendar, Gauge, FolderTree, RotateCcw, Save, Trash2, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { RangeSlider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';
import { useGraphStore, ALL_ENTITY_TYPES, isNodeTypeVisible } from '@/stores/graphStore';
import { graphApi } from '@/lib/api';
import type { EntityType, GraphStats } from '@/types/api';

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
  const {
    visibleNodeTypes,
    colorMode,
    sourceFilter,
    confidenceRange,
    timeRange,
    savedFilterPresets,
    toggleNodeType,
    showAllNodeTypes,
    toggleColorMode,
    setSourceFilter,
    setConfidenceRange,
    setTimeRange,
    clearFilters,
    saveFilterPreset,
    loadFilterPreset,
    deleteFilterPreset,
  } = useGraphStore();

  // Available sources from graph stats
  const [availableSources, setAvailableSources] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [showSaveInput, setShowSaveInput] = useState(false);

  // Fetch available sources from backend
  useEffect(() => {
    let mounted = true;
    const fetchSources = async () => {
      setIsLoading(true);
      try {
        const stats: GraphStats = await graphApi.getStats();
        if (mounted && stats.nodes_by_source) {
          setAvailableSources(Object.keys(stats.nodes_by_source));
        }
      } catch (error) {
        console.warn('[GraphFilterPanel] Failed to fetch sources:', error);
      } finally {
        if (mounted) setIsLoading(false);
      }
    };
    fetchSources();
    return () => { mounted = false; };
  }, []);

  // Count how many types are hidden
  const hiddenCount = visibleNodeTypes.length === 0
    ? 0  // Empty means all visible
    : ALL_ENTITY_TYPES.length - visibleNodeTypes.length;

  // Check if any advanced filters are active
  const hasActiveFilters =
    sourceFilter.length > 0 ||
    confidenceRange[0] > 0 ||
    confidenceRange[1] < 1 ||
    timeRange.start !== null ||
    timeRange.end !== null;

  const totalActiveFilters = hiddenCount + (hasActiveFilters ? 1 : 0);

  // Toggle source in filter
  const toggleSource = useCallback((source: string) => {
    if (sourceFilter.includes(source)) {
      setSourceFilter(sourceFilter.filter((s) => s !== source));
    } else {
      setSourceFilter([...sourceFilter, source]);
    }
  }, [sourceFilter, setSourceFilter]);

  // Handle confidence range change
  const handleConfidenceChange = useCallback((range: [number, number]) => {
    setConfidenceRange(range);
  }, [setConfidenceRange]);

  // Handle date changes
  const handleStartDateChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value || null;
    setTimeRange({ ...timeRange, start: value });
  }, [timeRange, setTimeRange]);

  const handleEndDateChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value || null;
    setTimeRange({ ...timeRange, end: value });
  }, [timeRange, setTimeRange]);

  // Handle saving preset
  const handleSavePreset = useCallback(() => {
    if (!presetName.trim()) return;
    saveFilterPreset(presetName.trim());
    setPresetName('');
    setShowSaveInput(false);
  }, [presetName, saveFilterPreset]);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            'h-8 px-2 text-white/60 hover:text-white hover:bg-white/10 gap-1.5',
            totalActiveFilters > 0 && 'text-amber-400/80',
            className
          )}
        >
          <Filter className="h-4 w-4" />
          <span className="text-xs">
            {totalActiveFilters > 0 ? `${totalActiveFilters} active` : 'Filter'}
          </span>
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        align="end"
        className="w-80 p-3 bg-black/95 border-white/10 backdrop-blur-sm max-h-[80vh] overflow-y-auto"
      >
        <div className="space-y-4">
          {/* Header with reset */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-white/80 font-medium">Filters</span>
            {(hiddenCount > 0 || hasActiveFilters) && (
              <button
                onClick={() => {
                  showAllNodeTypes();
                  clearFilters();
                }}
                className="flex items-center gap-1 text-xs text-white/40 hover:text-white/80 transition-colors"
              >
                <RotateCcw className="h-3 w-3" />
                Reset all
              </button>
            )}
          </div>

          {/* Node Types Section */}
          <div className="space-y-2">
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
          </div>

          {/* Source Filter Section */}
          <div className="pt-2 border-t border-white/10 space-y-2">
            <div className="flex items-center gap-1.5">
              <FolderTree className="h-3 w-3 text-white/40" />
              <span className="text-xs text-white/60 font-medium">Sources</span>
              {sourceFilter.length > 0 && (
                <span className="text-xs text-amber-400/80">({sourceFilter.length})</span>
              )}
            </div>

            {isLoading ? (
              <div className="text-xs text-white/40">Loading sources...</div>
            ) : availableSources.length === 0 ? (
              <div className="text-xs text-white/40">No sources available</div>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {availableSources.map((source) => {
                  const isSelected = sourceFilter.includes(source);
                  // Extract display name from path
                  const displayName = source.split('/').pop() || source;

                  return (
                    <button
                      key={source}
                      onClick={() => toggleSource(source)}
                      title={source}
                      className={cn(
                        'px-2 py-1 text-xs rounded transition-all',
                        'border max-w-[120px] truncate',
                        isSelected
                          ? 'bg-white/15 border-white/30 text-white/90 hover:bg-white/20'
                          : 'bg-transparent border-white/10 text-white/40 hover:text-white/60 hover:border-white/20'
                      )}
                    >
                      {displayName}
                    </button>
                  );
                })}
              </div>
            )}
            {sourceFilter.length > 0 && (
              <button
                onClick={() => setSourceFilter([])}
                className="text-xs text-white/40 hover:text-white/80 transition-colors"
              >
                Clear sources
              </button>
            )}
          </div>

          {/* Confidence Range Section */}
          <div className="pt-2 border-t border-white/10 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Gauge className="h-3 w-3 text-white/40" />
                <span className="text-xs text-white/60 font-medium">Confidence</span>
              </div>
              <span className="text-xs text-white/50">
                {Math.round(confidenceRange[0] * 100)}% - {Math.round(confidenceRange[1] * 100)}%
              </span>
            </div>

            <RangeSlider
              min={0}
              max={1}
              step={0.05}
              value={confidenceRange}
              onChange={handleConfidenceChange}
            />
          </div>

          {/* Time Range Section */}
          <div className="pt-2 border-t border-white/10 space-y-2">
            <div className="flex items-center gap-1.5">
              <Calendar className="h-3 w-3 text-white/40" />
              <span className="text-xs text-white/60 font-medium">Time Range</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="text-xs text-white/40">From</label>
                <input
                  type="date"
                  value={timeRange.start || ''}
                  onChange={handleStartDateChange}
                  className={cn(
                    'w-full px-2 py-1.5 text-xs rounded',
                    'bg-white/5 border border-white/10 text-white/80',
                    'focus:outline-none focus:border-white/30',
                    '[color-scheme:dark]'
                  )}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-white/40">To</label>
                <input
                  type="date"
                  value={timeRange.end || ''}
                  onChange={handleEndDateChange}
                  className={cn(
                    'w-full px-2 py-1.5 text-xs rounded',
                    'bg-white/5 border border-white/10 text-white/80',
                    'focus:outline-none focus:border-white/30',
                    '[color-scheme:dark]'
                  )}
                />
              </div>
            </div>
            {(timeRange.start || timeRange.end) && (
              <button
                onClick={() => setTimeRange({ start: null, end: null })}
                className="text-xs text-white/40 hover:text-white/80 transition-colors"
              >
                Clear dates
              </button>
            )}
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

          {/* Saved Presets Section */}
          <div className="pt-2 border-t border-white/10 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-white/60 font-medium">Saved Presets</span>
              {!showSaveInput && (
                <button
                  onClick={() => setShowSaveInput(true)}
                  className="flex items-center gap-1 text-xs text-white/40 hover:text-white/80 transition-colors"
                >
                  <Save className="h-3 w-3" />
                  Save current
                </button>
              )}
            </div>

            {/* Save preset input */}
            {showSaveInput && (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  placeholder="Preset name..."
                  className={cn(
                    'flex-1 px-2 py-1.5 text-xs rounded',
                    'bg-white/5 border border-white/10 text-white/80',
                    'placeholder:text-white/30',
                    'focus:outline-none focus:border-white/30'
                  )}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSavePreset();
                    if (e.key === 'Escape') {
                      setShowSaveInput(false);
                      setPresetName('');
                    }
                  }}
                  autoFocus
                />
                <button
                  onClick={handleSavePreset}
                  disabled={!presetName.trim()}
                  className={cn(
                    'p-1.5 rounded transition-colors',
                    presetName.trim()
                      ? 'text-green-400 hover:bg-green-500/20'
                      : 'text-white/20 cursor-not-allowed'
                  )}
                >
                  <Check className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {/* Preset list */}
            {savedFilterPresets.length === 0 ? (
              <div className="text-xs text-white/30 py-1">
                No saved presets
              </div>
            ) : (
              <div className="space-y-1">
                {savedFilterPresets.map((preset) => (
                  <div
                    key={preset.id}
                    className="flex items-center justify-between group"
                  >
                    <button
                      onClick={() => loadFilterPreset(preset.id)}
                      className="flex-1 text-left px-2 py-1.5 text-xs text-white/60 hover:text-white/90 hover:bg-white/5 rounded transition-colors truncate"
                      title={`Load "${preset.name}"`}
                    >
                      {preset.name}
                    </button>
                    <button
                      onClick={() => deleteFilterPreset(preset.id)}
                      className="p-1 text-white/20 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                      title="Delete preset"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Help text */}
          <div className="pt-2 border-t border-white/10">
            <p className="text-xs text-white/40">
              Filters apply client-side. Select sources to focus on specific data.
            </p>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export default GraphFilterPanel;

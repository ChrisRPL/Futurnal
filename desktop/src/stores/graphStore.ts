/**
 * Graph Store - Zustand state management for knowledge graph visualization
 *
 * Manages graph interaction state including node selection, hover,
 * zoom level, and display preferences.
 * Uses persist middleware for localStorage persistence of preferences.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { EntityType } from '@/types/api';

/**
 * Color mode for graph visualization.
 * - 'monochrome': White/gray nodes with opacity variation (default)
 * - 'colored': Distinct colors per entity type
 */
export type ColorMode = 'monochrome' | 'colored';

/**
 * Layout mode for graph visualization.
 * - 'force': Force-directed physics simulation
 * - 'timeline': Horizontal time-based layout
 */
export type LayoutMode = 'force' | 'timeline';

/**
 * Time granularity for timeline view.
 * Controls how nodes are grouped/bucketed on the timeline.
 */
export type TimeGranularity = 'auto' | 'hour' | 'day' | 'week' | 'month' | 'year';

/**
 * Time range filter for graph.
 */
export interface TimeRange {
  start: string | null;
  end: string | null;
}

/**
 * Saved filter preset.
 */
export interface FilterPreset {
  id: string;
  name: string;
  createdAt: string;
  visibleNodeTypes: EntityType[];
  sourceFilter: string[];
  confidenceRange: [number, number];
  timeRange: TimeRange;
}

/**
 * All available entity types for filtering.
 */
export const ALL_ENTITY_TYPES: EntityType[] = [
  'Event',
  'Document',
  'Person',
  'Code',
  'Concept',
  'Email',
  'Mailbox',
  'Source',
  'Organization',
];

interface GraphState {
  // === Core State ===
  /** Currently selected node ID for detail panel */
  selectedNodeId: string | null;
  /** Currently hovered node ID for highlight */
  hoveredNodeId: string | null;
  /** Current zoom level (1.0 = 100%) */
  zoomLevel: number;
  /** Center position for the graph view */
  centerPosition: { x: number; y: number };
  /** Whether the graph is expanded to full-screen */
  isExpanded: boolean;

  // === Display Preferences ===
  /** Visible node types (empty = all visible) */
  visibleNodeTypes: EntityType[];
  /** Whether breathing animation is enabled */
  breathingEnabled: boolean;
  /** Color mode for node visualization */
  colorMode: ColorMode;
  /** Layout mode (force-directed or timeline) */
  layoutMode: LayoutMode;
  /** Time granularity for timeline view */
  timeGranularity: TimeGranularity;

  // === Advanced Filters ===
  /** Filter by source identifiers */
  sourceFilter: string[];
  /** Confidence range filter [min, max] (0-1) */
  confidenceRange: [number, number];
  /** Time range filter */
  timeRange: TimeRange;

  // === Highlighting (for search integration) ===
  /** Node IDs highlighted from search results */
  highlightedNodeIds: string[];

  // === Bookmarks ===
  /** Bookmarked node IDs (synced with backend) */
  bookmarkedNodeIds: string[];

  // === Filter Presets ===
  /** Saved filter presets */
  savedFilterPresets: FilterPreset[];

  // === Actions ===
  /** Set selected node ID (null to deselect) */
  setSelectedNode: (id: string | null) => void;
  /** Set hovered node ID (null to clear hover) */
  setHoveredNode: (id: string | null) => void;
  /** Set zoom level */
  setZoomLevel: (level: number) => void;
  /** Set center position */
  setCenterPosition: (pos: { x: number; y: number }) => void;
  /** Toggle visibility of a node type */
  toggleNodeType: (type: EntityType) => void;
  /** Set all node types visible */
  showAllNodeTypes: () => void;
  /** Set breathing animation enabled/disabled */
  setBreathingEnabled: (enabled: boolean) => void;
  /** Set expanded state */
  setExpanded: (expanded: boolean) => void;
  /** Toggle color mode */
  toggleColorMode: () => void;
  /** Set layout mode */
  setLayoutMode: (mode: LayoutMode) => void;
  /** Set time granularity for timeline view */
  setTimeGranularity: (granularity: TimeGranularity) => void;
  /** Set source filter */
  setSourceFilter: (sources: string[]) => void;
  /** Set confidence range */
  setConfidenceRange: (range: [number, number]) => void;
  /** Set time range */
  setTimeRange: (range: TimeRange) => void;
  /** Clear all advanced filters */
  clearFilters: () => void;
  /** Set highlighted nodes (from search) */
  setHighlightedNodes: (ids: string[]) => void;
  /** Clear highlighted nodes */
  clearHighlights: () => void;
  /** Set bookmarked nodes */
  setBookmarkedNodes: (ids: string[]) => void;
  /** Toggle bookmark for a node */
  toggleBookmark: (id: string) => void;
  /** Save current filter settings as a preset */
  saveFilterPreset: (name: string) => void;
  /** Load a saved filter preset */
  loadFilterPreset: (id: string) => void;
  /** Delete a saved filter preset */
  deleteFilterPreset: (id: string) => void;
  /** Reset to initial state */
  reset: () => void;
}

const initialState = {
  // Core state
  selectedNodeId: null as string | null,
  hoveredNodeId: null as string | null,
  zoomLevel: 1.0,
  centerPosition: { x: 0, y: 0 },
  isExpanded: false,
  // Display preferences
  visibleNodeTypes: [] as EntityType[], // Empty means all visible
  breathingEnabled: true,
  colorMode: 'colored' as ColorMode,
  layoutMode: 'force' as LayoutMode,
  timeGranularity: 'auto' as TimeGranularity,
  // Advanced filters
  sourceFilter: [] as string[],
  confidenceRange: [0, 1] as [number, number],
  timeRange: { start: null, end: null } as TimeRange,
  // Highlighting
  highlightedNodeIds: [] as string[],
  // Bookmarks
  bookmarkedNodeIds: [] as string[],
  // Filter presets
  savedFilterPresets: [] as FilterPreset[],
};

export const useGraphStore = create<GraphState>()(
  persist(
    (set, get) => ({
      ...initialState,

      setSelectedNode: (id) => set({ selectedNodeId: id }),

      setHoveredNode: (id) => set({ hoveredNodeId: id }),

      setZoomLevel: (level) => set({ zoomLevel: level }),

      setCenterPosition: (pos) => set({ centerPosition: pos }),

      toggleNodeType: (type) => {
        const { visibleNodeTypes } = get();

        // If empty (all visible), switch to all-except-this
        if (visibleNodeTypes.length === 0) {
          set({
            visibleNodeTypes: ALL_ENTITY_TYPES.filter((t) => t !== type),
          });
          return;
        }

        // Toggle the type
        if (visibleNodeTypes.includes(type)) {
          // Remove it (but don't allow empty - that means "all")
          const updated = visibleNodeTypes.filter((t) => t !== type);
          // If removing this would leave empty, show all instead
          set({
            visibleNodeTypes: updated.length === 0 ? [] : updated,
          });
        } else {
          // Add it
          const updated = [...visibleNodeTypes, type];
          // If all types are now visible, switch to empty (meaning "all")
          set({
            visibleNodeTypes:
              updated.length === ALL_ENTITY_TYPES.length ? [] : updated,
          });
        }
      },

      showAllNodeTypes: () => set({ visibleNodeTypes: [] }),

      setBreathingEnabled: (enabled) => set({ breathingEnabled: enabled }),

      setExpanded: (expanded) => set({ isExpanded: expanded }),

      toggleColorMode: () => {
        const { colorMode } = get();
        set({ colorMode: colorMode === 'monochrome' ? 'colored' : 'monochrome' });
      },

      setLayoutMode: (mode) => set({ layoutMode: mode }),

      setTimeGranularity: (granularity) => set({ timeGranularity: granularity }),

      setSourceFilter: (sources) => set({ sourceFilter: sources }),

      setConfidenceRange: (range) => set({ confidenceRange: range }),

      setTimeRange: (range) => set({ timeRange: range }),

      clearFilters: () =>
        set({
          sourceFilter: [],
          confidenceRange: [0, 1],
          timeRange: { start: null, end: null },
        }),

      setHighlightedNodes: (ids) => set({ highlightedNodeIds: ids }),

      clearHighlights: () => set({ highlightedNodeIds: [] }),

      setBookmarkedNodes: (ids) => set({ bookmarkedNodeIds: ids }),

      toggleBookmark: (id) => {
        const { bookmarkedNodeIds } = get();
        if (bookmarkedNodeIds.includes(id)) {
          set({ bookmarkedNodeIds: bookmarkedNodeIds.filter((bid) => bid !== id) });
        } else {
          set({ bookmarkedNodeIds: [...bookmarkedNodeIds, id] });
        }
      },

      saveFilterPreset: (name) => {
        const {
          visibleNodeTypes,
          sourceFilter,
          confidenceRange,
          timeRange,
          savedFilterPresets,
        } = get();

        const newPreset: FilterPreset = {
          id: `preset-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          name,
          createdAt: new Date().toISOString(),
          visibleNodeTypes: [...visibleNodeTypes],
          sourceFilter: [...sourceFilter],
          confidenceRange: [...confidenceRange] as [number, number],
          timeRange: { ...timeRange },
        };

        set({ savedFilterPresets: [...savedFilterPresets, newPreset] });
      },

      loadFilterPreset: (id) => {
        const { savedFilterPresets } = get();
        const preset = savedFilterPresets.find((p) => p.id === id);
        if (!preset) return;

        set({
          visibleNodeTypes: [...preset.visibleNodeTypes],
          sourceFilter: [...preset.sourceFilter],
          confidenceRange: [...preset.confidenceRange] as [number, number],
          timeRange: { ...preset.timeRange },
        });
      },

      deleteFilterPreset: (id) => {
        const { savedFilterPresets } = get();
        set({ savedFilterPresets: savedFilterPresets.filter((p) => p.id !== id) });
      },

      reset: () =>
        set({
          selectedNodeId: null,
          hoveredNodeId: null,
          zoomLevel: 1.0,
          centerPosition: { x: 0, y: 0 },
          highlightedNodeIds: [],
          // Keep preferences: visibleNodeTypes, breathingEnabled, colorMode, layoutMode
          // Keep filters: sourceFilter, confidenceRange, timeRange
          // Keep bookmarks: bookmarkedNodeIds
        }),
    }),
    {
      name: 'futurnal-graph',
      version: 5, // Bump version for filter presets
      // Only persist user preferences, not transient state
      partialize: (state) => ({
        visibleNodeTypes: state.visibleNodeTypes,
        breathingEnabled: state.breathingEnabled,
        colorMode: state.colorMode,
        layoutMode: state.layoutMode,
        timeGranularity: state.timeGranularity,
        savedFilterPresets: state.savedFilterPresets,
      }),
      // Migration from old versions
      migrate: (persistedState, version) => {
        const state = persistedState as {
          visibleNodeTypes?: EntityType[];
          breathingEnabled?: boolean;
          colorMode?: ColorMode;
          layoutMode?: LayoutMode;
          timeGranularity?: TimeGranularity;
          savedFilterPresets?: FilterPreset[];
        } | null;

        if (version < 3) {
          return {
            visibleNodeTypes: state?.visibleNodeTypes ?? [],
            breathingEnabled: state?.breathingEnabled ?? true,
            colorMode: state?.colorMode ?? 'colored',
            layoutMode: 'force' as LayoutMode,
            timeGranularity: 'auto' as TimeGranularity,
            savedFilterPresets: [] as FilterPreset[],
          };
        }
        if (version < 4) {
          return {
            visibleNodeTypes: state?.visibleNodeTypes ?? [],
            breathingEnabled: state?.breathingEnabled ?? true,
            colorMode: state?.colorMode ?? 'colored',
            layoutMode: state?.layoutMode ?? 'force',
            timeGranularity: 'auto' as TimeGranularity,
            savedFilterPresets: [] as FilterPreset[],
          };
        }
        if (version < 5) {
          return {
            visibleNodeTypes: state?.visibleNodeTypes ?? [],
            breathingEnabled: state?.breathingEnabled ?? true,
            colorMode: state?.colorMode ?? 'colored',
            layoutMode: state?.layoutMode ?? 'force',
            timeGranularity: state?.timeGranularity ?? 'auto',
            savedFilterPresets: [] as FilterPreset[],
          };
        }
        return {
          visibleNodeTypes: state?.visibleNodeTypes ?? [],
          breathingEnabled: state?.breathingEnabled ?? true,
          colorMode: state?.colorMode ?? 'colored',
          layoutMode: state?.layoutMode ?? 'force',
          timeGranularity: state?.timeGranularity ?? 'auto',
          savedFilterPresets: state?.savedFilterPresets ?? [],
        };
      },
    }
  )
);

/**
 * Helper to check if a node type is visible.
 * Empty visibleNodeTypes means all types are visible.
 */
export function isNodeTypeVisible(
  type: EntityType,
  visibleNodeTypes: EntityType[]
): boolean {
  if (visibleNodeTypes.length === 0) return true;
  return visibleNodeTypes.includes(type);
}

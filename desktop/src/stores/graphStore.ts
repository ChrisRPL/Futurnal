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
  /** Currently selected node ID for detail panel */
  selectedNodeId: string | null;
  /** Currently hovered node ID for highlight */
  hoveredNodeId: string | null;
  /** Current zoom level (1.0 = 100%) */
  zoomLevel: number;
  /** Center position for the graph view */
  centerPosition: { x: number; y: number };
  /** Visible node types (empty = all visible) */
  visibleNodeTypes: EntityType[];
  /** Whether breathing animation is enabled */
  breathingEnabled: boolean;
  /** Whether the graph is expanded to full-screen */
  isExpanded: boolean;
  /** Color mode for node visualization */
  colorMode: ColorMode;

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
  /** Reset to initial state */
  reset: () => void;
}

const initialState = {
  selectedNodeId: null as string | null,
  hoveredNodeId: null as string | null,
  zoomLevel: 1.0,
  centerPosition: { x: 0, y: 0 },
  visibleNodeTypes: [] as EntityType[], // Empty means all visible
  breathingEnabled: true,
  isExpanded: false,
  colorMode: 'colored' as ColorMode, // Default to colored for semantic visualization
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

      reset: () =>
        set({
          selectedNodeId: null,
          hoveredNodeId: null,
          zoomLevel: 1.0,
          centerPosition: { x: 0, y: 0 },
          // Keep preferences: visibleNodeTypes, breathingEnabled, colorMode
        }),
    }),
    {
      name: 'futurnal-graph',
      version: 2, // Bump version to reset persisted state with new colored default
      // Only persist user preferences, not transient state
      partialize: (state) => ({
        visibleNodeTypes: state.visibleNodeTypes,
        breathingEnabled: state.breathingEnabled,
        colorMode: state.colorMode,
      }),
      // Migration from old version - reset to new defaults
      migrate: (persistedState, version) => {
        const state = persistedState as { visibleNodeTypes?: EntityType[]; breathingEnabled?: boolean; colorMode?: ColorMode } | null;
        if (version < 2) {
          // Reset colorMode to new default when upgrading
          return {
            visibleNodeTypes: state?.visibleNodeTypes ?? [],
            breathingEnabled: state?.breathingEnabled ?? true,
            colorMode: 'colored' as ColorMode,
          };
        }
        return {
          visibleNodeTypes: state?.visibleNodeTypes ?? [],
          breathingEnabled: state?.breathingEnabled ?? true,
          colorMode: state?.colorMode ?? 'colored',
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

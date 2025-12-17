/**
 * Schema Store - Zustand state management for schema evolution
 *
 * Step 08: Frontend Intelligence Integration - Phase 5
 *
 * Research Foundation:
 * - GFM-RAG: Schema-aware graph construction
 * - ACE: Adaptive schema evolution
 *
 * Manages schema statistics and evolution timeline.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Entity type statistics */
export interface EntityTypeStat {
  type: string;
  count: number;
  firstSeen: string | null;
  lastSeen: string | null;
}

/** Relationship type statistics */
export interface RelationshipTypeStat {
  type: string;
  count: number;
  confidenceAvg: number;
}

/** Quality metrics */
export interface QualityMetrics {
  precision: number;
  recall: number;
  temporalAccuracy: number;
}

/** Schema evolution event */
export interface SchemaEvolutionEvent {
  timestamp: string | null;
  changeType: string;
  details: string;
}

/** Response from get_schema_stats command */
interface SchemaStatsResponse {
  success: boolean;
  entityTypes: EntityTypeStat[];
  relationshipTypes: RelationshipTypeStat[];
  qualityMetrics: QualityMetrics;
  evolutionTimeline: SchemaEvolutionEvent[];
  error?: string;
}

// ============================================================================
// Store
// ============================================================================

interface SchemaState {
  /** Entity type statistics */
  entityTypes: EntityTypeStat[];
  /** Relationship type statistics */
  relationshipTypes: RelationshipTypeStat[];
  /** Quality metrics */
  qualityMetrics: QualityMetrics | null;
  /** Evolution timeline */
  evolutionTimeline: SchemaEvolutionEvent[];
  /** Loading state */
  isLoading: boolean;
  /** Error message */
  error: string | null;

  // Actions
  /** Fetch schema statistics */
  fetchSchemaStats: () => Promise<void>;
  /** Clear error */
  clearError: () => void;
}

export const useSchemaStore = create<SchemaState>()((set) => ({
  // Initial state
  entityTypes: [],
  relationshipTypes: [],
  qualityMetrics: null,
  evolutionTimeline: [],
  isLoading: false,
  error: null,

  // Fetch schema statistics
  fetchSchemaStats: async () => {
    set({ isLoading: true, error: null });

    try {
      const response = await invoke<SchemaStatsResponse>('get_schema_stats');

      if (response.success) {
        set({
          entityTypes: response.entityTypes,
          relationshipTypes: response.relationshipTypes,
          qualityMetrics: response.qualityMetrics,
          evolutionTimeline: response.evolutionTimeline,
          isLoading: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch schema stats',
          isLoading: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Fetch schema stats failed';
      set({ error: errorMsg, isLoading: false });
    }
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },
}));

export default useSchemaStore;

/**
 * Causal Store - Zustand state management for causal chain exploration
 *
 * Step 08: Frontend Intelligence Integration - Phase 2
 *
 * Research Foundation:
 * - Youtu-GraphRAG: Multi-hop causal reasoning
 * - CausalRAG: Causal-aware retrieval
 *
 * Manages causal chain state: causes, effects, and paths.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Causal cause from find_causes query */
export interface CausalCause {
  causeId: string;
  causeName: string;
  causeTimestamp: string | null;
  distance: number;
  confidenceScores: number[];
  aggregateConfidence: number;
  temporalOrderingValid: boolean;
}

/** Causal effect from find_effects query */
export interface CausalEffect {
  effectId: string;
  effectName: string;
  effectTimestamp: string | null;
  distance: number;
  confidenceScores: number[];
  aggregateConfidence: number;
  temporalOrderingValid: boolean;
}

/** Causal path between two events */
export interface CausalPath {
  startEventId: string;
  endEventId: string;
  path: string[];
  causalConfidence: number;
  confidenceScores: number[];
  temporalOrderingValid: boolean;
  causalEvidence: string[];
}

/** Response from find_causes command */
interface FindCausesResponse {
  success: boolean;
  targetEventId: string;
  causes: CausalCause[];
  maxHopsRequested: number;
  minConfidenceRequested: number;
  queryTimeMs: number;
  error?: string;
}

/** Response from find_effects command */
interface FindEffectsResponse {
  success: boolean;
  sourceEventId: string;
  effects: CausalEffect[];
  maxHopsRequested: number;
  minConfidenceRequested: number;
  queryTimeMs: number;
  error?: string;
}

/** Response from find_causal_path command */
interface FindCausalPathResponse {
  success: boolean;
  pathFound: boolean;
  path: CausalPath | null;
  startEventId: string;
  endEventId: string;
  maxHopsRequested: number;
  queryTimeMs: number;
  error?: string;
}

// ============================================================================
// Store
// ============================================================================

interface CausalState {
  /** Anchor event ID for exploration */
  anchorEventId: string | null;
  /** Causes of the anchor event */
  causes: CausalCause[];
  /** Effects of the anchor event */
  effects: CausalEffect[];
  /** Selected causal path */
  selectedPath: CausalPath | null;
  /** Loading state */
  isLoading: boolean;
  /** Error message */
  error: string | null;
  /** Query timing */
  lastQueryTimeMs: number | null;

  // Actions
  /** Set anchor event and clear results */
  setAnchorEvent: (eventId: string | null) => void;
  /** Find causes of an event */
  findCauses: (eventId: string, maxHops?: number, minConfidence?: number) => Promise<void>;
  /** Find effects of an event */
  findEffects: (eventId: string, maxHops?: number, minConfidence?: number) => Promise<void>;
  /** Find causal path between two events */
  findPath: (startId: string, endId: string, maxHops?: number) => Promise<void>;
  /** Clear all results */
  clearResults: () => void;
  /** Clear error */
  clearError: () => void;
}

export const useCausalStore = create<CausalState>()((set, get) => ({
  // Initial state
  anchorEventId: null,
  causes: [],
  effects: [],
  selectedPath: null,
  isLoading: false,
  error: null,
  lastQueryTimeMs: null,

  // Set anchor event
  setAnchorEvent: (eventId) => {
    set({
      anchorEventId: eventId,
      causes: [],
      effects: [],
      selectedPath: null,
      error: null,
    });
  },

  // Find causes of an event
  findCauses: async (eventId, maxHops = 3, minConfidence = 0.6) => {
    set({ isLoading: true, error: null, anchorEventId: eventId });

    try {
      const response = await invoke<FindCausesResponse>('find_causes', {
        eventId,
        maxHops,
        minConfidence,
      });

      if (response.success) {
        set({
          causes: response.causes,
          isLoading: false,
          lastQueryTimeMs: response.queryTimeMs,
        });
      } else {
        set({
          error: response.error || 'Failed to find causes',
          isLoading: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Find causes failed';
      set({ error: errorMsg, isLoading: false });
    }
  },

  // Find effects of an event
  findEffects: async (eventId, maxHops = 3, minConfidence = 0.6) => {
    set({ isLoading: true, error: null, anchorEventId: eventId });

    try {
      const response = await invoke<FindEffectsResponse>('find_effects', {
        eventId,
        maxHops,
        minConfidence,
      });

      if (response.success) {
        set({
          effects: response.effects,
          isLoading: false,
          lastQueryTimeMs: response.queryTimeMs,
        });
      } else {
        set({
          error: response.error || 'Failed to find effects',
          isLoading: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Find effects failed';
      set({ error: errorMsg, isLoading: false });
    }
  },

  // Find causal path between two events
  findPath: async (startId, endId, maxHops = 5) => {
    set({ isLoading: true, error: null });

    try {
      const response = await invoke<FindCausalPathResponse>('find_causal_path', {
        startId,
        endId,
        maxHops,
      });

      if (response.success) {
        set({
          selectedPath: response.pathFound ? response.path : null,
          isLoading: false,
          lastQueryTimeMs: response.queryTimeMs,
        });

        if (!response.pathFound) {
          set({ error: `No causal path found from ${startId} to ${endId}` });
        }
      } else {
        set({
          error: response.error || 'Failed to find causal path',
          isLoading: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Find causal path failed';
      set({ error: errorMsg, isLoading: false });
    }
  },

  // Clear all results
  clearResults: () => {
    set({
      anchorEventId: null,
      causes: [],
      effects: [],
      selectedPath: null,
      error: null,
      lastQueryTimeMs: null,
    });
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },
}));

export default useCausalStore;

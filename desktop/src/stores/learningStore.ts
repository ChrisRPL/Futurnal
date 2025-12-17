/**
 * Learning Store - Zustand state management for experiential learning
 *
 * Step 08: Frontend Intelligence Integration - Phase 6
 *
 * Research Foundation:
 * - RLHI: Reinforcement Learning from Human Interactions
 * - AgentFlow: Learning from user feedback
 * - Option B: Ghost frozen, learning via token priors
 *
 * Manages learning progress metrics and quality gates.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Quality progression metrics */
export interface QualityProgression {
  before: number;
  after: number;
  improvement: number;
}

/** Pattern learning statistics */
export interface PatternLearning {
  entityPriors: number;
  relationPriors: number;
  temporalPriors: number;
}

/** Quality gates status */
export interface QualityGates {
  ghostFrozen: boolean;
  improvementThreshold: number;
  meetsThreshold: boolean;
}

/** Response from get_learning_progress command */
interface LearningProgressResponse {
  success: boolean;
  documentsProcessed: number;
  successRate: number;
  qualityProgression: QualityProgression;
  patternLearning: PatternLearning;
  qualityGates: QualityGates;
  error?: string;
}

// ============================================================================
// Store
// ============================================================================

interface LearningState {
  /** Documents processed count */
  documentsProcessed: number;
  /** Success rate (0-1) */
  successRate: number;
  /** Quality progression */
  qualityProgression: QualityProgression | null;
  /** Pattern learning stats */
  patternLearning: PatternLearning | null;
  /** Quality gates status */
  qualityGates: QualityGates | null;
  /** Loading state */
  isLoading: boolean;
  /** Error message */
  error: string | null;

  // Actions
  /** Fetch learning progress */
  fetchLearningProgress: () => Promise<void>;
  /** Clear error */
  clearError: () => void;
}

export const useLearningStore = create<LearningState>()((set) => ({
  // Initial state
  documentsProcessed: 0,
  successRate: 0,
  qualityProgression: null,
  patternLearning: null,
  qualityGates: null,
  isLoading: false,
  error: null,

  // Fetch learning progress
  fetchLearningProgress: async () => {
    set({ isLoading: true, error: null });

    try {
      const response = await invoke<LearningProgressResponse>('get_learning_progress');

      if (response.success) {
        set({
          documentsProcessed: response.documentsProcessed,
          successRate: response.successRate,
          qualityProgression: response.qualityProgression,
          patternLearning: response.patternLearning,
          qualityGates: response.qualityGates,
          isLoading: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch learning progress',
          isLoading: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Fetch learning progress failed';
      set({ error: errorMsg, isLoading: false });
    }
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },
}));

export default useLearningStore;

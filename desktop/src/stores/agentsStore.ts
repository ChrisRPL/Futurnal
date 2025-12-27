/**
 * Agents Store - Zustand state management for Phase 2E AgentFlow
 *
 * Manages memory buffer, correlation hypotheses, and verification state.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Memory buffer entry. */
export interface MemoryEntry {
  entryId: string;
  entryType: string;
  content: string;
  priority: string;
  timestamp: string;
  relatedEntries: string[];
  metadata: Record<string, unknown>;
  accessCount: number;
  lastAccessed?: string;
}

/** Memory buffer statistics. */
export interface MemoryStats {
  totalEntries: number;
  maxEntries: number;
  utilization: number;
  byType: Record<string, number>;
  byPriority: Record<string, number>;
}

/** Correlation hypothesis. */
export interface Hypothesis {
  hypothesisId: string;
  hypothesisType: string;
  description: string;
  eventTypeA: string;
  eventTypeB: string;
  confidence: number;
  evidenceFor: string[];
  evidenceAgainst: string[];
  status: string;
  createdAt: string;
  lastUpdated: string;
  metadata: Record<string, unknown>;
}

/** Query plan for investigating a hypothesis. */
export interface QueryPlan {
  hypothesisId: string;
  queries: unknown[];
  expectedResults: string[];
  completionCriteria: string;
}

/** Verification report for a hypothesis. */
export interface VerificationReport {
  hypothesisId: string;
  result: string;
  confidence: number;
  evidenceSummary: string;
  criteriaMet: string[];
  criteriaViolated: string[];
  recommendation: string;
  createdAt: string;
}

/** AgentFlow system status. */
export interface AgentFlowStatus {
  memoryBuffer: unknown;
  correlationPlanner: unknown;
  correlationVerifier: unknown;
}

// ============================================================================
// Response Types
// ============================================================================

interface MemoryStatsResponse {
  success: boolean;
  totalEntries: number;
  maxEntries: number;
  utilization: number;
  byType: Record<string, number>;
  byPriority: Record<string, number>;
  error?: string;
}

interface MemoryEntriesResponse {
  success: boolean;
  entries: MemoryEntry[];
  count: number;
  error?: string;
}

interface MemorySearchResponse {
  success: boolean;
  query?: string;
  entries: MemoryEntry[];
  count: number;
  error?: string;
}

interface HypothesesResponse {
  success: boolean;
  hypotheses: Hypothesis[];
  count: number;
  error?: string;
}

interface InvestigateResponse {
  success: boolean;
  hypothesisId?: string;
  queryPlan?: QueryPlan;
  error?: string;
}

interface VerifyResponse {
  success: boolean;
  hypothesisId?: string;
  report?: VerificationReport;
  error?: string;
}

interface VerificationHistoryResponse {
  success: boolean;
  hypothesisId?: string;
  history: VerificationReport[];
  count: number;
  error?: string;
}

interface AgentFlowStatusResponse {
  success: boolean;
  memoryBuffer?: unknown;
  correlationPlanner?: unknown;
  correlationVerifier?: unknown;
  error?: string;
}

interface MemoryClearResponse {
  success: boolean;
  clearedCount: number;
  error?: string;
}

interface ExportPriorsResponse {
  success: boolean;
  content: string;
  error?: string;
}

// ============================================================================
// Store
// ============================================================================

interface AgentsState {
  // Memory buffer
  memoryStats: MemoryStats | null;
  recentEntries: MemoryEntry[];
  searchResults: MemoryEntry[];
  lastSearchQuery: string | null;

  // Hypotheses
  hypotheses: Hypothesis[];
  selectedHypothesis: Hypothesis | null;
  currentQueryPlan: QueryPlan | null;
  verificationHistory: VerificationReport[];

  // Status
  agentFlowStatus: AgentFlowStatus | null;
  tokenPriors: string | null;

  // Loading states
  isLoadingMemory: boolean;
  isLoadingHypotheses: boolean;
  isLoadingStatus: boolean;
  isVerifying: boolean;
  isInvestigating: boolean;

  // Error states
  error: string | null;

  // Actions - Memory Buffer
  fetchMemoryStats: () => Promise<void>;
  fetchRecentEntries: (limit?: number) => Promise<void>;
  searchMemory: (query: string, limit?: number) => Promise<void>;
  clearMemory: () => Promise<number>;

  // Actions - Hypotheses
  fetchHypotheses: (status?: string) => Promise<void>;
  generateHypotheses: (eventTypes: string) => Promise<number>;
  selectHypothesis: (hypothesis: Hypothesis | null) => void;
  investigateHypothesis: (hypothesisId: string) => Promise<QueryPlan | null>;
  verifyHypothesis: (hypothesisId: string) => Promise<VerificationReport | null>;
  fetchVerificationHistory: (hypothesisId: string) => Promise<void>;

  // Actions - Status
  fetchAgentFlowStatus: () => Promise<void>;
  exportTokenPriors: () => Promise<string | null>;

  // Actions - Utility
  clearError: () => void;
  refreshAll: () => Promise<void>;
}

export const useAgentsStore = create<AgentsState>()((set, get) => ({
  // Initial state
  memoryStats: null,
  recentEntries: [],
  searchResults: [],
  lastSearchQuery: null,
  hypotheses: [],
  selectedHypothesis: null,
  currentQueryPlan: null,
  verificationHistory: [],
  agentFlowStatus: null,
  tokenPriors: null,
  isLoadingMemory: false,
  isLoadingHypotheses: false,
  isLoadingStatus: false,
  isVerifying: false,
  isInvestigating: false,
  error: null,

  // -------------------------------------------------------------------------
  // Memory Buffer Actions
  // -------------------------------------------------------------------------

  fetchMemoryStats: async () => {
    set({ isLoadingMemory: true, error: null });

    try {
      const response = await invoke<MemoryStatsResponse>('get_memory_stats');

      if (response.success) {
        set({
          memoryStats: {
            totalEntries: response.totalEntries,
            maxEntries: response.maxEntries,
            utilization: response.utilization,
            byType: response.byType,
            byPriority: response.byPriority,
          },
          isLoadingMemory: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch memory stats',
          isLoadingMemory: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch memory stats failed';
      set({ error: errorMsg, isLoadingMemory: false });
    }
  },

  fetchRecentEntries: async (limit?: number) => {
    set({ isLoadingMemory: true, error: null });

    try {
      const response = await invoke<MemoryEntriesResponse>('get_memory_recent', {
        limit,
      });

      if (response.success) {
        set({
          recentEntries: response.entries,
          isLoadingMemory: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch recent entries',
          isLoadingMemory: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch recent entries failed';
      set({ error: errorMsg, isLoadingMemory: false });
    }
  },

  searchMemory: async (query: string, limit?: number) => {
    set({ isLoadingMemory: true, error: null, lastSearchQuery: query });

    try {
      const response = await invoke<MemorySearchResponse>('search_memory', {
        query,
        limit,
      });

      if (response.success) {
        set({
          searchResults: response.entries,
          isLoadingMemory: false,
        });
      } else {
        set({
          error: response.error || 'Failed to search memory',
          isLoadingMemory: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Search memory failed';
      set({ error: errorMsg, isLoadingMemory: false });
    }
  },

  clearMemory: async () => {
    try {
      const response = await invoke<MemoryClearResponse>('clear_memory');

      if (response.success) {
        set({
          memoryStats: null,
          recentEntries: [],
          searchResults: [],
        });
        return response.clearedCount;
      }
      return 0;
    } catch (error) {
      console.error('Failed to clear memory:', error);
      return 0;
    }
  },

  // -------------------------------------------------------------------------
  // Hypotheses Actions
  // -------------------------------------------------------------------------

  fetchHypotheses: async (status?: string) => {
    set({ isLoadingHypotheses: true, error: null });

    try {
      const response = await invoke<HypothesesResponse>('get_hypotheses', {
        status,
      });

      if (response.success) {
        set({
          hypotheses: response.hypotheses,
          isLoadingHypotheses: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch hypotheses',
          isLoadingHypotheses: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch hypotheses failed';
      set({ error: errorMsg, isLoadingHypotheses: false });
    }
  },

  generateHypotheses: async (eventTypes: string) => {
    set({ isLoadingHypotheses: true, error: null });

    try {
      const response = await invoke<HypothesesResponse>('generate_hypotheses', {
        eventTypes,
      });

      if (response.success) {
        set({
          hypotheses: response.hypotheses,
          isLoadingHypotheses: false,
        });
        return response.count;
      } else {
        set({
          error: response.error || 'Failed to generate hypotheses',
          isLoadingHypotheses: false,
        });
        return 0;
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Generate hypotheses failed';
      set({ error: errorMsg, isLoadingHypotheses: false });
      return 0;
    }
  },

  selectHypothesis: (hypothesis: Hypothesis | null) => {
    set({ selectedHypothesis: hypothesis });
  },

  investigateHypothesis: async (hypothesisId: string) => {
    set({ isInvestigating: true, error: null });

    try {
      const response = await invoke<InvestigateResponse>(
        'investigate_hypothesis',
        { hypothesisId }
      );

      if (response.success && response.queryPlan) {
        set({
          currentQueryPlan: response.queryPlan,
          isInvestigating: false,
        });
        return response.queryPlan;
      } else {
        set({
          error: response.error || 'Failed to investigate hypothesis',
          isInvestigating: false,
        });
        return null;
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Investigate hypothesis failed';
      set({ error: errorMsg, isInvestigating: false });
      return null;
    }
  },

  verifyHypothesis: async (hypothesisId: string) => {
    set({ isVerifying: true, error: null });

    try {
      const response = await invoke<VerifyResponse>('verify_hypothesis', {
        hypothesisId,
      });

      if (response.success && response.report) {
        // Refresh hypotheses to get updated status
        await get().fetchHypotheses();
        set({ isVerifying: false });
        return response.report;
      } else {
        set({
          error: response.error || 'Failed to verify hypothesis',
          isVerifying: false,
        });
        return null;
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Verify hypothesis failed';
      set({ error: errorMsg, isVerifying: false });
      return null;
    }
  },

  fetchVerificationHistory: async (hypothesisId: string) => {
    try {
      const response = await invoke<VerificationHistoryResponse>(
        'get_verification_history',
        { hypothesisId }
      );

      if (response.success) {
        set({ verificationHistory: response.history });
      }
    } catch (error) {
      console.error('Failed to fetch verification history:', error);
    }
  },

  // -------------------------------------------------------------------------
  // Status Actions
  // -------------------------------------------------------------------------

  fetchAgentFlowStatus: async () => {
    set({ isLoadingStatus: true, error: null });

    try {
      const response = await invoke<AgentFlowStatusResponse>(
        'get_agentflow_status'
      );

      if (response.success) {
        set({
          agentFlowStatus: {
            memoryBuffer: response.memoryBuffer,
            correlationPlanner: response.correlationPlanner,
            correlationVerifier: response.correlationVerifier,
          },
          isLoadingStatus: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch AgentFlow status',
          isLoadingStatus: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch status failed';
      set({ error: errorMsg, isLoadingStatus: false });
    }
  },

  exportTokenPriors: async () => {
    try {
      const response = await invoke<ExportPriorsResponse>('export_token_priors');

      if (response.success) {
        set({ tokenPriors: response.content });
        return response.content;
      }
      return null;
    } catch (error) {
      console.error('Failed to export token priors:', error);
      return null;
    }
  },

  // -------------------------------------------------------------------------
  // Utility Actions
  // -------------------------------------------------------------------------

  clearError: () => {
    set({ error: null });
  },

  refreshAll: async () => {
    const state = get();
    await Promise.all([
      state.fetchMemoryStats(),
      state.fetchRecentEntries(),
      state.fetchHypotheses(),
      state.fetchAgentFlowStatus(),
    ]);
  },
}));

export default useAgentsStore;

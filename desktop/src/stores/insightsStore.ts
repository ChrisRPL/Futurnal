/**
 * Insights Store - Zustand state management for emergent insights
 *
 * AGI Phase 8: Frontend Integration
 *
 * Research Foundation:
 * - CuriosityEngine: Information-gain gap detection
 * - EmergentInsights: Correlation to NL insights
 * - ICDA (2024): Interactive Causal Discovery
 *
 * Manages insights, knowledge gaps, and causal verifications.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Insight type classification */
export type InsightType =
  | 'correlation'
  | 'causal_hypothesis'
  | 'pattern'
  | 'anomaly'
  | 'trend'
  | 'knowledge_gap'
  // Extended types from Python backend
  | 'temporal_correlation'
  | 'behavioral_pattern'
  | 'aspiration_misalignment'
  | 'productivity_pattern'
  | 'weekly_rhythm'
  | 'sequence_pattern';

/** Insight priority level */
export type InsightPriority = 'high' | 'medium' | 'low';

/** An emergent insight from the intelligence engine */
export interface EmergentInsight {
  insightId: string;
  insightType: InsightType;
  title: string;
  description: string;
  confidence: number;
  relevance: number;
  priority: InsightPriority;
  sourceEvents: string[];
  suggestedActions: string[];
  createdAt: string;
  expiresAt?: string;
  isRead: boolean;
}

/** Knowledge gap type */
export type GapType =
  | 'isolated_cluster'
  | 'forgotten_memory'
  | 'bridge_opportunity'
  | 'missing_synthesis'
  | 'aspiration_disconnect';

/** A knowledge gap detected by CuriosityEngine */
export interface KnowledgeGap {
  gapId: string;
  gapType: GapType;
  title: string;
  description: string;
  informationGain: number;
  relatedTopics: string[];
  explorationPrompts: string[];
  createdAt: string;
  isAddressed: boolean;
}

/** Causal response type from user */
export type CausalResponseType =
  | 'yes_causal'
  | 'no_correlation'
  | 'reverse_causation'
  | 'confounder'
  | 'uncertain'
  | 'skip';

/** Response option for causal question */
export interface CausalResponseOption {
  value: CausalResponseType;
  label: string;
}

/** A causal verification question from ICDA */
export interface CausalVerificationQuestion {
  questionId: string;
  candidateId: string;
  causeEvent: string;
  effectEvent: string;
  mainQuestion: string;
  context: string;
  evidenceSummary: string;
  responseOptions: CausalResponseOption[];
  initialConfidence: number;
}

/** Insight statistics */
export interface InsightStats {
  totalInsights: number;
  unreadInsights: number;
  totalGaps: number;
  pendingVerifications: number;
  verifiedCausalCount: number;
  lastScanAt?: string;
}

// ============================================================================
// Response Types
// ============================================================================

interface InsightsResponse {
  success: boolean;
  insights: EmergentInsight[];
  totalCount: number;
  unreadCount: number;
  error?: string;
}

interface KnowledgeGapsResponse {
  success: boolean;
  gaps: KnowledgeGap[];
  totalCount: number;
  error?: string;
}

interface PendingVerificationsResponse {
  success: boolean;
  questions: CausalVerificationQuestion[];
  totalPending: number;
  error?: string;
}

interface VerificationResultResponse {
  success: boolean;
  candidateId: string;
  newConfidence: number;
  confidenceDelta: number;
  status: string;
  error?: string;
}

interface InsightStatsResponse {
  success: boolean;
  totalInsights: number;
  unreadInsights: number;
  totalGaps: number;
  pendingVerifications: number;
  verifiedCausalCount: number;
  lastScanAt?: string;
  error?: string;
}

// ============================================================================
// Store
// ============================================================================

interface InsightsState {
  // Emergent insights
  insights: EmergentInsight[];
  totalInsights: number;
  unreadCount: number;

  // Knowledge gaps
  gaps: KnowledgeGap[];
  totalGaps: number;

  // Causal verifications
  pendingVerifications: CausalVerificationQuestion[];
  totalPending: number;
  verifiedCount: number;

  // Statistics
  stats: InsightStats | null;
  lastScanAt: string | null;

  // Loading states
  isLoadingInsights: boolean;
  isLoadingGaps: boolean;
  isLoadingVerifications: boolean;
  isScanning: boolean;

  // Error states
  error: string | null;

  // Actions - Insights
  fetchInsights: (type?: InsightType, limit?: number) => Promise<void>;
  markInsightRead: (insightId: string) => Promise<void>;
  dismissInsight: (insightId: string) => Promise<void>;

  // Actions - Knowledge Gaps
  fetchKnowledgeGaps: (limit?: number) => Promise<void>;
  markGapAddressed: (gapId: string) => Promise<void>;

  // Actions - Causal Verification
  fetchPendingVerifications: (limit?: number) => Promise<void>;
  submitVerification: (
    questionId: string,
    response: CausalResponseType,
    explanation?: string
  ) => Promise<VerificationResultResponse | null>;

  // Actions - Statistics
  fetchStats: () => Promise<void>;
  triggerScan: () => Promise<void>;

  // Actions - Utility
  clearError: () => void;
  refreshAll: () => Promise<void>;
}

export const useInsightsStore = create<InsightsState>()((set, get) => ({
  // Initial state
  insights: [],
  totalInsights: 0,
  unreadCount: 0,

  gaps: [],
  totalGaps: 0,

  pendingVerifications: [],
  totalPending: 0,
  verifiedCount: 0,

  stats: null,
  lastScanAt: null,

  isLoadingInsights: false,
  isLoadingGaps: false,
  isLoadingVerifications: false,
  isScanning: false,

  error: null,

  // -------------------------------------------------------------------------
  // Insights Actions
  // -------------------------------------------------------------------------

  fetchInsights: async (type?: InsightType, limit?: number) => {
    set({ isLoadingInsights: true, error: null });

    try {
      const response = await invoke<InsightsResponse>('get_insights', {
        insightType: type,
        limit,
      });

      if (response.success) {
        set({
          insights: response.insights,
          totalInsights: response.totalCount,
          unreadCount: response.unreadCount,
          isLoadingInsights: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch insights',
          isLoadingInsights: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Fetch insights failed';
      set({ error: errorMsg, isLoadingInsights: false });
    }
  },

  markInsightRead: async (insightId: string) => {
    try {
      const success = await invoke<boolean>('mark_insight_read', { insightId });

      if (success) {
        set((state) => ({
          insights: state.insights.map((i) =>
            i.insightId === insightId ? { ...i, isRead: true } : i
          ),
          unreadCount: Math.max(0, state.unreadCount - 1),
        }));
      }
    } catch (error) {
      console.error('Failed to mark insight read:', error);
    }
  },

  dismissInsight: async (insightId: string) => {
    try {
      const success = await invoke<boolean>('dismiss_insight', { insightId });

      if (success) {
        set((state) => ({
          insights: state.insights.filter((i) => i.insightId !== insightId),
          totalInsights: state.totalInsights - 1,
          unreadCount: state.insights.find((i) => i.insightId === insightId)?.isRead
            ? state.unreadCount
            : state.unreadCount - 1,
        }));
      }
    } catch (error) {
      console.error('Failed to dismiss insight:', error);
    }
  },

  // -------------------------------------------------------------------------
  // Knowledge Gaps Actions
  // -------------------------------------------------------------------------

  fetchKnowledgeGaps: async (limit?: number) => {
    set({ isLoadingGaps: true, error: null });

    try {
      const response = await invoke<KnowledgeGapsResponse>('get_knowledge_gaps', {
        limit,
      });

      if (response.success) {
        set({
          gaps: response.gaps,
          totalGaps: response.totalCount,
          isLoadingGaps: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch knowledge gaps',
          isLoadingGaps: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Fetch gaps failed';
      set({ error: errorMsg, isLoadingGaps: false });
    }
  },

  markGapAddressed: async (gapId: string) => {
    try {
      const success = await invoke<boolean>('mark_gap_addressed', { gapId });

      if (success) {
        set((state) => ({
          gaps: state.gaps.map((g) =>
            g.gapId === gapId ? { ...g, isAddressed: true } : g
          ),
        }));
      }
    } catch (error) {
      console.error('Failed to mark gap addressed:', error);
    }
  },

  // -------------------------------------------------------------------------
  // Causal Verification Actions
  // -------------------------------------------------------------------------

  fetchPendingVerifications: async (limit?: number) => {
    set({ isLoadingVerifications: true, error: null });

    try {
      const response = await invoke<PendingVerificationsResponse>(
        'get_pending_verifications',
        { limit }
      );

      if (response.success) {
        set({
          pendingVerifications: response.questions,
          totalPending: response.totalPending,
          isLoadingVerifications: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch pending verifications',
          isLoadingVerifications: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch verifications failed';
      set({ error: errorMsg, isLoadingVerifications: false });
    }
  },

  submitVerification: async (
    questionId: string,
    response: CausalResponseType,
    explanation?: string
  ) => {
    try {
      const result = await invoke<VerificationResultResponse>(
        'submit_causal_verification',
        {
          request: {
            questionId,
            response,
            explanation,
          },
        }
      );

      if (result.success) {
        // Remove from pending
        set((state) => ({
          pendingVerifications: state.pendingVerifications.filter(
            (q) => q.questionId !== questionId
          ),
          totalPending: state.totalPending - 1,
          verifiedCount: state.verifiedCount + 1,
        }));
      }

      return result;
    } catch (error) {
      console.error('Failed to submit verification:', error);
      return null;
    }
  },

  // -------------------------------------------------------------------------
  // Statistics Actions
  // -------------------------------------------------------------------------

  fetchStats: async () => {
    try {
      const response = await invoke<InsightStatsResponse>('get_insight_stats');

      if (response.success) {
        set({
          stats: {
            totalInsights: response.totalInsights,
            unreadInsights: response.unreadInsights,
            totalGaps: response.totalGaps,
            pendingVerifications: response.pendingVerifications,
            verifiedCausalCount: response.verifiedCausalCount,
            lastScanAt: response.lastScanAt,
          },
          lastScanAt: response.lastScanAt || null,
        });
      }
    } catch (error) {
      console.error('Failed to fetch insight stats:', error);
    }
  },

  triggerScan: async () => {
    set({ isScanning: true, error: null });

    try {
      const response = await invoke<InsightStatsResponse>('trigger_insight_scan');

      if (response.success) {
        set({
          stats: {
            totalInsights: response.totalInsights,
            unreadInsights: response.unreadInsights,
            totalGaps: response.totalGaps,
            pendingVerifications: response.pendingVerifications,
            verifiedCausalCount: response.verifiedCausalCount,
            lastScanAt: response.lastScanAt,
          },
          lastScanAt: response.lastScanAt || null,
          isScanning: false,
        });

        // Refresh data after scan
        const state = get();
        await Promise.all([
          state.fetchInsights(),
          state.fetchKnowledgeGaps(),
          state.fetchPendingVerifications(),
        ]);
      } else {
        set({
          error: response.error || 'Failed to trigger scan',
          isScanning: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Scan failed';
      set({ error: errorMsg, isScanning: false });
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
      state.fetchInsights(),
      state.fetchKnowledgeGaps(),
      state.fetchPendingVerifications(),
      state.fetchStats(),
    ]);
  },
}));

export default useInsightsStore;

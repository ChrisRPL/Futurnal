/**
 * Insights Page - Emergent insights, knowledge gaps, and causal verification
 *
 * AGI Phase 8: Frontend Integration
 *
 * Features:
 * - InsightsFeed with tabbed navigation
 * - Knowledge gap exploration
 * - ICDA causal verification workflow
 * - Learning progress indicator
 */

import { useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Lightbulb,
  Brain,
  GitBranch,
  TrendingUp,
  ArrowLeft,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { ThemeLogo } from '@/components/ThemeLogo';
import { InsightsFeed, AgentFlowPanel, PatternVisualization } from '@/components/insights';
import { useLearningStore } from '@/stores/learningStore';
import type { EmergentInsight, KnowledgeGap } from '@/stores/insightsStore';

/**
 * Learning Progress Card
 */
function LearningProgressCard() {
  const {
    documentsProcessed,
    successRate,
    qualityProgression,
    patternLearning,
    qualityGates,
    isLoading,
    fetchLearningProgress,
  } = useLearningStore();

  useEffect(() => {
    fetchLearningProgress();
  }, [fetchLearningProgress]);

  if (isLoading) {
    return (
      <div className="p-4 bg-white/[0.02] border border-white/10 animate-pulse">
        <div className="h-4 w-32 bg-white/10 rounded mb-3" />
        <div className="h-8 w-24 bg-white/10 rounded" />
      </div>
    );
  }

  const improvementPercent = qualityProgression
    ? Math.round(qualityProgression.improvement * 100)
    : 0;

  return (
    <div className="p-4 bg-white/[0.02] border border-white/10">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="w-4 h-4 text-white/60" />
        <span className="text-sm font-medium text-white/80">
          Learning Progress
        </span>
        {qualityGates?.ghostFrozen && (
          <span className="ml-auto px-1.5 py-0.5 text-[10px] bg-emerald-500/20 text-emerald-400 rounded">
            Ghost Frozen
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Documents processed */}
        <div>
          <div className="text-2xl font-bold text-white/90">
            {documentsProcessed}
          </div>
          <div className="text-xs text-white/40">Documents processed</div>
        </div>

        {/* Success rate */}
        <div>
          <div className="text-2xl font-bold text-white/90">
            {Math.round(successRate * 100)}%
          </div>
          <div className="text-xs text-white/40">Success rate</div>
        </div>
      </div>

      {/* Quality improvement bar */}
      {qualityProgression && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-white/50 mb-1">
            <span>Quality improvement</span>
            <span className={improvementPercent >= 0 ? 'text-emerald-400' : 'text-red-400'}>
              {improvementPercent >= 0 ? '+' : ''}{improvementPercent}%
            </span>
          </div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                improvementPercent >= 5 ? 'bg-emerald-400' :
                improvementPercent >= 0 ? 'bg-white/40' : 'bg-red-400'
              )}
              style={{ width: `${Math.min(100, Math.abs(improvementPercent) * 10)}%` }}
            />
          </div>
        </div>
      )}

      {/* Pattern learning stats */}
      {patternLearning && (
        <div className="mt-4 flex items-center gap-4 text-xs text-white/40">
          <span>{patternLearning.entityPriors} entity priors</span>
          <span>{patternLearning.relationPriors} relation priors</span>
          <span>{patternLearning.temporalPriors} temporal priors</span>
        </div>
      )}
    </div>
  );
}

/**
 * Insights Page Component
 */
export function InsightsPage() {
  const navigate = useNavigate();

  // Handle exploring a knowledge gap
  const handleExploreGap = (prompt: string, gap: KnowledgeGap) => {
    // Navigate to chat with the exploration prompt
    navigate('/dashboard', { state: { chatPrompt: prompt } });
  };

  // Handle insight actions
  const handleInsightAction = (action: string, insight: EmergentInsight) => {
    console.log('[Insights] Action:', action, 'for insight:', insight.insightId);

    if (action === 'explore') {
      // Navigate to graph with related entities
      if (insight.sourceEvents.length > 0) {
        navigate(`/graph?select=${encodeURIComponent(insight.sourceEvents[0])}`);
      }
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--color-border)] flex-shrink-0">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Back + Logo */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/dashboard')}
              className="p-1.5 text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors"
              title="Back to dashboard"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <Link to="/dashboard" className="no-underline">
              <ThemeLogo variant="horizontal" className="h-6 w-auto" />
            </Link>
          </div>

          {/* Title */}
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[var(--color-text-muted)]" />
            <span className="text-sm font-medium text-[var(--color-text-primary)]">
              Intelligence Insights
            </span>
          </div>

          {/* Placeholder for right side */}
          <div className="w-24" />
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Left sidebar - Learning Progress */}
        <aside className="w-80 border-r border-[var(--color-border)] p-4 space-y-4 hidden lg:block">
          <LearningProgressCard />

          {/* Quick stats */}
          <div className="p-4 bg-white/[0.02] border border-white/10">
            <div className="text-xs font-medium text-white/50 uppercase tracking-wider mb-3">
              Intelligence Overview
            </div>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-white/5 rounded">
                  <Lightbulb className="w-4 h-4 text-white/60" />
                </div>
                <div>
                  <div className="text-sm text-white/80">Emergent Insights</div>
                  <div className="text-xs text-white/40">
                    Patterns detected in your data
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-white/5 rounded">
                  <Brain className="w-4 h-4 text-white/60" />
                </div>
                <div>
                  <div className="text-sm text-white/80">Knowledge Gaps</div>
                  <div className="text-xs text-white/40">
                    Areas worth exploring
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-white/5 rounded">
                  <GitBranch className="w-4 h-4 text-white/60" />
                </div>
                <div>
                  <div className="text-sm text-white/80">Causal Verification</div>
                  <div className="text-xs text-white/40">
                    Validate cause-effect relationships
                  </div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main content - InsightsFeed */}
        <main className="flex-1 flex flex-col min-h-0">
          <InsightsFeed
            onExploreGap={handleExploreGap}
            onInsightAction={handleInsightAction}
            className="flex-1"
          />
        </main>

        {/* Right sidebar - AgentFlow and Patterns */}
        <aside className="w-96 border-l border-[var(--color-border)] p-4 space-y-4 hidden xl:block overflow-y-auto">
          <PatternVisualization />
          <AgentFlowPanel />
        </aside>
      </div>
    </div>
  );
}

export default InsightsPage;

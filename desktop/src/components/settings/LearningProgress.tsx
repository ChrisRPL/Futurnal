/**
 * Learning Progress Section
 *
 * Step 08: Frontend Intelligence Integration - Phase 6
 *
 * Research Foundation:
 * - RLHI: Reinforcement Learning from Human Interactions
 * - AgentFlow: Learning from user feedback
 * - Option B: Ghost frozen, learning via token priors
 *
 * Displays experiential learning progress and quality gates.
 * Part of the Settings page.
 */

import { useEffect } from 'react';
import {
  Brain,
  Loader2,
  AlertCircle,
  RefreshCw,
  TrendingUp,
  Shield,
  FileText,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLearningStore } from '@/stores/learningStore';

/** Stat card component */
function StatCard({
  icon: Icon,
  label,
  value,
  subvalue,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  subvalue?: string;
}) {
  return (
    <div className="p-4 bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="h-4 w-4 text-[var(--color-text-tertiary)]" />
        <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-wide">
          {label}
        </span>
      </div>
      <p className="text-2xl font-medium text-[var(--color-text-primary)]">{value}</p>
      {subvalue && <p className="text-xs text-[var(--color-text-muted)] mt-1">{subvalue}</p>}
    </div>
  );
}

/** Progress indicator */
function ProgressIndicator({
  before,
  after,
  improvement,
}: {
  before: number;
  after: number;
  improvement: number;
}) {
  const improvementPercent = Math.round(improvement * 100);
  const isPositive = improvement > 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-[var(--color-text-secondary)]">Before</span>
        <span className="text-[var(--color-text-primary)]">{(before * 100).toFixed(1)}%</span>
      </div>
      <div className="flex items-center justify-between text-sm">
        <span className="text-[var(--color-text-secondary)]">After</span>
        <span className="text-[var(--color-text-primary)]">{(after * 100).toFixed(1)}%</span>
      </div>
      <div className="flex items-center justify-between text-sm pt-2 border-t border-[var(--color-border)]">
        <span className="text-[var(--color-text-secondary)]">Improvement</span>
        <span
          className={cn(
            'font-medium',
            isPositive ? 'text-white/70' : 'text-white/40'
          )}
        >
          {isPositive ? '+' : ''}
          {improvementPercent}%
        </span>
      </div>
    </div>
  );
}

export function LearningProgress() {
  const {
    documentsProcessed,
    successRate,
    qualityProgression,
    patternLearning,
    qualityGates,
    isLoading,
    error,
    fetchLearningProgress,
    clearError,
  } = useLearningStore();

  // Fetch on mount
  useEffect(() => {
    fetchLearningProgress();
  }, [fetchLearningProgress]);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">
            Learning Progress
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Ghost&apos;s improving understanding through experiential learning.
          </p>
        </div>
        <button
          onClick={() => fetchLearningProgress()}
          disabled={isLoading}
          className="p-2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn('h-4 w-4', isLoading && 'animate-spin')} />
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 text-sm text-red-400">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span className="flex-1">{error}</span>
          <button onClick={clearError} className="text-red-300 hover:text-red-200">
            Dismiss
          </button>
        </div>
      )}

      {/* Loading */}
      {isLoading && !qualityGates && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 text-[var(--color-text-muted)] animate-spin" />
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        <StatCard
          icon={FileText}
          label="Documents Processed"
          value={documentsProcessed.toLocaleString()}
        />
        <StatCard
          icon={TrendingUp}
          label="Success Rate"
          value={`${(successRate * 100).toFixed(1)}%`}
        />
      </div>

      {/* Quality Progression */}
      {qualityProgression && (
        <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h3 className="text-base font-medium text-[var(--color-text-primary)]">
              Quality Progression
            </h3>
          </div>
          <ProgressIndicator
            before={qualityProgression.before}
            after={qualityProgression.after}
            improvement={qualityProgression.improvement}
          />
        </div>
      )}

      {/* Pattern Learning */}
      {patternLearning && (
        <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h3 className="text-base font-medium text-[var(--color-text-primary)]">
              Pattern Learning
            </h3>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-2xl font-medium text-[var(--color-text-primary)]">
                {patternLearning.entityPriors}
              </p>
              <p className="text-xs text-[var(--color-text-muted)]">Entity Priors</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-medium text-[var(--color-text-primary)]">
                {patternLearning.relationPriors}
              </p>
              <p className="text-xs text-[var(--color-text-muted)]">Relation Priors</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-medium text-[var(--color-text-primary)]">
                {patternLearning.temporalPriors}
              </p>
              <p className="text-xs text-[var(--color-text-muted)]">Temporal Priors</p>
            </div>
          </div>
          <p className="text-xs text-[var(--color-text-muted)] mt-4 text-center">
            Token priors learned from experiential feedback
          </p>
        </div>
      )}

      {/* Quality Gates */}
      {qualityGates && (
        <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h3 className="text-base font-medium text-[var(--color-text-primary)]">
              Quality Gates
            </h3>
          </div>
          <div className="space-y-3">
            {/* Ghost Frozen */}
            <div className="flex items-center justify-between py-2 border-b border-[var(--color-border)]">
              <span className="text-sm text-[var(--color-text-secondary)]">Ghost Model Frozen</span>
              <div className="flex items-center gap-2">
                {qualityGates.ghostFrozen ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-white/70" />
                    <span className="text-sm text-[var(--color-text-primary)]">Yes</span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 text-white/40" />
                    <span className="text-sm text-[var(--color-text-muted)]">No</span>
                  </>
                )}
              </div>
            </div>

            {/* Improvement Threshold */}
            <div className="flex items-center justify-between py-2 border-b border-[var(--color-border)]">
              <span className="text-sm text-[var(--color-text-secondary)]">
                Improvement Threshold
              </span>
              <span className="text-sm text-[var(--color-text-primary)]">
                {(qualityGates.improvementThreshold * 100).toFixed(0)}%
              </span>
            </div>

            {/* Meets Threshold */}
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-[var(--color-text-secondary)]">Meets Threshold</span>
              <div className="flex items-center gap-2">
                {qualityGates.meetsThreshold ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-white/70" />
                    <span className="text-sm text-[var(--color-text-primary)]">Yes</span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 text-white/40" />
                    <span className="text-sm text-[var(--color-text-muted)]">Not yet</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <p className="text-xs text-[var(--color-text-muted)] mt-4">
            Option B: Ghost frozen, learning through token priors only
          </p>
        </div>
      )}
    </div>
  );
}

export default LearningProgress;

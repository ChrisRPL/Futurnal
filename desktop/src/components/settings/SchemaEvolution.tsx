/**
 * Schema Evolution Section
 *
 * Step 08: Frontend Intelligence Integration - Phase 5
 *
 * Research Foundation:
 * - GFM-RAG: Schema-aware graph construction
 * - ACE: Adaptive schema evolution
 *
 * Displays schema statistics and evolution timeline.
 * Part of the Settings page.
 */

import { useEffect } from 'react';
import {
  Database,
  GitBranch,
  Loader2,
  AlertCircle,
  RefreshCw,
  TrendingUp,
  Link,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useSchemaStore } from '@/stores/schemaStore';

/** Progress bar component */
function ProgressBar({ value, label }: { value: number; label: string }) {
  const percentage = Math.round(value * 100);
  const color = value >= 0.8 ? 'bg-white/70' : value >= 0.6 ? 'bg-white/50' : 'bg-white/30';

  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-sm text-[var(--color-text-secondary)]">{label}</span>
      <div className="flex items-center gap-2">
        <div className="w-24 h-1.5 bg-[var(--color-surface)] overflow-hidden">
          <div className={cn('h-full', color)} style={{ width: `${percentage}%` }} />
        </div>
        <span className="text-sm text-[var(--color-text-primary)] w-10 text-right">
          {percentage}%
        </span>
      </div>
    </div>
  );
}

export function SchemaEvolution() {
  const {
    entityTypes,
    relationshipTypes,
    qualityMetrics,
    evolutionTimeline,
    isLoading,
    error,
    fetchSchemaStats,
    clearError,
  } = useSchemaStore();

  // Fetch on mount
  useEffect(() => {
    fetchSchemaStats();
  }, [fetchSchemaStats]);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">
            Schema Evolution
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Knowledge graph structure and quality metrics.
          </p>
        </div>
        <button
          onClick={() => fetchSchemaStats()}
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
      {isLoading && entityTypes.length === 0 && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 text-[var(--color-text-muted)] animate-spin" />
        </div>
      )}

      {/* Entity Types */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Database className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Entity Types ({entityTypes.length})
          </h3>
        </div>
        {entityTypes.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">No entity types discovered yet.</p>
        ) : (
          <div className="space-y-2">
            {entityTypes.slice(0, 10).map((et) => (
              <div
                key={et.type}
                className="flex items-center justify-between py-1 border-b border-[var(--color-border)] last:border-0"
              >
                <span className="text-sm text-[var(--color-text-secondary)]">{et.type}</span>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-[var(--color-text-primary)]">
                    {et.count.toLocaleString()} entities
                  </span>
                  {et.firstSeen && (
                    <span className="text-xs text-[var(--color-text-muted)]">
                      since {new Date(et.firstSeen).toLocaleDateString()}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {entityTypes.length > 10 && (
              <p className="text-xs text-[var(--color-text-muted)] pt-2">
                +{entityTypes.length - 10} more types
              </p>
            )}
          </div>
        )}
      </div>

      {/* Relationship Types */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Link className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Relationship Types ({relationshipTypes.length})
          </h3>
        </div>
        {relationshipTypes.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">
            No relationship types discovered yet.
          </p>
        ) : (
          <div className="space-y-2">
            {relationshipTypes.slice(0, 10).map((rt) => (
              <div
                key={rt.type}
                className="flex items-center justify-between py-1 border-b border-[var(--color-border)] last:border-0"
              >
                <span className="text-sm text-[var(--color-text-secondary)]">{rt.type}</span>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-[var(--color-text-primary)]">
                    {rt.count.toLocaleString()} links
                  </span>
                  <span className="text-xs text-[var(--color-text-muted)]">
                    {(rt.confidenceAvg * 100).toFixed(0)}% avg conf
                  </span>
                </div>
              </div>
            ))}
            {relationshipTypes.length > 10 && (
              <p className="text-xs text-[var(--color-text-muted)] pt-2">
                +{relationshipTypes.length - 10} more types
              </p>
            )}
          </div>
        )}
      </div>

      {/* Quality Metrics */}
      {qualityMetrics && (
        <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h3 className="text-base font-medium text-[var(--color-text-primary)]">
              Quality Metrics
            </h3>
          </div>
          <div className="space-y-3">
            <ProgressBar value={qualityMetrics.precision} label="Precision" />
            <ProgressBar value={qualityMetrics.recall} label="Recall" />
            <ProgressBar value={qualityMetrics.temporalAccuracy} label="Temporal Accuracy" />
          </div>
          <p className="text-xs text-[var(--color-text-muted)] mt-4">
            Target: Precision ≥80%, Temporal Accuracy ≥85%
          </p>
        </div>
      )}

      {/* Evolution Timeline */}
      {evolutionTimeline.length > 0 && (
        <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-3 mb-4">
            <Clock className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h3 className="text-base font-medium text-[var(--color-text-primary)]">
              Recent Changes
            </h3>
          </div>
          <div className="space-y-2">
            {evolutionTimeline.slice(0, 5).map((event, idx) => (
              <div
                key={idx}
                className="flex items-center gap-3 py-1 border-b border-[var(--color-border)] last:border-0"
              >
                <GitBranch className="h-3 w-3 text-[var(--color-text-muted)] flex-shrink-0" />
                <span className="text-sm text-[var(--color-text-secondary)] flex-1">
                  {event.details}
                </span>
                {event.timestamp && (
                  <span className="text-xs text-[var(--color-text-muted)]">
                    {new Date(event.timestamp).toLocaleDateString()}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SchemaEvolution;

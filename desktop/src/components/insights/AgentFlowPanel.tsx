/**
 * AgentFlow Panel - Dashboard for Phase 2E AgentFlow Architecture
 *
 * Features:
 * - Memory buffer statistics and recent entries
 * - Hypothesis management and investigation
 * - Verification status and history
 */

import { useEffect, useState } from 'react';
import {
  Brain,
  Database,
  GitBranch,
  Search,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  XCircle,
  HelpCircle,
  Loader2,
  Trash2,
  AlertTriangle,
  FileSearch,
  ClipboardCheck,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAgentsStore } from '@/stores/agentsStore';
import type { Hypothesis } from '@/stores/agentsStore';

// ============================================================================
// Memory Buffer Section
// ============================================================================

function MemoryBufferCard() {
  const {
    memoryStats,
    recentEntries,
    isLoadingMemory,
    fetchMemoryStats,
    fetchRecentEntries,
    clearMemory,
  } = useAgentsStore();

  const [showEntries, setShowEntries] = useState(false);

  useEffect(() => {
    fetchMemoryStats();
    fetchRecentEntries(5);
  }, [fetchMemoryStats, fetchRecentEntries]);

  const utilizationColor =
    (memoryStats?.utilization ?? 0) > 0.8
      ? 'text-red-400'
      : (memoryStats?.utilization ?? 0) > 0.5
        ? 'text-amber-400'
        : 'text-green-400';

  const handleClear = async () => {
    if (confirm('Clear all memory buffer entries? This cannot be undone.')) {
      await clearMemory();
      await fetchMemoryStats();
    }
  };

  return (
    <div className="p-4 bg-white/[0.02] border border-white/10">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-white/60" />
          <span className="text-sm font-medium text-white/80">Memory Buffer</span>
        </div>
        <button
          onClick={() => {
            fetchMemoryStats();
            fetchRecentEntries(5);
          }}
          disabled={isLoadingMemory}
          className="p-1 text-white/40 hover:text-white/60"
        >
          <RefreshCw className={cn('w-3.5 h-3.5', isLoadingMemory && 'animate-spin')} />
        </button>
      </div>

      {memoryStats ? (
        <>
          <div className="grid grid-cols-3 gap-3 mb-3">
            <div className="text-center">
              <div className="text-xl font-bold text-white/90">
                {memoryStats.totalEntries}
              </div>
              <div className="text-[10px] text-white/40">Entries</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-white/90">
                {memoryStats.maxEntries}
              </div>
              <div className="text-[10px] text-white/40">Max</div>
            </div>
            <div className="text-center">
              <div className={cn('text-xl font-bold', utilizationColor)}>
                {Math.round(memoryStats.utilization * 100)}%
              </div>
              <div className="text-[10px] text-white/40">Used</div>
            </div>
          </div>

          {/* Utilization bar */}
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden mb-3">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                memoryStats.utilization > 0.8
                  ? 'bg-red-500'
                  : memoryStats.utilization > 0.5
                    ? 'bg-amber-500'
                    : 'bg-green-500'
              )}
              style={{ width: `${memoryStats.utilization * 100}%` }}
            />
          </div>

          {/* Entry types */}
          {Object.keys(memoryStats.byType).length > 0 && (
            <div className="flex flex-wrap gap-1 mb-3">
              {Object.entries(memoryStats.byType).map(([type, count]) => (
                <span
                  key={type}
                  className="px-1.5 py-0.5 text-[10px] bg-white/5 text-white/50 rounded"
                >
                  {type}: {count as number}
                </span>
              ))}
            </div>
          )}

          {/* Recent entries toggle */}
          <button
            onClick={() => setShowEntries(!showEntries)}
            className="flex items-center gap-1 text-xs text-white/50 hover:text-white/70"
          >
            {showEntries ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            <span>Recent entries</span>
          </button>

          {showEntries && recentEntries.length > 0 && (
            <div className="mt-2 space-y-1">
              {recentEntries.map((entry) => (
                <div
                  key={entry.entryId}
                  className="p-2 bg-white/[0.02] border border-white/5 text-xs"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-white/70">{entry.entryType}</span>
                    <span className="text-[10px] text-white/30">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-white/50 line-clamp-2">{entry.content}</p>
                </div>
              ))}
            </div>
          )}

          {/* Clear button */}
          {memoryStats.totalEntries > 0 && (
            <button
              onClick={handleClear}
              className="mt-3 flex items-center gap-1 text-xs text-red-400/70 hover:text-red-400"
            >
              <Trash2 className="w-3 h-3" />
              <span>Clear buffer</span>
            </button>
          )}
        </>
      ) : isLoadingMemory ? (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="w-5 h-5 text-white/40 animate-spin" />
        </div>
      ) : (
        <div className="text-center py-4 text-xs text-white/40">
          No memory data available
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Hypothesis Card
// ============================================================================

function HypothesisCard({
  hypothesis,
  onInvestigate,
  onVerify,
  isInvestigating,
  isVerifying,
}: {
  hypothesis: Hypothesis;
  onInvestigate: (id: string) => void;
  onVerify: (id: string) => void;
  isInvestigating: boolean;
  isVerifying: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const statusColors: Record<string, string> = {
    pending: 'bg-amber-500/20 text-amber-400',
    investigating: 'bg-blue-500/20 text-blue-400',
    confirmed: 'bg-green-500/20 text-green-400',
    refuted: 'bg-red-500/20 text-red-400',
    inconclusive: 'bg-white/10 text-white/50',
  };

  const StatusIcon =
    hypothesis.status === 'confirmed'
      ? CheckCircle
      : hypothesis.status === 'refuted'
        ? XCircle
        : hypothesis.status === 'investigating'
          ? Search
          : HelpCircle;

  return (
    <div className="p-3 bg-white/[0.02] border border-white/10">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <StatusIcon className="w-4 h-4 text-white/40 flex-shrink-0" />
            <span
              className={cn(
                'px-1.5 py-0.5 text-[10px] rounded',
                statusColors[hypothesis.status] || 'bg-white/10 text-white/50'
              )}
            >
              {hypothesis.status}
            </span>
            <span className="text-[10px] text-white/30 truncate">
              {hypothesis.hypothesisType}
            </span>
          </div>

          <h4 className="text-sm font-medium text-white/90 line-clamp-2 mb-1">
            {hypothesis.description}
          </h4>

          <div className="flex items-center gap-3 text-[10px] text-white/40">
            <span>
              {hypothesis.eventTypeA} → {hypothesis.eventTypeB}
            </span>
            <span>Confidence: {Math.round(hypothesis.confidence * 100)}%</span>
          </div>
        </div>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 text-white/30 hover:text-white/60"
        >
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-white/5">
          {/* Evidence */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div>
              <div className="text-[10px] text-green-400/70 mb-1">Evidence For</div>
              <div className="space-y-0.5">
                {hypothesis.evidenceFor.length > 0 ? (
                  hypothesis.evidenceFor.slice(0, 3).map((e, i) => (
                    <div key={i} className="text-[10px] text-white/50 truncate">
                      + {e}
                    </div>
                  ))
                ) : (
                  <div className="text-[10px] text-white/30">None yet</div>
                )}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-red-400/70 mb-1">Evidence Against</div>
              <div className="space-y-0.5">
                {hypothesis.evidenceAgainst.length > 0 ? (
                  hypothesis.evidenceAgainst.slice(0, 3).map((e, i) => (
                    <div key={i} className="text-[10px] text-white/50 truncate">
                      - {e}
                    </div>
                  ))
                ) : (
                  <div className="text-[10px] text-white/30">None yet</div>
                )}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={() => onInvestigate(hypothesis.hypothesisId)}
              disabled={isInvestigating || hypothesis.status === 'confirmed' || hypothesis.status === 'refuted'}
              className={cn(
                'flex-1 flex items-center justify-center gap-1.5',
                'px-2 py-1.5 text-xs',
                'bg-white/5 hover:bg-white/10',
                'text-white/60 hover:text-white/80',
                'border border-white/10',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              {isInvestigating ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <FileSearch className="w-3 h-3" />
              )}
              <span>Investigate</span>
            </button>

            <button
              onClick={() => onVerify(hypothesis.hypothesisId)}
              disabled={isVerifying || hypothesis.status === 'confirmed' || hypothesis.status === 'refuted'}
              className={cn(
                'flex-1 flex items-center justify-center gap-1.5',
                'px-2 py-1.5 text-xs',
                'bg-white/5 hover:bg-white/10',
                'text-white/60 hover:text-white/80',
                'border border-white/10',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              {isVerifying ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <ClipboardCheck className="w-3 h-3" />
              )}
              <span>Verify</span>
            </button>
          </div>

          {/* Timestamps */}
          <div className="flex items-center gap-3 mt-3 text-[10px] text-white/30">
            <span>Created: {new Date(hypothesis.createdAt).toLocaleDateString()}</span>
            <span>Updated: {new Date(hypothesis.lastUpdated).toLocaleDateString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Hypotheses Section
// ============================================================================

function HypothesesSection() {
  const {
    hypotheses,
    isLoadingHypotheses,
    isInvestigating,
    isVerifying,
    fetchHypotheses,
    investigateHypothesis,
    verifyHypothesis,
  } = useAgentsStore();

  const [statusFilter, setStatusFilter] = useState<string | null>(null);

  useEffect(() => {
    fetchHypotheses();
  }, [fetchHypotheses]);

  const filteredHypotheses = statusFilter
    ? hypotheses.filter((h) => h.status === statusFilter)
    : hypotheses;

  const statusCounts = hypotheses.reduce(
    (acc, h) => {
      acc[h.status] = (acc[h.status] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  const handleInvestigate = async (id: string) => {
    await investigateHypothesis(id);
  };

  const handleVerify = async (id: string) => {
    const report = await verifyHypothesis(id);
    if (report) {
      console.log('[AgentFlow] Verification result:', report.result);
    }
  };

  return (
    <div className="p-4 bg-white/[0.02] border border-white/10">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-white/60" />
          <span className="text-sm font-medium text-white/80">Correlation Hypotheses</span>
        </div>
        <button
          onClick={() => fetchHypotheses()}
          disabled={isLoadingHypotheses}
          className="p-1 text-white/40 hover:text-white/60"
        >
          <RefreshCw className={cn('w-3.5 h-3.5', isLoadingHypotheses && 'animate-spin')} />
        </button>
      </div>

      {/* Status filters */}
      <div className="flex flex-wrap gap-1 mb-3">
        <button
          onClick={() => setStatusFilter(null)}
          className={cn(
            'px-2 py-1 text-[10px] rounded',
            statusFilter === null
              ? 'bg-white/20 text-white/80'
              : 'bg-white/5 text-white/50 hover:text-white/70'
          )}
        >
          All ({hypotheses.length})
        </button>
        {Object.entries(statusCounts).map(([status, count]) => (
          <button
            key={status}
            onClick={() => setStatusFilter(status)}
            className={cn(
              'px-2 py-1 text-[10px] rounded',
              statusFilter === status
                ? 'bg-white/20 text-white/80'
                : 'bg-white/5 text-white/50 hover:text-white/70'
            )}
          >
            {status} ({count})
          </button>
        ))}
      </div>

      {/* Hypotheses list */}
      {isLoadingHypotheses ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-5 h-5 text-white/40 animate-spin" />
        </div>
      ) : filteredHypotheses.length > 0 ? (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredHypotheses.map((hypothesis) => (
            <HypothesisCard
              key={hypothesis.hypothesisId}
              hypothesis={hypothesis}
              onInvestigate={handleInvestigate}
              onVerify={handleVerify}
              isInvestigating={isInvestigating}
              isVerifying={isVerifying}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-xs text-white/40">
          <AlertTriangle className="w-6 h-6 mx-auto mb-2 opacity-50" />
          <p>No hypotheses found</p>
          <p className="text-[10px] text-white/30 mt-1">
            Run an insight scan to generate hypotheses
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main AgentFlow Panel
// ============================================================================

export function AgentFlowPanel({ className }: { className?: string }) {
  const { agentFlowStatus, isLoadingStatus, fetchAgentFlowStatus, error, clearError } =
    useAgentsStore();

  useEffect(() => {
    fetchAgentFlowStatus();
  }, [fetchAgentFlowStatus]);

  return (
    <div className={cn('space-y-4', className)}>
      {/* Error banner */}
      {error && (
        <div className="p-2 bg-red-500/10 border border-red-500/20 flex items-center justify-between">
          <span className="text-xs text-red-400">{error}</span>
          <button onClick={clearError} className="text-xs text-red-300 hover:text-red-200">
            Dismiss
          </button>
        </div>
      )}

      {/* Status indicator */}
      <div className="flex items-center gap-2 text-xs text-white/50">
        <Brain className="w-4 h-4" />
        <span>AgentFlow</span>
        {isLoadingStatus ? (
          <Loader2 className="w-3 h-3 animate-spin ml-auto" />
        ) : agentFlowStatus ? (
          <span className="ml-auto px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded text-[10px]">
            Active
          </span>
        ) : (
          <span className="ml-auto px-1.5 py-0.5 bg-white/10 text-white/40 rounded text-[10px]">
            Inactive
          </span>
        )}
      </div>

      {/* Memory buffer */}
      <MemoryBufferCard />

      {/* Hypotheses */}
      <HypothesesSection />
    </div>
  );
}

export default AgentFlowPanel;

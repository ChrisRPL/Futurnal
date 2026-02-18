/**
 * InsightsFeed - Main insights feed component
 *
 * AGI Phase 8: Frontend Integration
 *
 * Research Foundation:
 * - Proactive intelligence surfacing
 * - CuriosityEngine + EmergentInsights integration
 * - ICDA interactive verification
 *
 * Features:
 * - Tabbed view: All | Insights | Gaps | Verifications
 * - Unified feed with mixed content
 * - Statistics header
 * - Manual scan trigger
 * - Responsive layout
 */

import { useEffect, useState } from 'react';
import {
  Lightbulb,
  Brain,
  GitBranch,
  RefreshCw,
  Bell,
  Clock,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useInsightsStore } from '@/stores/insightsStore';
import { InsightCard } from './InsightCard';
import { KnowledgeGapCard } from './KnowledgeGapCard';
import { CausalVerification, CausalVerificationCompact } from './CausalVerification';
import type {
  EmergentInsight,
  KnowledgeGap,
  CausalVerificationQuestion,
} from '@/stores/insightsStore';

type TabType = 'all' | 'insights' | 'gaps' | 'verifications';

interface InsightsFeedProps {
  onExploreGap?: (prompt: string, gap: KnowledgeGap) => void;
  onInsightAction?: (action: string, insight: EmergentInsight) => void;
  className?: string;
}

/**
 * Statistics bar component
 */
function StatsBar({
  stats,
  isScanning,
  onScan,
}: {
  stats: {
    totalInsights: number;
    unreadInsights: number;
    totalGaps: number;
    pendingVerifications: number;
  } | null;
  isScanning: boolean;
  onScan: () => void;
}) {
  return (
    <div className="flex items-center justify-between p-3 bg-white/[0.02] border-b border-white/10">
      <div className="flex items-center gap-4 text-xs">
        {stats ? (
          <>
            <div className="flex items-center gap-1.5">
              <Lightbulb className="w-3.5 h-3.5 text-white/40" />
              <span className="text-white/60">{stats.totalInsights}</span>
              {stats.unreadInsights > 0 && (
                <span className="px-1 py-0.5 bg-blue-500/20 text-blue-400 text-[10px] rounded">
                  {stats.unreadInsights} new
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5 text-white/40" />
              <span className="text-white/60">{stats.totalGaps} gaps</span>
            </div>
            <div className="flex items-center gap-1.5">
              <GitBranch className="w-3.5 h-3.5 text-white/40" />
              <span className="text-white/60">
                {stats.pendingVerifications} pending
              </span>
            </div>
          </>
        ) : (
          <span className="text-white/40">Loading stats...</span>
        )}
      </div>

      <button
        onClick={onScan}
        disabled={isScanning}
        className={cn(
          'flex items-center gap-1.5 px-2 py-1',
          'text-xs text-white/50 hover:text-white/80',
          'border border-white/10 hover:border-white/20',
          'transition-colors',
          isScanning && 'opacity-50 cursor-not-allowed'
        )}
      >
        <RefreshCw className={cn('w-3 h-3', isScanning && 'animate-spin')} />
        <span>{isScanning ? 'Scanning...' : 'Scan Now'}</span>
      </button>
    </div>
  );
}

/**
 * Tab button component
 */
function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof Lightbulb;
  label: string;
  count?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-1.5 px-3 py-2',
        'text-xs transition-colors',
        active
          ? 'text-white border-b-2 border-white/60'
          : 'text-white/50 hover:text-white/70 border-b-2 border-transparent'
      )}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
      {count !== undefined && count > 0 && (
        <span
          className={cn(
            'px-1.5 py-0.5 text-[10px] rounded',
            active ? 'bg-white/20' : 'bg-white/10'
          )}
        >
          {count}
        </span>
      )}
    </button>
  );
}

/**
 * Empty state component with helpful guidance
 */
function EmptyState({ type }: { type: TabType }) {
  const messages: Record<TabType, { icon: typeof Lightbulb; message: string; guidance?: string[] }> = {
    all: {
      icon: Bell,
      message: 'No insights yet',
      guidance: [
        'Add more documents with timestamps',
        'Documents need different creation dates',
        'Minimum 5+ documents recommended for patterns',
        'Click "Scan Now" after adding more data',
      ],
    },
    insights: {
      icon: Lightbulb,
      message: 'No emergent insights detected',
      guidance: [
        'Insights emerge from recurring patterns',
        'Add documents with overlapping topics',
        'Include dated content for temporal patterns',
      ],
    },
    gaps: {
      icon: Brain,
      message: 'No knowledge gaps identified',
      guidance: [
        'Gaps are detected between related concepts',
        'Add more interconnected documents',
      ],
    },
    verifications: {
      icon: GitBranch,
      message: 'No causal hypotheses pending',
      guidance: [
        'Causal relationships are detected from temporal data',
        'Add documents with clear cause-effect relationships',
      ],
    },
  };

  const { icon: Icon, message, guidance } = messages[type];

  return (
    <div className="flex flex-col items-center justify-center py-12 text-center px-6">
      <Icon className="w-8 h-8 text-white/20 mb-3" />
      <p className="text-sm text-white/40 mb-4">{message}</p>
      {guidance && guidance.length > 0 && (
        <div className="text-left max-w-xs">
          <p className="text-xs text-white/30 mb-2">To generate insights:</p>
          <ul className="space-y-1.5">
            {guidance.map((item, i) => (
              <li key={i} className="text-xs text-white/25 flex items-start gap-2">
                <span className="text-white/20 mt-0.5">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * Main InsightsFeed component
 */
export function InsightsFeed({
  onExploreGap,
  onInsightAction,
  className,
}: InsightsFeedProps) {
  const [activeTab, setActiveTab] = useState<TabType>('all');
  const [expandedVerification, setExpandedVerification] = useState<string | null>(null);

  const {
    insights,
    gaps,
    pendingVerifications,
    stats,
    unreadCount,
    totalGaps,
    totalPending,
    isLoadingInsights,
    isLoadingGaps,
    isLoadingVerifications,
    isScanning,
    error,
    fetchInsights,
    fetchKnowledgeGaps,
    fetchPendingVerifications,
    fetchStats,
    triggerScan,
    markInsightRead,
    dismissInsight,
    markGapAddressed,
    submitVerification,
    clearError,
  } = useInsightsStore();

  // Load data on mount
  useEffect(() => {
    fetchStats();
    fetchInsights();
    fetchKnowledgeGaps();
    fetchPendingVerifications();
  }, [fetchStats, fetchInsights, fetchKnowledgeGaps, fetchPendingVerifications]);

  const isLoading = isLoadingInsights || isLoadingGaps || isLoadingVerifications;

  // Handle verification submission
  const handleVerificationSubmit = async (
    questionId: string,
    response: Parameters<typeof submitVerification>[1],
    explanation?: string
  ) => {
    const result = await submitVerification(questionId, response, explanation);
    if (result?.success) {
      setExpandedVerification(null);
    }
    return result;
  };

  // Filter content based on active tab
  const renderContent = () => {
    if (isLoading && insights.length === 0 && gaps.length === 0) {
      return (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 text-white/40 animate-spin" />
        </div>
      );
    }

    switch (activeTab) {
      case 'insights':
        return insights.length > 0 ? (
          <div className="space-y-2 p-3">
            {insights.map((insight) => (
              <InsightCard
                key={insight.insightId}
                insight={insight}
                onMarkRead={markInsightRead}
                onDismiss={dismissInsight}
                onActionClick={onInsightAction}
              />
            ))}
          </div>
        ) : (
          <EmptyState type="insights" />
        );

      case 'gaps':
        return gaps.length > 0 ? (
          <div className="space-y-2 p-3">
            {gaps.map((gap) => (
              <KnowledgeGapCard
                key={gap.gapId}
                gap={gap}
                onExplore={onExploreGap}
                onMarkAddressed={markGapAddressed}
              />
            ))}
          </div>
        ) : (
          <EmptyState type="gaps" />
        );

      case 'verifications':
        return pendingVerifications.length > 0 ? (
          <div className="space-y-2 p-3">
            {pendingVerifications.map((question) =>
              expandedVerification === question.questionId ? (
                <CausalVerification
                  key={question.questionId}
                  question={question}
                  onSubmit={handleVerificationSubmit}
                  onSkip={() => setExpandedVerification(null)}
                />
              ) : (
                <CausalVerificationCompact
                  key={question.questionId}
                  question={question}
                  onClick={() => setExpandedVerification(question.questionId)}
                />
              )
            )}
          </div>
        ) : (
          <EmptyState type="verifications" />
        );

      case 'all':
      default:
        // Mixed feed - prioritize by recency and importance
        const allItems: Array<{
          type: 'insight' | 'gap' | 'verification';
          data: EmergentInsight | KnowledgeGap | CausalVerificationQuestion;
          sortKey: number;
        }> = [
          ...insights.map((i) => ({
            type: 'insight' as const,
            data: i,
            sortKey: i.priority === 'high' ? 3 : i.priority === 'medium' ? 2 : 1,
          })),
          ...gaps
            .filter((g) => !g.isAddressed)
            .map((g) => ({
              type: 'gap' as const,
              data: g,
              sortKey: g.informationGain > 0.7 ? 3 : g.informationGain > 0.4 ? 2 : 1,
            })),
          ...pendingVerifications.map((v) => ({
            type: 'verification' as const,
            data: v,
            sortKey: 2.5, // Verifications are important
          })),
        ];

        // Sort by priority
        allItems.sort((a, b) => b.sortKey - a.sortKey);

        if (allItems.length === 0) {
          return <EmptyState type="all" />;
        }

        return (
          <div className="space-y-2 p-3">
            {allItems.slice(0, 20).map((item) => {
              if (item.type === 'insight') {
                return (
                  <InsightCard
                    key={`insight-${(item.data as EmergentInsight).insightId}`}
                    insight={item.data as EmergentInsight}
                    onMarkRead={markInsightRead}
                    onDismiss={dismissInsight}
                    onActionClick={onInsightAction}
                  />
                );
              }
              if (item.type === 'gap') {
                return (
                  <KnowledgeGapCard
                    key={`gap-${(item.data as KnowledgeGap).gapId}`}
                    gap={item.data as KnowledgeGap}
                    onExplore={onExploreGap}
                    onMarkAddressed={markGapAddressed}
                  />
                );
              }
              if (item.type === 'verification') {
                const question = item.data as CausalVerificationQuestion;
                return expandedVerification === question.questionId ? (
                  <CausalVerification
                    key={`ver-${question.questionId}`}
                    question={question}
                    onSubmit={handleVerificationSubmit}
                    onSkip={() => setExpandedVerification(null)}
                  />
                ) : (
                  <CausalVerificationCompact
                    key={`ver-${question.questionId}`}
                    question={question}
                    onClick={() => setExpandedVerification(question.questionId)}
                  />
                );
              }
              return null;
            })}
          </div>
        );
    }
  };

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Error banner */}
      {error && (
        <div className="px-3 py-2 bg-red-500/10 border-b border-red-500/20 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
          <button
            onClick={clearError}
            className="text-xs text-red-300 hover:text-red-200"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Stats bar */}
      <StatsBar
        stats={
          stats
            ? {
                totalInsights: stats.totalInsights,
                unreadInsights: stats.unreadInsights,
                totalGaps: stats.totalGaps,
                pendingVerifications: stats.pendingVerifications,
              }
            : null
        }
        isScanning={isScanning}
        onScan={triggerScan}
      />

      {/* Tabs */}
      <div className="flex items-center border-b border-white/10">
        <TabButton
          active={activeTab === 'all'}
          onClick={() => setActiveTab('all')}
          icon={Bell}
          label="All"
        />
        <TabButton
          active={activeTab === 'insights'}
          onClick={() => setActiveTab('insights')}
          icon={Lightbulb}
          label="Insights"
          count={unreadCount}
        />
        <TabButton
          active={activeTab === 'gaps'}
          onClick={() => setActiveTab('gaps')}
          icon={Brain}
          label="Gaps"
          count={totalGaps}
        />
        <TabButton
          active={activeTab === 'verifications'}
          onClick={() => setActiveTab('verifications')}
          icon={GitBranch}
          label="Verify"
          count={totalPending}
        />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">{renderContent()}</div>

      {/* Last scan timestamp */}
      {stats?.lastScanAt && (
        <div className="px-3 py-2 border-t border-white/5 flex items-center gap-1.5 text-[10px] text-white/30">
          <Clock className="w-3 h-3" />
          <span>Last scan: {new Date(stats.lastScanAt).toLocaleString()}</span>
        </div>
      )}
    </div>
  );
}

export default InsightsFeed;

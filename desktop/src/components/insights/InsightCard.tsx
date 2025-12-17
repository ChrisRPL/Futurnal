/**
 * InsightCard - Display individual emergent insights
 *
 * AGI Phase 8: Frontend Integration
 *
 * Research Foundation:
 * - EmergentInsights: Correlation to NL insights
 * - Proactive intelligence surfacing
 *
 * Features:
 * - Insight type badge with icon
 * - Confidence/relevance indicators
 * - Suggested actions
 * - Mark read/dismiss actions
 */

import { useState } from 'react';
import {
  Lightbulb,
  TrendingUp,
  AlertTriangle,
  GitBranch,
  Activity,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Check,
  X,
  Clock,
  ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { EmergentInsight, InsightType, InsightPriority } from '@/stores/insightsStore';

interface InsightCardProps {
  insight: EmergentInsight;
  onMarkRead?: (insightId: string) => void;
  onDismiss?: (insightId: string) => void;
  onActionClick?: (action: string, insight: EmergentInsight) => void;
  className?: string;
}

/** Icon mapping for insight types */
const TYPE_ICONS: Record<InsightType, typeof Lightbulb> = {
  correlation: Activity,
  causal_hypothesis: GitBranch,
  pattern: TrendingUp,
  anomaly: AlertTriangle,
  trend: TrendingUp,
  knowledge_gap: HelpCircle,
};

/** Color mapping for insight types */
const TYPE_COLORS: Record<InsightType, string> = {
  correlation: 'bg-blue-500/20 text-blue-400',
  causal_hypothesis: 'bg-purple-500/20 text-purple-400',
  pattern: 'bg-green-500/20 text-green-400',
  anomaly: 'bg-red-500/20 text-red-400',
  trend: 'bg-cyan-500/20 text-cyan-400',
  knowledge_gap: 'bg-amber-500/20 text-amber-400',
};

/** Priority colors */
const PRIORITY_COLORS: Record<InsightPriority, string> = {
  high: 'border-l-red-500',
  medium: 'border-l-amber-500',
  low: 'border-l-white/20',
};

/** Format insight type for display */
function formatInsightType(type: InsightType): string {
  return type
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Confidence bar component
 */
function ConfidenceBar({
  value,
  label,
  className,
}: {
  value: number;
  label: string;
  className?: string;
}) {
  const percentage = Math.round(value * 100);

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      <div className="flex items-center justify-between text-[10px] text-white/40">
        <span>{label}</span>
        <span>{percentage}%</span>
      </div>
      <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-white/40 rounded-full transition-all"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Main InsightCard component
 */
export function InsightCard({
  insight,
  onMarkRead,
  onDismiss,
  onActionClick,
  className,
}: InsightCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const TypeIcon = TYPE_ICONS[insight.insightType] || Lightbulb;
  const typeColor = TYPE_COLORS[insight.insightType] || 'bg-white/20 text-white/60';
  const priorityBorder = PRIORITY_COLORS[insight.priority];

  const handleMarkRead = () => {
    if (!insight.isRead && onMarkRead) {
      onMarkRead(insight.insightId);
    }
  };

  return (
    <div
      className={cn(
        'bg-white/[0.02] border border-white/10',
        'border-l-2',
        priorityBorder,
        !insight.isRead && 'bg-white/[0.04]',
        className
      )}
      onClick={handleMarkRead}
    >
      {/* Header */}
      <div className="p-3">
        <div className="flex items-start justify-between gap-3">
          {/* Type badge and title */}
          <div className="flex items-start gap-2 flex-1 min-w-0">
            <div className={cn('p-1.5 rounded', typeColor)}>
              <TypeIcon className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-medium text-white/90 truncate">
                  {insight.title}
                </h4>
                {!insight.isRead && (
                  <span className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />
                )}
              </div>
              <span className={cn('text-[10px] px-1.5 py-0.5 rounded', typeColor)}>
                {formatInsightType(insight.insightType)}
              </span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsExpanded(!isExpanded);
              }}
              className="p-1 text-white/30 hover:text-white/60 transition-colors"
            >
              {isExpanded ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>
            {onDismiss && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDismiss(insight.insightId);
                }}
                className="p-1 text-white/30 hover:text-red-400 transition-colors"
                title="Dismiss"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Description */}
        <p className="mt-2 text-xs text-white/60 line-clamp-2">{insight.description}</p>

        {/* Confidence bars */}
        <div className="mt-3 grid grid-cols-2 gap-3">
          <ConfidenceBar value={insight.confidence} label="Confidence" />
          <ConfidenceBar value={insight.relevance} label="Relevance" />
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-white/5 pt-3 space-y-3">
          {/* Source events */}
          {insight.sourceEvents.length > 0 && (
            <div>
              <p className="text-[10px] text-white/40 mb-1">Related Events</p>
              <div className="flex flex-wrap gap-1">
                {insight.sourceEvents.slice(0, 5).map((event, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-1.5 py-0.5 bg-white/5 text-white/50 rounded"
                  >
                    {event}
                  </span>
                ))}
                {insight.sourceEvents.length > 5 && (
                  <span className="text-[10px] text-white/30">
                    +{insight.sourceEvents.length - 5} more
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Suggested actions */}
          {insight.suggestedActions.length > 0 && (
            <div>
              <p className="text-[10px] text-white/40 mb-1">Suggested Actions</p>
              <div className="space-y-1">
                {insight.suggestedActions.map((action, i) => (
                  <button
                    key={i}
                    onClick={(e) => {
                      e.stopPropagation();
                      onActionClick?.(action, insight);
                    }}
                    className={cn(
                      'flex items-center gap-2 w-full text-left',
                      'text-xs text-white/60 hover:text-white/80',
                      'p-2 bg-white/[0.02] hover:bg-white/[0.05]',
                      'transition-colors'
                    )}
                  >
                    <Check className="w-3 h-3 text-white/30" />
                    <span className="flex-1">{action}</span>
                    <ExternalLink className="w-3 h-3 text-white/20" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Timestamps */}
          <div className="flex items-center gap-3 text-[10px] text-white/30">
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>Created: {new Date(insight.createdAt).toLocaleString()}</span>
            </div>
            {insight.expiresAt && (
              <span>Expires: {new Date(insight.expiresAt).toLocaleDateString()}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default InsightCard;

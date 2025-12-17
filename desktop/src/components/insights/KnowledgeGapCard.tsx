/**
 * KnowledgeGapCard - Display knowledge gaps from CuriosityEngine
 *
 * AGI Phase 8: Frontend Integration
 *
 * Research Foundation:
 * - CuriosityEngine: Information-gain gap detection
 * - DyMemR (2024): Memory forgetting curve
 * - Curiosity-driven Autotelic AI (Oudeyer 2024)
 *
 * Features:
 * - Gap type visualization
 * - Information gain indicator
 * - Exploration prompts with one-click explore
 * - Related topics
 */

import { useState } from 'react';
import {
  Brain,
  Network,
  Clock,
  Link2,
  Layers,
  Target,
  ChevronDown,
  ChevronUp,
  Check,
  Sparkles,
  MessageSquare,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { KnowledgeGap, GapType } from '@/stores/insightsStore';

interface KnowledgeGapCardProps {
  gap: KnowledgeGap;
  onExplore?: (prompt: string, gap: KnowledgeGap) => void;
  onMarkAddressed?: (gapId: string) => void;
  className?: string;
}

/** Icon mapping for gap types */
const GAP_TYPE_ICONS: Record<GapType, typeof Brain> = {
  isolated_cluster: Network,
  forgotten_memory: Clock,
  bridge_opportunity: Link2,
  missing_synthesis: Layers,
  aspiration_disconnect: Target,
};

/** Color mapping for gap types */
const GAP_TYPE_COLORS: Record<GapType, string> = {
  isolated_cluster: 'bg-purple-500/20 text-purple-400',
  forgotten_memory: 'bg-amber-500/20 text-amber-400',
  bridge_opportunity: 'bg-green-500/20 text-green-400',
  missing_synthesis: 'bg-blue-500/20 text-blue-400',
  aspiration_disconnect: 'bg-red-500/20 text-red-400',
};

/** Description for gap types */
const GAP_TYPE_DESCRIPTIONS: Record<GapType, string> = {
  isolated_cluster: 'Disconnected knowledge cluster',
  forgotten_memory: 'Potentially forgotten information',
  bridge_opportunity: 'Connection opportunity between topics',
  missing_synthesis: 'Unprocessed recent information',
  aspiration_disconnect: 'Gap between goals and knowledge',
};

/** Format gap type for display */
function formatGapType(type: GapType): string {
  return type
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Information gain indicator
 */
function InformationGainBar({
  gain,
  className,
}: {
  gain: number;
  className?: string;
}) {
  const percentage = Math.min(100, Math.round(gain * 100));
  const color =
    gain >= 0.7
      ? 'bg-green-500/60'
      : gain >= 0.4
        ? 'bg-amber-500/60'
        : 'bg-white/40';

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <Sparkles className="w-3 h-3 text-white/40" />
      <div className="flex-1">
        <div className="flex items-center justify-between text-[10px] text-white/40 mb-0.5">
          <span>Information Gain</span>
          <span>{percentage}%</span>
        </div>
        <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
          <div
            className={cn('h-full rounded-full transition-all', color)}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
}

/**
 * Main KnowledgeGapCard component
 */
export function KnowledgeGapCard({
  gap,
  onExplore,
  onMarkAddressed,
  className,
}: KnowledgeGapCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const GapIcon = GAP_TYPE_ICONS[gap.gapType] || Brain;
  const gapColor = GAP_TYPE_COLORS[gap.gapType] || 'bg-white/20 text-white/60';
  const gapDescription = GAP_TYPE_DESCRIPTIONS[gap.gapType];

  return (
    <div
      className={cn(
        'bg-white/[0.02] border border-white/10',
        gap.isAddressed && 'opacity-60',
        className
      )}
    >
      {/* Header */}
      <div className="p-3">
        <div className="flex items-start justify-between gap-3">
          {/* Type badge and title */}
          <div className="flex items-start gap-2 flex-1 min-w-0">
            <div className={cn('p-1.5 rounded', gapColor)}>
              <GapIcon className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-medium text-white/90 truncate">
                {gap.title}
              </h4>
              <div className="flex items-center gap-2 mt-0.5">
                <span className={cn('text-[10px] px-1.5 py-0.5 rounded', gapColor)}>
                  {formatGapType(gap.gapType)}
                </span>
                {gap.isAddressed && (
                  <span className="flex items-center gap-0.5 text-[10px] text-green-400">
                    <Check className="w-3 h-3" />
                    Addressed
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 text-white/30 hover:text-white/60 transition-colors"
            >
              {isExpanded ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        {/* Description */}
        <p className="mt-2 text-xs text-white/60 line-clamp-2">{gap.description}</p>

        {/* Information gain */}
        <div className="mt-3">
          <InformationGainBar gain={gap.informationGain} />
        </div>

        {/* Related topics preview */}
        {gap.relatedTopics.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1">
            {gap.relatedTopics.slice(0, 4).map((topic, i) => (
              <span
                key={i}
                className="text-[10px] px-1.5 py-0.5 bg-white/5 text-white/50 rounded"
              >
                {topic}
              </span>
            ))}
            {gap.relatedTopics.length > 4 && (
              <span className="text-[10px] text-white/30">
                +{gap.relatedTopics.length - 4} more
              </span>
            )}
          </div>
        )}
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-white/5 pt-3 space-y-3">
          {/* Gap type description */}
          <div className="flex items-center gap-2 text-[10px] text-white/40">
            <GapIcon className="w-3 h-3" />
            <span>{gapDescription}</span>
          </div>

          {/* Exploration prompts */}
          {gap.explorationPrompts.length > 0 && (
            <div>
              <p className="text-[10px] text-white/40 mb-2 flex items-center gap-1">
                <MessageSquare className="w-3 h-3" />
                Exploration Prompts
              </p>
              <div className="space-y-1">
                {gap.explorationPrompts.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => onExplore?.(prompt, gap)}
                    className={cn(
                      'flex items-start gap-2 w-full text-left',
                      'text-xs text-white/70 hover:text-white',
                      'p-2 bg-white/[0.02] hover:bg-white/[0.05]',
                      'border border-white/5 hover:border-white/10',
                      'transition-colors group'
                    )}
                  >
                    <Sparkles className="w-3 h-3 text-white/30 mt-0.5 group-hover:text-amber-400" />
                    <span className="flex-1">{prompt}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* All related topics */}
          {gap.relatedTopics.length > 4 && (
            <div>
              <p className="text-[10px] text-white/40 mb-1">All Related Topics</p>
              <div className="flex flex-wrap gap-1">
                {gap.relatedTopics.map((topic, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-1.5 py-0.5 bg-white/5 text-white/50 rounded"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Mark addressed */}
          {!gap.isAddressed && onMarkAddressed && (
            <button
              onClick={() => onMarkAddressed(gap.gapId)}
              className={cn(
                'flex items-center gap-2 w-full justify-center',
                'text-xs text-white/50 hover:text-green-400',
                'p-2 border border-white/10 hover:border-green-500/30',
                'transition-colors'
              )}
            >
              <Check className="w-3 h-3" />
              <span>Mark as Addressed</span>
            </button>
          )}

          {/* Timestamp */}
          <div className="text-[10px] text-white/30">
            Detected: {new Date(gap.createdAt).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}

export default KnowledgeGapCard;

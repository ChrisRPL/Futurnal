/**
 * CausalChain - Full causal chain visualization component
 *
 * Step 08: Frontend Intelligence Integration - Phase 2
 *
 * Research Foundation:
 * - Youtu-GraphRAG: Multi-hop causal patterns
 * - CausalRAG: Causal-aware visualization
 *
 * Features:
 * - Horizontal flow: Causes <- Anchor -> Effects
 * - Confidence bars on each connection
 * - Clickable nodes navigate to graph
 * - Expand/collapse detail panels
 * - Temporal ordering validation markers
 */

import { useState } from 'react';
import {
  GitBranch,
  ChevronRight,
  ChevronLeft,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Clock,
  Check,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useCausalStore, type CausalCause, type CausalEffect, type CausalPath } from '@/stores/causalStore';

interface CausalChainProps {
  /** Event ID to explore from */
  eventId: string;
  /** Event name/label */
  eventName?: string;
  /** Callback when a node is clicked */
  onNodeClick?: (nodeId: string) => void;
  /** Additional class names */
  className?: string;
}

/**
 * Confidence bar component
 */
function ConfidenceBar({
  confidence,
  className,
}: {
  confidence: number;
  className?: string;
}) {
  const percentage = Math.round(confidence * 100);
  const color =
    confidence >= 0.8
      ? 'bg-white/70'
      : confidence >= 0.6
        ? 'bg-white/50'
        : 'bg-white/30';

  return (
    <div className={cn('flex items-center gap-1.5', className)}>
      <div className="w-12 h-1 bg-white/10 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-[10px] text-white/40 w-8">{percentage}%</span>
    </div>
  );
}

/**
 * Temporal validity marker
 */
function TemporalMarker({ valid }: { valid: boolean }) {
  return valid ? (
    <Check className="w-3 h-3 text-white/60" title="Temporal ordering valid" />
  ) : (
    <AlertCircle className="w-3 h-3 text-white/40" title="Temporal ordering uncertain" />
  );
}

/**
 * Cause node component
 */
function CauseNode({
  cause,
  onNodeClick,
  expanded,
  onToggle,
}: {
  cause: CausalCause;
  onNodeClick?: (nodeId: string) => void;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="group">
      <div className="flex items-center gap-2">
        {/* Confidence connection line */}
        <div className="flex flex-col items-center gap-0.5">
          <div className="h-px w-8 bg-white/20" />
          <ConfidenceBar confidence={cause.aggregateConfidence} />
        </div>

        {/* Arrow */}
        <ChevronRight className="w-3 h-3 text-white/30" />

        {/* Node */}
        <div
          className={cn(
            'flex items-center gap-2 px-3 py-2',
            'bg-white/5 border border-white/10 hover:bg-white/10',
            'cursor-pointer transition-colors'
          )}
          onClick={() => onNodeClick?.(cause.causeId)}
        >
          <span className="text-sm text-white/80 max-w-[150px] truncate">
            {cause.causeName}
          </span>
          <TemporalMarker valid={cause.temporalOrderingValid} />
          <ExternalLink className="w-3 h-3 text-white/30 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>

        {/* Distance badge */}
        <span className="text-[10px] text-white/30 bg-white/5 px-1 rounded">
          -{cause.distance}
        </span>

        {/* Expand button */}
        <button
          onClick={onToggle}
          className="p-0.5 text-white/30 hover:text-white/60"
        >
          {expanded ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </button>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="ml-16 mt-2 p-2 bg-white/[0.02] border-l border-white/10 text-xs text-white/50">
          {cause.causeTimestamp && (
            <div className="flex items-center gap-1 mb-1">
              <Clock className="w-3 h-3" />
              <span>{new Date(cause.causeTimestamp).toLocaleString()}</span>
            </div>
          )}
          <div>Confidence: {(cause.aggregateConfidence * 100).toFixed(0)}%</div>
          <div>Hops: {cause.distance}</div>
        </div>
      )}
    </div>
  );
}

/**
 * Effect node component
 */
function EffectNode({
  effect,
  onNodeClick,
  expanded,
  onToggle,
}: {
  effect: CausalEffect;
  onNodeClick?: (nodeId: string) => void;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="group">
      <div className="flex items-center gap-2">
        {/* Distance badge */}
        <span className="text-[10px] text-white/30 bg-white/5 px-1 rounded">
          +{effect.distance}
        </span>

        {/* Expand button */}
        <button
          onClick={onToggle}
          className="p-0.5 text-white/30 hover:text-white/60"
        >
          {expanded ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </button>

        {/* Node */}
        <div
          className={cn(
            'flex items-center gap-2 px-3 py-2',
            'bg-white/5 border border-white/10 hover:bg-white/10',
            'cursor-pointer transition-colors'
          )}
          onClick={() => onNodeClick?.(effect.effectId)}
        >
          <span className="text-sm text-white/80 max-w-[150px] truncate">
            {effect.effectName}
          </span>
          <TemporalMarker valid={effect.temporalOrderingValid} />
          <ExternalLink className="w-3 h-3 text-white/30 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>

        {/* Arrow */}
        <ChevronRight className="w-3 h-3 text-white/30" />

        {/* Confidence connection line */}
        <div className="flex flex-col items-center gap-0.5">
          <div className="h-px w-8 bg-white/20" />
          <ConfidenceBar confidence={effect.aggregateConfidence} />
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="ml-12 mt-2 p-2 bg-white/[0.02] border-l border-white/10 text-xs text-white/50">
          {effect.effectTimestamp && (
            <div className="flex items-center gap-1 mb-1">
              <Clock className="w-3 h-3" />
              <span>{new Date(effect.effectTimestamp).toLocaleString()}</span>
            </div>
          )}
          <div>Confidence: {(effect.aggregateConfidence * 100).toFixed(0)}%</div>
          <div>Hops: {effect.distance}</div>
        </div>
      )}
    </div>
  );
}

/**
 * Main CausalChain component
 */
export function CausalChain({
  eventId,
  eventName,
  onNodeClick,
  className,
}: CausalChainProps) {
  const [expandedCauses, setExpandedCauses] = useState<Set<string>>(new Set());
  const [expandedEffects, setExpandedEffects] = useState<Set<string>>(new Set());

  const {
    anchorEventId,
    causes,
    effects,
    isLoading,
    error,
    lastQueryTimeMs,
    findCauses,
    findEffects,
    clearError,
  } = useCausalStore();

  // Check if this event is already explored
  const isExplored = anchorEventId === eventId;

  // Handle explore
  const handleExplore = async () => {
    await Promise.all([
      findCauses(eventId),
      findEffects(eventId),
    ]);
  };

  // Toggle expanded state
  const toggleCauseExpanded = (causeId: string) => {
    setExpandedCauses((prev) => {
      const next = new Set(prev);
      if (next.has(causeId)) {
        next.delete(causeId);
      } else {
        next.add(causeId);
      }
      return next;
    });
  };

  const toggleEffectExpanded = (effectId: string) => {
    setExpandedEffects((prev) => {
      const next = new Set(prev);
      if (next.has(effectId)) {
        next.delete(effectId);
      } else {
        next.add(effectId);
      }
      return next;
    });
  };

  return (
    <div className={cn('p-4 bg-black border border-white/10', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-white/60" />
          <span className="text-sm font-medium text-white/80">
            Causal Chain
          </span>
          {lastQueryTimeMs && (
            <span className="text-xs text-white/30">
              {lastQueryTimeMs.toFixed(0)}ms
            </span>
          )}
        </div>

        {/* Explore button */}
        {!isExplored && (
          <button
            onClick={handleExplore}
            disabled={isLoading}
            className={cn(
              'px-3 py-1 text-xs',
              'bg-white/10 hover:bg-white/20',
              'text-white/70 hover:text-white',
              'transition-colors',
              isLoading && 'opacity-50 cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              'Explore'
            )}
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 px-3 py-2 bg-red-500/10 border border-red-500/20 text-xs text-red-400">
          {error}
          <button
            onClick={clearError}
            className="ml-2 text-red-300 hover:text-red-200"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 text-white/40 animate-spin" />
        </div>
      )}

      {/* Causal flow visualization */}
      {isExplored && !isLoading && (
        <div className="flex items-start gap-4">
          {/* Causes (left side) */}
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-1 text-xs text-white/40 mb-2">
              <ChevronLeft className="w-3 h-3" />
              <span>What led to this ({causes.length})</span>
            </div>
            {causes.length === 0 ? (
              <p className="text-xs text-white/30">No causes found</p>
            ) : (
              causes.map((cause) => (
                <CauseNode
                  key={cause.causeId}
                  cause={cause}
                  onNodeClick={onNodeClick}
                  expanded={expandedCauses.has(cause.causeId)}
                  onToggle={() => toggleCauseExpanded(cause.causeId)}
                />
              ))
            )}
          </div>

          {/* Anchor node (center) */}
          <div className="flex flex-col items-center">
            <div
              className={cn(
                'px-4 py-3',
                'bg-white/10 border-2 border-white/20',
                'text-white font-medium',
                'cursor-pointer hover:bg-white/15',
                'transition-colors'
              )}
              onClick={() => onNodeClick?.(eventId)}
            >
              {eventName || eventId}
            </div>
            <div className="text-[10px] text-white/40 mt-1">Anchor</div>
          </div>

          {/* Effects (right side) */}
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-1 text-xs text-white/40 mb-2">
              <span>What resulted ({effects.length})</span>
              <ChevronRight className="w-3 h-3" />
            </div>
            {effects.length === 0 ? (
              <p className="text-xs text-white/30">No effects found</p>
            ) : (
              effects.map((effect) => (
                <EffectNode
                  key={effect.effectId}
                  effect={effect}
                  onNodeClick={onNodeClick}
                  expanded={expandedEffects.has(effect.effectId)}
                  onToggle={() => toggleEffectExpanded(effect.effectId)}
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Initial state */}
      {!isExplored && !isLoading && (
        <div className="text-center py-6 text-white/40 text-sm">
          <p>Click &quot;Explore&quot; to discover causes and effects</p>
        </div>
      )}
    </div>
  );
}

export default CausalChain;

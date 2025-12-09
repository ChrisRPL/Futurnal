/**
 * IntentBadge - Display query intent classification
 *
 * Shows the detected intent type (temporal, causal, exploratory, lookup)
 * with a monochrome styling using opacity-based differentiation.
 */

import { Clock, GitBranch, Compass, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { IntentType } from '@/stores/searchStore';

interface IntentBadgeProps {
  /** The detected intent type */
  intent: IntentType;
  /** Optional confidence percentage (0-1) */
  confidence?: number;
  /** Additional class names */
  className?: string;
}

/**
 * Intent configuration with monochrome styling
 * Uses opacity-based differentiation instead of colors
 */
const INTENT_CONFIG: Record<
  IntentType,
  {
    label: string;
    icon: typeof Clock;
    className: string;
  }
> = {
  temporal: {
    label: 'Temporal',
    icon: Clock,
    className: 'bg-white/10 text-white border border-white/20',
  },
  causal: {
    label: 'Causal',
    icon: GitBranch,
    className: 'bg-white/15 text-white border border-white/30',
  },
  exploratory: {
    label: 'Exploratory',
    icon: Compass,
    className: 'bg-white/5 text-white/70 border border-white/10',
  },
  lookup: {
    label: 'Lookup',
    icon: Search,
    className: 'bg-white/5 text-white/70 border border-white/10',
  },
};

export function IntentBadge({ intent, confidence, className }: IntentBadgeProps) {
  const config = INTENT_CONFIG[intent];
  const Icon = config.icon;

  return (
    <div
      data-slot="intent-badge"
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs',
        config.className,
        className
      )}
    >
      <Icon className="h-3 w-3" />
      <span>{config.label}</span>
      {confidence !== undefined && confidence > 0 && (
        <span className="opacity-60">({Math.round(confidence * 100)}%)</span>
      )}
    </div>
  );
}

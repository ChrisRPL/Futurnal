/**
 * ScoreIndicator - Display relevance score with monochrome color coding
 *
 * Shows percentage score with opacity-based color coding:
 * - 80%+ → full white (high relevance)
 * - 60%+ → white/70 (medium relevance)
 * - Below 60% → white/50 (low relevance)
 */

import { cn } from '@/lib/utils';

interface ScoreIndicatorProps {
  /** Relevance score from 0 to 1 */
  score: number;
  /** Additional class names */
  className?: string;
}

export function ScoreIndicator({ score, className }: ScoreIndicatorProps) {
  const percentage = Math.round(score * 100);

  // Determine opacity class based on score threshold
  const colorClass =
    percentage >= 80
      ? 'text-white'
      : percentage >= 60
        ? 'text-white/70'
        : 'text-white/50';

  return (
    <div
      data-slot="score-indicator"
      className={cn('text-xs font-medium tabular-nums', colorClass, className)}
      title={`Relevance: ${percentage}%`}
    >
      {percentage}%
    </div>
  );
}

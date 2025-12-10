/**
 * ConfidenceIndicator - 3-bar visualization for confidence level
 *
 * Displays confidence as 3 vertical bars:
 * - 0-33% → 1 bar filled
 * - 34-66% → 2 bars filled
 * - 67-100% → 3 bars filled
 */

import { cn } from '@/lib/utils';

interface ConfidenceIndicatorProps {
  /** Confidence level from 0 to 1 */
  confidence: number;
  /** Additional class names */
  className?: string;
}

export function ConfidenceIndicator({
  confidence,
  className,
}: ConfidenceIndicatorProps) {
  const percentage = Math.round(confidence * 100);
  const totalBars = 3;

  // Calculate filled bars based on confidence level
  const filledBars = Math.max(1, Math.ceil(confidence * totalBars));

  return (
    <div
      data-slot="confidence-indicator"
      className={cn('flex items-end gap-0.5', className)}
      title={`Confidence: ${percentage}%`}
    >
      {Array.from({ length: totalBars }).map((_, index) => {
        const isFilled = index < filledBars;
        // Each bar is slightly taller than the previous
        const height = 8 + index * 2;

        return (
          <div
            key={index}
            className={cn(
              'w-1 rounded-sm transition-colors',
              isFilled ? 'bg-white' : 'bg-white/20'
            )}
            style={{ height: `${height}px` }}
          />
        );
      })}
    </div>
  );
}

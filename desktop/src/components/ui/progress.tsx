/**
 * Progress Component
 *
 * Simple progress bar following Futurnal monochrome design system.
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Progress value from 0 to 100 */
  value: number;
  /** Optional label to show above the progress bar */
  label?: string;
  /** Optional secondary text (e.g., "50 / 100") */
  secondaryLabel?: string;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, label, secondaryLabel, ...props }, ref) => {
    const clampedValue = Math.min(100, Math.max(0, value));

    return (
      <div ref={ref} data-slot="progress" className={cn('w-full', className)} {...props}>
        {(label || secondaryLabel) && (
          <div className="flex justify-between text-xs text-white/60 mb-1">
            {label && <span>{label}</span>}
            {secondaryLabel && <span>{secondaryLabel}</span>}
          </div>
        )}
        <div className="h-1 w-full bg-white/10 overflow-hidden">
          <div
            className="h-full bg-white transition-all duration-300 ease-out"
            style={{ width: `${clampedValue}%` }}
            role="progressbar"
            aria-valuenow={clampedValue}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
      </div>
    );
  }
);
Progress.displayName = 'Progress';

export { Progress };

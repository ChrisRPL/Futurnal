/**
 * Label Component
 *
 * Form label following Futurnal monochrome design system.
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

export interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {
  /** Optional description text below the label */
  description?: string;
}

const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, description, children, ...props }, ref) => (
    <div className="space-y-1">
      <label
        ref={ref}
        data-slot="label"
        className={cn('text-sm font-medium text-white/80', className)}
        {...props}
      >
        {children}
      </label>
      {description && (
        <p className="text-xs text-white/50">{description}</p>
      )}
    </div>
  )
);
Label.displayName = 'Label';

export { Label };

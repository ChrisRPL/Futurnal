import * as React from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: string;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, error, ...props }, ref) => {
    return (
      <div className="w-full">
        <input
          type={type}
          data-slot="input"
          className={cn(
            'flex h-10 w-full rounded-md border bg-background-elevated px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary transition-all duration-150',
            'focus:outline-none focus:ring-2 focus:ring-primary/20',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-text-primary',
            error
              ? 'border-error focus:border-error focus:ring-error/20'
              : 'border-border focus:border-primary',
            className
          )}
          ref={ref}
          {...props}
        />
        {error && (
          <p data-slot="input-error" className="mt-1.5 text-xs text-error">
            {error}
          </p>
        )}
      </div>
    );
  }
);
Input.displayName = 'Input';

export { Input };

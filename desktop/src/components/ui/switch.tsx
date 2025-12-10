/**
 * Switch Component
 *
 * Toggle switch using Radix UI primitives.
 * Follows Futurnal monochrome design system.
 */

import * as React from 'react';
import * as SwitchPrimitive from '@radix-ui/react-switch';
import { cn } from '@/lib/utils';

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitive.Root
    data-slot="switch"
    className={cn(
      'peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center border border-white/30 transition-colors',
      'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-black',
      'disabled:cursor-not-allowed disabled:opacity-50',
      'data-[state=checked]:bg-white data-[state=unchecked]:bg-white/10',
      className
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitive.Thumb
      data-slot="switch-thumb"
      className={cn(
        'pointer-events-none block h-4 w-4 transition-transform',
        'data-[state=checked]:translate-x-4 data-[state=unchecked]:translate-x-0',
        'data-[state=checked]:bg-black data-[state=unchecked]:bg-white'
      )}
    />
  </SwitchPrimitive.Root>
));
Switch.displayName = SwitchPrimitive.Root.displayName;

export { Switch };

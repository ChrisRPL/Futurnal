Summary: Implement TailwindCSS design system with dark-mode tokens and base UI components from shadcn/ui.

# 02 · Design System & Components

## Purpose

Establish the visual foundation for Futurnal's desktop shell by implementing a comprehensive design system using TailwindCSS v4 with custom design tokens from FRONTEND_DESIGN.md, integrated with shadcn/ui components and micro-interaction animations.

**Criticality**: HIGH - Visual foundation for all UI components

## Scope

- TailwindCSS v4 configuration with CSS custom properties
- Design tokens from FRONTEND_DESIGN.md (colors, typography, shadows)
- Typography scale with Inter (UI) + JetBrains Mono (code)
- shadcn/ui component integration with custom dark theme
- Base components: Button, Input, Card, Dialog, Tooltip, Badge, ScrollArea
- Micro-interaction animations (hover, focus, transitions)
- Responsive utilities for 13"–16" displays
- Accessibility: focus states, color contrast (WCAG AA)

## Requirements Alignment

- **Feature Requirement**: "Dark-mode-first aesthetic with keyboard-centric navigation"
- **Design Philosophy**: "Sophisticated, Minimalist, Dark-Mode First"
- **Accessibility**: "Accessible keyboard shortcuts"
- **Performance**: Minimal CSS footprint via Tailwind's JIT

## Component Design

### Design Tokens (CSS Custom Properties)

```css
/* src/styles/globals.css */
@import 'tailwindcss';

@layer base {
  :root {
    /* Backgrounds */
    --background-deep: 0 0% 4%;           /* #0A0A0A */
    --background-surface: 0 0% 9%;        /* #161616 */
    --background-elevated: 0 0% 13%;      /* #222222 */

    /* Borders */
    --border: 0 0% 20%;                   /* #333333 */
    --border-hover: 0 0% 40%;             /* #666666 */

    /* Typography */
    --text-primary: 0 0% 93%;             /* #EDEDED */
    --text-secondary: 0 0% 63%;           /* #A0A0A0 */
    --text-tertiary: 0 0% 40%;            /* #666666 */

    /* Brand Colors */
    --primary: 217 91% 60%;               /* #3B82F6 Electric Blue - Ghost */
    --primary-foreground: 0 0% 100%;
    --secondary: 160 84% 39%;             /* #10B981 Emerald - Animal */
    --secondary-foreground: 0 0% 100%;
    --accent: 258 90% 66%;                /* #8B5CF6 Violet - Insights */
    --accent-foreground: 0 0% 100%;

    /* Semantic Colors */
    --destructive: 0 84% 60%;             /* #EF4444 */
    --destructive-foreground: 0 0% 100%;
    --warning: 38 92% 50%;                /* #F59E0B */
    --warning-foreground: 0 0% 0%;
    --success: 160 84% 39%;               /* #10B981 */
    --success-foreground: 0 0% 100%;

    /* Component Tokens */
    --card: var(--background-surface);
    --card-foreground: var(--text-primary);
    --popover: var(--background-surface);
    --popover-foreground: var(--text-primary);
    --muted: var(--background-elevated);
    --muted-foreground: var(--text-secondary);
    --input: var(--background-elevated);
    --ring: var(--primary);

    /* Shadows */
    --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
    --shadow-modal: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(0, 0, 0, 0.5);
    --shadow-glow-primary: 0 4px 12px rgba(59, 130, 246, 0.3);
    --shadow-glow-accent: 0 4px 12px rgba(139, 92, 246, 0.3);

    /* Radii */
    --radius: 6px;
    --radius-lg: 8px;
    --radius-sm: 4px;

    /* Typography Scale */
    --font-sans: 'Inter', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  }
}

@layer base {
  * {
    @apply border-border;
  }

  body {
    @apply bg-background-deep text-text-primary font-sans antialiased;
  }

  /* Custom scrollbar for dark mode */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: hsl(var(--background-deep));
  }

  ::-webkit-scrollbar-thumb {
    background: hsl(var(--border));
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: hsl(var(--border-hover));
  }
}
```

### Tailwind Configuration

```typescript
// tailwind.config.ts
import type { Config } from 'tailwindcss';

export default {
  darkMode: 'class',
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: {
          deep: 'hsl(var(--background-deep))',
          surface: 'hsl(var(--background-surface))',
          elevated: 'hsl(var(--background-elevated))',
        },
        border: 'hsl(var(--border))',
        'border-hover': 'hsl(var(--border-hover))',
        text: {
          primary: 'hsl(var(--text-primary))',
          secondary: 'hsl(var(--text-secondary))',
          tertiary: 'hsl(var(--text-tertiary))',
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      borderRadius: {
        lg: 'var(--radius-lg)',
        md: 'var(--radius)',
        sm: 'var(--radius-sm)',
      },
      boxShadow: {
        card: 'var(--shadow-card)',
        modal: 'var(--shadow-modal)',
        'glow-primary': 'var(--shadow-glow-primary)',
        'glow-accent': 'var(--shadow-glow-accent)',
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.2s ease-out',
        'scale-in': 'scaleIn 0.15s ease-out',
        'pulse-subtle': 'pulseSubtle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
} satisfies Config;
```

### Button Component

```tsx
// src/components/ui/button.tsx
import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background-deep disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default:
          'bg-primary text-primary-foreground hover:bg-primary/90 shadow-glow-primary hover:-translate-y-0.5',
        destructive:
          'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline:
          'border border-border bg-transparent hover:bg-background-elevated hover:border-border-hover text-text-secondary hover:text-text-primary',
        secondary:
          'bg-background-elevated text-text-secondary hover:text-text-primary border border-border hover:border-border-hover',
        ghost:
          'text-text-secondary hover:bg-background-elevated hover:text-text-primary',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-10 rounded-md px-6',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
```

### Input Component

```tsx
// src/components/ui/input.tsx
import * as React from 'react';
import { cn } from '@/lib/utils';

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          'flex h-9 w-full rounded-md border border-border bg-background-elevated px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary transition-all duration-150',
          'focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
```

### Card Component

```tsx
// src/components/ui/card.tsx
import * as React from 'react';
import { cn } from '@/lib/utils';

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      'rounded-lg border border-border bg-background-surface shadow-card',
      className
    )}
    {...props}
  />
));
Card.displayName = 'Card';

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex flex-col space-y-1.5 p-6', className)}
    {...props}
  />
));
CardHeader.displayName = 'CardHeader';

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn('text-lg font-semibold text-text-primary', className)}
    {...props}
  />
));
CardTitle.displayName = 'CardTitle';

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn('text-sm text-text-secondary', className)}
    {...props}
  />
));
CardDescription.displayName = 'CardDescription';

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('p-6 pt-0', className)} {...props} />
));
CardContent.displayName = 'CardContent';

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex items-center p-6 pt-0', className)}
    {...props}
  />
));
CardFooter.displayName = 'CardFooter';

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent };
```

### Badge Component

```tsx
// src/components/ui/badge.tsx
import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-primary/20 text-primary border border-primary/30',
        secondary: 'bg-background-elevated text-text-secondary border border-border',
        success: 'bg-secondary/20 text-secondary border border-secondary/30',
        warning: 'bg-warning/20 text-warning border border-warning/30',
        destructive: 'bg-destructive/20 text-destructive border border-destructive/30',
        accent: 'bg-accent/20 text-accent border border-accent/30',
        outline: 'border border-border text-text-secondary',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
```

### Utility Function

```typescript
// src/lib/utils.ts
import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

## Acceptance Criteria

- [ ] TailwindCSS v4 configured with all design tokens
- [ ] Inter and JetBrains Mono fonts loaded
- [ ] All base components render correctly
- [ ] Dark mode is default (no light mode needed yet)
- [ ] Color contrast meets WCAG AA (4.5:1 for text)
- [ ] Focus states visible for keyboard navigation
- [ ] Animations are smooth (60fps)
- [ ] Components work with screen readers
- [ ] Responsive from 900px to 1920px width

## Test Plan

### Unit Tests
```typescript
import { render, screen } from '@testing-library/react';
import { Button } from '@/components/ui/button';

describe('Button', () => {
  it('renders with default variant', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-primary');
  });

  it('applies variant classes correctly', () => {
    render(<Button variant="outline">Outline</Button>);
    expect(screen.getByRole('button')).toHaveClass('border-border');
  });
});
```

### Visual Tests
- Screenshot comparison for all components
- Test in different viewport sizes
- Verify animation smoothness

## Dependencies

- TailwindCSS v4
- class-variance-authority
- clsx + tailwind-merge
- @radix-ui/react-slot
- tailwindcss-animate
- Google Fonts (Inter, JetBrains Mono)

## Next Steps

After design system complete:
1. Proceed to Module 03 (Authentication)
2. Begin shadcn/ui component additions as needed
3. Create custom components for specific features

**This design system ensures visual consistency across the entire Futurnal desktop experience.**

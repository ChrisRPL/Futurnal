Summary: Implement TailwindCSS v4 design system with monochrome aesthetic and Cinzel typography.

# 02 · Design System & Components

## Purpose

Establish the visual foundation for Futurnal's desktop shell by implementing a comprehensive design system using TailwindCSS v4 with a pure monochrome aesthetic (black/white only) and elegant serif typography.

**Criticality**: HIGH - Visual foundation for all UI components

## Scope

- TailwindCSS v4 configuration with `@tailwindcss/vite` plugin
- Monochrome color palette (pure black #000, pure white #FFF, opacity-based grays)
- Typography: Cinzel (brand headlines) + Times New Roman (taglines)
- Custom CSS classes: `.font-brand`, `.font-tagline`
- Base components: Button, Input, Card (all with sharp corners, no rounded edges)
- Micro-interaction animations (hover, focus, transitions)
- Responsive utilities for 13"–16" displays
- Accessibility: focus states, color contrast (WCAG AA)

## Requirements Alignment

- **Feature Requirement**: "Dark-mode-first aesthetic with keyboard-centric navigation"
- **Design Philosophy**: "Sophisticated, Minimalist, Dark-Mode First"
- **Accessibility**: "Accessible keyboard shortcuts"
- **Performance**: Minimal CSS footprint via Tailwind's JIT

## Design System

### Color Palette

The Futurnal design system uses a pure monochrome palette - no accent colors.

**Dark Mode (Default)**
| Token | Value | Tailwind Class |
|-------|-------|----------------|
| Background | `#000000` | `bg-black` |
| Text Primary | `#FFFFFF` | `text-white` |
| Text Secondary | `rgba(255,255,255,0.7)` | `text-white/70` |
| Text Tertiary | `rgba(255,255,255,0.5)` | `text-white/50` |
| Text Muted | `rgba(255,255,255,0.3)` | `text-white/30` |
| Border | `rgba(255,255,255,0.1)` | `border-white/10` |
| Surface | `rgba(255,255,255,0.05)` | `bg-white/5` |

**Light Mode**
| Token | Value | Tailwind Class |
|-------|-------|----------------|
| Background | `#FFFFFF` | `bg-white` |
| Text Primary | `#000000` | `text-black` |
| Text Secondary | `rgba(0,0,0,0.7)` | `text-black/70` |
| Border | `rgba(0,0,0,0.1)` | `border-black/10` |

### Typography

**Fonts**
- **Cinzel** (Google Fonts): Brand headlines, page titles, buttons
- **Times New Roman** (System): Taglines, italic emphasis

**CSS Classes**
```css
.font-brand {
  font-family: 'Cinzel', serif;
  letter-spacing: 0.05em;
}

.font-tagline {
  font-family: 'Times New Roman', 'Georgia', serif;
  font-style: italic;
}
```

**Usage**
| Element | Class | Example |
|---------|-------|---------|
| Page Title | `text-3xl font-brand tracking-wide text-white` | "Welcome Back" |
| Tagline | `font-tagline italic text-white/40` | "Know Yourself More" |
| Body Text | `text-white/80` | Description paragraphs |
| Labels | `text-sm text-white/60` | Form labels |

## CSS Configuration

### globals.css (Actual Implementation)

```css
/* ===== GOOGLE FONTS - Cinzel for Brand Headlines ===== */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&display=swap');

/* ===== TAILWIND CSS v4 ===== */
@import "tailwindcss";

/* ===== BASE STYLES ===== */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  margin: 0;
  padding: 0;
  background-color: #000000;
  color: #FFFFFF;
  font-family: 'Cinzel', serif;
  min-height: 100vh;
  line-height: 1.6;
}

/* ===== CUSTOM FONT CLASSES ===== */
.font-brand {
  font-family: 'Cinzel', serif;
  letter-spacing: 0.05em;
}

.font-tagline {
  font-family: 'Times New Roman', 'Georgia', serif;
  font-style: italic;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes pulseSubtle {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}

.animate-pulse-subtle {
  animation: pulseSubtle 2s ease-in-out infinite;
}

/* ===== LINK STYLES ===== */
a {
  color: inherit;
  text-decoration: none;
}

a:hover {
  text-decoration: none;
}

.no-underline {
  text-decoration: none !important;
}
```

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // IMPORTANT: tailwindcss() MUST come before react()
  plugins: [tailwindcss(), react()],
  // ...
});
```

## Component Patterns

### Primary Button (Dark Mode)

```tsx
<button className="w-full py-4 bg-white text-black font-medium text-lg transition-all hover:bg-white/90 disabled:opacity-50 disabled:cursor-not-allowed">
  Get Started
</button>
```

### Secondary Button (Outline)

```tsx
<button className="px-8 py-3 bg-transparent text-white border border-white/30 font-medium transition-all hover:border-white/60">
  Sign In
</button>
```

### Text Input

```tsx
<input
  type="email"
  className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
  placeholder="your@email.com"
/>
```

### Form Label

```tsx
<label className="block text-sm text-white/60 mb-2">
  Email
</label>
```

### Error Message

```tsx
<div className="px-4 py-3 bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
  {errorMessage}
</div>
```

## Logo Assets

See `FRONTEND_DESIGN.md` section 1.4 for complete logo asset documentation.

**Quick Reference**:
| Context | Dark Mode Asset | Light Mode Asset |
|---------|-----------------|------------------|
| Header Icon | `/logo_dark.png` | `/logo.png` |
| Welcome Hero | `/logo_big_dark.png` | `/logo_big.png` |
| Dashboard Header | `/logo_text_horizon_dark.png` | `/logo_text_horizon.png` |

## Acceptance Criteria

- [x] TailwindCSS v4 configured with `@tailwindcss/vite` plugin
- [x] Cinzel font loaded from Google Fonts
- [x] Times New Roman available as system fallback
- [x] Monochrome color palette (black/white only)
- [x] Dark mode is default
- [x] Color contrast meets WCAG AA (4.5:1 for text)
- [x] Focus states visible for keyboard navigation
- [x] Animations are smooth (60fps)
- [x] Responsive from 900px to 1920px width

## Dependencies

- TailwindCSS v4 (`@tailwindcss/vite` plugin)
- class-variance-authority (optional, for component variants)
- clsx + tailwind-merge (optional, for class merging)
- Google Fonts (Cinzel)

## Important Notes

### Tailwind v4 Plugin Order

The `@tailwindcss/vite` plugin MUST be listed before `react()` in vite.config.ts:

```typescript
plugins: [tailwindcss(), react()]  // Correct
plugins: [react(), tailwindcss()]  // WRONG - will break Tailwind
```

### CSS Import Order

In globals.css, the Google Fonts import can come before `@import "tailwindcss"`, but any custom `@font-face` declarations must come AFTER the Tailwind import.

### No @theme Block

Tailwind v4's `@theme` block can conflict with default utilities. The current implementation uses direct CSS values instead of a `@theme` block for simplicity and reliability.

## Next Steps

After design system complete:
1. Proceed to Module 03 (Authentication) - COMPLETED
2. Implement Module 13 (Light Mode Theme) when ready
3. Create additional components as needed

**This design system ensures visual consistency across the entire Futurnal desktop experience with a timeless monochrome aesthetic.**

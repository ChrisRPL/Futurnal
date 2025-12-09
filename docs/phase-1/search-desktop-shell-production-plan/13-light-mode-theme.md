Summary: Implement light mode theme for all pages in the desktop application with theme toggle and system preference detection.

# 13 · Light Mode Theme

## Purpose

Implement a light mode theme variant for all pages in the Futurnal desktop application. Currently, all pages use a dark theme (black background, white text). This module adds a light mode alternative with proper color inversion and a theme toggle mechanism.

> *"Adapt to your environment while maintaining the Futurnal aesthetic."*

**Criticality**: ENHANCEMENT - Improves accessibility and user preference accommodation

## Scope

- Create light mode color palette following Futurnal brand guidelines
- Implement theme context/store for global theme state management
- Add theme toggle component in Dashboard header
- Support system preference detection (prefers-color-scheme)
- Persist theme preference across sessions
- Update all pages to support both themes:
  - Welcome page
  - Login page
  - Signup page
  - ForgotPassword page
  - Dashboard (HomePage)
- Create light mode variants of logo assets (already available)

## Requirements Alignment

- **Accessibility**: Support users who prefer light interfaces
- **System Integration**: Respect OS-level dark/light mode preferences
- **Brand Consistency**: Maintain Futurnal aesthetic in both modes

## Design Specifications

### Light Mode Color Palette

| Token | Light Mode Value | Dark Mode Value | Usage |
|-------|-----------------|-----------------|-------|
| `--bg-primary` | `#FFFFFF` | `#000000` | Page background |
| `--bg-secondary` | `#F5F5F5` | `#0A0A0A` | Card/surface background |
| `--bg-tertiary` | `#EBEBEB` | `#141414` | Elevated surfaces |
| `--text-primary` | `#000000` | `#FFFFFF` | Primary text |
| `--text-secondary` | `rgba(0,0,0,0.7)` | `rgba(255,255,255,0.7)` | Secondary text |
| `--text-tertiary` | `rgba(0,0,0,0.5)` | `rgba(255,255,255,0.5)` | Tertiary text |
| `--border` | `rgba(0,0,0,0.1)` | `rgba(255,255,255,0.1)` | Borders |

### Typography

Fonts remain consistent across themes:
- **Headlines**: Cinzel (serif)
- **Taglines**: Times New Roman (serif, italic)

### Logo Assets

Light mode uses standard logo files, dark mode uses inverted variants:

| Context | Light Mode | Dark Mode |
|---------|-----------|-----------|
| Header icon | `/logo.png` | `/logo_dark.png` |
| Welcome hero | `/logo_big.png` | `/logo_big_dark.png` |
| Dashboard header | `/logo_text_horizon.png` | `/logo_text_horizon_dark.png` |

## Component Design

### Theme Context

```typescript
// src/contexts/ThemeContext.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextType {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('futurnal-theme');
    return (stored as Theme) || 'system';
  });

  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('dark');

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const updateResolvedTheme = () => {
      if (theme === 'system') {
        setResolvedTheme(mediaQuery.matches ? 'dark' : 'light');
      } else {
        setResolvedTheme(theme);
      }
    };

    updateResolvedTheme();
    mediaQuery.addEventListener('change', updateResolvedTheme);

    return () => mediaQuery.removeEventListener('change', updateResolvedTheme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('futurnal-theme', theme);
    document.documentElement.classList.toggle('dark', resolvedTheme === 'dark');
    document.documentElement.classList.toggle('light', resolvedTheme === 'light');
  }, [theme, resolvedTheme]);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
```

### Theme Toggle Component

```tsx
// src/components/ui/ThemeToggle.tsx
import { Sun, Moon, Monitor } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  const cycleTheme = () => {
    const themes: Array<'light' | 'dark' | 'system'> = ['light', 'dark', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex]);
  };

  const Icon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor;

  return (
    <button
      onClick={cycleTheme}
      className="p-2 text-current opacity-60 hover:opacity-100 transition-opacity"
      aria-label={`Current theme: ${theme}. Click to change.`}
    >
      <Icon className="w-4 h-4" />
    </button>
  );
}
```

### CSS Theme Variables

```css
/* src/styles/globals.css - Theme additions */

:root {
  /* Light mode (default) */
  --bg-primary: #FFFFFF;
  --bg-secondary: #F5F5F5;
  --bg-tertiary: #EBEBEB;
  --text-primary: #000000;
  --text-secondary: rgba(0, 0, 0, 0.7);
  --text-tertiary: rgba(0, 0, 0, 0.5);
  --border-color: rgba(0, 0, 0, 0.1);
}

.dark {
  --bg-primary: #000000;
  --bg-secondary: #0A0A0A;
  --bg-tertiary: #141414;
  --text-primary: #FFFFFF;
  --text-secondary: rgba(255, 255, 255, 0.7);
  --text-tertiary: rgba(255, 255, 255, 0.5);
  --border-color: rgba(255, 255, 255, 0.1);
}

body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
}
```

### Updated Page Example (Welcome.tsx)

```tsx
// Theme-aware Welcome page
import { useTheme } from '@/contexts/ThemeContext';

export default function Welcome() {
  const { resolvedTheme } = useTheme();
  const logoSrc = resolvedTheme === 'dark' ? '/logo_dark.png' : '/logo.png';
  const bigLogoSrc = resolvedTheme === 'dark' ? '/logo_big_dark.png' : '/logo_big.png';

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] flex flex-col">
      <header className="w-full px-8 py-6">
        <img src={logoSrc} alt="Futurnal" className="h-10 w-auto" />
      </header>
      {/* ... rest of component using CSS variables */}
    </div>
  );
}
```

## Pages to Update

| Page | File Path | Changes Required |
|------|-----------|------------------|
| Welcome | `desktop/src/pages/Welcome.tsx` | Theme-aware logos, CSS variable colors |
| Login | `desktop/src/pages/Login.tsx` | Theme-aware logo, CSS variable colors |
| Signup | `desktop/src/pages/Signup.tsx` | Theme-aware logo, CSS variable colors |
| ForgotPassword | `desktop/src/pages/ForgotPassword.tsx` | Theme-aware logo, CSS variable colors |
| Dashboard | `desktop/src/App.tsx` (HomePage) | Theme toggle in header, theme-aware logo, CSS variable colors |

## Acceptance Criteria

- [ ] Theme context provides current theme state to all components
- [ ] Theme toggle cycles through light/dark/system modes
- [ ] System preference detection works correctly
- [ ] Theme preference persists across app restarts
- [ ] All pages render correctly in light mode
- [ ] All pages render correctly in dark mode
- [ ] Logo assets switch based on current theme
- [ ] No flash of wrong theme on page load
- [ ] Text remains readable in both themes
- [ ] Contrast ratios meet WCAG AA standards

## Test Plan

### Manual Testing

1. Open app - should use system preference by default
2. Toggle to light mode - verify all pages use light colors
3. Toggle to dark mode - verify all pages use dark colors
4. Toggle to system - verify follows OS preference
5. Restart app - verify theme preference persisted
6. Change OS theme while on system mode - verify app updates

### Unit Tests

```typescript
describe('ThemeContext', () => {
  it('should default to system preference', () => {});
  it('should persist theme to localStorage', () => {});
  it('should resolve system theme correctly', () => {});
});
```

## Dependencies

- No new packages required
- Uses existing Tailwind CSS v4 setup
- Uses existing localStorage for persistence

## Implementation Notes

1. **CSS-First Approach**: Use CSS custom properties (variables) for colors to enable smooth theme transitions
2. **Tailwind Integration**: Can use Tailwind's dark mode variant with `.dark` class on root
3. **No Flash**: Set theme class before React hydration using inline script in index.html
4. **Transitions**: Add `transition-colors` to body for smooth theme switching

## Next Steps

After light mode implementation:
1. Consider adding more granular theme customization
2. Add high contrast mode for accessibility
3. Consider per-page theme overrides if needed

---

**This module enhances user experience by providing theme flexibility while maintaining the Futurnal brand aesthetic across both light and dark modes.**

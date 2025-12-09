# Module 03: Authentication UI Specification

## Brand Identity

### Logo Assets
Located in `/assets/` and copied to `/desktop/public/`:
- `logo.png` - Icon mark (black geometric shapes with center dot)
- `logo_text_horizontal.png` - Full logo with "FUTURNAL" and "Know Yourself More" tagline

### Typography

| Use Case | Font | Weight | Style |
|----------|------|--------|-------|
| Brand Headlines | Cinzel | Bold (700) | Normal |
| Taglines | Times New Roman | Regular | Italic |
| Body Text | Inter | Regular (400) | Normal |
| UI Labels | Inter | Medium (500) | Normal |
| Code | JetBrains Mono | Regular | Normal |

**Font Loading:**
```css
/* Cinzel via Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&display=swap');

/* CSS Classes */
.font-brand { font-family: 'Cinzel', serif; letter-spacing: 0.05em; }
.font-tagline { font-family: 'Times New Roman', Georgia, serif; font-style: italic; }
```

### Color Palette

**Monochrome Only** - No blues, greens, purples, or colored glows.

| Token | Value | Usage |
|-------|-------|-------|
| Background | `#000000` | Page background |
| Surface | `#0A0A0A` | Elevated surfaces |
| Elevated | `#141414` | Inputs, hover states |
| Border | `#262626` | Dividers, borders |
| Border Hover | `#404040` | Active borders |
| Text Primary | `#FFFFFF` | Headlines, primary text |
| Text Secondary | `#A3A3A3` | Descriptions, labels |
| Text Tertiary | `#737373` | Placeholders, hints |
| Error | `#DC2626` | Error messages |

**Opacity Utilities:**
- `white/80` - Primary text in body
- `white/60` - Secondary text
- `white/40` - Tertiary text, icons
- `white/30` - Hints, disabled
- `white/10` - Borders
- `white/5` - Subtle backgrounds

---

## Layout Principles

### CRITICAL: No Squeezed Content

**Problem:** Content was constrained to narrow centered columns making text unreadable.

**Solution:** Use full-width layouts with proper max-widths:

```tsx
// ❌ WRONG - Too narrow, squeezed
<div className="max-w-md mx-auto">
  <p className="max-w-lg">Long text here...</p>
</div>

// ✅ CORRECT - Full width with comfortable max-width
<main className="px-8 py-12">
  <div className="max-w-7xl mx-auto">
    <p className="text-xl leading-relaxed max-w-2xl">
      Long text here...
    </p>
  </div>
</main>
```

### Page Structure

```
┌─────────────────────────────────────────────────────────┐
│ Header: Logo left, nav/user right                       │
│ px-8 py-6                                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Main Content: flex-1, centered vertically for auth      │
│ px-8 py-12, max-w-7xl mx-auto                          │
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │ Content container                                │  │
│   │ max-w-3xl for welcome, max-w-md for forms       │  │
│   └─────────────────────────────────────────────────┘  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Footer: border-t, px-8 py-6                            │
└─────────────────────────────────────────────────────────┘
```

### Spacing Scale

| Name | Value | Usage |
|------|-------|-------|
| px-8 | 2rem | Page horizontal padding |
| py-6 | 1.5rem | Header/footer vertical |
| py-12 | 3rem | Main content vertical |
| gap-4 | 1rem | Grid gaps |
| gap-6 | 1.5rem | Section spacing |
| space-y-12 | 3rem | Major section spacing |
| mb-6 | 1.5rem | Heading bottom margin |

---

## Wording Guidelines

### CRITICAL: No Ghost/Animal Terminology

These are internal technical concepts users won't understand:
- ❌ "Ghost Active" / "Ghost Idle"
- ❌ "Your Ghost awaits"
- ❌ "Ghost→Animal evolution"
- ❌ "Animal behavior"

**Use Instead:**
- ✅ "Active" / "Idle" (for system status)
- ✅ "Your personal knowledge graph awaits"
- ✅ "AI evolution" / "personalized intelligence"
- ✅ "Developed understanding"

### Approved Copy

**Welcome Page:**
- Headline: (Use logo with tagline image)
- Subhead: "The world's first AI evolution platform that transforms generic AI into deeply personalized intelligence."
- Description: "Your AI learns continuously from your unique personal data stream, developing genuine understanding of your patterns and growth."
- CTA: "Get Started" / "Sign In"
- Footer: "Privacy-first. Local-first. Your data remains yours."

**Login Page:**
- Title: "Welcome Back"
- Subtitle: "Sign in to continue your journey"
- Footer: "Your data remains yours. We only authenticate your identity."

**Signup Page:**
- Title: "Create Account"
- Subtitle: "Begin your journey to self-knowledge"
- Footer: "By creating an account, you agree to our privacy-first approach."

**Dashboard:**
- Greeting: "Good morning/afternoon/evening" (time-based)
- Subtitle: "Your personal knowledge graph awaits."
- Status: "Active" / "Idle" (not "Ghost Active")

**Tagline (always in footer):**
- "Know Yourself More" (italic, Times New Roman)

---

## Component Specifications

### Welcome Page

```
Full-screen black background
├── Header (px-8 py-6)
│   └── Logo icon only (h-10)
├── Main (flex-1, centered)
│   └── Container (max-w-3xl)
│       ├── Logo with text (h-24 md:h-32, centered)
│       ├── Value proposition (text-xl md:text-2xl, white/80)
│       ├── Description (text-lg, font-tagline italic, white/60)
│       └── CTAs (flex, gap-4)
│           ├── Primary: bg-white text-black
│           └── Secondary: border border-white/30
└── Footer (border-t border-white/10, px-8 py-8)
    └── Privacy note + tagline
```

### Login/Signup Pages

```
Full-screen black background
├── Header (px-8 py-6, flex justify-between)
│   ├── Logo icon (h-10)
│   └── Link to other auth page
├── Main (flex-1, centered)
│   └── Form container (max-w-md)
│       ├── Title (text-3xl font-brand)
│       ├── Subtitle (text-white/60)
│       ├── Form fields
│       │   └── Input: bg-white/5 border-white/10
│       ├── Error message (if any)
│       ├── Submit button (bg-white text-black, full width)
│       └── Links (forgot password, switch auth mode)
└── Footer (text-center, text-xs text-white/30)
```

### Dashboard

```
Full-screen black background
├── Header (border-b border-white/10)
│   └── Container (max-w-7xl mx-auto, px-8 py-5)
│       ├── Logo + "FUTURNAL" text
│       └── Status indicator + User email + Logout
├── Main (px-8 py-12)
│   └── Container (max-w-7xl mx-auto, space-y-12)
│       ├── Greeting section
│       │   ├── H1 (font-brand, text-3xl md:text-4xl)
│       │   └── Subtitle (text-white/60)
│       ├── Stats grid (3 columns on md+)
│       │   └── Card: p-6 bg-white/5 border-white/10
│       ├── Graph placeholder
│       │   └── Centered, border border-white/10, p-12
│       ├── Quick actions grid (4 columns on lg+)
│       │   ├── Primary: bg-white text-black
│       │   └── Secondary: border border-white/10
│       └── Recent activity
│           └── Card: border border-white/10 p-6
└── Footer (border-t border-white/5, px-8 py-6)
    └── Version + tagline
```

---

## Input Styles

```css
/* Base input */
.input {
  width: 100%;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: white;
  transition: border-color 150ms;
}

.input::placeholder {
  color: rgba(255, 255, 255, 0.3);
}

.input:focus {
  outline: none;
  border-color: rgba(255, 255, 255, 0.3);
}

/* Tailwind equivalent */
className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
```

---

## Button Styles

### Primary Button
```css
className="w-full py-4 bg-white text-black font-medium text-lg transition-all hover:bg-white/90 disabled:opacity-50 disabled:cursor-not-allowed"
```

### Secondary/Outline Button
```css
className="px-8 py-3 bg-transparent text-white border border-white/30 font-medium transition-all hover:border-white/60"
```

### Ghost Button (for logout, etc)
```css
className="text-white/40 hover:text-white transition-colors"
```

---

## Animation Classes

```css
.animate-fade-in {
  animation: fade-in 0.6s ease-out;
}

.animate-pulse-subtle {
  animation: pulse-subtle 2s ease-in-out infinite;
}

@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes pulse-subtle {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

---

## Known Issues to Fix

### 1. Squeezed Content
**Status:** Needs verification after refresh
**Solution:** Ensure all pages use `max-w-7xl` for dashboard, `max-w-3xl` for welcome, proper padding `px-8`

### 2. Font Loading
**Status:** Cinzel loaded via Google Fonts import in globals.css
**Verify:** Check that `font-brand` class renders in Cinzel

### 3. Logo Display
**Status:** Assets copied to `/desktop/public/`
**Verify:** `logo.png` and `logo_text_horizontal.png` load correctly

---

## File References

| File | Purpose |
|------|---------|
| `src/styles/globals.css` | Design tokens, fonts, animations |
| `src/pages/Welcome.tsx` | Welcome/landing page |
| `src/pages/Login.tsx` | Sign in page |
| `src/pages/Signup.tsx` | Registration page |
| `src/pages/ForgotPassword.tsx` | Password reset |
| `src/App.tsx` | Dashboard + routing |
| `src/components/auth/ProtectedRoute.tsx` | Auth guard |

---

## Testing Checklist

- [ ] Welcome page renders with full-width layout
- [ ] Logo images load correctly
- [ ] Cinzel font displays for headlines
- [ ] Times New Roman italic for taglines
- [ ] No "Ghost" or "Animal" text visible
- [ ] Forms are readable and not squeezed
- [ ] Buttons have correct black/white styling
- [ ] Status shows "Active/Idle" not "Ghost Active/Idle"
- [ ] Auth flow works end-to-end

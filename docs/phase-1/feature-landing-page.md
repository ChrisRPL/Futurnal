# Feature: Futurnal Landing Page & Web Portal

**Status**: Planned
**Priority**: HIGH
**Phase**: 1 (Required for desktop app distribution)

## Overview

The Futurnal Landing Page is a separate web application that serves as the public face of Futurnal, handling marketing, user signup, subscription management, and desktop app distribution. This is a companion project to the Tauri desktop app.

> **"Know Yourself More"** — The world's first AI evolution platform that transforms generic AI into deeply personalized intelligence.

## Purpose

1. **Marketing & Conversion**: Showcase Futurnal's value proposition, features, and differentiation
2. **User Acquisition**: Handle signup flow with tier selection
3. **Payment Processing**: Integrate Stripe for Pro subscriptions
4. **Subscription Management**: Provide access to Stripe Customer Portal
5. **App Distribution**: Download links for macOS, Windows, Linux desktop app

## Responsibilities

| Function | Landing Page | Desktop App |
|----------|-------------|-------------|
| Marketing/Features | ✓ | - |
| Pricing display | ✓ | - |
| User signup | ✓ | Redirect here |
| User login | ✓ | ✓ |
| Stripe Checkout | ✓ | - |
| Subscription portal | ✓ | Opens browser |
| Tier verification | ✓ (API) | ✓ (calls API) |
| Desktop download | ✓ | - |
| Core product features | - | ✓ |

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Framework | Next.js 14+ | SSR for SEO, app router |
| Styling | TailwindCSS v4 | Matches desktop app design system |
| Auth | Firebase Authentication | Shared with desktop app |
| Payments | Stripe | Industry standard |
| Database | Firestore | Subscription storage |
| Hosting | Vercel | Optimal for Next.js |
| Analytics | Vercel Analytics / Plausible | Privacy-respecting |

## Design System

The landing page must follow the same design guidelines as the desktop app:

### Typography
- **Headlines**: Cinzel (Google Fonts) - elegant serif
- **Taglines**: Times New Roman / Georgia - italic
- **Body**: System sans-serif for readability on web

### Colors
Pure monochrome aesthetic:
- **Background**: `#000000` (Dark mode default)
- **Text Primary**: `#FFFFFF`
- **Text Secondary**: `rgba(255, 255, 255, 0.7)`
- **Borders**: `rgba(255, 255, 255, 0.1)`
- **Accents**: Minimal, only for CTAs

### Visual Style
- Dark-mode first (matching desktop app)
- Sharp corners (no rounded elements)
- Generous whitespace
- Elegant, sophisticated, premium feel

## Page Structure

### 1. Home / Landing Page (`/`)
The main marketing page showcasing Futurnal's value proposition.

**Sections**:
- **Hero**: Large logo, tagline, value proposition, primary CTA
- **Problem/Solution**: Why personal AI needs to evolve
- **Three Phases**: Visual timeline of Archivist → Analyst → Guide
- **Features**: Key capabilities with elegant icons
- **Privacy First**: Emphasis on local-first, data sovereignty
- **Pricing Preview**: Quick tier comparison
- **CTA**: Download / Get Started

**Copy Direction**:
> "The world's first AI evolution platform that transforms generic AI into deeply personalized intelligence."

> "From 'What did I know?' to 'Why do I think this?'"

> "Your data remains yours. Always."

### 2. Pricing Page (`/pricing`)
Detailed pricing comparison and subscription selection.

**Tiers**:

#### Free: "Explorer"
- Core search functionality
- Up to 3 data sources
- Personal Knowledge Graph
- Interactive PKG visualization
- Local-only processing
- Privacy controls & audit logs

#### Pro: "Futurnal Pro" ($XX/month)
- Everything in Free
- **Unlimited data sources**
- Encrypted cloud backup
- **Phase 2: The Analyst**
  - Emergent Insights Engine
  - Curiosity Engine
  - Proactive notifications
- **Phase 3: The Guide**
  - Conversational causal exploration
  - Causal Inference Engine
  - Aspirational Self goals
- Priority support

**UI**: Two elegant cards side-by-side, monochrome with subtle differentiation

### 3. Signup Page (`/signup`)
New user registration with tier selection.

**Flow**:
1. Email + Password form
2. Tier selection (Free / Pro)
3. If Pro: Redirect to Stripe Checkout
4. If Free: Direct to success page
5. Success: Download links + welcome

### 4. Login Page (`/login`)
Existing user authentication.

**Features**:
- Email/Password login
- "Forgot Password" link
- Link to signup

### 5. Account Page (`/account`)
Authenticated user's account management.

**Sections**:
- Profile (email, change password)
- Subscription status
- "Manage Subscription" → Stripe Customer Portal
- Download links for desktop app

### 6. Download Page (`/download`)
Desktop app distribution.

**Content**:
- Platform detection (show relevant download first)
- Download buttons: macOS (.dmg), Windows (.exe/.msi), Linux (.AppImage/.deb)
- System requirements
- Installation instructions
- Link to documentation

## Stripe Integration

### Products & Prices
```
Product: Futurnal Pro
├── Price: pro_monthly ($XX/month)
└── Price: pro_yearly ($XX/year, 2 months free)
```

### Checkout Flow
1. User selects Pro tier on signup/pricing
2. Create Stripe Checkout Session (server-side)
3. Redirect to Stripe Checkout
4. On success: Return to `/signup/success?session_id=xxx`
5. Webhook updates Firestore subscription document

### Customer Portal
- Accessed via `/account` "Manage Subscription" button
- Handles: Update payment method, Cancel, View invoices
- Stripe-hosted (no custom UI needed)

### Webhook Events
```typescript
// Handle these Stripe webhook events:
'checkout.session.completed'  // New subscription
'customer.subscription.updated'  // Tier/status change
'customer.subscription.deleted'  // Cancellation
'invoice.payment_failed'  // Payment issue
```

## Firebase Integration

### Shared Authentication
Both landing page and desktop app use the same Firebase project:
- Users created on web can login on desktop
- Same email/password credentials
- Shared user UIDs

### Firestore Schema
```typescript
// /users/{userId}
{
  email: string;
  createdAt: Timestamp;
  lastLogin: Timestamp;
}

// /subscriptions/{userId}
{
  tier: 'free' | 'pro';
  status: 'active' | 'canceled' | 'past_due';
  stripeCustomerId: string;
  stripeSubscriptionId: string;
  currentPeriodEnd: Timestamp;
  cancelAtPeriodEnd: boolean;
}
```

## API Endpoints

### GET /api/subscription
Called by desktop app to verify tier.

**Request**:
```
Headers: Authorization: Bearer <firebase-id-token>
```

**Response**:
```json
{
  "tier": "free" | "pro",
  "isActive": true,
  "expiresAt": "2025-01-15T00:00:00Z" | null
}
```

### POST /api/checkout
Create Stripe Checkout session.

**Request**:
```json
{
  "priceId": "price_xxx",
  "successUrl": "https://futurnal.com/signup/success",
  "cancelUrl": "https://futurnal.com/pricing"
}
```

**Response**:
```json
{
  "sessionId": "cs_xxx",
  "url": "https://checkout.stripe.com/..."
}
```

### POST /api/portal
Create Stripe Customer Portal session.

**Response**:
```json
{
  "url": "https://billing.stripe.com/..."
}
```

## SEO & Meta

### Key Pages
- `/` - Main landing, full SEO optimization
- `/pricing` - Pricing comparison
- `/download` - App download

### Meta Tags
```html
<title>Futurnal - Know Yourself More | Personal AI Evolution Platform</title>
<meta name="description" content="The world's first AI evolution platform that transforms generic AI into deeply personalized intelligence. Privacy-first, local-first." />
```

### Open Graph
- Custom OG images per page
- Twitter card support

## Analytics

Privacy-respecting analytics only:
- Vercel Analytics (built-in, privacy-first)
- Or Plausible Analytics (self-hosted option)

Track:
- Page views
- Conversion funnel (signup, checkout, download)
- No personal data collection

## Implementation Plan

### Phase 1: Core Pages
1. Landing page with hero, features, pricing preview
2. Pricing page with tier comparison
3. Download page with platform detection

### Phase 2: Authentication
4. Signup page with Firebase
5. Login page
6. Account page

### Phase 3: Payments
7. Stripe Checkout integration
8. Webhook handling
9. Customer Portal integration

### Phase 4: Polish
10. SEO optimization
11. Analytics setup
12. Performance optimization

## Environment Variables

```env
# Firebase
NEXT_PUBLIC_FIREBASE_API_KEY=xxx
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=xxx
NEXT_PUBLIC_FIREBASE_PROJECT_ID=xxx

# Stripe
STRIPE_SECRET_KEY=sk_xxx
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_PRO_MONTHLY_PRICE_ID=price_xxx
STRIPE_PRO_YEARLY_PRICE_ID=price_xxx

# App URLs
NEXT_PUBLIC_APP_URL=https://futurnal.com
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Landing page load time | <2s |
| Lighthouse score | >90 |
| Signup conversion | Track |
| Pro conversion | Track |
| Download completion | Track |

## Dependencies

### NPM Packages
- next
- react
- tailwindcss
- firebase
- stripe
- @stripe/stripe-js

### External Services
- Firebase (shared project with desktop app)
- Stripe
- Vercel (hosting)

## Project Structure

```
landing/
├── app/
│   ├── page.tsx           # Home/Landing
│   ├── pricing/
│   │   └── page.tsx
│   ├── signup/
│   │   ├── page.tsx
│   │   └── success/
│   │       └── page.tsx
│   ├── login/
│   │   └── page.tsx
│   ├── account/
│   │   └── page.tsx
│   ├── download/
│   │   └── page.tsx
│   └── api/
│       ├── subscription/
│       ├── checkout/
│       ├── portal/
│       └── webhook/
├── components/
├── lib/
│   ├── firebase.ts
│   └── stripe.ts
├── styles/
│   └── globals.css
└── public/
    └── assets/
```

## Next Steps

1. Initialize Next.js project in `/landing` directory
2. Configure TailwindCSS with shared design tokens
3. Implement landing page hero and features
4. Set up Firebase (shared config with desktop)
5. Integrate Stripe products and checkout

---

**The landing page is the gateway to Futurnal—converting visitors into users and handling the complexity of payments so the desktop app can focus on delivering the core product experience.**

Summary: Implement tier verification and feature gating in the desktop app. Pricing/payments handled on landing page.

# 04 · Tier Verification

## Purpose

Implement subscription tier verification and feature gating in the desktop application. The desktop app does **not** handle payments directly—all pricing, signup, and subscription management happens on the landing page (web). The desktop app simply verifies the user's tier and gates features accordingly.

**Criticality**: HIGH - Feature gating and tier enforcement

## Architecture Decision

### Why This Approach?

1. **Stripe works best in browsers** - Checkout, Customer Portal, payment forms are web-optimized
2. **SEO/Marketing** - Landing page is crawlable, shareable, linkable
3. **Subscription management complexity** - Stripe Customer Portal handles upgrades, cancellations, invoices
4. **Desktop app stays simple** - Just verify tier via API call, no payment logic

### Responsibility Split

| Component | Desktop App | Landing Page (Web) |
|-----------|-------------|-------------------|
| Login | ✓ | ✓ |
| Signup | Redirect to web | ✓ |
| Tier verification | ✓ (via API) | - |
| Feature gating | ✓ | - |
| Pricing display | - | ✓ |
| Stripe Checkout | - | ✓ |
| Subscription management | Opens browser | ✓ (Stripe Portal) |

## Scope

- Tier verification hook (`useSubscription`)
- Feature gating hook (`useTierLimits`)
- API call to backend for subscription status
- Upgrade prompts when hitting limits
- "Manage Subscription" button → opens browser to web portal
- Graceful handling when backend unavailable

## Component Design

### Subscription Hook

```typescript
// src/hooks/useSubscription.ts
import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';

export type SubscriptionTier = 'free' | 'pro';

interface Subscription {
  tier: SubscriptionTier;
  isActive: boolean;
  expiresAt: Date | null;
}

export function useSubscription() {
  const { user } = useAuth();
  const [subscription, setSubscription] = useState<Subscription>({
    tier: 'free',
    isActive: true,
    expiresAt: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) {
      setSubscription({ tier: 'free', isActive: true, expiresAt: null });
      setLoading(false);
      return;
    }

    const fetchSubscription = async () => {
      try {
        const token = await user.getIdToken();
        const response = await fetch(
          `${import.meta.env.VITE_API_URL}/subscription`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );

        if (!response.ok) {
          throw new Error('Failed to fetch subscription');
        }

        const data = await response.json();
        setSubscription({
          tier: data.tier || 'free',
          isActive: data.isActive ?? true,
          expiresAt: data.expiresAt ? new Date(data.expiresAt) : null,
        });
      } catch (err) {
        console.error('Subscription fetch error:', err);
        setError('Unable to verify subscription. Using free tier.');
        // Default to free tier on error
        setSubscription({ tier: 'free', isActive: true, expiresAt: null });
      } finally {
        setLoading(false);
      }
    };

    fetchSubscription();
  }, [user]);

  const isPro = subscription.tier === 'pro' && subscription.isActive;

  return {
    ...subscription,
    isPro,
    loading,
    error,
  };
}
```

### Tier Limits Hook

```typescript
// src/hooks/useTierLimits.ts
import { useSubscription } from './useSubscription';

interface TierLimits {
  maxDataSources: number;
  hasCloudBackup: boolean;
  // Phase 2 features
  hasEmergentInsights: boolean;
  hasCuriosityEngine: boolean;
  // Phase 3 features
  hasCausalExploration: boolean;
  hasAspirationalSelf: boolean;
}

const TIER_LIMITS: Record<'free' | 'pro', TierLimits> = {
  free: {
    maxDataSources: 3,
    hasCloudBackup: false,
    hasEmergentInsights: false,
    hasCuriosityEngine: false,
    hasCausalExploration: false,
    hasAspirationalSelf: false,
  },
  pro: {
    maxDataSources: Infinity,
    hasCloudBackup: true,
    hasEmergentInsights: true,
    hasCuriosityEngine: true,
    hasCausalExploration: true,
    hasAspirationalSelf: true,
  },
};

export function useTierLimits() {
  const { tier, isPro } = useSubscription();
  const limits = TIER_LIMITS[tier];

  const canAddDataSource = (currentCount: number): boolean => {
    return currentCount < limits.maxDataSources;
  };

  const getUpgradeReason = (feature: string): string => {
    return `${feature} is available with Futurnal Pro. Visit futurnal.com to upgrade.`;
  };

  return {
    limits,
    isPro,
    canAddDataSource,
    getUpgradeReason,
  };
}
```

### Manage Subscription Handler

```typescript
// src/lib/subscription.ts
import { open } from '@tauri-apps/plugin-shell';

const PORTAL_URL = import.meta.env.VITE_LANDING_URL || 'https://futurnal.com';

export async function openSubscriptionPortal() {
  await open(`${PORTAL_URL}/account`);
}

export async function openUpgradePage() {
  await open(`${PORTAL_URL}/pricing`);
}

export async function openSignupPage() {
  await open(`${PORTAL_URL}/signup`);
}
```

### Upgrade Prompt Component

```tsx
// src/components/UpgradePrompt.tsx
import { openUpgradePage } from '@/lib/subscription';

interface UpgradePromptProps {
  reason: string;
  onClose: () => void;
}

export function UpgradePrompt({ reason, onClose }: UpgradePromptProps) {
  const handleUpgrade = async () => {
    await openUpgradePage();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
      <div className="w-full max-w-md p-8 bg-black border border-white/10">
        <h2 className="text-2xl font-brand text-white mb-4">
          Upgrade to Pro
        </h2>
        <p className="text-white/60 mb-6">{reason}</p>
        <div className="flex gap-4">
          <button
            onClick={onClose}
            className="flex-1 py-3 border border-white/30 text-white/70 hover:border-white/60"
          >
            Maybe Later
          </button>
          <button
            onClick={handleUpgrade}
            className="flex-1 py-3 bg-white text-black font-medium hover:bg-white/90"
          >
            View Plans
          </button>
        </div>
      </div>
    </div>
  );
}
```

### Dashboard Integration

```tsx
// In Dashboard header or settings
import { openSubscriptionPortal } from '@/lib/subscription';
import { useSubscription } from '@/hooks/useSubscription';

function UserMenu() {
  const { tier, isPro } = useSubscription();

  return (
    <div className="flex items-center gap-4">
      <span className="text-sm text-white/40">
        {isPro ? 'Pro' : 'Free'}
      </span>
      <button
        onClick={openSubscriptionPortal}
        className="text-sm text-white/60 hover:text-white"
      >
        Manage Subscription
      </button>
    </div>
  );
}
```

## User Flows

### New User (No Account)
1. Opens desktop app → Welcome screen
2. Clicks "Get Started" → Opens browser to futurnal.com/signup
3. Creates account on web, selects tier
4. Downloads desktop app (if not already)
5. Returns to desktop app → Logs in
6. Tier synced from backend

### Existing User (Free → Pro Upgrade)
1. Hits feature limit (e.g., 4th data source)
2. Sees upgrade prompt in desktop app
3. Clicks "View Plans" → Opens browser to futurnal.com/pricing
4. Completes payment on web
5. Returns to desktop app → Tier auto-updates via API

### Pro User (Manage Subscription)
1. Clicks "Manage Subscription" in settings
2. Opens browser to Stripe Customer Portal
3. Can update payment, cancel, view invoices
4. Changes sync to desktop app via API

## Acceptance Criteria

- [ ] useSubscription hook fetches tier from backend API
- [ ] useTierLimits correctly gates features by tier
- [ ] Free users limited to 3 data sources
- [ ] Upgrade prompts appear when hitting limits
- [ ] "Manage Subscription" opens browser to web portal
- [ ] "Get Started" (unauthenticated) opens browser to signup
- [ ] Graceful fallback to free tier if API unavailable
- [ ] Tier updates reflect without app restart

## API Contract

### GET /subscription
**Headers**: `Authorization: Bearer <firebase-token>`

**Response (200)**:
```json
{
  "tier": "free" | "pro",
  "isActive": true,
  "expiresAt": "2025-01-15T00:00:00Z" | null
}
```

**Response (401)**: Unauthorized
**Response (500)**: Server error (fallback to free tier)

## Dependencies

- @tauri-apps/plugin-shell (for opening browser)
- Firebase Auth (for user token)
- Backend API endpoint

## Environment Variables

```env
VITE_API_URL=https://api.futurnal.com
VITE_LANDING_URL=https://futurnal.com
```

## Next Steps

1. Implement backend `/subscription` endpoint
2. Create landing page with Stripe integration (separate project)
3. Integrate tier limits with connector management (Module 08)

**This module enables feature gating while keeping payment complexity on the web platform.**

Summary: Implement Stripe integration for Pro tier subscriptions with pricing selection UI and tier enforcement.

# 04 · Pricing Tier & Stripe

## Purpose

Implement the pricing tier selection flow and Stripe payment integration for Pro subscriptions. This module creates the onboarding pricing screen, Stripe Checkout integration, subscription status management, and tier enforcement middleware that limits Free users to 3 data sources.

**Criticality**: HIGH - Revenue enablement and tier enforcement

## Scope

- Pricing selection screen (onboarding step 3)
- Two-card layout: Free "Archivist" vs Pro "Futurnal Pro"
- Stripe Checkout integration for Pro tier
- Subscription status storage (Firestore)
- Tier enforcement middleware (Free: 3 sources, Pro: unlimited)
- Upgrade prompts when hitting limits
- Subscription management portal link
- Webhook handling for subscription events

## Requirements Alignment

- **Business Model**: Freemium with Pro tier from FRONTEND_DESIGN.md
- **Free Tier "The Archivist"**: Up to 3 data sources, local-only, Phase 1 features only
- **Pro Tier "Futurnal Pro"**: Unlimited sources, cloud backup, Phase 2 + Phase 3 features
- **Privacy Note**: Payment info handled by Stripe, not stored locally

## Futurnal's Three-Phase Evolution

Futurnal transforms a generic AI ("Ghost") into a deeply personalized intelligence ("Animal") across three phases:

### Phase 1: The Archivist (Months 1-4)
**Codename**: "Grounding the Ghost"
- Experiential connectors (Local Files, Obsidian, IMAP, GitHub)
- Personal Knowledge Graph (PKG) construction
- Hybrid search interface (semantic + graph traversal)
- Interactive PKG visualization
- Privacy-first foundation with consent/audit logging
- **Available in Free tier**

### Phase 2: The Analyst (Months 5-9)
**Codename**: "Awakening Animal Instincts"
- **Emergent Insights Engine**: Autonomous pattern analysis surfacing non-obvious correlations
  - *Example: "75% of proposals written Monday are accepted vs 30% on Friday"*
- **Curiosity Engine**: Identifies knowledge gaps and unexplored connections
  - *Example: "15 notes reference 'Project Titan' but none link to your aspiration 'Lead high-impact projects'"*
- Intelligent ranking aligned with user feedback
- Proactive notification system
- **Pro tier exclusive**

### Phase 3: The Guide (Months 10-15)
**Codename**: "The Emergent Animal Brain"
- **Conversational Causal Exploration**: Dialogue-driven hypothesis investigation
- **Causal Inference Engine**: The Animal's "world model" understanding the "why" behind patterns
  - Hypothesis generation (Judea Pearl-inspired)
  - Confounder identification
  - Guided temporal reasoning
- **Aspirational Self Integration**: Tracking progress toward user-defined goals
- **Reward Signal Dashboard**: Visualization of goal alignment
  - *Example: "✓ 3 new notes aligned with your goal 'Learn Causal Inference'"*
  - *Example: "⚠️ 80% of articles saved unrelated to stated aspirations"*
- **Pro tier exclusive**

## Component Design

### Stripe Configuration

```typescript
// src/lib/stripe.ts
import { loadStripe, type Stripe } from '@stripe/stripe-js';

let stripePromise: Promise<Stripe | null>;

export function getStripe() {
  if (!stripePromise) {
    stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY);
  }
  return stripePromise;
}

// Stripe product/price IDs (from Stripe Dashboard)
export const STRIPE_PRICES = {
  pro_monthly: import.meta.env.VITE_STRIPE_PRO_MONTHLY_PRICE_ID,
  pro_yearly: import.meta.env.VITE_STRIPE_PRO_YEARLY_PRICE_ID,
};

export async function createCheckoutSession(
  userId: string,
  priceId: string,
  successUrl: string,
  cancelUrl: string
): Promise<string> {
  // Call your backend/Firebase function to create Stripe Checkout Session
  const response = await fetch(import.meta.env.VITE_API_URL + '/create-checkout-session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      userId,
      priceId,
      successUrl,
      cancelUrl,
    }),
  });

  const { sessionId } = await response.json();
  return sessionId;
}

export async function redirectToCheckout(sessionId: string) {
  const stripe = await getStripe();
  if (!stripe) throw new Error('Stripe failed to load');

  const { error } = await stripe.redirectToCheckout({ sessionId });
  if (error) throw error;
}

export async function getCustomerPortalUrl(userId: string): Promise<string> {
  const response = await fetch(import.meta.env.VITE_API_URL + '/customer-portal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId }),
  });

  const { url } = await response.json();
  return url;
}
```

### Subscription Hook

```typescript
// src/hooks/useSubscription.ts
import { useEffect, useState } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '@/lib/firebase';
import { useAuth } from '@/contexts/AuthContext';

export type SubscriptionTier = 'free' | 'pro';
export type SubscriptionStatus = 'active' | 'canceled' | 'past_due' | 'trialing';

interface Subscription {
  tier: SubscriptionTier;
  status: SubscriptionStatus | null;
  currentPeriodEnd: Date | null;
  cancelAtPeriodEnd: boolean;
}

export function useSubscription() {
  const { user } = useAuth();
  const [subscription, setSubscription] = useState<Subscription>({
    tier: 'free',
    status: null,
    currentPeriodEnd: null,
    cancelAtPeriodEnd: false,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      setSubscription({
        tier: 'free',
        status: null,
        currentPeriodEnd: null,
        cancelAtPeriodEnd: false,
      });
      setLoading(false);
      return;
    }

    // Listen to Firestore subscription document
    const unsubscribe = onSnapshot(
      doc(db, 'subscriptions', user.uid),
      (doc) => {
        if (doc.exists()) {
          const data = doc.data();
          setSubscription({
            tier: data.tier || 'free',
            status: data.status || null,
            currentPeriodEnd: data.currentPeriodEnd?.toDate() || null,
            cancelAtPeriodEnd: data.cancelAtPeriodEnd || false,
          });
        } else {
          setSubscription({
            tier: 'free',
            status: null,
            currentPeriodEnd: null,
            cancelAtPeriodEnd: false,
          });
        }
        setLoading(false);
      },
      (error) => {
        console.error('Subscription error:', error);
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, [user]);

  const isPro = subscription.tier === 'pro' && subscription.status === 'active';

  return {
    ...subscription,
    isPro,
    loading,
  };
}
```

### Pricing Page

```tsx
// src/pages/Pricing.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useSubscription } from '@/hooks/useSubscription';
import { createCheckoutSession, redirectToCheckout, STRIPE_PRICES } from '@/lib/stripe';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Check, Sparkles, Zap } from 'lucide-react';

const FEATURES = {
  free: [
    'Phase 1: The Archivist',
    'Up to 3 data sources',
    'Local-only storage',
    'Hybrid search (semantic + graph)',
    'Personal Knowledge Graph',
    'Interactive PKG visualization',
    'Entity & relationship extraction',
    'Privacy controls & audit logs',
  ],
  pro: [
    'Everything in Free, plus:',
    'Unlimited data sources',
    'Encrypted cloud backup',
    '─── Phase 2: The Analyst ───',
    'Emergent Insights Engine',
    'Curiosity Engine',
    'Proactive notifications',
    'Intelligent ranking',
    '─── Phase 3: The Guide ───',
    'Conversational causal exploration',
    'Causal Inference Engine',
    'Aspirational Self goals',
    'Reward Signal Dashboard',
    'Priority support',
  ],
};

export default function Pricing() {
  const [selectedPlan, setSelectedPlan] = useState<'free' | 'pro'>('free');
  const [isLoading, setIsLoading] = useState(false);
  const { user } = useAuth();
  const { isPro } = useSubscription();
  const navigate = useNavigate();

  const handleContinue = async () => {
    if (selectedPlan === 'free') {
      navigate('/onboarding');
      return;
    }

    if (!user) return;

    setIsLoading(true);
    try {
      const sessionId = await createCheckoutSession(
        user.uid,
        STRIPE_PRICES.pro_monthly,
        `${window.location.origin}/onboarding?success=true`,
        `${window.location.origin}/pricing?canceled=true`
      );
      await redirectToCheckout(sessionId);
    } catch (error) {
      console.error('Checkout error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isPro) {
    navigate('/dashboard');
    return null;
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background-deep p-4">
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/3 left-1/3 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/3 right-1/3 w-[500px] h-[500px] bg-accent/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 w-full max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-text-primary mb-2">
            Choose Your Path
          </h1>
          <p className="text-text-secondary">
            Start free or unlock the full potential of your knowledge
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Free Tier */}
          <Card
            className={`relative cursor-pointer transition-all duration-200 ${
              selectedPlan === 'free'
                ? 'border-primary ring-2 ring-primary/20'
                : 'hover:border-border-hover'
            }`}
            onClick={() => setSelectedPlan('free')}
          >
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">The Archivist</CardTitle>
                <Badge variant="secondary">Free</Badge>
              </div>
              <CardDescription>For personal exploration</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-6">
                <span className="text-4xl font-bold text-text-primary">$0</span>
                <span className="text-text-secondary">/month</span>
              </div>
              <ul className="space-y-3">
                {FEATURES.free.map((feature) => (
                  <li key={feature} className="flex items-center gap-2 text-sm text-text-secondary">
                    <Check className="h-4 w-4 text-secondary" />
                    {feature}
                  </li>
                ))}
              </ul>
            </CardContent>
            {selectedPlan === 'free' && (
              <div className="absolute top-4 right-4">
                <div className="h-5 w-5 rounded-full bg-primary flex items-center justify-center">
                  <Check className="h-3 w-3 text-white" />
                </div>
              </div>
            )}
          </Card>

          {/* Pro Tier */}
          <Card
            className={`relative cursor-pointer transition-all duration-200 ${
              selectedPlan === 'pro'
                ? 'border-accent ring-2 ring-accent/20'
                : 'hover:border-border-hover'
            }`}
            onClick={() => setSelectedPlan('pro')}
          >
            <div className="absolute -top-3 left-1/2 -translate-x-1/2">
              <Badge className="bg-accent text-white border-0">
                <Sparkles className="h-3 w-3 mr-1" />
                Recommended
              </Badge>
            </div>
            <CardHeader className="pt-8">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">Futurnal Pro</CardTitle>
                <Badge variant="accent">Pro</Badge>
              </div>
              <CardDescription>For serious knowledge architects</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-6">
                <span className="text-4xl font-bold text-text-primary">$19</span>
                <span className="text-text-secondary">/month</span>
              </div>
              <ul className="space-y-3">
                {FEATURES.pro.map((feature) => (
                  <li key={feature} className="flex items-center gap-2 text-sm text-text-secondary">
                    <Zap className="h-4 w-4 text-accent" />
                    {feature}
                  </li>
                ))}
              </ul>
            </CardContent>
            {selectedPlan === 'pro' && (
              <div className="absolute top-4 right-4">
                <div className="h-5 w-5 rounded-full bg-accent flex items-center justify-center">
                  <Check className="h-3 w-3 text-white" />
                </div>
              </div>
            )}
          </Card>
        </div>

        <div className="mt-8 text-center">
          <Button
            size="lg"
            onClick={handleContinue}
            disabled={isLoading}
            className={selectedPlan === 'pro' ? 'bg-accent hover:bg-accent/90' : ''}
          >
            {isLoading
              ? 'Processing...'
              : selectedPlan === 'free'
              ? 'Continue with Free'
              : 'Upgrade to Pro'}
          </Button>
        </div>
      </div>
    </div>
  );
}
```

### Tier Enforcement Hook

```typescript
// src/hooks/useTierLimits.ts
import { useSubscription } from './useSubscription';

interface TierLimits {
  maxDataSources: number;
  hasCloudBackup: boolean;
  // Phase 2: The Analyst features
  hasEmergentInsights: boolean;
  hasCuriosityEngine: boolean;
  hasProactiveNotifications: boolean;
  hasIntelligentRanking: boolean;
  // Phase 3: The Guide features
  hasCausalExploration: boolean;
  hasCausalInferenceEngine: boolean;
  hasAspirationalSelf: boolean;
  hasRewardDashboard: boolean;
}

const TIER_LIMITS: Record<'free' | 'pro', TierLimits> = {
  free: {
    maxDataSources: 3,
    hasCloudBackup: false,
    // Phase 2 locked
    hasEmergentInsights: false,
    hasCuriosityEngine: false,
    hasProactiveNotifications: false,
    hasIntelligentRanking: false,
    // Phase 3 locked
    hasCausalExploration: false,
    hasCausalInferenceEngine: false,
    hasAspirationalSelf: false,
    hasRewardDashboard: false,
  },
  pro: {
    maxDataSources: Infinity,
    hasCloudBackup: true,
    // Phase 2: The Analyst
    hasEmergentInsights: true,
    hasCuriosityEngine: true,
    hasProactiveNotifications: true,
    hasIntelligentRanking: true,
    // Phase 3: The Guide
    hasCausalExploration: true,
    hasCausalInferenceEngine: true,
    hasAspirationalSelf: true,
    hasRewardDashboard: true,
  },
};

export function useTierLimits() {
  const { tier, isPro } = useSubscription();
  const limits = TIER_LIMITS[tier];

  const canAddDataSource = (currentCount: number): boolean => {
    return currentCount < limits.maxDataSources;
  };

  const getUpgradeReason = (currentCount: number): string | null => {
    if (!canAddDataSource(currentCount)) {
      return `Free tier is limited to ${limits.maxDataSources} data sources. Upgrade to Pro for unlimited sources.`;
    }
    return null;
  };

  return {
    limits,
    isPro,
    canAddDataSource,
    getUpgradeReason,
  };
}
```

### Upgrade Prompt Component

```tsx
// src/components/pricing/UpgradePrompt.tsx
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Sparkles, X } from 'lucide-react';

interface UpgradePromptProps {
  reason: string;
  onClose: () => void;
}

export function UpgradePrompt({ reason, onClose }: UpgradePromptProps) {
  const navigate = useNavigate();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <Card className="w-full max-w-md relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-text-secondary hover:text-text-primary"
        >
          <X className="h-4 w-4" />
        </button>
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-accent/20 flex items-center justify-center">
            <Sparkles className="h-6 w-6 text-accent" />
          </div>
          <CardTitle>Upgrade to Pro</CardTitle>
          <CardDescription>{reason}</CardDescription>
        </CardHeader>
        <CardContent className="flex gap-3 justify-center">
          <Button variant="outline" onClick={onClose}>
            Maybe later
          </Button>
          <Button
            className="bg-accent hover:bg-accent/90"
            onClick={() => navigate('/pricing')}
          >
            View Plans
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
```

## Firebase Cloud Functions (Backend)

```typescript
// functions/src/stripe.ts
import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import Stripe from 'stripe';

const stripe = new Stripe(functions.config().stripe.secret_key, {
  apiVersion: '2024-06-20',
});

export const createCheckoutSession = functions.https.onRequest(async (req, res) => {
  const { userId, priceId, successUrl, cancelUrl } = req.body;

  const session = await stripe.checkout.sessions.create({
    mode: 'subscription',
    payment_method_types: ['card'],
    line_items: [{ price: priceId, quantity: 1 }],
    success_url: successUrl,
    cancel_url: cancelUrl,
    client_reference_id: userId,
    metadata: { userId },
  });

  res.json({ sessionId: session.id });
});

export const stripeWebhook = functions.https.onRequest(async (req, res) => {
  const sig = req.headers['stripe-signature'] as string;
  const endpointSecret = functions.config().stripe.webhook_secret;

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(req.rawBody, sig, endpointSecret);
  } catch (err) {
    res.status(400).send(`Webhook Error: ${err}`);
    return;
  }

  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object as Stripe.Checkout.Session;
      const userId = session.metadata?.userId;
      if (userId) {
        await admin.firestore().collection('subscriptions').doc(userId).set({
          tier: 'pro',
          status: 'active',
          stripeCustomerId: session.customer,
          stripeSubscriptionId: session.subscription,
          currentPeriodEnd: new Date(),
          cancelAtPeriodEnd: false,
        });
      }
      break;
    }
    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      // Handle subscription updates
      break;
    }
  }

  res.json({ received: true });
});
```

## Acceptance Criteria

- [ ] Pricing page displays Free and Pro tiers
- [ ] Free tier selection proceeds to onboarding
- [ ] Pro tier selection redirects to Stripe Checkout
- [ ] Successful payment updates subscription in Firestore
- [ ] Subscription status reflects in useSubscription hook
- [ ] Tier limits enforced (3 sources for Free)
- [ ] Upgrade prompts appear when hitting limits
- [ ] Customer portal link works for subscription management
- [ ] Webhook handles subscription lifecycle events

## Test Plan

### Unit Tests
```typescript
describe('useTierLimits', () => {
  it('should allow adding source within limit', () => {
    // Mock free tier with 2 sources
    expect(canAddDataSource(2)).toBe(true);
  });

  it('should block adding source at limit', () => {
    // Mock free tier with 3 sources
    expect(canAddDataSource(3)).toBe(false);
  });
});
```

### E2E Tests
```typescript
test('upgrade flow', async ({ page }) => {
  await page.goto('/pricing');
  await page.click('[data-tier="pro"]');
  await page.click('button:has-text("Upgrade")');
  // Verify Stripe redirect
  await expect(page).toHaveURL(/checkout\.stripe\.com/);
});
```

## Dependencies

- @stripe/stripe-js
- Firebase Cloud Functions (for backend)
- Firestore (for subscription storage)

## Next Steps

After pricing complete:
1. Proceed to Module 05 (Command Palette & Search)
2. Integrate tier limits with connector management
3. Add usage analytics

**This module enables monetization through Pro subscriptions with clear tier differentiation.**

## Complete Tier Comparison

| Feature | Free (Archivist) | Pro (Analyst + Guide) |
|---------|:----------------:|:---------------------:|
| **Phase 1: The Archivist** | | |
| Data source connectors | ✓ (up to 3) | ✓ (unlimited) |
| Local Files, Obsidian, IMAP, GitHub | ✓ | ✓ |
| Personal Knowledge Graph | ✓ | ✓ |
| Hybrid search (semantic + graph) | ✓ | ✓ |
| Interactive PKG visualization | ✓ | ✓ |
| Entity & relationship extraction | ✓ | ✓ |
| Privacy controls & audit logging | ✓ | ✓ |
| Local-only processing | ✓ | ✓ |
| **Phase 2: The Analyst** | | |
| Emergent Insights Engine | ✗ | ✓ |
| Curiosity Engine | ✗ | ✓ |
| Proactive notifications | ✗ | ✓ |
| Intelligent ranking | ✗ | ✓ |
| **Phase 3: The Guide** | | |
| Conversational causal exploration | ✗ | ✓ |
| Causal Inference Engine | ✗ | ✓ |
| Aspirational Self goals | ✗ | ✓ |
| Reward Signal Dashboard | ✗ | ✓ |
| **Additional Pro Benefits** | | |
| Encrypted cloud backup | ✗ | ✓ |
| Cloud escalation for reasoning | ✗ | ✓ |
| Priority support | ✗ | ✓ |

## Value Proposition by Tier

### Free: "The Archivist"
> *"Your Ghost: The most powerful and private personal search intelligence."*

Experience superior personalized search that proves Futurnal's core value. Perfect for personal exploration and proving the concept before committing.

### Pro: "Futurnal Pro"
> *"Your evolving AI: Discovering patterns in your experience you never knew existed."*
> *"Your Animal companion: True AI consciousness evolved from your stream of experience."*

Unlock the full evolution from Ghost to Animal—proactive insights, causal understanding, and goal-aligned guidance that transforms your data into genuine self-knowledge.

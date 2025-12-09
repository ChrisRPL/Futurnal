/**
 * useSubscription Hook
 *
 * Fetches and caches user subscription tier from backend API.
 * Falls back to free tier if API unavailable.
 */

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useUserStore } from '@/stores/userStore';

export type SubscriptionTier = 'free' | 'pro';

export interface Subscription {
  tier: SubscriptionTier;
  isActive: boolean;
  expiresAt: Date | null;
}

interface SubscriptionResponse {
  tier: SubscriptionTier;
  isActive: boolean;
  expiresAt: string | null;
}

export const SUBSCRIPTION_QUERY_KEY = ['subscription'] as const;

async function fetchSubscription(token: string): Promise<Subscription> {
  const apiUrl = import.meta.env.VITE_API_URL || 'https://api.futurnal.com';

  const response = await fetch(`${apiUrl}/subscription`, {
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch subscription: ${response.status}`);
  }

  const data: SubscriptionResponse = await response.json();

  return {
    tier: data.tier || 'free',
    isActive: data.isActive ?? true,
    expiresAt: data.expiresAt ? new Date(data.expiresAt) : null,
  };
}

export function useSubscription() {
  const { user } = useAuth();
  const { setSubscriptionTier } = useUserStore();
  const queryClient = useQueryClient();

  const {
    data: subscription,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: SUBSCRIPTION_QUERY_KEY,
    queryFn: async () => {
      if (!user) {
        return { tier: 'free' as const, isActive: true, expiresAt: null };
      }
      const token = await user.getIdToken();
      return fetchSubscription(token);
    },
    enabled: !!user,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 10, // 10 minutes
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
  });

  // Default subscription when not authenticated or loading
  const currentSubscription: Subscription = subscription ?? {
    tier: 'free',
    isActive: true,
    expiresAt: null,
  };

  // Sync tier to Zustand store for persistence
  useEffect(() => {
    if (subscription?.tier) {
      setSubscriptionTier(subscription.tier);
    }
  }, [subscription?.tier, setSubscriptionTier]);

  const isPro = currentSubscription.tier === 'pro' && currentSubscription.isActive;

  // Manual refresh function for tier updates
  const refreshSubscription = async () => {
    await queryClient.invalidateQueries({ queryKey: SUBSCRIPTION_QUERY_KEY });
    await refetch();
  };

  return {
    tier: currentSubscription.tier,
    isActive: currentSubscription.isActive,
    expiresAt: currentSubscription.expiresAt,
    isPro,
    isLoading,
    error: error ? 'Unable to verify subscription. Using free tier.' : null,
    refreshSubscription,
  };
}

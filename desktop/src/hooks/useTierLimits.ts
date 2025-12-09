/**
 * useTierLimits Hook
 *
 * Provides tier-based feature limits and gating utilities.
 * Used throughout the app to check if features are available.
 */

import { useSubscription, type SubscriptionTier } from './useSubscription';
import { TIER_LIMITS, type UserTierLimits } from '@/types/api';

interface TierLimitsReturn {
  limits: UserTierLimits;
  tier: SubscriptionTier;
  isPro: boolean;
  isLoading: boolean;
  /** Check if user can add another data source */
  canAddDataSource: (currentCount: number) => boolean;
  /** Check if a specific feature is available */
  hasFeature: (feature: keyof Omit<UserTierLimits, 'maxSources'>) => boolean;
  /** Get upgrade message for a feature */
  getUpgradeReason: (featureName: string) => string;
}

export function useTierLimits(): TierLimitsReturn {
  const { tier, isPro, isLoading } = useSubscription();

  const limits = TIER_LIMITS[tier];

  const canAddDataSource = (currentCount: number): boolean => {
    return currentCount < limits.maxSources;
  };

  const hasFeature = (feature: keyof Omit<UserTierLimits, 'maxSources'>): boolean => {
    return limits[feature] === true;
  };

  const getUpgradeReason = (featureName: string): string => {
    return `${featureName} is available with Futurnal Pro. Upgrade to unlock unlimited data sources, cloud backup, and advanced insights.`;
  };

  return {
    limits,
    tier,
    isPro,
    isLoading,
    canAddDataSource,
    hasFeature,
    getUpgradeReason,
  };
}

/**
 * User Store - Zustand state management for user data
 *
 * Manages Firebase user state and subscription tier.
 * Uses persist middleware for local storage persistence.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from 'firebase/auth';

interface UserState {
  /** Firebase user object (not persisted - contains methods) */
  firebaseUser: User | null;
  /** User's subscription tier */
  subscriptionTier: 'free' | 'pro';
  /** Set Firebase user */
  setFirebaseUser: (user: User | null) => void;
  /** Set subscription tier */
  setSubscriptionTier: (tier: 'free' | 'pro') => void;
  /** Reset store to initial state */
  reset: () => void;
}

const initialState = {
  firebaseUser: null,
  subscriptionTier: 'free' as const,
};

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      ...initialState,
      setFirebaseUser: (user) => set({ firebaseUser: user }),
      setSubscriptionTier: (tier) => set({ subscriptionTier: tier }),
      reset: () => set(initialState),
    }),
    {
      name: 'futurnal-user',
      // Only persist subscriptionTier, not firebaseUser (has methods, managed by Firebase)
      partialize: (state) => ({ subscriptionTier: state.subscriptionTier }),
    }
  )
);

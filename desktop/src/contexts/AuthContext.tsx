/**
 * Authentication Context for Futurnal Desktop Shell
 *
 * Provides authentication state and methods to the component tree.
 * Uses Tauri store for session persistence across app restarts.
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import { Store } from '@tauri-apps/plugin-store';
import {
  subscribeToAuthChanges,
  signInWithEmail as firebaseSignInWithEmail,
  signUpWithEmail as firebaseSignUpWithEmail,
  logout as firebaseLogout,
  resetPassword as firebaseResetPassword,
  type User,
} from '@/lib/firebase';
import { useUserStore } from '@/stores/userStore';

interface AuthContextType {
  /** Current authenticated user */
  user: User | null;
  /** Whether auth state is being determined */
  loading: boolean;
  /** Current error message */
  error: string | null;
  /** Sign in with email and password */
  signInWithEmail: (email: string, password: string) => Promise<void>;
  /** Create account with email and password */
  signUpWithEmail: (email: string, password: string) => Promise<void>;
  /** Sign out current user */
  logout: () => Promise<void>;
  /** Send password reset email */
  resetPassword: (email: string) => Promise<void>;
  /** Clear current error */
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Initialize Tauri store for auth persistence
let store: Store | null = null;

async function getStore(): Promise<Store> {
  if (!store) {
    store = await Store.load('auth.dat');
  }
  return store;
}

/**
 * Map Firebase error codes to user-friendly messages
 */
function getAuthErrorMessage(error: unknown): string {
  const code = (error as { code?: string })?.code;

  const messages: Record<string, string> = {
    'auth/user-not-found': 'No account found with this email.',
    'auth/wrong-password': 'Incorrect password. Please try again.',
    'auth/email-already-in-use': 'An account with this email already exists.',
    'auth/weak-password': 'Password should be at least 6 characters.',
    'auth/invalid-email': 'Please enter a valid email address.',
    'auth/invalid-credential': 'Invalid email or password. Please try again.',
    'auth/too-many-requests': 'Too many attempts. Please try again later.',
    'auth/network-request-failed': 'Network error. Please check your connection.',
    'auth/user-disabled': 'This account has been disabled.',
    'auth/operation-not-allowed': 'This sign-in method is not enabled.',
  };

  return messages[code || ''] || 'An unexpected error occurred. Please try again.';
}

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setFirebaseUser, reset: resetUserStore } = useUserStore();

  // Subscribe to Firebase auth state changes
  useEffect(() => {
    const unsubscribe = subscribeToAuthChanges(async (firebaseUser) => {
      setUser(firebaseUser);
      setFirebaseUser(firebaseUser);
      setLoading(false);

      // Persist user ID to Tauri store
      try {
        const authStore = await getStore();
        if (firebaseUser) {
          await authStore.set('userId', firebaseUser.uid);
          await authStore.set('email', firebaseUser.email);
          await authStore.save();
        } else {
          await authStore.delete('userId');
          await authStore.delete('email');
          await authStore.save();
        }
      } catch (err) {
        console.error('Failed to persist auth state:', err);
      }
    });

    return () => unsubscribe();
  }, [setFirebaseUser]);

  const handleAuthError = (err: unknown) => {
    const message = getAuthErrorMessage(err);
    setError(message);
    throw err;
  };

  const value: AuthContextType = {
    user,
    loading,
    error,
    clearError: () => setError(null),

    signInWithEmail: async (email: string, password: string) => {
      try {
        setError(null);
        await firebaseSignInWithEmail(email, password);
      } catch (err) {
        handleAuthError(err);
      }
    },

    signUpWithEmail: async (email: string, password: string) => {
      try {
        setError(null);
        await firebaseSignUpWithEmail(email, password);
      } catch (err) {
        handleAuthError(err);
      }
    },

    logout: async () => {
      try {
        setError(null);
        await firebaseLogout();
        resetUserStore();

        // Clear Tauri store
        try {
          const authStore = await getStore();
          await authStore.clear();
          await authStore.save();
        } catch (err) {
          console.error('Failed to clear auth store:', err);
        }
      } catch (err) {
        handleAuthError(err);
      }
    },

    resetPassword: async (email: string) => {
      try {
        setError(null);
        await firebaseResetPassword(email);
      } catch (err) {
        handleAuthError(err);
      }
    },
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Hook to access authentication context
 * @throws Error if used outside AuthProvider
 */
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

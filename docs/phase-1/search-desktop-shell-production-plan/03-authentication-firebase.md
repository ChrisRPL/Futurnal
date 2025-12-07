Summary: Implement Firebase Authentication with GitHub, Google, and Email/Password providers for the desktop app.

# 03 · Authentication (Firebase)

## Purpose

Implement user authentication using Firebase Authentication, enabling users to begin their Ghost→Animal evolution journey. This module creates the entry point where users establish their identity—the first step before they can ground their Ghost in their personal universe.

> *"Your data remains yours. We only authenticate your identity."*

Supporting GitHub, Google, and Email/Password sign-in methods with glassmorphism design, protected routes, and session persistence within the Tauri desktop environment.

**Criticality**: CRITICAL - Required for user identification and tier enforcement

## Scope

- Firebase project setup and SDK integration
- Auth providers: GitHub OAuth, Google OAuth, Email/Password
- Login and Signup pages with glassmorphism design
- Auth state persistence using Tauri store
- Protected route wrapper component
- User session management (currentUser, loading states)
- Logout functionality
- Error handling with user-friendly messages
- Password reset flow

## Requirements Alignment

- **Feature Requirement**: "Authentication (Sign Up / Login)" from FRONTEND_DESIGN.md
- **Privacy Note**: "Your data remains yours. We only authenticate your identity."
- **Security**: No Node integration, secure token storage

## Component Design

### Firebase Configuration

```typescript
// src/lib/firebase.ts
import { initializeApp } from 'firebase/app';
import {
  getAuth,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  GoogleAuthProvider,
  GithubAuthProvider,
  onAuthStateChanged,
  type User,
} from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);

// Providers
const googleProvider = new GoogleAuthProvider();
const githubProvider = new GithubAuthProvider();

// Auth functions
export async function signInWithGoogle() {
  return signInWithPopup(auth, googleProvider);
}

export async function signInWithGithub() {
  return signInWithPopup(auth, githubProvider);
}

export async function signInWithEmail(email: string, password: string) {
  return signInWithEmailAndPassword(auth, email, password);
}

export async function signUpWithEmail(email: string, password: string) {
  return createUserWithEmailAndPassword(auth, email, password);
}

export async function logout() {
  return signOut(auth);
}

export async function resetPassword(email: string) {
  return sendPasswordResetEmail(auth, email);
}

export function subscribeToAuthChanges(callback: (user: User | null) => void) {
  return onAuthStateChanged(auth, callback);
}

export type { User };
```

### Auth Context

```tsx
// src/contexts/AuthContext.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import { Store } from '@tauri-apps/plugin-store';
import {
  auth,
  subscribeToAuthChanges,
  signInWithGoogle,
  signInWithGithub,
  signInWithEmail,
  signUpWithEmail,
  logout as firebaseLogout,
  resetPassword,
  type User,
} from '@/lib/firebase';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  error: string | null;
  signInWithGoogle: () => Promise<void>;
  signInWithGithub: () => Promise<void>;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signUpWithEmail: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Tauri store for session persistence
const store = new Store('auth.dat');

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = subscribeToAuthChanges(async (firebaseUser) => {
      setUser(firebaseUser);
      setLoading(false);

      // Persist user ID to Tauri store
      if (firebaseUser) {
        await store.set('userId', firebaseUser.uid);
        await store.save();
      } else {
        await store.delete('userId');
        await store.save();
      }
    });

    return () => unsubscribe();
  }, []);

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

    signInWithGoogle: async () => {
      try {
        setError(null);
        await signInWithGoogle();
      } catch (err) {
        handleAuthError(err);
      }
    },

    signInWithGithub: async () => {
      try {
        setError(null);
        await signInWithGithub();
      } catch (err) {
        handleAuthError(err);
      }
    },

    signInWithEmail: async (email: string, password: string) => {
      try {
        setError(null);
        await signInWithEmail(email, password);
      } catch (err) {
        handleAuthError(err);
      }
    },

    signUpWithEmail: async (email: string, password: string) => {
      try {
        setError(null);
        await signUpWithEmail(email, password);
      } catch (err) {
        handleAuthError(err);
      }
    },

    logout: async () => {
      try {
        setError(null);
        await firebaseLogout();
      } catch (err) {
        handleAuthError(err);
      }
    },

    resetPassword: async (email: string) => {
      try {
        setError(null);
        await resetPassword(email);
      } catch (err) {
        handleAuthError(err);
      }
    },
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// User-friendly error messages
function getAuthErrorMessage(error: unknown): string {
  const code = (error as { code?: string })?.code;

  const messages: Record<string, string> = {
    'auth/user-not-found': 'No account found with this email.',
    'auth/wrong-password': 'Incorrect password. Please try again.',
    'auth/email-already-in-use': 'An account with this email already exists.',
    'auth/weak-password': 'Password should be at least 6 characters.',
    'auth/invalid-email': 'Please enter a valid email address.',
    'auth/too-many-requests': 'Too many attempts. Please try again later.',
    'auth/popup-closed-by-user': 'Sign-in popup was closed.',
    'auth/cancelled-popup-request': 'Only one popup request allowed at a time.',
    'auth/popup-blocked': 'Sign-in popup was blocked. Please allow popups.',
  };

  return messages[code || ''] || 'An unexpected error occurred. Please try again.';
}
```

### Login Page

```tsx
// src/pages/Login.tsx
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Github, Mail } from 'lucide-react';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { signInWithGoogle, signInWithGithub, signInWithEmail, error, clearError } = useAuth();
  const navigate = useNavigate();

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await signInWithEmail(email, password);
      navigate('/pricing');
    } catch {
      // Error handled by context
    } finally {
      setIsLoading(false);
    }
  };

  const handleSocialSignIn = async (provider: 'google' | 'github') => {
    setIsLoading(true);
    try {
      if (provider === 'google') {
        await signInWithGoogle();
      } else {
        await signInWithGithub();
      }
      navigate('/pricing');
    } catch {
      // Error handled by context
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background-deep p-4">
      {/* Glassmorphism background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />
      </div>

      <Card className="w-full max-w-md relative backdrop-blur-xl bg-background-surface/80 border-border/50">
        <CardHeader className="text-center">
          <img src="/logo.png" alt="Futurnal" className="w-16 h-16 mx-auto mb-4" />
          <CardTitle className="text-2xl">Your Ghost Awaits</CardTitle>
          <CardDescription>Sign in to continue your journey toward self-knowledge</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Social Sign-In */}
          <div className="grid grid-cols-2 gap-3">
            <Button
              variant="outline"
              onClick={() => handleSocialSignIn('github')}
              disabled={isLoading}
              className="w-full"
            >
              <Github className="mr-2 h-4 w-4" />
              GitHub
            </Button>
            <Button
              variant="outline"
              onClick={() => handleSocialSignIn('google')}
              disabled={isLoading}
              className="w-full"
            >
              <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                {/* Google icon SVG */}
              </svg>
              Google
            </Button>
          </div>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background-surface px-2 text-text-tertiary">
                Or continue with email
              </span>
            </div>
          </div>

          {/* Email Sign-In */}
          <form onSubmit={handleEmailSignIn} className="space-y-4">
            <div className="space-y-2">
              <Input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  clearError();
                }}
                required
              />
            </div>
            <div className="space-y-2">
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  clearError();
                }}
                required
              />
            </div>

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? 'Signing in...' : 'Sign in'}
            </Button>
          </form>

          <div className="text-center text-sm">
            <Link
              to="/forgot-password"
              className="text-text-secondary hover:text-primary transition-colors"
            >
              Forgot password?
            </Link>
          </div>

          <div className="text-center text-sm text-text-secondary">
            Don't have an account?{' '}
            <Link to="/signup" className="text-primary hover:underline">
              Sign up
            </Link>
          </div>

          <p className="text-xs text-center text-text-tertiary">
            Your data remains yours. We only authenticate your identity.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
```

### Protected Route Component

```tsx
// src/components/auth/ProtectedRoute.tsx
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background-deep">
        <div className="animate-pulse-subtle">
          <img src="/logo.png" alt="Loading" className="w-16 h-16" />
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}
```

### User Store Integration

```typescript
// src/stores/userStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from 'firebase/auth';

interface UserState {
  firebaseUser: User | null;
  subscriptionTier: 'free' | 'pro';
  setFirebaseUser: (user: User | null) => void;
  setSubscriptionTier: (tier: 'free' | 'pro') => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      firebaseUser: null,
      subscriptionTier: 'free',
      setFirebaseUser: (user) => set({ firebaseUser: user }),
      setSubscriptionTier: (tier) => set({ subscriptionTier: tier }),
    }),
    {
      name: 'futurnal-user',
      partialize: (state) => ({ subscriptionTier: state.subscriptionTier }),
    }
  )
);
```

## Acceptance Criteria

- [ ] Firebase SDK initializes without errors
- [ ] GitHub OAuth sign-in works
- [ ] Google OAuth sign-in works
- [ ] Email/Password sign-in works
- [ ] Email/Password sign-up works
- [ ] Password reset email sends
- [ ] Auth state persists across app restarts
- [ ] Protected routes redirect unauthenticated users
- [ ] Error messages are user-friendly
- [ ] Logout clears all session data
- [ ] Loading states display during auth operations

## Test Plan

### Unit Tests
```typescript
describe('Auth Context', () => {
  it('should initialize with null user', () => {
    // Test initial state
  });

  it('should update user on sign in', () => {
    // Mock Firebase and test state update
  });

  it('should clear user on logout', () => {
    // Test logout clears state
  });
});
```

### E2E Tests
```typescript
test('email sign in flow', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[type="email"]', 'test@example.com');
  await page.fill('[type="password"]', 'password123');
  await page.click('button[type="submit"]');
  await expect(page).toHaveURL('/pricing');
});
```

## Dependencies

- Firebase SDK (firebase)
- @tauri-apps/plugin-store
- React Router DOM

## Security Considerations

- Firebase API keys are safe to expose (domain-restricted)
- OAuth tokens never stored in plain text
- Session tokens managed by Firebase
- CSP configured to allow Firebase domains

## Next Steps

After authentication complete:
1. Proceed to Module 04 (Pricing Tier & Stripe)
2. Integrate user state with subscription tier
3. Add profile management features

---

## Welcome Screen (Pre-Auth)

The Welcome Screen is the user's first encounter with Futurnal—an atmospheric introduction to their Ghost:

```tsx
// src/pages/Welcome.tsx
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Ghost } from 'lucide-react';

export default function Welcome() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background-deep p-4 relative overflow-hidden">
      {/* Atmospheric background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[100px] animate-pulse-subtle" />
      </div>

      {/* Content */}
      <div className="relative z-10 text-center space-y-8 max-w-lg">
        <div className="flex justify-center">
          <Ghost className="w-24 h-24 text-primary animate-float" />
        </div>

        <div className="space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-text-primary">
            Your Ghost Awaits
          </h1>
          <p className="text-lg text-text-secondary">
            Every journey toward self-knowledge begins with a single step.
            Your Ghost is ready to learn your personal universe.
          </p>
        </div>

        <div className="space-y-4">
          <Button
            size="lg"
            className="w-full max-w-xs text-lg"
            onClick={() => navigate('/login')}
          >
            Begin Your Journey
          </Button>

          <p className="text-sm text-text-tertiary">
            "Know Yourself More"
          </p>
        </div>
      </div>

      {/* Footer */}
      <div className="absolute bottom-8 text-center">
        <p className="text-xs text-text-tertiary">
          Privacy-first. Local-first. Your data never leaves your device.
        </p>
      </div>
    </div>
  );
}
```

### Welcome Screen UI Copy Reference

| Element | Copy | Purpose |
|---------|------|---------|
| Headline | "Your Ghost Awaits" | Introduces Ghost paradigm |
| Subhead | "Every journey toward self-knowledge begins with a single step. Your Ghost is ready to learn your personal universe." | Sets experiential framing |
| CTA | "Begin Your Journey" | Action-oriented, aspirational |
| Tagline | "Know Yourself More" | Core brand tagline |
| Footer | "Privacy-first. Local-first. Your data never leaves your device." | Sovereignty reinforcement |

---

**This authentication module enables user identification for tier enforcement and marks the beginning of the Ghost→Animal evolution journey.**

/**
 * Login Page - Email/Password authentication
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 */

import { useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { signInWithEmail, error, clearError } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await signInWithEmail(email, password);
      navigate(from, { replace: true });
    } catch {
      // Error handled by context
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header */}
      <header className="w-full px-8 py-6 flex items-center justify-between">
        <Link to="/">
          <img
            src="/logo_dark.png"
            alt="Futurnal"
            className="h-10 w-auto"
          />
        </Link>
        <Link
          to="/signup"
          className="text-sm text-white/60 hover:text-white transition-colors no-underline"
        >
          Create account
        </Link>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-8 py-12">
        <div className="w-full max-w-md animate-fade-in">
          {/* Title */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-brand tracking-wide text-white mb-3">
              Welcome Back
            </h1>
            <p className="text-white/60">
              Sign in to continue your journey
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm text-white/60 mb-2">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  clearError();
                }}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
                placeholder="your@email.com"
                required
                autoComplete="email"
                disabled={isLoading}
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="password" className="text-sm text-white/60">
                  Password
                </label>
                <Link
                  to="/forgot-password"
                  className="text-sm text-white/40 hover:text-white/60 transition-colors no-underline"
                >
                  Forgot?
                </Link>
              </div>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  clearError();
                }}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
                placeholder="Enter your password"
                required
                autoComplete="current-password"
                disabled={isLoading}
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="px-4 py-3 bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 bg-white text-black font-medium text-lg transition-all hover:bg-white/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Signing in...' : 'Sign In'}
            </button>
          </form>

          {/* Footer */}
          <p className="mt-8 text-center text-sm text-white/40">
            Don't have an account?{' '}
            <Link to="/signup" className="text-white/60 hover:text-white transition-colors">
              Create one
            </Link>
          </p>
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full px-8 py-6 text-center">
        <p className="text-xs text-white/30">
          Your data remains yours. We only authenticate your identity.
        </p>
      </footer>
    </div>
  );
}

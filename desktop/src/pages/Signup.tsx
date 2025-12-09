/**
 * Signup Page - New account registration
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 */

import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { signUpWithEmail, error, clearError } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (password !== confirmPassword) {
      setLocalError('Passwords do not match.');
      return;
    }

    if (password.length < 6) {
      setLocalError('Password must be at least 6 characters.');
      return;
    }

    setIsLoading(true);

    try {
      await signUpWithEmail(email, password);
      navigate('/dashboard', { replace: true });
    } catch {
      // Error handled by context
    } finally {
      setIsLoading(false);
    }
  };

  const displayError = localError || error;

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
          to="/login"
          className="text-sm text-white/60 hover:text-white transition-colors no-underline"
        >
          Sign in
        </Link>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-8 py-12">
        <div className="w-full max-w-md animate-fade-in">
          {/* Title */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-brand tracking-wide text-white mb-3">
              Create Account
            </h1>
            <p className="text-white/60">
              Begin your journey to self-knowledge
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
                  setLocalError(null);
                }}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
                placeholder="your@email.com"
                required
                autoComplete="email"
                disabled={isLoading}
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm text-white/60 mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  clearError();
                  setLocalError(null);
                }}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
                placeholder="At least 6 characters"
                required
                autoComplete="new-password"
                disabled={isLoading}
              />
            </div>

            <div>
              <label htmlFor="confirm-password" className="block text-sm text-white/60 mb-2">
                Confirm Password
              </label>
              <input
                id="confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => {
                  setConfirmPassword(e.target.value);
                  setLocalError(null);
                }}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30 transition-colors"
                placeholder="Confirm your password"
                required
                autoComplete="new-password"
                disabled={isLoading}
              />
            </div>

            {/* Error Message */}
            {displayError && (
              <div className="px-4 py-3 bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                {displayError}
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 bg-white text-black font-medium text-lg transition-all hover:bg-white/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Creating account...' : 'Create Account'}
            </button>
          </form>

          {/* Footer */}
          <p className="mt-8 text-center text-sm text-white/40">
            Already have an account?{' '}
            <Link to="/login" className="text-white/60 hover:text-white transition-colors">
              Sign in
            </Link>
          </p>

          {/* Privacy Note */}
          <p className="mt-6 text-center text-xs text-white/30">
            By creating an account, you agree to our privacy-first approach.
            Your data remains yours.
          </p>
        </div>
      </main>
    </div>
  );
}

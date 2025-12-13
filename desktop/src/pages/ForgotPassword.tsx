/**
 * Forgot Password Page - Password reset request
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 */

import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeLogo } from '@/components/ThemeLogo';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const { resetPassword, error, clearError } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await resetPassword(email);
      setIsSuccess(true);
    } catch {
      // Error handled by context
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] flex flex-col">
      {/* Header */}
      <header className="w-full px-8 py-6 flex items-center justify-between">
        <Link to="/">
          <ThemeLogo variant="small" className="h-10 w-auto" />
        </Link>
        <Link
          to="/login"
          className="text-sm text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors no-underline"
        >
          Back to sign in
        </Link>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-8 py-12">
        <div className="w-full max-w-md animate-fade-in">
          {isSuccess ? (
            // Success State
            <div className="text-center">
              <h1 className="text-3xl font-brand tracking-wide text-[var(--color-text-primary)] mb-4">
                Check Your Email
              </h1>
              <p className="text-[var(--color-text-tertiary)] mb-8">
                We've sent password reset instructions to{' '}
                <span className="text-[var(--color-text-primary)]">{email}</span>
              </p>
              <button
                onClick={() => navigate('/login')}
                className="px-8 py-3 bg-transparent text-[var(--color-text-primary)] border border-[var(--color-border-active)] font-medium transition-all hover:border-[var(--color-text-tertiary)]"
              >
                Back to Sign In
              </button>
            </div>
          ) : (
            // Form State
            <>
              <div className="text-center mb-10">
                <h1 className="text-3xl font-brand tracking-wide text-[var(--color-text-primary)] mb-3">
                  Reset Password
                </h1>
                <p className="text-[var(--color-text-tertiary)]">
                  Enter your email to receive reset instructions
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label htmlFor="email" className="block text-sm text-[var(--color-text-tertiary)] mb-2">
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
                    className="w-full px-4 py-3 bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-primary)] placeholder-[var(--color-text-faint)] focus:outline-none focus:border-[var(--color-border-active)] transition-colors"
                    placeholder="your@email.com"
                    required
                    autoComplete="email"
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
                  className="w-full py-4 bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)] font-medium text-lg transition-all hover:bg-[var(--color-inverse-bg-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Sending...' : 'Send Reset Link'}
                </button>
              </form>

              <p className="mt-8 text-center text-sm text-[var(--color-text-muted)]">
                Remember your password?{' '}
                <Link to="/login" className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors">
                  Sign in
                </Link>
              </p>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

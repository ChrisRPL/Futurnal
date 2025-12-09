/**
 * Forgot Password Page - Password reset request
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 */

import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

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
          Back to sign in
        </Link>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-8 py-12">
        <div className="w-full max-w-md animate-fade-in">
          {isSuccess ? (
            // Success State
            <div className="text-center">
              <h1 className="text-3xl font-brand tracking-wide text-white mb-4">
                Check Your Email
              </h1>
              <p className="text-white/60 mb-8">
                We've sent password reset instructions to{' '}
                <span className="text-white">{email}</span>
              </p>
              <button
                onClick={() => navigate('/login')}
                className="px-8 py-3 bg-transparent text-white border border-white/30 font-medium transition-all hover:border-white/60"
              >
                Back to Sign In
              </button>
            </div>
          ) : (
            // Form State
            <>
              <div className="text-center mb-10">
                <h1 className="text-3xl font-brand tracking-wide text-white mb-3">
                  Reset Password
                </h1>
                <p className="text-white/60">
                  Enter your email to receive reset instructions
                </p>
              </div>

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
                  {isLoading ? 'Sending...' : 'Send Reset Link'}
                </button>
              </form>

              <p className="mt-8 text-center text-sm text-white/40">
                Remember your password?{' '}
                <Link to="/login" className="text-white/60 hover:text-white transition-colors">
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

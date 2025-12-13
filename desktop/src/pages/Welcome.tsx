/**
 * Welcome Page - Entry point for Futurnal
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 * Cinzel for headlines, Times New Roman for taglines, black & white aesthetic.
 */

import { useNavigate } from 'react-router-dom';
import { ThemeLogo } from '@/components/ThemeLogo';

export default function Welcome() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] flex flex-col">
      {/* Header */}
      <header className="w-full px-8 py-6">
        <ThemeLogo variant="small" className="h-10 w-auto" />
      </header>

      {/* Main Content - Centered */}
      <main className="flex-1 flex flex-col items-center justify-center px-8 pb-24">
        <div className="w-full max-w-3xl animate-fade-in">
          {/* Big Logo */}
          <div className="mb-16">
            <ThemeLogo
              variant="big"
              alt="Futurnal - Know Yourself More"
              className="h-32 md:h-48 w-auto mx-auto"
            />
          </div>

          {/* Value Proposition */}
          <div className="text-center mb-16 space-y-6">
            <p className="text-xl md:text-2xl text-[var(--color-text-secondary)] leading-relaxed max-w-2xl mx-auto">
              The world's first AI evolution platform that transforms generic AI
              into deeply personalized intelligence.
            </p>
            <p className="text-lg text-[var(--color-text-tertiary)] leading-relaxed max-w-xl mx-auto font-tagline italic">
              Your AI learns continuously from your unique personal data stream,
              developing genuine understanding of your patterns and growth.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate('/signup')}
              className="w-full sm:w-auto px-10 py-4 bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)] font-medium text-lg transition-all hover:bg-[var(--color-inverse-bg-hover)]"
            >
              Get Started
            </button>
            <button
              onClick={() => navigate('/login')}
              className="w-full sm:w-auto px-10 py-4 bg-transparent text-[var(--color-text-primary)] border border-[var(--color-border-active)] font-medium text-lg transition-all hover:border-[var(--color-text-tertiary)]"
            >
              Sign In
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full px-8 py-8 border-t border-[var(--color-border)]">
        <div className="max-w-3xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-[var(--color-text-muted)]">
          <p>Privacy-first. Local-first. Your data remains yours.</p>
          <p className="font-tagline italic">For developers, researchers, and knowledge workers.</p>
        </div>
      </footer>
    </div>
  );
}

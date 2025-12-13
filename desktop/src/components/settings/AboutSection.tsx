/**
 * About Section
 *
 * Displays app version, links, and keyboard shortcuts reference.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { ExternalLink, Keyboard, Heart, Info } from 'lucide-react';

const SHORTCUTS = [
  { keys: ['Cmd/Ctrl', 'K'], description: 'Open command palette' },
  { keys: ['Cmd/Ctrl', 'F'], description: 'Focus search' },
  { keys: ['Cmd/Ctrl', ','], description: 'Open settings' },
  { keys: ['Esc'], description: 'Close dialogs' },
  { keys: ['G', 'then', 'H'], description: 'Go to dashboard' },
  { keys: ['G', 'then', 'G'], description: 'Go to graph view' },
  { keys: ['G', 'then', 'C'], description: 'Go to connectors' },
];

export function AboutSection() {
  const appVersion = import.meta.env.VITE_APP_VERSION ?? '0.1.0';

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">About</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Version information and helpful links.
        </p>
      </div>

      {/* Version Info */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Info className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Futurnal</h3>
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-[var(--color-text-secondary)]">Version</span>
            <span className="text-[var(--color-text-primary)]">{appVersion}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[var(--color-text-secondary)]">Build</span>
            <span className="text-[var(--color-text-primary)]">Desktop (Tauri)</span>
          </div>
        </div>
        <p className="text-xs text-[var(--color-text-muted)] mt-4 font-tagline italic">
          "Know Yourself More"
        </p>
      </div>

      {/* Links */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <ExternalLink className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Resources</h3>
        </div>
        <div className="space-y-3">
          <a
            href="https://futurnal.com/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors group"
          >
            <span>Documentation</span>
            <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <a
            href="https://futurnal.com/privacy"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors group"
          >
            <span>Privacy Policy</span>
            <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <a
            href="https://futurnal.com/terms"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors group"
          >
            <span>Terms of Service</span>
            <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <a
            href="https://github.com/futurnal/futurnal"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors group"
          >
            <span>GitHub Repository</span>
            <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
        </div>
      </div>

      {/* Keyboard Shortcuts */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Keyboard className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Keyboard Shortcuts</h3>
        </div>
        <div className="space-y-2">
          {SHORTCUTS.map(({ keys, description }) => (
            <div key={description} className="flex items-center justify-between">
              <span className="text-sm text-[var(--color-text-secondary)]">{description}</span>
              <div className="flex items-center gap-1">
                {keys.map((key, idx) => (
                  <span key={idx}>
                    {key === 'then' ? (
                      <span className="text-[var(--color-text-muted)] text-xs mx-1">then</span>
                    ) : (
                      <kbd className="px-2 py-0.5 text-xs bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)]">
                        {key}
                      </kbd>
                    )}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Credits */}
      <div className="text-center py-4">
        <p className="text-xs text-[var(--color-text-muted)] flex items-center justify-center gap-1">
          Made with <Heart className="h-3 w-3 text-[var(--color-text-muted)]" /> by Futurnal
        </p>
      </div>
    </div>
  );
}

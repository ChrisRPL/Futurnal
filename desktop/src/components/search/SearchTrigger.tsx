/**
 * SearchTrigger - Dashboard button to open command palette
 *
 * A compact button that displays the keyboard shortcut hint
 * and triggers the search palette when clicked.
 */

import { Search, Command } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SearchTriggerProps {
  /** Handler when trigger is clicked */
  onClick: () => void;
  /** Additional class names */
  className?: string;
}

/**
 * Detect if user is on macOS for keyboard shortcut display
 */
function isMac(): boolean {
  if (typeof navigator === 'undefined') return false;
  return navigator.platform.toLowerCase().includes('mac');
}

export function SearchTrigger({ onClick, className }: SearchTriggerProps) {
  const shortcutKey = isMac() ? '⌘' : 'Ctrl';

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-64 flex items-center justify-between px-4 py-2',
        'bg-transparent border border-white/10',
        'text-white/50 hover:text-white/70 hover:border-white/20',
        'transition-all',
        className
      )}
    >
      <div className="flex items-center gap-2">
        <Search className="h-4 w-4" />
        <span className="text-sm">Search your knowledge...</span>
      </div>
      <kbd className="flex items-center gap-0.5 bg-white/5 px-1.5 py-0.5 text-xs text-white/40 rounded">
        {isMac() ? <Command className="h-3 w-3" /> : <span>{shortcutKey}+</span>}
        <span>K</span>
      </kbd>
    </button>
  );
}

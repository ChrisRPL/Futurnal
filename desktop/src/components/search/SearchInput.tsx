/**
 * SearchInput - Query input field for the command palette
 *
 * Simple, unstyled input that integrates with the CommandPalette header.
 * Uses monochrome styling with transparent background.
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

interface SearchInputProps {
  /** Current input value */
  value: string;
  /** Value change handler */
  onChange: (value: string) => void;
  /** Keyboard event handler */
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Auto-focus on mount */
  autoFocus?: boolean;
  /** Additional class names */
  className?: string;
}

export function SearchInput({
  value,
  onChange,
  onKeyDown,
  placeholder = 'Search your knowledge...',
  autoFocus = false,
  className,
}: SearchInputProps) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      autoFocus={autoFocus}
      data-slot="search-input"
      className={cn(
        'flex-1 bg-transparent text-white text-sm',
        'placeholder:text-white/50',
        'outline-none border-none',
        'min-w-0',
        className
      )}
    />
  );
}

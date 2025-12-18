/**
 * SlashCommandPopover - Command dropdown for chat input
 *
 * Phase B: Slash Commands in Chat
 *
 * Features:
 * - Shows available commands when "/" is typed
 * - Filters commands as user types
 * - Keyboard navigation (arrows, enter, escape)
 * - Category grouping
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { filterCommands, type SlashCommand } from '@/data/commands';

interface SlashCommandPopoverProps {
  /** Current input text after "/" */
  filter: string;
  /** Whether the popover is visible */
  isOpen: boolean;
  /** Called when a command is selected */
  onSelect: (command: SlashCommand) => void;
  /** Called when the popover should close */
  onClose: () => void;
  /** Position anchor */
  anchorRef?: React.RefObject<HTMLElement>;
  /** Additional class names */
  className?: string;
}

export function SlashCommandPopover({
  filter,
  isOpen,
  onSelect,
  onClose,
  className,
}: SlashCommandPopoverProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  // Filter commands based on input
  const filteredCommands = filterCommands(filter);

  // Reset selection when filter changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [filter]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && selectedIndex >= 0) {
      const items = listRef.current.querySelectorAll('[data-command-item]');
      const selectedItem = items[selectedIndex] as HTMLElement;
      if (selectedItem) {
        selectedItem.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [selectedIndex]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < filteredCommands.length - 1 ? prev + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev > 0 ? prev - 1 : filteredCommands.length - 1
          );
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            onSelect(filteredCommands[selectedIndex]);
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
        case 'Tab':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            onSelect(filteredCommands[selectedIndex]);
          }
          break;
      }
    },
    [isOpen, filteredCommands, selectedIndex, onSelect, onClose]
  );

  // Attach keyboard listener
  useEffect(() => {
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown, true);
      return () => window.removeEventListener('keydown', handleKeyDown, true);
    }
  }, [isOpen, handleKeyDown]);

  if (!isOpen || filteredCommands.length === 0) return null;

  // Group commands by category
  const groupedCommands = filteredCommands.reduce(
    (acc, cmd) => {
      if (!acc[cmd.category]) acc[cmd.category] = [];
      acc[cmd.category].push(cmd);
      return acc;
    },
    {} as Record<string, SlashCommand[]>
  );

  const categoryLabels: Record<string, string> = {
    navigation: 'Navigate',
    action: 'Actions',
    query: 'Query',
  };

  // Flatten for index tracking
  let flatIndex = -1;

  return (
    <div
      ref={listRef}
      className={cn(
        'absolute bottom-full left-0 right-0 mb-2 mx-4',
        'bg-black/95 backdrop-blur-xl border border-white/10 rounded-lg',
        'shadow-2xl overflow-hidden',
        'max-h-[300px] overflow-y-auto',
        'animate-fade-in',
        className
      )}
    >
      {Object.entries(groupedCommands).map(([category, commands]) => (
        <div key={category}>
          {/* Category header */}
          <div className="px-3 py-1.5 text-[10px] font-medium text-white/40 uppercase tracking-wider bg-white/5 sticky top-0">
            {categoryLabels[category] || category}
          </div>

          {/* Commands */}
          {commands.map((command) => {
            flatIndex++;
            const index = flatIndex;
            const isSelected = index === selectedIndex;
            const Icon = command.icon;

            return (
              <button
                key={command.name}
                data-command-item
                onClick={() => onSelect(command)}
                onMouseEnter={() => setSelectedIndex(index)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2 text-left transition-colors',
                  isSelected ? 'bg-white/10' : 'hover:bg-white/5'
                )}
              >
                <Icon
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    isSelected ? 'text-white/80' : 'text-white/50'
                  )}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'text-sm font-medium',
                        isSelected ? 'text-white' : 'text-white/80'
                      )}
                    >
                      /{command.name}
                    </span>
                    {command.hasArgs && (
                      <span className="text-xs text-white/30">
                        {command.argPlaceholder}
                      </span>
                    )}
                    {command.shortcut && (
                      <kbd className="ml-auto px-1.5 py-0.5 text-[10px] bg-white/5 border border-white/10 rounded text-white/40">
                        {command.shortcut}
                      </kbd>
                    )}
                  </div>
                  <p className="text-xs text-white/40 truncate">
                    {command.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      ))}

      {/* Hint footer */}
      <div className="px-3 py-2 border-t border-white/5 flex items-center gap-3 text-[10px] text-white/30">
        <span className="flex items-center gap-1">
          <kbd className="px-1 bg-white/5 rounded">↑↓</kbd> navigate
        </span>
        <span className="flex items-center gap-1">
          <kbd className="px-1 bg-white/5 rounded">↵</kbd> select
        </span>
        <span className="flex items-center gap-1">
          <kbd className="px-1 bg-white/5 rounded">esc</kbd> close
        </span>
      </div>
    </div>
  );
}

export default SlashCommandPopover;

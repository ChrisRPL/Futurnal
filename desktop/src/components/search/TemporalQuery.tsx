/**
 * TemporalQuery - Natural language temporal query interface
 *
 * Step 08: Frontend Intelligence Integration - Phase 4
 *
 * Research Foundation:
 * - TemporalMed: Temporal-aware data retrieval
 * - Youtu-GraphRAG: Temporal context integration
 *
 * Features:
 * - Natural language input: "last week", "yesterday", "before 2024"
 * - Date range picker with presets
 * - Preview of parsed date range
 * - Apply button sends filter to searchStore
 */

import { useState, useCallback, useMemo } from 'react';
import {
  Calendar,
  Clock,
  X,
  Check,
  ChevronDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useSearchStore, type TemporalRange } from '@/stores/searchStore';

interface TemporalQueryProps {
  /** Callback when range is applied */
  onApply?: () => void;
  /** Callback when popover closes */
  onClose?: () => void;
  /** Additional class names */
  className?: string;
}

/** Preset temporal ranges */
interface TemporalPreset {
  label: string;
  value: string;
  getRange: () => TemporalRange;
}

const TEMPORAL_PRESETS: TemporalPreset[] = [
  {
    label: 'Today',
    value: 'today',
    getRange: () => {
      const now = new Date();
      const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const end = new Date(start.getTime() + 24 * 60 * 60 * 1000 - 1);
      return { start, end, label: 'Today' };
    },
  },
  {
    label: 'Yesterday',
    value: 'yesterday',
    getRange: () => {
      const now = new Date();
      const start = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
      const end = new Date(start.getTime() + 24 * 60 * 60 * 1000 - 1);
      return { start, end, label: 'Yesterday' };
    },
  },
  {
    label: 'Last 7 days',
    value: 'last_7_days',
    getRange: () => {
      const now = new Date();
      const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);
      const start = new Date(end.getTime() - 7 * 24 * 60 * 60 * 1000);
      return { start, end, label: 'Last 7 days' };
    },
  },
  {
    label: 'Last 30 days',
    value: 'last_30_days',
    getRange: () => {
      const now = new Date();
      const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);
      const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
      return { start, end, label: 'Last 30 days' };
    },
  },
  {
    label: 'This month',
    value: 'this_month',
    getRange: () => {
      const now = new Date();
      const start = new Date(now.getFullYear(), now.getMonth(), 1);
      const end = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59);
      return { start, end, label: 'This month' };
    },
  },
  {
    label: 'Last month',
    value: 'last_month',
    getRange: () => {
      const now = new Date();
      const start = new Date(now.getFullYear(), now.getMonth() - 1, 1);
      const end = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59);
      return { start, end, label: 'Last month' };
    },
  },
  {
    label: 'This year',
    value: 'this_year',
    getRange: () => {
      const now = new Date();
      const start = new Date(now.getFullYear(), 0, 1);
      const end = new Date(now.getFullYear(), 11, 31, 23, 59, 59);
      return { start, end, label: 'This year' };
    },
  },
  {
    label: 'Last year',
    value: 'last_year',
    getRange: () => {
      const now = new Date();
      const start = new Date(now.getFullYear() - 1, 0, 1);
      const end = new Date(now.getFullYear() - 1, 11, 31, 23, 59, 59);
      return { start, end, label: 'Last year' };
    },
  },
];

/**
 * Parse natural language temporal expression
 * Returns null if not recognized
 */
function parseTemporalExpression(expression: string): TemporalRange | null {
  const lower = expression.toLowerCase().trim();

  // Check presets first
  for (const preset of TEMPORAL_PRESETS) {
    if (lower === preset.value || lower === preset.label.toLowerCase()) {
      return preset.getRange();
    }
  }

  // Common patterns
  if (lower === 'today') return TEMPORAL_PRESETS[0].getRange();
  if (lower === 'yesterday') return TEMPORAL_PRESETS[1].getRange();
  if (lower.includes('last week') || lower.includes('last 7 days')) {
    return TEMPORAL_PRESETS[2].getRange();
  }
  if (lower.includes('last month')) return TEMPORAL_PRESETS[5].getRange();
  if (lower.includes('this month')) return TEMPORAL_PRESETS[4].getRange();
  if (lower.includes('this year')) return TEMPORAL_PRESETS[6].getRange();
  if (lower.includes('last year')) return TEMPORAL_PRESETS[7].getRange();

  // "N days ago" pattern
  const daysAgoMatch = lower.match(/(\d+)\s*days?\s*ago/);
  if (daysAgoMatch) {
    const days = parseInt(daysAgoMatch[1], 10);
    const now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate() - days);
    const end = new Date(start.getTime() + 24 * 60 * 60 * 1000 - 1);
    return { start, end, label: `${days} days ago` };
  }

  // "last N days" pattern
  const lastNDaysMatch = lower.match(/last\s*(\d+)\s*days?/);
  if (lastNDaysMatch) {
    const days = parseInt(lastNDaysMatch[1], 10);
    const now = new Date();
    const end = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);
    const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
    return { start, end, label: `Last ${days} days` };
  }

  return null;
}

/**
 * Format date range for display
 */
function formatDateRange(range: TemporalRange): string {
  const startStr = range.start.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: range.start.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined,
  });
  const endStr = range.end.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: range.end.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined,
  });

  // Same day
  if (range.start.toDateString() === range.end.toDateString()) {
    return startStr;
  }

  return `${startStr} - ${endStr}`;
}

export function TemporalQuery({
  onApply,
  onClose,
  className,
}: TemporalQueryProps) {
  const {
    temporalRange,
    parsedTemporalExpression,
    setTemporalRange,
    clearTemporalRange,
  } = useSearchStore();

  const [inputValue, setInputValue] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [showPresets, setShowPresets] = useState(true);

  // Preview the parsed range
  const previewRange = useMemo(() => {
    if (inputValue.trim()) {
      return parseTemporalExpression(inputValue);
    }
    return null;
  }, [inputValue]);

  // Current range to display (preview or actual)
  const displayRange = previewRange || temporalRange;

  // Handle preset click
  const handlePresetClick = useCallback((preset: TemporalPreset) => {
    setSelectedPreset(preset.value);
    setInputValue(preset.label);
    const range = preset.getRange();
    setTemporalRange(range);
  }, [setTemporalRange]);

  // Handle input change
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
    setSelectedPreset(null);

    // Auto-parse and set
    const parsed = parseTemporalExpression(e.target.value);
    if (parsed) {
      setTemporalRange(parsed);
    }
  }, [setTemporalRange]);

  // Handle apply
  const handleApply = useCallback(() => {
    if (displayRange) {
      setTemporalRange(displayRange);
    }
    onApply?.();
    onClose?.();
  }, [displayRange, setTemporalRange, onApply, onClose]);

  // Handle clear
  const handleClear = useCallback(() => {
    setInputValue('');
    setSelectedPreset(null);
    clearTemporalRange();
  }, [clearTemporalRange]);

  return (
    <div className={cn('p-4 bg-[var(--color-surface)] border border-[var(--color-border)]', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-[var(--color-text-tertiary)]" />
          <span className="text-sm font-medium text-[var(--color-text-primary)]">
            Time Range
          </span>
        </div>
        {temporalRange && (
          <button
            onClick={handleClear}
            className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] flex items-center gap-1"
          >
            <X className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>

      {/* Natural language input */}
      <div className="mb-3">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          placeholder='e.g., "last week", "yesterday", "30 days ago"'
          className={cn(
            'w-full px-3 py-2 text-sm',
            'bg-[var(--color-bg-primary)] border border-[var(--color-border)]',
            'text-[var(--color-text-primary)] placeholder-[var(--color-text-faint)]',
            'focus:outline-none focus:border-[var(--color-border-active)]'
          )}
        />
      </div>

      {/* Presets toggle */}
      <button
        onClick={() => setShowPresets(!showPresets)}
        className="flex items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] mb-2"
      >
        <ChevronDown className={cn('w-3 h-3 transition-transform', showPresets && 'rotate-180')} />
        Quick presets
      </button>

      {/* Presets */}
      {showPresets && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {TEMPORAL_PRESETS.map((preset) => (
            <button
              key={preset.value}
              onClick={() => handlePresetClick(preset)}
              className={cn(
                'px-2 py-1 text-xs transition-colors',
                selectedPreset === preset.value
                  ? 'bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)]'
                  : 'bg-[var(--color-bg-primary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)]'
              )}
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}

      {/* Preview */}
      {displayRange && (
        <div className="flex items-center gap-2 px-3 py-2 bg-[var(--color-bg-primary)] border border-[var(--color-border)] mb-3">
          <Calendar className="w-4 h-4 text-[var(--color-text-tertiary)]" />
          <span className="text-sm text-[var(--color-text-secondary)]">
            {formatDateRange(displayRange)}
          </span>
          {displayRange.label && (
            <span className="text-xs text-[var(--color-text-muted)]">
              ({displayRange.label})
            </span>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-end gap-2">
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
        >
          Cancel
        </button>
        <button
          onClick={handleApply}
          disabled={!displayRange}
          className={cn(
            'flex items-center gap-1 px-3 py-1.5 text-xs transition-colors',
            displayRange
              ? 'bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)] hover:bg-[var(--color-inverse-bg-hover)]'
              : 'bg-[var(--color-surface)] text-[var(--color-text-faint)] cursor-not-allowed'
          )}
        >
          <Check className="w-3 h-3" />
          Apply
        </button>
      </div>
    </div>
  );
}

export default TemporalQuery;

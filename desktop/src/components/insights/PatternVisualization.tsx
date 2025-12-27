/**
 * Pattern Visualization - Charts for Phase 2B pattern detection
 *
 * Features:
 * - Day-of-week activity patterns
 * - Time-lagged correlation visualization
 * - Pattern significance indicators
 */

import { useEffect, useState } from 'react';
import {
  Calendar,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Loader2,
  ChevronDown,
  ChevronUp,
  BarChart3,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

interface DayOfWeekPattern {
  dayIndex: number;
  dayName: string;
  eventCount: number;
  averageCount: number;
  deviationPct: number;
  isPeak: boolean;
  isTrough: boolean;
  eventType: string;
}

interface TimeLaggedPattern {
  lagHours: number;
  lagRange: string;
  occurrenceCount: number;
  avgActualLagHours: number;
  proportion: number;
  eventTypeA: string;
  eventTypeB: string;
  isSignificant: boolean;
}

interface PatternsData {
  dayOfWeek: DayOfWeekPattern[];
  timeLagged: TimeLaggedPattern[];
}

interface TimeRange {
  start: string;
  end: string;
}

interface PatternsResponse {
  success: boolean;
  timeRange?: TimeRange;
  patterns: PatternsData;
  error?: string;
}

// ============================================================================
// Day-of-Week Chart
// ============================================================================

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function DayOfWeekChart({ patterns }: { patterns: DayOfWeekPattern[] }) {
  if (patterns.length === 0) {
    return (
      <div className="text-center py-6 text-xs text-white/40">
        <Calendar className="w-6 h-6 mx-auto mb-2 opacity-50" />
        <p>No weekly patterns detected</p>
      </div>
    );
  }

  // Find max count for scaling
  const maxCount = Math.max(...patterns.map((p) => p.eventCount));

  // Group by day
  const dayPatterns = DAYS.map((_, idx) => {
    return patterns.find((p) => p.dayIndex === idx) || null;
  });

  return (
    <div className="space-y-3">
      {/* Bar chart */}
      <div className="flex items-end justify-between gap-1 h-24">
        {dayPatterns.map((pattern, idx) => {
          const height = pattern ? (pattern.eventCount / maxCount) * 100 : 0;
          const isPeak = pattern?.isPeak ?? false;
          const isTrough = pattern?.isTrough ?? false;

          return (
            <div key={idx} className="flex-1 flex flex-col items-center gap-1">
              <div className="relative w-full flex justify-center">
                <div
                  className={cn(
                    'w-full max-w-8 rounded-t transition-all',
                    isPeak
                      ? 'bg-green-500'
                      : isTrough
                        ? 'bg-red-500/70'
                        : 'bg-white/20'
                  )}
                  style={{ height: `${Math.max(height, 4)}px` }}
                />
                {isPeak && (
                  <TrendingUp className="w-3 h-3 text-green-400 absolute -top-4" />
                )}
                {isTrough && (
                  <TrendingDown className="w-3 h-3 text-red-400 absolute -top-4" />
                )}
              </div>
              <span className="text-[10px] text-white/40">{DAYS[idx]}</span>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 text-[10px] text-white/50">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded" />
          <span>Peak</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-red-500/70 rounded" />
          <span>Trough</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-white/20 rounded" />
          <span>Normal</span>
        </div>
      </div>

      {/* Stats table */}
      <div className="space-y-1 max-h-32 overflow-y-auto">
        {patterns
          .filter((p) => p.isPeak || p.isTrough)
          .map((pattern, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-2 bg-white/[0.02] text-xs"
            >
              <div className="flex items-center gap-2">
                {pattern.isPeak ? (
                  <TrendingUp className="w-3 h-3 text-green-400" />
                ) : (
                  <TrendingDown className="w-3 h-3 text-red-400" />
                )}
                <span className="text-white/70">{pattern.dayName}</span>
                <span className="text-white/40">({pattern.eventType})</span>
              </div>
              <span
                className={cn(
                  'text-[10px]',
                  pattern.deviationPct > 0 ? 'text-green-400' : 'text-red-400'
                )}
              >
                {pattern.deviationPct > 0 ? '+' : ''}
                {Math.round(pattern.deviationPct)}%
              </span>
            </div>
          ))}
      </div>
    </div>
  );
}

// ============================================================================
// Time-Lagged Patterns
// ============================================================================

function TimeLaggedChart({ patterns }: { patterns: TimeLaggedPattern[] }) {
  if (patterns.length === 0) {
    return (
      <div className="text-center py-6 text-xs text-white/40">
        <Clock className="w-6 h-6 mx-auto mb-2 opacity-50" />
        <p>No time-lagged correlations detected</p>
      </div>
    );
  }

  // Group by event pair
  const groupedPatterns = patterns.reduce(
    (acc, p) => {
      const key = `${p.eventTypeA} → ${p.eventTypeB}`;
      if (!acc[key]) acc[key] = [];
      acc[key].push(p);
      return acc;
    },
    {} as Record<string, TimeLaggedPattern[]>
  );

  return (
    <div className="space-y-3">
      {Object.entries(groupedPatterns).map(([pairKey, pairPatterns]) => (
        <div key={pairKey} className="p-3 bg-white/[0.02] border border-white/5">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-3 h-3 text-white/40" />
            <span className="text-xs font-medium text-white/70">{pairKey}</span>
          </div>

          {/* Lag distribution */}
          <div className="flex gap-1 mb-2">
            {pairPatterns.map((p, idx) => (
              <div
                key={idx}
                className="flex-1 text-center"
                title={`${p.lagRange}: ${p.occurrenceCount} occurrences`}
              >
                <div
                  className={cn(
                    'h-8 rounded-sm flex items-end justify-center mb-1',
                    p.isSignificant ? 'bg-blue-500/50' : 'bg-white/10'
                  )}
                  style={{ height: `${Math.max(p.proportion * 100, 20)}px` }}
                >
                  <span className="text-[10px] text-white/70 pb-1">
                    {p.occurrenceCount}
                  </span>
                </div>
                <span className="text-[8px] text-white/40 line-clamp-1">
                  {p.lagRange}
                </span>
              </div>
            ))}
          </div>

          {/* Significant patterns */}
          {pairPatterns.filter((p) => p.isSignificant).length > 0 && (
            <div className="flex items-center gap-2 text-[10px] text-blue-400/70">
              <BarChart3 className="w-3 h-3" />
              <span>
                Significant at{' '}
                {pairPatterns
                  .filter((p) => p.isSignificant)
                  .map((p) => p.lagRange)
                  .join(', ')}
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Main Pattern Visualization Component
// ============================================================================

export function PatternVisualization({ className }: { className?: string }) {
  const [patterns, setPatterns] = useState<PatternsData | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'weekly' | 'lagged'>('weekly');
  const [isExpanded, setIsExpanded] = useState(true);
  const [days, setDays] = useState(30);

  const fetchPatterns = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await invoke<PatternsResponse>('detect_patterns', {
        days,
      });

      if (response.success) {
        setPatterns(response.patterns);
        setTimeRange(response.timeRange || null);
      } else {
        setError(response.error || 'Failed to fetch patterns');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Pattern detection failed');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPatterns();
  }, [days]);

  return (
    <div className={cn('p-4 bg-white/[0.02] border border-white/10', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2"
        >
          <TrendingUp className="w-4 h-4 text-white/60" />
          <span className="text-sm font-medium text-white/80">Pattern Detection</span>
          {isExpanded ? (
            <ChevronUp className="w-3 h-3 text-white/40" />
          ) : (
            <ChevronDown className="w-3 h-3 text-white/40" />
          )}
        </button>

        <div className="flex items-center gap-2">
          {/* Days selector */}
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="px-2 py-1 text-[10px] bg-white/5 border border-white/10 text-white/60 rounded"
          >
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
          </select>

          <button
            onClick={fetchPatterns}
            disabled={isLoading}
            className="p-1 text-white/40 hover:text-white/60"
          >
            <RefreshCw className={cn('w-3.5 h-3.5', isLoading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Error */}
          {error && (
            <div className="mb-3 p-2 bg-red-500/10 border border-red-500/20 text-xs text-red-400">
              {error}
            </div>
          )}

          {/* Time range */}
          {timeRange && (
            <div className="mb-3 text-[10px] text-white/40">
              Analyzing: {new Date(timeRange.start).toLocaleDateString()} -{' '}
              {new Date(timeRange.end).toLocaleDateString()}
            </div>
          )}

          {/* Tabs */}
          <div className="flex gap-1 mb-3">
            <button
              onClick={() => setActiveTab('weekly')}
              className={cn(
                'flex items-center gap-1 px-2 py-1 text-xs rounded',
                activeTab === 'weekly'
                  ? 'bg-white/20 text-white/80'
                  : 'bg-white/5 text-white/50 hover:text-white/70'
              )}
            >
              <Calendar className="w-3 h-3" />
              <span>Weekly Rhythm</span>
            </button>
            <button
              onClick={() => setActiveTab('lagged')}
              className={cn(
                'flex items-center gap-1 px-2 py-1 text-xs rounded',
                activeTab === 'lagged'
                  ? 'bg-white/20 text-white/80'
                  : 'bg-white/5 text-white/50 hover:text-white/70'
              )}
            >
              <Clock className="w-3 h-3" />
              <span>Time-Lagged</span>
            </button>
          </div>

          {/* Content */}
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-5 h-5 text-white/40 animate-spin" />
            </div>
          ) : patterns ? (
            activeTab === 'weekly' ? (
              <DayOfWeekChart patterns={patterns.dayOfWeek} />
            ) : (
              <TimeLaggedChart patterns={patterns.timeLagged} />
            )
          ) : (
            <div className="text-center py-8 text-xs text-white/40">
              <Minus className="w-6 h-6 mx-auto mb-2 opacity-50" />
              <p>No patterns available</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default PatternVisualization;

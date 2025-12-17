/**
 * Activity Store - Zustand state management for activity stream
 *
 * Step 08: Frontend Intelligence Integration - Phase 3
 *
 * Research Foundation:
 * - AgentFlow: Activity tracking patterns
 * - RLHI: User interaction history
 *
 * Manages activity stream state: events, filters, pagination.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Activity event categories */
export type ActivityCategory = 'data' | 'interaction' | 'learning';

/** Activity event types */
export type ActivityEventType =
  | 'search'
  | 'document'
  | 'chat'
  | 'insight'
  | 'schema'
  | 'entity'
  | 'ingestion';

/** Single activity event */
export interface ActivityEvent {
  id: string;
  type: ActivityEventType;
  category: ActivityCategory;
  title: string;
  description?: string;
  timestamp: string;
  relatedEntityIds: string[];
  metadata: Record<string, unknown>;
}

/** Date range filter */
export interface DateRange {
  start: Date;
  end: Date;
}

/** Activity filters */
export interface ActivityFilters {
  types: ActivityEventType[];
  dateRange: DateRange | null;
}

/** Response from get_activity_log command */
interface ActivityListResponse {
  success: boolean;
  events: ActivityEvent[];
  total: number;
  limit: number;
  offset: number;
  error?: string;
}

// ============================================================================
// Helpers
// ============================================================================

/** Group activities by relative time */
export function groupActivitiesByTime(events: ActivityEvent[]): Map<string, ActivityEvent[]> {
  const groups = new Map<string, ActivityEvent[]>();
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  for (const event of events) {
    const eventDate = new Date(event.timestamp);
    const eventDay = new Date(eventDate.getFullYear(), eventDate.getMonth(), eventDate.getDate());

    let group: string;
    if (eventDay.getTime() >= today.getTime()) {
      group = 'Today';
    } else if (eventDay.getTime() >= yesterday.getTime()) {
      group = 'Yesterday';
    } else if (eventDay.getTime() >= lastWeek.getTime()) {
      group = 'This Week';
    } else {
      group = 'Earlier';
    }

    const existing = groups.get(group) || [];
    existing.push(event);
    groups.set(group, existing);
  }

  return groups;
}

/** Get icon name for activity type */
export function getActivityIcon(type: ActivityEventType): string {
  switch (type) {
    case 'search':
      return 'Search';
    case 'document':
      return 'FileText';
    case 'chat':
      return 'MessageSquare';
    case 'insight':
      return 'Lightbulb';
    case 'schema':
      return 'Database';
    case 'entity':
      return 'Circle';
    case 'ingestion':
      return 'Download';
    default:
      return 'Activity';
  }
}

/** Get category color class */
export function getCategoryColor(category: ActivityCategory): string {
  switch (category) {
    case 'data':
      return 'text-white/70';
    case 'interaction':
      return 'text-white/60';
    case 'learning':
      return 'text-white/50';
    default:
      return 'text-white/40';
  }
}

// ============================================================================
// Store
// ============================================================================

interface ActivityState {
  /** All loaded activity events */
  events: ActivityEvent[];
  /** Current filters */
  filters: ActivityFilters;
  /** Whether more events are available */
  hasMore: boolean;
  /** Total count of events */
  total: number;
  /** Current pagination offset */
  offset: number;
  /** Loading state */
  isLoading: boolean;
  /** Error message */
  error: string | null;

  // Actions
  /** Fetch activities with current filters */
  fetchActivities: (reset?: boolean) => Promise<void>;
  /** Load more activities (pagination) */
  loadMore: () => Promise<void>;
  /** Get recent activities (for widget) */
  getRecentActivities: (limit?: number) => ActivityEvent[];
  /** Set type filter */
  setTypeFilter: (types: ActivityEventType[]) => void;
  /** Set date range filter */
  setDateRange: (range: DateRange | null) => void;
  /** Clear all filters */
  clearFilters: () => void;
  /** Clear error */
  clearError: () => void;
}

const DEFAULT_LIMIT = 50;

export const useActivityStore = create<ActivityState>()((set, get) => ({
  // Initial state
  events: [],
  filters: {
    types: [],
    dateRange: null,
  },
  hasMore: true,
  total: 0,
  offset: 0,
  isLoading: false,
  error: null,

  // Fetch activities
  fetchActivities: async (reset = true) => {
    const state = get();
    const offset = reset ? 0 : state.offset;

    set({ isLoading: true, error: null });

    try {
      // Build filter args
      const eventTypes = state.filters.types.length > 0 ? state.filters.types : undefined;
      const dateFrom = state.filters.dateRange?.start.toISOString();
      const dateTo = state.filters.dateRange?.end.toISOString();

      const response = await invoke<ActivityListResponse>('get_activity_log', {
        limit: DEFAULT_LIMIT,
        offset,
        eventTypes,
        dateFrom,
        dateTo,
      });

      if (response.success) {
        const newEvents = reset ? response.events : [...state.events, ...response.events];
        const newOffset = offset + response.events.length;
        const hasMore = newOffset < response.total;

        set({
          events: newEvents,
          total: response.total,
          offset: newOffset,
          hasMore,
          isLoading: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch activities',
          isLoading: false,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Fetch activities failed';
      set({ error: errorMsg, isLoading: false });
    }
  },

  // Load more (pagination)
  loadMore: async () => {
    const state = get();
    if (state.isLoading || !state.hasMore) return;
    await state.fetchActivities(false);
  },

  // Get recent activities (for widget)
  getRecentActivities: (limit = 10) => {
    const state = get();
    return state.events.slice(0, limit);
  },

  // Set type filter
  setTypeFilter: (types) => {
    set((state) => ({
      filters: { ...state.filters, types },
    }));
  },

  // Set date range filter
  setDateRange: (range) => {
    set((state) => ({
      filters: { ...state.filters, dateRange: range },
    }));
  },

  // Clear all filters
  clearFilters: () => {
    set({
      filters: {
        types: [],
        dateRange: null,
      },
    });
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },
}));

export default useActivityStore;

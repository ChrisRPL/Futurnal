/**
 * Notification Store - Zustand state management for Phase 2D notifications
 *
 * Manages notification preferences, history, and delivery status.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// Types
// ============================================================================

/** Do-not-disturb schedule. */
export interface DndSchedule {
  enabled: boolean;
  startTime: string;
  endTime: string;
  days: number[];
  isActive: boolean;
}

/** Channel preferences. */
export interface ChannelPreferences {
  dashboardEnabled: boolean;
  desktopEnabled: boolean;
  minPriorityDesktop: string;
}

/** Notification preferences. */
export interface NotificationPreferences {
  frequency: string;
  maxDailyNotifications: number;
  minInsightConfidence: number;
  dndSchedule: DndSchedule;
  channels: ChannelPreferences;
}

/** A notification item. */
export interface NotificationItem {
  notificationId: string;
  title: string;
  body: string;
  insightId?: string;
  priority: string;
  createdAt: string;
  deliveredAt?: string;
  read: boolean;
  actionUrl?: string;
  metadata: Record<string, unknown>;
}

/** Notification system status. */
export interface NotificationStatus {
  pendingInsights: number;
  unreadNotifications: number;
  notificationsToday: number;
  maxDaily: number;
  frequency: string;
  dndActive: boolean;
  desktopEnabled: boolean;
  dashboardEnabled: boolean;
}

// ============================================================================
// Response Types
// ============================================================================

interface NotificationPreferencesResponse {
  success: boolean;
  frequency: string;
  maxDailyNotifications: number;
  minInsightConfidence: number;
  dndSchedule: DndSchedule;
  channels: ChannelPreferences;
  error?: string;
}

interface SetFrequencyResponse {
  success: boolean;
  frequency: string;
  error?: string;
}

interface SetDndResponse {
  success: boolean;
  dndSchedule?: DndSchedule;
  isActive: boolean;
  error?: string;
}

interface NotificationHistoryResponse {
  success: boolean;
  notifications: NotificationItem[];
  totalCount: number;
  unreadCount: number;
  error?: string;
}

interface NotificationStatusResponse {
  success: boolean;
  pendingInsights: number;
  unreadNotifications: number;
  notificationsToday: number;
  maxDaily: number;
  frequency: string;
  dndActive: boolean;
  desktopEnabled: boolean;
  dashboardEnabled: boolean;
  error?: string;
}

// ============================================================================
// Store
// ============================================================================

interface NotificationState {
  // Preferences
  preferences: NotificationPreferences | null;

  // History
  notifications: NotificationItem[];
  totalCount: number;
  unreadCount: number;

  // Status
  status: NotificationStatus | null;

  // Loading states
  isLoadingPreferences: boolean;
  isLoadingHistory: boolean;
  isLoadingStatus: boolean;
  isSaving: boolean;

  // Error states
  error: string | null;

  // Actions - Preferences
  fetchPreferences: () => Promise<void>;
  setFrequency: (frequency: string) => Promise<boolean>;
  setDnd: (options: {
    enabled?: boolean;
    startTime?: string;
    endTime?: string;
  }) => Promise<boolean>;

  // Actions - History
  fetchHistory: (limit?: number, unreadOnly?: boolean) => Promise<void>;
  markNotificationRead: (notificationId: string) => Promise<void>;
  clearNotifications: () => Promise<number>;

  // Actions - Status
  fetchStatus: () => Promise<void>;
  deliverNotifications: (force?: boolean) => Promise<number>;

  // Actions - Utility
  clearError: () => void;
  refreshAll: () => Promise<void>;
}

export const useNotificationStore = create<NotificationState>()((set, get) => ({
  // Initial state
  preferences: null,
  notifications: [],
  totalCount: 0,
  unreadCount: 0,
  status: null,
  isLoadingPreferences: false,
  isLoadingHistory: false,
  isLoadingStatus: false,
  isSaving: false,
  error: null,

  // -------------------------------------------------------------------------
  // Preferences Actions
  // -------------------------------------------------------------------------

  fetchPreferences: async () => {
    set({ isLoadingPreferences: true, error: null });

    try {
      const response = await invoke<NotificationPreferencesResponse>(
        'get_notification_preferences'
      );

      if (response.success) {
        set({
          preferences: {
            frequency: response.frequency,
            maxDailyNotifications: response.maxDailyNotifications,
            minInsightConfidence: response.minInsightConfidence,
            dndSchedule: response.dndSchedule,
            channels: response.channels,
          },
          isLoadingPreferences: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch preferences',
          isLoadingPreferences: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch preferences failed';
      set({ error: errorMsg, isLoadingPreferences: false });
    }
  },

  setFrequency: async (frequency: string) => {
    set({ isSaving: true, error: null });

    try {
      const response = await invoke<SetFrequencyResponse>(
        'set_notification_frequency',
        { request: { frequency } }
      );

      if (response.success) {
        set((state) => ({
          preferences: state.preferences
            ? { ...state.preferences, frequency: response.frequency }
            : null,
          isSaving: false,
        }));
        return true;
      } else {
        set({ error: response.error || 'Failed to set frequency', isSaving: false });
        return false;
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Set frequency failed';
      set({ error: errorMsg, isSaving: false });
      return false;
    }
  },

  setDnd: async (options) => {
    set({ isSaving: true, error: null });

    try {
      const response = await invoke<SetDndResponse>('set_notification_dnd', {
        request: {
          enabled: options.enabled,
          startTime: options.startTime,
          endTime: options.endTime,
        },
      });

      if (response.success && response.dndSchedule) {
        set((state) => ({
          preferences: state.preferences
            ? { ...state.preferences, dndSchedule: response.dndSchedule! }
            : null,
          isSaving: false,
        }));
        return true;
      } else {
        set({ error: response.error || 'Failed to set DND', isSaving: false });
        return false;
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Set DND failed';
      set({ error: errorMsg, isSaving: false });
      return false;
    }
  },

  // -------------------------------------------------------------------------
  // History Actions
  // -------------------------------------------------------------------------

  fetchHistory: async (limit?: number, unreadOnly?: boolean) => {
    set({ isLoadingHistory: true, error: null });

    try {
      const response = await invoke<NotificationHistoryResponse>(
        'get_notification_history',
        { limit, unreadOnly }
      );

      if (response.success) {
        set({
          notifications: response.notifications,
          totalCount: response.totalCount,
          unreadCount: response.unreadCount,
          isLoadingHistory: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch history',
          isLoadingHistory: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch history failed';
      set({ error: errorMsg, isLoadingHistory: false });
    }
  },

  markNotificationRead: async (notificationId: string) => {
    try {
      const success = await invoke<boolean>('mark_notification_read', {
        notificationId,
      });

      if (success) {
        set((state) => ({
          notifications: state.notifications.map((n) =>
            n.notificationId === notificationId ? { ...n, read: true } : n
          ),
          unreadCount: Math.max(0, state.unreadCount - 1),
        }));
      }
    } catch (error) {
      console.error('Failed to mark notification read:', error);
    }
  },

  clearNotifications: async () => {
    try {
      const clearedCount = await invoke<number>('clear_notifications');
      set({
        notifications: [],
        totalCount: 0,
        unreadCount: 0,
      });
      return clearedCount;
    } catch (error) {
      console.error('Failed to clear notifications:', error);
      return 0;
    }
  },

  // -------------------------------------------------------------------------
  // Status Actions
  // -------------------------------------------------------------------------

  fetchStatus: async () => {
    set({ isLoadingStatus: true, error: null });

    try {
      const response = await invoke<NotificationStatusResponse>(
        'get_notification_status'
      );

      if (response.success) {
        set({
          status: {
            pendingInsights: response.pendingInsights,
            unreadNotifications: response.unreadNotifications,
            notificationsToday: response.notificationsToday,
            maxDaily: response.maxDaily,
            frequency: response.frequency,
            dndActive: response.dndActive,
            desktopEnabled: response.desktopEnabled,
            dashboardEnabled: response.dashboardEnabled,
          },
          isLoadingStatus: false,
        });
      } else {
        set({
          error: response.error || 'Failed to fetch status',
          isLoadingStatus: false,
        });
      }
    } catch (error) {
      const errorMsg =
        error instanceof Error ? error.message : 'Fetch status failed';
      set({ error: errorMsg, isLoadingStatus: false });
    }
  },

  deliverNotifications: async (force?: boolean) => {
    try {
      const deliveredCount = await invoke<number>('deliver_notifications', {
        force,
      });

      // Refresh after delivery
      await get().fetchStatus();
      await get().fetchHistory();

      return deliveredCount;
    } catch (error) {
      console.error('Failed to deliver notifications:', error);
      return 0;
    }
  },

  // -------------------------------------------------------------------------
  // Utility Actions
  // -------------------------------------------------------------------------

  clearError: () => {
    set({ error: null });
  },

  refreshAll: async () => {
    const state = get();
    await Promise.all([
      state.fetchPreferences(),
      state.fetchHistory(),
      state.fetchStatus(),
    ]);
  },
}));

export default useNotificationStore;

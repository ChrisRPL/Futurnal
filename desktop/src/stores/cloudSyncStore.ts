/**
 * Cloud Sync Store - Zustand state management for Firebase cloud sync
 *
 * Manages cloud sync consent status, sync operations, and periodic sync scheduling.
 * Coordinates with the Python backend for consent storage and the Firebase client for sync.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import type {
  CloudSyncScope,
  CloudSyncConsentStatus,
  GrantCloudSyncRequest,
} from '@/types/api';
import {
  performCloudSync,
  deleteAllCloudData,
  type CloudSyncResult,
} from '@/lib/cloudSync';

/** Sync operation that's queued for when online */
interface PendingOperation {
  id: string;
  type: 'sync' | 'delete';
  timestamp: string;
  data?: Record<string, unknown>;
}

interface CloudSyncState {
  // Consent status
  consentStatus: CloudSyncConsentStatus | null;
  isLoadingConsent: boolean;
  consentError: string | null;

  // Sync status
  syncEnabled: boolean;
  syncInProgress: boolean;
  lastSyncAt: string | null;
  lastSyncError: string | null;
  nodesSynced: number;

  // Sync settings
  autoSyncIntervalMinutes: number;
  syncOnClose: boolean;

  // Offline queue
  pendingOperations: PendingOperation[];
  isOnline: boolean;

  // Periodic sync
  syncIntervalId: number | null;

  // Actions - Consent
  loadConsentStatus: () => Promise<void>;
  grantConsent: (scopes: CloudSyncScope[], operator?: string) => Promise<void>;
  revokeConsent: (operator?: string) => Promise<void>;

  // Actions - Sync
  enableSync: () => void;
  disableSync: () => void;
  syncNow: () => Promise<void>;
  startPeriodicSync: () => void;
  stopPeriodicSync: () => void;

  // Actions - Settings
  setAutoSyncInterval: (minutes: number) => void;
  setSyncOnClose: (enabled: boolean) => void;

  // Actions - Offline
  setOnlineStatus: (isOnline: boolean) => void;
  addPendingOperation: (op: Omit<PendingOperation, 'id' | 'timestamp'>) => void;
  clearPendingOperations: () => void;

  // Actions - Internal
  reset: () => void;
}

const DEFAULT_SYNC_INTERVAL_MINUTES = 15;

const initialState = {
  consentStatus: null,
  isLoadingConsent: false,
  consentError: null,
  syncEnabled: false,
  syncInProgress: false,
  lastSyncAt: null,
  lastSyncError: null,
  nodesSynced: 0,
  autoSyncIntervalMinutes: DEFAULT_SYNC_INTERVAL_MINUTES,
  syncOnClose: true,
  pendingOperations: [],
  isOnline: true,
  syncIntervalId: null,
};

export const useCloudSyncStore = create<CloudSyncState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Load consent status from Python backend
      loadConsentStatus: async () => {
        set({ isLoadingConsent: true, consentError: null });
        try {
          const status = await invoke<CloudSyncConsentStatus>('get_cloud_sync_consent');
          set({
            consentStatus: status,
            isLoadingConsent: false,
            // Auto-enable sync if consent is granted
            syncEnabled: status.has_consent && get().syncEnabled,
          });
        } catch (error) {
          console.error('Failed to load cloud sync consent:', error);
          set({
            isLoadingConsent: false,
            consentError: error instanceof Error ? error.message : 'Failed to load consent status',
          });
        }
      },

      // Grant consent for specified scopes
      grantConsent: async (scopes: CloudSyncScope[], operator?: string) => {
        set({ isLoadingConsent: true, consentError: null });
        try {
          const request: GrantCloudSyncRequest = { scopes, operator };
          const status = await invoke<CloudSyncConsentStatus>('grant_cloud_sync_consent', { request });
          set({
            consentStatus: status,
            isLoadingConsent: false,
            syncEnabled: true, // Auto-enable sync after granting consent
          });

          // Start periodic sync if enabled
          if (status.has_consent) {
            get().startPeriodicSync();
          }
        } catch (error) {
          console.error('Failed to grant cloud sync consent:', error);
          set({
            isLoadingConsent: false,
            consentError: error instanceof Error ? error.message : 'Failed to grant consent',
          });
          throw error;
        }
      },

      // Revoke consent and delete cloud data
      revokeConsent: async (operator?: string) => {
        set({ isLoadingConsent: true, consentError: null });
        try {
          // Stop sync first
          get().stopPeriodicSync();
          set({ syncEnabled: false, syncInProgress: false });

          // Delete all cloud data from Firebase
          try {
            const deletedCount = await deleteAllCloudData();
            console.log(`Deleted ${deletedCount} documents from cloud`);

            // Log data deletion to audit
            await invoke('log_cloud_sync_audit', {
              action: 'data_deleted',
              success: true,
              nodes_affected: deletedCount,
            });
          } catch (deleteError) {
            console.error('Failed to delete cloud data:', deleteError);
            // Continue with consent revocation even if delete fails
            // User should be able to revoke consent regardless
          }

          // Revoke consent in Python backend
          await invoke('revoke_cloud_sync_consent', { operator });

          set({
            consentStatus: {
              has_consent: false,
              granted_scopes: [],
              is_syncing: false,
            },
            isLoadingConsent: false,
            lastSyncAt: null,
            nodesSynced: 0,
          });
        } catch (error) {
          console.error('Failed to revoke cloud sync consent:', error);
          set({
            isLoadingConsent: false,
            consentError: error instanceof Error ? error.message : 'Failed to revoke consent',
          });
          throw error;
        }
      },

      // Enable sync (after consent is granted)
      enableSync: () => {
        const { consentStatus } = get();
        if (!consentStatus?.has_consent) {
          console.warn('Cannot enable sync without consent');
          return;
        }
        set({ syncEnabled: true });
        get().startPeriodicSync();
      },

      // Disable sync (but keep consent)
      disableSync: () => {
        get().stopPeriodicSync();
        set({ syncEnabled: false, syncInProgress: false });
      },

      // Trigger immediate sync
      syncNow: async () => {
        const { consentStatus, syncEnabled, syncInProgress, isOnline } = get();

        if (!consentStatus?.has_consent) {
          console.warn('Cannot sync without consent');
          return;
        }

        if (!syncEnabled) {
          console.warn('Sync is disabled');
          return;
        }

        if (syncInProgress) {
          console.warn('Sync already in progress');
          return;
        }

        if (!isOnline) {
          // Queue for later
          get().addPendingOperation({ type: 'sync' });
          return;
        }

        set({ syncInProgress: true, lastSyncError: null });

        try {
          // Log sync started
          await invoke('log_cloud_sync_audit', {
            action: 'sync_started',
            success: true,
            nodes_affected: 0,
          });

          // Perform the actual sync via Firebase client
          const result: CloudSyncResult = await performCloudSync(
            consentStatus.granted_scopes as CloudSyncScope[]
          );

          const now = new Date().toISOString();

          if (result.success) {
            // Log sync completed
            await invoke('log_cloud_sync_audit', {
              action: 'sync_completed',
              success: true,
              nodes_affected: result.nodesSynced,
            });

            set({
              syncInProgress: false,
              lastSyncAt: now,
              lastSyncError: null,
              nodesSynced: result.nodesSynced,
            });
          } else {
            throw new Error(result.error || 'Sync failed');
          }
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Sync failed';
          console.error('Cloud sync failed:', error);

          // Log sync failed
          try {
            await invoke('log_cloud_sync_audit', {
              action: 'sync_failed',
              success: false,
              nodes_affected: 0,
              error_message: errorMessage,
            });
          } catch {
            // Ignore audit logging errors
          }

          set({
            syncInProgress: false,
            lastSyncError: errorMessage,
          });
        }
      },

      // Start periodic sync
      startPeriodicSync: () => {
        const { syncIntervalId, autoSyncIntervalMinutes, consentStatus, syncEnabled } = get();

        // Don't start if already running or no consent
        if (syncIntervalId !== null) return;
        if (!consentStatus?.has_consent) return;
        if (!syncEnabled) return;

        const intervalMs = autoSyncIntervalMinutes * 60 * 1000;
        const id = window.setInterval(() => {
          get().syncNow();
        }, intervalMs);

        set({ syncIntervalId: id });
        console.log(`Started periodic cloud sync every ${autoSyncIntervalMinutes} minutes`);
      },

      // Stop periodic sync
      stopPeriodicSync: () => {
        const { syncIntervalId } = get();
        if (syncIntervalId !== null) {
          window.clearInterval(syncIntervalId);
          set({ syncIntervalId: null });
          console.log('Stopped periodic cloud sync');
        }
      },

      // Update sync interval
      setAutoSyncInterval: (minutes: number) => {
        const wasRunning = get().syncIntervalId !== null;
        if (wasRunning) {
          get().stopPeriodicSync();
        }
        set({ autoSyncIntervalMinutes: minutes });
        if (wasRunning) {
          get().startPeriodicSync();
        }
      },

      // Update sync on close setting
      setSyncOnClose: (enabled: boolean) => {
        set({ syncOnClose: enabled });
      },

      // Update online status
      setOnlineStatus: (isOnline: boolean) => {
        const wasOffline = !get().isOnline;
        set({ isOnline });

        // Process pending operations when coming back online
        if (wasOffline && isOnline) {
          const { pendingOperations, consentStatus, syncEnabled } = get();
          if (pendingOperations.length > 0 && consentStatus?.has_consent && syncEnabled) {
            console.log('Back online, processing pending sync operations');
            get().syncNow();
            get().clearPendingOperations();
          }
        }
      },

      // Add pending operation for offline queue
      addPendingOperation: (op) => {
        const id = `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        set((state) => ({
          pendingOperations: [
            ...state.pendingOperations,
            { ...op, id, timestamp: new Date().toISOString() },
          ],
        }));
      },

      // Clear pending operations
      clearPendingOperations: () => {
        set({ pendingOperations: [] });
      },

      // Reset to initial state
      reset: () => {
        get().stopPeriodicSync();
        set(initialState);
      },
    }),
    {
      name: 'futurnal-cloud-sync',
      // Only persist these fields (not intervalId or loading states)
      partialize: (state) => ({
        syncEnabled: state.syncEnabled,
        autoSyncIntervalMinutes: state.autoSyncIntervalMinutes,
        syncOnClose: state.syncOnClose,
        lastSyncAt: state.lastSyncAt,
        nodesSynced: state.nodesSynced,
      }),
    }
  )
);

// Convenience selectors
export const useCloudSyncConsent = () =>
  useCloudSyncStore((state) => ({
    status: state.consentStatus,
    isLoading: state.isLoadingConsent,
    error: state.consentError,
    hasConsent: state.consentStatus?.has_consent ?? false,
  }));

export const useCloudSyncStatus = () =>
  useCloudSyncStore((state) => ({
    syncEnabled: state.syncEnabled,
    syncInProgress: state.syncInProgress,
    lastSyncAt: state.lastSyncAt,
    lastSyncError: state.lastSyncError,
    nodesSynced: state.nodesSynced,
    isOnline: state.isOnline,
  }));

export const useCloudSyncSettings = () =>
  useCloudSyncStore((state) => ({
    autoSyncIntervalMinutes: state.autoSyncIntervalMinutes,
    syncOnClose: state.syncOnClose,
  }));

// Hook to initialize cloud sync on app start
export function useCloudSyncInit() {
  const { loadConsentStatus, startPeriodicSync, setOnlineStatus, consentStatus, syncEnabled } =
    useCloudSyncStore();

  // This should be called once on app mount
  const initialize = async () => {
    // Load consent status
    await loadConsentStatus();

    // Set up online/offline listeners
    const handleOnline = () => setOnlineStatus(true);
    const handleOffline = () => setOnlineStatus(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    setOnlineStatus(navigator.onLine);

    // Start periodic sync if consent granted and sync enabled
    if (consentStatus?.has_consent && syncEnabled) {
      startPeriodicSync();
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  };

  return { initialize };
}

/**
 * Tauri Store Persistence Layer
 *
 * Provides persistent storage using Tauri's secure store plugin.
 * This is used for Zustand persist middleware integration.
 */

import { Store } from '@tauri-apps/plugin-store';

let store: Store | null = null;

/**
 * Get or create the Tauri store instance.
 */
async function getStore(): Promise<Store> {
  if (!store) {
    store = await Store.load('futurnal-store.json');
  }
  return store;
}

/**
 * Tauri storage adapter for general app data.
 */
export const tauriStorage = {
  /**
   * Get an item from storage.
   */
  async getItem(key: string): Promise<string | null> {
    try {
      const s = await getStore();
      const value = await s.get<string>(key);
      return value ?? null;
    } catch (error) {
      console.warn('[Storage] Failed to get item:', key, error);
      return null;
    }
  },

  /**
   * Set an item in storage.
   */
  async setItem(key: string, value: string): Promise<void> {
    try {
      const s = await getStore();
      await s.set(key, value);
      await s.save();
    } catch (error) {
      console.warn('[Storage] Failed to set item:', key, error);
    }
  },

  /**
   * Remove an item from storage.
   */
  async removeItem(key: string): Promise<void> {
    try {
      const s = await getStore();
      await s.delete(key);
      await s.save();
    } catch (error) {
      console.warn('[Storage] Failed to remove item:', key, error);
    }
  },
};

/**
 * Create a Zustand-compatible storage adapter for Tauri store.
 *
 * Note: Zustand's persist middleware expects synchronous operations,
 * but Tauri store is async. This adapter handles the async nature
 * by using localStorage as a synchronous cache with async persistence.
 */
export function createTauriStorage() {
  // Sync to Tauri store on setItem
  return {
    getItem: (name: string): string | null => {
      // Use localStorage for sync access, Tauri for persistence
      const value = localStorage.getItem(name);

      // Also sync from Tauri store in background
      tauriStorage.getItem(name).then((tauriValue) => {
        if (tauriValue && tauriValue !== value) {
          localStorage.setItem(name, tauriValue);
        }
      });

      return value;
    },

    setItem: (name: string, value: string): void => {
      // Write to both localStorage (sync) and Tauri store (async)
      localStorage.setItem(name, value);
      tauriStorage.setItem(name, value);
    },

    removeItem: (name: string): void => {
      // Remove from both
      localStorage.removeItem(name);
      tauriStorage.removeItem(name);
    },
  };
}

/**
 * Initialize storage by syncing Tauri store to localStorage.
 * Call this on app startup.
 */
export async function initializeStorage(): Promise<void> {
  try {
    const s = await getStore();
    const keys = await s.keys();

    for (const key of keys) {
      const value = await s.get<string>(key);
      if (value !== null && value !== undefined) {
        localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value));
      }
    }

    console.info('[Storage] Initialized with', keys.length, 'keys from Tauri store');
  } catch (error) {
    console.warn('[Storage] Failed to initialize from Tauri store:', error);
  }
}

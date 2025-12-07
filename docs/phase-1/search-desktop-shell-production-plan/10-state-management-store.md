Summary: Implement Zustand state management with TanStack Query for API caching and persistence.

# 10 · State Management

## Purpose

Create the state management layer using Zustand for local UI state and TanStack Query for server state, with persistence via Tauri's secure store plugin. This provides reactive updates, efficient caching, and seamless state synchronization across the application.

**Criticality**: MEDIUM - Essential for app coherence but uses standard patterns

## Scope

- Zustand stores for:
  - Search state (query, results, history, filters)
  - Connectors state (sources, status, sync state)
  - Settings state (preferences, theme, privacy)
  - User state (auth, subscription tier)
- TanStack Query for API state caching
- Persistence layer (Tauri store / localStorage)
- Search history with max 50 entries
- Reactive subscriptions for real-time updates
- DevTools integration for debugging

## Requirements Alignment

- **Architecture**: Lightweight, reactive state management
- **Performance**: Minimal re-renders, efficient updates
- **Persistence**: Settings and history survive app restarts

## Component Design

### Search Store

```typescript
// src/stores/searchStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { SearchResult, QueryIntent, SearchFilters } from '@/types/api';

interface SearchHistoryItem {
  id: string;
  query: string;
  timestamp: string;
  resultCount: number;
}

interface SearchState {
  // Current search
  query: string;
  results: SearchResult[];
  isSearching: boolean;
  error: string | null;
  intent: QueryIntent | null;
  executionTime: number | null;

  // Filters
  filters: SearchFilters;

  // History
  history: SearchHistoryItem[];

  // Actions
  setQuery: (query: string) => void;
  setResults: (results: SearchResult[], intent: QueryIntent, executionTime: number) => void;
  setSearching: (isSearching: boolean) => void;
  setError: (error: string | null) => void;
  setFilters: (filters: Partial<SearchFilters>) => void;
  clearFilters: () => void;
  addToHistory: (query: string, resultCount: number) => void;
  clearHistory: () => void;
  removeFromHistory: (id: string) => void;
  reset: () => void;
}

const MAX_HISTORY_ITEMS = 50;

const initialFilters: SearchFilters = {
  entity_types: undefined,
  source_types: undefined,
  date_range: undefined,
  sources: undefined,
};

export const useSearchStore = create<SearchState>()(
  persist(
    immer((set) => ({
      // Initial state
      query: '',
      results: [],
      isSearching: false,
      error: null,
      intent: null,
      executionTime: null,
      filters: initialFilters,
      history: [],

      // Actions
      setQuery: (query) =>
        set((state) => {
          state.query = query;
        }),

      setResults: (results, intent, executionTime) =>
        set((state) => {
          state.results = results;
          state.intent = intent;
          state.executionTime = executionTime;
          state.isSearching = false;
          state.error = null;
        }),

      setSearching: (isSearching) =>
        set((state) => {
          state.isSearching = isSearching;
          if (isSearching) {
            state.error = null;
          }
        }),

      setError: (error) =>
        set((state) => {
          state.error = error;
          state.isSearching = false;
        }),

      setFilters: (filters) =>
        set((state) => {
          state.filters = { ...state.filters, ...filters };
        }),

      clearFilters: () =>
        set((state) => {
          state.filters = initialFilters;
        }),

      addToHistory: (query, resultCount) =>
        set((state) => {
          // Don't add duplicates of the same query
          const exists = state.history.some(
            (h) => h.query.toLowerCase() === query.toLowerCase()
          );
          if (exists) {
            // Move to top
            state.history = [
              { id: crypto.randomUUID(), query, timestamp: new Date().toISOString(), resultCount },
              ...state.history.filter(
                (h) => h.query.toLowerCase() !== query.toLowerCase()
              ),
            ].slice(0, MAX_HISTORY_ITEMS);
          } else {
            state.history = [
              { id: crypto.randomUUID(), query, timestamp: new Date().toISOString(), resultCount },
              ...state.history,
            ].slice(0, MAX_HISTORY_ITEMS);
          }
        }),

      clearHistory: () =>
        set((state) => {
          state.history = [];
        }),

      removeFromHistory: (id) =>
        set((state) => {
          state.history = state.history.filter((h) => h.id !== id);
        }),

      reset: () =>
        set((state) => {
          state.query = '';
          state.results = [];
          state.isSearching = false;
          state.error = null;
          state.intent = null;
          state.executionTime = null;
          state.filters = initialFilters;
        }),
    })),
    {
      name: 'futurnal-search',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        history: state.history,
        filters: state.filters,
      }),
    }
  )
);

// Selectors
export const useSearchQuery = () => useSearchStore((state) => state.query);
export const useSearchResults = () => useSearchStore((state) => state.results);
export const useSearchHistory = () => useSearchStore((state) => state.history);
export const useSearchFilters = () => useSearchStore((state) => state.filters);
export const useIsSearching = () => useSearchStore((state) => state.isSearching);
```

### User Store

```typescript
// src/stores/userStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User } from 'firebase/auth';

type SubscriptionTier = 'free' | 'pro';

interface UserState {
  // Auth state
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Subscription
  tier: SubscriptionTier;
  subscriptionId: string | null;
  subscriptionEndsAt: string | null;

  // Actions
  setUser: (user: User | null) => void;
  setLoading: (isLoading: boolean) => void;
  setSubscription: (tier: SubscriptionTier, subscriptionId?: string, endsAt?: string) => void;
  logout: () => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      // Initial state
      user: null,
      isAuthenticated: false,
      isLoading: true,
      tier: 'free',
      subscriptionId: null,
      subscriptionEndsAt: null,

      // Actions
      setUser: (user) =>
        set({
          user,
          isAuthenticated: !!user,
          isLoading: false,
        }),

      setLoading: (isLoading) => set({ isLoading }),

      setSubscription: (tier, subscriptionId, endsAt) =>
        set({
          tier,
          subscriptionId: subscriptionId ?? null,
          subscriptionEndsAt: endsAt ?? null,
        }),

      logout: () =>
        set({
          user: null,
          isAuthenticated: false,
          tier: 'free',
          subscriptionId: null,
          subscriptionEndsAt: null,
        }),
    }),
    {
      name: 'futurnal-user',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        tier: state.tier,
        subscriptionId: state.subscriptionId,
        subscriptionEndsAt: state.subscriptionEndsAt,
      }),
    }
  )
);

// Selectors
export const useUser = () => useUserStore((state) => state.user);
export const useIsAuthenticated = () => useUserStore((state) => state.isAuthenticated);
export const useTier = () => useUserStore((state) => state.tier);
export const useIsPro = () => useUserStore((state) => state.tier === 'pro');
```

### Settings Store

```typescript
// src/stores/settingsStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface SettingsState {
  // Appearance
  theme: 'dark' | 'light' | 'system';
  accentColor: string;
  fontSize: 'small' | 'medium' | 'large';

  // Search preferences
  defaultSearchMode: 'hybrid' | 'semantic' | 'keyword';
  resultsPerPage: number;
  showConfidenceScores: boolean;
  showProvenance: boolean;

  // Privacy
  telemetryEnabled: boolean;
  crashReportsEnabled: boolean;
  analyticsEnabled: boolean;

  // Connectors
  autoSyncEnabled: boolean;
  syncIntervalMinutes: number;
  notifyOnSyncComplete: boolean;

  // Graph
  graphNodeLimit: number;
  graphAnimations: boolean;

  // Actions
  setSetting: <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => void;
  resetSettings: () => void;
}

const defaultSettings: Omit<SettingsState, 'setSetting' | 'resetSettings'> = {
  // Appearance
  theme: 'dark',
  accentColor: '#3B82F6',
  fontSize: 'medium',

  // Search
  defaultSearchMode: 'hybrid',
  resultsPerPage: 20,
  showConfidenceScores: true,
  showProvenance: false,

  // Privacy
  telemetryEnabled: false,
  crashReportsEnabled: true,
  analyticsEnabled: false,

  // Connectors
  autoSyncEnabled: true,
  syncIntervalMinutes: 60,
  notifyOnSyncComplete: true,

  // Graph
  graphNodeLimit: 1000,
  graphAnimations: true,
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...defaultSettings,

      setSetting: (key, value) => set({ [key]: value }),

      resetSettings: () => set(defaultSettings),
    }),
    {
      name: 'futurnal-settings',
      storage: createJSONStorage(() => localStorage),
    }
  )
);

// Convenience hooks
export const useTheme = () => useSettingsStore((state) => state.theme);
export const useSearchPreferences = () =>
  useSettingsStore((state) => ({
    defaultSearchMode: state.defaultSearchMode,
    resultsPerPage: state.resultsPerPage,
    showConfidenceScores: state.showConfidenceScores,
    showProvenance: state.showProvenance,
  }));
export const usePrivacySettings = () =>
  useSettingsStore((state) => ({
    telemetryEnabled: state.telemetryEnabled,
    crashReportsEnabled: state.crashReportsEnabled,
    analyticsEnabled: state.analyticsEnabled,
  }));
```

### UI Store (Ephemeral State)

```typescript
// src/stores/uiStore.ts
import { create } from 'zustand';

interface UIState {
  // Command palette
  isCommandPaletteOpen: boolean;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  toggleCommandPalette: () => void;

  // Sidebar
  isSidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // Modals
  activeModal: string | null;
  modalProps: Record<string, unknown>;
  openModal: (modalId: string, props?: Record<string, unknown>) => void;
  closeModal: () => void;

  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // Graph
  selectedNodeId: string | null;
  setSelectedNode: (id: string | null) => void;
  graphFullscreen: boolean;
  toggleGraphFullscreen: () => void;
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message?: string;
  duration?: number;
}

export const useUIStore = create<UIState>((set) => ({
  // Command palette
  isCommandPaletteOpen: false,
  openCommandPalette: () => set({ isCommandPaletteOpen: true }),
  closeCommandPalette: () => set({ isCommandPaletteOpen: false }),
  toggleCommandPalette: () =>
    set((state) => ({ isCommandPaletteOpen: !state.isCommandPaletteOpen })),

  // Sidebar
  isSidebarCollapsed: false,
  toggleSidebar: () =>
    set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),

  // Modals
  activeModal: null,
  modalProps: {},
  openModal: (modalId, props = {}) =>
    set({ activeModal: modalId, modalProps: props }),
  closeModal: () => set({ activeModal: null, modalProps: {} }),

  // Notifications
  notifications: [],
  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: crypto.randomUUID() },
      ],
    })),
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
  clearNotifications: () => set({ notifications: [] }),

  // Graph
  selectedNodeId: null,
  setSelectedNode: (id) => set({ selectedNodeId: id }),
  graphFullscreen: false,
  toggleGraphFullscreen: () =>
    set((state) => ({ graphFullscreen: !state.graphFullscreen })),
}));

// Convenience hooks
export const useCommandPalette = () => ({
  isOpen: useUIStore((state) => state.isCommandPaletteOpen),
  open: useUIStore((state) => state.openCommandPalette),
  close: useUIStore((state) => state.closeCommandPalette),
  toggle: useUIStore((state) => state.toggleCommandPalette),
});

export const useNotifications = () => ({
  notifications: useUIStore((state) => state.notifications),
  add: useUIStore((state) => state.addNotification),
  remove: useUIStore((state) => state.removeNotification),
  clear: useUIStore((state) => state.clearNotifications),
});
```

### TanStack Query Configuration

```typescript
// src/lib/queryClient.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 30, // 30 minutes (formerly cacheTime)
      retry: 1,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 0,
    },
  },
});

// Query keys factory
export const queryKeys = {
  // Search
  search: (query: string, filters?: object) => ['search', query, filters] as const,
  searchHistory: ['searchHistory'] as const,

  // Connectors
  connectors: ['connectors'] as const,
  connector: (id: string) => ['connectors', id] as const,

  // Privacy
  consent: (sourceId?: string) => ['consent', sourceId] as const,
  auditLogs: (query?: object) => ['auditLogs', query] as const,

  // Orchestrator
  orchestratorStatus: ['orchestratorStatus'] as const,

  // Graph
  knowledgeGraph: (limit?: number) => ['knowledgeGraph', limit] as const,

  // User
  subscription: ['subscription'] as const,
};
```

### Persistence with Tauri Store

```typescript
// src/lib/storage.ts
import { Store } from '@tauri-apps/plugin-store';

let store: Store | null = null;

async function getStore(): Promise<Store> {
  if (!store) {
    store = await Store.load('futurnal-store.json');
  }
  return store;
}

export const tauriStorage = {
  async getItem(key: string): Promise<string | null> {
    const store = await getStore();
    const value = await store.get<string>(key);
    return value ?? null;
  },

  async setItem(key: string, value: string): Promise<void> {
    const store = await getStore();
    await store.set(key, value);
    await store.save();
  },

  async removeItem(key: string): Promise<void> {
    const store = await getStore();
    await store.delete(key);
    await store.save();
  },
};

// For Zustand persist middleware
export const createTauriStorage = () => ({
  getItem: tauriStorage.getItem,
  setItem: tauriStorage.setItem,
  removeItem: tauriStorage.removeItem,
});
```

### Secure Storage for Sensitive Data

```typescript
// src/lib/secureStorage.ts
import { Store } from '@tauri-apps/plugin-store';

// Separate store for sensitive data with encryption
let secureStore: Store | null = null;

async function getSecureStore(): Promise<Store> {
  if (!secureStore) {
    // This store should be encrypted at rest
    secureStore = await Store.load('futurnal-secure.json', {
      // Encryption options would go here if supported
    });
  }
  return secureStore;
}

export const secureStorage = {
  async saveToken(key: string, token: string): Promise<void> {
    const store = await getSecureStore();
    // In production, consider additional encryption
    await store.set(key, token);
    await store.save();
  },

  async getToken(key: string): Promise<string | null> {
    const store = await getSecureStore();
    const value = await store.get<string>(key);
    return value ?? null;
  },

  async deleteToken(key: string): Promise<void> {
    const store = await getSecureStore();
    await store.delete(key);
    await store.save();
  },

  async clearAll(): Promise<void> {
    const store = await getSecureStore();
    await store.clear();
    await store.save();
  },
};
```

### Store Provider

```tsx
// src/providers/StoreProvider.tsx
import { ReactNode, useEffect } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { queryClient } from '@/lib/queryClient';
import { useUserStore } from '@/stores/userStore';
import { onAuthStateChanged } from 'firebase/auth';
import { auth } from '@/lib/firebase';

interface StoreProviderProps {
  children: ReactNode;
}

export function StoreProvider({ children }: StoreProviderProps) {
  const setUser = useUserStore((state) => state.setUser);
  const setLoading = useUserStore((state) => state.setLoading);

  // Sync Firebase auth state with store
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [setUser, setLoading]);

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  );
}
```

### DevTools Integration

```typescript
// src/lib/devtools.ts
import { mountStoreDevtool } from 'simple-zustand-devtools';
import { useSearchStore } from '@/stores/searchStore';
import { useUserStore } from '@/stores/userStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useUIStore } from '@/stores/uiStore';

// Mount devtools in development
if (import.meta.env.DEV) {
  mountStoreDevtool('SearchStore', useSearchStore);
  mountStoreDevtool('UserStore', useUserStore);
  mountStoreDevtool('SettingsStore', useSettingsStore);
  mountStoreDevtool('UIStore', useUIStore);
}
```

## Acceptance Criteria

- [ ] Search state persists query history
- [ ] User subscription tier persists across sessions
- [ ] Settings persist and apply correctly
- [ ] UI state resets appropriately on navigation
- [ ] TanStack Query caches API responses
- [ ] Polling updates connector status
- [ ] DevTools accessible in development
- [ ] Store subscriptions trigger re-renders correctly
- [ ] Secure storage works for tokens

## Test Plan

### Unit Tests
```typescript
describe('useSearchStore', () => {
  beforeEach(() => {
    useSearchStore.setState({ history: [] });
  });

  it('should add query to history', () => {
    const { addToHistory } = useSearchStore.getState();
    addToHistory('test query', 10);

    const { history } = useSearchStore.getState();
    expect(history).toHaveLength(1);
    expect(history[0].query).toBe('test query');
  });

  it('should limit history to 50 items', () => {
    const { addToHistory } = useSearchStore.getState();

    for (let i = 0; i < 60; i++) {
      addToHistory(`query ${i}`, i);
    }

    const { history } = useSearchStore.getState();
    expect(history).toHaveLength(50);
  });

  it('should move duplicate queries to top', () => {
    const { addToHistory } = useSearchStore.getState();
    addToHistory('first', 1);
    addToHistory('second', 2);
    addToHistory('first', 3);

    const { history } = useSearchStore.getState();
    expect(history[0].query).toBe('first');
    expect(history).toHaveLength(2);
  });
});
```

### Integration Tests
```typescript
describe('Store Persistence', () => {
  it('should persist settings to localStorage', () => {
    const { setSetting } = useSettingsStore.getState();
    setSetting('theme', 'light');

    // Simulate page reload
    const stored = JSON.parse(localStorage.getItem('futurnal-settings') || '{}');
    expect(stored.state.theme).toBe('light');
  });
});
```

## Dependencies

- zustand (with immer, persist middlewares)
- @tanstack/react-query
- @tanstack/react-query-devtools
- @tauri-apps/plugin-store
- simple-zustand-devtools

## Next Steps

After state management complete:
1. Add optimistic updates for mutations
2. Implement offline mode with queue
3. Add state migration for schema changes
4. Create state snapshot/restore for debugging

**This state management layer provides a reactive, persistent, and efficient foundation for the entire application.**

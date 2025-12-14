/**
 * Search Store - Zustand state management for search functionality
 *
 * Manages search query state, results, filters, and recent searches.
 * Uses persist middleware for localStorage persistence of recent searches.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { searchApi } from '@/lib/api';
import type { SearchResult, SearchFilters } from '@/types/api';

/**
 * Filter chip for UI display
 */
export interface SearchFilter {
  type: 'intent' | 'entityType' | 'sourceType' | 'timeRange';
  value: string;
  label: string;
}

/**
 * Intent type for classification
 */
export type IntentType = 'temporal' | 'causal' | 'exploratory' | 'lookup';

/**
 * Search history item with metadata
 */
export interface SearchHistoryItem {
  /** Unique ID for the history entry */
  id: string;
  /** The search query */
  query: string;
  /** When the search was performed */
  timestamp: string;
  /** Number of results returned */
  resultCount: number;
  /** Detected intent (if any) */
  intent?: IntentType | null;
  /** Execution time in milliseconds */
  executionTimeMs?: number;
}

interface SearchState {
  /** Current query string */
  query: string;
  /** Search results */
  results: SearchResult[];
  /** Loading state */
  isLoading: boolean;
  /** Detected query intent */
  intent: IntentType | null;
  /** Intent classification confidence (0-1) */
  intentConfidence: number;
  /** Active filter chips */
  filters: SearchFilter[];
  /** Recent search history with metadata (persisted) */
  searchHistory: SearchHistoryItem[];
  /** Error message if search failed */
  error: string | null;
  /** Currently selected result ID for detail panel */
  selectedResultId: string | null;

  /** Set the query string */
  setQuery: (query: string) => void;
  /** Set active filters */
  setFilters: (filters: SearchFilter[]) => void;
  /** Add a single filter */
  addFilter: (filter: SearchFilter) => void;
  /** Remove a filter */
  removeFilter: (filter: SearchFilter) => void;
  /** Add a search to history with metadata */
  addSearchToHistory: (item: Omit<SearchHistoryItem, 'id'>) => void;
  /** Remove a search from history by ID */
  removeSearchFromHistory: (id: string) => void;
  /** Clear all search history */
  clearSearchHistory: () => void;
  /** Get recent queries (for backwards compatibility) */
  getRecentQueries: () => string[];
  /** Execute the search */
  executeSearch: () => Promise<void>;
  /** Clear results and reset state */
  clearResults: () => void;
  /** Set selected result ID */
  setSelectedResult: (id: string | null) => void;
}

const MAX_SEARCH_HISTORY = 50;

/**
 * Generate unique ID for search history entries
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

const initialState = {
  query: '',
  results: [],
  isLoading: false,
  intent: null as IntentType | null,
  intentConfidence: 0,
  filters: [] as SearchFilter[],
  searchHistory: [] as SearchHistoryItem[],
  error: null as string | null,
  selectedResultId: null as string | null,
};

/**
 * Convert SearchFilter[] to SearchFilters for API
 */
function filtersToApiFormat(filters: SearchFilter[]): SearchFilters {
  const apiFilters: SearchFilters = {};

  for (const filter of filters) {
    if (filter.type === 'entityType') {
      if (!apiFilters.entity_types) apiFilters.entity_types = [];
      apiFilters.entity_types.push(filter.value);
    } else if (filter.type === 'sourceType') {
      if (!apiFilters.source_types) apiFilters.source_types = [];
      apiFilters.source_types.push(filter.value);
    }
    // Note: timeRange and intent filters are handled differently
  }

  return apiFilters;
}

export const useSearchStore = create<SearchState>()(
  persist(
    (set, get) => ({
      ...initialState,

      setQuery: (query) => set({ query }),

      setFilters: (filters) => set({ filters }),

      addFilter: (filter) => {
        const { filters } = get();
        // Prevent duplicates
        const exists = filters.some(
          (f) => f.type === filter.type && f.value === filter.value
        );
        if (!exists) {
          set({ filters: [...filters, filter] });
        }
      },

      removeFilter: (filter) => {
        const { filters } = get();
        set({
          filters: filters.filter(
            (f) => !(f.type === filter.type && f.value === filter.value)
          ),
        });
      },

      addSearchToHistory: (item) => {
        const { searchHistory } = get();
        // Create new entry with generated ID
        const newEntry: SearchHistoryItem = {
          ...item,
          id: generateId(),
        };
        // Remove duplicates of the same query, add new one at front
        const filtered = searchHistory.filter((h) => h.query !== item.query);
        const updated = [newEntry, ...filtered].slice(0, MAX_SEARCH_HISTORY);
        set({ searchHistory: updated });
      },

      removeSearchFromHistory: (id) => {
        const { searchHistory } = get();
        set({
          searchHistory: searchHistory.filter((h) => h.id !== id),
        });
      },

      clearSearchHistory: () => set({ searchHistory: [] }),

      getRecentQueries: () => {
        const { searchHistory } = get();
        return searchHistory.map((h) => h.query);
      },

      executeSearch: async () => {
        const { query, filters, addSearchToHistory } = get();
        const trimmedQuery = query.trim();

        if (!trimmedQuery) return;

        set({ isLoading: true, error: null });
        const startTime = Date.now();

        try {
          const response = await searchApi.search({
            query: trimmedQuery,
            top_k: 20,
            filters: filtersToApiFormat(filters),
          });

          // Debug logging for search results
          console.log('[SearchStore] Search response:', {
            query: trimmedQuery,
            resultCount: response.results.length,
            intent: response.intent,
            executionTimeMs: response.execution_time_ms,
          });
          if (response.results.length > 0) {
            console.log('[SearchStore] Sample result structure:', {
              id: response.results[0].id,
              entity_type: response.results[0].entity_type,
              source_type: response.results[0].source_type,
              score: response.results[0].score,
              contentPreview: response.results[0].content?.substring(0, 100),
              metadataKeys: Object.keys(response.results[0].metadata || {}),
            });
          }

          const executionTimeMs = Date.now() - startTime;

          // Extract intent from response
          const intentPrimary = response.intent?.primary ?? null;

          // Add to search history with full metadata
          addSearchToHistory({
            query: trimmedQuery,
            timestamp: new Date().toISOString(),
            resultCount: response.results.length,
            intent: intentPrimary,
            executionTimeMs,
          });

          set({
            results: response.results,
            intent: intentPrimary,
            intentConfidence: 0.85, // Default confidence since API doesn't provide it
            isLoading: false,
          });
        } catch (error) {
          console.error('Search error:', error);
          set({
            results: [],
            intent: null,
            intentConfidence: 0,
            isLoading: false,
            error: error instanceof Error ? error.message : 'Search failed',
          });
        }
      },

      clearResults: () =>
        set({
          results: [],
          intent: null,
          intentConfidence: 0,
          error: null,
          selectedResultId: null,
        }),

      setSelectedResult: (id) => set({ selectedResultId: id }),
    }),
    {
      name: 'futurnal-search',
      // Only persist searchHistory, not query/results/filters
      partialize: (state) => ({ searchHistory: state.searchHistory }),
    }
  )
);

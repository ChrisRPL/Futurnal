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
  /** Recent search queries (persisted) */
  recentSearches: string[];
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
  /** Add a query to recent searches */
  addRecentSearch: (query: string) => void;
  /** Clear all recent searches */
  clearRecentSearches: () => void;
  /** Execute the search */
  executeSearch: () => Promise<void>;
  /** Clear results and reset state */
  clearResults: () => void;
  /** Set selected result ID */
  setSelectedResult: (id: string | null) => void;
}

const MAX_RECENT_SEARCHES = 50;

const initialState = {
  query: '',
  results: [],
  isLoading: false,
  intent: null as IntentType | null,
  intentConfidence: 0,
  filters: [] as SearchFilter[],
  recentSearches: [] as string[],
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

      addRecentSearch: (query) => {
        const { recentSearches } = get();
        // Remove if exists, add to front
        const filtered = recentSearches.filter((q) => q !== query);
        const updated = [query, ...filtered].slice(0, MAX_RECENT_SEARCHES);
        set({ recentSearches: updated });
      },

      clearRecentSearches: () => set({ recentSearches: [] }),

      executeSearch: async () => {
        const { query, filters, addRecentSearch } = get();
        const trimmedQuery = query.trim();

        if (!trimmedQuery) return;

        set({ isLoading: true, error: null });

        try {
          const response = await searchApi.search({
            query: trimmedQuery,
            top_k: 20,
            filters: filtersToApiFormat(filters),
          });

          // Add to recent searches
          addRecentSearch(trimmedQuery);

          // Extract intent from response
          const intentPrimary = response.intent?.primary ?? null;

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
      // Only persist recentSearches, not query/results/filters
      partialize: (state) => ({ recentSearches: state.recentSearches }),
    }
  )
);

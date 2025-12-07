Summary: Implement OmniCommandPalette search interface with natural language queries, filters, and keyboard navigation.

# 05 · Command Palette & Search Interface

## Purpose

Implement the primary search interface—the window through which users query their Ghost's understanding of their personal universe. Using an adapted OmniCommandPalette from 21st.dev, this interface enables natural language queries that leverage the Ghost's evolving comprehension of the user's experiential data.

> **The Core Interaction**: Every query is a conversation with your Ghost, asking it to search not just *what* you know, but to reveal the connections and patterns it has discovered in your stream of experience.

This module features natural language query input, filter chips for experiential refinement, query history, intent classification display, and comprehensive keyboard navigation.

**Criticality**: CRITICAL - Primary user interaction for search

## Scope

- OmniCommandPalette component adaptation from 21st.dev
- ⌘K / Ctrl+K global keyboard trigger
- Natural language query input
- Filter chips: temporal, causal, entity type, source type
- Query history (recent searches with max 50 entries)
- Intent classification display (temporal/causal/exploratory/lookup)
- Loading states with skeleton UI
- Fuzzy search for commands and navigation
- Search result integration with IPC layer

## Requirements Alignment

- **Feature Requirement**: "Search input with command palette behavior, supporting natural language and filters"
- **UX Requirement**: "Keyboard-centric navigation"
- **Performance**: "Sub-second feedback" for search queries

## Component Design

### Command Palette Component

```tsx
// src/components/search/CommandPalette.tsx
import * as React from 'react';
import { useCallback, useEffect, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { invoke } from '@tauri-apps/api/core';
import { useSearchStore } from '@/stores/searchStore';
import { SearchInput } from './SearchInput';
import { FilterChips } from './FilterChips';
import { SearchResults } from './SearchResults';
import { RecentSearches } from './RecentSearches';
import { IntentBadge } from './IntentBadge';
import { Loader2, Search, X, Command } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CommandPaletteProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function CommandPalette({ open: controlledOpen, onOpenChange }: CommandPaletteProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;

  const {
    query,
    setQuery,
    results,
    isLoading,
    intent,
    filters,
    setFilters,
    recentSearches,
    addRecentSearch,
    executeSearch,
  } = useSearchStore();

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen(!open);
      }
      if (e.key === 'Escape' && open) {
        setOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, setOpen]);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    addRecentSearch(query);
    await executeSearch();
  }, [query, addRecentSearch, executeSearch]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const handleRecentSelect = (recentQuery: string) => {
    setQuery(recentQuery);
    handleSearch();
  };

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm data-[state=open]:animate-fade-in" />
        <Dialog.Content
          className={cn(
            'fixed z-50 left-1/2 top-[15%] -translate-x-1/2',
            'w-[min(640px,calc(100%-2rem))]',
            'rounded-xl border border-border bg-background-surface shadow-modal',
            'data-[state=open]:animate-scale-in',
            'outline-none'
          )}
        >
          {/* Header */}
          <div className="flex items-center gap-3 border-b border-border p-4">
            <Search className="h-5 w-5 text-text-tertiary" />
            <SearchInput
              value={query}
              onChange={setQuery}
              onKeyDown={handleKeyDown}
              placeholder="Ask your Ghost anything..."
              autoFocus
            />
            {isLoading && <Loader2 className="h-4 w-4 animate-spin text-text-tertiary" />}
            {intent && <IntentBadge intent={intent} />}
            <Dialog.Close asChild>
              <button className="rounded p-1 text-text-tertiary hover:bg-background-elevated hover:text-text-secondary">
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </div>

          {/* Filters */}
          <div className="border-b border-border px-4 py-2">
            <FilterChips filters={filters} onChange={setFilters} />
          </div>

          {/* Content */}
          <div className="max-h-[60vh] overflow-y-auto p-2">
            {!query && recentSearches.length > 0 && (
              <RecentSearches
                searches={recentSearches}
                onSelect={handleRecentSelect}
              />
            )}

            {query && results.length > 0 && (
              <SearchResults results={results} onSelect={() => setOpen(false)} />
            )}

            {query && !isLoading && results.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-text-tertiary">
                <Search className="h-8 w-8 mb-3 opacity-50" />
                <p className="text-sm">Your Ghost found nothing yet</p>
                <p className="text-xs mt-1">Try different keywords, or feed your Ghost more experiential data</p>
              </div>
            )}

            {isLoading && (
              <div className="space-y-2 p-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="animate-pulse">
                    <div className="h-16 rounded-lg bg-background-elevated" />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-text-tertiary">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="rounded bg-background-elevated px-1.5 py-0.5">↵</kbd>
                to search
              </span>
              <span className="flex items-center gap-1">
                <kbd className="rounded bg-background-elevated px-1.5 py-0.5">↑</kbd>
                <kbd className="rounded bg-background-elevated px-1.5 py-0.5">↓</kbd>
                to navigate
              </span>
            </div>
            <span className="flex items-center gap-1">
              <kbd className="rounded bg-background-elevated px-1.5 py-0.5">esc</kbd>
              to close
            </span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
```

### Search Input Component

```tsx
// src/components/search/SearchInput.tsx
import * as React from 'react';
import { cn } from '@/lib/utils';

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
  placeholder?: string;
  autoFocus?: boolean;
}

export function SearchInput({
  value,
  onChange,
  onKeyDown,
  placeholder = 'Search...',
  autoFocus = false,
}: SearchInputProps) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      autoFocus={autoFocus}
      className={cn(
        'flex-1 bg-transparent text-text-primary text-sm',
        'placeholder:text-text-tertiary',
        'outline-none'
      )}
    />
  );
}
```

### Filter Chips Component

```tsx
// src/components/search/FilterChips.tsx
import { Badge } from '@/components/ui/badge';
import { X, Clock, GitBranch, FileText, Mic, Image, Code } from 'lucide-react';

export type SearchFilter = {
  type: 'intent' | 'entityType' | 'sourceType' | 'timeRange';
  value: string;
  label: string;
};

interface FilterChipsProps {
  filters: SearchFilter[];
  onChange: (filters: SearchFilter[]) => void;
}

const INTENT_OPTIONS = [
  { value: 'temporal', label: 'Temporal', icon: Clock },
  { value: 'causal', label: 'Causal', icon: GitBranch },
  { value: 'exploratory', label: 'Exploratory', icon: null },
  { value: 'lookup', label: 'Lookup', icon: null },
];

const ENTITY_TYPE_OPTIONS = [
  { value: 'Event', label: 'Events' },
  { value: 'Document', label: 'Documents' },
  { value: 'Person', label: 'People' },
  { value: 'Code', label: 'Code' },
];

const SOURCE_TYPE_OPTIONS = [
  { value: 'text', label: 'Text', icon: FileText },
  { value: 'ocr', label: 'OCR', icon: Image },
  { value: 'audio', label: 'Audio', icon: Mic },
  { value: 'code', label: 'Code', icon: Code },
];

export function FilterChips({ filters, onChange }: FilterChipsProps) {
  const addFilter = (filter: SearchFilter) => {
    if (!filters.find((f) => f.type === filter.type && f.value === filter.value)) {
      onChange([...filters, filter]);
    }
  };

  const removeFilter = (filter: SearchFilter) => {
    onChange(filters.filter((f) => !(f.type === filter.type && f.value === filter.value)));
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      {/* Active filters */}
      {filters.map((filter) => (
        <Badge
          key={`${filter.type}-${filter.value}`}
          variant="default"
          className="cursor-pointer gap-1"
          onClick={() => removeFilter(filter)}
        >
          {filter.label}
          <X className="h-3 w-3" />
        </Badge>
      ))}

      {/* Quick filter buttons */}
      {filters.length === 0 && (
        <div className="flex gap-1.5 text-xs text-text-tertiary">
          <span>Quick filters:</span>
          {INTENT_OPTIONS.slice(0, 2).map((opt) => (
            <button
              key={opt.value}
              onClick={() =>
                addFilter({ type: 'intent', value: opt.value, label: opt.label })
              }
              className="rounded bg-background-elevated px-2 py-0.5 hover:bg-border hover:text-text-secondary transition-colors"
            >
              {opt.label}
            </button>
          ))}
          {ENTITY_TYPE_OPTIONS.slice(0, 2).map((opt) => (
            <button
              key={opt.value}
              onClick={() =>
                addFilter({ type: 'entityType', value: opt.value, label: opt.label })
              }
              className="rounded bg-background-elevated px-2 py-0.5 hover:bg-border hover:text-text-secondary transition-colors"
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
```

### Intent Badge Component

```tsx
// src/components/search/IntentBadge.tsx
import { Badge } from '@/components/ui/badge';
import { Clock, GitBranch, Compass, Search } from 'lucide-react';

type Intent = 'temporal' | 'causal' | 'exploratory' | 'lookup';

interface IntentBadgeProps {
  intent: Intent;
  confidence?: number;
}

const INTENT_CONFIG: Record<Intent, { label: string; icon: typeof Clock; variant: 'default' | 'accent' | 'secondary' }> = {
  temporal: { label: 'Temporal', icon: Clock, variant: 'default' },
  causal: { label: 'Causal', icon: GitBranch, variant: 'accent' },
  exploratory: { label: 'Exploratory', icon: Compass, variant: 'secondary' },
  lookup: { label: 'Lookup', icon: Search, variant: 'secondary' },
};

export function IntentBadge({ intent, confidence }: IntentBadgeProps) {
  const config = INTENT_CONFIG[intent];
  const Icon = config.icon;

  return (
    <Badge variant={config.variant} className="gap-1 text-xs">
      <Icon className="h-3 w-3" />
      {config.label}
      {confidence && <span className="opacity-60">({Math.round(confidence * 100)}%)</span>}
    </Badge>
  );
}
```

### Search Store

```typescript
// src/stores/searchStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import type { SearchFilter } from '@/components/search/FilterChips';

interface SearchResult {
  id: string;
  content: string;
  score: number;
  confidence: number;
  timestamp?: string;
  entityType?: string;
  sourceType?: string;
  metadata: Record<string, unknown>;
}

interface SearchState {
  query: string;
  results: SearchResult[];
  isLoading: boolean;
  intent: 'temporal' | 'causal' | 'exploratory' | 'lookup' | null;
  intentConfidence: number;
  filters: SearchFilter[];
  recentSearches: string[];

  setQuery: (query: string) => void;
  setFilters: (filters: SearchFilter[]) => void;
  addRecentSearch: (query: string) => void;
  clearRecentSearches: () => void;
  executeSearch: () => Promise<void>;
}

const MAX_RECENT_SEARCHES = 50;

export const useSearchStore = create<SearchState>()(
  persist(
    (set, get) => ({
      query: '',
      results: [],
      isLoading: false,
      intent: null,
      intentConfidence: 0,
      filters: [],
      recentSearches: [],

      setQuery: (query) => set({ query }),
      setFilters: (filters) => set({ filters }),

      addRecentSearch: (query) => {
        const { recentSearches } = get();
        const filtered = recentSearches.filter((q) => q !== query);
        const updated = [query, ...filtered].slice(0, MAX_RECENT_SEARCHES);
        set({ recentSearches: updated });
      },

      clearRecentSearches: () => set({ recentSearches: [] }),

      executeSearch: async () => {
        const { query, filters } = get();
        if (!query.trim()) return;

        set({ isLoading: true });

        try {
          const response = await invoke<{
            results: SearchResult[];
            intent: string;
            intentConfidence: number;
          }>('search_query', {
            query,
            topK: 20,
            filters: filters.reduce(
              (acc, f) => ({ ...acc, [f.type]: f.value }),
              {}
            ),
          });

          set({
            results: response.results,
            intent: response.intent as SearchState['intent'],
            intentConfidence: response.intentConfidence,
            isLoading: false,
          });
        } catch (error) {
          console.error('Search error:', error);
          set({ results: [], isLoading: false });
        }
      },
    }),
    {
      name: 'futurnal-search',
      partialize: (state) => ({ recentSearches: state.recentSearches }),
    }
  )
);
```

### Global Search Trigger Button

```tsx
// src/components/search/SearchTrigger.tsx
import { Search, Command } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SearchTriggerProps {
  onClick: () => void;
}

export function SearchTrigger({ onClick }: SearchTriggerProps) {
  return (
    <Button
      variant="outline"
      onClick={onClick}
      className="w-64 justify-between text-text-tertiary hover:text-text-secondary"
    >
      <div className="flex items-center gap-2">
        <Search className="h-4 w-4" />
        <span>Ask your Ghost...</span>
      </div>
      <kbd className="flex items-center gap-0.5 rounded bg-background-elevated px-1.5 py-0.5 text-xs">
        <Command className="h-3 w-3" />K
      </kbd>
    </Button>
  );
}
```

## Acceptance Criteria

- [ ] Command palette opens with ⌘K / Ctrl+K
- [ ] Command palette closes with Escape
- [ ] Natural language query input works
- [ ] Filter chips add/remove correctly
- [ ] Query history persists across sessions
- [ ] Maximum 50 recent searches stored
- [ ] Intent classification badge displays
- [ ] Loading skeleton shows during search
- [ ] Results display after search
- [ ] Keyboard navigation works (↑↓ arrows)
- [ ] Enter key triggers search
- [ ] Empty state displays when no results

## Test Plan

### Unit Tests
```typescript
describe('SearchStore', () => {
  it('should add recent search', () => {
    const { addRecentSearch, recentSearches } = useSearchStore.getState();
    addRecentSearch('test query');
    expect(recentSearches).toContain('test query');
  });

  it('should limit recent searches to 50', () => {
    // Add 60 searches
    // Verify only 50 are stored
  });
});
```

### E2E Tests
```typescript
test('search flow', async ({ page }) => {
  await page.keyboard.press('Meta+k');
  await expect(page.locator('[role="dialog"]')).toBeVisible();
  await page.fill('input[placeholder*="Search"]', 'meeting notes');
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
});
```

## Dependencies

- @radix-ui/react-dialog
- @tauri-apps/api
- Zustand

## Next Steps

After command palette complete:
1. Proceed to Module 06 (Results & Provenance View)
2. Integrate with IPC layer for real search
3. Add advanced filter options

---

## UI Copy Reference

| Element | Copy | Purpose |
|---------|------|---------|
| Input placeholder | "Ask your Ghost anything..." | Frames interaction as conversation |
| Empty state | "Your Ghost found nothing yet" | Personalizes the AI |
| Empty state hint | "Try different keywords, or feed your Ghost more experiential data" | Guides toward connectors |
| Search trigger | "Ask your Ghost..." | Consistent terminology |
| Filter labels | "Temporal", "Causal", "Exploratory" | Maps to intent classification |

### Query Examples (Tooltips/Hints)

- *"What did I write about machine learning last month?"* (Temporal + Entity)
- *"Show me all interactions with Sarah"* (Entity lookup)
- *"How does my sleep affect my productivity?"* (Causal exploration - Phase 2+)
- *"What patterns exist in my project notes?"* (Exploratory analysis)

---

**This command palette provides the primary search interaction—the conversation with your Ghost about your personal universe.**

/**
 * CommandPaletteOverlay - Standalone overlay window for global command palette
 *
 * This renders the command palette with glassmorphism design in a transparent
 * frameless window. Triggered by global hotkey (Cmd+Shift+Space).
 *
 * On macOS, this runs in an NSPanel which doesn't steal focus from other apps
 * (Spotlight-like behavior using tauri-nspanel).
 *
 * Features:
 * - Glassmorphism design with frosted glass effect (backdrop-filter blur)
 * - Auto-dismiss on Escape or click outside
 * - Truly transparent background showing desktop behind
 * - Rounded corners with glass border effect
 */

import { useCallback, useEffect, useState, useLayoutEffect } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { Search, Loader2, X, Sparkles, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useSearchStore } from '@/stores/searchStore';
import { SearchInput } from '@/components/search/SearchInput';
import { FilterChips } from '@/components/search/FilterChips';
import { RecentSearches } from '@/components/search/RecentSearches';
import { IntentBadge } from '@/components/search/IntentBadge';
import { ModelSelector } from '@/components/search/ModelSelector';
import { StreamingAnswer } from '@/components/search/StreamingAnswer';
import { ResultsList, ResultDetailPanel } from '@/components/results';
import { ANSWER_MODELS } from '@/types/api';
import type { SearchResult } from '@/types/api';
import '@/styles/globals.css';

/**
 * Hook to make the window transparent for glassmorphism effect.
 * Adds 'overlay-transparent' class to HTML element to remove background.
 */
function useTransparentWindow() {
  useLayoutEffect(() => {
    // Add transparent class immediately on mount
    document.documentElement.classList.add('overlay-transparent');
    document.body.style.background = 'transparent';

    return () => {
      // Cleanup on unmount (shouldn't happen in this window, but good practice)
      document.documentElement.classList.remove('overlay-transparent');
    };
  }, []);
}

// Create a query client for this isolated window
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

/**
 * Glassmorphism Command Palette Content
 */
function GlassCommandPalette() {
  // Search store
  const {
    query,
    setQuery,
    results,
    isLoading,
    intent,
    intentConfidence,
    filters,
    setFilters,
    searchHistory,
    executeSearch,
    selectedResultId,
    setSelectedResult,
    answer,
    answerSources,
    generateAnswers,
    toggleAnswerGeneration,
    selectedModel,
  } = useSearchStore();

  // Get the selected result object
  const selectedResult = results.find((r) => r.id === selectedResultId) ?? null;
  const modelLabel = ANSWER_MODELS.find(m => m.id === selectedModel)?.label ?? selectedModel;

  // Keyboard navigation state
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);

  // Calculate total navigable items
  const hasResults = results.length > 0;
  const hasSearchHistory = !query && searchHistory.length > 0;

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(-1);
  }, [results, query]);

  // Auto-collapse results when AI answer is received
  useEffect(() => {
    if (generateAnswers && answer && results.length > 0) {
      setResultsCollapsed(true);
    }
  }, [generateAnswers, answer, results.length]);

  // Reset results collapsed state when starting new search
  useEffect(() => {
    if (isLoading) {
      setResultsCollapsed(false);
    }
  }, [isLoading]);

  // Handle dismiss - hide the overlay window
  const handleDismiss = useCallback(async () => {
    await invoke('hide_command_palette');
  }, []);

  // Listen for window blur (click outside)
  // Fixed race condition: properly track mounted state and cleanup
  useEffect(() => {
    let mounted = true;
    let unlisten: (() => void) | null = null;

    const setupBlurListener = async () => {
      const currentWindow = getCurrentWindow();
      const unlistenFn = await currentWindow.onFocusChanged(({ payload: focused }) => {
        // Only dismiss if still mounted and lost focus
        if (!focused && mounted) {
          handleDismiss();
        }
      });

      // Check if still mounted before storing unlisten
      if (mounted) {
        unlisten = unlistenFn;
      } else {
        // Component unmounted during async setup, cleanup immediately
        unlistenFn();
      }
    };

    setupBlurListener();

    return () => {
      mounted = false;
      if (unlisten) {
        unlisten();
      }
    };
  }, [handleDismiss]);

  // Handle search execution
  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    await executeSearch();
  }, [query, executeSearch]);

  // Handle recent search selection
  const handleRecentSelect = useCallback(
    (recentQuery: string) => {
      setQuery(recentQuery);
      setTimeout(() => {
        useSearchStore.getState().executeSearch();
      }, 0);
    },
    [setQuery]
  );

  // Handle result selection
  const handleResultSelect = useCallback(
    (result: SearchResult) => {
      setSelectedResult(result.id);
    },
    [setSelectedResult]
  );

  // Handle close detail panel
  const handleCloseDetailPanel = useCallback(() => {
    setSelectedResult(null);
  }, [setSelectedResult]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: ReactKeyboardEvent) => {
      const totalItems = hasResults
        ? results.length
        : hasSearchHistory
          ? searchHistory.length
          : 0;

      switch (e.key) {
        case 'Escape':
          e.preventDefault();
          handleDismiss();
          break;

        case 'ArrowDown':
          e.preventDefault();
          if (totalItems > 0) {
            setSelectedIndex((prev) =>
              prev < totalItems - 1 ? prev + 1 : prev
            );
          }
          break;

        case 'ArrowUp':
          e.preventDefault();
          if (totalItems > 0) {
            setSelectedIndex((prev) => (prev > 0 ? prev - 1 : prev));
          }
          break;

        case 'Enter':
          e.preventDefault();
          if (selectedIndex >= 0) {
            if (hasResults && results[selectedIndex]) {
              handleResultSelect(results[selectedIndex]);
            } else if (hasSearchHistory && searchHistory[selectedIndex]) {
              handleRecentSelect(searchHistory[selectedIndex].query);
            }
          } else {
            handleSearch();
          }
          break;
      }
    },
    [
      results,
      searchHistory,
      selectedIndex,
      hasResults,
      hasSearchHistory,
      handleDismiss,
      handleResultSelect,
      handleRecentSelect,
      handleSearch,
    ]
  );

  // Determine what to show
  const showAIAnswer = generateAnswers && (answer || isLoading);
  const showCollapsibleResults = generateAnswers && answer && results.length > 0;

  return (
    <div
      onKeyDown={handleKeyDown}
      className={cn(
        // Glassmorphism container - authentic frosted glass with prominent styling
        // NOTE: overflow-hidden removed - it breaks backdrop-filter on WebKit
        'rounded-3xl',
        // Semi-transparent dark background for glass effect
        'bg-black/40',
        // Strong backdrop blur for frosted glass
        'backdrop-blur-3xl backdrop-saturate-150',
        // Subtle white border for glass edge
        'border border-white/10',
        // Complex shadow for depth and glass feel
        'shadow-2xl shadow-black/50',
        // Inner glow for glass effect
        'ring-1 ring-inset ring-white/5',
        'transition-all duration-300 ease-out',
        // Width based on detail panel
        selectedResult
          ? 'w-[min(960px,calc(100%-2rem))]'
          : 'w-[min(680px,calc(100%-2rem))]'
      )}
      style={{
        // Ensure backdrop filter works with WebKit prefix
        WebkitBackdropFilter: 'blur(40px) saturate(150%)',
        backdropFilter: 'blur(40px) saturate(150%)',
        // Explicit border radius to ensure it renders
        borderRadius: '24px',
      }}
    >
      {/* Header with glass input - rounded top */}
      <div className="flex items-center gap-3 border-b border-white/[0.1] p-5 bg-white/[0.02] rounded-t-[24px]">
        <Search className="h-5 w-5 text-white/50" />
        <SearchInput
          value={query}
          onChange={setQuery}
          onKeyDown={handleKeyDown}
          placeholder={generateAnswers ? "Ask a question..." : "Search your knowledge..."}
          autoFocus
          className="bg-transparent"
        />
        {isLoading && (
          <Loader2 className="h-4 w-4 animate-spin text-white/40" />
        )}
        {intent && !generateAnswers && (
          <IntentBadge intent={intent} confidence={intentConfidence} />
        )}

        {/* AI Mode toggle */}
        <button
          onClick={toggleAnswerGeneration}
          className={cn(
            'flex items-center gap-1.5 rounded-xl px-3 py-2',
            'text-xs font-medium transition-all duration-200',
            generateAnswers
              ? 'bg-white/[0.15] text-white border border-white/25 shadow-[0_0_12px_rgba(255,255,255,0.1)]'
              : 'text-white/50 hover:text-white/70 hover:bg-white/[0.08] border border-transparent'
          )}
          title={generateAnswers ? 'Disable AI answers' : 'Enable AI answers'}
        >
          <Sparkles className="h-3.5 w-3.5" />
          {generateAnswers && <span>AI</span>}
        </button>

        {generateAnswers && <ModelSelector />}

        {/* Close button */}
        <button
          onClick={handleDismiss}
          className={cn(
            'rounded-xl p-2',
            'text-white/40 hover:text-white/70',
            'hover:bg-white/[0.08] transition-all duration-200'
          )}
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Filters - glass style */}
      {!generateAnswers && (
        <div className="border-b border-white/[0.08] px-5 py-3 bg-white/[0.01]">
          <FilterChips filters={filters} onChange={setFilters} />
        </div>
      )}

      {/* Content area */}
      <div className="flex max-h-[55vh]">
        {/* Results area */}
        <div
          className={cn(
            'flex-1 overflow-y-auto p-4',
            selectedResult && 'border-r border-white/[0.08]'
          )}
        >
          {/* AI answer */}
          {showAIAnswer && (
            <StreamingAnswer
              answer={answer || ''}
              sources={answerSources}
              isLoading={isLoading}
              modelName={modelLabel}
            />
          )}

          {/* Collapsible results */}
          {showCollapsibleResults && (
            <div className="mt-4">
              <button
                onClick={() => setResultsCollapsed(!resultsCollapsed)}
                className={cn(
                  'flex items-center gap-2 w-full px-4 py-3 rounded-2xl',
                  'bg-white/[0.06] border border-white/[0.12]',
                  'text-sm text-white/70 hover:text-white/90',
                  'hover:bg-white/[0.1] transition-all duration-200'
                )}
              >
                <FileText className="h-4 w-4" />
                <span>{results.length} supporting documents</span>
                <span className="ml-auto">
                  {resultsCollapsed ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronUp className="h-4 w-4" />
                  )}
                </span>
              </button>

              {!resultsCollapsed && (
                <div className="mt-2">
                  <ResultsList
                    results={results}
                    query={query}
                    selectedId={selectedResultId}
                    onSelect={handleResultSelect}
                    selectedIndex={selectedIndex}
                  />
                </div>
              )}
            </div>
          )}

          {/* Regular results */}
          {!generateAnswers && query && results.length > 0 && (
            <ResultsList
              results={results}
              query={query}
              selectedId={selectedResultId}
              onSelect={handleResultSelect}
              selectedIndex={selectedIndex}
            />
          )}

          {/* Recent searches */}
          {!query && searchHistory.length > 0 && (
            <RecentSearches
              searches={searchHistory}
              onSelect={handleRecentSelect}
              selectedIndex={selectedIndex}
            />
          )}

          {/* Empty state */}
          {query && !isLoading && results.length === 0 && !answer && (
            <div className="flex flex-col items-center justify-center py-16 text-white/60">
              <div className="w-14 h-14 rounded-2xl bg-white/[0.08] flex items-center justify-center mb-4">
                <Search className="h-6 w-6 opacity-60" />
              </div>
              <p className="text-sm font-medium">No results found</p>
              <p className="text-xs mt-2 text-white/40">
                Try different keywords
              </p>
            </div>
          )}

          {/* Loading */}
          {isLoading && !generateAnswers && (
            <div className="space-y-3 p-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse">
                  <div className="h-18 rounded-2xl bg-white/[0.06]" />
                </div>
              ))}
            </div>
          )}

          {/* Initial state */}
          {!query && searchHistory.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center py-16 text-white/60">
              {generateAnswers ? (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-white/[0.08] flex items-center justify-center mb-4">
                    <Sparkles className="h-6 w-6 opacity-60" />
                  </div>
                  <p className="text-sm font-medium">Ask anything about your knowledge</p>
                  <p className="text-xs mt-2 text-white/40">
                    AI will synthesize an answer from your sources
                  </p>
                </>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-white/[0.08] flex items-center justify-center mb-4">
                    <Search className="h-6 w-6 opacity-60" />
                  </div>
                  <p className="text-sm font-medium">Search your personal knowledge</p>
                  <p className="text-xs mt-2 text-white/40">
                    Press ⌘⇧Space anywhere to open
                  </p>
                </>
              )}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selectedResult && (
          <div className="w-[320px] flex-shrink-0 animate-slide-in-right bg-white/[0.03]">
            <ResultDetailPanel
              result={selectedResult}
              query={query}
              onClose={handleCloseDetailPanel}
            />
          </div>
        )}
      </div>

      {/* Footer with keyboard hints - glass style - rounded bottom */}
      <div className="flex items-center justify-between border-t border-white/[0.1] px-5 py-3 text-xs text-white/50 bg-white/[0.02] rounded-b-[24px]">
        <div className="flex items-center gap-5">
          <span className="flex items-center gap-2">
            <kbd className="rounded-lg bg-white/[0.1] px-2 py-1 text-white/60 font-medium">↵</kbd>
            {generateAnswers ? 'to ask' : 'to search'}
          </span>
          <span className="flex items-center gap-2">
            <kbd className="rounded-lg bg-white/[0.1] px-2 py-1 text-white/60 font-medium">↑↓</kbd>
            navigate
          </span>
        </div>
        <span className="flex items-center gap-2">
          <kbd className="rounded-lg bg-white/[0.1] px-2 py-1 text-white/60 font-medium">esc</kbd>
          close
        </span>
      </div>
    </div>
  );
}

/**
 * CommandPaletteOverlay Page Component
 *
 * This is the entry point for the overlay window.
 * It provides necessary context providers and renders the glass palette.
 */
export default function CommandPaletteOverlay() {
  // Make the window background transparent for glassmorphism
  useTransparentWindow();

  return (
    <QueryClientProvider client={queryClient}>
      {/* Fully transparent container - glassmorphism only on the palette itself */}
      <div
        className="h-screen w-screen flex items-start justify-center pt-[10vh]"
        style={{ background: 'transparent' }}
      >
        <GlassCommandPalette />
      </div>
    </QueryClientProvider>
  );
}

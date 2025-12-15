/**
 * CommandPalette - Main search interface dialog
 *
 * Global search interface triggered by Cmd+K / Ctrl+K.
 * Combines all search components into a unified experience.
 * Includes slide-in detail panel for result exploration.
 */

import { useCallback, useEffect, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Search, Loader2, X, Sparkles, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useSearchStore } from '@/stores/searchStore';
import { SearchInput } from './SearchInput';
import { FilterChips } from './FilterChips';
import { RecentSearches } from './RecentSearches';
import { IntentBadge } from './IntentBadge';
import { ModelSelector } from './ModelSelector';
import { StreamingAnswer } from './StreamingAnswer';
import { ResultsList, ResultDetailPanel } from '@/components/results';
import { ANSWER_MODELS } from '@/types/api';
import type { SearchResult } from '@/types/api';

interface CommandPaletteProps {
  /** Controlled open state */
  open?: boolean;
  /** Handler for open state changes */
  onOpenChange?: (open: boolean) => void;
}

export function CommandPalette({
  open: controlledOpen,
  onOpenChange,
}: CommandPaletteProps) {
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
    // Answer generation state
    answer,
    answerSources,
    generateAnswers,
    toggleAnswerGeneration,
    selectedModel,
  } = useSearchStore();

  // Get the selected result object
  const selectedResult = results.find((r) => r.id === selectedResultId) ?? null;

  // Get model label for display
  const modelLabel = ANSWER_MODELS.find(m => m.id === selectedModel)?.label ?? selectedModel;

  // Internal state for uncontrolled dialog usage
  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;

  // Keyboard navigation state
  const [selectedIndex, setSelectedIndex] = useState(-1);

  // Results collapsed state (for AI mode)
  const [resultsCollapsed, setResultsCollapsed] = useState(false);

  // Calculate total navigable items
  const hasResults = results.length > 0;
  const hasSearchHistory = !query && searchHistory.length > 0;
  const totalItems = hasResults
    ? results.length
    : hasSearchHistory
    ? searchHistory.length
    : 0;

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

  // Global keyboard shortcut for opening palette
  useEffect(() => {
    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      // Cmd+K or Ctrl+K to toggle
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen(!open);
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [open, setOpen]);

  // Clear state when closing
  useEffect(() => {
    if (!open) {
      setSelectedIndex(-1);
      setSelectedResult(null);
      setResultsCollapsed(false);
      // Don't clear query/results immediately to avoid flicker
    }
  }, [open, setSelectedResult]);

  // Handle search execution
  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    await executeSearch();
  }, [query, executeSearch]);

  // Handle recent search selection
  const handleRecentSelect = useCallback(
    (recentQuery: string) => {
      setQuery(recentQuery);
      // Execute search after setting query
      setTimeout(() => {
        useSearchStore.getState().executeSearch();
      }, 0);
    },
    [setQuery]
  );

  // Handle result selection - opens detail panel
  const handleResultSelect = useCallback(
    (result: SearchResult) => {
      setSelectedResult(result.id);
    },
    [setSelectedResult]
  );

  // Handle closing the detail panel
  const handleCloseDetailPanel = useCallback(() => {
    setSelectedResult(null);
  }, [setSelectedResult]);

  // Keyboard navigation within palette
  const handleKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        e.stopPropagation();
        setSelectedIndex((prev) =>
          prev < totalItems - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        e.stopPropagation();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
        break;
      case 'Enter':
        e.preventDefault();
        e.stopPropagation(); // Prevent double execution from event bubbling
        if (selectedIndex >= 0) {
          // Handle selection based on current view
          if (hasResults && results[selectedIndex]) {
            handleResultSelect(results[selectedIndex]);
          } else if (hasSearchHistory && searchHistory[selectedIndex]) {
            handleRecentSelect(searchHistory[selectedIndex].query);
          }
        } else {
          // Execute search
          handleSearch();
        }
        break;
      case 'Escape':
        e.preventDefault();
        e.stopPropagation();
        setOpen(false);
        break;
    }
  };

  // Check if we're in AI answer mode with content
  const showAIAnswer = generateAnswers && (answer || isLoading) && query;
  const showCollapsibleResults = showAIAnswer && hasResults && !isLoading;

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        {/* Overlay */}
        <Dialog.Overlay
          className={cn(
            'fixed inset-0 z-50 bg-black/50 backdrop-blur-sm',
            'data-[state=open]:animate-fade-in'
          )}
        />

        {/* Content */}
        <Dialog.Content
          onKeyDown={handleKeyDown}
          className={cn(
            'fixed z-50 left-1/2 top-[12%] -translate-x-1/2',
            'rounded-xl border bg-black shadow-2xl',
            'data-[state=open]:animate-scale-in',
            'outline-none overflow-hidden',
            'transition-all duration-200',
            // Visual indicator for AI mode
            generateAnswers
              ? 'border-white/20'
              : 'border-white/10',
            // Wider when detail panel is open
            selectedResult
              ? 'w-[min(960px,calc(100%-2rem))]'
              : 'w-[min(680px,calc(100%-2rem))]'
          )}
        >
          {/* Header */}
          <div className="flex items-center gap-3 border-b border-white/10 p-4">
            <Search className="h-5 w-5 text-white/40" />
            <SearchInput
              value={query}
              onChange={setQuery}
              onKeyDown={handleKeyDown}
              placeholder={generateAnswers ? "Ask a question..." : "Search your knowledge..."}
              autoFocus
            />
            {isLoading && (
              <Loader2 className="h-4 w-4 animate-spin text-white/40" />
            )}
            {intent && !generateAnswers && (
              <IntentBadge intent={intent} confidence={intentConfidence} />
            )}

            {/* AI Mode indicator */}
            <button
              onClick={toggleAnswerGeneration}
              className={cn(
                'flex items-center gap-1.5 rounded px-2 py-1',
                'text-xs font-medium transition-all',
                generateAnswers
                  ? 'bg-white/10 text-white/90 border border-white/20'
                  : 'text-white/40 hover:text-white/60 hover:bg-white/5'
              )}
              title={generateAnswers ? 'Disable AI answers' : 'Enable AI answers'}
            >
              <Sparkles className="h-3.5 w-3.5" />
              {generateAnswers && <span>AI</span>}
            </button>

            {/* Model selector - only show when answers enabled */}
            {generateAnswers && <ModelSelector />}

            <Dialog.Close asChild>
              <button
                className={cn(
                  'rounded p-1',
                  'text-white/40 hover:text-white/60',
                  'hover:bg-white/5 transition-colors'
                )}
              >
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </button>
            </Dialog.Close>
          </div>

          {/* Filters - hide when AI mode to reduce clutter */}
          {!generateAnswers && (
            <div className="border-b border-white/10 px-4 py-2">
              <FilterChips filters={filters} onChange={setFilters} />
            </div>
          )}

          {/* Content - flex layout for side panel */}
          <div className="flex max-h-[65vh]">
            {/* Results area */}
            <div
              className={cn(
                'flex-1 overflow-y-auto p-3',
                selectedResult && 'border-r border-white/10'
              )}
            >
              {/* AI-generated answer - prominent when in AI mode */}
              {showAIAnswer && (
                <StreamingAnswer
                  answer={answer || ''}
                  sources={answerSources}
                  isLoading={isLoading}
                  modelName={modelLabel}
                />
              )}

              {/* Collapsible results section when AI mode is on */}
              {showCollapsibleResults && (
                <div className="mt-3">
                  <button
                    onClick={() => setResultsCollapsed(!resultsCollapsed)}
                    className={cn(
                      'flex items-center gap-2 w-full px-3 py-2 rounded-lg',
                      'bg-white/[0.02] border border-white/5',
                      'text-sm text-white/60 hover:text-white/80',
                      'hover:bg-white/[0.04] transition-colors'
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

              {/* Regular results when NOT in AI mode */}
              {!generateAnswers && query && results.length > 0 && (
                <ResultsList
                  results={results}
                  query={query}
                  selectedId={selectedResultId}
                  onSelect={handleResultSelect}
                  selectedIndex={selectedIndex}
                />
              )}

              {/* Recent searches - show when no query */}
              {!query && searchHistory.length > 0 && (
                <RecentSearches
                  searches={searchHistory}
                  onSelect={handleRecentSelect}
                  selectedIndex={selectedIndex}
                />
              )}

              {/* Empty state */}
              {query && !isLoading && results.length === 0 && !answer && (
                <div className="flex flex-col items-center justify-center py-12 text-white/50">
                  <Search className="h-8 w-8 mb-3 opacity-50" />
                  <p className="text-sm">No results found</p>
                  <p className="text-xs mt-1 text-white/30">
                    Try different keywords, or connect more data sources
                  </p>
                </div>
              )}

              {/* Loading skeletons - only show when not in AI mode (AI mode has its own loading) */}
              {isLoading && !generateAnswers && (
                <div className="space-y-2 p-2">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="animate-pulse">
                      <div className="h-16 rounded bg-white/5" />
                    </div>
                  ))}
                </div>
              )}

              {/* Initial state - no query, no search history */}
              {!query && searchHistory.length === 0 && !isLoading && (
                <div className="flex flex-col items-center justify-center py-12 text-white/50">
                  {generateAnswers ? (
                    <>
                      <Sparkles className="h-8 w-8 mb-3 opacity-50" />
                      <p className="text-sm">Ask anything about your knowledge</p>
                      <p className="text-xs mt-1 text-white/30">
                        AI will synthesize an answer from your connected sources
                      </p>
                    </>
                  ) : (
                    <>
                      <Search className="h-8 w-8 mb-3 opacity-50" />
                      <p className="text-sm">Search your personal knowledge</p>
                      <p className="text-xs mt-1 text-white/30">
                        Type a query to search across all your connected sources
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Detail panel - slide in from right */}
            {selectedResult && (
              <div className="w-[320px] flex-shrink-0 animate-slide-in-right">
                <ResultDetailPanel
                  result={selectedResult}
                  query={query}
                  onClose={handleCloseDetailPanel}
                />
              </div>
            )}
          </div>

          {/* Footer with keyboard hints */}
          <div className="flex items-center justify-between border-t border-white/10 px-4 py-2 text-xs text-white/40">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="rounded bg-white/5 px-1.5 py-0.5">↵</kbd>
                {generateAnswers ? 'to ask' : 'to search'}
              </span>
              <span className="flex items-center gap-1">
                <kbd className="rounded bg-white/5 px-1.5 py-0.5">↑</kbd>
                <kbd className="rounded bg-white/5 px-1.5 py-0.5">↓</kbd>
                to navigate
              </span>
            </div>
            <span className="flex items-center gap-1">
              <kbd className="rounded bg-white/5 px-1.5 py-0.5">esc</kbd>
              to close
            </span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

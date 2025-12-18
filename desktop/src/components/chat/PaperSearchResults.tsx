/**
 * Paper Search Results - Phase D: Academic Paper Agent
 *
 * Displays academic paper search results in the chat interface
 * with options to download and add papers to the knowledge graph.
 *
 * Research Foundation:
 * - Semantic Scholar API integration
 * - PDF download and processing pipeline
 */

import { useState, useCallback } from 'react';
import {
  FileText,
  Download,
  ExternalLink,
  Check,
  Loader2,
  Users,
  Calendar,
  Quote,
  ChevronDown,
  ChevronUp,
  CheckSquare,
  Square,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { papersApi } from '@/lib/api';

/**
 * Paper metadata matching backend models (camelCase to match CLI JSON output)
 */
export interface PaperMetadata {
  paperId: string;
  title: string;
  authors: Array<{ name: string; authorId?: string }>;
  year?: number;
  abstractText?: string;
  venue?: string;
  citationCount?: number;
  pdfUrl?: string;
  semanticScholarUrl?: string;
  doi?: string;
  arxivId?: string;
  fieldsOfStudy?: string[];
}

export interface SearchResult {
  query: string;
  total: number;
  papers: PaperMetadata[];
  searchTimeMs?: number;
}

interface PaperSearchResultsProps {
  /** Search results from the paper agent */
  results: SearchResult;
  /** Callback when papers are selected for download */
  onDownload?: (papers: PaperMetadata[]) => void;
  /** Optional className */
  className?: string;
}

type DownloadState = 'idle' | 'downloading' | 'downloaded' | 'error';
type IngestionState = 'idle' | 'queued' | 'processing' | 'completed' | 'error';

interface PaperItemState {
  selected: boolean;
  downloadState: DownloadState;
  ingestionState: IngestionState;
  expanded: boolean;
}

/**
 * Individual paper card
 */
function PaperCard({
  paper,
  state,
  onToggleSelect,
  onToggleExpand,
  onDownload,
}: {
  paper: PaperMetadata;
  state: PaperItemState;
  onToggleSelect: () => void;
  onToggleExpand: () => void;
  onDownload: () => void;
}) {
  const authorsList = paper.authors
    .slice(0, 3)
    .map((a) => a.name)
    .join(', ');
  const hasMoreAuthors = paper.authors.length > 3;

  return (
    <div
      className={cn(
        'border border-[var(--color-border)] rounded-lg p-3 transition-all',
        state.selected && 'border-[var(--color-border-active)] bg-[var(--color-surface)]'
      )}
    >
      {/* Header with checkbox and title */}
      <div className="flex items-start gap-2">
        <button
          onClick={onToggleSelect}
          className="mt-0.5 text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
        >
          {state.selected ? (
            <CheckSquare className="w-4 h-4 text-[var(--color-text-secondary)]" />
          ) : (
            <Square className="w-4 h-4" />
          )}
        </button>

        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-[var(--color-text-primary)] leading-tight">
            {paper.title}
          </h4>

          {/* Authors and year */}
          <div className="flex items-center gap-3 mt-1 text-xs text-[var(--color-text-muted)]">
            <span className="flex items-center gap-1">
              <Users className="w-3 h-3" />
              {authorsList}
              {hasMoreAuthors && ` +${paper.authors.length - 3}`}
            </span>
            {paper.year && (
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {paper.year}
              </span>
            )}
            {paper.citationCount !== undefined && (
              <span className="flex items-center gap-1">
                <Quote className="w-3 h-3" />
                {paper.citationCount.toLocaleString()}
              </span>
            )}
          </div>

          {/* Venue, fields, and status badges */}
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            {/* Status badges */}
            {state.downloadState === 'downloaded' && state.ingestionState === 'idle' && (
              <span className="text-xs px-1.5 py-0.5 bg-emerald-500/10 border border-emerald-500/30 rounded text-emerald-400">
                Downloaded
              </span>
            )}
            {state.ingestionState === 'queued' && (
              <span className="text-xs px-1.5 py-0.5 bg-amber-500/10 border border-amber-500/30 rounded text-amber-400">
                Processing...
              </span>
            )}
            {state.ingestionState === 'completed' && (
              <span className="text-xs px-1.5 py-0.5 bg-blue-500/10 border border-blue-500/30 rounded text-blue-400">
                In KG
              </span>
            )}
            {state.ingestionState === 'error' && (
              <span className="text-xs px-1.5 py-0.5 bg-red-500/10 border border-red-500/30 rounded text-red-400">
                Failed
              </span>
            )}

            {paper.venue && (
              <span className="text-xs px-1.5 py-0.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded">
                {paper.venue}
              </span>
            )}
            {paper.fieldsOfStudy?.slice(0, 2).map((field) => (
              <span
                key={field}
                className="text-xs px-1.5 py-0.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[var(--color-text-tertiary)]"
              >
                {field}
              </span>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {paper.pdfUrl && (
            <button
              onClick={onDownload}
              disabled={state.downloadState === 'downloading' || state.downloadState === 'downloaded'}
              className={cn(
                'p-1.5 rounded transition-colors',
                state.downloadState === 'idle' &&
                  'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)]',
                state.downloadState === 'downloading' && 'text-[var(--color-text-muted)] cursor-wait',
                state.downloadState === 'downloaded' && 'text-emerald-500',
                state.downloadState === 'error' && 'text-red-500'
              )}
              title={state.downloadState === 'downloaded' ? 'Downloaded' : 'Download PDF'}
            >
              {state.downloadState === 'downloading' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : state.downloadState === 'downloaded' ? (
                <Check className="w-4 h-4" />
              ) : (
                <Download className="w-4 h-4" />
              )}
            </button>
          )}

          {paper.semanticScholarUrl && (
            <a
              href={paper.semanticScholarUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors"
              title="View on Semantic Scholar"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          )}

          <button
            onClick={onToggleExpand}
            className="p-1.5 rounded text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface)] transition-colors"
            title={state.expanded ? 'Hide abstract' : 'Show abstract'}
          >
            {state.expanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded abstract */}
      {state.expanded && paper.abstractText && (
        <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
          <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
            {paper.abstractText}
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Paper search results component
 */
export function PaperSearchResults({
  results,
  onDownload,
  className,
}: PaperSearchResultsProps) {
  const [paperStates, setPaperStates] = useState<Map<string, PaperItemState>>(() => {
    const map = new Map<string, PaperItemState>();
    results.papers.forEach((paper) => {
      map.set(paper.paperId, {
        selected: false,
        downloadState: 'idle',
        ingestionState: 'idle',
        expanded: false,
      });
    });
    return map;
  });

  const toggleSelect = useCallback((paperId: string) => {
    setPaperStates((prev) => {
      const newMap = new Map(prev);
      const current = newMap.get(paperId);
      if (current) {
        newMap.set(paperId, { ...current, selected: !current.selected });
      }
      return newMap;
    });
  }, []);

  const toggleExpand = useCallback((paperId: string) => {
    setPaperStates((prev) => {
      const newMap = new Map(prev);
      const current = newMap.get(paperId);
      if (current) {
        newMap.set(paperId, { ...current, expanded: !current.expanded });
      }
      return newMap;
    });
  }, []);

  const downloadPaper = useCallback(
    async (paper: PaperMetadata) => {
      if (!paper.pdfUrl) {
        console.warn('No PDF URL available for paper:', paper.title);
        return;
      }

      setPaperStates((prev) => {
        const newMap = new Map(prev);
        const current = newMap.get(paper.paperId);
        if (current) {
          newMap.set(paper.paperId, { ...current, downloadState: 'downloading' });
        }
        return newMap;
      });

      try {
        // Call the Tauri backend API to download the paper
        const response = await papersApi.download({
          paperId: paper.paperId,
          pdfUrl: paper.pdfUrl,
          title: paper.title,
          year: paper.year,
        });

        if (!response.success) {
          throw new Error(response.error || 'Download failed');
        }

        setPaperStates((prev) => {
          const newMap = new Map(prev);
          const current = newMap.get(paper.paperId);
          if (current) {
            newMap.set(paper.paperId, { ...current, downloadState: 'downloaded' });
          }
          return newMap;
        });
      } catch (error) {
        console.error('Failed to download paper:', error);
        setPaperStates((prev) => {
          const newMap = new Map(prev);
          const current = newMap.get(paper.paperId);
          if (current) {
            newMap.set(paper.paperId, { ...current, downloadState: 'error' });
          }
          return newMap;
        });
      }
    },
    []
  );

  const selectedPapers = results.papers.filter(
    (p) => paperStates.get(p.paperId)?.selected
  );

  const selectAll = useCallback(() => {
    setPaperStates((prev) => {
      const newMap = new Map(prev);
      const allSelected = results.papers.every((p) => newMap.get(p.paperId)?.selected);
      results.papers.forEach((paper) => {
        const current = newMap.get(paper.paperId);
        if (current) {
          newMap.set(paper.paperId, { ...current, selected: !allSelected });
        }
      });
      return newMap;
    });
  }, [results.papers]);

  const [isAddingToKG, setIsAddingToKG] = useState(false);

  const addToKnowledgeGraph = useCallback(async () => {
    if (selectedPapers.length === 0) return;

    setIsAddingToKG(true);
    const papersWithPdf = selectedPapers.filter((p) => p.pdfUrl);

    try {
      // Step 1: Download papers that haven't been downloaded yet
      for (const paper of papersWithPdf) {
        const state = paperStates.get(paper.paperId);
        if (state?.downloadState !== 'downloaded') {
          await downloadPaper(paper);
        }
      }

      // Step 2: Trigger ingestion for all papers
      const paperIds = papersWithPdf.map((p) => p.paperId);

      // Update states to show queued
      setPaperStates((prev) => {
        const newMap = new Map(prev);
        for (const paperId of paperIds) {
          const current = newMap.get(paperId);
          if (current) {
            newMap.set(paperId, { ...current, ingestionState: 'queued' });
          }
        }
        return newMap;
      });

      const response = await papersApi.ingest({ paperIds });

      if (response.success) {
        console.log('[Papers] Queued', response.queued, 'papers for ingestion');
        // Update states based on response
        for (const paperInfo of response.papers) {
          setPaperStates((prev) => {
            const newMap = new Map(prev);
            const current = newMap.get(paperInfo.paperId);
            if (current) {
              newMap.set(paperInfo.paperId, {
                ...current,
                ingestionState: paperInfo.status === 'queued' ? 'queued' : 'error',
              });
            }
            return newMap;
          });
        }
      } else {
        console.error('[Papers] Ingestion failed:', response.error);
        // Mark as error
        setPaperStates((prev) => {
          const newMap = new Map(prev);
          for (const paperId of paperIds) {
            const current = newMap.get(paperId);
            if (current) {
              newMap.set(paperId, { ...current, ingestionState: 'error' });
            }
          }
          return newMap;
        });
      }

      onDownload?.(papersWithPdf);
    } catch (error) {
      console.error('[Papers] Failed to add to KG:', error);
      // Mark as error
      setPaperStates((prev) => {
        const newMap = new Map(prev);
        for (const paper of papersWithPdf) {
          const current = newMap.get(paper.paperId);
          if (current) {
            newMap.set(paper.paperId, { ...current, ingestionState: 'error' });
          }
        }
        return newMap;
      });
    } finally {
      setIsAddingToKG(false);
    }
  }, [selectedPapers, paperStates, downloadPaper, onDownload]);

  if (results.papers.length === 0) {
    return (
      <div className={cn('p-4 text-center', className)}>
        <FileText className="w-8 h-8 mx-auto text-[var(--color-text-faint)]" />
        <p className="mt-2 text-sm text-[var(--color-text-muted)]">
          No papers found for "{results.query}"
        </p>
      </div>
    );
  }

  const allSelected = results.papers.every((p) => paperStates.get(p.paperId)?.selected);

  return (
    <div className={cn('space-y-3', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-[var(--color-text-muted)]">
          Found <span className="font-medium text-[var(--color-text-primary)]">{results.total}</span> papers
          {results.searchTimeMs && (
            <span className="text-[var(--color-text-faint)]">
              {' '}
              ({(results.searchTimeMs / 1000).toFixed(2)}s)
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={selectAll}
            className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
          >
            {allSelected ? 'Deselect all' : 'Select all'}
          </button>

          {selectedPapers.length > 0 && (
            <button
              onClick={addToKnowledgeGraph}
              disabled={isAddingToKG}
              className={cn(
                'flex items-center gap-1.5 px-2.5 py-1 text-xs',
                'border border-[var(--color-border)] rounded',
                'transition-colors',
                isAddingToKG
                  ? 'text-[var(--color-text-muted)] cursor-wait'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-surface)]'
              )}
            >
              {isAddingToKG ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Download className="w-3.5 h-3.5" />
              )}
              Add {selectedPapers.length} to KG
            </button>
          )}
        </div>
      </div>

      {/* Paper list */}
      <div className="space-y-2">
        {results.papers.map((paper) => {
          const state = paperStates.get(paper.paperId) || {
            selected: false,
            downloadState: 'idle' as DownloadState,
            ingestionState: 'idle' as IngestionState,
            expanded: false,
          };

          return (
            <PaperCard
              key={paper.paperId}
              paper={paper}
              state={state}
              onToggleSelect={() => toggleSelect(paper.paperId)}
              onToggleExpand={() => toggleExpand(paper.paperId)}
              onDownload={() => downloadPaper(paper)}
            />
          );
        })}
      </div>
    </div>
  );
}

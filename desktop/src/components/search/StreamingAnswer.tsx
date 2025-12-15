/**
 * Streaming Answer Component for LLM Answer Display
 *
 * Step 02: LLM Answer Generation
 * Displays synthesized answers with source citations.
 *
 * Research Foundation:
 * - CausalRAG (ACL 2025): Causal-aware generation with citations
 * - LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach
 *
 * Styling: Per frontend-design.mdc - monochrome, dark-mode first
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Sparkles, BookOpen, Loader2, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StreamingAnswerProps {
  /** The generated answer text */
  answer: string;
  /** Source documents used for the answer */
  sources: string[];
  /** Whether the answer is currently being generated */
  isLoading?: boolean;
  /** Model name used for generation */
  modelName?: string;
  /** Additional CSS classes */
  className?: string;
}

// Threshold for showing collapse button (approximate character count)
const COLLAPSIBLE_LENGTH_THRESHOLD = 500;

export function StreamingAnswer({
  answer,
  sources,
  isLoading = false,
  modelName,
  className,
}: StreamingAnswerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isExpanded, setIsExpanded] = useState(true); // Start expanded by default
  const [copied, setCopied] = useState(false);

  // Check if answer is long enough to allow collapsing
  const isCollapsible = answer.length > COLLAPSIBLE_LENGTH_THRESHOLD;

  // Auto-scroll while loading
  useEffect(() => {
    if (isLoading && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [answer, isLoading]);

  // Copy answer to clipboard
  const handleCopy = useCallback(async () => {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [answer]);

  // Don't render if no answer and not loading
  if (!answer && !isLoading) return null;

  return (
    <div
      className={cn(
        'bg-white/[0.02] border border-white/10 rounded-lg',
        'shadow-lg',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5">
        <Sparkles className="h-4 w-4 text-white/60" />
        <span className="text-sm font-medium text-white/80">AI Answer</span>
        {modelName && (
          <span className="text-xs text-white/30 ml-1">
            via {modelName}
          </span>
        )}

        <div className="ml-auto flex items-center gap-2">
          {isLoading && (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-white/40" />
          )}

          {/* Copy button */}
          {!isLoading && answer && (
            <button
              onClick={handleCopy}
              className={cn(
                'flex items-center gap-1 text-xs px-2 py-1 rounded',
                'text-white/40 hover:text-white/60 hover:bg-white/5',
                'transition-colors'
              )}
              title="Copy answer"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 text-green-500" />
                  <span className="text-green-500">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          )}

          {/* Expand/Collapse button */}
          {!isLoading && isCollapsible && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className={cn(
                'flex items-center gap-1 text-xs px-2 py-1 rounded',
                'text-white/40 hover:text-white/60 hover:bg-white/5',
                'transition-colors'
              )}
            >
              {isExpanded ? (
                <>
                  <ChevronUp className="h-3 w-3" />
                  <span>Collapse</span>
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3" />
                  <span>Expand</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Answer content - scrollable container */}
      <div
        ref={containerRef}
        className={cn(
          'overflow-y-auto transition-all duration-200',
          'px-4 py-3 text-sm text-white/90 leading-relaxed whitespace-pre-wrap',
          // Custom scrollbar styling
          'scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent',
          'hover:scrollbar-thumb-white/30',
          isExpanded ? 'max-h-[45vh]' : 'max-h-24'
        )}
        style={{
          // Fallback scrollbar styling for webkit browsers
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(255,255,255,0.2) transparent',
        }}
      >
        {answer || (
          <span className="text-white/40 italic">Generating answer...</span>
        )}
        {isLoading && (
          <span className="inline-block w-1.5 h-4 bg-white/50 animate-pulse ml-0.5 align-text-bottom" />
        )}
      </div>

      {/* Gradient fade when collapsed */}
      {!isExpanded && isCollapsible && !isLoading && (
        <div className="h-8 -mt-8 relative pointer-events-none bg-gradient-to-t from-[#0a0a0a] to-transparent" />
      )}

      {/* Sources footer */}
      {sources.length > 0 && !isLoading && (
        <div className="px-4 py-2.5 border-t border-white/5 bg-white/[0.01]">
          <div className="flex items-center gap-1.5 text-xs text-white/40 mb-2">
            <BookOpen className="h-3 w-3" />
            <span>Sources ({sources.length})</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {sources.map((source, i) => (
              <span
                key={i}
                className={cn(
                  'text-xs px-2 py-1 rounded',
                  'bg-white/5 text-white/60',
                  'border border-white/5',
                  'hover:bg-white/10 hover:text-white/80 transition-colors cursor-default'
                )}
              >
                {source}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

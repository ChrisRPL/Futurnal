/**
 * Insight Save Button - Phase C: Save Insight
 *
 * Allows users to save important discoveries from chat conversations
 * to the knowledge graph with a single click.
 *
 * Research Foundation:
 * - Training-Free GRPO (2510.08191v1): Natural language learning
 * - ProPerSim (2509.21730v1): Multi-turn context preservation
 */

import { useState } from 'react';
import { Bookmark, Check, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { userInsightsApi } from '@/lib/api';

interface InsightSaveButtonProps {
  /** The message content to save as insight */
  messageContent: string;
  /** Conversation/session ID for context */
  conversationId?: string;
  /** Related entity IDs mentioned in conversation */
  relatedEntities?: string[];
  /** Callback when insight is saved */
  onSave?: (insightId: string) => void;
  /** Optional className */
  className?: string;
}

type SaveState = 'idle' | 'saving' | 'saved' | 'error';

export function InsightSaveButton({
  messageContent,
  conversationId,
  relatedEntities = [],
  onSave,
  className,
}: InsightSaveButtonProps) {
  const [state, setState] = useState<SaveState>('idle');
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    if (state === 'saving' || state === 'saved') return;

    setState('saving');
    setError(null);

    try {
      // Call the Tauri backend API to save the insight
      const response = await userInsightsApi.saveInsight({
        content: messageContent,
        conversationId: conversationId,
        relatedEntities: relatedEntities,
        source: 'user_explicit',
      });

      if (!response.success) {
        throw new Error(response.error || 'Failed to save insight');
      }

      setState('saved');
      if (response.insightId) {
        onSave?.(response.insightId);
      }

      // Reset to idle after a delay to allow saving again
      setTimeout(() => {
        setState('idle');
      }, 3000);
    } catch (err) {
      console.error('Failed to save insight:', err);
      setState('error');
      setError(err instanceof Error ? err.message : 'Failed to save');

      // Reset to idle after showing error
      setTimeout(() => {
        setState('idle');
        setError(null);
      }, 3000);
    }
  };

  const getIcon = () => {
    switch (state) {
      case 'saving':
        return <Loader2 className="w-3.5 h-3.5 animate-spin" />;
      case 'saved':
        return <Check className="w-3.5 h-3.5" />;
      default:
        return <Bookmark className="w-3.5 h-3.5" />;
    }
  };

  const getLabel = () => {
    switch (state) {
      case 'saving':
        return 'Saving...';
      case 'saved':
        return 'Saved';
      case 'error':
        return error || 'Error';
      default:
        return 'Save insight';
    }
  };

  return (
    <button
      onClick={handleSave}
      disabled={state === 'saving' || state === 'saved'}
      title={state === 'error' ? error || 'Error saving' : 'Save this insight to your knowledge graph'}
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-1 text-xs',
        'border border-[var(--color-border)] rounded',
        'transition-all duration-200',
        state === 'idle' && 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-active)] hover:bg-[var(--color-surface)]',
        state === 'saving' && 'text-[var(--color-text-muted)] cursor-wait',
        state === 'saved' && 'text-emerald-500 border-emerald-500/30 bg-emerald-500/5',
        state === 'error' && 'text-red-500 border-red-500/30 bg-red-500/5',
        className
      )}
    >
      {getIcon()}
      <span>{getLabel()}</span>
    </button>
  );
}

/**
 * Compact version for inline use in message actions
 */
export function InsightSaveButtonCompact({
  messageContent,
  conversationId,
  relatedEntities = [],
  onSave,
  className,
}: InsightSaveButtonProps) {
  const [state, setState] = useState<SaveState>('idle');

  const handleSave = async () => {
    if (state === 'saving' || state === 'saved') return;

    setState('saving');

    try {
      // Call the Tauri backend API to save the insight
      const response = await userInsightsApi.saveInsight({
        content: messageContent,
        conversationId: conversationId,
        relatedEntities: relatedEntities,
        source: 'user_explicit',
      });

      if (!response.success) {
        throw new Error(response.error || 'Failed to save insight');
      }

      setState('saved');
      if (response.insightId) {
        onSave?.(response.insightId);
      }
    } catch (err) {
      console.error('Failed to save insight:', err);
      setState('error');

      setTimeout(() => {
        setState('idle');
      }, 2000);
    }
  };

  return (
    <button
      onClick={handleSave}
      disabled={state === 'saving' || state === 'saved'}
      title="Save insight"
      className={cn(
        'p-1.5 rounded transition-colors',
        state === 'idle' && 'text-[var(--color-text-faint)] hover:text-[var(--color-text-muted)] hover:bg-[var(--color-surface)]',
        state === 'saving' && 'text-[var(--color-text-muted)] cursor-wait',
        state === 'saved' && 'text-emerald-500',
        state === 'error' && 'text-red-500',
        className
      )}
    >
      {state === 'saving' ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : state === 'saved' ? (
        <Check className="w-4 h-4" />
      ) : (
        <Bookmark className="w-4 h-4" />
      )}
    </button>
  );
}

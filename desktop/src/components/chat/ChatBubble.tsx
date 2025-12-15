/**
 * ChatBubble - Message bubble component with avatar and actions
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Session tracking
 * - Causal-Copilot (2504.13263v2): Confidence scoring
 *
 * Styling: Per frontend-design.mdc - monochrome, dark-mode first
 */

import { useState } from 'react';
import { Copy, Check, BookOpen, Link2, Bot, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ChatMessage } from '@/types/chat';

interface ChatBubbleProps {
  message: ChatMessage;
  onEntityClick?: (entityId: string) => void;
  className?: string;
}

/**
 * Avatar component for chat messages
 */
function ChatAvatar({
  role,
  className,
}: {
  role: 'user' | 'assistant';
  className?: string;
}) {
  const isUser = role === 'user';

  return (
    <div
      className={cn(
        'flex-shrink-0 w-8 h-8 flex items-center justify-center',
        isUser
          ? 'bg-white text-black'
          : 'bg-white/10 text-white/80 border border-white/20',
        className
      )}
    >
      {isUser ? (
        <User className="h-4 w-4" />
      ) : (
        <Bot className="h-4 w-4" />
      )}
    </div>
  );
}

/**
 * Message actions (copy, etc.) shown on hover
 */
function MessageActions({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div
      className={cn(
        'flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity',
        className
      )}
    >
      <button
        onClick={handleCopy}
        className={cn(
          'p-1 text-white/40 hover:text-white/70 hover:bg-white/5',
          'transition-colors'
        )}
        title={copied ? 'Copied!' : 'Copy message'}
      >
        {copied ? (
          <Check className="h-3 w-3 text-green-400" />
        ) : (
          <Copy className="h-3 w-3" />
        )}
      </button>
    </div>
  );
}

/**
 * Confidence indicator bar
 */
function ConfidenceIndicator({
  confidence,
  className,
}: {
  confidence: number;
  className?: string;
}) {
  // Don't show for 100% confidence
  if (confidence >= 1.0) return null;

  const percentage = Math.round(confidence * 100);

  // Color based on confidence level
  const barColor =
    confidence >= 0.7
      ? 'bg-white/40'
      : confidence >= 0.5
        ? 'bg-yellow-400/40'
        : 'bg-red-400/40';

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className="flex-1 h-1 bg-white/10 overflow-hidden">
        <div
          className={cn('h-full transition-all', barColor)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs text-white/40 tabular-nums">
        {percentage}%
      </span>
    </div>
  );
}

/**
 * Sources list component
 */
function SourcesList({
  sources,
  className,
}: {
  sources: string[];
  className?: string;
}) {
  if (sources.length === 0) return null;

  return (
    <div className={cn('pt-2 border-t border-white/5', className)}>
      <div className="flex items-center gap-1 text-xs text-white/40 mb-1.5">
        <BookOpen className="h-3 w-3" />
        <span>Sources</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {sources.map((source, i) => (
          <span
            key={i}
            className="text-xs px-2 py-0.5 bg-white/5 text-white/60 border border-white/10"
          >
            {source}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * Entity references list component
 */
function EntityRefsList({
  entityRefs,
  onEntityClick,
  className,
}: {
  entityRefs: string[];
  onEntityClick?: (entityId: string) => void;
  className?: string;
}) {
  if (entityRefs.length === 0) return null;

  return (
    <div className={cn('flex flex-wrap gap-1', className)}>
      <div className="flex items-center gap-1 text-xs text-white/40 mr-1">
        <Link2 className="h-3 w-3" />
      </div>
      {entityRefs.map((ref) => (
        <button
          key={ref}
          onClick={() => onEntityClick?.(ref)}
          className={cn(
            'text-xs px-2 py-0.5',
            'bg-white/5 text-white/70 border border-white/10',
            'hover:bg-white/10 hover:text-white hover:border-white/20',
            'transition-colors cursor-pointer'
          )}
        >
          {ref}
        </button>
      ))}
    </div>
  );
}

/**
 * Main ChatBubble component
 */
export function ChatBubble({
  message,
  onEntityClick,
  className,
}: ChatBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        'group flex gap-3',
        isUser ? 'flex-row-reverse' : 'flex-row',
        className
      )}
    >
      {/* Avatar */}
      <ChatAvatar role={message.role} />

      {/* Message content */}
      <div
        className={cn(
          'flex flex-col max-w-[80%]',
          isUser ? 'items-end' : 'items-start'
        )}
      >
        {/* Bubble */}
        <div
          className={cn(
            'p-3',
            isUser
              ? 'bg-white text-black'
              : 'bg-white/5 text-white/90 border border-white/10'
          )}
        >
          {/* Message text */}
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>

          {/* Confidence indicator (assistant only) */}
          {!isUser && (
            <ConfidenceIndicator
              confidence={message.confidence}
              className="mt-2"
            />
          )}

          {/* Sources (assistant only) */}
          {!isUser && (
            <SourcesList sources={message.sources} className="mt-2" />
          )}

          {/* Entity refs (assistant only) */}
          {!isUser && (
            <EntityRefsList
              entityRefs={message.entityRefs}
              onEntityClick={onEntityClick}
              className="mt-2"
            />
          )}
        </div>

        {/* Message actions */}
        <div className="flex items-center gap-2 mt-1 px-1">
          <MessageActions content={message.content} />
          <span className="text-xs text-white/30">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
        </div>
      </div>
    </div>
  );
}

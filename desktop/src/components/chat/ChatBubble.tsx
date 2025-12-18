/**
 * ChatBubble - Message bubble component with avatar and actions
 *
 * Step 03: Chat Interface & Conversational AI
 * Step 08: Experiential Learning Integration
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Session tracking
 * - Causal-Copilot (2504.13263v2): Confidence scoring
 * - RLHI: Reinforcement Learning from Human Interactions
 * - AgentFlow: Learning from explicit user feedback
 *
 * Styling: Per frontend-design.mdc - monochrome, dark-mode first
 */

import { useState } from 'react';
import { Copy, Check, BookOpen, Link2, Bot, User, FileText, ImageIcon, AlertCircle, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import { multimodalApi } from '@/lib/multimodalApi';
import { InsightSaveButtonCompact } from './InsightSaveButton';
import type { ChatMessage, ChatAttachment } from '@/types/chat';

interface ChatBubbleProps {
  message: ChatMessage;
  onEntityClick?: (entityId: string) => void;
  onInsightSaved?: (insightId: string) => void;
  sessionId?: string;
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
 * Feedback buttons for explicit learning
 *
 * Research Foundation:
 * - RLHI: Reinforcement Learning from Human Interactions
 * - AgentFlow: Explicit feedback for continuous improvement
 *
 * When users provide feedback, it's recorded to the learning pipeline
 * to improve future responses. This is what makes the system feel smarter
 * over time - their feedback actually matters.
 */
function FeedbackButtons({
  content,
  entityRefs,
  className,
}: {
  content: string;
  entityRefs: string[];
  className?: string;
}) {
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  const handleFeedback = async (isPositive: boolean) => {
    if (feedback !== null || isRecording) return;

    setIsRecording(true);
    const feedbackType = isPositive ? 'up' : 'down';

    try {
      // Record to learning pipeline
      await multimodalApi.recordDocumentLearning({
        content: content.slice(0, 500),
        source: 'chat_feedback',
        contentType: 'user_feedback',
        success: isPositive,
        qualityScore: isPositive ? 0.9 : 0.3,
        entityTypes: ['UserFeedback', ...(isPositive ? ['GoodResponse'] : ['BadResponse']), ...entityRefs.slice(0, 3)],
        relationTypes: isPositive ? ['APPROVED_BY_USER'] : ['REJECTED_BY_USER'],
      });

      setFeedback(feedbackType);
      console.log(`[Chat] User feedback recorded: ${feedbackType}`);
    } catch (err) {
      console.warn('[Chat] Failed to record feedback:', err);
      // Still show the feedback state locally even if recording failed
      setFeedback(feedbackType);
    } finally {
      setIsRecording(false);
    }
  };

  return (
    <div className={cn('flex items-center gap-0.5', className)}>
      <button
        onClick={() => handleFeedback(true)}
        disabled={feedback !== null || isRecording}
        className={cn(
          'p-1 transition-all',
          feedback === 'up'
            ? 'text-green-400'
            : feedback === 'down'
              ? 'text-white/20 cursor-default'
              : 'text-white/40 hover:text-green-400 hover:bg-white/5',
          isRecording && 'opacity-50'
        )}
        title="Good response - helps me learn"
      >
        <ThumbsUp className="h-3 w-3" />
      </button>
      <button
        onClick={() => handleFeedback(false)}
        disabled={feedback !== null || isRecording}
        className={cn(
          'p-1 transition-all',
          feedback === 'down'
            ? 'text-red-400'
            : feedback === 'up'
              ? 'text-white/20 cursor-default'
              : 'text-white/40 hover:text-red-400 hover:bg-white/5',
          isRecording && 'opacity-50'
        )}
        title="Needs improvement - helps me learn"
      >
        <ThumbsDown className="h-3 w-3" />
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
 * Attachment chip component - displays file attachments as compact widgets
 */
function AttachmentChip({
  attachment,
  isUserMessage,
  className,
}: {
  attachment: ChatAttachment;
  isUserMessage: boolean;
  className?: string;
}) {
  const isImage = attachment.type === 'image';
  const hasError = attachment.status === 'error';

  return (
    <div
      className={cn(
        'flex items-center gap-2 px-2 py-1.5 rounded',
        isUserMessage
          ? 'bg-black/10'
          : 'bg-white/5 border border-white/10',
        hasError && 'border-red-500/30',
        className
      )}
    >
      {/* Icon or image preview */}
      {isImage && attachment.preview ? (
        <img
          src={attachment.preview}
          alt={attachment.name}
          className="w-8 h-8 object-cover rounded"
        />
      ) : isImage ? (
        <ImageIcon className={cn(
          'w-4 h-4',
          isUserMessage ? 'text-black/60' : 'text-white/60'
        )} />
      ) : (
        <FileText className={cn(
          'w-4 h-4',
          isUserMessage ? 'text-black/60' : 'text-white/60'
        )} />
      )}

      {/* File name */}
      <span className={cn(
        'text-xs truncate max-w-[120px]',
        isUserMessage ? 'text-black/80' : 'text-white/80'
      )}>
        {attachment.name}
      </span>

      {/* Status indicator */}
      {hasError && (
        <AlertCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
      )}
    </div>
  );
}

/**
 * Attachments list component
 */
function AttachmentsList({
  attachments,
  isUserMessage,
  className,
}: {
  attachments?: ChatAttachment[];
  isUserMessage: boolean;
  className?: string;
}) {
  if (!attachments || attachments.length === 0) return null;

  return (
    <div className={cn('flex flex-wrap gap-1.5', className)}>
      {attachments.map((attachment) => (
        <AttachmentChip
          key={attachment.id}
          attachment={attachment}
          isUserMessage={isUserMessage}
        />
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
  onInsightSaved,
  sessionId,
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
          {/* Attachments (shown first for user messages) */}
          {isUser && message.attachments && message.attachments.length > 0 && (
            <AttachmentsList
              attachments={message.attachments}
              isUserMessage={isUser}
              className="mb-2"
            />
          )}

          {/* Message text with markdown support */}
          {message.content && (
            <div className={cn(
              'text-sm leading-relaxed chat-markdown',
              isUser ? 'chat-markdown-light' : 'chat-markdown-dark'
            )}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}

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
          {/* Save insight button (assistant only) */}
          {!isUser && (
            <InsightSaveButtonCompact
              messageContent={message.content}
              conversationId={sessionId}
              relatedEntities={message.entityRefs}
              onSave={onInsightSaved}
            />
          )}
          {/* Feedback buttons (assistant only) - RLHI learning */}
          {!isUser && (
            <FeedbackButtons
              content={message.content}
              entityRefs={message.entityRefs}
            />
          )}
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

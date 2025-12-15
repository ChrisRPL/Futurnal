/**
 * ChatInterface - Conversational interface to Personal Knowledge Graph
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Multi-turn context, session management
 * - Causal-Copilot (2504.13263v2): Natural language exploration, confidence
 *
 * Features:
 * - Multi-modal input (text, voice, images, files)
 * - Message bubbles with avatars and actions
 * - Typing indicator animation
 * - Confidence scoring display
 * - Source citations and entity references
 *
 * Styling: Per frontend-design.mdc - monochrome, dark-mode first
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { MessageSquare, X, Link2, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/stores/chatStore';
import { ChatBubble } from './ChatBubble';
import { MessageLoading } from './MessageLoading';
import { ChatInput, type Attachment } from './ChatInput';

interface ChatInterfaceProps {
  /** Session ID to use (creates new if not provided) */
  sessionId?: string;
  /** Entity ID for "Ask about this" feature */
  contextEntityId?: string;
  /** Callback when an entity reference is clicked */
  onEntityClick?: (entityId: string) => void;
  /** Callback to close the chat panel */
  onClose?: () => void;
  /** Enable voice input */
  enableVoice?: boolean;
  /** Enable image upload */
  enableImages?: boolean;
  /** Enable file attachments */
  enableFiles?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function ChatInterface({
  sessionId,
  contextEntityId,
  onEntityClick,
  onClose,
  enableVoice = true,
  enableImages = true,
  enableFiles = true,
  className,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const {
    currentSessionId,
    messages,
    isLoading,
    error,
    contextEntityId: storeContextEntity,
    createSession,
    setCurrentSession,
    sendMessage,
    setContextEntity,
    clearError,
    clearSession,
  } = useChatStore();

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      if (sessionId && sessionId !== currentSessionId) {
        await setCurrentSession(sessionId);
      } else if (!currentSessionId) {
        await createSession();
      }
    };
    initSession();
  }, [sessionId, currentSessionId, setCurrentSession, createSession]);

  // Set context entity if provided
  useEffect(() => {
    if (contextEntityId && contextEntityId !== storeContextEntity) {
      setContextEntity(contextEntityId);
    }
  }, [contextEntityId, storeContextEntity, setContextEntity]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(clearError, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, clearError]);

  // Handle send message
  const handleSubmit = useCallback(
    async (text: string, attachments?: Attachment[]) => {
      if (!text.trim() && (!attachments || attachments.length === 0)) return;
      if (isLoading) return;

      // TODO: Handle attachments (send to vision/OCR models)
      if (attachments && attachments.length > 0) {
        console.log('Attachments:', attachments);
        // For now, just append info about attachments to the message
        const attachmentInfo = attachments
          .map((a) => `[Attached: ${a.name}]`)
          .join(' ');
        text = `${text}\n\n${attachmentInfo}`;
      }

      setInput('');
      await sendMessage(text);
    },
    [isLoading, sendMessage]
  );

  // Handle clear conversation
  const handleClear = async () => {
    if (confirm('Clear all messages in this conversation?')) {
      await clearSession();
    }
  };

  return (
    <div className={cn('flex flex-col h-full bg-black', className)}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
        <MessageSquare className="h-4 w-4 text-white/60" />
        <span className="text-sm font-medium text-white/80">
          Chat with Knowledge
        </span>

        <div className="ml-auto flex items-center gap-1">
          {/* Clear button */}
          {messages.length > 0 && (
            <button
              onClick={handleClear}
              className="p-1.5 text-white/40 hover:text-white/60 hover:bg-white/5 transition-colors"
              title="Clear conversation"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}

          {/* Close button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 text-white/40 hover:text-white/60 hover:bg-white/5 transition-colors"
              title="Close chat"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Context indicator */}
      {(storeContextEntity || contextEntityId) && (
        <div className="px-4 py-2 border-b border-white/5 bg-white/[0.02]">
          <div className="flex items-center gap-2 text-xs text-white/60">
            <Link2 className="h-3 w-3" />
            <span>Discussing:</span>
            <span className="text-white/80 font-medium">
              {storeContextEntity || contextEntityId}
            </span>
            <button
              onClick={() => setContextEntity(null)}
              className="ml-auto text-white/40 hover:text-white/60 p-0.5"
              title="Clear context"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      {/* Messages */}
      <div
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {/* Empty state */}
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-white/50">
            <MessageSquare className="h-12 w-12 mb-4 opacity-30" />
            <p className="text-sm font-medium">Start a conversation</p>
            <p className="text-xs mt-2 text-white/30 text-center max-w-[200px]">
              Ask questions about your personal knowledge graph
            </p>
            <div className="mt-6 flex flex-wrap gap-2 justify-center max-w-[280px]">
              {[
                'What do I know about...',
                'Summarize my notes on...',
                'Find connections between...',
              ].map((hint) => (
                <button
                  key={hint}
                  onClick={() => setInput(hint)}
                  className="text-xs px-2 py-1 bg-white/5 border border-white/10 text-white/50 hover:text-white/70 hover:bg-white/10 transition-colors"
                >
                  {hint}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message list */}
        {messages.map((msg, i) => (
          <ChatBubble
            key={`${msg.timestamp}-${i}`}
            message={msg}
            onEntityClick={onEntityClick}
          />
        ))}

        {/* Loading indicator */}
        {isLoading && <MessageLoading />}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading}
          enableVoice={enableVoice}
          enableImages={enableImages}
          enableFiles={enableFiles}
          placeholder={
            storeContextEntity
              ? `Ask about ${storeContextEntity}...`
              : 'Ask about your knowledge...'
          }
        />

        {/* Message count */}
        {messages.length > 0 && (
          <div className="mt-2 text-xs text-white/30 text-right">
            {messages.length} messages in this conversation
          </div>
        )}
      </div>
    </div>
  );
}

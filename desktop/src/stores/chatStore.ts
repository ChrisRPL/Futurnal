/**
 * Chat Store - Zustand state management for conversational interface
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Multi-turn context, session management
 * - Causal-Copilot (2504.13263v2): Confidence tracking
 *
 * Manages chat sessions, messages, and conversation state.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { chatApi } from '@/lib/api';
import type { ChatMessage, ChatSessionInfo, ChatAttachment } from '@/types/chat';

/** Options for sending a message with attachments */
export interface SendMessageOptions {
  /** Message content visible to user */
  content: string;
  /** Hidden context sent to AI (extracted from attachments) */
  hiddenContext?: string;
  /** Attachment metadata for display */
  attachments?: ChatAttachment[];
}

interface ChatState {
  /** Current active session ID */
  currentSessionId: string | null;
  /** Messages in current session */
  messages: ChatMessage[];
  /** All available sessions */
  sessions: ChatSessionInfo[];
  /** Loading state */
  isLoading: boolean;
  /** Error message */
  error: string | null;
  /** Context entity ID for "Ask about this" feature */
  contextEntityId: string | null;
  /** Selected model (persisted) */
  selectedModel: string;

  // Actions
  /** Create a new chat session */
  createSession: () => Promise<string>;
  /** Set current session and load history */
  setCurrentSession: (sessionId: string) => Promise<void>;
  /** Send a message (string or options with attachments) */
  sendMessage: (contentOrOptions: string | SendMessageOptions) => Promise<void>;
  /** Set context entity for focused conversations */
  setContextEntity: (entityId: string | null) => void;
  /** Load session history */
  loadHistory: (sessionId: string) => Promise<void>;
  /** Load all sessions */
  loadSessions: () => Promise<void>;
  /** Clear current session messages */
  clearSession: () => Promise<void>;
  /** Delete a session */
  deleteSession: (sessionId: string) => Promise<void>;
  /** Set selected model */
  setSelectedModel: (model: string) => void;
  /** Clear error */
  clearError: () => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // Initial state
      currentSessionId: null,
      messages: [],
      sessions: [],
      isLoading: false,
      error: null,
      contextEntityId: null,
      selectedModel: 'llama3.1:8b-instruct-q4_0',

      // Create new session
      createSession: async () => {
        try {
          const sessionId = await chatApi.createSession();
          set({
            currentSessionId: sessionId,
            messages: [],
            contextEntityId: null,
            error: null,
          });
          // Refresh sessions list
          get().loadSessions();
          return sessionId;
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Failed to create session';
          set({ error: errorMsg });
          throw error;
        }
      },

      // Set current session and load history
      setCurrentSession: async (sessionId: string) => {
        set({ currentSessionId: sessionId, isLoading: true, error: null });
        try {
          const response = await chatApi.getHistory(sessionId);
          set({
            messages: response.messages || [],
            isLoading: false,
          });
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Failed to load session';
          set({ error: errorMsg, isLoading: false });
        }
      },

      // Send a message (supports both string and options with attachments)
      sendMessage: async (contentOrOptions: string | SendMessageOptions) => {
        const state = get();
        let sessionId = state.currentSessionId;

        // Parse input - support both string and options
        const options: SendMessageOptions = typeof contentOrOptions === 'string'
          ? { content: contentOrOptions }
          : contentOrOptions;

        const { content, hiddenContext, attachments } = options;

        // Create session if needed
        if (!sessionId) {
          sessionId = await get().createSession();
        }

        // Build message for AI (includes hidden context)
        const aiMessage = hiddenContext
          ? `${content}\n\n---\nAttached content:\n${hiddenContext}`
          : content;

        // Add user message optimistically (UI shows only content + attachments)
        const userMessage: ChatMessage = {
          role: 'user',
          content,
          hiddenContext,
          attachments,
          sources: [],
          entityRefs: [],
          confidence: 1.0,
          timestamp: new Date().toISOString(),
        };

        set((s) => ({
          messages: [...s.messages, userMessage],
          isLoading: true,
          error: null,
        }));

        try {
          const response = await chatApi.sendMessage({
            sessionId: sessionId!,
            message: aiMessage, // Send full message with context to AI
            contextEntityId: state.contextEntityId || undefined,
            model: state.selectedModel,
          });

          if (response.success) {
            set((s) => ({
              messages: [...s.messages, response.response],
              isLoading: false,
            }));
          } else {
            throw new Error('Chat request failed');
          }
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Failed to send message';
          // Add error message as assistant response
          const errorMessage: ChatMessage = {
            role: 'assistant',
            content: 'I encountered an error processing your request. Please try again.',
            sources: [],
            entityRefs: [],
            confidence: 0,
            timestamp: new Date().toISOString(),
          };
          set((s) => ({
            messages: [...s.messages, errorMessage],
            isLoading: false,
            error: errorMsg,
          }));
        }
      },

      // Set context entity
      setContextEntity: (entityId: string | null) => {
        set({ contextEntityId: entityId });
      },

      // Load session history
      loadHistory: async (sessionId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await chatApi.getHistory(sessionId);
          set({
            messages: response.messages || [],
            isLoading: false,
          });
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Failed to load history';
          set({ error: errorMsg, isLoading: false });
        }
      },

      // Load all sessions
      loadSessions: async () => {
        try {
          const response = await chatApi.listSessions();
          set({ sessions: response.sessions || [] });
        } catch (error) {
          console.error('[ChatStore] Failed to load sessions:', error);
        }
      },

      // Clear current session
      clearSession: async () => {
        const { currentSessionId } = get();
        if (!currentSessionId) return;

        try {
          await chatApi.clearSession(currentSessionId);
          set({ messages: [] });
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Failed to clear session';
          set({ error: errorMsg });
        }
      },

      // Delete a session
      deleteSession: async (sessionId: string) => {
        try {
          await chatApi.deleteSession(sessionId);
          const state = get();

          // If deleting current session, clear it
          if (state.currentSessionId === sessionId) {
            set({
              currentSessionId: null,
              messages: [],
              contextEntityId: null,
            });
          }

          // Refresh sessions list
          get().loadSessions();
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Failed to delete session';
          set({ error: errorMsg });
        }
      },

      // Set selected model
      setSelectedModel: (model: string) => {
        set({ selectedModel: model });
      },

      // Clear error
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'futurnal-chat-store',
      partialize: (state) => ({
        currentSessionId: state.currentSessionId,
        selectedModel: state.selectedModel,
      }),
    }
  )
);

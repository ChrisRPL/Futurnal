/**
 * Chat Types - TypeScript interfaces for conversational interface
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Session tracking with timestamps
 * - Causal-Copilot (2504.13263v2): Confidence scoring per response
 */

// ============================================================================
// Chat Message Types
// ============================================================================

/**
 * Individual chat message.
 *
 * Research Foundation:
 * - ProPerSim: Session tracking
 * - Causal-Copilot: Confidence scoring
 */
export interface ChatMessage {
  /** Message sender role */
  role: 'user' | 'assistant';
  /** Message content */
  content: string;
  /** Source documents used */
  sources: string[];
  /** PKG entity references */
  entityRefs: string[];
  /** Response confidence (0-1) per Causal-Copilot */
  confidence: number;
  /** Message timestamp ISO string */
  timestamp: string;
}

/**
 * Chat request to send a message.
 */
export interface ChatRequest {
  sessionId: string;
  message: string;
  contextEntityId?: string;
  model?: string;
}

/**
 * Response from chat send command.
 */
export interface ChatResponse {
  success: boolean;
  sessionId: string;
  response: ChatMessage;
}

// ============================================================================
// Session Types
// ============================================================================

/**
 * Chat session information.
 */
export interface ChatSessionInfo {
  id: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
  preview: string;
}

/**
 * Chat history response.
 */
export interface ChatHistoryResponse {
  success: boolean;
  sessionId: string;
  messages: ChatMessage[];
  total: number;
}

/**
 * Sessions list response.
 */
export interface SessionsListResponse {
  success: boolean;
  sessions: ChatSessionInfo[];
  total: number;
}

/**
 * Generic operation response.
 */
export interface OperationResponse {
  success: boolean;
  sessionId: string;
  message: string;
}

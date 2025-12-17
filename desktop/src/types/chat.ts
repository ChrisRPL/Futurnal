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
 * Attachment metadata for chat messages.
 */
export interface ChatAttachment {
  /** Unique attachment ID */
  id: string;
  /** Attachment type */
  type: 'image' | 'document';
  /** Original filename */
  name: string;
  /** MIME type */
  mimeType?: string;
  /** File size in bytes */
  size?: number;
  /** Preview URL (for images) */
  preview?: string;
  /** Processing status */
  status: 'pending' | 'processing' | 'success' | 'error';
  /** Error message if failed */
  error?: string;
}

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
  /** Message content (visible to user) */
  content: string;
  /** Hidden context sent to AI (extracted from attachments) */
  hiddenContext?: string;
  /** File attachments */
  attachments?: ChatAttachment[];
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

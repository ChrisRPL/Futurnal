/**
 * Futurnal Desktop Shell - Tauri API Client
 *
 * This module provides typed wrappers around Tauri's invoke function
 * for calling Rust IPC commands.
 */

import { invoke } from '@tauri-apps/api/core';
import type {
  SearchQuery,
  SearchResponse,
  SearchWithAnswerResponse,
  SearchHistoryItem,
  Connector,
  AddSourceRequest,
  SyncResult,
  ConsentRecord,
  GrantConsentRequest,
  AuditLogEntry,
  AuditLogQuery,
  OrchestratorStatus,
  GraphData,
  GraphFilter,
  GraphStats,
} from '@/types/api';

/**
 * API Error class for handling Tauri IPC errors.
 */
export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Default timeout for API calls in milliseconds.
 */
const DEFAULT_TIMEOUT_MS = 30000;

/**
 * Wrapper around invoke with timeout handling.
 */
async function invokeWithTimeout<T>(
  command: string,
  args?: Record<string, unknown>,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new ApiError(`Request timeout: ${command}`)), timeoutMs);
  });

  try {
    const result = await Promise.race([invoke<T>(command, args), timeout]);
    return result;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (typeof error === 'string') throw new ApiError(error);
    throw new ApiError(String(error));
  }
}

// ============================================================================
// Search API
// ============================================================================

export const searchApi = {
  /**
   * Execute a search query against the Hybrid Search API.
   * Returns real results from GraphRAG pipeline only.
   */
  async search(query: SearchQuery): Promise<SearchResponse> {
    try {
      const response = await invokeWithTimeout<SearchResponse>('search_query', { query });
      return response;
    } catch (error) {
      console.error('[Search API] Backend error:', error);
      // Return empty results on error - don't use mock data
      return {
        results: [],
        total: 0,
        query_id: `error-${Date.now()}`,
        intent: { primary: 'exploratory' },
        execution_time_ms: 0,
      };
    }
  },

  /**
   * Get search history.
   */
  async getHistory(limit?: number): Promise<SearchHistoryItem[]> {
    try {
      return await invokeWithTimeout('get_search_history', { limit });
    } catch {
      return []; // Graceful fallback
    }
  },

  /**
   * Execute search with LLM answer generation.
   *
   * Step 02: LLM Answer Generation
   * Research Foundation:
   * - CausalRAG (ACL 2025): Causal-aware generation
   * - LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach
   */
  async searchWithAnswer(
    query: SearchQuery,
    generateAnswer = true,
    model?: string
  ): Promise<SearchWithAnswerResponse> {
    try {
      const response = await invokeWithTimeout<SearchWithAnswerResponse>(
        'search_with_answer',
        {
          query,
          generateAnswer,
          model,
        },
        60000 // Longer timeout for answer generation
      );
      return response;
    } catch (error) {
      console.error('[Search API] Answer generation error:', error);
      // Return empty results on error - graceful degradation
      return {
        results: [],
        total: 0,
        query_id: `error-${Date.now()}`,
        intent: { primary: 'exploratory' },
        execution_time_ms: 0,
        answer: undefined,
        sources: undefined,
        model: undefined,
        generation_time_ms: undefined,
      };
    }
  },
};

// ============================================================================
// Connectors API
// ============================================================================

export const connectorsApi = {
  /**
   * List all configured data sources.
   */
  async list(): Promise<Connector[]> {
    return invokeWithTimeout('list_sources');
  },

  /**
   * Add a new data source.
   */
  async add(request: AddSourceRequest): Promise<Connector> {
    return invokeWithTimeout('add_source', { request });
  },

  /**
   * Pause a data source.
   */
  async pause(id: string): Promise<void> {
    return invokeWithTimeout('pause_source', { id });
  },

  /**
   * Resume a paused data source.
   */
  async resume(id: string): Promise<void> {
    return invokeWithTimeout('resume_source', { id });
  },

  /**
   * Delete a data source.
   * @param id Source ID to delete
   * @param connectorType Optional connector type for better cleanup (github, imap, obsidian, local_folder)
   */
  async delete(id: string, connectorType?: string): Promise<void> {
    return invokeWithTimeout('delete_source', { id, connectorType });
  },

  /**
   * Retry a failed source sync.
   */
  async retry(id: string): Promise<void> {
    return invokeWithTimeout('retry_source', { id });
  },

  /**
   * Pause all data sources.
   */
  async pauseAll(): Promise<void> {
    return invokeWithTimeout('pause_all_sources');
  },

  /**
   * Resume all data sources.
   */
  async resumeAll(): Promise<void> {
    return invokeWithTimeout('resume_all_sources');
  },

  /**
   * Sync a data source (clone/update repository).
   * For GitHub sources, this triggers a git clone/pull operation.
   * Returns sync result with files synced, bytes, and duration.
   */
  async sync(id: string, connectorType: string): Promise<SyncResult> {
    return invokeWithTimeout('sync_source', { id, connectorType }, 300000); // 5 min timeout
  },

  /**
   * Sync all GitHub sources.
   */
  async syncAllGitHub(): Promise<SyncResult[]> {
    return invokeWithTimeout('sync_all_github', undefined, 600000); // 10 min timeout
  },

  /**
   * Authenticate an IMAP mailbox using OAuth2.
   * Opens a browser for OAuth2 authentication and stores the tokens.
   * @param mailboxId The IMAP mailbox ID to authenticate
   * @param clientId OAuth2 client ID from Google Cloud Console
   * @param clientSecret OAuth2 client secret
   * @param provider OAuth provider: 'gmail' or 'office365'
   */
  async authenticateImap(
    mailboxId: string,
    clientId: string,
    clientSecret: string,
    provider?: string
  ): Promise<{ success: boolean; error?: string; credential_id?: string }> {
    return invokeWithTimeout('authenticate_imap', {
      mailboxId,
      clientId,
      clientSecret,
      provider: provider || 'gmail',
    }, 120000); // 2 min timeout for OAuth flow
  },
};

// ============================================================================
// Privacy API
// ============================================================================

export const privacyApi = {
  /**
   * Get consent records.
   */
  async getConsent(sourceId?: string): Promise<ConsentRecord[]> {
    try {
      return await invokeWithTimeout('get_consent', { sourceId });
    } catch {
      return [];
    }
  },

  /**
   * Grant consent for a data source.
   */
  async grantConsent(request: GrantConsentRequest): Promise<ConsentRecord> {
    return invokeWithTimeout('grant_consent', { request });
  },

  /**
   * Revoke consent for a data source.
   */
  async revokeConsent(sourceId: string, consentType: string): Promise<void> {
    return invokeWithTimeout('revoke_consent', { sourceId, consentType });
  },

  /**
   * Get audit logs.
   */
  async getAuditLogs(query: AuditLogQuery): Promise<AuditLogEntry[]> {
    try {
      return await invokeWithTimeout('get_audit_logs', { query });
    } catch {
      return [];
    }
  },
};

// ============================================================================
// Orchestrator API
// ============================================================================

export const orchestratorApi = {
  /**
   * Get orchestrator daemon status.
   */
  async getStatus(): Promise<OrchestratorStatus> {
    return invokeWithTimeout('get_orchestrator_status');
  },

  /**
   * Start the orchestrator daemon.
   */
  async start(): Promise<void> {
    return invokeWithTimeout('start_orchestrator');
  },

  /**
   * Stop the orchestrator daemon.
   */
  async stop(): Promise<void> {
    return invokeWithTimeout('stop_orchestrator');
  },

  /**
   * Ensure orchestrator is running (starts if not).
   * Returns true if the orchestrator was started, false if already running.
   */
  async ensureRunning(): Promise<boolean> {
    return invokeWithTimeout('ensure_orchestrator_running');
  },
};

// ============================================================================
// Graph API
// ============================================================================

export const graphApi = {
  /**
   * Get knowledge graph data for visualization.
   * Returns empty graph data if backend returns empty results or is unavailable.
   */
  async getGraph(limit?: number): Promise<GraphData> {
    try {
      const result = await invokeWithTimeout<GraphData>('get_knowledge_graph', { limit });
      return result;
    } catch (error) {
      console.warn('[Graph API] Failed to fetch graph data:', error);
      return { nodes: [], links: [] };
    }
  },

  /**
   * Get filtered subgraph based on filter parameters.
   */
  async getFilteredGraph(filter: GraphFilter, limit?: number, offset?: number): Promise<GraphData> {
    try {
      const result = await invokeWithTimeout<GraphData>('get_filtered_graph', { filter, limit, offset });
      return result;
    } catch (error) {
      console.warn('[Graph API] Failed to fetch filtered graph:', error);
      return { nodes: [], links: [] };
    }
  },

  /**
   * Get neighbors of a node within specified depth.
   */
  async getNodeNeighbors(nodeId: string, depth?: number): Promise<GraphData> {
    try {
      const result = await invokeWithTimeout<GraphData>('get_node_neighbors', { node_id: nodeId, depth });
      return result;
    } catch (error) {
      console.warn('[Graph API] Failed to fetch node neighbors:', error);
      return { nodes: [], links: [] };
    }
  },

  /**
   * Get graph statistics.
   */
  async getStats(): Promise<GraphStats> {
    try {
      return await invokeWithTimeout<GraphStats>('get_graph_stats');
    } catch (error) {
      console.warn('[Graph API] Failed to fetch graph stats:', error);
      return { total_nodes: 0, total_links: 0, nodes_by_type: {}, nodes_by_source: {} };
    }
  },

  /**
   * Get list of bookmarked node IDs.
   */
  async getBookmarks(): Promise<string[]> {
    try {
      return await invokeWithTimeout<string[]>('get_bookmarked_nodes');
    } catch (error) {
      console.warn('[Graph API] Failed to fetch bookmarks:', error);
      return [];
    }
  },

  /**
   * Add a node to bookmarks.
   */
  async bookmarkNode(nodeId: string): Promise<void> {
    return invokeWithTimeout('bookmark_node', { node_id: nodeId });
  },

  /**
   * Remove a node from bookmarks.
   */
  async unbookmarkNode(nodeId: string): Promise<void> {
    return invokeWithTimeout('unbookmark_node', { node_id: nodeId });
  },
};

// ============================================================================
// Ollama API
// ============================================================================

export interface OllamaModel {
  name: string;
  size: string;
  modified: string;
}

export interface ListModelsResponse {
  models: OllamaModel[];
}

export const ollamaApi = {
  /**
   * List installed Ollama models.
   */
  async listModels(): Promise<ListModelsResponse> {
    try {
      return await invokeWithTimeout<ListModelsResponse>('list_ollama_models');
    } catch (error) {
      console.warn('[Ollama API] Failed to list models:', error);
      return { models: [] };
    }
  },

  /**
   * Check if a specific model is installed.
   */
  async isModelInstalled(modelName: string): Promise<boolean> {
    try {
      return await invokeWithTimeout<boolean>('is_model_installed', { modelName });
    } catch {
      return false;
    }
  },

  /**
   * Pull (download) an Ollama model.
   * Progress updates are emitted via 'ollama-pull-progress' event.
   * Completion is emitted via 'ollama-pull-complete' event.
   */
  async pullModel(modelName: string): Promise<boolean> {
    return invokeWithTimeout<boolean>('pull_ollama_model', { modelName }, 600000); // 10 min timeout
  },
};

// ============================================================================
// Chat API
// ============================================================================

import type {
  ChatRequest,
  ChatResponse,
  ChatHistoryResponse,
  SessionsListResponse,
  OperationResponse,
} from '@/types/chat';

/**
 * Chat API for conversational interface.
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Multi-turn context
 * - Causal-Copilot (2504.13263v2): Confidence scoring
 */
export const chatApi = {
  /**
   * Send a chat message and get a response.
   */
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    return invokeWithTimeout<ChatResponse>('send_chat_message', { request }, 60000); // 60s timeout
  },

  /**
   * Get conversation history for a session.
   */
  async getHistory(sessionId: string, limit?: number): Promise<ChatHistoryResponse> {
    try {
      return await invokeWithTimeout<ChatHistoryResponse>('get_chat_history', {
        sessionId,
        limit,
      });
    } catch (error) {
      console.warn('[Chat API] Failed to get history:', error);
      return {
        success: false,
        sessionId,
        messages: [],
        total: 0,
      };
    }
  },

  /**
   * List all chat sessions.
   */
  async listSessions(limit?: number): Promise<SessionsListResponse> {
    try {
      return await invokeWithTimeout<SessionsListResponse>('list_chat_sessions', { limit });
    } catch (error) {
      console.warn('[Chat API] Failed to list sessions:', error);
      return {
        success: false,
        sessions: [],
        total: 0,
      };
    }
  },

  /**
   * Create a new chat session.
   */
  async createSession(): Promise<string> {
    return invokeWithTimeout<string>('create_chat_session');
  },

  /**
   * Clear messages from a session.
   */
  async clearSession(sessionId: string): Promise<OperationResponse> {
    return invokeWithTimeout<OperationResponse>('clear_chat_session', { sessionId });
  },

  /**
   * Delete a chat session.
   */
  async deleteSession(sessionId: string): Promise<OperationResponse> {
    return invokeWithTimeout<OperationResponse>('delete_chat_session', { sessionId });
  },
};

// ============================================================================
// User Insights API (Phase C: Save Insight)
// ============================================================================

export interface SaveUserInsightRequest {
  content: string;
  conversationId?: string;
  relatedEntities: string[];
  source: string;
}

export interface SaveUserInsightResponse {
  success: boolean;
  insightId?: string;
  error?: string;
}

export const userInsightsApi = {
  /**
   * Save a user-generated insight from chat conversation.
   */
  async saveInsight(request: SaveUserInsightRequest): Promise<SaveUserInsightResponse> {
    try {
      return await invokeWithTimeout<SaveUserInsightResponse>('save_user_insight', { request });
    } catch (error) {
      console.error('[User Insights API] Failed to save insight:', error);
      return {
        success: false,
        error: String(error),
      };
    }
  },
};

// ============================================================================
// Papers API (Phase D: Academic Paper Agent)
// ============================================================================

export interface PaperAuthor {
  name: string;
  authorId?: string;
}

export interface PaperMetadata {
  paperId: string;
  title: string;
  authors: PaperAuthor[];
  year?: number;
  abstractText?: string;
  venue?: string;
  citationCount?: number;
  pdfUrl?: string;
  semanticScholarUrl?: string;
  doi?: string;
  arxivId?: string;
  fieldsOfStudy: string[];
}

export interface PaperSearchRequest {
  query: string;
  limit?: number;
  yearFrom?: number;
  yearTo?: number;
  fields?: string[];
}

export interface PaperSearchResponse {
  success: boolean;
  query: string;
  papers: PaperMetadata[];
  total: number;
  searchTimeMs?: number;
  error?: string;
}

export interface PaperDownloadRequest {
  paperId: string;
  pdfUrl: string;
  title: string;
  year?: number;
}

export interface DownloadedPaper {
  paperId: string;
  title: string;
  localPath: string;
  fileSizeBytes: number;
}

export interface PaperDownloadResponse {
  success: boolean;
  downloaded?: DownloadedPaper;
  error?: string;
}

export interface PaperRecommendationsResponse {
  success: boolean;
  sourcePaperId: string;
  recommendations: PaperMetadata[];
  error?: string;
}

export const papersApi = {
  /**
   * Search for academic papers.
   */
  async search(request: PaperSearchRequest): Promise<PaperSearchResponse> {
    try {
      return await invokeWithTimeout<PaperSearchResponse>('search_papers', { request }, 60000);
    } catch (error) {
      console.error('[Papers API] Search failed:', error);
      return {
        success: false,
        query: request.query,
        papers: [],
        total: 0,
        error: String(error),
      };
    }
  },

  /**
   * Download a paper PDF.
   */
  async download(request: PaperDownloadRequest): Promise<PaperDownloadResponse> {
    try {
      return await invokeWithTimeout<PaperDownloadResponse>('download_paper', { request }, 120000);
    } catch (error) {
      console.error('[Papers API] Download failed:', error);
      return {
        success: false,
        error: String(error),
      };
    }
  },

  /**
   * Get paper recommendations based on a paper.
   */
  async getRecommendations(paperId: string, limit?: number): Promise<PaperRecommendationsResponse> {
    try {
      return await invokeWithTimeout<PaperRecommendationsResponse>('get_paper_recommendations', {
        paperId,
        limit,
      });
    } catch (error) {
      console.error('[Papers API] Get recommendations failed:', error);
      return {
        success: false,
        sourcePaperId: paperId,
        recommendations: [],
        error: String(error),
      };
    }
  },

  /**
   * Get details for a specific paper.
   */
  async getDetails(paperId: string): Promise<PaperMetadata | null> {
    try {
      return await invokeWithTimeout<PaperMetadata>('get_paper_details', { paperId });
    } catch (error) {
      console.error('[Papers API] Get details failed:', error);
      return null;
    }
  },

  /**
   * Intelligent agentic paper search with query understanding.
   *
   * Features:
   * - Query analysis and understanding
   * - Multiple search strategies (synonyms, expansions)
   * - Relevance scoring
   * - Synthesis and suggestions
   */
  async agenticSearch(query: string): Promise<AgenticSearchResponse> {
    try {
      return await invokeWithTimeout<AgenticSearchResponse>(
        'agentic_search_papers',
        { request: { query } },
        120000 // 2 minute timeout for comprehensive search
      );
    } catch (error) {
      console.error('[Papers API] Agentic search failed:', error);
      return {
        success: false,
        query,
        papers: [],
        totalEvaluated: 0,
        synthesis: '',
        suggestions: [],
        strategiesTried: [],
        error: String(error),
      };
    }
  },

  /**
   * Ingest papers into the knowledge graph.
   */
  async ingest(request: PaperIngestRequest): Promise<PaperIngestResponse> {
    try {
      return await invokeWithTimeout<PaperIngestResponse>(
        'ingest_papers',
        { request },
        60000
      );
    } catch (error) {
      console.error('[Papers API] Ingest failed:', error);
      return {
        success: false,
        queued: 0,
        papers: [],
        error: String(error),
      };
    }
  },

  /**
   * Get status for a specific paper.
   */
  async getStatus(paperId: string): Promise<PaperStatusResponse> {
    try {
      return await invokeWithTimeout<PaperStatusResponse>(
        'get_paper_status',
        { paperId }
      );
    } catch (error) {
      console.error('[Papers API] Get status failed:', error);
      return {
        success: false,
        error: String(error),
      };
    }
  },

  /**
   * Get status for all papers.
   */
  async getAllStatus(): Promise<AllPapersStatusResponse> {
    try {
      return await invokeWithTimeout<AllPapersStatusResponse>(
        'get_all_papers_status',
        {}
      );
    } catch (error) {
      console.error('[Papers API] Get all status failed:', error);
      return {
        success: false,
        total: 0,
        counts: { download: {}, ingestion: {} },
        papers: [],
        error: String(error),
      };
    }
  },
};

// ============================================================================
// Agentic Search Types
// ============================================================================

export interface SearchStrategy {
  query: string;
  type: string;
  rationale: string;
}

export interface ScoredPaper {
  paperId: string;
  title: string;
  authors: PaperAuthor[];
  year?: number;
  abstractText?: string;
  venue?: string;
  citationCount?: number;
  pdfUrl?: string;
  sourceUrl?: string;
  relevanceScore: number;
  rationale: string;
}

export interface AgenticSearchResponse {
  success: boolean;
  query: string;
  papers: ScoredPaper[];
  totalEvaluated: number;
  synthesis: string;
  suggestions: string[];
  strategiesTried: SearchStrategy[];
  searchTimeMs?: number;
  error?: string;
}

// ============================================================================
// Paper Ingestion Types
// ============================================================================

export interface PaperIngestRequest {
  paperIds: string[];
}

export interface IngestedPaperInfo {
  paperId: string;
  title: string;
  status: string;
}

export interface PaperIngestResponse {
  success: boolean;
  queued: number;
  papers: IngestedPaperInfo[];
  error?: string;
}

// ============================================================================
// Paper Status Types
// ============================================================================

export interface PaperStatusInfo {
  paperId: string;
  title: string;
  localPath?: string;
  downloadStatus: 'pending' | 'downloaded' | 'failed';
  ingestionStatus: 'pending' | 'queued' | 'processing' | 'completed' | 'failed';
  downloadedAt?: string;
  ingestedAt?: string;
  fileSizeBytes: number;
}

export interface PaperStatusResponse {
  success: boolean;
  paper?: PaperStatusInfo;
  error?: string;
}

export interface PaperStatusCounts {
  download: Record<string, number>;
  ingestion: Record<string, number>;
}

export interface AllPapersStatusResponse {
  success: boolean;
  total: number;
  counts: PaperStatusCounts;
  papers: PaperStatusInfo[];
  error?: string;
}

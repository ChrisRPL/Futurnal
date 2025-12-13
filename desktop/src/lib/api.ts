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

/**
 * Mock search results for UI testing when backend is unavailable.
 */
function getMockSearchResults(queryText: string): SearchResponse {
  const mockResults = [
    {
      id: 'mock-1',
      content: `Meeting notes from project kickoff discussing ${queryText}. The team agreed on the timeline and key deliverables. Action items were assigned to each team member with specific deadlines.`,
      score: 0.92,
      confidence: 0.88,
      timestamp: new Date().toISOString(),
      entity_type: 'Event' as const,
      source_type: 'text' as const,
      metadata: {
        source: '/Users/demo/notes/meetings/kickoff.md',
        extractionTimestamp: new Date().toISOString(),
        schemaVersion: 'v2',
        entityId: 'entity-001',
      },
      causal_chain: {
        causes: ['Project approval', 'Budget allocation'],
        anchor: 'Project kickoff meeting',
        effects: ['Sprint planning', 'Resource assignment'],
      },
    },
    {
      id: 'mock-2',
      content: `Research document about ${queryText} with detailed analysis and findings. This comprehensive study examines the key factors and their relationships.`,
      score: 0.85,
      confidence: 0.76,
      timestamp: new Date(Date.now() - 86400000).toISOString(), // Yesterday
      entity_type: 'Document' as const,
      source_type: 'text' as const,
      metadata: {
        source: '/Users/demo/research/analysis.md',
        extractionTimestamp: new Date().toISOString(),
        schemaVersion: 'v2',
        entityId: 'entity-002',
      },
    },
    {
      id: 'mock-3',
      content: `Code snippet implementing ${queryText} functionality. Uses TypeScript with React hooks for state management and includes error handling.`,
      score: 0.78,
      confidence: 0.92,
      timestamp: new Date(Date.now() - 172800000).toISOString(), // 2 days ago
      entity_type: 'Code' as const,
      source_type: 'code' as const,
      metadata: {
        source: '/Users/demo/projects/app/src/feature.ts',
        extractionTimestamp: new Date().toISOString(),
        schemaVersion: 'v2',
        entityId: 'entity-003',
        language: 'TypeScript',
      },
    },
    {
      id: 'mock-4',
      content: `Voice memo transcription discussing ${queryText}. Key points mentioned include prioritization, team coordination, and next steps for the upcoming sprint.`,
      score: 0.71,
      confidence: 0.65,
      timestamp: new Date(Date.now() - 604800000).toISOString(), // 1 week ago
      entity_type: 'Event' as const,
      source_type: 'audio' as const,
      source_confidence: 0.82,
      metadata: {
        source: '/Users/demo/voice-memos/standup-2024.m4a',
        extractionTimestamp: new Date().toISOString(),
        schemaVersion: 'v2',
        entityId: 'entity-004',
        duration: '5:32',
      },
    },
    {
      id: 'mock-5',
      content: `Scanned document about ${queryText} from the archives. Contains historical context and background information that provides important perspective on current initiatives.`,
      score: 0.64,
      confidence: 0.58,
      timestamp: new Date(Date.now() - 2592000000).toISOString(), // 30 days ago
      entity_type: 'Document' as const,
      source_type: 'ocr' as const,
      source_confidence: 0.74,
      metadata: {
        source: '/Users/demo/scans/archive-doc.pdf',
        extractionTimestamp: new Date().toISOString(),
        schemaVersion: 'v2',
        entityId: 'entity-005',
      },
    },
  ];

  return {
    results: mockResults,
    total: mockResults.length,
    query_id: `mock-${Date.now()}`,
    intent: {
      primary: 'exploratory',
    },
    execution_time_ms: 42,
  };
}

export const searchApi = {
  /**
   * Execute a search query against the Hybrid Search API.
   * Falls back to mock data if backend returns empty results or is unavailable.
   */
  async search(query: SearchQuery): Promise<SearchResponse> {
    try {
      const response = await invokeWithTimeout<SearchResponse>('search_query', { query });
      // If backend returns empty results, use mock data for UI testing
      if (!response.results || response.results.length === 0) {
        console.info('[Search API] Using mock data - no results from backend');
        return getMockSearchResults(query.query);
      }
      return response;
    } catch {
      // Return mock data for UI testing when backend is unavailable
      console.info('[Search API] Using mock data - backend unavailable');
      return getMockSearchResults(query.query);
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

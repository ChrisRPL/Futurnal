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
  ConsentRecord,
  GrantConsentRequest,
  AuditLogEntry,
  AuditLogQuery,
  OrchestratorStatus,
  GraphData,
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
   */
  async search(query: SearchQuery): Promise<SearchResponse> {
    return invokeWithTimeout('search_query', { query });
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
   */
  async delete(id: string): Promise<void> {
    return invokeWithTimeout('delete_source', { id });
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
   * Get orchestrator status.
   */
  async getStatus(): Promise<OrchestratorStatus> {
    return invokeWithTimeout('get_orchestrator_status');
  },
};

// ============================================================================
// Graph API
// ============================================================================

export const graphApi = {
  /**
   * Get knowledge graph data for visualization.
   */
  async getGraph(limit?: number): Promise<GraphData> {
    return invokeWithTimeout('get_knowledge_graph', { limit });
  },
};

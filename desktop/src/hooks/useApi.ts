/**
 * React Query hooks for Futurnal API
 *
 * These hooks provide a convenient way to interact with the Tauri IPC commands
 * with automatic caching, refetching, and error handling.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  searchApi,
  connectorsApi,
  privacyApi,
  orchestratorApi,
  graphApi,
} from '@/lib/api';
import { queryKeys } from '@/lib/queryClient';
import type {
  SearchQuery,
  AddSourceRequest,
  GrantConsentRequest,
  AuditLogQuery,
} from '@/types/api';

// ============================================================================
// Search Hooks
// ============================================================================

/**
 * Hook to execute a search query.
 */
export function useSearch(query: SearchQuery, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: queryKeys.search(query.query),
    queryFn: () => searchApi.search(query),
    enabled: options?.enabled !== false && !!query.query.trim(),
  });
}

/**
 * Hook to get search history.
 */
export function useSearchHistory(limit?: number) {
  return useQuery({
    queryKey: queryKeys.searchHistory(limit),
    queryFn: () => searchApi.getHistory(limit),
  });
}

// ============================================================================
// Connector Hooks
// ============================================================================

/**
 * Hook to list all connectors.
 */
export function useConnectors() {
  return useQuery({
    queryKey: queryKeys.connectors,
    queryFn: connectorsApi.list,
    refetchInterval: 5000, // Poll every 5s for status updates
  });
}

/**
 * Hook to add a new connector.
 */
export function useAddConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: AddSourceRequest) => connectorsApi.add(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

/**
 * Hook to pause a connector.
 */
export function usePauseConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => connectorsApi.pause(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

/**
 * Hook to resume a connector.
 */
export function useResumeConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => connectorsApi.resume(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

/**
 * Hook to delete a connector.
 */
export function useDeleteConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => connectorsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

/**
 * Hook to retry a failed connector.
 */
export function useRetryConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => connectorsApi.retry(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

/**
 * Hook to pause all connectors.
 */
export function usePauseAllConnectors() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => connectorsApi.pauseAll(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

/**
 * Hook to resume all connectors.
 */
export function useResumeAllConnectors() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => connectorsApi.resumeAll(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
    },
  });
}

// ============================================================================
// Privacy Hooks
// ============================================================================

/**
 * Hook to get consent records.
 */
export function useConsent(sourceId?: string) {
  return useQuery({
    queryKey: queryKeys.consent(sourceId),
    queryFn: () => privacyApi.getConsent(sourceId),
  });
}

/**
 * Hook to grant consent.
 */
export function useGrantConsent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: GrantConsentRequest) => privacyApi.grantConsent(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['consent'] });
    },
  });
}

/**
 * Hook to revoke consent.
 */
export function useRevokeConsent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ sourceId, consentType }: { sourceId: string; consentType: string }) =>
      privacyApi.revokeConsent(sourceId, consentType),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['consent'] });
    },
  });
}

/**
 * Hook to get audit logs.
 */
export function useAuditLogs(query: AuditLogQuery) {
  return useQuery({
    queryKey: queryKeys.auditLogs(query),
    queryFn: () => privacyApi.getAuditLogs(query),
  });
}

// ============================================================================
// Orchestrator Hooks
// ============================================================================

/**
 * Hook to get orchestrator status.
 */
export function useOrchestratorStatus() {
  return useQuery({
    queryKey: queryKeys.orchestratorStatus,
    queryFn: orchestratorApi.getStatus,
    refetchInterval: 3000, // Poll every 3s for real-time status
  });
}

// ============================================================================
// Graph Hooks
// ============================================================================

/**
 * Hook to get knowledge graph data.
 */
export function useKnowledgeGraph(limit?: number) {
  return useQuery({
    queryKey: queryKeys.knowledgeGraph(limit),
    queryFn: () => graphApi.getGraph(limit),
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

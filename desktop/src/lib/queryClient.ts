/**
 * TanStack Query configuration for Futurnal Desktop Shell
 */

import { QueryClient } from '@tanstack/react-query';
import type { AuditLogQuery } from '@/types/api';

/**
 * Query client with optimized defaults for desktop application.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Cache data for 5 minutes
      staleTime: 1000 * 60 * 5,
      // Keep unused data in cache for 10 minutes
      gcTime: 1000 * 60 * 10,
      // Retry failed requests once
      retry: 1,
      // Refetch on window focus for real-time data
      refetchOnWindowFocus: true,
      // Don't refetch on reconnect (desktop app is always "online")
      refetchOnReconnect: false,
    },
    mutations: {
      // Retry mutations once
      retry: 1,
    },
  },
});

/**
 * Query keys factory for consistent key management.
 */
export const queryKeys = {
  // Search
  search: (query: string) => ['search', query] as const,
  searchHistory: (limit?: number) => ['searchHistory', limit] as const,

  // Connectors
  connectors: ['connectors'] as const,
  connector: (id: string) => ['connectors', id] as const,

  // Privacy
  consent: (sourceId?: string) => ['consent', sourceId] as const,
  auditLogs: (query?: AuditLogQuery) => ['auditLogs', query] as const,

  // Orchestrator
  orchestratorStatus: ['orchestratorStatus'] as const,

  // Graph
  knowledgeGraph: (limit?: number) => ['knowledgeGraph', limit] as const,
};

/**
 * API Client Tests
 *
 * Tests for the Tauri IPC API wrappers.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { invoke } from '@tauri-apps/api/core';
import { searchApi, connectorsApi, orchestratorApi, graphApi, ApiError } from '@/lib/api';

// Type the mock
const mockInvoke = vi.mocked(invoke);

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('searchApi', () => {
    it('should call search_query with correct parameters', async () => {
      const mockResponse = {
        results: [],
        total: 0,
        query_id: 'test-id',
        intent: { primary: 'exploratory' },
        execution_time_ms: 10,
      };

      mockInvoke.mockResolvedValueOnce(mockResponse);

      const query = { query: 'test query', top_k: 10 };
      const result = await searchApi.search(query);

      expect(mockInvoke).toHaveBeenCalledWith('search_query', { query });
      expect(result).toEqual(mockResponse);
    });

    it('should return empty array for getHistory on error', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('Failed'));

      const result = await searchApi.getHistory();

      expect(result).toEqual([]);
    });
  });

  describe('connectorsApi', () => {
    it('should call list_sources', async () => {
      mockInvoke.mockResolvedValueOnce([]);

      const result = await connectorsApi.list();

      expect(mockInvoke).toHaveBeenCalledWith('list_sources', undefined);
      expect(result).toEqual([]);
    });

    it('should call pause_source with id', async () => {
      mockInvoke.mockResolvedValueOnce(undefined);

      await connectorsApi.pause('source-1');

      expect(mockInvoke).toHaveBeenCalledWith('pause_source', { id: 'source-1' });
    });
  });

  describe('orchestratorApi', () => {
    it('should call get_orchestrator_status', async () => {
      const mockStatus = {
        running: false,
        active_jobs: 0,
        pending_jobs: 0,
        failed_jobs: 0,
        sources: [],
        uptime_seconds: 0,
      };

      mockInvoke.mockResolvedValueOnce(mockStatus);

      const result = await orchestratorApi.getStatus();

      expect(mockInvoke).toHaveBeenCalledWith('get_orchestrator_status', undefined);
      expect(result).toEqual(mockStatus);
    });
  });

  describe('graphApi', () => {
    it('should call get_knowledge_graph with limit', async () => {
      const mockGraph = { nodes: [], links: [] };
      mockInvoke.mockResolvedValueOnce(mockGraph);

      const result = await graphApi.getGraph(500);

      expect(mockInvoke).toHaveBeenCalledWith('get_knowledge_graph', { limit: 500 });
      expect(result).toEqual(mockGraph);
    });
  });

  describe('Error handling', () => {
    it('should wrap string errors in ApiError', async () => {
      mockInvoke.mockRejectedValue('Backend error message');

      await expect(connectorsApi.list()).rejects.toThrow(ApiError);
    });

    it('should include error message in ApiError', async () => {
      mockInvoke.mockRejectedValue('Backend error message');

      await expect(connectorsApi.list()).rejects.toThrow('Backend error message');
    });
  });
});

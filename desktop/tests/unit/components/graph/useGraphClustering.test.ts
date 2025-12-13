/**
 * useGraphClustering Hook Tests
 *
 * Tests for the graph clustering functionality that groups large graphs
 * by node type for improved performance.
 */

import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import {
  useGraphClustering,
  isClusterNode,
  getClusterAwareSize,
} from '@/components/graph/useGraphClustering';
import type { GraphData, GraphNode, GraphLink, EntityType } from '@/types/api';

// Helper to create mock nodes
function createMockNodes(
  count: number,
  type: EntityType = 'Document',
  prefix: string = 'node'
): GraphNode[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `${prefix}-${type.toLowerCase()}-${i}`,
    label: `${type} ${i}`,
    node_type: type,
  }));
}

// Helper to create mock links
function createMockLinks(nodes: GraphNode[]): GraphLink[] {
  const links: GraphLink[] = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    links.push({
      source: nodes[i].id,
      target: nodes[i + 1].id,
      relationship: 'contains',
    });
  }
  return links;
}

describe('useGraphClustering', () => {
  describe('Below threshold behavior', () => {
    it('should return original data when below threshold', () => {
      const nodes = createMockNodes(100, 'Document');
      const links = createMockLinks(nodes);
      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 500 })
      );

      expect(result.current?.nodes.length).toBe(100);
      expect(result.current?.links.length).toBe(links.length);
    });

    it('should return undefined for undefined data', () => {
      const { result } = renderHook(() => useGraphClustering(undefined));

      expect(result.current).toBeUndefined();
    });

    it('should return original data when clustering is disabled', () => {
      const nodes = createMockNodes(600, 'Document');
      const links = createMockLinks(nodes);
      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 500, enabled: false })
      );

      expect(result.current?.nodes.length).toBe(600);
    });
  });

  describe('Clustering behavior', () => {
    it('should cluster when above threshold', () => {
      const nodes = [
        ...createMockNodes(100, 'Document'),
        ...createMockNodes(100, 'Email'),
        ...createMockNodes(100, 'Person'),
        ...createMockNodes(100, 'Event'),
        ...createMockNodes(100, 'Concept'),
        ...createMockNodes(100, 'Source'),
      ];
      const links = createMockLinks(nodes);
      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 500 })
      );

      // Should have fewer nodes due to clustering
      expect(result.current?.nodes.length).toBeLessThan(600);
    });

    it('should create cluster representatives for large type groups', () => {
      // Create 100 documents (above per-type threshold of 50)
      const nodes = [
        ...createMockNodes(100, 'Document'),
        ...createMockNodes(10, 'Person'), // Below threshold, not clustered
      ];
      const links: GraphLink[] = [];
      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 50 })
      );

      // Should have 1 Document cluster + 10 individual Person nodes
      expect(result.current?.nodes.length).toBe(11);

      // Find the cluster node
      const clusterNode = result.current?.nodes.find((n) => n.isCluster);
      expect(clusterNode).toBeDefined();
      expect(clusterNode?.node_type).toBe('Document');
      expect(clusterNode?.clusterSize).toBe(100);
      expect(clusterNode?.label).toBe('Document (100)');
    });

    it('should preserve small type groups as individual nodes', () => {
      const nodes = [
        ...createMockNodes(100, 'Document'),
        ...createMockNodes(30, 'Person'), // Below 50 threshold
      ];
      const links: GraphLink[] = [];
      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 50 })
      );

      // Person nodes should remain individual
      const personNodes = result.current?.nodes.filter(
        (n) => n.node_type === 'Person' && !n.isCluster
      );
      expect(personNodes?.length).toBe(30);
    });

    it('should store original node IDs in cluster', () => {
      const nodes = createMockNodes(100, 'Document');
      const data: GraphData = { nodes, links: [] };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 50 })
      );

      const clusterNode = result.current?.nodes.find((n) => n.isCluster);
      expect(clusterNode?.originalNodeIds).toHaveLength(100);
      expect(clusterNode?.originalNodeIds).toContain('node-document-0');
      expect(clusterNode?.originalNodeIds).toContain('node-document-99');
    });
  });

  describe('Link remapping', () => {
    it('should remap links to cluster nodes', () => {
      const docNodes = createMockNodes(100, 'Document');
      const personNodes = createMockNodes(5, 'Person');
      const nodes = [...docNodes, ...personNodes];

      // Create links from documents to persons
      const links: GraphLink[] = [
        { source: docNodes[0].id, target: personNodes[0].id, relationship: 'mentions' },
        { source: docNodes[50].id, target: personNodes[1].id, relationship: 'mentions' },
      ];

      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 50 })
      );

      // Links should be remapped to cluster
      const remappedLinks = result.current?.links;
      expect(remappedLinks?.length).toBeGreaterThan(0);

      // Links from documents should reference the cluster
      const clusterLinks = remappedLinks?.filter(
        (l) => l.source === 'cluster-Document' || l.target === 'cluster-Document'
      );
      expect(clusterLinks?.length).toBeGreaterThan(0);
    });

    it('should skip self-links within same cluster', () => {
      const docNodes = createMockNodes(100, 'Document');

      // Create links between documents (will all map to same cluster)
      const links: GraphLink[] = [
        { source: docNodes[0].id, target: docNodes[1].id, relationship: 'relates' },
        { source: docNodes[2].id, target: docNodes[3].id, relationship: 'relates' },
      ];

      const data: GraphData = { nodes: docNodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 50 })
      );

      // Should have no links (all were within same cluster)
      expect(result.current?.links.length).toBe(0);
    });

    it('should deduplicate cluster links', () => {
      const docNodes = createMockNodes(100, 'Document');
      const personNodes = createMockNodes(5, 'Person');
      const nodes = [...docNodes, ...personNodes];

      // Create multiple links from different docs to same person
      const links: GraphLink[] = [
        { source: docNodes[0].id, target: personNodes[0].id, relationship: 'mentions' },
        { source: docNodes[1].id, target: personNodes[0].id, relationship: 'mentions' },
        { source: docNodes[2].id, target: personNodes[0].id, relationship: 'mentions' },
      ];

      const data: GraphData = { nodes, links };

      const { result } = renderHook(() =>
        useGraphClustering(data, { threshold: 50 })
      );

      // Should have only one link from cluster to person
      const linksToPersonZero = result.current?.links.filter(
        (l) =>
          l.target === personNodes[0].id ||
          l.source === personNodes[0].id
      );
      expect(linksToPersonZero?.length).toBe(1);
    });
  });

  describe('Memoization', () => {
    it('should return same reference for unchanged data', () => {
      const nodes = createMockNodes(100, 'Document');
      const data: GraphData = { nodes, links: [] };

      const { result, rerender } = renderHook(
        ({ data }) => useGraphClustering(data, { threshold: 500 }),
        { initialProps: { data } }
      );

      const firstResult = result.current;
      rerender({ data });
      const secondResult = result.current;

      expect(firstResult).toBe(secondResult);
    });
  });
});

describe('isClusterNode', () => {
  it('should return true for cluster nodes', () => {
    const clusterNode = {
      id: 'cluster-Document',
      label: 'Document (100)',
      node_type: 'Document' as EntityType,
      isCluster: true,
      clusterSize: 100,
    };

    expect(isClusterNode(clusterNode)).toBe(true);
  });

  it('should return false for regular nodes', () => {
    const regularNode = {
      id: 'node-1',
      label: 'Regular Node',
      node_type: 'Document' as EntityType,
      isCluster: false,
    };

    expect(isClusterNode(regularNode)).toBe(false);
  });

  it('should return false for nodes without isCluster property', () => {
    const node = {
      id: 'node-1',
      label: 'Node',
      node_type: 'Document' as EntityType,
    };

    expect(isClusterNode(node as any)).toBe(false);
  });
});

describe('getClusterAwareSize', () => {
  const baseSize = 10;

  it('should return base size for non-cluster nodes', () => {
    const regularNode = {
      id: 'node-1',
      label: 'Node',
      node_type: 'Document' as EntityType,
      isCluster: false,
    };

    expect(getClusterAwareSize(regularNode, baseSize)).toBe(baseSize);
  });

  it('should return scaled size for cluster nodes', () => {
    const clusterNode = {
      id: 'cluster-Document',
      label: 'Document (100)',
      node_type: 'Document' as EntityType,
      isCluster: true,
      clusterSize: 100,
    };

    const size = getClusterAwareSize(clusterNode, baseSize);
    expect(size).toBeGreaterThan(baseSize);
  });

  it('should scale logarithmically with cluster size', () => {
    const smallCluster = {
      id: 'cluster-1',
      label: 'Small',
      node_type: 'Document' as EntityType,
      isCluster: true,
      clusterSize: 10,
    };

    const largeCluster = {
      id: 'cluster-2',
      label: 'Large',
      node_type: 'Document' as EntityType,
      isCluster: true,
      clusterSize: 1000,
    };

    const smallSize = getClusterAwareSize(smallCluster, baseSize);
    const largeSize = getClusterAwareSize(largeCluster, baseSize);

    // Large cluster should be bigger, but not 100x bigger (logarithmic)
    expect(largeSize).toBeGreaterThan(smallSize);
    expect(largeSize / smallSize).toBeLessThan(10);
  });

  it('should handle cluster with undefined size', () => {
    const clusterNode = {
      id: 'cluster-Document',
      label: 'Document',
      node_type: 'Document' as EntityType,
      isCluster: true,
      clusterSize: undefined,
    };

    // Should not throw and should return a valid size
    const size = getClusterAwareSize(clusterNode as any, baseSize);
    expect(size).toBeGreaterThan(0);
  });
});

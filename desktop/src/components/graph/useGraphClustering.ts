/**
 * useGraphClustering - Hook for clustering large graphs
 *
 * When graphs exceed the threshold (default: 500 nodes), this hook
 * groups nodes by type into cluster representatives to improve performance.
 *
 * Clusters show aggregate counts and can be expanded to show individual nodes.
 */

import { useMemo } from 'react';
import type { GraphData, GraphNode, GraphLink, EntityType } from '@/types/api';

interface ClusterOptions {
  /** Threshold above which clustering activates (default: 500) */
  threshold?: number;
  /** Enable/disable clustering (default: true) */
  enabled?: boolean;
}

/**
 * Extended node type that includes cluster information
 */
interface ClusteredNode extends GraphNode {
  /** Number of nodes in this cluster (undefined for non-clusters) */
  clusterSize?: number;
  /** Whether this is a cluster representative */
  isCluster?: boolean;
  /** Original node IDs if this is a cluster */
  originalNodeIds?: string[];
}

interface ClusteredGraphData {
  nodes: ClusteredNode[];
  links: GraphLink[];
}

/**
 * Hook that clusters large graphs by node type for performance
 *
 * @param data - Raw graph data
 * @param options - Clustering options
 * @returns Clustered graph data (or original if below threshold)
 */
export function useGraphClustering(
  data: GraphData | undefined,
  options: ClusterOptions = {}
): ClusteredGraphData | undefined {
  const { threshold = 500, enabled = true } = options;

  return useMemo(() => {
    if (!data) return undefined;

    // If clustering is disabled or below threshold, return original data
    if (!enabled || data.nodes.length < threshold) {
      return data as ClusteredGraphData;
    }

    // Group nodes by type
    const typeGroups = new Map<EntityType, GraphNode[]>();
    data.nodes.forEach((node) => {
      const nodeType = node.node_type || 'Document';
      const existing = typeGroups.get(nodeType) || [];
      typeGroups.set(nodeType, [...existing, node]);
    });

    // Create cluster nodes for large groups
    const clusterNodes: ClusteredNode[] = [];
    const nodeToCluster = new Map<string, string>();
    const clusterThresholdPerType = 50; // Cluster if type has more than 50 nodes

    typeGroups.forEach((nodes, type) => {
      if (nodes.length > clusterThresholdPerType) {
        // Create cluster representative
        const clusterId = `cluster-${type}`;
        clusterNodes.push({
          id: clusterId,
          label: `${type} (${nodes.length})`,
          node_type: type,
          isCluster: true,
          clusterSize: nodes.length,
          originalNodeIds: nodes.map((n) => n.id),
        });

        // Map original nodes to cluster
        nodes.forEach((n) => nodeToCluster.set(n.id, clusterId));
      } else {
        // Keep individual nodes
        nodes.forEach((n) => {
          clusterNodes.push({
            ...n,
            isCluster: false,
          });
        });
      }
    });

    // Rebuild links with cluster references
    const clusterLinks: GraphLink[] = [];
    const linkSet = new Set<string>(); // Prevent duplicate cluster links

    data.links.forEach((link) => {
      const sourceId = typeof link.source === 'string'
        ? link.source
        : (link.source as GraphNode).id;
      const targetId = typeof link.target === 'string'
        ? link.target
        : (link.target as GraphNode).id;

      const mappedSource = nodeToCluster.get(sourceId) || sourceId;
      const mappedTarget = nodeToCluster.get(targetId) || targetId;

      // Skip self-links to same cluster
      if (mappedSource === mappedTarget) return;

      // Prevent duplicate links
      const linkKey = `${mappedSource}-${mappedTarget}`;
      const reverseLinkKey = `${mappedTarget}-${mappedSource}`;
      if (linkSet.has(linkKey) || linkSet.has(reverseLinkKey)) return;

      linkSet.add(linkKey);

      clusterLinks.push({
        source: mappedSource,
        target: mappedTarget,
        relationship: link.relationship,
        weight: link.weight,
      });
    });

    return { nodes: clusterNodes, links: clusterLinks };
  }, [data, threshold, enabled]);
}

/**
 * Check if a node is a cluster representative
 */
export function isClusterNode(node: ClusteredNode): boolean {
  return node.isCluster === true;
}

/**
 * Get the display size for a node (larger for clusters)
 */
export function getClusterAwareSize(node: ClusteredNode, baseSize: number): number {
  if (!node.isCluster) return baseSize;
  // Scale cluster size logarithmically
  const clusterSize = node.clusterSize || 1;
  return baseSize * (1 + Math.log10(clusterSize) * 0.5);
}

export default useGraphClustering;

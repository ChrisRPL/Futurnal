"""
Community Detection Algorithms.

Implements multiple community detection algorithms:
1. Louvain: Fast modularity optimization
2. Leiden: Improved Louvain with guaranteed connectedness
3. DuallyPerceived: Combines structural + semantic (per Youtu-GraphRAG)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Represents a community of nodes."""
    id: str
    nodes: Set[str] = field(default_factory=set)
    level: int = 0
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Community properties
    internal_edges: int = 0
    external_edges: int = 0
    total_weight: float = 0.0

    # Semantic properties
    centroid: Optional[List[float]] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class Graph:
    """Simple graph representation for community detection."""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (src, dst) -> weight
    node_weights: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    node_embeddings: Dict[str, List[float]] = field(default_factory=dict)

    def add_edge(self, src: str, dst: str, weight: float = 1.0) -> None:
        """Add an edge to the graph."""
        self.nodes.add(src)
        self.nodes.add(dst)
        self.edges[(src, dst)] = weight
        self.edges[(dst, src)] = weight  # Undirected
        self.node_weights[src] += weight
        self.node_weights[dst] += weight

    def total_weight(self) -> float:
        """Total weight of all edges."""
        return sum(self.edges.values()) / 2  # Divide by 2 for undirected

    def neighbors(self, node: str) -> List[Tuple[str, float]]:
        """Get neighbors of a node with edge weights."""
        result = []
        for (src, dst), weight in self.edges.items():
            if src == node:
                result.append((dst, weight))
        return result


class CommunityDetector(ABC):
    """Abstract base class for community detection algorithms."""

    @abstractmethod
    def detect(self, graph: Graph) -> List[Community]:
        """
        Detect communities in the graph.

        Args:
            graph: Input graph

        Returns:
            List of detected communities
        """
        pass

    def detect_from_neo4j(
        self,
        neo4j_driver,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> List[Community]:
        """
        Detect communities from Neo4j graph.

        Args:
            neo4j_driver: Neo4j driver instance
            node_labels: Optional filter for node labels
            relationship_types: Optional filter for relationship types

        Returns:
            Detected communities
        """
        graph = self._load_from_neo4j(neo4j_driver, node_labels, relationship_types)
        return self.detect(graph)

    def _load_from_neo4j(
        self,
        neo4j_driver,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> Graph:
        """Load graph from Neo4j."""
        graph = Graph()

        # Build node filter
        label_filter = ""
        if node_labels:
            label_filter = " WHERE " + " OR ".join([f"n:{label}" for label in node_labels])

        # Query nodes
        node_query = f"""
        MATCH (n){label_filter}
        RETURN n.id as id, n.name as name
        """

        with neo4j_driver.session() as session:
            result = session.run(node_query)
            for record in result:
                graph.nodes.add(record["id"])

        # Build relationship filter
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)

        # Query edges
        edge_query = f"""
        MATCH (a)-[r{rel_filter}]->(b)
        {label_filter.replace('n:', 'a:')}
        RETURN a.id as src, b.id as dst, r.confidence as weight
        """

        with neo4j_driver.session() as session:
            result = session.run(edge_query)
            for record in result:
                weight = record["weight"] or 1.0
                if record["src"] in graph.nodes and record["dst"] in graph.nodes:
                    graph.add_edge(record["src"], record["dst"], weight)

        return graph


class LouvainDetector(CommunityDetector):
    """
    Louvain algorithm for community detection.

    Optimizes modularity through two phases:
    1. Local optimization: Move nodes to maximize modularity gain
    2. Aggregation: Build new network of communities
    """

    def __init__(
        self,
        resolution: float = 1.0,
        max_iterations: int = 100,
        min_modularity_gain: float = 1e-7
    ):
        self.resolution = resolution
        self.max_iterations = max_iterations
        self.min_modularity_gain = min_modularity_gain

    def detect(self, graph: Graph) -> List[Community]:
        """Detect communities using Louvain algorithm."""
        if not graph.nodes:
            return []

        # Initialize: each node is its own community
        node_to_community: Dict[str, str] = {node: node for node in graph.nodes}
        community_nodes: Dict[str, Set[str]] = {node: {node} for node in graph.nodes}

        # Community internal weights
        community_internal: Dict[str, float] = defaultdict(float)
        community_total: Dict[str, float] = defaultdict(float)

        for node in graph.nodes:
            community_total[node] = graph.node_weights[node]

        m = graph.total_weight()
        if m == 0:
            # No edges - each node is its own community
            return [
                Community(id=f"c_{node}", nodes={node}, level=0)
                for node in graph.nodes
            ]

        # Phase 1: Local optimization
        improved = True
        iteration = 0

        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1

            for node in list(graph.nodes):
                current_community = node_to_community[node]

                # Calculate modularity gain for moving to each neighbor's community
                best_community = current_community
                best_gain = 0.0

                # Get neighbor communities
                neighbor_communities: Dict[str, float] = defaultdict(float)
                for neighbor, weight in graph.neighbors(node):
                    nc = node_to_community[neighbor]
                    neighbor_communities[nc] += weight

                # Calculate gain for each candidate community
                for candidate, weight_to_candidate in neighbor_communities.items():
                    if candidate == current_community:
                        continue

                    gain = self._modularity_gain(
                        node, current_community, candidate,
                        weight_to_candidate, graph, community_total, m
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_community = candidate

                # Move node if there's improvement
                if best_gain > self.min_modularity_gain:
                    self._move_node(
                        node, current_community, best_community,
                        node_to_community, community_nodes,
                        community_internal, community_total, graph
                    )
                    improved = True

        # Build communities
        final_communities = []
        for comm_id, nodes in community_nodes.items():
            if nodes:
                community = Community(
                    id=f"louvain_{comm_id}",
                    nodes=nodes,
                    level=0,
                    internal_edges=int(community_internal.get(comm_id, 0)),
                    total_weight=community_total.get(comm_id, 0)
                )
                final_communities.append(community)

        logger.info(f"Louvain found {len(final_communities)} communities in {iteration} iterations")
        return final_communities

    def _modularity_gain(
        self,
        node: str,
        current_community: str,
        target_community: str,
        weight_to_target: float,
        graph: Graph,
        community_total: Dict[str, float],
        m: float
    ) -> float:
        """Calculate modularity gain for moving node to target community."""
        ki = graph.node_weights[node]
        sigma_tot = community_total[target_community]

        # Gain = weight_to_target - ki * sigma_tot / m
        gain = weight_to_target - (ki * sigma_tot * self.resolution) / (2 * m)
        return gain

    def _move_node(
        self,
        node: str,
        from_comm: str,
        to_comm: str,
        node_to_community: Dict[str, str],
        community_nodes: Dict[str, Set[str]],
        community_internal: Dict[str, float],
        community_total: Dict[str, float],
        graph: Graph
    ) -> None:
        """Move a node from one community to another."""
        # Update mappings
        node_to_community[node] = to_comm
        community_nodes[from_comm].discard(node)
        community_nodes[to_comm].add(node)

        # Update community weights
        ki = graph.node_weights[node]
        community_total[from_comm] -= ki
        community_total[to_comm] += ki


class LeidenDetector(CommunityDetector):
    """
    Leiden algorithm for community detection.

    Improvement over Louvain that guarantees connected communities
    and uses a refinement phase for better quality.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        max_iterations: int = 100,
        theta: float = 0.01  # Quality threshold
    ):
        self.resolution = resolution
        self.max_iterations = max_iterations
        self.theta = theta

    def detect(self, graph: Graph) -> List[Community]:
        """Detect communities using Leiden algorithm."""
        if not graph.nodes:
            return []

        # Initialize partitions
        node_to_community: Dict[str, str] = {node: node for node in graph.nodes}
        communities: Dict[str, Set[str]] = {node: {node} for node in graph.nodes}

        m = graph.total_weight()
        if m == 0:
            return [
                Community(id=f"c_{node}", nodes={node}, level=0)
                for node in graph.nodes
            ]

        # Main loop
        for iteration in range(self.max_iterations):
            # Phase 1: Local moving (similar to Louvain)
            changed = self._local_moving_phase(
                graph, node_to_community, communities, m
            )

            if not changed:
                break

            # Phase 2: Refinement
            self._refinement_phase(graph, node_to_community, communities, m)

            # Phase 3: Aggregate network
            graph, node_to_community, communities = self._aggregate_network(
                graph, node_to_community, communities
            )

            if len(communities) == len(graph.nodes):
                break

        # Build final communities
        final_communities = []
        for comm_id, nodes in communities.items():
            if nodes:
                community = Community(
                    id=f"leiden_{comm_id}",
                    nodes=nodes,
                    level=0
                )
                final_communities.append(community)

        logger.info(f"Leiden found {len(final_communities)} communities")
        return final_communities

    def _local_moving_phase(
        self,
        graph: Graph,
        node_to_community: Dict[str, str],
        communities: Dict[str, Set[str]],
        m: float
    ) -> bool:
        """Perform local moving phase."""
        changed = False
        queue = list(graph.nodes)
        random.shuffle(queue)

        while queue:
            node = queue.pop()
            current_comm = node_to_community[node]

            # Find best community
            best_comm = current_comm
            best_gain = 0.0

            neighbor_comms: Dict[str, float] = defaultdict(float)
            for neighbor, weight in graph.neighbors(node):
                neighbor_comms[node_to_community[neighbor]] += weight

            for comm, weight in neighbor_comms.items():
                if comm == current_comm:
                    continue

                gain = weight - self.resolution * graph.node_weights[node] * \
                       sum(graph.node_weights[n] for n in communities[comm]) / (2 * m)

                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm

            if best_comm != current_comm and best_gain > self.theta:
                # Move node
                communities[current_comm].discard(node)
                communities[best_comm].add(node)
                node_to_community[node] = best_comm
                changed = True

                # Add neighbors back to queue
                for neighbor, _ in graph.neighbors(node):
                    if neighbor not in queue:
                        queue.append(neighbor)

        return changed

    def _refinement_phase(
        self,
        graph: Graph,
        node_to_community: Dict[str, str],
        communities: Dict[str, Set[str]],
        m: float
    ) -> None:
        """Refinement phase to ensure connected communities."""
        # For each community, ensure it's connected
        for comm_id in list(communities.keys()):
            nodes = communities[comm_id]
            if len(nodes) <= 1:
                continue

            # Find connected components within community
            components = self._find_connected_components(graph, nodes)

            if len(components) > 1:
                # Split into separate communities
                for i, component in enumerate(components[1:], 1):
                    new_comm_id = f"{comm_id}_split_{i}"
                    communities[new_comm_id] = component
                    for node in component:
                        node_to_community[node] = new_comm_id
                    communities[comm_id] -= component

    def _find_connected_components(
        self,
        graph: Graph,
        nodes: Set[str]
    ) -> List[Set[str]]:
        """Find connected components within a set of nodes."""
        remaining = set(nodes)
        components = []

        while remaining:
            start = next(iter(remaining))
            component = set()
            queue = [start]

            while queue:
                node = queue.pop()
                if node in component:
                    continue
                component.add(node)
                remaining.discard(node)

                for neighbor, _ in graph.neighbors(node):
                    if neighbor in remaining and neighbor not in component:
                        queue.append(neighbor)

            components.append(component)

        return components

    def _aggregate_network(
        self,
        graph: Graph,
        node_to_community: Dict[str, str],
        communities: Dict[str, Set[str]]
    ) -> Tuple[Graph, Dict[str, str], Dict[str, Set[str]]]:
        """Build aggregated network for next iteration."""
        new_graph = Graph()
        new_node_to_comm = {}
        new_communities = {}

        # Create super-nodes
        for comm_id, nodes in communities.items():
            if nodes:
                new_graph.nodes.add(comm_id)
                new_node_to_comm[comm_id] = comm_id
                new_communities[comm_id] = {comm_id}

        # Create super-edges
        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        for (src, dst), weight in graph.edges.items():
            src_comm = node_to_community[src]
            dst_comm = node_to_community[dst]
            if src_comm != dst_comm:
                edge_key = tuple(sorted([src_comm, dst_comm]))
                edge_weights[edge_key] += weight

        for (src, dst), weight in edge_weights.items():
            new_graph.add_edge(src, dst, weight)

        return new_graph, new_node_to_comm, new_communities


class DuallyPerceivedDetector(CommunityDetector):
    """
    Dually-Perceived Community Detection (per Youtu-GraphRAG).

    Combines:
    1. Structural topology (graph connectivity)
    2. Subgraph semantics (embedding similarity)

    For comprehensive knowledge organization.
    """

    def __init__(
        self,
        structural_weight: float = 0.6,
        semantic_weight: float = 0.4,
        resolution: float = 1.0,
        similarity_threshold: float = 0.7
    ):
        self.structural_weight = structural_weight
        self.semantic_weight = semantic_weight
        self.resolution = resolution
        self.similarity_threshold = similarity_threshold

        # Use Leiden as base structural detector
        self.structural_detector = LeidenDetector(resolution=resolution)

    def detect(self, graph: Graph) -> List[Community]:
        """Detect communities using dual perception."""
        # Phase 1: Structural detection
        structural_communities = self.structural_detector.detect(graph)

        if not graph.node_embeddings:
            # No embeddings - return structural communities
            return structural_communities

        # Phase 2: Semantic refinement
        refined_communities = self._semantic_refinement(
            graph, structural_communities
        )

        # Phase 3: Compute community centroids
        for community in refined_communities:
            community.centroid = self._compute_centroid(graph, community.nodes)

        return refined_communities

    def _semantic_refinement(
        self,
        graph: Graph,
        communities: List[Community]
    ) -> List[Community]:
        """Refine communities based on semantic similarity."""
        refined = []

        for community in communities:
            if len(community.nodes) <= 2:
                refined.append(community)
                continue

            # Check semantic coherence
            nodes_list = list(community.nodes)
            coherent_groups = self._find_semantic_groups(graph, nodes_list)

            if len(coherent_groups) == 1:
                refined.append(community)
            else:
                # Split community based on semantic groups
                for i, group in enumerate(coherent_groups):
                    new_community = Community(
                        id=f"{community.id}_sem_{i}",
                        nodes=group,
                        level=community.level,
                        parent_id=community.parent_id
                    )
                    refined.append(new_community)

        return refined

    def _find_semantic_groups(
        self,
        graph: Graph,
        nodes: List[str]
    ) -> List[Set[str]]:
        """Find semantically coherent groups within nodes."""
        if len(nodes) <= 1:
            return [set(nodes)]

        # Build similarity matrix
        embeddings = []
        valid_nodes = []
        for node in nodes:
            if node in graph.node_embeddings:
                embeddings.append(graph.node_embeddings[node])
                valid_nodes.append(node)

        if not embeddings:
            return [set(nodes)]

        # Cluster based on similarity
        groups: List[Set[str]] = []
        assigned = set()

        for i, node in enumerate(valid_nodes):
            if node in assigned:
                continue

            group = {node}
            assigned.add(node)

            for j, other in enumerate(valid_nodes):
                if other in assigned:
                    continue

                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self.similarity_threshold:
                    group.add(other)
                    assigned.add(other)

            groups.append(group)

        # Add unassigned nodes to closest group
        for node in nodes:
            if node not in assigned:
                if groups:
                    groups[0].add(node)
                else:
                    groups.append({node})

        return groups

    def _compute_centroid(
        self,
        graph: Graph,
        nodes: Set[str]
    ) -> Optional[List[float]]:
        """Compute centroid embedding for a community."""
        embeddings = []
        for node in nodes:
            if node in graph.node_embeddings:
                embeddings.append(graph.node_embeddings[node])

        if not embeddings:
            return None

        # Average embeddings
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += v

        return [v / len(embeddings) for v in centroid]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

"""
Community Hierarchy Management.

Builds and maintains hierarchical community structure for:
- Multi-level knowledge organization
- Top-down filtering
- Bottom-up reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

from .detection import Community, Graph, CommunityDetector, LeidenDetector

logger = logging.getLogger(__name__)


@dataclass
class CommunityLevel:
    """Represents a level in the community hierarchy."""
    level: int
    communities: List[Community]
    resolution: float
    modularity: float = 0.0
    coverage: float = 1.0  # Fraction of nodes covered


@dataclass
class CommunityHierarchy:
    """
    Hierarchical community structure.

    Organizes communities into a tree structure for:
    - Efficient navigation
    - Multi-level summarization
    - Scope-based retrieval
    """
    levels: List[CommunityLevel] = field(default_factory=list)
    root_communities: List[str] = field(default_factory=list)

    # Index structures
    community_index: Dict[str, Community] = field(default_factory=dict)
    node_to_communities: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    parent_child_map: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    # Metadata
    total_nodes: int = 0
    total_communities: int = 0
    max_depth: int = 0

    def get_community(self, community_id: str) -> Optional[Community]:
        """Get a community by ID."""
        return self.community_index.get(community_id)

    def get_communities_for_node(self, node_id: str) -> List[Community]:
        """Get all communities containing a node."""
        comm_ids = self.node_to_communities.get(node_id, [])
        return [self.community_index[cid] for cid in comm_ids if cid in self.community_index]

    def get_communities_at_level(self, level: int) -> List[Community]:
        """Get all communities at a specific level."""
        if 0 <= level < len(self.levels):
            return self.levels[level].communities
        return []

    def get_children(self, community_id: str) -> List[Community]:
        """Get child communities."""
        child_ids = self.parent_child_map.get(community_id, [])
        return [self.community_index[cid] for cid in child_ids if cid in self.community_index]

    def get_ancestors(self, community_id: str) -> List[Community]:
        """Get all ancestor communities (path to root)."""
        ancestors = []
        comm = self.community_index.get(community_id)
        while comm and comm.parent_id:
            parent = self.community_index.get(comm.parent_id)
            if parent:
                ancestors.append(parent)
                comm = parent
            else:
                break
        return ancestors

    def get_subtree_nodes(self, community_id: str) -> Set[str]:
        """Get all nodes in the subtree rooted at a community."""
        nodes = set()
        queue = [community_id]

        while queue:
            cid = queue.pop()
            comm = self.community_index.get(cid)
            if comm:
                nodes.update(comm.nodes)
                queue.extend(self.parent_child_map.get(cid, []))

        return nodes


class HierarchyBuilder:
    """
    Builds hierarchical community structure.

    Uses recursive community detection with varying resolutions
    to create a multi-level hierarchy.
    """

    def __init__(
        self,
        detector: Optional[CommunityDetector] = None,
        min_community_size: int = 2,
        max_levels: int = 5,
        resolution_decay: float = 0.5
    ):
        self.detector = detector or LeidenDetector()
        self.min_community_size = min_community_size
        self.max_levels = max_levels
        self.resolution_decay = resolution_decay

    def build(self, graph: Graph) -> CommunityHierarchy:
        """
        Build hierarchical community structure.

        Args:
            graph: Input graph

        Returns:
            CommunityHierarchy with multiple levels
        """
        hierarchy = CommunityHierarchy()
        hierarchy.total_nodes = len(graph.nodes)

        # Build levels recursively
        current_resolution = 1.0

        for level in range(self.max_levels):
            # Detect communities at this resolution
            if isinstance(self.detector, LeidenDetector):
                self.detector.resolution = current_resolution

            if level == 0:
                communities = self.detector.detect(graph)
            else:
                # Detect within each parent community
                communities = self._detect_within_communities(
                    graph, hierarchy.levels[-1].communities, level
                )

            if not communities:
                break

            # Filter small communities
            communities = [c for c in communities if len(c.nodes) >= self.min_community_size]

            if not communities:
                break

            # Update level for communities
            for comm in communities:
                comm.level = level
                comm.id = f"L{level}_{comm.id}"

            # Create level
            level_obj = CommunityLevel(
                level=level,
                communities=communities,
                resolution=current_resolution
            )
            hierarchy.levels.append(level_obj)

            # Update indices
            for comm in communities:
                hierarchy.community_index[comm.id] = comm
                hierarchy.total_communities += 1
                for node in comm.nodes:
                    hierarchy.node_to_communities[node].append(comm.id)

            # Set root communities for first level
            if level == 0:
                hierarchy.root_communities = [c.id for c in communities]

            # Decrease resolution for finer granularity
            current_resolution *= self.resolution_decay

            # Stop if we have too many small communities
            avg_size = sum(len(c.nodes) for c in communities) / len(communities)
            if avg_size < self.min_community_size * 2:
                break

        # Build parent-child relationships
        self._build_parent_child_links(hierarchy)

        hierarchy.max_depth = len(hierarchy.levels)
        logger.info(
            f"Built hierarchy with {hierarchy.max_depth} levels, "
            f"{hierarchy.total_communities} communities"
        )

        return hierarchy

    def build_from_neo4j(
        self,
        neo4j_driver,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> CommunityHierarchy:
        """Build hierarchy from Neo4j graph."""
        # Load graph from Neo4j
        graph = self._load_graph_from_neo4j(neo4j_driver, node_labels, relationship_types)

        # Build hierarchy
        return self.build(graph)

    def _detect_within_communities(
        self,
        graph: Graph,
        parent_communities: List[Community],
        level: int
    ) -> List[Community]:
        """Detect sub-communities within parent communities."""
        all_communities = []

        for parent in parent_communities:
            # Create subgraph
            subgraph = self._create_subgraph(graph, parent.nodes)

            if len(subgraph.nodes) < self.min_community_size:
                continue

            # Detect communities in subgraph
            sub_communities = self.detector.detect(subgraph)

            for comm in sub_communities:
                comm.parent_id = parent.id
                all_communities.append(comm)

        return all_communities

    def _create_subgraph(self, graph: Graph, nodes: Set[str]) -> Graph:
        """Create a subgraph containing only specified nodes."""
        subgraph = Graph()
        subgraph.nodes = set(nodes)

        for (src, dst), weight in graph.edges.items():
            if src in nodes and dst in nodes:
                subgraph.edges[(src, dst)] = weight
                subgraph.node_weights[src] += weight
                subgraph.node_weights[dst] += weight

        # Copy embeddings
        for node in nodes:
            if node in graph.node_embeddings:
                subgraph.node_embeddings[node] = graph.node_embeddings[node]

        return subgraph

    def _build_parent_child_links(self, hierarchy: CommunityHierarchy) -> None:
        """Build parent-child relationships between communities."""
        for level in hierarchy.levels:
            for comm in level.communities:
                if comm.parent_id:
                    hierarchy.parent_child_map[comm.parent_id].append(comm.id)
                    parent = hierarchy.community_index.get(comm.parent_id)
                    if parent:
                        parent.child_ids.append(comm.id)

    def _load_graph_from_neo4j(
        self,
        neo4j_driver,
        node_labels: Optional[List[str]],
        relationship_types: Optional[List[str]]
    ) -> Graph:
        """Load graph from Neo4j."""
        graph = Graph()

        # Query nodes
        label_clause = ""
        if node_labels:
            label_clause = " WHERE " + " OR ".join([f"n:{l}" for l in node_labels])

        node_query = f"MATCH (n){label_clause} RETURN n.id as id"

        with neo4j_driver.session() as session:
            for record in session.run(node_query):
                graph.nodes.add(record["id"])

        # Query edges
        rel_clause = ""
        if relationship_types:
            rel_clause = ":" + "|".join(relationship_types)

        edge_query = f"""
        MATCH (a)-[r{rel_clause}]->(b)
        {label_clause.replace('n:', 'a:')}
        RETURN a.id as src, b.id as dst, coalesce(r.weight, r.confidence, 1.0) as weight
        """

        with neo4j_driver.session() as session:
            for record in session.run(edge_query):
                if record["src"] in graph.nodes and record["dst"] in graph.nodes:
                    graph.add_edge(record["src"], record["dst"], record["weight"])

        return graph

    def update_hierarchy(
        self,
        hierarchy: CommunityHierarchy,
        graph: Graph,
        new_nodes: Set[str]
    ) -> CommunityHierarchy:
        """
        Incrementally update hierarchy with new nodes.

        Args:
            hierarchy: Existing hierarchy
            graph: Updated graph
            new_nodes: Newly added nodes

        Returns:
            Updated hierarchy
        """
        # For each new node, find best community to join
        for node in new_nodes:
            best_community = None
            best_score = 0.0

            # Check communities at level 0
            for comm in hierarchy.levels[0].communities if hierarchy.levels else []:
                score = self._compute_membership_score(node, comm, graph)
                if score > best_score:
                    best_score = score
                    best_community = comm

            if best_community and best_score > 0.3:
                best_community.nodes.add(node)
                hierarchy.node_to_communities[node].append(best_community.id)
            else:
                # Create new community for orphan node
                new_comm = Community(
                    id=f"L0_new_{node}",
                    nodes={node},
                    level=0
                )
                if hierarchy.levels:
                    hierarchy.levels[0].communities.append(new_comm)
                hierarchy.community_index[new_comm.id] = new_comm
                hierarchy.node_to_communities[node].append(new_comm.id)

        hierarchy.total_nodes = len(graph.nodes)
        return hierarchy

    def _compute_membership_score(
        self,
        node: str,
        community: Community,
        graph: Graph
    ) -> float:
        """Compute how well a node fits in a community."""
        # Count edges to community members
        edges_to_community = 0
        total_edges = 0

        for (src, dst), weight in graph.edges.items():
            if src == node:
                total_edges += 1
                if dst in community.nodes:
                    edges_to_community += 1
            elif dst == node:
                total_edges += 1
                if src in community.nodes:
                    edges_to_community += 1

        if total_edges == 0:
            return 0.0

        return edges_to_community / total_edges

"""PKG Community Detection for Knowledge Gap Analysis.

AGI Phase 4: Detects isolated clusters and structural gaps in the
Personal Knowledge Graph for proactive insight generation.

Research Foundation:
- Curiosity-driven Autotelic AI (Oudeyer 2024): Information gain scoring
- DyMemR (2024): Memory forgetting curve analysis
- Community Detection: Louvain algorithm for graph clustering

Key Innovation:
Unlike standard graph analysis that reports static structure, this module
implements dynamic community detection that identifies:
1. Isolated knowledge islands (potential bridge opportunities)
2. Forgotten memories (DyMemR forgetting curve)
3. Missing connections to aspirations
4. Information gain opportunities

Option B Compliance:
- Graph analysis only, no model parameter updates
- All discoveries expressed as natural language for token priors
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A detected community (cluster) in the PKG.

    Represents a group of related nodes that form a coherent
    knowledge domain.
    """

    community_id: str
    nodes: Set[str] = field(default_factory=set)
    central_topics: List[str] = field(default_factory=list)
    density: float = 0.0  # Internal connectivity
    modularity_contribution: float = 0.0
    creation_dates: List[datetime] = field(default_factory=list)
    last_activity: Optional[datetime] = None

    @property
    def size(self) -> int:
        """Number of nodes in community."""
        return len(self.nodes)

    @property
    def is_isolated(self) -> bool:
        """Check if community has few external connections."""
        # Will be set by detector
        return getattr(self, "_is_isolated", False)


@dataclass
class MemoryDecay:
    """Memory decay analysis result following DyMemR principles.

    DyMemR (2024) models memory forgetting using decay curves
    to identify memories that need reinforcement.
    """

    node_id: str
    title: str
    last_access: datetime
    access_count: int
    decay_score: float  # 0 = forgotten, 1 = fresh
    importance_score: float  # Based on connections
    retrieval_probability: float  # P(recall)

    @property
    def needs_reinforcement(self) -> bool:
        """Check if memory needs reinforcement."""
        return self.decay_score < 0.3 and self.importance_score > 0.5


@dataclass
class BridgeOpportunity:
    """Opportunity to bridge two communities.

    When two isolated communities have potential semantic connections
    but no explicit links in the PKG.
    """

    community_a: str
    community_b: str
    shared_concepts: List[str]
    information_gain: float  # Expected info gain from bridging
    suggested_connections: List[Tuple[str, str]]  # (node_a, node_b) pairs


class PKGCommunityDetector:
    """Detects communities and structural gaps in the PKG.

    AGI Phase 4 core component for curiosity-driven gap detection.

    Capabilities:
    1. Community detection using modified Louvain algorithm
    2. Isolation detection for knowledge islands
    3. DyMemR-style forgetting analysis
    4. Information gain computation for exploration prioritization
    5. Bridge opportunity detection between communities

    Example:
        detector = PKGCommunityDetector()
        communities = detector.detect_communities(pkg_graph)
        isolated = detector.find_isolated_communities(communities)
        decaying = detector.analyze_memory_decay(pkg_graph)
        bridges = detector.find_bridge_opportunities(communities)
    """

    # Configuration
    MIN_COMMUNITY_SIZE = 3  # Minimum nodes for valid community
    ISOLATION_THRESHOLD = 0.1  # Max external edge ratio for isolation
    DECAY_HALF_LIFE_DAYS = 30  # Memory decay half-life
    IMPORTANCE_DECAY_FACTOR = 0.95  # Importance decay per level from aspirations

    def __init__(
        self,
        min_community_size: int = MIN_COMMUNITY_SIZE,
        isolation_threshold: float = ISOLATION_THRESHOLD,
        decay_half_life_days: int = DECAY_HALF_LIFE_DAYS,
    ):
        """Initialize community detector.

        Args:
            min_community_size: Minimum nodes for a valid community
            isolation_threshold: Maximum external edge ratio for isolation
            decay_half_life_days: DyMemR forgetting curve half-life
        """
        self._min_community_size = min_community_size
        self._isolation_threshold = isolation_threshold
        self._decay_half_life_days = decay_half_life_days

        logger.info(
            f"PKGCommunityDetector initialized "
            f"(min_size={min_community_size}, "
            f"isolation_threshold={isolation_threshold})"
        )

    def detect_communities(
        self,
        pkg_graph: Any,
    ) -> List[Community]:
        """Detect communities in the PKG using Louvain-style algorithm.

        Implements a simplified Louvain algorithm optimized for
        personal knowledge graphs (typically <10k nodes).

        Args:
            pkg_graph: Personal Knowledge Graph (Neo4j/NetworkX compatible)

        Returns:
            List of detected communities
        """
        # Get graph structure
        nodes, edges = self._extract_graph_structure(pkg_graph)

        if not nodes:
            logger.debug("Empty graph, no communities")
            return []

        # Initialize each node as its own community
        node_to_community: Dict[str, str] = {n: n for n in nodes}
        community_members: Dict[str, Set[str]] = {n: {n} for n in nodes}

        # Build adjacency for modularity computation
        adjacency = self._build_adjacency(nodes, edges)
        total_edges = len(edges) if edges else 1

        # Louvain Phase 1: Local optimization
        improved = True
        iterations = 0
        max_iterations = 100

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for node in nodes:
                current_community = node_to_community[node]
                neighbors = adjacency.get(node, set())

                # Find best community among neighbors
                best_community = current_community
                best_gain = 0.0

                neighbor_communities = set()
                for neighbor in neighbors:
                    neighbor_communities.add(node_to_community[neighbor])

                for target_community in neighbor_communities:
                    if target_community == current_community:
                        continue

                    # Calculate modularity gain
                    gain = self._calculate_modularity_gain(
                        node,
                        current_community,
                        target_community,
                        community_members,
                        adjacency,
                        total_edges,
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_community = target_community

                # Move node if beneficial
                if best_community != current_community:
                    # Remove from old community
                    community_members[current_community].discard(node)
                    if not community_members[current_community]:
                        del community_members[current_community]

                    # Add to new community
                    if best_community not in community_members:
                        community_members[best_community] = set()
                    community_members[best_community].add(node)
                    node_to_community[node] = best_community

                    improved = True

        logger.debug(
            f"Community detection converged in {iterations} iterations, "
            f"found {len(community_members)} communities"
        )

        # Build Community objects
        communities = []
        for community_id, members in community_members.items():
            if len(members) >= self._min_community_size:
                community = Community(
                    community_id=community_id,
                    nodes=members,
                )

                # Calculate density
                internal_edges = sum(
                    1 for n in members
                    for neighbor in adjacency.get(n, set())
                    if neighbor in members
                ) // 2
                max_edges = len(members) * (len(members) - 1) // 2
                community.density = internal_edges / max(max_edges, 1)

                # Extract central topics (nodes with most connections)
                node_degrees = {
                    n: len(adjacency.get(n, set()) & members)
                    for n in members
                }
                sorted_nodes = sorted(
                    node_degrees.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                community.central_topics = [n for n, _ in sorted_nodes[:3]]

                communities.append(community)

        return communities

    def _extract_graph_structure(
        self,
        pkg_graph: Any,
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract nodes and edges from PKG.

        Handles different graph formats (Neo4j, NetworkX, dict).
        """
        nodes: List[str] = []
        edges: List[Tuple[str, str]] = []

        # Handle different graph types
        if hasattr(pkg_graph, "nodes") and hasattr(pkg_graph, "edges"):
            # NetworkX-like
            nodes = list(pkg_graph.nodes())
            edges = list(pkg_graph.edges())

        elif isinstance(pkg_graph, dict):
            # Dict format: {"nodes": [...], "edges": [...]}
            nodes = pkg_graph.get("nodes", [])
            raw_edges = pkg_graph.get("edges", [])
            edges = [
                (e["source"], e["target"])
                for e in raw_edges
                if isinstance(e, dict) and "source" in e and "target" in e
            ]

        elif hasattr(pkg_graph, "get_all_nodes"):
            # Custom PKG interface
            try:
                node_objs = pkg_graph.get_all_nodes()
                nodes = [n.id if hasattr(n, "id") else str(n) for n in node_objs]
                edge_objs = pkg_graph.get_all_edges()
                edges = [
                    (e.source_id, e.target_id)
                    for e in edge_objs
                    if hasattr(e, "source_id")
                ]
            except Exception as e:
                logger.warning(f"Error extracting graph structure: {e}")

        return nodes, edges

    def _build_adjacency(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
    ) -> Dict[str, Set[str]]:
        """Build adjacency list from edges."""
        adjacency: Dict[str, Set[str]] = {n: set() for n in nodes}

        for source, target in edges:
            if source in adjacency:
                adjacency[source].add(target)
            if target in adjacency:
                adjacency[target].add(source)

        return adjacency

    def _calculate_modularity_gain(
        self,
        node: str,
        old_community: str,
        new_community: str,
        community_members: Dict[str, Set[str]],
        adjacency: Dict[str, Set[str]],
        total_edges: int,
    ) -> float:
        """Calculate modularity gain from moving node."""
        neighbors = adjacency.get(node, set())
        node_degree = len(neighbors)

        # Edges to new community
        edges_to_new = len(neighbors & community_members.get(new_community, set()))

        # Edges to old community (excluding self)
        old_members = community_members.get(old_community, set()) - {node}
        edges_to_old = len(neighbors & old_members)

        # Simplified modularity gain
        m = total_edges
        if m == 0:
            return 0.0

        # Gain = edges_to_new - expected - (edges_to_old - expected)
        gain = (edges_to_new - edges_to_old) / m

        return gain

    def find_isolated_communities(
        self,
        communities: List[Community],
        pkg_graph: Any,
    ) -> List[Community]:
        """Find communities with few external connections.

        These represent knowledge islands that could benefit from bridging.

        Args:
            communities: Detected communities
            pkg_graph: Original graph for edge analysis

        Returns:
            List of isolated communities
        """
        _, edges = self._extract_graph_structure(pkg_graph)
        edge_set = set(edges) | {(t, s) for s, t in edges}

        isolated = []

        for community in communities:
            members = community.nodes

            # Count external edges
            internal_edges = 0
            external_edges = 0

            for source, target in edge_set:
                source_in = source in members
                target_in = target in members

                if source_in and target_in:
                    internal_edges += 1
                elif source_in or target_in:
                    external_edges += 1

            total_edges = internal_edges + external_edges
            external_ratio = external_edges / max(total_edges, 1)

            # Mark as isolated if external ratio is low
            if external_ratio < self._isolation_threshold:
                community._is_isolated = True
                isolated.append(community)

                logger.debug(
                    f"Isolated community: {community.community_id} "
                    f"(external_ratio={external_ratio:.3f})"
                )

        return isolated

    def analyze_memory_decay(
        self,
        pkg_graph: Any,
        access_history: Optional[Dict[str, List[datetime]]] = None,
    ) -> List[MemoryDecay]:
        """Analyze memory decay using DyMemR forgetting curve.

        DyMemR models forgetting with exponential decay:
        P(recall) = e^(-t/λ) where λ is the half-life

        Identifies memories that are:
        1. Important (high connectivity)
        2. Decaying (not accessed recently)
        3. Need reinforcement

        Args:
            pkg_graph: Personal Knowledge Graph
            access_history: Optional dict of node_id -> list of access times

        Returns:
            List of memories sorted by need for reinforcement
        """
        nodes, edges = self._extract_graph_structure(pkg_graph)
        adjacency = self._build_adjacency(nodes, edges)

        now = datetime.utcnow()
        half_life_seconds = self._decay_half_life_days * 24 * 3600

        results = []

        for node in nodes:
            # Get access history
            accesses = []
            if access_history and node in access_history:
                accesses = access_history[node]

            # Default last access if no history
            if accesses:
                last_access = max(accesses)
                access_count = len(accesses)
            else:
                # Assume created 90 days ago if no history
                last_access = now - timedelta(days=90)
                access_count = 1

            # Calculate decay score using DyMemR formula
            time_since_access = (now - last_access).total_seconds()
            decay_score = math.exp(-0.693 * time_since_access / half_life_seconds)

            # Calculate importance from connectivity
            degree = len(adjacency.get(node, set()))
            max_degree = max(len(adj) for adj in adjacency.values()) if adjacency else 1
            importance_score = degree / max_degree if max_degree > 0 else 0

            # Retrieval probability combines decay and importance
            retrieval_probability = (
                0.6 * decay_score +
                0.4 * importance_score
            )

            results.append(MemoryDecay(
                node_id=node,
                title=self._get_node_title(pkg_graph, node),
                last_access=last_access,
                access_count=access_count,
                decay_score=decay_score,
                importance_score=importance_score,
                retrieval_probability=retrieval_probability,
            ))

        # Sort by need for reinforcement (high importance, low decay)
        results.sort(
            key=lambda m: (
                -m.importance_score if m.decay_score < 0.5 else 0,
                m.decay_score,
            )
        )

        return results

    def _get_node_title(
        self,
        pkg_graph: Any,
        node_id: str,
    ) -> str:
        """Extract title for a node."""
        if hasattr(pkg_graph, "get_node"):
            try:
                node = pkg_graph.get_node(node_id)
                if hasattr(node, "title"):
                    return node.title
                if hasattr(node, "name"):
                    return node.name
            except Exception:
                pass

        return node_id

    def find_bridge_opportunities(
        self,
        communities: List[Community],
        pkg_graph: Any,
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> List[BridgeOpportunity]:
        """Find opportunities to bridge isolated communities.

        Uses semantic similarity (if embeddings available) or
        shared concept analysis to identify potential bridges.

        Args:
            communities: Detected communities
            pkg_graph: Original graph
            embeddings: Optional node embeddings for semantic similarity

        Returns:
            List of bridge opportunities sorted by information gain
        """
        opportunities = []

        for i, community_a in enumerate(communities):
            for community_b in communities[i + 1:]:
                # Find shared concepts (common neighbors)
                shared = self._find_shared_concepts(
                    community_a,
                    community_b,
                    pkg_graph,
                )

                if not shared:
                    continue

                # Compute information gain from bridging
                info_gain = self._compute_bridge_info_gain(
                    community_a,
                    community_b,
                    shared,
                    embeddings,
                )

                if info_gain > 0.1:  # Minimum threshold
                    # Suggest specific connections
                    suggestions = self._suggest_connections(
                        community_a,
                        community_b,
                        shared,
                        pkg_graph,
                    )

                    opportunities.append(BridgeOpportunity(
                        community_a=community_a.community_id,
                        community_b=community_b.community_id,
                        shared_concepts=shared,
                        information_gain=info_gain,
                        suggested_connections=suggestions,
                    ))

        # Sort by information gain
        opportunities.sort(key=lambda o: -o.information_gain)

        return opportunities

    def _find_shared_concepts(
        self,
        community_a: Community,
        community_b: Community,
        pkg_graph: Any,
    ) -> List[str]:
        """Find concepts that could bridge two communities."""
        # Get all node titles/keywords
        _, edges = self._extract_graph_structure(pkg_graph)

        # Find nodes that connect to both communities
        shared = []

        # Check for common neighbors not in either community
        all_nodes = community_a.nodes | community_b.nodes
        edge_set = set(edges) | {(t, s) for s, t in edges}

        for source, target in edge_set:
            if source in community_a.nodes and target in community_b.nodes:
                shared.append(f"direct:{source}-{target}")
            elif source in community_b.nodes and target in community_a.nodes:
                shared.append(f"direct:{target}-{source}")

        return shared[:5]  # Limit to top 5

    def _compute_bridge_info_gain(
        self,
        community_a: Community,
        community_b: Community,
        shared_concepts: List[str],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> float:
        """Compute expected information gain from bridging.

        Based on curiosity-driven exploration principles:
        Higher gain for bridging large, diverse communities.
        """
        # Base gain from community sizes
        size_factor = math.log(1 + community_a.size * community_b.size) / 10

        # Boost for isolated communities
        isolation_boost = (
            (1.5 if community_a.is_isolated else 1.0) *
            (1.5 if community_b.is_isolated else 1.0)
        )

        # Penalty for communities that are too similar
        similarity_penalty = 1.0
        if embeddings:
            sim = self._compute_community_similarity(
                community_a,
                community_b,
                embeddings,
            )
            # Penalize very similar (already connected conceptually)
            # Penalize very different (no meaningful bridge possible)
            similarity_penalty = 1.0 - abs(sim - 0.5) * 2

        info_gain = size_factor * isolation_boost * similarity_penalty

        return min(1.0, info_gain)

    def _compute_community_similarity(
        self,
        community_a: Community,
        community_b: Community,
        embeddings: Dict[str, List[float]],
    ) -> float:
        """Compute semantic similarity between communities."""
        # Average embeddings for each community
        def avg_embedding(community: Community) -> Optional[List[float]]:
            vecs = [
                embeddings[n] for n in community.nodes
                if n in embeddings
            ]
            if not vecs:
                return None
            return [
                sum(v[i] for v in vecs) / len(vecs)
                for i in range(len(vecs[0]))
            ]

        emb_a = avg_embedding(community_a)
        emb_b = avg_embedding(community_b)

        if emb_a is None or emb_b is None:
            return 0.5  # Neutral

        # Cosine similarity
        dot = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = math.sqrt(sum(a * a for a in emb_a))
        norm_b = math.sqrt(sum(b * b for b in emb_b))

        if norm_a == 0 or norm_b == 0:
            return 0.5

        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1

    def _suggest_connections(
        self,
        community_a: Community,
        community_b: Community,
        shared_concepts: List[str],
        pkg_graph: Any,
    ) -> List[Tuple[str, str]]:
        """Suggest specific node pairs to connect."""
        suggestions = []

        # Connect central topics
        if community_a.central_topics and community_b.central_topics:
            for topic_a in community_a.central_topics[:2]:
                for topic_b in community_b.central_topics[:2]:
                    suggestions.append((topic_a, topic_b))

        return suggestions[:3]  # Limit suggestions

    def compute_information_gain(
        self,
        item: Any,
        pkg_graph: Any,
    ) -> float:
        """Compute information gain for exploring an item.

        Based on curiosity-driven exploration principles:
        - High gain for novel but accessible items
        - Low gain for highly familiar or highly alien items
        - Optimal is the "zone of proximal development"

        Args:
            item: Gap, community, or node to evaluate
            pkg_graph: Current knowledge graph

        Returns:
            Information gain score (0-1)
        """
        if isinstance(item, Community):
            # Gain from exploring isolated community
            isolation_factor = 1.5 if item.is_isolated else 1.0
            size_factor = min(1.0, item.size / 10)
            density_factor = 1.0 - item.density  # More gain from sparse communities

            return min(1.0, isolation_factor * size_factor * density_factor)

        elif isinstance(item, MemoryDecay):
            # Gain from reinforcing decaying memory
            # High importance + low decay = high gain
            return item.importance_score * (1.0 - item.decay_score)

        elif isinstance(item, BridgeOpportunity):
            return item.information_gain

        return 0.5  # Default neutral

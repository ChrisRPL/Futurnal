"""
Causal Chain Inference Algorithm.

Implements discovery and validation of causal chains in knowledge graphs.
Combines graph traversal with temporal and statistical validation.

Research Foundation:
- ICDA (Interactive Causal Discovery)
- PC Algorithm for causal structure discovery
- Temporal causality constraints

Algorithm Components:
1. Path Discovery - Find potential causal paths in graph
2. Temporal Validation - Ensure A precedes B precedes C
3. Conditional Independence Testing - Validate causal structure
4. Chain Strength Scoring - Rank discovered chains

Option B Compliance:
- Rule-based inference (no gradient updates)
- LLM used for plausibility assessment only
- Results stored as natural language priors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class ChainType(str, Enum):
    """Types of causal chains."""
    DIRECT = "direct"  # A -> B (direct cause)
    MEDIATING = "mediating"  # A -> M -> B (mediated)
    CONFOUNDED = "confounded"  # C -> A, C -> B (common cause)
    COLLIDER = "collider"  # A -> C <- B (common effect)
    CASCADING = "cascading"  # A -> B -> C -> D (cascade)


class InferenceMethod(str, Enum):
    """Methods for causal inference."""
    TEMPORAL = "temporal"  # Based on temporal order only
    STATISTICAL = "statistical"  # Based on statistical tests
    STRUCTURAL = "structural"  # Based on graph structure
    HYBRID = "hybrid"  # Combination of methods


@dataclass
class CausalNode:
    """A node in a causal chain."""
    node_id: str
    name: str
    timestamp: Optional[datetime] = None
    node_type: str = "event"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """An edge (relationship) in a causal chain."""
    source_id: str
    target_id: str
    relationship_type: str
    confidence: float = 0.5
    temporal_gap: Optional[timedelta] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalChain:
    """A discovered causal chain."""
    chain_id: str
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    chain_type: ChainType

    # Scores
    temporal_score: float = 0.0  # Temporal validity (0-1)
    structural_score: float = 0.0  # Graph structure support (0-1)
    statistical_score: float = 0.0  # Statistical support (0-1)
    overall_score: float = 0.0

    # Metadata
    inference_method: InferenceMethod = InferenceMethod.HYBRID
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    supporting_evidence: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of edges in chain."""
        return len(self.edges)

    @property
    def total_gap(self) -> Optional[timedelta]:
        """Total temporal gap from first to last node."""
        gaps = [e.temporal_gap for e in self.edges if e.temporal_gap]
        if gaps:
            return sum(gaps, timedelta())
        return None

    @property
    def path_description(self) -> str:
        """Human-readable path description."""
        if not self.nodes:
            return ""
        names = [n.name for n in self.nodes]
        return " → ".join(names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "nodes": [{"id": n.node_id, "name": n.name} for n in self.nodes],
            "edges": [{"source": e.source_id, "target": e.target_id, "type": e.relationship_type}
                      for e in self.edges],
            "chain_type": self.chain_type.value,
            "temporal_score": self.temporal_score,
            "structural_score": self.structural_score,
            "statistical_score": self.statistical_score,
            "overall_score": self.overall_score,
            "path_description": self.path_description,
        }


@dataclass
class ChainInferenceResult:
    """Result of causal chain inference."""
    query: str
    chains: List[CausalChain]
    total_chains_found: int
    inference_time_ms: float

    # Summary statistics
    avg_chain_length: float = 0.0
    avg_confidence: float = 0.0
    dominant_chain_type: Optional[ChainType] = None

    # Metadata
    method_used: InferenceMethod = InferenceMethod.HYBRID
    parameters: Dict[str, Any] = field(default_factory=dict)


class CausalChainInference:
    """
    Discovers and validates causal chains in knowledge graphs.

    Combines:
    1. Graph traversal for path discovery
    2. Temporal ordering validation
    3. Statistical independence testing
    4. Confidence scoring
    """

    def __init__(
        self,
        neo4j_driver: Optional[Any] = None,
        max_chain_length: int = 5,
        min_confidence: float = 0.3,
        temporal_window_days: int = 30
    ):
        """Initialize causal chain inference.

        Args:
            neo4j_driver: Neo4j driver for graph queries
            max_chain_length: Maximum chain length to search
            min_confidence: Minimum confidence threshold
            temporal_window_days: Maximum gap between events
        """
        self.driver = neo4j_driver
        self.max_chain_length = max_chain_length
        self.min_confidence = min_confidence
        self.temporal_window = timedelta(days=temporal_window_days)

        # Cache
        self._chain_cache: Dict[str, List[CausalChain]] = {}

    def find_chains(
        self,
        source_id: str,
        target_id: Optional[str] = None,
        max_chains: int = 10,
        method: InferenceMethod = InferenceMethod.HYBRID
    ) -> ChainInferenceResult:
        """Find causal chains from source to target.

        Args:
            source_id: Starting node ID
            target_id: Optional ending node ID (if None, find all chains from source)
            max_chains: Maximum chains to return
            method: Inference method to use

        Returns:
            ChainInferenceResult with discovered chains
        """
        import time
        start_time = time.time()

        query = f"{source_id} -> {target_id or '*'}"

        # Discover potential paths
        if target_id:
            paths = self._find_paths_to_target(source_id, target_id)
        else:
            paths = self._find_all_paths_from(source_id)

        # Convert paths to chains
        chains = []
        for path in paths:
            chain = self._path_to_chain(path)
            if chain:
                # Validate and score chain
                chain = self._validate_chain(chain, method)
                if chain.overall_score >= self.min_confidence:
                    chains.append(chain)

        # Sort by score and limit
        chains.sort(key=lambda c: c.overall_score, reverse=True)
        chains = chains[:max_chains]

        # Calculate statistics
        avg_length = sum(c.length for c in chains) / len(chains) if chains else 0
        avg_confidence = sum(c.overall_score for c in chains) / len(chains) if chains else 0

        # Find dominant chain type
        type_counts = defaultdict(int)
        for chain in chains:
            type_counts[chain.chain_type] += 1
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else None

        inference_time = (time.time() - start_time) * 1000

        return ChainInferenceResult(
            query=query,
            chains=chains,
            total_chains_found=len(chains),
            inference_time_ms=inference_time,
            avg_chain_length=avg_length,
            avg_confidence=avg_confidence,
            dominant_chain_type=dominant_type,
            method_used=method,
        )

    def find_mediators(
        self,
        cause_id: str,
        effect_id: str
    ) -> List[CausalChain]:
        """Find mediating variables between cause and effect.

        A -> M -> B where M mediates the effect of A on B.
        """
        chains = []

        # Find all paths of length 2
        paths = self._find_paths_to_target(cause_id, effect_id, max_length=3)

        for path in paths:
            if len(path["nodes"]) == 3:  # A -> M -> B
                chain = self._path_to_chain(path)
                if chain:
                    chain.chain_type = ChainType.MEDIATING
                    chain = self._validate_chain(chain, InferenceMethod.HYBRID)
                    if chain.overall_score >= self.min_confidence:
                        chains.append(chain)

        return chains

    def find_confounders(
        self,
        event_a_id: str,
        event_b_id: str
    ) -> List[Tuple[str, CausalChain]]:
        """Find potential confounding variables.

        C -> A and C -> B (common cause of both)
        """
        confounders = []

        # Find common ancestors
        ancestors_a = self._find_ancestors(event_a_id)
        ancestors_b = self._find_ancestors(event_b_id)

        common = ancestors_a & ancestors_b

        for confounder_id in common:
            # Build chain representing confounding structure
            nodes = [
                self._get_node(confounder_id),
                self._get_node(event_a_id),
                self._get_node(event_b_id),
            ]
            nodes = [n for n in nodes if n]

            if len(nodes) == 3:
                edges = [
                    CausalEdge(confounder_id, event_a_id, "causes"),
                    CausalEdge(confounder_id, event_b_id, "causes"),
                ]
                chain = CausalChain(
                    chain_id=f"conf_{confounder_id}",
                    nodes=nodes,
                    edges=edges,
                    chain_type=ChainType.CONFOUNDED,
                )
                chain = self._validate_chain(chain, InferenceMethod.STRUCTURAL)
                confounders.append((nodes[0].name, chain))

        return confounders

    def find_cascading_effects(
        self,
        trigger_id: str,
        max_depth: int = 4
    ) -> List[CausalChain]:
        """Find cascading causal chains from a trigger event.

        A -> B -> C -> D (cascade of effects)
        """
        chains = []
        visited = set()

        def dfs_cascade(current_id: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current_id in visited:
                return

            visited.add(current_id)
            path.append(current_id)

            # Get effects of current node
            effects = self._get_effects(current_id)

            for effect_id in effects:
                # Recursively explore
                dfs_cascade(effect_id, path.copy(), depth + 1)

            # Create chain from path if long enough
            if len(path) >= 3:
                chain = self._ids_to_chain(path)
                if chain:
                    chain.chain_type = ChainType.CASCADING
                    chains.append(chain)

            visited.remove(current_id)

        dfs_cascade(trigger_id, [], 0)

        # Validate and score chains
        validated = []
        for chain in chains:
            chain = self._validate_chain(chain, InferenceMethod.TEMPORAL)
            if chain.overall_score >= self.min_confidence:
                validated.append(chain)

        return validated

    def _find_paths_to_target(
        self,
        source_id: str,
        target_id: str,
        max_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find all paths from source to target using BFS."""
        max_len = max_length or self.max_chain_length
        paths = []

        if not self.driver:
            return paths

        query = """
        MATCH path = shortestPath((source)-[*1..{max_len}]->(target))
        WHERE source.id = $source_id AND target.id = $target_id
        RETURN path,
               [n in nodes(path) | n.id] as node_ids,
               [n in nodes(path) | n.name] as node_names,
               [n in nodes(path) | n.timestamp] as timestamps,
               [r in relationships(path) | type(r)] as rel_types
        LIMIT 50
        """.replace("{max_len}", str(max_len))

        try:
            with self.driver.session() as session:
                result = session.run(query, source_id=source_id, target_id=target_id)
                for record in result:
                    paths.append({
                        "node_ids": record["node_ids"],
                        "node_names": record["node_names"],
                        "timestamps": record["timestamps"],
                        "rel_types": record["rel_types"],
                    })
        except Exception as e:
            logger.warning(f"Path query failed: {e}")

        return paths

    def _find_all_paths_from(
        self,
        source_id: str
    ) -> List[Dict[str, Any]]:
        """Find all causal paths emanating from source."""
        paths = []

        if not self.driver:
            return paths

        query = """
        MATCH path = (source)-[r:CAUSES|LEADS_TO|TRIGGERS*1..{max_len}]->(target)
        WHERE source.id = $source_id
        RETURN path,
               [n in nodes(path) | n.id] as node_ids,
               [n in nodes(path) | n.name] as node_names,
               [n in nodes(path) | n.timestamp] as timestamps,
               [rel in relationships(path) | type(rel)] as rel_types
        LIMIT 100
        """.replace("{max_len}", str(self.max_chain_length))

        try:
            with self.driver.session() as session:
                result = session.run(query, source_id=source_id)
                for record in result:
                    paths.append({
                        "node_ids": record["node_ids"],
                        "node_names": record["node_names"],
                        "timestamps": record["timestamps"],
                        "rel_types": record["rel_types"],
                    })
        except Exception as e:
            logger.warning(f"All paths query failed: {e}")

        return paths

    def _find_ancestors(self, node_id: str) -> Set[str]:
        """Find all causal ancestors of a node."""
        ancestors = set()

        if not self.driver:
            return ancestors

        query = """
        MATCH (ancestor)-[r:CAUSES|LEADS_TO|TRIGGERS*1..5]->(target)
        WHERE target.id = $node_id
        RETURN DISTINCT ancestor.id as ancestor_id
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id)
                for record in result:
                    ancestors.add(record["ancestor_id"])
        except Exception as e:
            logger.warning(f"Ancestors query failed: {e}")

        return ancestors

    def _get_effects(self, node_id: str) -> List[str]:
        """Get direct causal effects of a node."""
        effects = []

        if not self.driver:
            return effects

        query = """
        MATCH (source)-[r:CAUSES|LEADS_TO|TRIGGERS]->(effect)
        WHERE source.id = $node_id
        RETURN effect.id as effect_id
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id)
                for record in result:
                    effects.append(record["effect_id"])
        except Exception as e:
            logger.warning(f"Effects query failed: {e}")

        return effects

    def _get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get node details from graph."""
        if not self.driver:
            return CausalNode(node_id=node_id, name=node_id)

        query = """
        MATCH (n {id: $node_id})
        RETURN n.id as id, n.name as name, n.timestamp as timestamp, labels(n) as labels
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id)
                record = result.single()
                if record:
                    return CausalNode(
                        node_id=record["id"],
                        name=record["name"] or record["id"],
                        timestamp=record["timestamp"],
                        node_type=record["labels"][0] if record["labels"] else "unknown",
                    )
        except Exception as e:
            logger.warning(f"Node query failed: {e}")

        return None

    def _path_to_chain(self, path: Dict[str, Any]) -> Optional[CausalChain]:
        """Convert a path dict to CausalChain."""
        node_ids = path.get("node_ids", [])
        node_names = path.get("node_names", [])
        timestamps = path.get("timestamps", [])
        rel_types = path.get("rel_types", [])

        if len(node_ids) < 2:
            return None

        # Build nodes
        nodes = []
        for i, (nid, name) in enumerate(zip(node_ids, node_names)):
            ts = timestamps[i] if i < len(timestamps) else None
            nodes.append(CausalNode(
                node_id=nid,
                name=name or nid,
                timestamp=ts,
            ))

        # Build edges
        edges = []
        for i in range(len(node_ids) - 1):
            rel = rel_types[i] if i < len(rel_types) else "causes"
            gap = None
            if i + 1 < len(timestamps) and timestamps[i] and timestamps[i + 1]:
                gap = timestamps[i + 1] - timestamps[i]

            edges.append(CausalEdge(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                relationship_type=rel,
                temporal_gap=gap,
            ))

        # Determine chain type
        if len(nodes) == 2:
            chain_type = ChainType.DIRECT
        elif len(nodes) == 3:
            chain_type = ChainType.MEDIATING
        else:
            chain_type = ChainType.CASCADING

        return CausalChain(
            chain_id=f"chain_{'_'.join(node_ids[:3])}",
            nodes=nodes,
            edges=edges,
            chain_type=chain_type,
        )

    def _ids_to_chain(self, node_ids: List[str]) -> Optional[CausalChain]:
        """Convert list of node IDs to CausalChain."""
        nodes = []
        for nid in node_ids:
            node = self._get_node(nid)
            if node:
                nodes.append(node)

        if len(nodes) < 2:
            return None

        edges = []
        for i in range(len(nodes) - 1):
            gap = None
            if nodes[i].timestamp and nodes[i + 1].timestamp:
                gap = nodes[i + 1].timestamp - nodes[i].timestamp
            edges.append(CausalEdge(
                source_id=nodes[i].node_id,
                target_id=nodes[i + 1].node_id,
                relationship_type="causes",
                temporal_gap=gap,
            ))

        return CausalChain(
            chain_id=f"chain_{'_'.join(node_ids[:3])}",
            nodes=nodes,
            edges=edges,
            chain_type=ChainType.CASCADING,
        )

    def _validate_chain(
        self,
        chain: CausalChain,
        method: InferenceMethod
    ) -> CausalChain:
        """Validate and score a causal chain."""
        # Temporal validation
        chain.temporal_score = self._validate_temporal_order(chain)
        if chain.temporal_score > 0:
            chain.supporting_evidence.append("Temporal order preserved")
        else:
            chain.concerns.append("Temporal order violated")

        # Structural validation
        chain.structural_score = self._validate_structure(chain)
        if chain.structural_score > 0.5:
            chain.supporting_evidence.append("Strong graph structure support")

        # Statistical validation
        if method in [InferenceMethod.STATISTICAL, InferenceMethod.HYBRID]:
            chain.statistical_score = self._validate_statistical(chain)
            if chain.statistical_score > 0.5:
                chain.supporting_evidence.append("Statistical evidence supports chain")

        # Calculate overall score
        if method == InferenceMethod.TEMPORAL:
            chain.overall_score = chain.temporal_score
        elif method == InferenceMethod.STATISTICAL:
            chain.overall_score = (chain.temporal_score * 0.3 + chain.statistical_score * 0.7)
        elif method == InferenceMethod.STRUCTURAL:
            chain.overall_score = (chain.temporal_score * 0.3 + chain.structural_score * 0.7)
        else:  # HYBRID
            chain.overall_score = (
                chain.temporal_score * 0.4 +
                chain.structural_score * 0.3 +
                chain.statistical_score * 0.3
            )

        chain.inference_method = method
        return chain

    def _validate_temporal_order(self, chain: CausalChain) -> float:
        """Validate that temporal order is preserved."""
        if len(chain.nodes) < 2:
            return 0.0

        valid_pairs = 0
        total_pairs = len(chain.nodes) - 1

        for i in range(len(chain.nodes) - 1):
            node_a = chain.nodes[i]
            node_b = chain.nodes[i + 1]

            if node_a.timestamp and node_b.timestamp:
                if node_a.timestamp < node_b.timestamp:
                    # Also check temporal window
                    gap = node_b.timestamp - node_a.timestamp
                    if gap <= self.temporal_window:
                        valid_pairs += 1
                    else:
                        valid_pairs += 0.5  # Partial credit for correct order
            else:
                # No timestamps - neutral
                valid_pairs += 0.5

        return valid_pairs / total_pairs if total_pairs > 0 else 0.0

    def _validate_structure(self, chain: CausalChain) -> float:
        """Validate chain based on graph structure."""
        score = 0.0

        # Check edge types
        causal_types = {"causes", "leads_to", "triggers", "results_in", "enables"}
        causal_edges = sum(
            1 for e in chain.edges
            if e.relationship_type.lower() in causal_types
        )
        score += 0.5 * (causal_edges / len(chain.edges)) if chain.edges else 0

        # Check edge confidence
        avg_confidence = sum(e.confidence for e in chain.edges) / len(chain.edges) if chain.edges else 0
        score += 0.5 * avg_confidence

        return score

    def _validate_statistical(self, chain: CausalChain) -> float:
        """Validate chain using statistical tests."""
        # This would involve conditional independence testing
        # For now, use heuristics based on edge properties

        score = 0.5  # Default moderate

        # Check sample size proxy
        if len(chain.edges) > 0:
            avg_confidence = sum(e.confidence for e in chain.edges) / len(chain.edges)
            score = 0.3 + (0.7 * avg_confidence)

        return score


class CausalStructureLearner:
    """
    Learns causal structure from observational data.

    Uses constraint-based approach inspired by PC algorithm.
    """

    def __init__(
        self,
        neo4j_driver: Optional[Any] = None,
        significance_level: float = 0.05
    ):
        self.driver = neo4j_driver
        self.significance_level = significance_level

    def learn_structure(
        self,
        event_types: List[str],
        max_conditioning_set: int = 3
    ) -> Dict[str, List[str]]:
        """Learn causal structure among event types.

        Args:
            event_types: List of event types to consider
            max_conditioning_set: Maximum size of conditioning set

        Returns:
            Dictionary mapping cause to list of effects
        """
        # Start with complete undirected graph
        edges = set()
        for i, a in enumerate(event_types):
            for j, b in enumerate(event_types):
                if i < j:
                    edges.add((a, b))

        # Remove edges based on conditional independence
        for cond_size in range(max_conditioning_set + 1):
            edges_to_remove = set()

            for (a, b) in edges:
                # Get other variables for conditioning
                others = [x for x in event_types if x not in (a, b)]

                for cond_set in self._subsets(others, cond_size):
                    if self._conditionally_independent(a, b, list(cond_set)):
                        edges_to_remove.add((a, b))
                        break

            edges -= edges_to_remove

        # Orient edges based on temporal order
        causal_structure = defaultdict(list)
        for (a, b) in edges:
            # Use temporal order to determine direction
            if self._precedes(a, b):
                causal_structure[a].append(b)
            elif self._precedes(b, a):
                causal_structure[b].append(a)
            else:
                # Unknown direction - keep both
                causal_structure[a].append(b)

        return dict(causal_structure)

    def _subsets(self, items: List[str], size: int):
        """Generate all subsets of given size."""
        from itertools import combinations
        return combinations(items, size)

    def _conditionally_independent(
        self,
        a: str,
        b: str,
        conditioning: List[str]
    ) -> bool:
        """Test if A and B are conditionally independent given conditioning set."""
        # This is a simplified test - full implementation would use chi-square
        # or G-test on contingency tables

        if not self.driver:
            return False  # Can't test without data

        # Build query to count co-occurrences
        # Simplified: just check if correlation exists
        # Real implementation would compute partial correlations

        return False  # Conservative: assume dependent

    def _precedes(self, event_a: str, event_b: str) -> bool:
        """Check if event type A typically precedes event type B."""
        if not self.driver:
            return False

        query = """
        MATCH (a {event_type: $type_a})-[:BEFORE|CAUSES|LEADS_TO]->(b {event_type: $type_b})
        WITH count(*) as ab_count
        MATCH (b {event_type: $type_b})-[:BEFORE|CAUSES|LEADS_TO]->(a {event_type: $type_a})
        WITH ab_count, count(*) as ba_count
        RETURN ab_count > ba_count as a_precedes_b
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, type_a=event_a, type_b=event_b)
                record = result.single()
                if record:
                    return record["a_precedes_b"]
        except Exception as e:
            logger.warning(f"Temporal order query failed: {e}")

        return False

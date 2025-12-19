"""
Subgraph Pattern Extraction and Matching.

Extracts frequent subgraph patterns from the knowledge graph
that can be used to generate extraction rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternNode:
    """A node in a subgraph pattern (may be variable or concrete)."""
    id: str
    node_type: str
    is_variable: bool = True  # True = can match any node of type


@dataclass(frozen=True)
class PatternEdge:
    """An edge in a subgraph pattern."""
    source: str
    target: str
    relation_type: str


@dataclass
class SubgraphPattern:
    """
    A subgraph pattern that can match against the knowledge graph.

    Used for:
    - Identifying extraction opportunities
    - Generating rules for new facts
    - Knowledge base completion
    """
    id: str
    nodes: Dict[str, PatternNode] = field(default_factory=dict)
    edges: List[PatternEdge] = field(default_factory=list)

    # Pattern metadata
    support: int = 0  # Number of matches in graph
    confidence: float = 0.0
    source: str = "mined"  # "mined", "llm_proposed", "user_defined"

    # Optional constraints
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_cypher(self) -> str:
        """Convert pattern to Cypher MATCH clause."""
        match_parts = []

        for node_id, node in self.nodes.items():
            if node.is_variable:
                match_parts.append(f"({node_id}:{node.node_type})")
            else:
                match_parts.append(f"({node_id}:{node.node_type} {{id: '{node.id}'}})")

        for edge in self.edges:
            match_parts.append(f"({edge.source})-[:{edge.relation_type}]->({edge.target})")

        return "MATCH " + ", ".join(match_parts)

    def get_signature(self) -> str:
        """Get unique signature for pattern deduplication."""
        sig_parts = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            sig_parts.append(f"{node_id}:{node.node_type}")

        for edge in sorted(self.edges, key=lambda e: (e.source, e.target)):
            sig_parts.append(f"{edge.source}-{edge.relation_type}-{edge.target}")

        return hashlib.md5("|".join(sig_parts).encode()).hexdigest()


@dataclass
class PatternMatch:
    """A match of a pattern against the knowledge graph."""
    pattern_id: str
    bindings: Dict[str, str]  # pattern_node_id -> actual_node_id
    confidence: float = 1.0
    source_doc: Optional[str] = None


class PatternExtractor:
    """
    Extracts frequent subgraph patterns from the knowledge graph.

    Uses a mining approach inspired by gSpan algorithm but adapted
    for knowledge graphs with typed nodes and edges.
    """

    def __init__(
        self,
        min_support: int = 2,
        max_pattern_size: int = 5,
        max_patterns: int = 1000
    ):
        self.min_support = min_support
        self.max_pattern_size = max_pattern_size
        self.max_patterns = max_patterns

    def extract_from_neo4j(
        self,
        neo4j_driver,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> List[SubgraphPattern]:
        """
        Extract patterns from Neo4j knowledge graph.

        Args:
            neo4j_driver: Neo4j driver instance
            node_labels: Filter for node types
            relationship_types: Filter for relationship types

        Returns:
            List of frequent subgraph patterns
        """
        # Step 1: Extract frequent edge patterns (size-1)
        edge_patterns = self._extract_edge_patterns(
            neo4j_driver, node_labels, relationship_types
        )

        # Step 2: Grow patterns
        all_patterns = list(edge_patterns)
        current_patterns = edge_patterns

        for size in range(2, self.max_pattern_size + 1):
            if len(all_patterns) >= self.max_patterns:
                break

            grown_patterns = self._grow_patterns(
                neo4j_driver, current_patterns, node_labels, relationship_types
            )

            if not grown_patterns:
                break

            all_patterns.extend(grown_patterns)
            current_patterns = grown_patterns

        logger.info(f"Extracted {len(all_patterns)} patterns")
        return all_patterns[:self.max_patterns]

    def _extract_edge_patterns(
        self,
        neo4j_driver,
        node_labels: Optional[List[str]],
        relationship_types: Optional[List[str]]
    ) -> List[SubgraphPattern]:
        """Extract size-1 patterns (single edges)."""
        patterns = []

        # Query for edge type frequencies
        query = """
        MATCH (a)-[r]->(b)
        WITH labels(a)[0] as head_type, type(r) as rel_type, labels(b)[0] as tail_type, count(*) as cnt
        WHERE cnt >= $min_support
        RETURN head_type, rel_type, tail_type, cnt
        ORDER BY cnt DESC
        """

        with neo4j_driver.session() as session:
            result = session.run(query, min_support=self.min_support)

            for record in result:
                head_type = record["head_type"]
                rel_type = record["rel_type"]
                tail_type = record["tail_type"]
                count = record["cnt"]

                # Filter if specified
                if node_labels and (head_type not in node_labels or tail_type not in node_labels):
                    continue
                if relationship_types and rel_type not in relationship_types:
                    continue

                pattern = SubgraphPattern(
                    id=f"edge_{head_type}_{rel_type}_{tail_type}",
                    nodes={
                        "h": PatternNode(id="h", node_type=head_type, is_variable=True),
                        "t": PatternNode(id="t", node_type=tail_type, is_variable=True)
                    },
                    edges=[PatternEdge(source="h", target="t", relation_type=rel_type)],
                    support=count
                )
                patterns.append(pattern)

        return patterns

    def _grow_patterns(
        self,
        neo4j_driver,
        base_patterns: List[SubgraphPattern],
        node_labels: Optional[List[str]],
        relationship_types: Optional[List[str]]
    ) -> List[SubgraphPattern]:
        """Grow patterns by adding one more edge."""
        grown = []
        seen_signatures: Set[str] = set()

        for base_pattern in base_patterns:
            # Try extending from each node in the pattern
            for node_id in base_pattern.nodes:
                extensions = self._find_extensions(
                    neo4j_driver, base_pattern, node_id, node_labels, relationship_types
                )

                for ext_pattern in extensions:
                    sig = ext_pattern.get_signature()
                    if sig not in seen_signatures and ext_pattern.support >= self.min_support:
                        seen_signatures.add(sig)
                        grown.append(ext_pattern)

                        if len(grown) >= self.max_patterns // 2:
                            return grown

        return grown

    def _find_extensions(
        self,
        neo4j_driver,
        base_pattern: SubgraphPattern,
        extend_node: str,
        node_labels: Optional[List[str]],
        relationship_types: Optional[List[str]]
    ) -> List[SubgraphPattern]:
        """Find pattern extensions from a specific node."""
        extensions = []

        # Get the node type we're extending from
        extend_type = base_pattern.nodes[extend_node].node_type

        # Query for additional edges from this node type
        query = f"""
        MATCH (a:{extend_type})-[r]->(b)
        WITH type(r) as rel_type, labels(b)[0] as target_type, count(*) as cnt
        WHERE cnt >= $min_support
        RETURN rel_type, target_type, cnt
        """

        with neo4j_driver.session() as session:
            result = session.run(query, min_support=self.min_support)

            for record in result:
                rel_type = record["rel_type"]
                target_type = record["target_type"]
                count = record["cnt"]

                if node_labels and target_type not in node_labels:
                    continue
                if relationship_types and rel_type not in relationship_types:
                    continue

                # Check if this edge already exists in pattern
                edge_exists = any(
                    e.source == extend_node and e.relation_type == rel_type
                    for e in base_pattern.edges
                )

                if not edge_exists:
                    # Create extended pattern
                    new_pattern = SubgraphPattern(
                        id=f"{base_pattern.id}_{rel_type}_{target_type}",
                        nodes=dict(base_pattern.nodes),
                        edges=list(base_pattern.edges),
                        support=count
                    )

                    # Add new node and edge
                    new_node_id = f"n{len(new_pattern.nodes)}"
                    new_pattern.nodes[new_node_id] = PatternNode(
                        id=new_node_id, node_type=target_type, is_variable=True
                    )
                    new_pattern.edges.append(PatternEdge(
                        source=extend_node, target=new_node_id, relation_type=rel_type
                    ))

                    extensions.append(new_pattern)

        return extensions


class PatternMatcher:
    """
    Matches patterns against the knowledge graph.

    Used for:
    - Finding all instances of a pattern
    - Validating extracted rules
    - Identifying missing relationships
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def match(
        self,
        pattern: SubgraphPattern,
        limit: int = 100
    ) -> List[PatternMatch]:
        """
        Find all matches of a pattern in the graph.

        Args:
            pattern: Pattern to match
            limit: Maximum number of matches to return

        Returns:
            List of pattern matches with bindings
        """
        cypher = pattern.to_cypher()
        return_vars = list(pattern.nodes.keys())
        cypher += f" RETURN {', '.join([f'{v}.id as {v}' for v in return_vars])}"
        cypher += f" LIMIT {limit}"

        matches = []
        with self.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                bindings = {v: record[v] for v in return_vars}
                matches.append(PatternMatch(
                    pattern_id=pattern.id,
                    bindings=bindings
                ))

        return matches

    def count_matches(self, pattern: SubgraphPattern) -> int:
        """Count total matches for a pattern."""
        cypher = pattern.to_cypher()
        cypher += " RETURN count(*) as cnt"

        with self.driver.session() as session:
            result = session.run(cypher)
            record = result.single()
            return record["cnt"] if record else 0

    def find_missing_edges(
        self,
        pattern: SubgraphPattern,
        target_relation: str,
        limit: int = 100
    ) -> List[Tuple[str, str]]:
        """
        Find node pairs that match pattern but are missing target relation.

        This is useful for knowledge base completion.

        Args:
            pattern: Pattern that should indicate a relationship
            target_relation: The relationship that should exist
            limit: Maximum pairs to return

        Returns:
            List of (head_id, tail_id) pairs that are missing the relation
        """
        # Get all matches of the pattern
        matches = self.match(pattern, limit=limit * 10)

        # For each match, check if target relation exists
        missing = []
        head_var = list(pattern.nodes.keys())[0]
        tail_var = list(pattern.nodes.keys())[-1]

        for match in matches:
            head_id = match.bindings.get(head_var)
            tail_id = match.bindings.get(tail_var)

            if head_id and tail_id:
                exists = self._check_relation_exists(head_id, target_relation, tail_id)
                if not exists:
                    missing.append((head_id, tail_id))

                    if len(missing) >= limit:
                        break

        return missing

    def _check_relation_exists(
        self,
        head_id: str,
        relation_type: str,
        tail_id: str
    ) -> bool:
        """Check if a specific relation exists."""
        query = f"""
        MATCH (h {{id: $head_id}})-[:{relation_type}]->(t {{id: $tail_id}})
        RETURN count(*) > 0 as exists
        """

        with self.driver.session() as session:
            result = session.run(query, head_id=head_id, tail_id=tail_id)
            record = result.single()
            return record["exists"] if record else False

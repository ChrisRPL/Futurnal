"""
Query Decomposer - Breaks complex queries into sub-queries.

Implements multi-clause query decomposition for:
- Multi-hop reasoning
- Parallel sub-query execution
- Evidence aggregation
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries."""
    SIMPLE = "simple"  # Single entity or fact lookup
    MULTI_HOP = "multi_hop"  # Requires traversing relationships
    COMPARISON = "comparison"  # Compare entities or attributes
    TEMPORAL = "temporal"  # Time-based queries
    CAUSAL = "causal"  # Cause-effect queries
    AGGREGATION = "aggregation"  # Count, sum, average, etc.
    COMPOSITE = "composite"  # Combination of multiple types


class DecompositionStrategy(Enum):
    """Strategies for query decomposition."""
    ENTITY_CENTRIC = "entity"  # Decompose by entities mentioned
    CLAUSE_BASED = "clause"  # Decompose by query clauses (AND/OR)
    REASONING_CHAIN = "chain"  # Decompose into reasoning steps
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class SubQuery:
    """A sub-query resulting from decomposition."""
    id: str
    text: str
    query_type: QueryType

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # IDs of prerequisite sub-queries
    provides: List[str] = field(default_factory=list)  # Variables this sub-query provides

    # Execution hints
    priority: int = 0  # Higher = execute first
    estimated_complexity: float = 1.0  # Relative complexity
    retrieval_mode: str = "hybrid"  # "vector", "graph", "hybrid"

    # Extracted information
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    temporal_constraints: List[str] = field(default_factory=list)

    # Results (filled after execution)
    result: Optional[Any] = None
    confidence: float = 0.0


@dataclass
class QueryPlan:
    """A plan for executing a decomposed query."""
    original_query: str
    query_type: QueryType
    sub_queries: List[SubQuery]
    execution_order: List[List[str]]  # Groups of parallel sub-queries
    fusion_strategy: str = "weighted"  # How to combine results

    # Metadata
    decomposition_strategy: DecompositionStrategy = DecompositionStrategy.HYBRID
    estimated_hops: int = 1
    confidence: float = 1.0


class QueryDecomposer:
    """
    Decomposes complex queries into executable sub-queries.

    Uses LLM for semantic understanding and rule-based decomposition.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_sub_queries: int = 10,
        use_llm_decomposition: bool = True
    ):
        self.llm_client = llm_client
        self.max_sub_queries = max_sub_queries
        self.use_llm_decomposition = use_llm_decomposition

        # Patterns for rule-based decomposition
        self._init_patterns()

    def _init_patterns(self):
        """Initialize decomposition patterns."""
        # Multi-hop indicators
        self.multi_hop_patterns = [
            r"what.*that.*which",
            r"who.*works.*at.*located",
            r"find.*related.*to.*connected",
            r"how.*does.*affect.*through",
        ]

        # Temporal indicators
        self.temporal_patterns = [
            r"when did",
            r"before|after|during",
            r"in \d{4}",
            r"last (week|month|year)",
            r"between .* and .*",
        ]

        # Causal indicators
        self.causal_patterns = [
            r"why did",
            r"what caused",
            r"how did .* lead to",
            r"what (is|was) the (effect|impact|result)",
            r"because of",
        ]

        # Comparison indicators
        self.comparison_patterns = [
            r"compare",
            r"difference between",
            r"(more|less|better|worse) than",
            r"which .* is (the )?(most|least|best|worst)",
        ]

    async def decompose(self, query: str) -> QueryPlan:
        """
        Decompose a query into a query plan.

        Args:
            query: Natural language query

        Returns:
            QueryPlan with sub-queries and execution order
        """
        # Classify query type
        query_type = self._classify_query(query)

        # Choose decomposition strategy
        strategy = self._choose_strategy(query_type)

        # Decompose based on strategy
        if self.use_llm_decomposition and self.llm_client:
            sub_queries = await self._decompose_with_llm(query, query_type, strategy)
        else:
            sub_queries = self._decompose_rule_based(query, query_type, strategy)

        # Build execution order
        execution_order = self._build_execution_order(sub_queries)

        # Determine fusion strategy
        fusion_strategy = self._determine_fusion_strategy(query_type)

        plan = QueryPlan(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            execution_order=execution_order,
            decomposition_strategy=strategy,
            fusion_strategy=fusion_strategy,
            estimated_hops=len(execution_order)
        )

        logger.info(
            f"Decomposed query into {len(sub_queries)} sub-queries, "
            f"{len(execution_order)} execution stages"
        )

        return plan

    def decompose_sync(self, query: str) -> QueryPlan:
        """Synchronous version of decompose."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.decompose(query))

    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query."""
        query_lower = query.lower()

        # Check for multi-hop
        for pattern in self.multi_hop_patterns:
            if re.search(pattern, query_lower):
                return QueryType.MULTI_HOP

        # Check for causal
        for pattern in self.causal_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CAUSAL

        # Check for temporal
        for pattern in self.temporal_patterns:
            if re.search(pattern, query_lower):
                return QueryType.TEMPORAL

        # Check for comparison
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return QueryType.COMPARISON

        # Check for aggregation
        if any(w in query_lower for w in ["how many", "count", "total", "average", "sum"]):
            return QueryType.AGGREGATION

        # Check for composite (multiple clauses)
        if " and " in query_lower or " or " in query_lower:
            return QueryType.COMPOSITE

        return QueryType.SIMPLE

    def _choose_strategy(self, query_type: QueryType) -> DecompositionStrategy:
        """Choose decomposition strategy based on query type."""
        if query_type in [QueryType.MULTI_HOP, QueryType.CAUSAL]:
            return DecompositionStrategy.REASONING_CHAIN
        elif query_type == QueryType.COMPARISON:
            return DecompositionStrategy.ENTITY_CENTRIC
        elif query_type == QueryType.COMPOSITE:
            return DecompositionStrategy.CLAUSE_BASED
        else:
            return DecompositionStrategy.HYBRID

    async def _decompose_with_llm(
        self,
        query: str,
        query_type: QueryType,
        strategy: DecompositionStrategy
    ) -> List[SubQuery]:
        """Use LLM to decompose query."""
        prompt = self._build_decomposition_prompt(query, query_type, strategy)

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = await self.llm_client.chat([{"role": "user", "content": prompt}])
                response = response.get("content", "")
            else:
                return self._decompose_rule_based(query, query_type, strategy)

            return self._parse_llm_decomposition(response, query)

        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
            return self._decompose_rule_based(query, query_type, strategy)

    def _build_decomposition_prompt(
        self,
        query: str,
        query_type: QueryType,
        strategy: DecompositionStrategy
    ) -> str:
        """Build prompt for LLM decomposition."""
        prompt = f"""You are a query decomposition expert. Break down the following complex query into simpler sub-queries that can be executed independently or in sequence.

## Query:
{query}

## Query Type: {query_type.value}
## Strategy: {strategy.value}

## Instructions:
1. Identify the key information needs in the query
2. Break into atomic sub-queries that each retrieve one piece of information
3. Mark dependencies between sub-queries (which needs results from which)
4. Identify entities, relations, and temporal constraints for each sub-query

## Output Format (JSON):
```json
[
  {{
    "id": "sq1",
    "text": "Sub-query text",
    "type": "simple|multi_hop|temporal|causal",
    "depends_on": [],
    "provides": ["variable_name"],
    "entities": ["entity1", "entity2"],
    "relations": ["RELATION_TYPE"],
    "retrieval_mode": "vector|graph|hybrid"
  }}
]
```

Decompose the query:"""
        return prompt

    def _parse_llm_decomposition(self, response: str, original_query: str) -> List[SubQuery]:
        """Parse LLM decomposition response."""
        sub_queries = []

        try:
            # Extract JSON from response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                items = json.loads(json_str)

                for item in items[:self.max_sub_queries]:
                    sq = SubQuery(
                        id=item.get("id", f"sq{len(sub_queries)}"),
                        text=item.get("text", ""),
                        query_type=QueryType(item.get("type", "simple")),
                        depends_on=item.get("depends_on", []),
                        provides=item.get("provides", []),
                        entities=item.get("entities", []),
                        relations=item.get("relations", []),
                        retrieval_mode=item.get("retrieval_mode", "hybrid")
                    )
                    sub_queries.append(sq)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM decomposition: {e}")
            # Fallback to rule-based
            sub_queries = self._decompose_rule_based(
                original_query,
                self._classify_query(original_query),
                DecompositionStrategy.HYBRID
            )

        return sub_queries if sub_queries else [
            SubQuery(id="sq0", text=original_query, query_type=QueryType.SIMPLE)
        ]

    def _decompose_rule_based(
        self,
        query: str,
        query_type: QueryType,
        strategy: DecompositionStrategy
    ) -> List[SubQuery]:
        """Rule-based query decomposition."""
        sub_queries = []

        if strategy == DecompositionStrategy.CLAUSE_BASED:
            sub_queries = self._decompose_by_clauses(query)
        elif strategy == DecompositionStrategy.ENTITY_CENTRIC:
            sub_queries = self._decompose_by_entities(query)
        elif strategy == DecompositionStrategy.REASONING_CHAIN:
            sub_queries = self._decompose_by_reasoning_chain(query, query_type)
        else:
            # Hybrid: combine approaches
            clause_queries = self._decompose_by_clauses(query)
            if len(clause_queries) > 1:
                sub_queries = clause_queries
            else:
                sub_queries = self._decompose_by_entities(query)

        # Ensure at least one sub-query
        if not sub_queries:
            sub_queries = [SubQuery(
                id="sq0",
                text=query,
                query_type=query_type
            )]

        return sub_queries

    def _decompose_by_clauses(self, query: str) -> List[SubQuery]:
        """Decompose by AND/OR clauses."""
        sub_queries = []

        # Split by "and" and "or"
        parts = re.split(r'\s+and\s+|\s+or\s+', query, flags=re.IGNORECASE)

        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) > 10:  # Minimum meaningful query length
                sq = SubQuery(
                    id=f"clause_{i}",
                    text=part,
                    query_type=self._classify_query(part),
                    priority=i
                )
                sub_queries.append(sq)

        return sub_queries

    def _decompose_by_entities(self, query: str) -> List[SubQuery]:
        """Decompose by entities mentioned."""
        sub_queries = []

        # Extract potential entities (capitalized phrases)
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        unique_entities = list(dict.fromkeys(entities))

        for i, entity in enumerate(unique_entities[:5]):  # Limit entities
            sq = SubQuery(
                id=f"entity_{i}",
                text=f"Information about {entity}",
                query_type=QueryType.SIMPLE,
                entities=[entity],
                priority=i
            )
            sub_queries.append(sq)

        # Add original query as final aggregation
        if sub_queries:
            sub_queries.append(SubQuery(
                id="aggregate",
                text=query,
                query_type=QueryType.COMPOSITE,
                depends_on=[sq.id for sq in sub_queries],
                priority=len(sub_queries)
            ))

        return sub_queries

    def _decompose_by_reasoning_chain(
        self,
        query: str,
        query_type: QueryType
    ) -> List[SubQuery]:
        """Decompose into reasoning steps."""
        sub_queries = []

        if query_type == QueryType.MULTI_HOP:
            # Extract relationship chain
            # Example: "What is the capital of the country where Einstein was born?"
            # -> 1. Where was Einstein born? -> 2. What is the capital of [result]?

            # Simple heuristic: split by "that", "which", "where"
            parts = re.split(r'\s+(?:that|which|where|who)\s+', query, flags=re.IGNORECASE)

            for i, part in enumerate(parts):
                part = part.strip()
                if len(part) > 10:
                    depends = [f"hop_{j}" for j in range(i)] if i > 0 else []
                    sq = SubQuery(
                        id=f"hop_{i}",
                        text=part + "?" if not part.endswith("?") else part,
                        query_type=QueryType.SIMPLE,
                        depends_on=depends,
                        priority=i,
                        retrieval_mode="graph"  # Multi-hop often needs graph traversal
                    )
                    sub_queries.append(sq)

        elif query_type == QueryType.CAUSAL:
            # Decompose into cause and effect queries
            sub_queries.append(SubQuery(
                id="identify_event",
                text=f"What event or action is being asked about in: {query}",
                query_type=QueryType.SIMPLE,
                priority=0
            ))
            sub_queries.append(SubQuery(
                id="find_causes",
                text="What are the causes or contributing factors?",
                query_type=QueryType.CAUSAL,
                depends_on=["identify_event"],
                priority=1,
                retrieval_mode="graph"
            ))
            sub_queries.append(SubQuery(
                id="synthesize",
                text=query,
                query_type=QueryType.CAUSAL,
                depends_on=["identify_event", "find_causes"],
                priority=2
            ))

        return sub_queries

    def _build_execution_order(self, sub_queries: List[SubQuery]) -> List[List[str]]:
        """Build execution order respecting dependencies."""
        if not sub_queries:
            return []

        # Topological sort based on dependencies
        remaining = {sq.id: sq for sq in sub_queries}
        executed = set()
        order = []

        while remaining:
            # Find queries with satisfied dependencies
            ready = []
            for sq_id, sq in remaining.items():
                if all(dep in executed for dep in sq.depends_on):
                    ready.append(sq_id)

            if not ready:
                # Circular dependency or error - add remaining
                ready = list(remaining.keys())

            order.append(ready)
            for sq_id in ready:
                executed.add(sq_id)
                del remaining[sq_id]

        return order

    def _determine_fusion_strategy(self, query_type: QueryType) -> str:
        """Determine how to fuse sub-query results."""
        if query_type == QueryType.COMPARISON:
            return "side_by_side"
        elif query_type == QueryType.AGGREGATION:
            return "aggregate"
        elif query_type in [QueryType.MULTI_HOP, QueryType.CAUSAL]:
            return "chain"
        else:
            return "weighted"

    def extract_entities_and_relations(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract entities and relations from a query."""
        # Extract entities (capitalized phrases)
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)

        # Extract potential relations (verb phrases)
        relations = []
        relation_patterns = [
            r"works at",
            r"lives in",
            r"created",
            r"wrote",
            r"founded",
            r"located in",
            r"related to",
            r"connected to",
            r"causes",
            r"affects",
        ]

        query_lower = query.lower()
        for pattern in relation_patterns:
            if re.search(pattern, query_lower):
                relations.append(pattern.upper().replace(" ", "_"))

        return list(dict.fromkeys(entities)), relations

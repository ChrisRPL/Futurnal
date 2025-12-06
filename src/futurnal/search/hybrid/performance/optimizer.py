"""Query Plan Optimizer for Hybrid Search.

Implements cost-based query plan optimization to select the best
retrieval strategy based on query intent, constraints, and cache state.

Key Features:
- Strategy selection based on query type and historical performance
- Cache-aware cost adjustment
- Parallel execution planning where possible
- Early termination thresholds

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md

Option B Compliance:
- Supports temporal-first queries (critical for Phase 3)
- Causal chain strategy for causal queries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from futurnal.search.hybrid.performance.cache import CacheLayer, MultiLayerCache
    from futurnal.search.hybrid.performance.profiler import PerformanceProfiler
    from futurnal.search.hybrid.types import QueryIntent

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    VECTOR_ONLY = "vector"
    GRAPH_ONLY = "graph"
    TEMPORAL_FIRST = "temporal_first"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    CAUSAL_CHAIN = "causal_chain"


@dataclass
class QueryPlanCost:
    """Estimated cost for a query plan."""

    latency_ms: float  # Estimated latency
    resource_usage: float  # 0-1 scale
    cache_probability: float  # Likelihood of cache hit
    accuracy_score: float  # Expected relevance

    @property
    def total_cost(self) -> float:
        """Weighted cost function for plan comparison.

        Lower is better. Balances latency, resources, cache efficiency,
        and accuracy.
        """
        return (
            self.latency_ms * 0.4
            + self.resource_usage * 100 * 0.2
            + (1 - self.cache_probability) * 100 * 0.2
            + (1 - self.accuracy_score) * 100 * 0.2
        )


@dataclass
class QueryPlan:
    """Execution plan for a query."""

    strategy: RetrievalStrategy
    estimated_cost: QueryPlanCost
    steps: List[Dict[str, Any]]
    parallel_steps: List[List[str]]  # Steps that can run in parallel
    early_termination_threshold: float  # Stop if confidence exceeds
    timeout_ms: float


# Baseline costs for each strategy (will be adjusted based on cache/history)
BASELINE_STRATEGY_COSTS: Dict[RetrievalStrategy, QueryPlanCost] = {
    RetrievalStrategy.VECTOR_ONLY: QueryPlanCost(
        latency_ms=50,
        resource_usage=0.3,
        cache_probability=0.7,
        accuracy_score=0.75,
    ),
    RetrievalStrategy.GRAPH_ONLY: QueryPlanCost(
        latency_ms=100,
        resource_usage=0.4,
        cache_probability=0.5,
        accuracy_score=0.85,
    ),
    RetrievalStrategy.TEMPORAL_FIRST: QueryPlanCost(
        latency_ms=150,
        resource_usage=0.5,
        cache_probability=0.4,
        accuracy_score=0.90,
    ),
    RetrievalStrategy.HYBRID_PARALLEL: QueryPlanCost(
        latency_ms=200,
        resource_usage=0.7,
        cache_probability=0.3,
        accuracy_score=0.92,
    ),
    RetrievalStrategy.HYBRID_SEQUENTIAL: QueryPlanCost(
        latency_ms=300,
        resource_usage=0.5,
        cache_probability=0.4,
        accuracy_score=0.93,
    ),
    RetrievalStrategy.CAUSAL_CHAIN: QueryPlanCost(
        latency_ms=500,
        resource_usage=0.8,
        cache_probability=0.2,
        accuracy_score=0.95,
    ),
}


# Strategy steps definitions
STRATEGY_STEPS: Dict[RetrievalStrategy, List[Dict[str, Any]]] = {
    RetrievalStrategy.VECTOR_ONLY: [
        {"name": "embed_query", "component": "embedding_router"},
        {"name": "vector_search", "component": "vector_store"},
        {"name": "rank_results", "component": "ranker"},
    ],
    RetrievalStrategy.GRAPH_ONLY: [
        {"name": "identify_seeds", "component": "seed_finder"},
        {"name": "graph_search", "component": "pkg"},
        {"name": "rank_results", "component": "ranker"},
    ],
    RetrievalStrategy.TEMPORAL_FIRST: [
        {"name": "extract_temporal", "component": "temporal_parser"},
        {"name": "temporal_filter", "component": "temporal_index"},
        {"name": "embed_query", "component": "embedding_router"},
        {"name": "vector_search", "component": "vector_store"},
        {"name": "temporal_boost", "component": "ranker"},
        {"name": "rank_results", "component": "ranker"},
    ],
    RetrievalStrategy.HYBRID_PARALLEL: [
        {"name": "embed_query", "component": "embedding_router"},
        {"name": "vector_search", "component": "vector_store"},
        {"name": "graph_search", "component": "pkg"},
        {"name": "fuse_results", "component": "fusion"},
        {"name": "rank_results", "component": "ranker"},
    ],
    RetrievalStrategy.HYBRID_SEQUENTIAL: [
        {"name": "embed_query", "component": "embedding_router"},
        {"name": "vector_search", "component": "vector_store"},
        {"name": "expand_graph", "component": "pkg"},
        {"name": "re_rank", "component": "ranker"},
    ],
    RetrievalStrategy.CAUSAL_CHAIN: [
        {"name": "identify_anchor", "component": "causal_retriever"},
        {"name": "traverse_causes", "component": "pkg"},
        {"name": "traverse_effects", "component": "pkg"},
        {"name": "build_chain", "component": "causal_retriever"},
        {"name": "rank_by_relevance", "component": "ranker"},
    ],
}


# Parallel step groups per strategy
PARALLEL_STEP_GROUPS: Dict[RetrievalStrategy, List[List[str]]] = {
    RetrievalStrategy.HYBRID_PARALLEL: [["vector_search", "graph_search"]],
    RetrievalStrategy.CAUSAL_CHAIN: [["traverse_causes", "traverse_effects"]],
}


# Intent to candidate strategies mapping
INTENT_STRATEGY_MAP: Dict[str, List[RetrievalStrategy]] = {
    "temporal": [
        RetrievalStrategy.TEMPORAL_FIRST,
        RetrievalStrategy.HYBRID_SEQUENTIAL,
    ],
    "causal": [
        RetrievalStrategy.CAUSAL_CHAIN,
        RetrievalStrategy.GRAPH_ONLY,
    ],
    "exploratory": [
        RetrievalStrategy.HYBRID_PARALLEL,
        RetrievalStrategy.VECTOR_ONLY,
    ],
    "factual": [
        RetrievalStrategy.VECTOR_ONLY,
        RetrievalStrategy.HYBRID_PARALLEL,
    ],
    "lookup": [
        RetrievalStrategy.VECTOR_ONLY,
        RetrievalStrategy.GRAPH_ONLY,
    ],
    "code": [
        RetrievalStrategy.VECTOR_ONLY,  # CodeBERT embeddings
        RetrievalStrategy.GRAPH_ONLY,
    ],
}


class QueryPlanOptimizer:
    """Cost-based query plan optimization.

    Optimizations:
    1. Strategy selection based on query type and history
    2. Parallel execution where possible
    3. Early termination for high-confidence results
    4. Cache-aware planning

    Integration Points:
    - MultiLayerCache: Checks cache state for planning
    - PerformanceProfiler: Uses historical latency data

    Example:
        >>> optimizer = QueryPlanOptimizer(cache, profiler)
        >>> intent = QueryIntent(primary_intent="temporal", confidence=0.9)
        >>> plan = optimizer.optimize("what happened last week", intent)
        >>> print(plan.strategy)
    """

    def __init__(
        self,
        cache: Optional["MultiLayerCache"] = None,
        profiler: Optional["PerformanceProfiler"] = None,
    ) -> None:
        """Initialize query plan optimizer.

        Args:
            cache: Optional MultiLayerCache for cache-aware planning
            profiler: Optional PerformanceProfiler for historical data
        """
        self.cache = cache
        self.profiler = profiler
        self.strategy_costs = BASELINE_STRATEGY_COSTS.copy()

    def optimize(
        self,
        query: str,
        intent: "QueryIntent",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        """Generate optimized query plan.

        Args:
            query: User query string
            intent: Classified query intent
            constraints: Optional constraints:
                - max_latency_ms: Maximum acceptable latency
                - min_accuracy: Minimum acceptable accuracy score

        Returns:
            Optimized QueryPlan for execution
        """
        constraints = constraints or {}
        max_latency = constraints.get("max_latency_ms", 1000)
        min_accuracy = constraints.get("min_accuracy", 0.7)

        # Get candidate strategies for this intent
        candidates = self._get_candidate_strategies(intent)

        # Adjust costs based on cache state
        adjusted_costs = self._adjust_for_cache_state(candidates, query)

        # Adjust costs based on historical performance
        if self.profiler:
            adjusted_costs = self._adjust_for_history(adjusted_costs, intent)

        # Select best strategy within constraints
        best_strategy = self._select_best_strategy(
            adjusted_costs, max_latency, min_accuracy
        )

        # Build execution plan
        plan = self._build_plan(best_strategy, intent, max_latency)

        logger.debug(
            f"Optimized plan: strategy={best_strategy.value}, "
            f"estimated_latency={plan.estimated_cost.latency_ms}ms"
        )

        return plan

    def _get_candidate_strategies(
        self,
        intent: "QueryIntent",
    ) -> List[RetrievalStrategy]:
        """Get strategies suitable for this intent.

        Args:
            intent: Classified query intent

        Returns:
            List of candidate retrieval strategies
        """
        # Get primary intent type
        intent_type = getattr(intent, "primary_intent", None)
        if intent_type is None:
            intent_type = str(intent.value) if hasattr(intent, "value") else "exploratory"

        # Handle enum vs string
        if hasattr(intent_type, "value"):
            intent_type = intent_type.value

        intent_type_str = str(intent_type).lower()

        return INTENT_STRATEGY_MAP.get(
            intent_type_str, [RetrievalStrategy.HYBRID_PARALLEL]
        )

    def _adjust_for_cache_state(
        self,
        candidates: List[RetrievalStrategy],
        query: str,
    ) -> Dict[RetrievalStrategy, QueryPlanCost]:
        """Adjust costs based on cache state.

        Args:
            candidates: Candidate strategies
            query: Query string for cache estimation

        Returns:
            Dict mapping strategies to adjusted costs
        """
        adjusted: Dict[RetrievalStrategy, QueryPlanCost] = {}

        for strategy in candidates:
            base_cost = self.strategy_costs[strategy]

            # Estimate cache hit probability
            cache_hit_prob = self._estimate_cache_hit(query, strategy)

            # Reduce latency and resource usage based on cache probability
            adjusted[strategy] = QueryPlanCost(
                latency_ms=base_cost.latency_ms * (1 - cache_hit_prob * 0.8),
                resource_usage=base_cost.resource_usage * (1 - cache_hit_prob * 0.5),
                cache_probability=cache_hit_prob,
                accuracy_score=base_cost.accuracy_score,
            )

        return adjusted

    def _estimate_cache_hit(
        self,
        query: str,
        strategy: RetrievalStrategy,
    ) -> float:
        """Estimate probability of cache hit.

        Args:
            query: Query string
            strategy: Retrieval strategy

        Returns:
            Estimated cache hit probability (0-1)
        """
        if not self.cache:
            return 0.3  # Default assumption

        from futurnal.search.hybrid.performance.cache import CacheLayer

        # Map strategies to primary cache layers
        layer_map = {
            RetrievalStrategy.VECTOR_ONLY: CacheLayer.EMBEDDING,
            RetrievalStrategy.GRAPH_ONLY: CacheLayer.GRAPH_TRAVERSAL,
            RetrievalStrategy.TEMPORAL_FIRST: CacheLayer.TEMPORAL_INDEX,
        }

        layer = layer_map.get(strategy, CacheLayer.QUERY_RESULT)
        return self.cache.stats.hit_rate(layer)

    def _adjust_for_history(
        self,
        costs: Dict[RetrievalStrategy, QueryPlanCost],
        intent: "QueryIntent",
    ) -> Dict[RetrievalStrategy, QueryPlanCost]:
        """Adjust costs based on historical performance.

        Args:
            costs: Current cost estimates
            intent: Query intent

        Returns:
            Adjusted cost estimates
        """
        adjusted: Dict[RetrievalStrategy, QueryPlanCost] = {}

        # Get primary intent type
        intent_type = getattr(intent, "primary_intent", None)
        if intent_type is None:
            intent_type = str(intent.value) if hasattr(intent, "value") else "exploratory"
        if hasattr(intent_type, "value"):
            intent_type = intent_type.value
        intent_type_str = str(intent_type).lower()

        for strategy, cost in costs.items():
            historical_latency = None
            if self.profiler:
                historical_latency = self.profiler.get_avg_latency(
                    strategy=strategy.value,
                    intent_type=intent_type_str,
                )

            if historical_latency:
                # Blend historical with baseline (50/50)
                blended_latency = (cost.latency_ms + historical_latency) / 2
                adjusted[strategy] = QueryPlanCost(
                    latency_ms=blended_latency,
                    resource_usage=cost.resource_usage,
                    cache_probability=cost.cache_probability,
                    accuracy_score=cost.accuracy_score,
                )
            else:
                adjusted[strategy] = cost

        return adjusted

    def _select_best_strategy(
        self,
        costs: Dict[RetrievalStrategy, QueryPlanCost],
        max_latency: float,
        min_accuracy: float,
    ) -> RetrievalStrategy:
        """Select best strategy within constraints.

        Args:
            costs: Strategy costs
            max_latency: Maximum acceptable latency
            min_accuracy: Minimum acceptable accuracy

        Returns:
            Best strategy meeting constraints
        """
        valid_strategies = [
            (strategy, cost)
            for strategy, cost in costs.items()
            if cost.latency_ms <= max_latency and cost.accuracy_score >= min_accuracy
        ]

        if not valid_strategies:
            # Fallback to fastest strategy if none meet constraints
            logger.warning("No strategy meets constraints, selecting fastest")
            return min(costs.items(), key=lambda x: x[1].latency_ms)[0]

        # Select lowest total cost
        return min(valid_strategies, key=lambda x: x[1].total_cost)[0]

    def _build_plan(
        self,
        strategy: RetrievalStrategy,
        intent: "QueryIntent",
        timeout_ms: float,
    ) -> QueryPlan:
        """Build detailed execution plan.

        Args:
            strategy: Selected strategy
            intent: Query intent
            timeout_ms: Query timeout

        Returns:
            Complete QueryPlan
        """
        cost = self.strategy_costs[strategy]
        steps = STRATEGY_STEPS.get(strategy, [])
        parallel_steps = PARALLEL_STEP_GROUPS.get(strategy, [])

        return QueryPlan(
            strategy=strategy,
            estimated_cost=cost,
            steps=steps,
            parallel_steps=parallel_steps,
            early_termination_threshold=0.95,
            timeout_ms=timeout_ms,
        )

"""
Query Planner - Optimizes query execution plans.

Creates efficient execution plans for sub-queries:
- Resource estimation
- Parallel execution optimization
- Result caching strategies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from .decomposer import SubQuery, QueryPlan, QueryType

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for query plan nodes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CACHED = "cached"
    STREAMING = "streaming"


@dataclass
class PlanNode:
    """A node in the execution plan tree."""
    id: str
    sub_query: SubQuery
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL

    # Resource estimates
    estimated_time_ms: float = 100.0
    estimated_memory_mb: float = 10.0
    estimated_results: int = 10

    # Execution details
    retrieval_source: str = "hybrid"  # "vector", "graph", "cache"
    cache_key: Optional[str] = None
    timeout_ms: int = 5000

    # Children for tree structure
    children: List[str] = field(default_factory=list)

    # Results
    executed: bool = False
    result: Optional[Any] = None
    actual_time_ms: float = 0.0


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    query_plan: QueryPlan
    nodes: Dict[str, PlanNode]
    root_nodes: List[str]

    # Execution stages
    stages: List[List[str]]  # Groups of nodes to execute together

    # Global settings
    total_timeout_ms: int = 30000
    max_parallel: int = 5
    use_caching: bool = True

    # Optimization metadata
    optimization_applied: List[str] = field(default_factory=list)
    estimated_total_time_ms: float = 0.0


class QueryPlanner:
    """
    Creates optimized execution plans for queries.

    Optimizations include:
    - Parallel execution of independent sub-queries
    - Result caching
    - Early termination
    - Resource-based scheduling
    """

    def __init__(
        self,
        cache_manager: Optional[Any] = None,
        max_parallel: int = 5,
        default_timeout_ms: int = 5000
    ):
        self.cache_manager = cache_manager
        self.max_parallel = max_parallel
        self.default_timeout_ms = default_timeout_ms

    def create_plan(self, query_plan: QueryPlan) -> ExecutionPlan:
        """
        Create an optimized execution plan.

        Args:
            query_plan: The decomposed query plan

        Returns:
            ExecutionPlan ready for execution
        """
        # Create plan nodes
        nodes = {}
        for sq in query_plan.sub_queries:
            node = self._create_plan_node(sq)
            nodes[sq.id] = node

        # Identify root nodes (no dependencies)
        root_nodes = [
            sq.id for sq in query_plan.sub_queries
            if not sq.depends_on
        ]

        # Build execution stages
        stages = self._build_execution_stages(query_plan, nodes)

        # Create execution plan
        plan = ExecutionPlan(
            query_plan=query_plan,
            nodes=nodes,
            root_nodes=root_nodes,
            stages=stages,
            max_parallel=self.max_parallel
        )

        # Apply optimizations
        plan = self._optimize_plan(plan)

        # Calculate estimated time
        plan.estimated_total_time_ms = self._estimate_total_time(plan)

        return plan

    def _create_plan_node(self, sub_query: SubQuery) -> PlanNode:
        """Create a plan node for a sub-query."""
        # Estimate resources based on query type
        time_estimate = self._estimate_time(sub_query)
        memory_estimate = self._estimate_memory(sub_query)
        result_estimate = self._estimate_results(sub_query)

        # Determine retrieval source
        retrieval_source = sub_query.retrieval_mode
        if retrieval_source == "hybrid":
            # Choose based on query type
            if sub_query.query_type in [QueryType.MULTI_HOP, QueryType.CAUSAL]:
                retrieval_source = "graph"
            elif sub_query.query_type == QueryType.SIMPLE:
                retrieval_source = "vector"

        # Generate cache key
        cache_key = self._generate_cache_key(sub_query) if self.cache_manager else None

        return PlanNode(
            id=sub_query.id,
            sub_query=sub_query,
            estimated_time_ms=time_estimate,
            estimated_memory_mb=memory_estimate,
            estimated_results=result_estimate,
            retrieval_source=retrieval_source,
            cache_key=cache_key,
            timeout_ms=self.default_timeout_ms,
            children=sub_query.depends_on
        )

    def _estimate_time(self, sub_query: SubQuery) -> float:
        """Estimate execution time for a sub-query."""
        base_time = 100.0  # Base time in ms

        # Adjust based on query type
        type_multipliers = {
            QueryType.SIMPLE: 1.0,
            QueryType.MULTI_HOP: 3.0,
            QueryType.TEMPORAL: 1.5,
            QueryType.CAUSAL: 4.0,
            QueryType.COMPARISON: 2.0,
            QueryType.AGGREGATION: 2.5,
            QueryType.COMPOSITE: 5.0,
        }

        multiplier = type_multipliers.get(sub_query.query_type, 1.0)

        # Adjust based on complexity
        complexity_factor = sub_query.estimated_complexity

        # Adjust based on retrieval mode
        mode_factors = {
            "vector": 1.0,
            "graph": 2.0,
            "hybrid": 1.5,
        }
        mode_factor = mode_factors.get(sub_query.retrieval_mode, 1.0)

        return base_time * multiplier * complexity_factor * mode_factor

    def _estimate_memory(self, sub_query: SubQuery) -> float:
        """Estimate memory usage for a sub-query."""
        base_memory = 10.0  # Base memory in MB

        # Adjust based on expected results
        if sub_query.query_type == QueryType.AGGREGATION:
            return base_memory * 0.5  # Less memory for aggregations
        elif sub_query.query_type == QueryType.COMPOSITE:
            return base_memory * 3.0  # More memory for composite queries

        return base_memory

    def _estimate_results(self, sub_query: SubQuery) -> int:
        """Estimate number of results."""
        if sub_query.query_type == QueryType.SIMPLE:
            return 5
        elif sub_query.query_type in [QueryType.MULTI_HOP, QueryType.CAUSAL]:
            return 10
        elif sub_query.query_type == QueryType.AGGREGATION:
            return 1
        return 10

    def _generate_cache_key(self, sub_query: SubQuery) -> str:
        """Generate cache key for a sub-query."""
        import hashlib
        content = f"{sub_query.text}|{sub_query.query_type.value}|{sub_query.retrieval_mode}"
        return hashlib.md5(content.encode()).hexdigest()

    def _build_execution_stages(
        self,
        query_plan: QueryPlan,
        nodes: Dict[str, PlanNode]
    ) -> List[List[str]]:
        """Build execution stages for parallel execution."""
        # Use the execution order from query plan, but optimize for parallelism
        stages = []

        for group in query_plan.execution_order:
            if len(group) <= self.max_parallel:
                stages.append(group)
            else:
                # Split large groups
                for i in range(0, len(group), self.max_parallel):
                    stages.append(group[i:i + self.max_parallel])

        return stages

    def _optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Apply optimizations to the execution plan."""
        optimizations = []

        # Optimization 1: Cache lookup
        plan = self._optimize_caching(plan)
        if any(n.execution_mode == ExecutionMode.CACHED for n in plan.nodes.values()):
            optimizations.append("cache_lookup")

        # Optimization 2: Parallel execution
        plan = self._optimize_parallelism(plan)
        parallel_stages = sum(1 for stage in plan.stages if len(stage) > 1)
        if parallel_stages > 0:
            optimizations.append(f"parallel_execution:{parallel_stages}_stages")

        # Optimization 3: Early termination for simple queries
        if plan.query_plan.query_type == QueryType.SIMPLE:
            for node in plan.nodes.values():
                node.timeout_ms = min(node.timeout_ms, 2000)
            optimizations.append("early_termination")

        # Optimization 4: Streaming for large result sets
        plan = self._optimize_streaming(plan)
        if any(n.execution_mode == ExecutionMode.STREAMING for n in plan.nodes.values()):
            optimizations.append("streaming")

        plan.optimization_applied = optimizations
        return plan

    def _optimize_caching(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Mark nodes that can use cached results."""
        if not self.cache_manager:
            return plan

        for node in plan.nodes.values():
            if node.cache_key:
                # Check if result is in cache
                cached = self.cache_manager.get(node.cache_key) if hasattr(self.cache_manager, 'get') else None
                if cached:
                    node.execution_mode = ExecutionMode.CACHED
                    node.result = cached

        return plan

    def _optimize_parallelism(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize for parallel execution."""
        for stage in plan.stages:
            if len(stage) > 1:
                for node_id in stage:
                    plan.nodes[node_id].execution_mode = ExecutionMode.PARALLEL

        return plan

    def _optimize_streaming(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Enable streaming for large result sets."""
        for node in plan.nodes.values():
            if node.estimated_results > 100:
                node.execution_mode = ExecutionMode.STREAMING

        return plan

    def _estimate_total_time(self, plan: ExecutionPlan) -> float:
        """Estimate total execution time."""
        total_time = 0.0

        for stage in plan.stages:
            # Time for a stage is max of parallel nodes
            stage_times = [
                plan.nodes[node_id].estimated_time_ms
                for node_id in stage
                if plan.nodes[node_id].execution_mode != ExecutionMode.CACHED
            ]
            if stage_times:
                total_time += max(stage_times)

        return total_time

    def replan_on_failure(
        self,
        plan: ExecutionPlan,
        failed_node_id: str,
        error: str
    ) -> ExecutionPlan:
        """
        Adjust plan after a node failure.

        Args:
            plan: Current execution plan
            failed_node_id: ID of the failed node
            error: Error message

        Returns:
            Updated execution plan
        """
        logger.warning(f"Replanning due to failure in {failed_node_id}: {error}")

        # Mark failed node
        if failed_node_id in plan.nodes:
            plan.nodes[failed_node_id].executed = True
            plan.nodes[failed_node_id].result = {"error": error}

        # Try alternative retrieval source
        node = plan.nodes.get(failed_node_id)
        if node:
            if node.retrieval_source == "graph":
                node.retrieval_source = "vector"
                node.executed = False
                logger.info(f"Switching {failed_node_id} to vector retrieval")
            elif node.retrieval_source == "vector":
                node.retrieval_source = "graph"
                node.executed = False
                logger.info(f"Switching {failed_node_id} to graph retrieval")

        return plan

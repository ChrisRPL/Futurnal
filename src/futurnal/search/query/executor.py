"""
Query Executor - Executes optimized query plans.

Handles:
- Parallel sub-query execution
- Result fusion and ranking
- Error handling and retries
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import time

from .decomposer import SubQuery, QueryPlan, QueryType
from .planner import ExecutionPlan, PlanNode, ExecutionMode, QueryPlanner

logger = logging.getLogger(__name__)


@dataclass
class SubQueryResult:
    """Result from a sub-query execution."""
    sub_query_id: str
    query_text: str
    results: List[Dict[str, Any]]
    scores: List[float]
    execution_time_ms: float
    source: str  # "vector", "graph", "cache"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    """Final fused result from all sub-queries."""
    original_query: str
    query_type: QueryType
    results: List[Dict[str, Any]]
    scores: List[float]
    sub_query_results: List[SubQueryResult]

    # Execution metadata
    total_time_ms: float
    num_sub_queries: int
    parallel_stages: int
    cache_hits: int

    # Quality metrics
    coverage: float  # Fraction of sub-queries that returned results
    confidence: float  # Overall confidence in results


class QueryExecutor:
    """
    Executes query plans and fuses results.

    Supports:
    - Parallel execution of independent sub-queries
    - Multiple retrieval backends (vector, graph, hybrid)
    - Result caching
    - Graceful error handling
    """

    def __init__(
        self,
        vector_retriever: Optional[Any] = None,
        graph_retriever: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        max_concurrent: int = 5,
        default_top_k: int = 10
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.cache_manager = cache_manager
        self.max_concurrent = max_concurrent
        self.default_top_k = default_top_k

        # Execution statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "vector_calls": 0,
            "graph_calls": 0,
            "errors": 0
        }

    async def execute(
        self,
        plan: ExecutionPlan,
        top_k: int = 10
    ) -> FusedResult:
        """
        Execute a query plan and return fused results.

        Args:
            plan: Execution plan from QueryPlanner
            top_k: Number of final results to return

        Returns:
            FusedResult with combined results from all sub-queries
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        sub_query_results: List[SubQueryResult] = []
        cache_hits = 0

        # Execute stages sequentially, sub-queries within stage in parallel
        for stage_idx, stage in enumerate(plan.stages):
            logger.debug(f"Executing stage {stage_idx + 1}/{len(plan.stages)} with {len(stage)} queries")

            # Execute all queries in this stage
            stage_tasks = []
            for node_id in stage:
                node = plan.nodes[node_id]

                # Check if already executed (cached)
                if node.execution_mode == ExecutionMode.CACHED and node.result:
                    sub_query_results.append(SubQueryResult(
                        sub_query_id=node.id,
                        query_text=node.sub_query.text,
                        results=node.result.get("results", []),
                        scores=node.result.get("scores", []),
                        execution_time_ms=0,
                        source="cache"
                    ))
                    cache_hits += 1
                    continue

                # Create execution task
                task = self._execute_node(node, top_k)
                stage_tasks.append(task)

            # Execute stage tasks with concurrency limit
            if stage_tasks:
                semaphore = asyncio.Semaphore(self.max_concurrent)

                async def limited_task(task):
                    async with semaphore:
                        return await task

                results = await asyncio.gather(
                    *[limited_task(t) for t in stage_tasks],
                    return_exceptions=True
                )

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Sub-query execution failed: {result}")
                        self.stats["errors"] += 1
                    else:
                        sub_query_results.append(result)

                        # Cache successful results
                        if self.cache_manager and result.error is None:
                            node = plan.nodes.get(result.sub_query_id)
                            if node and node.cache_key:
                                self.cache_manager.set(node.cache_key, {
                                    "results": result.results,
                                    "scores": result.scores
                                })

        # Fuse results from all sub-queries
        fused_results, fused_scores = self._fuse_results(
            sub_query_results, plan.query_plan, top_k
        )

        # Calculate coverage
        successful = sum(1 for r in sub_query_results if r.error is None and r.results)
        coverage = successful / len(sub_query_results) if sub_query_results else 0.0

        # Calculate confidence
        confidence = self._calculate_confidence(sub_query_results, fused_scores)

        total_time_ms = (time.time() - start_time) * 1000

        return FusedResult(
            original_query=plan.query_plan.original_query,
            query_type=plan.query_plan.query_type,
            results=fused_results,
            scores=fused_scores,
            sub_query_results=sub_query_results,
            total_time_ms=total_time_ms,
            num_sub_queries=len(plan.nodes),
            parallel_stages=len(plan.stages),
            cache_hits=cache_hits,
            coverage=coverage,
            confidence=confidence
        )

    def execute_sync(
        self,
        plan: ExecutionPlan,
        top_k: int = 10
    ) -> FusedResult:
        """Synchronous version of execute."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute(plan, top_k))

    async def _execute_node(
        self,
        node: PlanNode,
        top_k: int
    ) -> SubQueryResult:
        """Execute a single plan node."""
        start_time = time.time()

        try:
            if node.retrieval_source == "vector":
                results, scores = await self._vector_retrieve(node.sub_query, top_k)
                self.stats["vector_calls"] += 1
            elif node.retrieval_source == "graph":
                results, scores = await self._graph_retrieve(node.sub_query, top_k)
                self.stats["graph_calls"] += 1
            else:  # hybrid
                results, scores = await self._hybrid_retrieve(node.sub_query, top_k)
                self.stats["vector_calls"] += 1
                self.stats["graph_calls"] += 1

            execution_time_ms = (time.time() - start_time) * 1000

            # Update node
            node.executed = True
            node.actual_time_ms = execution_time_ms
            node.result = {"results": results, "scores": scores}

            return SubQueryResult(
                sub_query_id=node.id,
                query_text=node.sub_query.text,
                results=results,
                scores=scores,
                execution_time_ms=execution_time_ms,
                source=node.retrieval_source,
                metadata={
                    "query_type": node.sub_query.query_type.value,
                    "estimated_time_ms": node.estimated_time_ms
                }
            )

        except asyncio.TimeoutError:
            return SubQueryResult(
                sub_query_id=node.id,
                query_text=node.sub_query.text,
                results=[],
                scores=[],
                execution_time_ms=(time.time() - start_time) * 1000,
                source=node.retrieval_source,
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"Error executing node {node.id}: {e}")
            return SubQueryResult(
                sub_query_id=node.id,
                query_text=node.sub_query.text,
                results=[],
                scores=[],
                execution_time_ms=(time.time() - start_time) * 1000,
                source=node.retrieval_source,
                error=str(e)
            )

    async def _vector_retrieve(
        self,
        sub_query: SubQuery,
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Retrieve using vector similarity."""
        if not self.vector_retriever:
            return [], []

        # Apply time filter if specified
        filters = {}
        if sub_query.time_range:
            start, end = sub_query.time_range
            if start:
                filters["timestamp_gte"] = start.isoformat()
            if end:
                filters["timestamp_lte"] = end.isoformat()

        if hasattr(self.vector_retriever, "retrieve"):
            results = await self.vector_retriever.retrieve(
                query=sub_query.text,
                top_k=top_k,
                filters=filters if filters else None
            )
            return results.get("results", []), results.get("scores", [])
        elif hasattr(self.vector_retriever, "query"):
            # ChromaDB-style interface
            results = self.vector_retriever.query(
                query_texts=[sub_query.text],
                n_results=top_k
            )
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            formatted_results = []
            scores = []
            for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "id": meta.get("id", f"doc_{i}")
                })
                # Convert distance to similarity score
                scores.append(1.0 / (1.0 + dist))

            return formatted_results, scores

        return [], []

    async def _graph_retrieve(
        self,
        sub_query: SubQuery,
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Retrieve using graph traversal."""
        if not self.graph_retriever:
            return [], []

        if hasattr(self.graph_retriever, "retrieve"):
            results = await self.graph_retriever.retrieve(
                query=sub_query.text,
                top_k=top_k,
                query_type=sub_query.query_type.value
            )
            return results.get("results", []), results.get("scores", [])
        elif hasattr(self.graph_retriever, "search"):
            # Neo4j-style interface
            results = self.graph_retriever.search(
                query=sub_query.text,
                limit=top_k
            )
            formatted = []
            scores = []
            for r in results:
                formatted.append({
                    "content": r.get("description", r.get("name", "")),
                    "metadata": r,
                    "id": r.get("id", "")
                })
                scores.append(r.get("score", 0.5))
            return formatted, scores

        return [], []

    async def _hybrid_retrieve(
        self,
        sub_query: SubQuery,
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Retrieve using both vector and graph, then merge."""
        # Get results from both sources
        vector_results, vector_scores = await self._vector_retrieve(sub_query, top_k)
        graph_results, graph_scores = await self._graph_retrieve(sub_query, top_k)

        # Merge results using reciprocal rank fusion
        merged = self._reciprocal_rank_fusion(
            [vector_results, graph_results],
            [vector_scores, graph_scores],
            top_k
        )

        return merged

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict[str, Any]]],
        score_lists: List[List[float]],
        top_k: int,
        k: int = 60
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Fuse multiple result lists using reciprocal rank fusion."""
        # Build ID to result mapping
        id_to_result = {}
        id_to_rrf_score = {}

        for results, scores in zip(result_lists, score_lists):
            for rank, (result, score) in enumerate(zip(results, scores)):
                result_id = result.get("id", str(hash(str(result))))

                if result_id not in id_to_result:
                    id_to_result[result_id] = result
                    id_to_rrf_score[result_id] = 0.0

                # RRF score contribution
                id_to_rrf_score[result_id] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(
            id_to_rrf_score.keys(),
            key=lambda x: id_to_rrf_score[x],
            reverse=True
        )[:top_k]

        results = [id_to_result[id] for id in sorted_ids]
        scores = [id_to_rrf_score[id] for id in sorted_ids]

        return results, scores

    def _fuse_results(
        self,
        sub_query_results: List[SubQueryResult],
        query_plan: QueryPlan,
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Fuse results from multiple sub-queries."""
        if not sub_query_results:
            return [], []

        # Choose fusion strategy based on query type
        if query_plan.query_type == QueryType.COMPARISON:
            return self._fuse_comparison(sub_query_results, top_k)
        elif query_plan.query_type == QueryType.AGGREGATION:
            return self._fuse_aggregation(sub_query_results, top_k)
        elif query_plan.query_type == QueryType.COMPOSITE:
            return self._fuse_composite(sub_query_results, query_plan, top_k)
        else:
            # Default: reciprocal rank fusion
            result_lists = [r.results for r in sub_query_results if r.results]
            score_lists = [r.scores for r in sub_query_results if r.scores]
            return self._reciprocal_rank_fusion(result_lists, score_lists, top_k)

    def _fuse_comparison(
        self,
        sub_query_results: List[SubQueryResult],
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Fuse results for comparison queries - keep groups separate."""
        # Group results by sub-query
        grouped = []
        for sqr in sub_query_results:
            if sqr.results:
                grouped.append({
                    "query": sqr.query_text,
                    "results": sqr.results[:top_k // len(sub_query_results)],
                    "scores": sqr.scores[:top_k // len(sub_query_results)]
                })

        # Flatten for output, preserving group info
        all_results = []
        all_scores = []
        for group in grouped:
            for result, score in zip(group["results"], group["scores"]):
                result["comparison_group"] = group["query"]
                all_results.append(result)
                all_scores.append(score)

        return all_results[:top_k], all_scores[:top_k]

    def _fuse_aggregation(
        self,
        sub_query_results: List[SubQueryResult],
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Fuse results for aggregation queries - combine statistics."""
        # Collect all results for aggregation
        all_values = []
        for sqr in sub_query_results:
            for result in sqr.results:
                if "value" in result:
                    all_values.append(result["value"])

        if not all_values:
            # Return raw results if no aggregatable values
            result_lists = [r.results for r in sub_query_results if r.results]
            score_lists = [r.scores for r in sub_query_results if r.scores]
            return self._reciprocal_rank_fusion(result_lists, score_lists, top_k)

        # Create aggregated result
        aggregated = {
            "type": "aggregation",
            "count": len(all_values),
            "values": all_values,
            "metadata": {
                "sub_queries": len(sub_query_results)
            }
        }

        return [aggregated], [1.0]

    def _fuse_composite(
        self,
        sub_query_results: List[SubQueryResult],
        query_plan: QueryPlan,
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """Fuse results for composite queries - respect dependency order."""
        # Build dependency-ordered results
        ordered_results = []

        # Sort sub-queries by dependency order
        id_to_result = {sqr.sub_query_id: sqr for sqr in sub_query_results}

        for order_group in query_plan.execution_order:
            for sq_id in order_group:
                if sq_id in id_to_result:
                    sqr = id_to_result[sq_id]
                    for result in sqr.results:
                        result["reasoning_step"] = sq_id
                        ordered_results.append(result)

        # Score by position in reasoning chain
        scores = [1.0 - (i * 0.1) for i in range(len(ordered_results))]

        return ordered_results[:top_k], scores[:top_k]

    def _calculate_confidence(
        self,
        sub_query_results: List[SubQueryResult],
        fused_scores: List[float]
    ) -> float:
        """Calculate overall confidence in fused results."""
        if not fused_scores:
            return 0.0

        # Base confidence on average fused score
        avg_score = sum(fused_scores) / len(fused_scores)

        # Penalty for errors
        error_count = sum(1 for r in sub_query_results if r.error)
        error_penalty = error_count * 0.1

        # Bonus for multiple sources agreeing
        sources = set(r.source for r in sub_query_results if not r.error)
        source_bonus = 0.1 if len(sources) > 1 else 0.0

        confidence = min(1.0, max(0.0, avg_score - error_penalty + source_bonus))
        return confidence


class QueryPipeline:
    """
    End-to-end query processing pipeline.

    Combines decomposition, planning, and execution.
    """

    def __init__(
        self,
        decomposer: Optional[Any] = None,
        planner: Optional[QueryPlanner] = None,
        executor: Optional[QueryExecutor] = None
    ):
        from .decomposer import QueryDecomposer

        self.decomposer = decomposer or QueryDecomposer()
        self.planner = planner or QueryPlanner()
        self.executor = executor or QueryExecutor()

    async def process(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> FusedResult:
        """
        Process a query through the full pipeline.

        Args:
            query: User query
            top_k: Number of results
            context: Optional context (entity graph, time ranges, etc.)

        Returns:
            FusedResult with ranked results
        """
        # Step 1: Decompose query
        logger.info(f"Decomposing query: {query[:50]}...")
        query_plan = await self.decomposer.decompose(query, context)
        logger.info(
            f"Decomposed into {len(query_plan.sub_queries)} sub-queries "
            f"(type: {query_plan.query_type.value})"
        )

        # Step 2: Create execution plan
        logger.info("Creating execution plan...")
        execution_plan = self.planner.create_plan(query_plan)
        logger.info(
            f"Plan: {len(execution_plan.stages)} stages, "
            f"optimizations: {execution_plan.optimization_applied}"
        )

        # Step 3: Execute plan
        logger.info("Executing plan...")
        result = await self.executor.execute(execution_plan, top_k)
        logger.info(
            f"Execution complete: {len(result.results)} results in {result.total_time_ms:.1f}ms"
        )

        return result

    def process_sync(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> FusedResult:
        """Synchronous version of process."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.process(query, top_k, context))

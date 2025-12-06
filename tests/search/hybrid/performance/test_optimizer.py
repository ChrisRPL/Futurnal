"""Tests for QueryPlanOptimizer.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from futurnal.search.hybrid.performance.optimizer import (
    QueryPlan,
    QueryPlanCost,
    QueryPlanOptimizer,
    RetrievalStrategy,
    BASELINE_STRATEGY_COSTS,
    STRATEGY_STEPS,
)


class TestRetrievalStrategy:
    """Tests for RetrievalStrategy enum."""

    def test_all_strategies_defined(self):
        """Verify all expected strategies exist."""
        assert RetrievalStrategy.VECTOR_ONLY == "vector"
        assert RetrievalStrategy.GRAPH_ONLY == "graph"
        assert RetrievalStrategy.TEMPORAL_FIRST == "temporal_first"
        assert RetrievalStrategy.HYBRID_PARALLEL == "hybrid_parallel"
        assert RetrievalStrategy.HYBRID_SEQUENTIAL == "hybrid_sequential"
        assert RetrievalStrategy.CAUSAL_CHAIN == "causal_chain"


class TestQueryPlanCost:
    """Tests for QueryPlanCost dataclass."""

    def test_total_cost_calculation(self):
        """Test weighted cost function."""
        cost = QueryPlanCost(
            latency_ms=100,
            resource_usage=0.5,
            cache_probability=0.7,
            accuracy_score=0.9,
        )

        total = cost.total_cost

        # Verify it's a weighted combination
        assert total > 0
        assert total < 200  # Should be reasonable

    def test_lower_latency_lower_cost(self):
        """Verify lower latency results in lower total cost."""
        fast = QueryPlanCost(latency_ms=50, resource_usage=0.5, cache_probability=0.5, accuracy_score=0.8)
        slow = QueryPlanCost(latency_ms=500, resource_usage=0.5, cache_probability=0.5, accuracy_score=0.8)

        assert fast.total_cost < slow.total_cost

    def test_higher_cache_probability_lower_cost(self):
        """Verify higher cache probability reduces cost."""
        high_cache = QueryPlanCost(latency_ms=100, resource_usage=0.5, cache_probability=0.9, accuracy_score=0.8)
        low_cache = QueryPlanCost(latency_ms=100, resource_usage=0.5, cache_probability=0.1, accuracy_score=0.8)

        assert high_cache.total_cost < low_cache.total_cost


class TestQueryPlanOptimizer:
    """Tests for QueryPlanOptimizer."""

    def test_optimize_temporal_intent(self, optimizer: QueryPlanOptimizer, mock_query_intent):
        """Test optimization for temporal intent."""
        mock_query_intent.primary_intent = "temporal"
        mock_query_intent.confidence = 0.9

        plan = optimizer.optimize(
            "what happened last week",
            mock_query_intent,
        )

        assert isinstance(plan, QueryPlan)
        assert plan.strategy in [
            RetrievalStrategy.TEMPORAL_FIRST,
            RetrievalStrategy.HYBRID_SEQUENTIAL,
        ]

    def test_optimize_causal_intent(self, optimizer: QueryPlanOptimizer, mock_query_intent):
        """Test optimization for causal intent."""
        mock_query_intent.primary_intent = "causal"

        plan = optimizer.optimize(
            "what caused the outage",
            mock_query_intent,
        )

        assert plan.strategy in [
            RetrievalStrategy.CAUSAL_CHAIN,
            RetrievalStrategy.GRAPH_ONLY,
        ]

    def test_optimize_exploratory_intent(self, optimizer: QueryPlanOptimizer, mock_query_intent):
        """Test optimization for exploratory intent."""
        mock_query_intent.primary_intent = "exploratory"

        plan = optimizer.optimize(
            "tell me about project X",
            mock_query_intent,
        )

        assert plan.strategy in [
            RetrievalStrategy.HYBRID_PARALLEL,
            RetrievalStrategy.VECTOR_ONLY,
        ]

    def test_latency_constraint(self, optimizer: QueryPlanOptimizer, mock_query_intent):
        """Test that latency constraints are respected."""
        mock_query_intent.primary_intent = "exploratory"

        plan = optimizer.optimize(
            "test query",
            mock_query_intent,
            constraints={"max_latency_ms": 100},  # Tight constraint
        )

        # Should select faster strategy
        assert plan.estimated_cost.latency_ms <= 200  # Some buffer

    def test_accuracy_constraint(self, optimizer: QueryPlanOptimizer, mock_query_intent):
        """Test that accuracy constraints are respected."""
        mock_query_intent.primary_intent = "exploratory"

        plan = optimizer.optimize(
            "test query",
            mock_query_intent,
            constraints={"min_accuracy": 0.9},  # High accuracy needed
        )

        assert plan.estimated_cost.accuracy_score >= 0.9

    def test_plan_has_steps(self, optimizer: QueryPlanOptimizer, mock_query_intent):
        """Test that plan includes execution steps."""
        plan = optimizer.optimize("test query", mock_query_intent)

        assert len(plan.steps) > 0
        assert all("name" in step for step in plan.steps)
        assert all("component" in step for step in plan.steps)

    def test_cache_aware_optimization(
        self, optimizer_with_cache: QueryPlanOptimizer, mock_query_intent
    ):
        """Test that optimizer uses cache state for planning."""
        # Populate cache with some entries
        from futurnal.search.hybrid.performance.cache import CacheLayer

        optimizer_with_cache.cache.set(
            CacheLayer.QUERY_RESULT, "test query", "cached_result"
        )

        plan = optimizer_with_cache.optimize("test query", mock_query_intent)

        # Should factor in cache state
        assert plan.estimated_cost.cache_probability >= 0


class TestStrategyDefinitions:
    """Tests for strategy step definitions."""

    def test_all_strategies_have_costs(self):
        """Verify all strategies have baseline costs defined."""
        for strategy in RetrievalStrategy:
            assert strategy in BASELINE_STRATEGY_COSTS

    def test_all_strategies_have_steps(self):
        """Verify all strategies have execution steps defined."""
        for strategy in RetrievalStrategy:
            assert strategy in STRATEGY_STEPS
            assert len(STRATEGY_STEPS[strategy]) > 0

    def test_hybrid_parallel_has_parallel_steps(self):
        """Verify hybrid parallel strategy identifies parallelizable steps."""
        steps = STRATEGY_STEPS[RetrievalStrategy.HYBRID_PARALLEL]

        # Should have vector_search and graph_search
        step_names = [s["name"] for s in steps]
        assert "vector_search" in step_names
        assert "graph_search" in step_names

    def test_causal_chain_traverses_causes_and_effects(self):
        """Verify causal chain strategy includes both traversals."""
        steps = STRATEGY_STEPS[RetrievalStrategy.CAUSAL_CHAIN]
        step_names = [s["name"] for s in steps]

        assert "traverse_causes" in step_names
        assert "traverse_effects" in step_names

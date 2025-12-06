"""Tests for Query Router Orchestrator.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Success Metrics:
- Query routing latency <100ms
- Multi-strategy composition functional
- Result ranking quality high
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from futurnal.search.hybrid.types import QueryIntent, QueryPlan, QueryResult
from futurnal.search.hybrid.routing.orchestrator import QueryRouter
from futurnal.search.hybrid.routing.feedback import SearchQualityFeedback
from futurnal.search.hybrid.routing.templates import QueryTemplateDatabase


class TestQueryRouter:
    """Tests for QueryRouter."""

    def test_initialization_minimal(self):
        """Router initializes with minimal configuration."""
        router = QueryRouter()

        assert router.temporal is None
        assert router.causal is None
        assert router.schema is None
        assert router.classifier is not None

    def test_initialization_full(
        self,
        mock_temporal_engine,
        mock_causal_retrieval,
        mock_schema_retrieval,
        mock_intent_classifier,
        mock_grpo_engine,
    ):
        """Router initializes with all components."""
        feedback = SearchQualityFeedback(grpo_engine=mock_grpo_engine)
        template_db = QueryTemplateDatabase()

        router = QueryRouter(
            temporal_engine=mock_temporal_engine,
            causal_retrieval=mock_causal_retrieval,
            schema_retrieval=mock_schema_retrieval,
            intent_classifier=mock_intent_classifier,
            grpo_feedback=feedback,
            template_db=template_db,
        )

        assert router.temporal is mock_temporal_engine
        assert router.causal is mock_causal_retrieval
        assert router.schema is mock_schema_retrieval
        assert router.classifier is mock_intent_classifier
        assert router.grpo_feedback is feedback
        assert router.template_db is template_db

    def test_strategy_configs_defined(self):
        """Strategy configs exist for all intents."""
        configs = QueryRouter.STRATEGY_CONFIGS

        assert QueryIntent.TEMPORAL in configs
        assert QueryIntent.CAUSAL in configs
        assert QueryIntent.LOOKUP in configs
        assert QueryIntent.EXPLORATORY in configs

        for intent, config in configs.items():
            assert "primary_strategy" in config
            assert "weights" in config
            assert "estimated_latency_ms" in config

    def test_temporal_strategy_config(self):
        """Temporal strategy has correct configuration."""
        config = QueryRouter.STRATEGY_CONFIGS[QueryIntent.TEMPORAL]

        assert config["primary_strategy"] == "temporal_query"
        assert config["secondary_strategy"] == "hybrid_retrieval"
        assert config["weights"]["temporal"] > config["weights"]["hybrid"]

    def test_causal_strategy_config(self):
        """Causal strategy has correct configuration."""
        config = QueryRouter.STRATEGY_CONFIGS[QueryIntent.CAUSAL]

        assert config["primary_strategy"] == "causal_chain"
        assert config["secondary_strategy"] == "temporal_query"
        assert config["weights"]["causal"] > config["weights"]["temporal"]

    def test_lookup_strategy_config(self):
        """Lookup strategy has correct configuration."""
        config = QueryRouter.STRATEGY_CONFIGS[QueryIntent.LOOKUP]

        assert config["primary_strategy"] == "hybrid_retrieval"
        assert config["secondary_strategy"] is None
        assert config["weights"]["hybrid"] == 1.0


class TestQueryRouting:
    """Tests for query routing logic."""

    def test_route_query_returns_plan(self, mock_intent_classifier):
        """route_query returns a QueryPlan."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        plan = router.route_query("What happened yesterday?")

        assert isinstance(plan, QueryPlan)
        assert plan.query_id is not None
        assert plan.original_query == "What happened yesterday?"
        assert plan.intent == QueryIntent.TEMPORAL
        assert plan.intent_confidence == 0.9

    def test_route_query_temporal_strategy(self, mock_intent_classifier):
        """Temporal intent routes to temporal strategy."""
        mock_intent_classifier.classify_intent.return_value = {
            "intent": "temporal",
            "confidence": 0.9,
            "reasoning": "time keywords",
        }

        router = QueryRouter(intent_classifier=mock_intent_classifier)
        plan = router.route_query("When was the meeting?")

        assert plan.intent == QueryIntent.TEMPORAL
        assert plan.primary_strategy == "temporal_query"
        assert "temporal" in plan.weights

    def test_route_query_causal_strategy(self, mock_intent_classifier):
        """Causal intent routes to causal strategy."""
        mock_intent_classifier.classify_intent.return_value = {
            "intent": "causal",
            "confidence": 0.85,
            "reasoning": "why keyword",
        }

        router = QueryRouter(intent_classifier=mock_intent_classifier)
        plan = router.route_query("Why did the project fail?")

        assert plan.intent == QueryIntent.CAUSAL
        assert plan.primary_strategy == "causal_chain"
        assert "causal" in plan.weights

    def test_route_query_lookup_strategy(self, mock_intent_classifier):
        """Lookup intent routes to hybrid strategy."""
        mock_intent_classifier.classify_intent.return_value = {
            "intent": "lookup",
            "confidence": 0.9,
            "reasoning": "what is keyword",
        }

        router = QueryRouter(intent_classifier=mock_intent_classifier)
        plan = router.route_query("What is machine learning?")

        assert plan.intent == QueryIntent.LOOKUP
        assert plan.primary_strategy == "hybrid_retrieval"
        assert plan.secondary_strategy is None

    def test_route_query_records_feedback(self, mock_intent_classifier, mock_grpo_engine):
        """Routing records query for GRPO feedback."""
        feedback = SearchQualityFeedback(grpo_engine=mock_grpo_engine)
        router = QueryRouter(
            intent_classifier=mock_intent_classifier,
            grpo_feedback=feedback,
        )

        plan = router.route_query("What happened?")

        assert len(feedback.query_history) == 1
        assert feedback.query_history[0].query_id == plan.query_id


class TestQueryExecution:
    """Tests for query plan execution."""

    def test_execute_plan_returns_result(
        self,
        mock_intent_classifier,
        mock_schema_retrieval,
    ):
        """execute_plan returns a QueryResult."""
        router = QueryRouter(
            intent_classifier=mock_intent_classifier,
            schema_retrieval=mock_schema_retrieval,
        )

        plan = router.route_query("What is the status?")
        result = router.execute_plan(plan)

        assert isinstance(result, QueryResult)
        assert result.query_id == plan.query_id
        assert result.latency_ms >= 0

    def test_execute_plan_calls_primary_strategy(
        self,
        mock_intent_classifier,
        mock_schema_retrieval,
    ):
        """Execution calls primary strategy."""
        router = QueryRouter(
            intent_classifier=mock_intent_classifier,
            schema_retrieval=mock_schema_retrieval,
        )

        plan = router.route_query("What is the status?")
        router.execute_plan(plan)

        mock_schema_retrieval.hybrid_search.assert_called()

    def test_execute_plan_handles_missing_engine(self, mock_intent_classifier):
        """Execution handles missing temporal engine gracefully."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        mock_intent_classifier.classify_intent.return_value = {
            "intent": "temporal",
            "confidence": 0.9,
            "reasoning": "time",
        }

        plan = router.route_query("When was it?")
        result = router.execute_plan(plan)

        # Should not crash, just return empty results
        assert result.entities == []

    def test_route_and_execute_convenience(
        self,
        mock_intent_classifier,
        mock_schema_retrieval,
    ):
        """route_and_execute combines routing and execution."""
        router = QueryRouter(
            intent_classifier=mock_intent_classifier,
            schema_retrieval=mock_schema_retrieval,
        )

        result = router.route_and_execute("What is the status?")

        assert isinstance(result, QueryResult)
        assert len(result.entities) > 0


class TestResultFusion:
    """Tests for result fusion logic."""

    def test_fuse_results_deduplicates(self, mock_intent_classifier):
        """Fusion deduplicates by entity ID."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        results = [
            {"id": "entity-1", "type": "Event", "score": 0.9, "source_strategy": "temporal_query"},
            {"id": "entity-1", "type": "Event", "score": 0.8, "source_strategy": "hybrid_retrieval"},
            {"id": "entity-2", "type": "Event", "score": 0.7, "source_strategy": "temporal_query"},
        ]

        fused = router._fuse_results(
            results,
            {"temporal": 0.7, "hybrid": 0.3},
            QueryIntent.TEMPORAL,
        )

        # Should have 2 unique entities
        assert len(fused["entities"]) == 2

    def test_fuse_results_applies_weights(self, mock_intent_classifier):
        """Fusion applies strategy weights to scores."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        results = [
            {"id": "entity-1", "score": 1.0, "source_strategy": "temporal_query"},
        ]

        fused = router._fuse_results(
            results,
            {"temporal": 0.7},
            QueryIntent.TEMPORAL,
        )

        # Weighted score should be 1.0 * 0.7 = 0.7
        assert fused["entities"][0]["weighted_score"] == 0.7

    def test_fuse_results_sorts_by_score(self, mock_intent_classifier):
        """Fusion sorts results by weighted score."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        results = [
            {"id": "entity-1", "score": 0.5, "source_strategy": "hybrid_retrieval"},
            {"id": "entity-2", "score": 0.9, "source_strategy": "hybrid_retrieval"},
            {"id": "entity-3", "score": 0.7, "source_strategy": "hybrid_retrieval"},
        ]

        fused = router._fuse_results(
            results,
            {"hybrid": 1.0},
            QueryIntent.LOOKUP,
        )

        scores = [e["weighted_score"] for e in fused["entities"]]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_results_temporal_context(self, mock_intent_classifier):
        """Temporal queries include temporal context."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        fused = router._fuse_results(
            [],
            {"temporal": 0.7},
            QueryIntent.TEMPORAL,
        )

        assert fused["temporal_context"] is not None
        assert fused["temporal_context"]["query_type"] == "temporal"

    def test_fuse_results_causal_chain(self, mock_intent_classifier):
        """Causal queries include causal chain."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        results = [
            {"id": "entity-1", "score": 0.9, "source_strategy": "causal_chain"},
        ]

        fused = router._fuse_results(
            results,
            {"causal": 0.6},
            QueryIntent.CAUSAL,
        )

        assert fused["causal_chain"] is not None


class TestStrategyConfiguration:
    """Tests for strategy configuration updates."""

    def test_update_strategy_config(self, mock_intent_classifier):
        """Strategy config can be updated at runtime."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        original = router.get_strategy_config(QueryIntent.TEMPORAL)

        router.update_strategy_config(
            QueryIntent.TEMPORAL,
            {"weights": {"temporal": 0.8, "hybrid": 0.2}},
        )

        updated = router.get_strategy_config(QueryIntent.TEMPORAL)
        assert updated["weights"]["temporal"] == 0.8

    def test_get_strategy_config(self, mock_intent_classifier):
        """Strategy config retrieval works."""
        router = QueryRouter(intent_classifier=mock_intent_classifier)

        config = router.get_strategy_config(QueryIntent.LOOKUP)

        assert "primary_strategy" in config
        assert "weights" in config

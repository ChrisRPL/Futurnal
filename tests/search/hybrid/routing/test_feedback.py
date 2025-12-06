"""Tests for Search Quality Feedback and GRPO Integration.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Success Metrics:
- GRPO feedback loop operational
- Signal collection and aggregation
- Advantage extraction for experiential learning
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from futurnal.search.hybrid.types import QueryIntent, QueryPlan
from futurnal.search.hybrid.routing.feedback import (
    SearchQualityFeedback,
    SearchQualitySignal,
)


class TestSearchQualitySignal:
    """Tests for SearchQualitySignal model."""

    def test_signal_creation(self):
        """Signal creates with required fields."""
        signal = SearchQualitySignal(
            query_id="query-123",
            signal_type="click",
            signal_value=1.0,
        )

        assert signal.query_id == "query-123"
        assert signal.signal_type == "click"
        assert signal.signal_value == 1.0
        assert signal.timestamp is not None
        assert signal.context == {}

    def test_signal_with_context(self):
        """Signal accepts context dictionary."""
        signal = SearchQualitySignal(
            query_id="query-123",
            signal_type="click",
            signal_value=1.0,
            context={"entity_id": "entity-1", "position": 0},
        )

        assert signal.context["entity_id"] == "entity-1"
        assert signal.context["position"] == 0

    def test_signal_value_bounds(self):
        """Signal value is bounded -1 to 1."""
        # Valid values
        signal_pos = SearchQualitySignal(
            query_id="q", signal_type="click", signal_value=1.0
        )
        signal_neg = SearchQualitySignal(
            query_id="q", signal_type="no_results", signal_value=-1.0
        )

        assert signal_pos.signal_value == 1.0
        assert signal_neg.signal_value == -1.0


class TestSearchQualityFeedback:
    """Tests for SearchQualityFeedback."""

    def test_initialization_with_grpo(self, mock_grpo_engine):
        """Feedback initializes with GRPO engine."""
        feedback = SearchQualityFeedback(grpo_engine=mock_grpo_engine)

        assert feedback.grpo is mock_grpo_engine
        assert feedback.query_history == []
        assert feedback.signals == []

    def test_initialization_without_grpo(self):
        """Feedback initializes without GRPO engine."""
        feedback = SearchQualityFeedback()

        assert feedback.grpo is None
        assert feedback.query_history == []
        assert feedback.signals == []

    def test_record_query(self, sample_query_plan):
        """Query recording works."""
        feedback = SearchQualityFeedback()

        feedback.record_query(sample_query_plan)

        assert len(feedback.query_history) == 1
        assert feedback.query_history[0].query_id == sample_query_plan.query_id

    def test_record_query_limits_history(self, sample_query_plan):
        """Query history is limited to MAX_QUERY_HISTORY."""
        feedback = SearchQualityFeedback()
        feedback.MAX_QUERY_HISTORY = 5

        for i in range(10):
            plan = QueryPlan(
                query_id=f"query-{i}",
                original_query=f"Query {i}",
                intent=QueryIntent.LOOKUP,
                intent_confidence=0.9,
                primary_strategy="hybrid_retrieval",
                weights={"hybrid": 1.0},
            )
            feedback.record_query(plan)

        assert len(feedback.query_history) == 5

    def test_record_signal(self):
        """Signal recording works."""
        feedback = SearchQualityFeedback()

        feedback.record_signal(
            query_id="query-123",
            signal_type="click",
            signal_value=1.0,
            context={"entity_id": "entity-1"},
        )

        assert len(feedback.signals) == 1
        assert feedback.signals[0].query_id == "query-123"
        assert feedback.signals[0].signal_type == "click"

    def test_record_signal_clamps_value(self):
        """Signal value is clamped to -1, 1 range."""
        feedback = SearchQualityFeedback()

        feedback.record_signal("q", "click", 5.0)  # Too high
        feedback.record_signal("q", "no_results", -5.0)  # Too low

        assert feedback.signals[0].signal_value == 1.0
        assert feedback.signals[1].signal_value == -1.0

    def test_record_signal_triggers_grpo(self, mock_grpo_engine, sample_query_plan):
        """Recording signals triggers GRPO update at threshold."""
        feedback = SearchQualityFeedback(grpo_engine=mock_grpo_engine)
        feedback.GRPO_TRIGGER_THRESHOLD = 5

        feedback.record_query(sample_query_plan)

        for i in range(5):
            feedback.record_signal(
                query_id=sample_query_plan.query_id,
                signal_type="click",
                signal_value=1.0,
            )

        # GRPO should have been triggered
        mock_grpo_engine.update_experiential_knowledge.assert_called()

    def test_aggregate_signals(self, sample_query_plan):
        """Signal aggregation groups by query ID."""
        feedback = SearchQualityFeedback()
        feedback.record_query(sample_query_plan)

        # Add signals for different queries
        feedback.record_signal("query-1", "click", 1.0)
        feedback.record_signal("query-1", "click", 1.0)
        feedback.record_signal("query-2", "refinement", -0.5)

        grouped = feedback._aggregate_signals()

        assert "query-1" in grouped
        assert "query-2" in grouped
        assert len(grouped["query-1"]) == 2
        assert len(grouped["query-2"]) == 1

    def test_extract_advantages_clicks(self, sample_query_plan):
        """Advantages extracted from click signals."""
        feedback = SearchQualityFeedback()
        feedback.record_query(sample_query_plan)

        signals = {
            sample_query_plan.query_id: [
                SearchQualitySignal(
                    query_id=sample_query_plan.query_id,
                    signal_type="click",
                    signal_value=1.0,
                ),
                SearchQualitySignal(
                    query_id=sample_query_plan.query_id,
                    signal_type="click",
                    signal_value=1.0,
                ),
            ]
        }

        advantages = feedback._extract_advantages(signals)

        assert len(advantages) >= 1
        assert "better_approach" in advantages[0]
        assert "temporal" in advantages[0]["better_approach"].lower()

    def test_extract_advantages_refinement(self, sample_query_plan):
        """Advantages extracted from refinement signals."""
        feedback = SearchQualityFeedback()
        feedback.record_query(sample_query_plan)

        signals = {
            sample_query_plan.query_id: [
                SearchQualitySignal(
                    query_id=sample_query_plan.query_id,
                    signal_type="refinement",
                    signal_value=-0.5,
                ),
            ]
        }

        advantages = feedback._extract_advantages(signals)

        assert len(advantages) >= 1
        assert "refinement" in advantages[0]["reasoning"].lower()


class TestQualityMetrics:
    """Tests for quality metrics computation."""

    def test_get_quality_metrics_empty(self):
        """Metrics for empty signals."""
        feedback = SearchQualityFeedback()

        metrics = feedback.get_quality_metrics()

        assert metrics["insufficient_data"] is True
        assert metrics["signal_count"] == 0

    def test_get_quality_metrics_with_data(self):
        """Metrics computed correctly."""
        feedback = SearchQualityFeedback()

        # 8 clicks, 2 refinements
        for i in range(8):
            feedback.record_signal(f"q-{i}", "click", 1.0)
        for i in range(2):
            feedback.record_signal(f"q-{i+8}", "refinement", -0.5)

        metrics = feedback.get_quality_metrics()

        assert metrics["click_rate"] == pytest.approx(0.8)
        assert metrics["refinement_rate"] == pytest.approx(0.2)
        assert metrics["satisfaction_trend"] == pytest.approx(0.6)  # 0.8 - 0.2
        assert metrics["signal_count"] == 10

    def test_get_intent_performance(self, sample_query_plan):
        """Intent performance metrics work."""
        feedback = SearchQualityFeedback()
        feedback.record_query(sample_query_plan)

        feedback.record_signal(sample_query_plan.query_id, "click", 1.0)
        feedback.record_signal(sample_query_plan.query_id, "click", 1.0)

        perf = feedback.get_intent_performance()

        assert "temporal" in perf
        assert perf["temporal"]["click_rate"] == 1.0
        assert perf["temporal"]["signal_count"] == 2


class TestFeedbackManagement:
    """Tests for feedback management operations."""

    def test_clear_history(self, sample_query_plan):
        """Clear history removes all data."""
        feedback = SearchQualityFeedback()
        feedback.record_query(sample_query_plan)
        feedback.record_signal("q", "click", 1.0)

        feedback.clear_history()

        assert len(feedback.query_history) == 0
        assert len(feedback.signals) == 0

    def test_export_data(self, sample_query_plan):
        """Export data returns complete state."""
        feedback = SearchQualityFeedback()
        feedback.record_query(sample_query_plan)
        feedback.record_signal(sample_query_plan.query_id, "click", 1.0)

        data = feedback.export_data()

        assert "queries" in data
        assert "signals" in data
        assert "metrics" in data
        assert "intent_performance" in data

        assert len(data["queries"]) == 1
        assert len(data["signals"]) == 1

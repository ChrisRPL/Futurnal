"""Tests for causal search result models.

Tests result models and computed fields.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md
"""

from datetime import datetime, timedelta

import pytest

from futurnal.pkg.schema.models import EventNode
from futurnal.search.causal.results import (
    CausalCauseResult,
    CausalEffectResult,
    CausalPathResult,
    CausalSearchPath,
    CorrelationPatternResult,
    FindCausesResult,
    FindEffectsResult,
)
from tests.search.conftest import create_test_event


class TestCausalSearchPath:
    """Test CausalSearchPath model."""

    def test_basic_creation(self):
        """Test basic path creation."""
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="p1",
            path=["m1", "d1", "p1"],
            causal_confidence=0.72,
            temporal_ordering_valid=True,
        )
        assert path.start_event_id == "m1"
        assert path.end_event_id == "p1"
        assert len(path.path) == 3
        assert path.causal_confidence == 0.72
        assert path.temporal_ordering_valid is True

    def test_path_length_computed(self):
        """Test path_length is computed correctly."""
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="p1",
            path=["m1", "d1", "p1"],
            causal_confidence=0.72,
            temporal_ordering_valid=True,
        )
        assert path.path_length == 2  # 3 nodes = 2 hops

    def test_path_length_single_hop(self):
        """Test path_length for single hop."""
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="d1",
            path=["m1", "d1"],
            causal_confidence=0.9,
            temporal_ordering_valid=True,
        )
        assert path.path_length == 1

    def test_temporal_span_with_events(self):
        """Test temporal_span computed with events."""
        events = [
            create_test_event("m1", "Meeting", "meeting", datetime(2024, 1, 1, 10, 0)),
            create_test_event("d1", "Decision", "decision", datetime(2024, 1, 5, 14, 0)),
            create_test_event("p1", "Publication", "publication", datetime(2024, 1, 15, 9, 0)),
        ]
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="p1",
            path=["m1", "d1", "p1"],
            events=events,
            causal_confidence=0.72,
            temporal_ordering_valid=True,
        )
        # 14 days, 23 hours
        expected_span = events[2].timestamp - events[0].timestamp
        assert path.temporal_span == expected_span

    def test_temporal_span_without_events(self):
        """Test temporal_span is None without events."""
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="p1",
            path=["m1", "d1", "p1"],
            causal_confidence=0.72,
            temporal_ordering_valid=True,
        )
        assert path.temporal_span is None

    def test_confidence_scores(self):
        """Test confidence_scores field."""
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="p1",
            path=["m1", "d1", "p1"],
            causal_confidence=0.765,  # min of [0.9, 0.85]
            confidence_scores=[0.9, 0.85],
            temporal_ordering_valid=True,
        )
        assert path.confidence_scores == [0.9, 0.85]


class TestCausalCauseResult:
    """Test CausalCauseResult model."""

    def test_basic_creation(self):
        """Test basic cause result creation."""
        result = CausalCauseResult(
            cause_id="m1",
            cause_name="Meeting",
            cause_timestamp=datetime(2024, 1, 1, 10, 0),
            distance=1,
            confidence_scores=[0.9],
            aggregate_confidence=0.9,
            temporal_ordering_valid=True,
        )
        assert result.cause_id == "m1"
        assert result.cause_name == "Meeting"
        assert result.distance == 1
        assert result.aggregate_confidence == 0.9
        assert result.temporal_ordering_valid is True


class TestFindCausesResult:
    """Test FindCausesResult model."""

    def test_total_causes_computed(self):
        """Test total_causes is computed correctly."""
        causes = [
            CausalCauseResult(
                cause_id=f"c{i}",
                cause_name=f"Cause {i}",
                cause_timestamp=datetime(2024, 1, i, 10, 0),
                distance=i,
                aggregate_confidence=0.8,
                temporal_ordering_valid=True,
            )
            for i in range(1, 4)
        ]
        result = FindCausesResult(
            target_event_id="target",
            causes=causes,
            max_hops_requested=3,
            min_confidence_requested=0.6,
        )
        assert result.total_causes == 3

    def test_unique_root_causes_computed(self):
        """Test unique_root_causes identifies max distance causes."""
        causes = [
            CausalCauseResult(
                cause_id="c1",
                cause_name="Near Cause",
                cause_timestamp=datetime(2024, 1, 3, 10, 0),
                distance=1,
                aggregate_confidence=0.8,
                temporal_ordering_valid=True,
            ),
            CausalCauseResult(
                cause_id="c2",
                cause_name="Root Cause A",
                cause_timestamp=datetime(2024, 1, 1, 10, 0),
                distance=3,
                aggregate_confidence=0.7,
                temporal_ordering_valid=True,
            ),
            CausalCauseResult(
                cause_id="c3",
                cause_name="Root Cause B",
                cause_timestamp=datetime(2024, 1, 2, 10, 0),
                distance=3,
                aggregate_confidence=0.75,
                temporal_ordering_valid=True,
            ),
        ]
        result = FindCausesResult(
            target_event_id="target",
            causes=causes,
            max_hops_requested=3,
            min_confidence_requested=0.6,
        )
        assert set(result.unique_root_causes) == {"c2", "c3"}

    def test_empty_causes(self):
        """Test result with no causes."""
        result = FindCausesResult(
            target_event_id="target",
            causes=[],
            max_hops_requested=3,
            min_confidence_requested=0.6,
        )
        assert result.total_causes == 0
        assert result.unique_root_causes == []


class TestFindEffectsResult:
    """Test FindEffectsResult model."""

    def test_total_effects_computed(self):
        """Test total_effects is computed correctly."""
        effects = [
            CausalEffectResult(
                effect_id=f"e{i}",
                effect_name=f"Effect {i}",
                effect_timestamp=datetime(2024, 1, i + 5, 10, 0),
                distance=i,
                aggregate_confidence=0.8,
                temporal_ordering_valid=True,
            )
            for i in range(1, 4)
        ]
        result = FindEffectsResult(
            source_event_id="source",
            effects=effects,
            max_hops_requested=3,
            min_confidence_requested=0.6,
        )
        assert result.total_effects == 3


class TestCausalPathResult:
    """Test CausalPathResult model."""

    def test_path_found_true(self):
        """Test result when path is found."""
        path = CausalSearchPath(
            start_event_id="m1",
            end_event_id="p1",
            path=["m1", "d1", "p1"],
            causal_confidence=0.72,
            temporal_ordering_valid=True,
        )
        result = CausalPathResult(
            path_found=True,
            path=path,
            start_event_id="m1",
            end_event_id="p1",
            max_hops_requested=5,
        )
        assert result.path_found is True
        assert result.path is not None

    def test_path_not_found(self):
        """Test result when no path found."""
        result = CausalPathResult(
            path_found=False,
            path=None,
            start_event_id="m1",
            end_event_id="x1",
            max_hops_requested=5,
        )
        assert result.path_found is False
        assert result.path is None


class TestCorrelationPatternResult:
    """Test CorrelationPatternResult model."""

    def test_basic_creation(self):
        """Test basic creation."""
        result = CorrelationPatternResult(
            patterns_found=3,
            correlations=[{"a": 1}, {"b": 2}, {"c": 3}],
            causal_candidates=[{"a": 1}],
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 6, 30),
            min_correlation_strength=0.5,
        )
        assert result.patterns_found == 3
        assert len(result.correlations) == 3
        assert result.causal_candidate_count == 1

    def test_time_span_computed(self):
        """Test time_span is computed correctly."""
        result = CorrelationPatternResult(
            patterns_found=0,
            correlations=[],
            causal_candidates=[],
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 6, 30),
            min_correlation_strength=0.5,
        )
        expected_span = datetime(2024, 6, 30) - datetime(2024, 1, 1)
        assert result.time_span == expected_span

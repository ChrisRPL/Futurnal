"""Tests for DOTS Causal Ordering module.

Phase 2.5: Research Integration Sprint

Tests verify:
1. Correct temporal precedence computation
2. Topological sorting of causal order
3. Correlation filtering by causal direction
4. Option B compliance (algorithmic, no model updates)

Research basis: DOTS (arxiv:2510.24639) achieves F1 0.81 vs 0.63 baseline.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List

from futurnal.search.temporal.dots_ordering import (
    DOTSCausalOrdering,
    CausalOrder,
)


class TestCausalOrder:
    """Tests for CausalOrder dataclass."""

    def test_empty_order(self):
        """Test empty causal order."""
        order = CausalOrder()
        assert order.ordered_events == []
        assert order.confidence == 0.0
        assert order.total_observations == 0

    def test_get_causal_direction_known_events(self):
        """Test getting causal direction for known events."""
        order = CausalOrder(
            ordered_events=["meeting", "decision", "action"],
            precedence_matrix={
                ("meeting", "decision"): 0.8,
                ("decision", "action"): 0.9,
            },
            confidence=0.85,
        )

        cause, effect, conf = order.get_causal_direction("meeting", "decision")
        assert cause == "meeting"
        assert effect == "decision"
        assert conf == 0.8

    def test_get_causal_direction_reverse_order(self):
        """Test causal direction when second event is the cause."""
        order = CausalOrder(
            ordered_events=["meeting", "decision"],
            precedence_matrix={
                ("meeting", "decision"): 0.8,
            },
        )

        # Ask for decision -> meeting, should return meeting -> decision
        cause, effect, conf = order.get_causal_direction("decision", "meeting")
        assert cause == "meeting"
        assert effect == "decision"

    def test_get_causal_direction_unknown_event(self):
        """Test causal direction with unknown event."""
        order = CausalOrder(
            ordered_events=["meeting", "decision"],
        )

        cause, effect, conf = order.get_causal_direction("meeting", "unknown")
        # Unknown event returns default 0.5 confidence
        assert conf == 0.5

    def test_is_valid_causal_path_valid(self):
        """Test valid causal path."""
        order = CausalOrder(
            ordered_events=["meeting", "decision", "action"],
            precedence_matrix={
                ("meeting", "decision"): 0.8,
                ("decision", "action"): 0.9,
            },
        )

        assert order.is_valid_causal_path(["meeting", "decision", "action"]) is True

    def test_is_valid_causal_path_invalid(self):
        """Test invalid causal path (reversed order)."""
        order = CausalOrder(
            ordered_events=["meeting", "decision", "action"],
            precedence_matrix={
                ("meeting", "decision"): 0.8,
            },
        )

        # action -> meeting is invalid (wrong direction)
        assert order.is_valid_causal_path(["action", "meeting"]) is False

    def test_get_position(self):
        """Test getting position in causal order."""
        order = CausalOrder(
            ordered_events=["meeting", "decision", "action"],
        )

        assert order.get_position("meeting") == 0
        assert order.get_position("decision") == 1
        assert order.get_position("action") == 2
        assert order.get_position("unknown") is None

    def test_to_natural_language(self):
        """Test natural language export."""
        order = CausalOrder(
            ordered_events=["meeting", "decision"],
            confidence=0.85,
            total_observations=100,
        )

        nl = order.to_natural_language()
        assert "meeting" in nl
        assert "decision" in nl
        assert "85%" in nl
        assert "100" in nl

    def test_to_dict(self):
        """Test dictionary serialization."""
        order = CausalOrder(
            ordered_events=["meeting", "decision"],
            precedence_matrix={("meeting", "decision"): 0.8},
            confidence=0.85,
            total_observations=100,
        )

        d = order.to_dict()
        assert d["ordered_events"] == ["meeting", "decision"]
        assert d["confidence"] == 0.85
        assert "meeting|decision" in d["precedence_matrix"]


class TestDOTSCausalOrdering:
    """Tests for DOTSCausalOrdering class."""

    @pytest.fixture
    def dots(self):
        """Create DOTSCausalOrdering with default settings.

        Note: max_lag_hours=24 ensures that cross-pairs (46h gap) don't
        contaminate the natural pairs (2h gap) in test fixtures where
        events are spaced 2 days apart.
        """
        return DOTSCausalOrdering(
            min_observations=2,
            precedence_threshold=0.6,
            max_lag_hours=24,
        )

    @pytest.fixture
    def simple_events(self) -> List[Dict]:
        """Create simple event sequence where meeting always precedes decision."""
        now = datetime.utcnow()
        events = []

        # Create 5 meeting -> decision pairs
        for i in range(5):
            base_time = now - timedelta(days=i * 2)
            events.append({
                "event_type": "meeting",
                "timestamp": base_time.isoformat(),
            })
            events.append({
                "event_type": "decision",
                "timestamp": (base_time + timedelta(hours=2)).isoformat(),
            })

        return events

    @pytest.fixture
    def complex_events(self) -> List[Dict]:
        """Create complex event sequence with multiple event types."""
        now = datetime.utcnow()
        events = []

        # Create chain: meeting -> discussion -> decision
        for i in range(5):
            base_time = now - timedelta(days=i * 3)
            events.append({
                "event_type": "meeting",
                "timestamp": base_time.isoformat(),
            })
            events.append({
                "event_type": "discussion",
                "timestamp": (base_time + timedelta(hours=1)).isoformat(),
            })
            events.append({
                "event_type": "decision",
                "timestamp": (base_time + timedelta(hours=3)).isoformat(),
            })

        return events

    def test_compute_causal_order_insufficient_events(self, dots):
        """Test with insufficient events."""
        events = [{"event_type": "meeting", "timestamp": datetime.utcnow().isoformat()}]
        order = dots.compute_causal_order(events)

        assert order.ordered_events == []
        assert order.confidence == 0.0

    def test_compute_causal_order_simple(self, dots, simple_events):
        """Test causal ordering with simple event sequence."""
        order = dots.compute_causal_order(simple_events)

        assert len(order.ordered_events) == 2
        # Meeting should precede decision in the order
        assert order.ordered_events.index("meeting") < order.ordered_events.index("decision")
        assert order.confidence > 0.5

    def test_compute_causal_order_complex(self, dots, complex_events):
        """Test causal ordering with complex event sequence."""
        order = dots.compute_causal_order(complex_events)

        assert len(order.ordered_events) == 3
        # Should be meeting -> discussion -> decision
        assert order.get_position("meeting") < order.get_position("discussion")
        assert order.get_position("discussion") < order.get_position("decision")

    def test_precedence_matrix_computed(self, dots, simple_events):
        """Test that precedence matrix is computed."""
        order = dots.compute_causal_order(simple_events)

        # meeting -> decision should have high precedence
        assert ("meeting", "decision") in order.precedence_matrix
        assert order.precedence_matrix[("meeting", "decision")] > 0.5

    def test_total_observations_recorded(self, dots, simple_events):
        """Test that total observations are recorded."""
        order = dots.compute_causal_order(simple_events)
        assert order.total_observations > 0

    def test_filter_by_causal_order(self, dots, simple_events):
        """Test filtering correlations by causal order."""
        order = dots.compute_causal_order(simple_events)

        # Create correlations - one valid, one invalid
        correlations = [
            {"event_type_a": "meeting", "event_type_b": "decision"},  # Valid
            {"event_type_a": "decision", "event_type_b": "meeting"},  # Invalid (reversed)
        ]

        filtered = dots.filter_by_causal_order(correlations, order)

        # Should keep only the valid one
        assert len(filtered) == 1
        assert filtered[0]["event_type_a"] == "meeting"

    def test_caching(self, dots, simple_events):
        """Test that orderings are cached."""
        order1 = dots.compute_causal_order(simple_events)

        # Get cached order
        cached = dots.get_cached_order(["meeting", "decision"])

        assert cached is not None
        assert cached.confidence == order1.confidence

    def test_statistics(self, dots, simple_events):
        """Test statistics tracking."""
        dots.compute_causal_order(simple_events)
        stats = dots.get_statistics()

        assert stats["computations"] == 1
        assert stats["cached_orderings"] == 1

    def test_export_for_token_priors(self, dots, simple_events):
        """Test natural language export."""
        dots.compute_causal_order(simple_events)
        export = dots.export_for_token_priors()

        assert "DOTS" in export
        assert "Causal Orderings" in export

    def test_clear_cache(self, dots, simple_events):
        """Test cache clearing."""
        dots.compute_causal_order(simple_events)
        assert len(dots._ordering_cache) == 1

        dots.clear_cache()
        assert len(dots._ordering_cache) == 0


class TestDOTSAlgorithmProperties:
    """Tests verifying DOTS algorithm properties."""

    def test_temporal_consistency(self):
        """Events that always precede should be ordered first."""
        dots = DOTSCausalOrdering(min_observations=2, precedence_threshold=0.6)

        now = datetime.utcnow()
        events = []

        # A always happens before B, which always happens before C
        for i in range(10):
            base_time = now - timedelta(days=i)
            events.append({"event_type": "A", "timestamp": base_time.isoformat()})
            events.append({"event_type": "B", "timestamp": (base_time + timedelta(hours=1)).isoformat()})
            events.append({"event_type": "C", "timestamp": (base_time + timedelta(hours=2)).isoformat()})

        order = dots.compute_causal_order(events)

        # Verify strict ordering A -> B -> C
        assert order.get_position("A") < order.get_position("B")
        assert order.get_position("B") < order.get_position("C")

    def test_high_confidence_with_consistent_ordering(self):
        """Consistent temporal patterns should yield high confidence."""
        dots = DOTSCausalOrdering(min_observations=2, precedence_threshold=0.5)

        now = datetime.utcnow()
        events = []

        # Very consistent ordering
        for i in range(20):
            base_time = now - timedelta(days=i)
            events.append({"event_type": "cause", "timestamp": base_time.isoformat()})
            events.append({"event_type": "effect", "timestamp": (base_time + timedelta(hours=1)).isoformat()})

        order = dots.compute_causal_order(events)

        # Should have very high confidence
        assert order.confidence > 0.8

    def test_max_lag_respects_threshold(self):
        """Events outside max_lag should not contribute to precedence."""
        dots = DOTSCausalOrdering(
            min_observations=2,
            precedence_threshold=0.5,
            max_lag_hours=12,  # 12 hour max lag
        )

        now = datetime.utcnow()
        events = []

        # Create pairs with 24-hour gap (outside max_lag)
        for i in range(5):
            base_time = now - timedelta(days=i * 2)
            events.append({"event_type": "A", "timestamp": base_time.isoformat()})
            events.append({"event_type": "B", "timestamp": (base_time + timedelta(hours=24)).isoformat()})

        order = dots.compute_causal_order(events)

        # Should not find strong precedence (gap too large)
        if order.precedence_matrix:
            # If there is any precedence, it should be weak
            prec = order.precedence_matrix.get(("A", "B"), 0)
            assert prec < 0.6 or order.ordered_events == []

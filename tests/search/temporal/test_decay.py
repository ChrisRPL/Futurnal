"""Tests for Temporal Decay Scoring.

Tests the TemporalDecayScorer class for correctness of decay formula.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md
"""

import math
from datetime import datetime, timedelta

import pytest

from futurnal.search.config import TemporalSearchConfig
from futurnal.search.temporal.decay import TemporalDecayScorer, create_decay_scorer
from futurnal.search.temporal.exceptions import DecayScoringError
from tests.search.conftest import create_test_event


class TestTemporalDecayScorer:
    """Test TemporalDecayScorer functionality."""

    def test_init_default_half_life(self):
        """Test default half-life initialization."""
        scorer = TemporalDecayScorer()
        assert scorer.half_life_days == 30.0
        assert scorer.decay_constant == math.log(2) / 30.0

    def test_init_custom_half_life(self):
        """Test custom half-life initialization."""
        scorer = TemporalDecayScorer(half_life_days=14.0)
        assert scorer.half_life_days == 14.0
        assert scorer.decay_constant == math.log(2) / 14.0

    def test_init_from_config(self):
        """Test initialization from config."""
        config = TemporalSearchConfig(decay_half_life_days=7.0)
        scorer = TemporalDecayScorer(config=config)
        assert scorer.half_life_days == 7.0

    def test_init_invalid_half_life(self):
        """Test that negative half-life raises error."""
        with pytest.raises(ValueError):
            TemporalDecayScorer(half_life_days=0)

        with pytest.raises(ValueError):
            TemporalDecayScorer(half_life_days=-5)

    def test_score_at_half_life(self):
        """Score at half-life should be ~0.5."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Event 30 days ago
        reference = datetime(2024, 6, 1)
        event = create_test_event(
            "e1", "Test", "meeting",
            datetime(2024, 5, 2)  # 30 days before reference
        )

        score = scorer.score(event, reference_time=reference)
        assert abs(score - 0.5) < 0.01  # Allow small floating point error

    def test_score_at_double_half_life(self):
        """Score at 2x half-life should be ~0.25."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Event 60 days ago
        reference = datetime(2024, 6, 1)
        event = create_test_event(
            "e1", "Test", "meeting",
            datetime(2024, 4, 2)  # 60 days before reference
        )

        score = scorer.score(event, reference_time=reference)
        assert abs(score - 0.25) < 0.01

    def test_score_recent_event(self):
        """Recent event should have score close to 1.0."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Event 1 day ago
        reference = datetime(2024, 6, 1)
        event = create_test_event(
            "e1", "Test", "meeting",
            datetime(2024, 5, 31)  # 1 day before reference
        )

        score = scorer.score(event, reference_time=reference)
        assert score > 0.95

    def test_score_future_event(self):
        """Future event should have full score (no penalty)."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Event in future
        reference = datetime(2024, 6, 1)
        event = create_test_event(
            "e1", "Test", "meeting",
            datetime(2024, 6, 15)  # After reference
        )

        score = scorer.score(event, reference_time=reference)
        assert score == 1.0

    def test_score_with_base_score(self):
        """Test custom base score."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        reference = datetime(2024, 6, 1)
        event = create_test_event(
            "e1", "Test", "meeting",
            datetime(2024, 5, 2)  # 30 days ago
        )

        score = scorer.score(event, reference_time=reference, base_score=2.0)
        assert abs(score - 1.0) < 0.02  # 2.0 * 0.5 = 1.0

    def test_score_default_reference_time(self):
        """Test using current time as default reference."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Very old event
        event = create_test_event(
            "e1", "Test", "meeting",
            datetime(2020, 1, 1)
        )

        score = scorer.score(event)  # Uses current time
        assert score < 0.01  # Very old event should have very low score

    def test_compute_decay_factor(self):
        """Test decay factor computation."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        reference = datetime(2024, 6, 1)

        # At half-life
        factor = scorer.compute_decay_factor(
            datetime(2024, 5, 2),  # 30 days ago
            reference_time=reference
        )
        assert abs(factor - 0.5) < 0.01

        # Today
        factor = scorer.compute_decay_factor(reference, reference_time=reference)
        assert factor == 1.0

    def test_apply_decay_sorting(self):
        """Test that apply_decay sorts by score descending."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        reference = datetime(2024, 6, 1)
        events = [
            create_test_event("e1", "Old", "meeting", datetime(2024, 1, 1)),
            create_test_event("e2", "Recent", "meeting", datetime(2024, 5, 25)),
            create_test_event("e3", "Medium", "meeting", datetime(2024, 4, 1)),
        ]

        scored = scorer.apply_decay(events, reference_time=reference)

        # Should be sorted by score descending (most recent first)
        assert scored[0].event.id == "e2"  # Most recent
        assert scored[1].event.id == "e3"  # Medium
        assert scored[2].event.id == "e1"  # Oldest

    def test_apply_decay_with_base_scores(self):
        """Test apply_decay with custom base scores."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        reference = datetime(2024, 6, 1)
        events = [
            create_test_event("e1", "High relevance", "meeting", datetime(2024, 5, 1)),
            create_test_event("e2", "Low relevance", "meeting", datetime(2024, 5, 25)),
        ]
        base_scores = [2.0, 0.5]

        scored = scorer.apply_decay(
            events,
            reference_time=reference,
            base_scores=base_scores
        )

        # e1 has higher base but is older
        # e2 is more recent but has lower base
        # The sorting depends on final scores
        assert len(scored) == 2
        assert scored[0].base_score != scored[1].base_score

    def test_apply_decay_mismatched_lengths(self):
        """Test error when events and base_scores lengths don't match."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        events = [
            create_test_event("e1", "Test 1", "meeting", datetime(2024, 5, 1)),
            create_test_event("e2", "Test 2", "meeting", datetime(2024, 5, 2)),
        ]
        base_scores = [1.0]  # Wrong length

        with pytest.raises(DecayScoringError):
            scorer.apply_decay(events, base_scores=base_scores)

    def test_apply_decay_empty_list(self):
        """Test apply_decay with empty list."""
        scorer = TemporalDecayScorer(half_life_days=30.0)
        scored = scorer.apply_decay([])
        assert scored == []

    def test_apply_decay_tuples(self):
        """Test apply_decay_tuples returns simple tuples."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        reference = datetime(2024, 6, 1)
        events = [
            create_test_event("e1", "Test", "meeting", datetime(2024, 5, 25)),
        ]

        results = scorer.apply_decay_tuples(events, reference_time=reference)

        assert len(results) == 1
        assert isinstance(results[0], tuple)
        assert results[0][0].id == "e1"
        assert isinstance(results[0][1], float)

    def test_days_for_score(self):
        """Test days_for_score calculation."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Days to reach 0.5 of base
        days = scorer.days_for_score(0.5)
        assert abs(days - 30.0) < 0.01  # Should be half-life

        # Days to reach 0.25 of base
        days = scorer.days_for_score(0.25)
        assert abs(days - 60.0) < 0.01  # Should be 2x half-life

        # Days to reach 0.125 of base
        days = scorer.days_for_score(0.125)
        assert abs(days - 90.0) < 0.1  # Should be 3x half-life

    def test_days_for_score_invalid(self):
        """Test days_for_score with invalid inputs."""
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Target > base
        with pytest.raises(ValueError):
            scorer.days_for_score(1.5)

        # Target <= 0
        with pytest.raises(ValueError):
            scorer.days_for_score(0)

        with pytest.raises(ValueError):
            scorer.days_for_score(-0.5)

    def test_repr(self):
        """Test string representation."""
        scorer = TemporalDecayScorer(half_life_days=30.0)
        repr_str = repr(scorer)
        assert "TemporalDecayScorer" in repr_str
        assert "half_life_days=30.0" in repr_str

    def test_timezone_handling(self):
        """Test that timezones are handled correctly."""
        import pytz
        scorer = TemporalDecayScorer(half_life_days=30.0)

        # Event with timezone
        utc = pytz.UTC
        event_time = datetime(2024, 5, 2, tzinfo=utc)
        event = create_test_event("e1", "Test", "meeting", event_time)

        # Reference without timezone
        reference = datetime(2024, 6, 1)

        # Should not raise error
        score = scorer.score(event, reference_time=reference)
        assert 0 <= score <= 1


class TestCreateDecayScorer:
    """Test create_decay_scorer factory function."""

    def test_create_with_half_life(self):
        """Test creating scorer with explicit half-life."""
        scorer = create_decay_scorer(half_life_days=14.0)
        assert scorer.half_life_days == 14.0

    def test_create_with_config(self):
        """Test creating scorer from config."""
        config = TemporalSearchConfig(decay_half_life_days=7.0)
        scorer = create_decay_scorer(config=config)
        assert scorer.half_life_days == 7.0

    def test_create_half_life_takes_precedence(self):
        """Test that explicit half_life overrides config."""
        config = TemporalSearchConfig(decay_half_life_days=7.0)
        scorer = create_decay_scorer(half_life_days=21.0, config=config)
        assert scorer.half_life_days == 21.0

    def test_create_default(self):
        """Test creating scorer with defaults."""
        scorer = create_decay_scorer()
        assert scorer.half_life_days == 30.0

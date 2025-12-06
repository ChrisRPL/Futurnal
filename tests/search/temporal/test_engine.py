"""Tests for Temporal Query Engine.

Tests the TemporalQueryEngine with mock PKG queries.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md
"""

from datetime import datetime, timedelta

import pytest

from futurnal.search.config import TemporalSearchConfig
from futurnal.search.temporal.engine import TemporalQueryEngine
from futurnal.search.temporal.types import TemporalQuery, TemporalQueryType
from futurnal.search.temporal.results import (
    ScoredEvent,
    SequenceMatch,
    TemporalCorrelationResult,
    TemporalSearchResult,
)
from tests.search.conftest import (
    MockTemporalGraphQueries,
    create_test_event,
    create_pattern_events,
)


class TestTemporalQueryEngineInit:
    """Test TemporalQueryEngine initialization."""

    def test_init_basic(self, mock_pkg_queries, temporal_config):
        """Test basic initialization."""
        engine = TemporalQueryEngine(
            pkg_queries=mock_pkg_queries,
            config=temporal_config,
        )
        assert engine.config == temporal_config
        assert engine.decay_scorer is not None
        assert engine.pattern_matcher is not None
        assert engine.correlation_detector is not None

    def test_init_without_vector_store(self, mock_pkg_queries):
        """Test initialization without vector store."""
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)
        # Should initialize without error
        assert engine is not None

    def test_init_default_config(self, mock_pkg_queries):
        """Test initialization with default config."""
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)
        assert engine.config.decay_half_life_days == 30.0


class TestTimeRangeQueries:
    """Test time range query functionality."""

    def test_query_time_range_basic(self, mock_pkg_queries, sample_events):
        """Test basic time range query."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_time_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        assert isinstance(result, TemporalSearchResult)
        assert len(result.items) > 0
        assert result.decay_applied is True

        # All items should be ScoredEvent
        for item in result.items:
            assert isinstance(item, ScoredEvent)

    def test_query_time_range_with_event_type(self, mock_pkg_queries, sample_events):
        """Test time range query with event type filter."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_time_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 3, 31),
            event_types=["meeting"],
        )

        # All results should be meetings
        for item in result.items:
            assert item.event.event_type == "meeting"

    def test_query_time_range_without_decay(self, mock_pkg_queries, sample_events):
        """Test time range query without decay scoring."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_time_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            enable_decay=False,
        )

        assert result.decay_applied is False
        # All decay scores should be 1.0
        for item in result.items:
            assert item.decay_score == 1.0

    def test_query_time_range_empty_result(self, mock_pkg_queries, sample_events):
        """Test time range query with no results."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_time_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 31),
        )

        assert len(result.items) == 0


class TestBeforeAfterQueries:
    """Test before/after query functionality."""

    def test_query_before_with_event_id(self, mock_pkg_queries, sample_events):
        """Test before query with event ID."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        events = engine.query_before(
            reference_event_id="e5",  # Publication on Jan 25
            limit=10,
        )

        # All events should be before e5
        for event in events:
            assert event.timestamp < sample_events[4].timestamp

    def test_query_before_with_timestamp(self, mock_pkg_queries, sample_events):
        """Test before query with timestamp."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        events = engine.query_before(
            reference_timestamp=datetime(2024, 1, 20),
            time_window=timedelta(days=30),
            limit=10,
        )

        for event in events:
            assert event.timestamp < datetime(2024, 1, 20)

    def test_query_after_with_event_id(self, mock_pkg_queries, sample_events):
        """Test after query with event ID."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        events = engine.query_after(
            reference_event_id="e1",  # First meeting on Jan 1
            limit=10,
        )

        # All events should be after e1
        for event in events:
            assert event.timestamp > sample_events[0].timestamp


class TestTemporalNeighborhood:
    """Test temporal neighborhood queries."""

    def test_query_temporal_neighborhood(self, mock_pkg_queries, sample_events):
        """Test basic temporal neighborhood query."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        neighborhood = engine.query_temporal_neighborhood(
            entity_id="e3",  # Meeting on Jan 15
            time_window=timedelta(days=7),
        )

        assert neighborhood.center_id == "e3"
        # Should include events within 7 days of Jan 15


class TestPatternMatching:
    """Test pattern matching functionality."""

    def test_query_temporal_sequence(self, mock_pkg_queries, pattern_events):
        """Test temporal sequence query."""
        mock_pkg_queries.set_events(pattern_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        matches = engine.query_temporal_sequence(
            pattern=["meeting", "decision"],
            time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            max_gap=timedelta(days=7),
        )

        assert len(matches) > 0
        for match in matches:
            assert isinstance(match, SequenceMatch)
            assert match.pattern == ["meeting", "decision"]

    def test_find_recurring_patterns(self, mock_pkg_queries, pattern_events):
        """Test recurring pattern discovery."""
        mock_pkg_queries.set_events(pattern_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        patterns = engine.find_recurring_patterns(
            time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            min_occurrences=3,
        )

        # Should find the meeting -> decision pattern
        assert len(patterns) > 0


class TestCorrelationDetection:
    """Test correlation detection functionality."""

    def test_query_temporal_correlation(self, mock_pkg_queries, pattern_events):
        """Test correlation detection."""
        mock_pkg_queries.set_events(pattern_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_temporal_correlation(
            event_type_a="meeting",
            event_type_b="decision",
            max_gap=timedelta(days=7),
            min_occurrences=3,
        )

        assert isinstance(result, TemporalCorrelationResult)
        # With pattern_events, correlation should be found
        if result.correlation_found:
            assert result.co_occurrences >= 3
            assert result.avg_gap_days is not None

    def test_scan_all_correlations(self, mock_pkg_queries, pattern_events):
        """Test scanning for all correlations."""
        mock_pkg_queries.set_events(pattern_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        correlations = engine.scan_all_correlations(
            time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            min_occurrences=3,
        )

        # Should return list of correlations
        assert isinstance(correlations, list)


class TestUnifiedQueryInterface:
    """Test the unified query() interface."""

    def test_query_dispatch_time_range(self, mock_pkg_queries, sample_events):
        """Test query dispatch for TIME_RANGE."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        query = TemporalQuery(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )

        result = engine.query(query)
        assert isinstance(result, TemporalSearchResult)

    def test_query_dispatch_before(self, mock_pkg_queries, sample_events):
        """Test query dispatch for BEFORE."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        query = TemporalQuery(
            query_type=TemporalQueryType.BEFORE,
            reference_event_id="e5",
        )

        result = engine.query(query)
        assert isinstance(result, TemporalSearchResult)

    def test_query_dispatch_temporal_correlation(self, mock_pkg_queries, pattern_events):
        """Test query dispatch for TEMPORAL_CORRELATION."""
        mock_pkg_queries.set_events(pattern_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        query = TemporalQuery(
            query_type=TemporalQueryType.TEMPORAL_CORRELATION,
            event_type_a="meeting",
            event_type_b="decision",
        )

        result = engine.query(query)
        assert isinstance(result, TemporalCorrelationResult)

    def test_query_dispatch_temporal_sequence(self, mock_pkg_queries, pattern_events):
        """Test query dispatch for TEMPORAL_SEQUENCE."""
        mock_pkg_queries.set_events(pattern_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        query = TemporalQuery(
            query_type=TemporalQueryType.TEMPORAL_SEQUENCE,
            pattern=["meeting", "decision"],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 6, 30),
        )

        result = engine.query(query)
        assert isinstance(result, list)


class TestDecayScoringIntegration:
    """Test decay scoring integration with engine."""

    def test_decay_sorting(self, mock_pkg_queries):
        """Test that results are sorted by decay score."""
        # Create events with known timestamps
        events = [
            create_test_event("old", "Old Event", "meeting", datetime(2024, 1, 1)),
            create_test_event("recent", "Recent Event", "meeting", datetime(2024, 5, 1)),
            create_test_event("very_recent", "Very Recent", "meeting", datetime(2024, 5, 25)),
        ]
        mock_pkg_queries.set_events(events)

        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_time_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
            enable_decay=True,
        )

        # Results should be sorted by final_score descending
        scores = [item.final_score for item in result.items]
        assert scores == sorted(scores, reverse=True)

        # Most recent should be first
        assert result.items[0].event.id == "very_recent"


class TestCausalChainQueries:
    """Test causal chain query delegation."""

    def test_query_causal_chain(self, mock_pkg_queries, sample_events):
        """Test causal chain query delegates to PKG."""
        mock_pkg_queries.set_events(sample_events)
        engine = TemporalQueryEngine(pkg_queries=mock_pkg_queries)

        result = engine.query_causal_chain(
            start_event_id="e1",
            max_hops=5,
        )

        # MockTemporalGraphQueries returns empty result
        assert result.start_event_id == "e1"
        assert result.max_hops_requested == 5

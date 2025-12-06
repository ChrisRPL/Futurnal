"""Tests for causal query types.

Tests CausalQueryType enum and CausalQuery validation.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md
"""

from datetime import datetime, timedelta

import pytest

from futurnal.search.causal.types import CausalQuery, CausalQueryType


class TestCausalQueryType:
    """Test CausalQueryType enum."""

    def test_enum_values(self):
        """Verify all query types exist."""
        assert CausalQueryType.FIND_CAUSES.value == "find_causes"
        assert CausalQueryType.FIND_EFFECTS.value == "find_effects"
        assert CausalQueryType.CAUSAL_PATH.value == "causal_path"
        assert CausalQueryType.CAUSAL_CHAIN.value == "causal_chain"
        assert CausalQueryType.CORRELATION_PATTERN.value == "correlation_pattern"

    def test_enum_count(self):
        """Verify expected number of query types."""
        assert len(CausalQueryType) == 5


class TestCausalQuery:
    """Test CausalQuery model validation."""

    def test_find_causes_valid(self):
        """Test valid FIND_CAUSES query."""
        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="event_123",
            max_hops=3,
            min_confidence=0.6,
        )
        assert query.event_id == "event_123"
        assert query.max_hops == 3
        assert query.min_confidence == 0.6

    def test_find_causes_missing_event_id(self):
        """Test FIND_CAUSES requires event_id."""
        with pytest.raises(ValueError, match="event_id"):
            CausalQuery(
                query_type=CausalQueryType.FIND_CAUSES,
                # event_id missing
            )

    def test_find_effects_valid(self):
        """Test valid FIND_EFFECTS query."""
        query = CausalQuery(
            query_type=CausalQueryType.FIND_EFFECTS,
            event_id="event_456",
        )
        assert query.event_id == "event_456"

    def test_causal_path_valid(self):
        """Test valid CAUSAL_PATH query."""
        query = CausalQuery(
            query_type=CausalQueryType.CAUSAL_PATH,
            start_event_id="meeting_1",
            end_event_id="publication_1",
            max_hops=5,
        )
        assert query.start_event_id == "meeting_1"
        assert query.end_event_id == "publication_1"
        assert query.max_hops == 5

    def test_causal_path_missing_start(self):
        """Test CAUSAL_PATH requires start_event_id."""
        with pytest.raises(ValueError, match="start_event_id and end_event_id"):
            CausalQuery(
                query_type=CausalQueryType.CAUSAL_PATH,
                end_event_id="publication_1",
            )

    def test_causal_path_missing_end(self):
        """Test CAUSAL_PATH requires end_event_id."""
        with pytest.raises(ValueError, match="start_event_id and end_event_id"):
            CausalQuery(
                query_type=CausalQueryType.CAUSAL_PATH,
                start_event_id="meeting_1",
            )

    def test_causal_chain_valid(self):
        """Test valid CAUSAL_CHAIN query."""
        query = CausalQuery(
            query_type=CausalQueryType.CAUSAL_CHAIN,
            event_id="root_event",
        )
        assert query.event_id == "root_event"

    def test_correlation_pattern_valid(self):
        """Test valid CORRELATION_PATTERN query."""
        query = CausalQuery(
            query_type=CausalQueryType.CORRELATION_PATTERN,
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 6, 30),
            min_correlation_strength=0.7,
        )
        assert query.time_range_start == datetime(2024, 1, 1)
        assert query.time_range_end == datetime(2024, 6, 30)
        assert query.min_correlation_strength == 0.7

    def test_correlation_pattern_missing_time_range(self):
        """Test CORRELATION_PATTERN requires time range."""
        with pytest.raises(ValueError, match="time_range"):
            CausalQuery(
                query_type=CausalQueryType.CORRELATION_PATTERN,
                time_range_start=datetime(2024, 1, 1),
                # time_range_end missing
            )

    def test_correlation_pattern_invalid_range(self):
        """Test CORRELATION_PATTERN validates time range order."""
        with pytest.raises(ValueError, match="time_range_start must be before"):
            CausalQuery(
                query_type=CausalQueryType.CORRELATION_PATTERN,
                time_range_start=datetime(2024, 6, 30),  # Later
                time_range_end=datetime(2024, 1, 1),  # Earlier
            )

    def test_max_hops_bounds(self):
        """Test max_hops has valid bounds (1-10)."""
        # Valid minimum
        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="e1",
            max_hops=1,
        )
        assert query.max_hops == 1

        # Valid maximum
        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="e1",
            max_hops=10,
        )
        assert query.max_hops == 10

        # Invalid: too low
        with pytest.raises(ValueError):
            CausalQuery(
                query_type=CausalQueryType.FIND_CAUSES,
                event_id="e1",
                max_hops=0,
            )

        # Invalid: too high
        with pytest.raises(ValueError):
            CausalQuery(
                query_type=CausalQueryType.FIND_CAUSES,
                event_id="e1",
                max_hops=11,
            )

    def test_confidence_bounds(self):
        """Test min_confidence has valid bounds (0.0-1.0)."""
        # Valid minimum
        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="e1",
            min_confidence=0.0,
        )
        assert query.min_confidence == 0.0

        # Valid maximum
        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="e1",
            min_confidence=1.0,
        )
        assert query.min_confidence == 1.0

        # Invalid: negative
        with pytest.raises(ValueError):
            CausalQuery(
                query_type=CausalQueryType.FIND_CAUSES,
                event_id="e1",
                min_confidence=-0.1,
            )

        # Invalid: > 1.0
        with pytest.raises(ValueError):
            CausalQuery(
                query_type=CausalQueryType.FIND_CAUSES,
                event_id="e1",
                min_confidence=1.1,
            )

    def test_defaults(self):
        """Test default parameter values."""
        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="e1",
        )
        assert query.max_hops == 3  # Default
        assert query.min_confidence == 0.6  # Default
        assert query.limit == 20  # Default
        assert query.min_correlation_strength == 0.5  # Default

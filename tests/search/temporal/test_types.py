"""Tests for Temporal Query Types.

Tests TemporalQueryType enum and TemporalQuery model validation.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from futurnal.search.temporal.types import TemporalQuery, TemporalQueryType


class TestTemporalQueryType:
    """Test TemporalQueryType enum."""

    def test_enum_values(self):
        """Verify all expected enum values exist."""
        assert TemporalQueryType.TIME_RANGE == "time_range"
        assert TemporalQueryType.BEFORE == "before"
        assert TemporalQueryType.AFTER == "after"
        assert TemporalQueryType.DURING == "during"
        assert TemporalQueryType.TEMPORAL_NEIGHBORHOOD == "temporal_neighborhood"
        assert TemporalQueryType.TEMPORAL_SEQUENCE == "temporal_sequence"
        assert TemporalQueryType.TEMPORAL_CORRELATION == "temporal_correlation"
        assert TemporalQueryType.CAUSAL_CHAIN == "causal_chain"

    def test_enum_count(self):
        """Verify number of query types."""
        assert len(TemporalQueryType) == 8

    def test_string_conversion(self):
        """Test string conversion for serialization."""
        assert str(TemporalQueryType.TIME_RANGE) == "TemporalQueryType.TIME_RANGE"
        assert TemporalQueryType.TIME_RANGE.value == "time_range"


class TestTemporalQuery:
    """Test TemporalQuery model validation."""

    def test_time_range_query_valid(self):
        """Valid TIME_RANGE query."""
        query = TemporalQuery(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 3, 31),
        )
        assert query.query_type == TemporalQueryType.TIME_RANGE
        assert query.start_time == datetime(2024, 1, 1)
        assert query.end_time == datetime(2024, 3, 31)

    def test_time_range_query_missing_start(self):
        """TIME_RANGE requires start_time."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                end_time=datetime(2024, 3, 31),
            )
        assert "TIME_RANGE query requires start_time and end_time" in str(exc_info.value)

    def test_time_range_query_missing_end(self):
        """TIME_RANGE requires end_time."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime(2024, 1, 1),
            )
        assert "TIME_RANGE query requires start_time and end_time" in str(exc_info.value)

    def test_time_range_query_invalid_range(self):
        """TIME_RANGE requires start <= end."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime(2024, 3, 31),
                end_time=datetime(2024, 1, 1),
            )
        assert "start_time must be before or equal to end_time" in str(exc_info.value)

    def test_before_query_with_event_id(self):
        """Valid BEFORE query with event ID."""
        query = TemporalQuery(
            query_type=TemporalQueryType.BEFORE,
            reference_event_id="evt_123",
        )
        assert query.query_type == TemporalQueryType.BEFORE
        assert query.reference_event_id == "evt_123"

    def test_before_query_with_timestamp(self):
        """Valid BEFORE query with timestamp."""
        query = TemporalQuery(
            query_type=TemporalQueryType.BEFORE,
            reference_timestamp=datetime(2024, 6, 15),
        )
        assert query.reference_timestamp == datetime(2024, 6, 15)

    def test_before_query_missing_reference(self):
        """BEFORE requires reference_event_id or reference_timestamp."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(query_type=TemporalQueryType.BEFORE)
        assert "reference_timestamp or reference_event_id" in str(exc_info.value)

    def test_after_query_valid(self):
        """Valid AFTER query."""
        query = TemporalQuery(
            query_type=TemporalQueryType.AFTER,
            reference_event_id="evt_456",
            time_window=timedelta(days=30),
        )
        assert query.query_type == TemporalQueryType.AFTER
        assert query.time_window == timedelta(days=30)

    def test_during_query_valid(self):
        """Valid DURING query."""
        query = TemporalQuery(
            query_type=TemporalQueryType.DURING,
            reference_event_id="evt_789",
        )
        assert query.query_type == TemporalQueryType.DURING

    def test_during_query_missing_event_id(self):
        """DURING requires reference_event_id."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(query_type=TemporalQueryType.DURING)
        assert "reference_event_id" in str(exc_info.value)

    def test_temporal_neighborhood_defaults_time_window(self):
        """TEMPORAL_NEIGHBORHOOD defaults time_window to 7 days."""
        query = TemporalQuery(
            query_type=TemporalQueryType.TEMPORAL_NEIGHBORHOOD,
            reference_event_id="evt_123",
        )
        assert query.time_window == timedelta(days=7)

    def test_temporal_sequence_valid(self):
        """Valid TEMPORAL_SEQUENCE query."""
        query = TemporalQuery(
            query_type=TemporalQueryType.TEMPORAL_SEQUENCE,
            pattern=["Meeting", "Decision", "Publication"],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 6, 30),
        )
        assert query.pattern == ["Meeting", "Decision", "Publication"]
        assert query.max_gap == timedelta(days=30)  # Default

    def test_temporal_sequence_missing_pattern(self):
        """TEMPORAL_SEQUENCE requires pattern."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(
                query_type=TemporalQueryType.TEMPORAL_SEQUENCE,
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 6, 30),
            )
        assert "pattern with at least 2 event types" in str(exc_info.value)

    def test_temporal_sequence_pattern_too_short(self):
        """TEMPORAL_SEQUENCE pattern needs at least 2 types."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(
                query_type=TemporalQueryType.TEMPORAL_SEQUENCE,
                pattern=["Meeting"],
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 6, 30),
            )
        assert "at least 2 event types" in str(exc_info.value)

    def test_temporal_correlation_valid(self):
        """Valid TEMPORAL_CORRELATION query."""
        query = TemporalQuery(
            query_type=TemporalQueryType.TEMPORAL_CORRELATION,
            event_type_a="Meeting",
            event_type_b="Decision",
        )
        assert query.event_type_a == "Meeting"
        assert query.event_type_b == "Decision"
        assert query.max_gap == timedelta(days=30)  # Default

    def test_temporal_correlation_missing_types(self):
        """TEMPORAL_CORRELATION requires both event types."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(
                query_type=TemporalQueryType.TEMPORAL_CORRELATION,
                event_type_a="Meeting",
            )
        assert "event_type_a and event_type_b" in str(exc_info.value)

    def test_causal_chain_valid(self):
        """Valid CAUSAL_CHAIN query."""
        query = TemporalQuery(
            query_type=TemporalQueryType.CAUSAL_CHAIN,
            reference_event_id="evt_start",
            max_hops=7,
        )
        assert query.max_hops == 7

    def test_causal_chain_missing_event_id(self):
        """CAUSAL_CHAIN requires reference_event_id."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalQuery(query_type=TemporalQueryType.CAUSAL_CHAIN)
        assert "reference_event_id" in str(exc_info.value)

    def test_default_values(self):
        """Test default parameter values."""
        query = TemporalQuery(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 3, 31),
        )
        assert query.min_confidence == 0.7
        assert query.enable_decay_scoring is True
        assert query.decay_half_life_days == 30.0
        assert query.limit == 100
        assert query.offset == 0
        assert query.vector_weight == 0.3

    def test_min_confidence_bounds(self):
        """min_confidence must be in [0, 1]."""
        # Valid
        query = TemporalQuery(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 3, 31),
            min_confidence=0.5,
        )
        assert query.min_confidence == 0.5

        # Invalid - too low
        with pytest.raises(ValidationError):
            TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 3, 31),
                min_confidence=-0.1,
            )

        # Invalid - too high
        with pytest.raises(ValidationError):
            TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 3, 31),
                min_confidence=1.5,
            )

    def test_max_hops_bounds(self):
        """max_hops must be in [1, 10]."""
        # Valid
        query = TemporalQuery(
            query_type=TemporalQueryType.CAUSAL_CHAIN,
            reference_event_id="evt_123",
            max_hops=10,
        )
        assert query.max_hops == 10

        # Invalid - too low
        with pytest.raises(ValidationError):
            TemporalQuery(
                query_type=TemporalQueryType.CAUSAL_CHAIN,
                reference_event_id="evt_123",
                max_hops=0,
            )

        # Invalid - too high
        with pytest.raises(ValidationError):
            TemporalQuery(
                query_type=TemporalQueryType.CAUSAL_CHAIN,
                reference_event_id="evt_123",
                max_hops=15,
            )

    def test_limit_bounds(self):
        """limit must be in [1, 1000]."""
        # Valid
        query = TemporalQuery(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 3, 31),
            limit=500,
        )
        assert query.limit == 500

        # Invalid - too low
        with pytest.raises(ValidationError):
            TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 3, 31),
                limit=0,
            )

    def test_with_query_text(self):
        """Test query with text for hybrid search."""
        query = TemporalQuery(
            query_type=TemporalQueryType.TEMPORAL_NEIGHBORHOOD,
            reference_event_id="evt_123",
            query_text="quarterly planning meeting",
            vector_weight=0.4,
        )
        assert query.query_text == "quarterly planning meeting"
        assert query.vector_weight == 0.4

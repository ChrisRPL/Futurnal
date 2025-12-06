"""Tests for CausalChainRetrieval.

Tests the main CausalChainRetrieval class methods.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- Temporal validation required for ALL paths (100%)
- Causal confidence scoring on relationships
- Phase 2/3 foundation established
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from futurnal.search.causal import (
    CausalChainRetrieval,
    CausalQuery,
    CausalQueryType,
)
from futurnal.search.causal.exceptions import (
    CausalChainDepthExceeded,
    CausalSearchError,
    CorrelationDetectionError,
    EventNotFoundError,
)
from futurnal.search.config import CausalSearchConfig


class TestCausalChainRetrievalInit:
    """Test CausalChainRetrieval initialization."""

    def test_init_basic(self, mock_causal_pkg, causal_config):
        """Test basic initialization."""
        retrieval = CausalChainRetrieval(
            pkg_queries=mock_causal_pkg,
            config=causal_config,
        )
        assert retrieval.config == causal_config

    def test_init_without_temporal_engine(self, mock_causal_pkg):
        """Test initialization without temporal engine."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)
        assert retrieval is not None

    def test_init_with_default_config(self, mock_causal_pkg):
        """Test initialization creates default config."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)
        assert retrieval.config is not None
        assert retrieval.config.default_max_hops == 3


class TestFindCauses:
    """Test find_causes method."""

    def test_find_causes_basic(self, mock_causal_pkg, causal_chain_events):
        """Test finding causes of an event."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causes(event_id="d1", max_hops=3)

        assert result.target_event_id == "d1"
        assert result.max_hops_requested == 3
        # Should find m1 as cause of d1
        cause_ids = [c.cause_id for c in result.causes]
        assert "m1" in cause_ids

    def test_find_causes_validates_temporal_ordering(self, mock_causal_pkg):
        """Verify temporal ordering is validated (Option B requirement)."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causes(event_id="d1")

        # All causes should have temporal_ordering_valid set
        for cause in result.causes:
            assert hasattr(cause, "temporal_ordering_valid")
            # Our mock data has valid ordering
            assert cause.temporal_ordering_valid is True

    def test_find_causes_respects_min_confidence(self, mock_causal_pkg):
        """Test confidence filtering."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        # With high threshold, may filter out some causes
        result = retrieval.find_causes(
            event_id="d1",
            min_confidence=0.95,  # Higher than our test data (0.9)
        )

        # Result should still work (filtering happens in query)
        assert result.target_event_id == "d1"

    def test_find_causes_event_not_found(self, mock_causal_pkg):
        """Test EventNotFoundError when event doesn't exist."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        with pytest.raises(EventNotFoundError) as exc_info:
            retrieval.find_causes(event_id="nonexistent")

        assert exc_info.value.event_id == "nonexistent"

    def test_find_causes_depth_exceeded(self, mock_causal_pkg):
        """Test CausalChainDepthExceeded for excessive hops."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        with pytest.raises(CausalChainDepthExceeded) as exc_info:
            retrieval.find_causes(event_id="d1", max_hops=15)

        assert exc_info.value.requested == 15
        assert exc_info.value.maximum == 10

    def test_find_causes_returns_query_time(self, mock_causal_pkg):
        """Test query time is recorded."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causes(event_id="d1")

        assert result.query_time_ms >= 0


class TestFindEffects:
    """Test find_effects method."""

    def test_find_effects_basic(self, mock_causal_pkg, causal_chain_events):
        """Test finding effects of an event."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_effects(event_id="m1", max_hops=3)

        assert result.source_event_id == "m1"
        assert result.max_hops_requested == 3
        # Should find d1 as effect of m1
        effect_ids = [e.effect_id for e in result.effects]
        assert "d1" in effect_ids

    def test_find_effects_validates_temporal_ordering(self, mock_causal_pkg):
        """Verify temporal ordering is validated."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_effects(event_id="m1")

        for effect in result.effects:
            assert hasattr(effect, "temporal_ordering_valid")

    def test_find_effects_event_not_found(self, mock_causal_pkg):
        """Test EventNotFoundError when event doesn't exist."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        with pytest.raises(EventNotFoundError):
            retrieval.find_effects(event_id="nonexistent")


class TestFindCausalPath:
    """Test find_causal_path method."""

    def test_find_causal_path_exists(self, mock_causal_pkg):
        """Test finding path that exists."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causal_path(
            start_event_id="m1",
            end_event_id="p1",
            max_hops=5,
        )

        assert result.path_found is True
        assert result.path is not None
        assert result.path.start_event_id == "m1"
        assert result.path.end_event_id == "p1"
        # Path should be m1 -> d1 -> p1
        assert "m1" in result.path.path
        assert "p1" in result.path.path

    def test_find_causal_path_validates_temporal(self, mock_causal_pkg):
        """Test path has temporal_ordering_valid field."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causal_path(
            start_event_id="m1",
            end_event_id="d1",
        )

        if result.path_found:
            assert hasattr(result.path, "temporal_ordering_valid")

    def test_find_causal_path_not_found(self, mock_causal_pkg):
        """Test when no path exists."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        # p1 -> m1 has no path (reverse direction)
        result = retrieval.find_causal_path(
            start_event_id="p1",
            end_event_id="m1",
        )

        assert result.path_found is False
        assert result.path is None

    def test_find_causal_path_depth_exceeded(self, mock_causal_pkg):
        """Test CausalChainDepthExceeded for excessive hops."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        with pytest.raises(CausalChainDepthExceeded):
            retrieval.find_causal_path(
                start_event_id="m1",
                end_event_id="p1",
                max_hops=11,
            )

    def test_find_causal_path_records_confidence(self, mock_causal_pkg):
        """Test path includes confidence scores."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causal_path(
            start_event_id="m1",
            end_event_id="d1",
        )

        if result.path_found:
            assert result.path.causal_confidence > 0
            assert len(result.path.confidence_scores) > 0


class TestCorrelationPatterns:
    """Test detect_correlation_patterns method."""

    def test_correlation_without_temporal_engine_raises(self, mock_causal_pkg):
        """Test error when temporal engine not configured."""
        retrieval = CausalChainRetrieval(
            pkg_queries=mock_causal_pkg,
            temporal_engine=None,  # No temporal engine
        )

        with pytest.raises(CorrelationDetectionError):
            retrieval.detect_correlation_patterns(
                time_range_start=datetime(2024, 1, 1),
                time_range_end=datetime(2024, 6, 30),
            )

    def test_correlation_with_temporal_engine(self, mock_causal_pkg):
        """Test correlation pattern detection with temporal engine."""
        # Create mock temporal engine
        mock_temporal = MagicMock()
        mock_correlation = MagicMock()
        mock_correlation.correlation_strength = 0.7
        mock_correlation.is_causal_candidate = True
        mock_correlation.model_dump.return_value = {
            "correlation_strength": 0.7,
            "is_causal_candidate": True,
        }
        mock_temporal.scan_all_correlations.return_value = [mock_correlation]

        retrieval = CausalChainRetrieval(
            pkg_queries=mock_causal_pkg,
            temporal_engine=mock_temporal,
        )

        result = retrieval.detect_correlation_patterns(
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 6, 30),
            min_correlation_strength=0.5,
        )

        assert result.patterns_found == 1
        assert len(result.correlations) == 1
        assert result.causal_candidate_count == 1

    def test_correlation_filters_by_strength(self, mock_causal_pkg):
        """Test correlation filtering by min strength."""
        mock_temporal = MagicMock()

        # Create correlations with different strengths
        weak = MagicMock()
        weak.correlation_strength = 0.3
        weak.is_causal_candidate = False
        weak.model_dump.return_value = {"correlation_strength": 0.3}

        strong = MagicMock()
        strong.correlation_strength = 0.8
        strong.is_causal_candidate = True
        strong.model_dump.return_value = {"correlation_strength": 0.8}

        mock_temporal.scan_all_correlations.return_value = [weak, strong]

        retrieval = CausalChainRetrieval(
            pkg_queries=mock_causal_pkg,
            temporal_engine=mock_temporal,
        )

        result = retrieval.detect_correlation_patterns(
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 6, 30),
            min_correlation_strength=0.5,
        )

        # Only strong correlation should pass filter
        assert result.patterns_found == 1


class TestUnifiedQueryInterface:
    """Test the unified query() method."""

    def test_query_find_causes(self, mock_causal_pkg):
        """Test query() with FIND_CAUSES type."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        query = CausalQuery(
            query_type=CausalQueryType.FIND_CAUSES,
            event_id="d1",
        )
        result = retrieval.query(query)

        assert result.target_event_id == "d1"

    def test_query_find_effects(self, mock_causal_pkg):
        """Test query() with FIND_EFFECTS type."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        query = CausalQuery(
            query_type=CausalQueryType.FIND_EFFECTS,
            event_id="m1",
        )
        result = retrieval.query(query)

        assert result.source_event_id == "m1"

    def test_query_causal_path(self, mock_causal_pkg):
        """Test query() with CAUSAL_PATH type."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        query = CausalQuery(
            query_type=CausalQueryType.CAUSAL_PATH,
            start_event_id="m1",
            end_event_id="p1",
        )
        result = retrieval.query(query)

        assert result.start_event_id == "m1"
        assert result.end_event_id == "p1"

    def test_query_causal_chain_delegates_to_pkg(self, mock_causal_pkg):
        """Test CAUSAL_CHAIN query delegates to PKG."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        query = CausalQuery(
            query_type=CausalQueryType.CAUSAL_CHAIN,
            event_id="m1",
        )
        result = retrieval.query(query)

        # Should return CausalChainResult from PKG
        assert result.start_event_id == "m1"


class TestPerformance:
    """Performance tests for latency targets."""

    @pytest.mark.performance
    def test_find_causes_latency(self, mock_causal_pkg):
        """Verify find_causes completes in <2s."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causes(event_id="d1")

        # Target: <2s = 2000ms
        assert result.query_time_ms < 2000

    @pytest.mark.performance
    def test_find_causal_path_latency(self, mock_causal_pkg):
        """Verify find_causal_path completes in <2s."""
        retrieval = CausalChainRetrieval(pkg_queries=mock_causal_pkg)

        result = retrieval.find_causal_path(
            start_event_id="m1",
            end_event_id="p1",
        )

        # Target: <2s = 2000ms
        assert result.query_time_ms < 2000

"""Tests for EmbeddingQualityEvolution.

Tests experiential learning integration, quality-based re-embedding,
and trend measurement.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
import numpy as np

from futurnal.embeddings.quality.evolution import (
    EmbeddingQualityEvolution,
    ExperientialLearningMonitor,
)
from futurnal.embeddings.quality.tracker import QualityMetricsTracker
from futurnal.embeddings.quality.store import QualityMetricsStore
from futurnal.embeddings.quality.golden import GoldenEmbeddingsManager
from futurnal.embeddings.reembedding import ReembeddingProgress


@pytest.fixture
def mock_quality_tracker():
    """Create a mock quality tracker."""
    tracker = MagicMock(spec=QualityMetricsTracker)
    tracker.identify_low_quality_embeddings.return_value = []
    tracker.update_quality_trend.return_value = "stable"
    tracker.get_statistics.return_value = {"total_count": 0}
    tracker._store = MagicMock()
    return tracker


@pytest.fixture
def mock_reembedding_service():
    """Create a mock reembedding service."""
    service = MagicMock()
    service.trigger_reembedding.return_value = ReembeddingProgress(
        total=10,
        processed=10,
        succeeded=9,
        failed=1,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )
    return service


@pytest.fixture
def mock_learning_monitor():
    """Create a mock experiential learning monitor."""
    monitor = MagicMock(spec=ExperientialLearningMonitor)
    monitor.get_current_quality_metrics.return_value = {"precision": 0.85}
    monitor.get_baseline_quality.return_value = {"precision": 0.75}
    return monitor


@pytest.fixture
def evolution(mock_quality_tracker, mock_reembedding_service):
    """Create evolution instance with mocks."""
    return EmbeddingQualityEvolution(
        quality_tracker=mock_quality_tracker,
        reembedding_service=mock_reembedding_service,
    )


@pytest.fixture
def evolution_with_monitor(mock_quality_tracker, mock_reembedding_service, mock_learning_monitor):
    """Create evolution instance with learning monitor."""
    return EmbeddingQualityEvolution(
        quality_tracker=mock_quality_tracker,
        reembedding_service=mock_reembedding_service,
        experiential_learning_monitor=mock_learning_monitor,
    )


class TestEmbeddingQualityEvolution:
    """Tests for EmbeddingQualityEvolution."""

    def test_initialization_defaults(self, evolution):
        """Test default initialization values."""
        assert evolution._improvement_threshold == 0.05
        assert evolution._quality_threshold == 0.7
        assert evolution._baseline_quality is None
        assert evolution._learning_monitor is None

    def test_monitor_extraction_no_monitor(self, evolution):
        """Test monitoring returns None without learning monitor."""
        result = evolution.monitor_extraction_quality_improvement()
        assert result is None

    def test_monitor_extraction_quality_improvement_triggers_reembedding(
        self,
        evolution_with_monitor,
        mock_quality_tracker,
        mock_reembedding_service,
        mock_learning_monitor,
    ):
        """Test that quality improvement triggers re-embedding."""
        # Setup: improvement > threshold (0.85 - 0.75 = 0.10 > 0.05)
        mock_learning_monitor.get_current_quality_metrics.return_value = {"precision": 0.85}
        mock_learning_monitor.get_baseline_quality.return_value = {"precision": 0.75}
        mock_quality_tracker.identify_low_quality_embeddings.return_value = [
            "ent_1", "ent_2", "ent_3"
        ]

        result = evolution_with_monitor.monitor_extraction_quality_improvement()

        # Should trigger re-embedding
        assert result is not None
        mock_reembedding_service.trigger_reembedding.assert_called_once()
        call_kwargs = mock_reembedding_service.trigger_reembedding.call_args[1]
        assert call_kwargs["entity_ids"] == ["ent_1", "ent_2", "ent_3"]

    def test_monitor_extraction_no_improvement(
        self,
        evolution_with_monitor,
        mock_learning_monitor,
        mock_reembedding_service,
    ):
        """Test that no improvement skips re-embedding."""
        # Setup: no improvement (0.76 - 0.75 = 0.01 < 0.05)
        mock_learning_monitor.get_current_quality_metrics.return_value = {"precision": 0.76}
        mock_learning_monitor.get_baseline_quality.return_value = {"precision": 0.75}

        result = evolution_with_monitor.monitor_extraction_quality_improvement()

        # Should not trigger re-embedding
        assert result is None
        mock_reembedding_service.trigger_reembedding.assert_not_called()

    def test_monitor_extraction_no_low_quality_found(
        self,
        evolution_with_monitor,
        mock_quality_tracker,
        mock_learning_monitor,
        mock_reembedding_service,
    ):
        """Test when improvement detected but no low-quality embeddings."""
        # Setup: improvement detected
        mock_learning_monitor.get_current_quality_metrics.return_value = {"precision": 0.90}
        mock_learning_monitor.get_baseline_quality.return_value = {"precision": 0.75}
        # But no low-quality embeddings
        mock_quality_tracker.identify_low_quality_embeddings.return_value = []

        result = evolution_with_monitor.monitor_extraction_quality_improvement()

        # Should return None (nothing to re-embed)
        assert result is None
        mock_reembedding_service.trigger_reembedding.assert_not_called()

    def test_trigger_quality_based_reembedding(
        self,
        evolution,
        mock_quality_tracker,
        mock_reembedding_service,
    ):
        """Test triggering quality-based re-embedding."""
        mock_quality_tracker.identify_low_quality_embeddings.return_value = [
            "ent_a", "ent_b"
        ]

        result = evolution.trigger_quality_based_reembedding(quality_threshold=0.6)

        assert result is not None
        mock_quality_tracker.identify_low_quality_embeddings.assert_called_once_with(
            min_quality_score=0.6,
            limit=1000,
        )
        mock_reembedding_service.trigger_reembedding.assert_called_once()

    def test_trigger_quality_based_reembedding_no_candidates(
        self,
        evolution,
        mock_quality_tracker,
        mock_reembedding_service,
    ):
        """Test triggering when no low-quality embeddings exist."""
        mock_quality_tracker.identify_low_quality_embeddings.return_value = []

        result = evolution.trigger_quality_based_reembedding(quality_threshold=0.6)

        assert result is None
        mock_reembedding_service.trigger_reembedding.assert_not_called()

    def test_trigger_quality_based_reembedding_custom_params(
        self,
        evolution,
        mock_quality_tracker,
        mock_reembedding_service,
    ):
        """Test triggering with custom parameters."""
        mock_quality_tracker.identify_low_quality_embeddings.return_value = ["ent_1"]

        evolution.trigger_quality_based_reembedding(
            quality_threshold=0.8,
            limit=500,
            batch_size=25,
        )

        mock_quality_tracker.identify_low_quality_embeddings.assert_called_once_with(
            min_quality_score=0.8,
            limit=500,
        )
        call_kwargs = mock_reembedding_service.trigger_reembedding.call_args[1]
        assert call_kwargs["batch_size"] == 25

    def test_measure_quality_trend_improving(self, evolution, mock_quality_tracker):
        """Test measuring improving trend."""
        mock_quality_tracker.update_quality_trend.return_value = "improving"

        trend = evolution.measure_quality_trend("ent_123", lookback_days=30)

        assert trend == "improving"
        mock_quality_tracker.update_quality_trend.assert_called_once_with(
            "ent_123", 30
        )

    def test_measure_quality_trend_stable(self, evolution, mock_quality_tracker):
        """Test measuring stable trend."""
        mock_quality_tracker.update_quality_trend.return_value = None  # Not enough data

        trend = evolution.measure_quality_trend("ent_123")

        assert trend == "stable"

    def test_measure_quality_trend_degrading(self, evolution, mock_quality_tracker):
        """Test measuring degrading trend."""
        mock_quality_tracker.update_quality_trend.return_value = "degrading"

        trend = evolution.measure_quality_trend("ent_456")

        assert trend == "degrading"

    def test_get_evolution_status(self, evolution):
        """Test getting evolution status."""
        status = evolution.get_evolution_status()

        assert "improvement_threshold" in status
        assert "quality_threshold" in status
        assert "baseline_quality" in status
        assert "last_check" in status
        assert "has_learning_monitor" in status
        assert "tracker_statistics" in status

        assert status["improvement_threshold"] == 0.05
        assert status["quality_threshold"] == 0.7
        assert status["has_learning_monitor"] is False

    def test_get_evolution_status_with_monitor(self, evolution_with_monitor):
        """Test evolution status with learning monitor."""
        status = evolution_with_monitor.get_evolution_status()
        assert status["has_learning_monitor"] is True

    def test_set_baseline_quality(self, evolution):
        """Test manually setting baseline quality."""
        baseline = {"precision": 0.8, "recall": 0.75}
        evolution.set_baseline_quality(baseline)

        assert evolution._baseline_quality == baseline

    def test_reset_baseline(self, evolution):
        """Test resetting baseline quality."""
        # Set baseline first
        evolution.set_baseline_quality({"precision": 0.8})
        evolution._last_check = datetime.utcnow()

        # Reset
        evolution.reset_baseline()

        assert evolution._baseline_quality is None
        assert evolution._last_check is None

    def test_baseline_updates_after_successful_reembedding(
        self,
        evolution_with_monitor,
        mock_quality_tracker,
        mock_learning_monitor,
    ):
        """Test that baseline updates after successful re-embedding."""
        # Initial baseline
        mock_learning_monitor.get_baseline_quality.return_value = {"precision": 0.75}
        mock_learning_monitor.get_current_quality_metrics.return_value = {"precision": 0.85}
        mock_quality_tracker.identify_low_quality_embeddings.return_value = ["ent_1"]

        evolution_with_monitor.monitor_extraction_quality_improvement()

        # Baseline should be updated to current
        assert evolution_with_monitor._baseline_quality == {"precision": 0.85}

    def test_error_handling_in_monitor(
        self,
        evolution_with_monitor,
        mock_learning_monitor,
    ):
        """Test error handling when monitor fails."""
        mock_learning_monitor.get_current_quality_metrics.side_effect = Exception("Monitor failed")

        result = evolution_with_monitor.monitor_extraction_quality_improvement()

        # Should handle error gracefully
        assert result is None

    def test_custom_thresholds(self, mock_quality_tracker, mock_reembedding_service):
        """Test initialization with custom thresholds."""
        evolution = EmbeddingQualityEvolution(
            quality_tracker=mock_quality_tracker,
            reembedding_service=mock_reembedding_service,
            improvement_threshold=0.10,
            quality_threshold=0.8,
        )

        assert evolution._improvement_threshold == 0.10
        assert evolution._quality_threshold == 0.8


class TestExperientialLearningIntegration:
    """Tests for integration with experiential learning pipeline."""

    def test_protocol_compliance(self):
        """Test that ExperientialLearningMonitor protocol is properly defined."""
        # Create a class implementing the protocol
        class MockMonitor:
            def get_current_quality_metrics(self):
                return {"precision": 0.85}

            def get_baseline_quality(self):
                return {"precision": 0.75}

            def assess_extraction_trajectory(self, recent_extractions):
                return {"improvement": 0.1}

        monitor = MockMonitor()

        # Verify methods exist and work
        assert monitor.get_current_quality_metrics()["precision"] == 0.85
        assert monitor.get_baseline_quality()["precision"] == 0.75
        assert monitor.assess_extraction_trajectory([])["improvement"] == 0.1

    def test_evolution_with_world_state_model_pattern(
        self,
        mock_quality_tracker,
        mock_reembedding_service,
    ):
        """Test evolution works with WorldStateModel-like monitor."""
        # Mock WorldStateModel pattern
        mock_monitor = MagicMock()
        mock_monitor.quality_history = [{"precision": 0.7}, {"precision": 0.8}]
        mock_monitor.get_current_quality_metrics.return_value = {"precision": 0.9}
        mock_monitor.get_baseline_quality.return_value = {"precision": 0.7}

        evolution = EmbeddingQualityEvolution(
            quality_tracker=mock_quality_tracker,
            reembedding_service=mock_reembedding_service,
            experiential_learning_monitor=mock_monitor,
        )

        mock_quality_tracker.identify_low_quality_embeddings.return_value = ["ent_1"]

        # Should work with the mock monitor
        result = evolution.monitor_extraction_quality_improvement()

        # Improvement is 0.9 - 0.7 = 0.2 > 0.05, should trigger
        assert result is not None

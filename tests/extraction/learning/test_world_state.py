"""Tests for World State Model.

Tests quality metrics computation, trajectory tracking, and success/failure
pattern identification per SEAgent World State Model research.

Research Reference:
- SEAgent (2508.04700v2): World State Model for step-wise trajectory assessment

Quality Gates:
- Quality metrics must accurately assess extraction quality
- Trajectory tracking must identify success/failure patterns
- Quality progression must show >5% improvement over 50 documents
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from futurnal.learning.world_state import (
    QualityMetrics,
    ExtractionTrajectory,
    WorldStateAssessor,
    ENTITY_COVERAGE_WEIGHT,
    RELATION_COVERAGE_WEIGHT,
    TEMPORAL_AWARENESS_WEIGHT,
    CONFIDENCE_WEIGHT,
    CONSISTENCY_WEIGHT,
)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_quality_metrics_creation(self):
        """Test basic QualityMetrics creation."""
        metrics = QualityMetrics(
            entity_count=5,
            relation_count=3,
            temporal_coverage=0.8,
            confidence_avg=0.9,
            consistency_score=1.0,
        )
        assert metrics.entity_count == 5
        assert metrics.relation_count == 3
        assert metrics.temporal_coverage == 0.8
        assert metrics.confidence_avg == 0.9
        assert metrics.consistency_score == 1.0

    def test_weighted_quality_computation(self):
        """Test weighted quality score computation."""
        metrics = QualityMetrics(
            entity_count=10,  # Normalized to 1.0
            relation_count=5,  # Normalized to 1.0
            temporal_coverage=1.0,
            confidence_avg=1.0,
            consistency_score=1.0,
        )
        # All metrics maxed out should give 1.0
        quality = metrics.compute_weighted_quality()
        assert quality == pytest.approx(1.0, abs=0.01)

    def test_weighted_quality_partial_scores(self):
        """Test weighted quality with partial scores."""
        metrics = QualityMetrics(
            entity_count=5,  # Normalized to 0.5
            relation_count=2,  # Normalized to 0.4
            temporal_coverage=0.5,
            confidence_avg=0.5,
            consistency_score=0.5,
        )
        quality = metrics.compute_weighted_quality()
        # Expected: 0.2*0.5 + 0.2*0.4 + 0.3*0.5 + 0.15*0.5 + 0.15*0.5
        expected = (0.2 * 0.5) + (0.2 * 0.4) + (0.3 * 0.5) + (0.15 * 0.5) + (0.15 * 0.5)
        assert quality == pytest.approx(expected, abs=0.01)

    def test_quality_metrics_to_dict(self):
        """Test QualityMetrics serialization."""
        metrics = QualityMetrics(
            entity_count=5,
            relation_count=3,
            temporal_coverage=0.8,
            confidence_avg=0.9,
            consistency_score=1.0,
            document_id="test_doc",
        )
        data = metrics.to_dict()
        assert data["entity_count"] == 5
        assert data["relation_count"] == 3
        assert data["temporal_coverage"] == 0.8
        assert data["document_id"] == "test_doc"
        assert "weighted_quality" in data

    def test_weight_sum_is_one(self):
        """Verify quality weights sum to 1.0."""
        total = (
            ENTITY_COVERAGE_WEIGHT +
            RELATION_COVERAGE_WEIGHT +
            TEMPORAL_AWARENESS_WEIGHT +
            CONFIDENCE_WEIGHT +
            CONSISTENCY_WEIGHT
        )
        assert total == pytest.approx(1.0, abs=0.001)


class TestExtractionTrajectory:
    """Tests for ExtractionTrajectory dataclass."""

    def test_trajectory_success_detection(self):
        """Test trajectory correctly identifies success."""
        trajectory = ExtractionTrajectory(
            document_id="doc1",
            quality_before=0.5,
            quality_after=0.7,
        )
        assert trajectory.success is True
        assert trajectory.improvement == pytest.approx(0.2)
        assert trajectory.improvement_percentage == pytest.approx(40.0)

    def test_trajectory_failure_detection(self):
        """Test trajectory correctly identifies failure."""
        trajectory = ExtractionTrajectory(
            document_id="doc1",
            quality_before=0.7,
            quality_after=0.5,
        )
        assert trajectory.success is False
        assert trajectory.improvement == pytest.approx(-0.2)

    def test_trajectory_no_change(self):
        """Test trajectory with no quality change."""
        trajectory = ExtractionTrajectory(
            document_id="doc1",
            quality_before=0.5,
            quality_after=0.5,
        )
        assert trajectory.success is False  # No improvement = not success
        assert trajectory.improvement == 0.0

    def test_trajectory_patterns_tracking(self):
        """Test trajectory tracks applied patterns."""
        trajectory = ExtractionTrajectory(
            document_id="doc1",
            quality_before=0.5,
            quality_after=0.7,
            patterns_applied=["entity:Person", "relation:works_at"],
        )
        assert len(trajectory.patterns_applied) == 2
        assert "entity:Person" in trajectory.patterns_applied

    def test_trajectory_to_dict(self):
        """Test trajectory serialization."""
        trajectory = ExtractionTrajectory(
            document_id="doc1",
            quality_before=0.5,
            quality_after=0.7,
        )
        data = trajectory.to_dict()
        assert data["document_id"] == "doc1"
        assert data["success"] is True
        assert "improvement" in data
        assert "improvement_percentage" in data


class TestWorldStateAssessor:
    """Tests for WorldStateAssessor class."""

    def test_assessor_creation(self):
        """Test WorldStateAssessor creation."""
        assessor = WorldStateAssessor()
        assert assessor.trajectory_capacity == 1000
        assert len(assessor.trajectories) == 0

    def test_assess_extraction_direct_values(self):
        """Test assess_extraction with direct values."""
        assessor = WorldStateAssessor()
        metrics = assessor.assess_extraction(
            entity_count=5,
            relation_count=3,
            confidence=0.8,
            document_id="test_doc",
        )
        assert metrics.entity_count == 5
        assert metrics.relation_count == 3
        assert metrics.confidence_avg == 0.8
        assert metrics.document_id == "test_doc"

    def test_assess_extraction_with_extraction_result(self):
        """Test assess_extraction with mock extraction result."""
        assessor = WorldStateAssessor()

        # Create mock extraction result
        mock_result = MagicMock()
        mock_result.entity_count = 8
        mock_result.relation_count = 4
        mock_result.confidence = 0.85

        metrics = assessor.assess_extraction(
            extraction_result=mock_result,
            document_id="test_doc",
        )
        assert metrics.entity_count == 8
        assert metrics.relation_count == 4
        assert metrics.confidence_avg == 0.85

    def test_record_trajectory(self):
        """Test trajectory recording."""
        assessor = WorldStateAssessor()
        trajectory = assessor.record_trajectory(
            document_id="doc1",
            quality_before=0.5,
            quality_after=0.7,
            patterns_applied=["pattern1"],
        )
        assert trajectory.document_id == "doc1"
        assert trajectory.success is True
        assert len(assessor.trajectories) == 1

    def test_trajectory_capacity_pruning(self):
        """Test trajectories are pruned when over capacity."""
        assessor = WorldStateAssessor(trajectory_capacity=5)

        for i in range(10):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.6,
            )

        assert len(assessor.trajectories) == 5
        # Should keep most recent
        assert assessor.trajectories[-1].document_id == "doc9"

    def test_identify_success_patterns(self):
        """Test success pattern identification."""
        assessor = WorldStateAssessor()

        # Record multiple successful trajectories with same pattern
        for i in range(5):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.8,  # High quality = success
                patterns_applied=["good_pattern"],
            )

        success_patterns = assessor.identify_success_patterns(min_occurrences=3)
        assert "good_pattern" in success_patterns

    def test_identify_failure_patterns(self):
        """Test failure pattern identification."""
        assessor = WorldStateAssessor()

        # Record multiple failed trajectories with same pattern
        for i in range(5):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.2,  # Low quality = failure
                patterns_applied=["bad_pattern"],
            )

        failure_patterns = assessor.identify_failure_patterns(min_occurrences=3)
        assert "bad_pattern" in failure_patterns

    def test_compute_quality_progression_insufficient_data(self):
        """Test quality progression with insufficient data."""
        assessor = WorldStateAssessor()

        for i in range(5):  # Less than window_size * 2
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.6,
            )

        progression = assessor.compute_quality_progression()
        assert progression.get("insufficient_data") is True

    def test_compute_quality_progression_with_improvement(self):
        """Test quality progression showing improvement."""
        assessor = WorldStateAssessor()

        # First batch: lower quality
        for i in range(15):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.4,
                quality_after=0.5,
            )

        # Second batch: higher quality
        for i in range(15, 30):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.7,
            )

        progression = assessor.compute_quality_progression(window_size=10)
        assert progression.get("insufficient_data") is False
        assert progression["last_window_avg"] > progression["first_window_avg"]
        assert progression["improvement"] > 0

    def test_get_quality_summary(self):
        """Test quality summary generation."""
        assessor = WorldStateAssessor()

        for i in range(5):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.6,
            )

        summary = assessor.get_quality_summary()
        assert summary["status"] == "active"
        assert summary["trajectories"] == 5
        assert "success_rate" in summary
        assert "avg_quality" in summary

    def test_clear_trajectories(self):
        """Test clearing trajectories."""
        assessor = WorldStateAssessor()

        for i in range(5):
            assessor.record_trajectory(
                document_id=f"doc{i}",
                quality_before=0.5,
                quality_after=0.6,
            )

        count = assessor.clear_trajectories()
        assert count == 5
        assert len(assessor.trajectories) == 0


class TestQualityGateCompliance:
    """Tests specifically for quality gate compliance."""

    def test_temporal_weight_is_highest(self):
        """Verify temporal weight is highest (30%) per Step 06 spec."""
        assert TEMPORAL_AWARENESS_WEIGHT == 0.30
        assert TEMPORAL_AWARENESS_WEIGHT > ENTITY_COVERAGE_WEIGHT
        assert TEMPORAL_AWARENESS_WEIGHT > RELATION_COVERAGE_WEIGHT
        assert TEMPORAL_AWARENESS_WEIGHT > CONFIDENCE_WEIGHT
        assert TEMPORAL_AWARENESS_WEIGHT > CONSISTENCY_WEIGHT

    def test_five_percent_improvement_detection(self):
        """Test detection of >5% quality improvement."""
        assessor = WorldStateAssessor()

        # First window: quality 0.5
        for i in range(20):
            assessor.record_trajectory(
                document_id=f"early_{i}",
                quality_before=0.4,
                quality_after=0.5,
            )

        # Second window: quality 0.6 (20% improvement)
        for i in range(30):
            assessor.record_trajectory(
                document_id=f"late_{i}",
                quality_before=0.5,
                quality_after=0.6,
            )

        progression = assessor.compute_quality_progression(window_size=10)
        assert progression["passes_quality_gate"] is True
        assert progression["improvement_percentage"] >= 5.0

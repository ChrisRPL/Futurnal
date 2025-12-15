"""
World State Model for Experiential Learning

Implements SEAgent World State Model for step-wise trajectory assessment
and quality metric computation.

Research Foundation:
- SEAgent (2508.04700v2): World State Model for step-wise trajectory assessment
- Training-Free GRPO (2510.08191v1): Quality assessment without parameter updates

Quality Gates (.cursor/rules/quality-gates.mdc):
- Temporal extraction: >85% accuracy
- Schema evolution: >90% semantic alignment
- Experiential learning: demonstrable quality improvement (>5% over 50 docs)

Option B Compliance:
- Ghost model frozen (no parameter updates)
- Quality metrics guide token prior updates, NOT model weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.extraction.schema.evolution import ExtractionResult
    from futurnal.extraction.temporal.models import (
        TemporalExtractionResult,
        ValidationResult,
    )

logger = logging.getLogger(__name__)


# Quality metric weights per Step 06 specification
ENTITY_COVERAGE_WEIGHT = 0.20
RELATION_COVERAGE_WEIGHT = 0.20
TEMPORAL_AWARENESS_WEIGHT = 0.30  # CRITICAL for Phase 3 causal inference
CONFIDENCE_WEIGHT = 0.15
CONSISTENCY_WEIGHT = 0.15

# Normalization targets
TARGET_ENTITY_COUNT = 10  # Entities per document target
TARGET_RELATION_COUNT = 5  # Relations per document target


@dataclass
class QualityMetrics:
    """Quality assessment metrics per SEAgent World State Model.

    Implements weighted quality scoring for extraction assessment:
    - Entity coverage: 20%
    - Relation coverage: 20%
    - Temporal awareness: 30% (CRITICAL for Phase 3 causal inference)
    - Confidence: 15%
    - Consistency: 15%

    Research Reference: SEAgent Section 3 - "World State Model for step-wise
    trajectory assessment"
    """

    entity_count: int = 0
    relation_count: int = 0
    temporal_coverage: float = 0.0  # 0-1, fraction of temporally-grounded entities
    confidence_avg: float = 0.0  # Average extraction confidence
    consistency_score: float = 1.0  # From ValidationResult (default to 1.0 = consistent)

    # Optional metadata
    document_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def _normalize_entity_count(self) -> float:
        """Normalize entity count to 0-1 scale."""
        return min(self.entity_count / TARGET_ENTITY_COUNT, 1.0)

    def _normalize_relation_count(self) -> float:
        """Normalize relation count to 0-1 scale."""
        return min(self.relation_count / TARGET_RELATION_COUNT, 1.0)

    def compute_weighted_quality(self) -> float:
        """Compute overall quality score with research-defined weights.

        Returns:
            float: Quality score between 0-1

        Formula:
            quality = (0.20 * entity_score) +
                     (0.20 * relation_score) +
                     (0.30 * temporal_coverage) +
                     (0.15 * confidence_avg) +
                     (0.15 * consistency_score)
        """
        return (
            ENTITY_COVERAGE_WEIGHT * self._normalize_entity_count()
            + RELATION_COVERAGE_WEIGHT * self._normalize_relation_count()
            + TEMPORAL_AWARENESS_WEIGHT * self.temporal_coverage
            + CONFIDENCE_WEIGHT * self.confidence_avg
            + CONSISTENCY_WEIGHT * self.consistency_score
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "temporal_coverage": self.temporal_coverage,
            "confidence_avg": self.confidence_avg,
            "consistency_score": self.consistency_score,
            "weighted_quality": self.compute_weighted_quality(),
            "document_id": self.document_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExtractionTrajectory:
    """Track quality progression over documents for success/failure analysis.

    Per SEAgent: Step-wise trajectory for identifying success and failure
    patterns that guide experiential learning.

    Attributes:
        document_id: Unique identifier for the processed document
        quality_before: Quality score before applying experiential knowledge
        quality_after: Quality score after extraction with priors
        success: Whether quality improved (quality_after > quality_before)
        patterns_applied: List of experiential patterns used in this extraction
        metrics_before: Detailed metrics before extraction
        metrics_after: Detailed metrics after extraction
    """

    document_id: str
    quality_before: float
    quality_after: float
    success: bool = field(init=False)
    patterns_applied: List[str] = field(default_factory=list)
    metrics_before: Optional[QualityMetrics] = None
    metrics_after: Optional[QualityMetrics] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Compute success based on quality improvement."""
        self.success = self.quality_after > self.quality_before

    @property
    def improvement(self) -> float:
        """Compute quality improvement (can be negative)."""
        return self.quality_after - self.quality_before

    @property
    def improvement_percentage(self) -> float:
        """Compute percentage improvement."""
        if self.quality_before == 0:
            return 100.0 if self.quality_after > 0 else 0.0
        return ((self.quality_after - self.quality_before) / self.quality_before) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "document_id": self.document_id,
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "success": self.success,
            "improvement": self.improvement,
            "improvement_percentage": self.improvement_percentage,
            "patterns_applied": self.patterns_applied,
            "metrics_before": self.metrics_before.to_dict() if self.metrics_before else None,
            "metrics_after": self.metrics_after.to_dict() if self.metrics_after else None,
            "timestamp": self.timestamp.isoformat(),
        }


class WorldStateAssessor:
    """Complete implementation of SEAgent World State Model.

    Assesses extraction quality using weighted metrics and tracks trajectories
    for success/failure pattern identification. This is the core component for
    experiential learning quality feedback.

    Research Reference:
    - SEAgent (2508.04700v2) Section 3: "World State Model for step-wise
      trajectory assessment"

    Quality Gates:
    - Must demonstrate quality improvement >5% over 50 documents
    - Ghost model parameters must remain unchanged

    Example:
        >>> assessor = WorldStateAssessor()
        >>> metrics = assessor.assess_extraction(extraction_result, temporal_result, validation)
        >>> print(f"Quality: {metrics.compute_weighted_quality():.2f}")
    """

    def __init__(
        self,
        trajectory_capacity: int = 1000,
        success_pattern_threshold: float = 0.7,
        failure_pattern_threshold: float = 0.3,
    ):
        """Initialize World State Assessor.

        Args:
            trajectory_capacity: Maximum trajectories to store in memory
            success_pattern_threshold: Min quality for success pattern identification
            failure_pattern_threshold: Max quality for failure pattern identification
        """
        self.trajectories: List[ExtractionTrajectory] = []
        self.trajectory_capacity = trajectory_capacity
        self.success_pattern_threshold = success_pattern_threshold
        self.failure_pattern_threshold = failure_pattern_threshold

        # Pattern tracking
        self.success_patterns: Dict[str, int] = {}  # pattern -> success count
        self.failure_patterns: Dict[str, int] = {}  # pattern -> failure count

    def assess_extraction(
        self,
        extraction_result: Optional[ExtractionResult] = None,
        temporal_result: Optional[TemporalExtractionResult] = None,
        consistency_result: Optional[ValidationResult] = None,
        entity_count: Optional[int] = None,
        relation_count: Optional[int] = None,
        confidence: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> QualityMetrics:
        """Assess extraction quality using weighted metrics.

        Can be called with either:
        1. Full extraction results (extraction_result, temporal_result, consistency_result)
        2. Direct metric values (entity_count, relation_count, etc.)

        Args:
            extraction_result: Result from schema evolution extraction
            temporal_result: Result from temporal extraction pipeline
            consistency_result: Result from temporal consistency validation
            entity_count: Direct entity count (if not using extraction_result)
            relation_count: Direct relation count (if not using extraction_result)
            confidence: Direct confidence value (if not using extraction_result)
            document_id: Optional document identifier

        Returns:
            QualityMetrics: Computed quality metrics with weighted score
        """
        # Extract values from result objects if provided
        if extraction_result is not None:
            entity_count = entity_count or getattr(extraction_result, "entity_count", 0)
            relation_count = relation_count or getattr(extraction_result, "relation_count", 0)
            confidence = confidence or getattr(extraction_result, "confidence", 0.5)

        # Compute temporal coverage from temporal result
        temporal_coverage = 0.0
        if temporal_result is not None:
            events = getattr(temporal_result, "events", [])
            if events:
                # Count events with timestamps
                events_with_timestamps = sum(
                    1 for e in events if getattr(e, "timestamp", None) is not None
                )
                temporal_coverage = events_with_timestamps / len(events)
            else:
                # Check temporal markers as fallback
                markers = getattr(temporal_result, "temporal_markers", [])
                if markers:
                    temporal_coverage = min(len(markers) / 5, 1.0)  # Normalize to target

        # Get consistency score from validation result
        consistency_score = 1.0  # Default to fully consistent
        if consistency_result is not None:
            if getattr(consistency_result, "valid", True):
                consistency_score = 1.0
            else:
                # Reduce score based on error count
                errors = getattr(consistency_result, "errors", [])
                consistency_score = max(0.0, 1.0 - (len(errors) * 0.1))

        return QualityMetrics(
            entity_count=entity_count or 0,
            relation_count=relation_count or 0,
            temporal_coverage=temporal_coverage,
            confidence_avg=confidence or 0.5,
            consistency_score=consistency_score,
            document_id=document_id,
        )

    def record_trajectory(
        self,
        document_id: str,
        quality_before: float,
        quality_after: float,
        patterns_applied: Optional[List[str]] = None,
        metrics_before: Optional[QualityMetrics] = None,
        metrics_after: Optional[QualityMetrics] = None,
    ) -> ExtractionTrajectory:
        """Record extraction trajectory for experiential learning.

        Per SEAgent: Track success/failure patterns for GRPO on successes
        and adversarial imitation on failures.

        Args:
            document_id: Unique document identifier
            quality_before: Quality score before applying experiential knowledge
            quality_after: Quality score after extraction
            patterns_applied: List of experiential patterns used
            metrics_before: Detailed quality metrics before
            metrics_after: Detailed quality metrics after

        Returns:
            ExtractionTrajectory: Recorded trajectory
        """
        trajectory = ExtractionTrajectory(
            document_id=document_id,
            quality_before=quality_before,
            quality_after=quality_after,
            patterns_applied=patterns_applied or [],
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )

        # Store trajectory
        self.trajectories.append(trajectory)

        # Prune if over capacity
        if len(self.trajectories) > self.trajectory_capacity:
            self.trajectories = self.trajectories[-self.trajectory_capacity:]

        # Update pattern tracking
        self._update_pattern_counts(trajectory)

        logger.debug(
            f"Recorded trajectory for {document_id}: "
            f"quality {quality_before:.3f} -> {quality_after:.3f} "
            f"({'success' if trajectory.success else 'failure'})"
        )

        return trajectory

    def _update_pattern_counts(self, trajectory: ExtractionTrajectory) -> None:
        """Update success/failure pattern counts from trajectory."""
        for pattern in trajectory.patterns_applied:
            if trajectory.success:
                self.success_patterns[pattern] = self.success_patterns.get(pattern, 0) + 1
            else:
                self.failure_patterns[pattern] = self.failure_patterns.get(pattern, 0) + 1

    def identify_success_patterns(
        self,
        trajectories: Optional[List[ExtractionTrajectory]] = None,
        min_occurrences: int = 3,
    ) -> List[str]:
        """Identify patterns that led to successful extractions.

        Per SEAgent: GRPO on successful patterns to reinforce what works.

        Args:
            trajectories: Trajectories to analyze (uses stored if None)
            min_occurrences: Minimum times pattern must appear to be identified

        Returns:
            List of pattern descriptions that correlate with success
        """
        trajectories = trajectories or self.trajectories

        # Count patterns in successful trajectories
        pattern_counts: Dict[str, int] = {}
        for traj in trajectories:
            if traj.success and traj.quality_after >= self.success_pattern_threshold:
                for pattern in traj.patterns_applied:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Filter by minimum occurrences and sort by count
        success_patterns = [
            pattern for pattern, count in pattern_counts.items()
            if count >= min_occurrences
        ]

        return sorted(
            success_patterns,
            key=lambda p: pattern_counts[p],
            reverse=True,
        )

    def identify_failure_patterns(
        self,
        trajectories: Optional[List[ExtractionTrajectory]] = None,
        min_occurrences: int = 3,
    ) -> List[str]:
        """Identify patterns that led to failed extractions.

        Per SEAgent: Adversarial imitation on failure patterns to learn
        what to avoid.

        Args:
            trajectories: Trajectories to analyze (uses stored if None)
            min_occurrences: Minimum times pattern must appear to be identified

        Returns:
            List of pattern descriptions that correlate with failure
        """
        trajectories = trajectories or self.trajectories

        # Count patterns in failed trajectories
        pattern_counts: Dict[str, int] = {}
        for traj in trajectories:
            if not traj.success or traj.quality_after <= self.failure_pattern_threshold:
                for pattern in traj.patterns_applied:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Filter by minimum occurrences and sort by count
        failure_patterns = [
            pattern for pattern, count in pattern_counts.items()
            if count >= min_occurrences
        ]

        return sorted(
            failure_patterns,
            key=lambda p: pattern_counts[p],
            reverse=True,
        )

    def compute_quality_progression(
        self,
        trajectories: Optional[List[ExtractionTrajectory]] = None,
        window_size: int = 10,
    ) -> Dict[str, float]:
        """Compute quality progression metrics over trajectories.

        Used to validate >5% improvement over 50 documents quality gate.

        Args:
            trajectories: Trajectories to analyze (uses stored if None)
            window_size: Size of comparison windows

        Returns:
            Dict with progression metrics including improvement percentage
        """
        trajectories = trajectories or self.trajectories

        if len(trajectories) < window_size * 2:
            return {
                "insufficient_data": True,
                "trajectories_needed": window_size * 2,
                "trajectories_available": len(trajectories),
            }

        # Compare first and last windows
        first_window = trajectories[:window_size]
        last_window = trajectories[-window_size:]

        first_avg = sum(t.quality_after for t in first_window) / window_size
        last_avg = sum(t.quality_after for t in last_window) / window_size

        improvement = last_avg - first_avg
        improvement_pct = (improvement / first_avg * 100) if first_avg > 0 else 0.0

        # Calculate success rates
        first_success_rate = sum(1 for t in first_window if t.success) / window_size
        last_success_rate = sum(1 for t in last_window if t.success) / window_size

        return {
            "insufficient_data": False,
            "first_window_avg": first_avg,
            "last_window_avg": last_avg,
            "improvement": improvement,
            "improvement_percentage": improvement_pct,
            "passes_quality_gate": improvement_pct >= 5.0,  # >5% target
            "first_success_rate": first_success_rate,
            "last_success_rate": last_success_rate,
            "total_trajectories": len(trajectories),
        }

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of current quality state.

        Returns:
            Dict with quality statistics and trends
        """
        if not self.trajectories:
            return {"status": "no_data", "trajectories": 0}

        qualities = [t.quality_after for t in self.trajectories]
        successes = sum(1 for t in self.trajectories if t.success)

        return {
            "status": "active",
            "trajectories": len(self.trajectories),
            "success_rate": successes / len(self.trajectories),
            "avg_quality": sum(qualities) / len(qualities),
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "top_success_patterns": self.identify_success_patterns()[:5],
            "top_failure_patterns": self.identify_failure_patterns()[:5],
            "progression": self.compute_quality_progression(),
        }

    def clear_trajectories(self) -> int:
        """Clear stored trajectories.

        Returns:
            Number of trajectories cleared
        """
        count = len(self.trajectories)
        self.trajectories = []
        self.success_patterns = {}
        self.failure_patterns = {}
        logger.info(f"Cleared {count} trajectories from World State Assessor")
        return count

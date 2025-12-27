"""Insight Quality Gates for Phase 2F.

Validates insight quality before delivery to ensure users receive
only valuable, well-evidenced insights.

Research Foundation:
- Bradford Hill criteria for causal inference
- Statistical significance thresholds
- User feedback integration for personalized quality assessment

Option B Compliance:
- No model parameter updates
- Quality thresholds based on evidence, not learned weights
- Ghost model FROZEN
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.insights.emergent_insights import EmergentInsight

logger = logging.getLogger(__name__)


class QualityGateResult(str, Enum):
    """Result of quality gate validation."""

    PASSED = "passed"  # Meets all quality criteria
    FAILED = "failed"  # Does not meet minimum criteria
    PENDING = "pending"  # Needs more evidence
    REVIEW = "review"  # Borderline, needs manual review


class QualityDimension(str, Enum):
    """Dimensions of insight quality."""

    CONFIDENCE = "confidence"  # Statistical confidence
    EVIDENCE = "evidence"  # Supporting evidence count
    NOVELTY = "novelty"  # How new/interesting is this
    RELEVANCE = "relevance"  # User interest alignment
    ACTIONABILITY = "actionability"  # Can user act on this
    TEMPORAL = "temporal"  # Temporal grounding quality


@dataclass
class QualityScore:
    """Quality score for a single dimension.

    Attributes:
        dimension: Which quality dimension
        score: Score from 0 to 1
        reason: Explanation for the score
        threshold: Minimum threshold for this dimension
        passed: Whether score meets threshold
    """

    dimension: QualityDimension
    score: float
    reason: str = ""
    threshold: float = 0.5
    passed: bool = False

    def __post_init__(self):
        self.passed = self.score >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "reason": self.reason,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass
class QualityReport:
    """Complete quality assessment report for an insight.

    Attributes:
        insight_id: ID of the insight assessed
        result: Overall quality gate result
        overall_score: Combined quality score (0-1)
        dimension_scores: Individual dimension scores
        recommendation: What to do with this insight
        created_at: When assessment was performed
    """

    insight_id: str
    result: QualityGateResult
    overall_score: float = 0.0
    dimension_scores: List[QualityScore] = field(default_factory=list)
    recommendation: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "insight_id": self.insight_id,
            "result": self.result.value,
            "overall_score": self.overall_score,
            "dimension_scores": [s.to_dict() for s in self.dimension_scores],
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
        }

    def to_natural_language(self) -> str:
        """Convert to natural language summary."""
        lines = [
            f"Quality Assessment: {self.result.value.upper()}",
            f"Overall Score: {self.overall_score:.0%}",
        ]

        passed = [s for s in self.dimension_scores if s.passed]
        failed = [s for s in self.dimension_scores if not s.passed]

        if passed:
            lines.append(f"\nPassed ({len(passed)}):")
            for s in passed:
                lines.append(f"  + {s.dimension.value}: {s.score:.0%}")

        if failed:
            lines.append(f"\nFailed ({len(failed)}):")
            for s in failed:
                lines.append(f"  - {s.dimension.value}: {s.score:.0%} (needs {s.threshold:.0%})")

        if self.recommendation:
            lines.append(f"\nRecommendation: {self.recommendation}")

        return "\n".join(lines)


@dataclass
class QualityGateConfig:
    """Configuration for quality gates.

    Attributes:
        min_confidence: Minimum confidence threshold
        min_evidence_count: Minimum evidence points needed
        min_novelty: Minimum novelty score
        min_relevance: Minimum relevance score
        min_overall: Minimum overall score to pass
        require_temporal: Whether temporal grounding is required
        strict_mode: If True, all dimensions must pass
    """

    min_confidence: float = 0.6
    min_evidence_count: int = 3
    min_novelty: float = 0.3
    min_relevance: float = 0.4
    min_overall: float = 0.5
    require_temporal: bool = True
    strict_mode: bool = False


class InsightQualityGate:
    """Validates insight quality before delivery.

    Ensures users only receive valuable, well-evidenced insights by
    applying multi-dimensional quality assessment.

    Key Quality Dimensions:
    1. Confidence - Statistical confidence of the insight
    2. Evidence - Number of supporting evidence points
    3. Novelty - How new or interesting is this insight
    4. Relevance - Alignment with user interests
    5. Actionability - Can the user act on this insight
    6. Temporal - Quality of temporal grounding

    Usage:
        gate = InsightQualityGate()
        report = gate.validate(insight)
        if report.result == QualityGateResult.PASSED:
            # Deliver insight
        else:
            # Filter out or queue for more evidence
    """

    def __init__(
        self,
        config: Optional[QualityGateConfig] = None,
        user_interests: Optional[List[str]] = None,
    ):
        """Initialize quality gate.

        Args:
            config: Quality threshold configuration
            user_interests: Optional list of user interest keywords
        """
        self.config = config or QualityGateConfig()
        self.user_interests = user_interests or []

        logger.info(
            f"InsightQualityGate initialized "
            f"(min_confidence={self.config.min_confidence}, "
            f"min_evidence={self.config.min_evidence_count})"
        )

    def validate(self, insight: "EmergentInsight") -> QualityReport:
        """Validate an insight against quality gates.

        Args:
            insight: The insight to validate

        Returns:
            QualityReport with assessment details
        """
        dimension_scores = []

        # 1. Confidence score
        confidence_score = self._assess_confidence(insight)
        dimension_scores.append(confidence_score)

        # 2. Evidence score
        evidence_score = self._assess_evidence(insight)
        dimension_scores.append(evidence_score)

        # 3. Novelty score
        novelty_score = self._assess_novelty(insight)
        dimension_scores.append(novelty_score)

        # 4. Relevance score
        relevance_score = self._assess_relevance(insight)
        dimension_scores.append(relevance_score)

        # 5. Actionability score
        actionability_score = self._assess_actionability(insight)
        dimension_scores.append(actionability_score)

        # 6. Temporal score
        temporal_score = self._assess_temporal(insight)
        dimension_scores.append(temporal_score)

        # Calculate overall score (weighted average)
        weights = {
            QualityDimension.CONFIDENCE: 0.25,
            QualityDimension.EVIDENCE: 0.20,
            QualityDimension.NOVELTY: 0.15,
            QualityDimension.RELEVANCE: 0.20,
            QualityDimension.ACTIONABILITY: 0.10,
            QualityDimension.TEMPORAL: 0.10,
        }

        overall_score = sum(
            s.score * weights.get(s.dimension, 0.15)
            for s in dimension_scores
        )

        # Determine result
        result = self._determine_result(dimension_scores, overall_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            result, dimension_scores, overall_score
        )

        report = QualityReport(
            insight_id=getattr(insight, "insight_id", str(id(insight))),
            result=result,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            recommendation=recommendation,
        )

        logger.debug(
            f"Quality assessment: {report.result.value} "
            f"(score={overall_score:.2f})"
        )

        return report

    def _assess_confidence(self, insight: "EmergentInsight") -> QualityScore:
        """Assess confidence dimension."""
        confidence = getattr(insight, "confidence", 0.5)

        if confidence >= 0.9:
            reason = "Very high confidence based on strong evidence"
        elif confidence >= 0.7:
            reason = "Good confidence with solid evidence"
        elif confidence >= 0.5:
            reason = "Moderate confidence, more evidence may help"
        else:
            reason = "Low confidence, needs more supporting data"

        return QualityScore(
            dimension=QualityDimension.CONFIDENCE,
            score=confidence,
            reason=reason,
            threshold=self.config.min_confidence,
        )

    def _assess_evidence(self, insight: "EmergentInsight") -> QualityScore:
        """Assess evidence dimension."""
        # Count evidence sources
        evidence_count = 0

        # Check for evidence in various attributes
        if hasattr(insight, "supporting_evidence"):
            evidence_count += len(getattr(insight, "supporting_evidence", []))
        if hasattr(insight, "source_nodes"):
            evidence_count += len(getattr(insight, "source_nodes", []))
        if hasattr(insight, "evidence_documents"):
            evidence_count += len(getattr(insight, "evidence_documents", []))

        # Normalize to 0-1 score
        max_expected = self.config.min_evidence_count * 2
        score = min(1.0, evidence_count / max_expected)

        if evidence_count >= self.config.min_evidence_count * 2:
            reason = f"Strong evidence base ({evidence_count} sources)"
        elif evidence_count >= self.config.min_evidence_count:
            reason = f"Adequate evidence ({evidence_count} sources)"
        else:
            reason = f"Insufficient evidence ({evidence_count}/{self.config.min_evidence_count} needed)"

        threshold = self.config.min_evidence_count / max_expected

        return QualityScore(
            dimension=QualityDimension.EVIDENCE,
            score=score,
            reason=reason,
            threshold=threshold,
        )

    def _assess_novelty(self, insight: "EmergentInsight") -> QualityScore:
        """Assess novelty dimension."""
        # Check for novelty indicators
        novelty = getattr(insight, "novelty_score", None)

        if novelty is None:
            # Estimate based on insight type and properties
            insight_type = getattr(insight, "insight_type", "").lower()

            if "anomaly" in insight_type or "unexpected" in insight_type:
                novelty = 0.8
            elif "pattern" in insight_type:
                novelty = 0.6
            elif "correlation" in insight_type:
                novelty = 0.5
            else:
                novelty = 0.4

        if novelty >= 0.7:
            reason = "Highly novel discovery"
        elif novelty >= 0.5:
            reason = "Moderately interesting finding"
        else:
            reason = "Expected or common pattern"

        return QualityScore(
            dimension=QualityDimension.NOVELTY,
            score=novelty,
            reason=reason,
            threshold=self.config.min_novelty,
        )

    def _assess_relevance(self, insight: "EmergentInsight") -> QualityScore:
        """Assess relevance dimension."""
        # Check for relevance score or compute from user interests
        relevance = getattr(insight, "relevance_score", None)

        if relevance is None and self.user_interests:
            # Simple keyword matching against insight content
            title = getattr(insight, "title", "").lower()
            description = getattr(insight, "description", "").lower()
            content = f"{title} {description}"

            matches = sum(
                1 for interest in self.user_interests
                if interest.lower() in content
            )

            relevance = min(1.0, matches / max(1, len(self.user_interests)))
        elif relevance is None:
            relevance = 0.5  # Default neutral relevance

        if relevance >= 0.7:
            reason = "Highly relevant to user interests"
        elif relevance >= 0.5:
            reason = "Moderately relevant"
        else:
            reason = "Limited relevance to stated interests"

        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=relevance,
            reason=reason,
            threshold=self.config.min_relevance,
        )

    def _assess_actionability(self, insight: "EmergentInsight") -> QualityScore:
        """Assess actionability dimension."""
        # Check for actionable indicators
        title = getattr(insight, "title", "").lower()
        description = getattr(insight, "description", "").lower()
        content = f"{title} {description}"

        # Look for action-oriented language
        action_keywords = [
            "can", "should", "try", "consider", "improve",
            "increase", "decrease", "change", "adjust", "optimize",
            "focus", "prioritize", "avoid", "limit", "expand",
        ]

        action_count = sum(1 for kw in action_keywords if kw in content)
        score = min(1.0, action_count / 3)  # 3 action words = full score

        if score >= 0.7:
            reason = "Clearly actionable with specific suggestions"
        elif score >= 0.4:
            reason = "Some actionable elements"
        else:
            reason = "Informational only, no clear action"

        return QualityScore(
            dimension=QualityDimension.ACTIONABILITY,
            score=score,
            reason=reason,
            threshold=0.3,  # Low threshold since not all insights need actions
        )

    def _assess_temporal(self, insight: "EmergentInsight") -> QualityScore:
        """Assess temporal grounding dimension."""
        # Check for temporal metadata
        has_timestamp = hasattr(insight, "timestamp") or hasattr(insight, "created_at")
        has_temporal_range = (
            hasattr(insight, "start_date") or
            hasattr(insight, "time_range") or
            hasattr(insight, "temporal_context")
        )

        insight_type = getattr(insight, "insight_type", "").lower()
        is_temporal_type = any(
            t in insight_type
            for t in ["temporal", "correlation", "pattern", "trend", "cycle"]
        )

        # Score based on temporal grounding
        if has_timestamp and has_temporal_range:
            score = 1.0
            reason = "Fully temporally grounded"
        elif has_timestamp:
            score = 0.7
            reason = "Has timestamp but limited temporal context"
        elif is_temporal_type:
            score = 0.5
            reason = "Temporal insight type but missing metadata"
        else:
            score = 0.3
            reason = "Limited temporal grounding"

        threshold = 0.5 if self.config.require_temporal else 0.2

        return QualityScore(
            dimension=QualityDimension.TEMPORAL,
            score=score,
            reason=reason,
            threshold=threshold,
        )

    def _determine_result(
        self,
        scores: List[QualityScore],
        overall: float,
    ) -> QualityGateResult:
        """Determine overall quality gate result."""
        passed_count = sum(1 for s in scores if s.passed)
        total_count = len(scores)

        # Strict mode: all must pass
        if self.config.strict_mode:
            if passed_count == total_count:
                return QualityGateResult.PASSED
            elif passed_count >= total_count * 0.8:
                return QualityGateResult.REVIEW
            else:
                return QualityGateResult.FAILED

        # Normal mode: overall score + critical dimensions
        if overall >= self.config.min_overall:
            # Check critical dimensions (confidence and evidence)
            critical_passed = all(
                s.passed for s in scores
                if s.dimension in (QualityDimension.CONFIDENCE, QualityDimension.EVIDENCE)
            )

            if critical_passed:
                return QualityGateResult.PASSED
            else:
                return QualityGateResult.PENDING

        # Borderline cases
        if overall >= self.config.min_overall * 0.8:
            return QualityGateResult.REVIEW

        return QualityGateResult.FAILED

    def _generate_recommendation(
        self,
        result: QualityGateResult,
        scores: List[QualityScore],
        overall: float,
    ) -> str:
        """Generate recommendation based on assessment."""
        if result == QualityGateResult.PASSED:
            return "Insight meets quality standards. Ready for delivery."

        if result == QualityGateResult.PENDING:
            failed_dims = [s for s in scores if not s.passed]
            dim_names = ", ".join(s.dimension.value for s in failed_dims)
            return f"Needs more evidence for: {dim_names}. Queue for future reassessment."

        if result == QualityGateResult.REVIEW:
            return f"Borderline quality (score: {overall:.0%}). Manual review recommended."

        # Failed
        failed_dims = [s for s in scores if not s.passed]
        if len(failed_dims) >= 3:
            return "Multiple quality dimensions failed. Do not deliver."
        else:
            dim_names = ", ".join(s.dimension.value for s in failed_dims)
            return f"Failed on: {dim_names}. Gather more evidence or discard."

    def filter_low_quality(
        self,
        insights: List["EmergentInsight"],
    ) -> List["EmergentInsight"]:
        """Filter out low-quality insights.

        Args:
            insights: List of insights to filter

        Returns:
            List of insights that passed quality gates
        """
        passed = []

        for insight in insights:
            report = self.validate(insight)
            if report.result in (QualityGateResult.PASSED, QualityGateResult.REVIEW):
                passed.append(insight)
            else:
                logger.debug(
                    f"Filtered low-quality insight: "
                    f"{getattr(insight, 'title', 'Unknown')[:50]}... "
                    f"({report.result.value})"
                )

        logger.info(
            f"Quality filter: {len(passed)}/{len(insights)} insights passed"
        )

        return passed

    def batch_validate(
        self,
        insights: List["EmergentInsight"],
    ) -> List[QualityReport]:
        """Validate multiple insights.

        Args:
            insights: List of insights to validate

        Returns:
            List of quality reports
        """
        return [self.validate(insight) for insight in insights]

    def get_quality_summary(
        self,
        reports: List[QualityReport],
    ) -> Dict[str, Any]:
        """Get summary of quality assessment batch.

        Args:
            reports: List of quality reports

        Returns:
            Summary statistics
        """
        total = len(reports)
        if total == 0:
            return {"total": 0, "passed": 0, "failed": 0, "pending": 0, "review": 0}

        by_result = {}
        for r in reports:
            by_result[r.result.value] = by_result.get(r.result.value, 0) + 1

        avg_score = sum(r.overall_score for r in reports) / total

        return {
            "total": total,
            "passed": by_result.get("passed", 0),
            "failed": by_result.get("failed", 0),
            "pending": by_result.get("pending", 0),
            "review": by_result.get("review", 0),
            "average_score": avg_score,
            "pass_rate": by_result.get("passed", 0) / total,
        }


# Global instance
_quality_gate: Optional[InsightQualityGate] = None


def get_quality_gate() -> InsightQualityGate:
    """Get the default quality gate singleton."""
    global _quality_gate
    if _quality_gate is None:
        _quality_gate = InsightQualityGate()
    return _quality_gate

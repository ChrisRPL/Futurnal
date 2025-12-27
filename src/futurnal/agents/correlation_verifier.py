"""Correlation Verifier for AgentFlow Analysis.

Phase 2E: AgentFlow Architecture - Step 14

Validates correlation evidence and determines hypothesis status.
Applies Bradford Hill criteria-inspired checks for causal inference.

Research Foundation:
- Bradford Hill criteria (1965): Causal inference guidelines
- Event-CausNet (2025): Causal feature extraction
- ICDA (2024): Interactive Causal Discovery

Option B Compliance:
- No model parameter updates
- Verification results as natural language
- Ghost model FROZEN
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from futurnal.agents.memory_buffer import (
    EvolvingMemoryBuffer,
    MemoryEntry,
    MemoryEntryType,
    MemoryPriority,
    get_memory_buffer,
)
from futurnal.agents.correlation_planner import (
    CorrelationHypothesis,
    HypothesisStatus,
    HypothesisType,
    QueryPlan,
)

logger = logging.getLogger(__name__)


class VerificationResult(str, Enum):
    """Possible verification outcomes."""

    CONFIRMED = "confirmed"  # Strong evidence supports causation
    REFUTED = "refuted"  # Evidence contradicts causation
    INCONCLUSIVE = "inconclusive"  # Evidence is mixed or insufficient
    EXHAUSTED = "exhausted"  # No more evidence available to gather
    NEEDS_MORE_DATA = "needs_more_data"  # Specific data gaps identified


class BradfordHillCriterion(str, Enum):
    """Bradford Hill criteria for causal inference.

    These criteria guide assessment of whether an observed
    correlation represents a true causal relationship.
    """

    STRENGTH = "strength"  # Strong association
    CONSISTENCY = "consistency"  # Observed repeatedly
    SPECIFICITY = "specificity"  # Specific relationship
    TEMPORALITY = "temporality"  # Cause precedes effect
    BIOLOGICAL_GRADIENT = "biological_gradient"  # Dose-response
    PLAUSIBILITY = "plausibility"  # Mechanism makes sense
    COHERENCE = "coherence"  # Doesn't conflict with known facts
    EXPERIMENT = "experiment"  # Experimental evidence
    ANALOGY = "analogy"  # Similar known relationships


@dataclass
class EvidenceItem:
    """A piece of evidence for or against a hypothesis.

    Attributes:
        evidence_id: Unique identifier
        description: Natural language description
        is_supporting: True if supports the hypothesis
        criterion: Which Bradford Hill criterion this addresses
        strength: How strong is this evidence (0-1)
        source: Where the evidence came from
        timestamp: When evidence was gathered
        metadata: Additional data
    """

    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    is_supporting: bool = True
    criterion: Optional[BradfordHillCriterion] = None
    strength: float = 0.5
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "description": self.description,
            "is_supporting": self.is_supporting,
            "criterion": self.criterion.value if self.criterion else None,
            "strength": self.strength,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        """Create from dictionary."""
        return cls(
            evidence_id=data.get("evidence_id", str(uuid4())),
            description=data.get("description", ""),
            is_supporting=data.get("is_supporting", True),
            criterion=BradfordHillCriterion(data["criterion"])
            if data.get("criterion")
            else None,
            strength=data.get("strength", 0.5),
            source=data.get("source", ""),
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.utcnow().isoformat())
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VerificationReport:
    """Report summarizing verification of a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis that was verified
        result: Verification outcome
        confidence: Final confidence (0-1)
        evidence_summary: Summary of evidence
        criteria_met: Which Bradford Hill criteria were satisfied
        criteria_violated: Which criteria were violated
        recommendation: Suggested next steps
        created_at: When report was created
    """

    hypothesis_id: str
    result: VerificationResult
    confidence: float = 0.5
    evidence_summary: str = ""
    criteria_met: List[BradfordHillCriterion] = field(default_factory=list)
    criteria_violated: List[BradfordHillCriterion] = field(default_factory=list)
    recommendation: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "result": self.result.value,
            "confidence": self.confidence,
            "evidence_summary": self.evidence_summary,
            "criteria_met": [c.value for c in self.criteria_met],
            "criteria_violated": [c.value for c in self.criteria_violated],
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
        }

    def to_natural_language(self) -> str:
        """Convert to natural language for token priors."""
        lines = [
            f"Verification Result: {self.result.value.upper()}",
            f"Confidence: {self.confidence:.0%}",
            "",
            "Evidence Summary:",
            self.evidence_summary,
        ]

        if self.criteria_met:
            lines.append(f"\nCriteria Met: {', '.join(c.value for c in self.criteria_met)}")

        if self.criteria_violated:
            lines.append(
                f"Criteria Violated: {', '.join(c.value for c in self.criteria_violated)}"
            )

        if self.recommendation:
            lines.append(f"\nRecommendation: {self.recommendation}")

        return "\n".join(lines)


class CorrelationVerifier:
    """Verifies correlation evidence and determines causal validity.

    AgentFlow Module 3: Applies Bradford Hill-inspired criteria to
    evaluate whether observed correlations represent true causation.

    Key Capabilities:
    1. Evaluate evidence against Bradford Hill criteria
    2. Determine verification result (CONFIRMED/REFUTED/etc)
    3. Generate natural language verification reports
    4. Track verification history in memory buffer

    Bradford Hill Criteria Applied:
    - Strength: Is the correlation statistically strong?
    - Consistency: Is it observed repeatedly?
    - Temporality: Does cause precede effect?
    - Plausibility: Does the relationship make sense?
    - Coherence: Is it consistent with other knowledge?

    Option B Compliance:
    - No model updates
    - Results as natural language
    - Ghost model FROZEN

    Usage:
        verifier = CorrelationVerifier()
        evidence = [EvidenceItem(...), ...]
        report = verifier.verify_evidence(hypothesis, evidence)
        if report.result == VerificationResult.CONFIRMED:
            # Promote to confirmed causal relationship
    """

    # Minimum thresholds for different outcomes
    CONFIRMATION_THRESHOLD = 0.75  # Confidence needed for CONFIRMED
    REFUTATION_THRESHOLD = 0.25  # Below this is REFUTED
    MIN_EVIDENCE_FOR_CONCLUSION = 3  # Need at least 3 evidence items
    CRITICAL_CRITERIA = [
        BradfordHillCriterion.TEMPORALITY,
        BradfordHillCriterion.STRENGTH,
    ]

    def __init__(
        self,
        memory_buffer: Optional[EvolvingMemoryBuffer] = None,
    ):
        """Initialize correlation verifier.

        Args:
            memory_buffer: Memory buffer for tracking state
        """
        self.memory = memory_buffer or get_memory_buffer()

        # Track verified hypotheses
        self._verification_history: Dict[str, List[VerificationReport]] = {}

        logger.info("CorrelationVerifier initialized")

    def verify_evidence(
        self,
        hypothesis: CorrelationHypothesis,
        evidence: List[EvidenceItem],
    ) -> VerificationReport:
        """Verify evidence for a hypothesis.

        Applies Bradford Hill criteria-inspired checks to determine
        if the correlation represents true causation.

        Args:
            hypothesis: The hypothesis being verified
            evidence: List of evidence items

        Returns:
            VerificationReport with result and details
        """
        if not evidence:
            return self._create_report(
                hypothesis,
                VerificationResult.NEEDS_MORE_DATA,
                confidence=hypothesis.confidence,
                summary="No evidence provided for verification.",
                recommendation="Gather initial evidence by executing the query plan.",
            )

        # Separate supporting and contradicting evidence
        supporting = [e for e in evidence if e.is_supporting]
        contradicting = [e for e in evidence if not e.is_supporting]

        # Calculate evidence balance
        total_strength = sum(e.strength for e in evidence)
        if total_strength == 0:
            total_strength = 1  # Avoid division by zero

        supporting_weight = sum(e.strength for e in supporting) / total_strength
        contradicting_weight = sum(e.strength for e in contradicting) / total_strength

        # Check criteria
        criteria_met, criteria_violated = self._evaluate_criteria(
            hypothesis, evidence
        )

        # Calculate final confidence
        base_confidence = hypothesis.confidence
        evidence_adjustment = (supporting_weight - contradicting_weight) * 0.3

        # Criteria bonuses/penalties
        criteria_adjustment = 0.0
        for criterion in criteria_met:
            if criterion in self.CRITICAL_CRITERIA:
                criteria_adjustment += 0.1
            else:
                criteria_adjustment += 0.05

        for criterion in criteria_violated:
            if criterion in self.CRITICAL_CRITERIA:
                criteria_adjustment -= 0.15
            else:
                criteria_adjustment -= 0.05

        final_confidence = max(
            0.0, min(1.0, base_confidence + evidence_adjustment + criteria_adjustment)
        )

        # Determine result
        result = self._determine_result(
            final_confidence, len(evidence), criteria_met, criteria_violated
        )

        # Generate summary
        summary = self._generate_summary(
            hypothesis, supporting, contradicting, criteria_met, criteria_violated
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            result, hypothesis, criteria_met, criteria_violated
        )

        report = self._create_report(
            hypothesis,
            result,
            final_confidence,
            summary,
            criteria_met,
            criteria_violated,
            recommendation,
        )

        # Update hypothesis status based on result
        self._update_hypothesis_status(hypothesis, result, final_confidence)

        # Store in verification history
        if hypothesis.hypothesis_id not in self._verification_history:
            self._verification_history[hypothesis.hypothesis_id] = []
        self._verification_history[hypothesis.hypothesis_id].append(report)

        # Store in memory buffer
        self.memory.add_entry(
            MemoryEntry(
                entry_type=MemoryEntryType.VERIFICATION,
                content=report.to_natural_language(),
                priority=MemoryPriority.HIGH
                if result == VerificationResult.CONFIRMED
                else MemoryPriority.NORMAL,
                metadata={
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "result": result.value,
                    "confidence": final_confidence,
                },
            )
        )

        logger.info(
            f"Verified hypothesis {hypothesis.hypothesis_id}: "
            f"{result.value} (confidence={final_confidence:.2f})"
        )

        return report

    def _evaluate_criteria(
        self,
        hypothesis: CorrelationHypothesis,
        evidence: List[EvidenceItem],
    ) -> tuple[List[BradfordHillCriterion], List[BradfordHillCriterion]]:
        """Evaluate which Bradford Hill criteria are met/violated.

        Args:
            hypothesis: The hypothesis
            evidence: Evidence items

        Returns:
            Tuple of (criteria_met, criteria_violated)
        """
        criteria_met = []
        criteria_violated = []

        # Group evidence by criterion
        by_criterion: Dict[BradfordHillCriterion, List[EvidenceItem]] = {}
        for e in evidence:
            if e.criterion:
                if e.criterion not in by_criterion:
                    by_criterion[e.criterion] = []
                by_criterion[e.criterion].append(e)

        # Evaluate each criterion with evidence
        for criterion, items in by_criterion.items():
            supporting = [e for e in items if e.is_supporting]
            contradicting = [e for e in items if not e.is_supporting]

            if len(supporting) > len(contradicting):
                avg_strength = sum(e.strength for e in supporting) / len(supporting)
                if avg_strength >= 0.5:
                    criteria_met.append(criterion)
            elif len(contradicting) > len(supporting):
                criteria_violated.append(criterion)

        # Special checks for hypothesis type
        if hypothesis.hypothesis_type == HypothesisType.TEMPORAL_SEQUENCE:
            # Temporality must be verified for temporal hypotheses
            if BradfordHillCriterion.TEMPORALITY not in criteria_met:
                if BradfordHillCriterion.TEMPORALITY not in criteria_violated:
                    # Need to explicitly verify temporality
                    pass  # Will be handled by NEEDS_MORE_DATA

        return criteria_met, criteria_violated

    def _determine_result(
        self,
        confidence: float,
        evidence_count: int,
        criteria_met: List[BradfordHillCriterion],
        criteria_violated: List[BradfordHillCriterion],
    ) -> VerificationResult:
        """Determine the verification result.

        Args:
            confidence: Final confidence score
            evidence_count: Number of evidence items
            criteria_met: Satisfied criteria
            criteria_violated: Violated criteria

        Returns:
            VerificationResult
        """
        # Check for critical violations
        for criterion in self.CRITICAL_CRITERIA:
            if criterion in criteria_violated:
                return VerificationResult.REFUTED

        # Not enough evidence
        if evidence_count < self.MIN_EVIDENCE_FOR_CONCLUSION:
            return VerificationResult.NEEDS_MORE_DATA

        # High confidence with criteria met
        if confidence >= self.CONFIRMATION_THRESHOLD:
            # Must have at least one critical criterion met
            if any(c in criteria_met for c in self.CRITICAL_CRITERIA):
                return VerificationResult.CONFIRMED
            else:
                return VerificationResult.INCONCLUSIVE

        # Low confidence
        if confidence <= self.REFUTATION_THRESHOLD:
            return VerificationResult.REFUTED

        # Middle ground
        return VerificationResult.INCONCLUSIVE

    def _generate_summary(
        self,
        hypothesis: CorrelationHypothesis,
        supporting: List[EvidenceItem],
        contradicting: List[EvidenceItem],
        criteria_met: List[BradfordHillCriterion],
        criteria_violated: List[BradfordHillCriterion],
    ) -> str:
        """Generate natural language summary of evidence.

        Args:
            hypothesis: The hypothesis
            supporting: Supporting evidence
            contradicting: Contradicting evidence
            criteria_met: Satisfied criteria
            criteria_violated: Violated criteria

        Returns:
            Summary string
        """
        lines = []

        lines.append(f"Investigated: {hypothesis.description}")
        lines.append("")

        if supporting:
            lines.append(f"Supporting evidence ({len(supporting)} items):")
            for e in supporting[:3]:
                lines.append(f"  + {e.description[:100]}")
            if len(supporting) > 3:
                lines.append(f"  ... and {len(supporting) - 3} more")

        if contradicting:
            lines.append(f"\nContradicting evidence ({len(contradicting)} items):")
            for e in contradicting[:3]:
                lines.append(f"  - {e.description[:100]}")
            if len(contradicting) > 3:
                lines.append(f"  ... and {len(contradicting) - 3} more")

        if criteria_met:
            lines.append(
                f"\nCausal criteria satisfied: {', '.join(c.value for c in criteria_met)}"
            )

        if criteria_violated:
            lines.append(
                f"Causal criteria violated: {', '.join(c.value for c in criteria_violated)}"
            )

        return "\n".join(lines)

    def _generate_recommendation(
        self,
        result: VerificationResult,
        hypothesis: CorrelationHypothesis,
        criteria_met: List[BradfordHillCriterion],
        criteria_violated: List[BradfordHillCriterion],
    ) -> str:
        """Generate recommendation for next steps.

        Args:
            result: Verification result
            hypothesis: The hypothesis
            criteria_met: Satisfied criteria
            criteria_violated: Violated criteria

        Returns:
            Recommendation string
        """
        if result == VerificationResult.CONFIRMED:
            return (
                f"This correlation appears to be causal. "
                f"Consider surfacing as an insight: '{hypothesis.description}'"
            )

        if result == VerificationResult.REFUTED:
            violated_str = ", ".join(c.value for c in criteria_violated) if criteria_violated else "insufficient evidence"
            return (
                f"This correlation does not appear causal ({violated_str}). "
                f"Archive hypothesis and explore alternative explanations."
            )

        if result == VerificationResult.NEEDS_MORE_DATA:
            missing_criteria = [
                c for c in self.CRITICAL_CRITERIA if c not in criteria_met
            ]
            if missing_criteria:
                return (
                    f"Gather more evidence to evaluate: "
                    f"{', '.join(c.value for c in missing_criteria)}"
                )
            return "Gather at least 3 evidence items before drawing conclusions."

        if result == VerificationResult.INCONCLUSIVE:
            return (
                "Evidence is mixed. Consider gathering more data on "
                f"{'temporality' if BradfordHillCriterion.TEMPORALITY not in criteria_met else 'strength'} "
                "to reach a conclusion."
            )

        return "Continue investigation with targeted queries."

    def _create_report(
        self,
        hypothesis: CorrelationHypothesis,
        result: VerificationResult,
        confidence: float,
        summary: str,
        criteria_met: Optional[List[BradfordHillCriterion]] = None,
        criteria_violated: Optional[List[BradfordHillCriterion]] = None,
        recommendation: str = "",
    ) -> VerificationReport:
        """Create a verification report.

        Args:
            hypothesis: The hypothesis
            result: Verification result
            confidence: Final confidence
            summary: Evidence summary
            criteria_met: Satisfied criteria
            criteria_violated: Violated criteria
            recommendation: Next steps

        Returns:
            VerificationReport
        """
        return VerificationReport(
            hypothesis_id=hypothesis.hypothesis_id,
            result=result,
            confidence=confidence,
            evidence_summary=summary,
            criteria_met=criteria_met or [],
            criteria_violated=criteria_violated or [],
            recommendation=recommendation,
        )

    def _update_hypothesis_status(
        self,
        hypothesis: CorrelationHypothesis,
        result: VerificationResult,
        confidence: float,
    ) -> None:
        """Update hypothesis status based on verification result.

        Args:
            hypothesis: The hypothesis to update
            result: Verification result
            confidence: Final confidence
        """
        hypothesis.confidence = confidence
        hypothesis.last_updated = datetime.utcnow()

        status_map = {
            VerificationResult.CONFIRMED: HypothesisStatus.CONFIRMED,
            VerificationResult.REFUTED: HypothesisStatus.REFUTED,
            VerificationResult.INCONCLUSIVE: HypothesisStatus.INCONCLUSIVE,
            VerificationResult.EXHAUSTED: HypothesisStatus.INCONCLUSIVE,
            VerificationResult.NEEDS_MORE_DATA: HypothesisStatus.NEEDS_EVIDENCE,
        }

        hypothesis.status = status_map.get(result, HypothesisStatus.INVESTIGATING)

    def evaluate_temporal_evidence(
        self,
        hypothesis: CorrelationHypothesis,
        forward_correlation: float,
        reverse_correlation: float,
        sample_size: int,
    ) -> EvidenceItem:
        """Evaluate temporal correlation evidence.

        Checks if A precedes B more often than B precedes A.

        Args:
            hypothesis: The hypothesis
            forward_correlation: Correlation of A -> B
            reverse_correlation: Correlation of B -> A
            sample_size: Number of event pairs examined

        Returns:
            EvidenceItem summarizing temporal evidence
        """
        # Temporality criterion: forward should be stronger than reverse
        is_supporting = forward_correlation > reverse_correlation * 1.2

        if is_supporting:
            strength = min(1.0, (forward_correlation - reverse_correlation) * 2)
            description = (
                f"Temporal ordering confirmed: {hypothesis.event_type_a} precedes "
                f"{hypothesis.event_type_b} with {forward_correlation:.0%} correlation "
                f"(reverse: {reverse_correlation:.0%}, n={sample_size})"
            )
        else:
            strength = 0.7 if forward_correlation < reverse_correlation else 0.3
            description = (
                f"Temporal ordering unclear: forward correlation {forward_correlation:.0%} "
                f"vs reverse {reverse_correlation:.0%} (n={sample_size})"
            )

        return EvidenceItem(
            description=description,
            is_supporting=is_supporting,
            criterion=BradfordHillCriterion.TEMPORALITY,
            strength=strength,
            source="temporal_correlation_analysis",
            metadata={
                "forward_correlation": forward_correlation,
                "reverse_correlation": reverse_correlation,
                "sample_size": sample_size,
            },
        )

    def evaluate_strength_evidence(
        self,
        hypothesis: CorrelationHypothesis,
        correlation_coefficient: float,
        p_value: float,
        sample_size: int,
    ) -> EvidenceItem:
        """Evaluate correlation strength evidence.

        Args:
            hypothesis: The hypothesis
            correlation_coefficient: Correlation strength (-1 to 1)
            p_value: Statistical significance
            sample_size: Sample size

        Returns:
            EvidenceItem summarizing strength evidence
        """
        abs_correlation = abs(correlation_coefficient)

        # Strength criterion: significant and reasonably strong
        is_significant = p_value < 0.05
        is_strong = abs_correlation >= 0.3

        is_supporting = is_significant and is_strong

        if is_supporting:
            strength = min(1.0, abs_correlation)
            description = (
                f"Strong correlation found: r={correlation_coefficient:.2f}, "
                f"p={p_value:.4f} (n={sample_size})"
            )
        else:
            strength = 0.3
            if not is_significant:
                description = f"Correlation not significant: p={p_value:.4f} > 0.05"
            else:
                description = f"Weak correlation: r={correlation_coefficient:.2f}"

        return EvidenceItem(
            description=description,
            is_supporting=is_supporting,
            criterion=BradfordHillCriterion.STRENGTH,
            strength=strength,
            source="statistical_analysis",
            metadata={
                "correlation_coefficient": correlation_coefficient,
                "p_value": p_value,
                "sample_size": sample_size,
            },
        )

    def evaluate_consistency_evidence(
        self,
        hypothesis: CorrelationHypothesis,
        occurrences: int,
        time_periods: int,
        consistency_rate: float,
    ) -> EvidenceItem:
        """Evaluate consistency evidence.

        Checks if the correlation is observed repeatedly across time.

        Args:
            hypothesis: The hypothesis
            occurrences: Total occurrences observed
            time_periods: Number of distinct time periods
            consistency_rate: Rate of consistent observation

        Returns:
            EvidenceItem summarizing consistency evidence
        """
        # Consistency criterion: observed in multiple time periods
        is_supporting = time_periods >= 3 and consistency_rate >= 0.6

        if is_supporting:
            strength = min(1.0, consistency_rate)
            description = (
                f"Consistent pattern: observed in {time_periods} time periods "
                f"with {consistency_rate:.0%} consistency ({occurrences} total occurrences)"
            )
        else:
            strength = 0.3
            description = (
                f"Inconsistent pattern: only {time_periods} periods, "
                f"{consistency_rate:.0%} consistency"
            )

        return EvidenceItem(
            description=description,
            is_supporting=is_supporting,
            criterion=BradfordHillCriterion.CONSISTENCY,
            strength=strength,
            source="pattern_analysis",
            metadata={
                "occurrences": occurrences,
                "time_periods": time_periods,
                "consistency_rate": consistency_rate,
            },
        )

    def get_verification_history(
        self, hypothesis_id: str
    ) -> List[VerificationReport]:
        """Get verification history for a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            List of verification reports
        """
        return self._verification_history.get(hypothesis_id, [])

    def get_latest_verification(
        self, hypothesis_id: str
    ) -> Optional[VerificationReport]:
        """Get the most recent verification for a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            Latest VerificationReport, or None
        """
        history = self.get_verification_history(hypothesis_id)
        return history[-1] if history else None

    def export_for_token_priors(self) -> str:
        """Export verifier state as natural language for token priors."""
        lines = ["Verification State:"]

        confirmed = []
        refuted = []

        for hypothesis_id, reports in self._verification_history.items():
            if reports:
                latest = reports[-1]
                if latest.result == VerificationResult.CONFIRMED:
                    confirmed.append(latest)
                elif latest.result == VerificationResult.REFUTED:
                    refuted.append(latest)

        if confirmed:
            lines.append(f"\nConfirmed Causal Relationships: {len(confirmed)}")
            for report in confirmed[:3]:
                lines.append(f"- {report.evidence_summary[:100]}...")

        if refuted:
            lines.append(f"\nRefuted Correlations: {len(refuted)}")
            for report in refuted[:3]:
                lines.append(f"- {report.evidence_summary[:100]}...")

        return "\n".join(lines)


# Global instance
_correlation_verifier: Optional[CorrelationVerifier] = None


def get_correlation_verifier() -> CorrelationVerifier:
    """Get the default correlation verifier singleton."""
    global _correlation_verifier
    if _correlation_verifier is None:
        _correlation_verifier = CorrelationVerifier()
    return _correlation_verifier

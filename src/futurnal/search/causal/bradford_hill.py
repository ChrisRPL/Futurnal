"""
Bradford-Hill Criteria Validation for Causal Inference.

Implements the 9 Bradford-Hill criteria for evaluating causality
from observational data. These criteria help distinguish true
causal relationships from mere correlations.

Research Foundation:
- Bradford Hill, A.B. (1965): "The Environment and Disease: Association or Causation?"
- Modern interpretations for computational causality assessment

The 9 Criteria:
1. Strength - Strong associations more likely causal
2. Consistency - Repeated observation across contexts
3. Specificity - Specific cause leads to specific effect
4. Temporality - Cause precedes effect
5. Biological Gradient - Dose-response relationship
6. Plausibility - Mechanism is biologically/logically plausible
7. Coherence - Doesn't conflict with known facts
8. Experiment - Experimental evidence supports causation
9. Analogy - Similar relationships are known to be causal

Option B Compliance:
- Scoring uses rule-based heuristics
- LLM assists with plausibility assessment only
- No model parameter updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import math

if TYPE_CHECKING:
    from futurnal.search.temporal.results import TemporalCorrelationResult
    from futurnal.insights.hypothesis_generation import CausalHypothesis

logger = logging.getLogger(__name__)


class CriterionStrength(str, Enum):
    """Strength levels for criterion assessment."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class CausalityVerdict(str, Enum):
    """Overall causality verdict."""
    LIKELY_CAUSAL = "likely_causal"
    POSSIBLY_CAUSAL = "possibly_causal"
    UNCERTAIN = "uncertain"
    POSSIBLY_NOT_CAUSAL = "possibly_not_causal"
    LIKELY_NOT_CAUSAL = "likely_not_causal"


@dataclass
class CriterionAssessment:
    """Assessment of a single Bradford-Hill criterion."""
    criterion: str
    score: float  # 0.0 to 1.0
    strength: CriterionStrength
    evidence: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    weight: float = 1.0  # Importance weight for this criterion
    applicable: bool = True  # Whether this criterion applies

    @property
    def weighted_score(self) -> float:
        """Score weighted by importance."""
        return self.score * self.weight if self.applicable else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "score": self.score,
            "strength": self.strength.value,
            "evidence": self.evidence,
            "concerns": self.concerns,
            "weight": self.weight,
            "applicable": self.applicable,
        }


@dataclass
class BradfordHillReport:
    """Complete Bradford-Hill criteria assessment report."""
    assessments: Dict[str, CriterionAssessment]
    overall_score: float
    verdict: CausalityVerdict
    summary: str
    recommendations: List[str]

    # Metadata
    cause_event: str = ""
    effect_event: str = ""
    assessed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def strong_criteria(self) -> List[str]:
        """Criteria with strong support."""
        return [
            name for name, a in self.assessments.items()
            if a.strength in [CriterionStrength.STRONG, CriterionStrength.VERY_STRONG]
        ]

    @property
    def weak_criteria(self) -> List[str]:
        """Criteria with weak support."""
        return [
            name for name, a in self.assessments.items()
            if a.strength in [CriterionStrength.WEAK, CriterionStrength.VERY_WEAK]
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessments": {k: v.to_dict() for k, v in self.assessments.items()},
            "overall_score": self.overall_score,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "cause_event": self.cause_event,
            "effect_event": self.effect_event,
            "strong_criteria": self.strong_criteria,
            "weak_criteria": self.weak_criteria,
        }


class BradfordHillValidator:
    """
    Validates causal hypotheses using Bradford-Hill criteria.

    Provides structured assessment of potential causal relationships
    based on observational evidence.
    """

    # Criterion weights (sum to 9.0 for equal weighting)
    DEFAULT_WEIGHTS = {
        "strength": 1.0,
        "consistency": 1.0,
        "specificity": 1.0,
        "temporality": 1.5,  # Higher weight - fundamental for causation
        "gradient": 1.0,
        "plausibility": 1.0,
        "coherence": 1.0,
        "experiment": 0.5,  # Lower weight - often not available
        "analogy": 1.0,
    }

    # Score thresholds for strength classification
    STRENGTH_THRESHOLDS = {
        CriterionStrength.VERY_STRONG: 0.8,
        CriterionStrength.STRONG: 0.65,
        CriterionStrength.MODERATE: 0.45,
        CriterionStrength.WEAK: 0.25,
        CriterionStrength.VERY_WEAK: 0.0,
    }

    # Verdict thresholds
    VERDICT_THRESHOLDS = {
        CausalityVerdict.LIKELY_CAUSAL: 0.75,
        CausalityVerdict.POSSIBLY_CAUSAL: 0.55,
        CausalityVerdict.UNCERTAIN: 0.40,
        CausalityVerdict.POSSIBLY_NOT_CAUSAL: 0.25,
        CausalityVerdict.LIKELY_NOT_CAUSAL: 0.0,
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
        analogies_database: Optional[Dict[str, List[str]]] = None
    ):
        """Initialize validator.

        Args:
            llm_client: Optional LLM for plausibility assessment
            weights: Custom criterion weights
            analogies_database: Known causal analogies for comparison
        """
        self.llm_client = llm_client
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.analogies_database = analogies_database or {}

        # Cache for assessments
        self._assessment_cache: Dict[str, BradfordHillReport] = {}

    async def validate(
        self,
        correlation: "TemporalCorrelationResult",
        hypothesis: Optional["CausalHypothesis"] = None,
        additional_evidence: Optional[Dict[str, Any]] = None
    ) -> BradfordHillReport:
        """Perform full Bradford-Hill validation.

        Args:
            correlation: The temporal correlation to assess
            hypothesis: Optional hypothesis with mechanism information
            additional_evidence: Additional evidence for assessment

        Returns:
            BradfordHillReport with complete assessment
        """
        cause = correlation.event_type_a
        effect = correlation.event_type_b
        additional_evidence = additional_evidence or {}

        assessments = {}

        # 1. Strength of association
        assessments["strength"] = self._assess_strength(correlation)

        # 2. Consistency
        assessments["consistency"] = self._assess_consistency(correlation, additional_evidence)

        # 3. Specificity
        assessments["specificity"] = self._assess_specificity(correlation, additional_evidence)

        # 4. Temporality
        assessments["temporality"] = self._assess_temporality(correlation)

        # 5. Biological gradient (dose-response)
        assessments["gradient"] = self._assess_gradient(correlation, additional_evidence)

        # 6. Plausibility
        assessments["plausibility"] = await self._assess_plausibility(
            correlation, hypothesis, additional_evidence
        )

        # 7. Coherence
        assessments["coherence"] = self._assess_coherence(correlation, hypothesis, additional_evidence)

        # 8. Experiment
        assessments["experiment"] = self._assess_experiment(additional_evidence)

        # 9. Analogy
        assessments["analogy"] = self._assess_analogy(correlation)

        # Calculate overall score
        overall_score = self._calculate_overall_score(assessments)

        # Determine verdict
        verdict = self._determine_verdict(overall_score, assessments)

        # Generate summary and recommendations
        summary = self._generate_summary(cause, effect, assessments, verdict)
        recommendations = self._generate_recommendations(assessments)

        report = BradfordHillReport(
            assessments=assessments,
            overall_score=overall_score,
            verdict=verdict,
            summary=summary,
            recommendations=recommendations,
            cause_event=cause,
            effect_event=effect,
        )

        # Cache
        cache_key = f"{cause}|{effect}"
        self._assessment_cache[cache_key] = report

        logger.info(
            f"Bradford-Hill validation: {cause} -> {effect} = {verdict.value} "
            f"(score: {overall_score:.2f})"
        )

        return report

    def validate_sync(
        self,
        correlation: "TemporalCorrelationResult",
        hypothesis: Optional["CausalHypothesis"] = None,
        additional_evidence: Optional[Dict[str, Any]] = None
    ) -> BradfordHillReport:
        """Synchronous version of validate."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.validate(correlation, hypothesis, additional_evidence)
        )

    def _assess_strength(
        self,
        correlation: "TemporalCorrelationResult"
    ) -> CriterionAssessment:
        """Assess strength of association."""
        evidence = []
        concerns = []

        # Get correlation metrics
        strength = correlation.correlation_strength or 0
        effect_size = correlation.effect_size or 0
        p_value = correlation.p_value

        # Score based on multiple factors
        score = 0.0

        # Correlation strength component
        score += min(strength, 1.0) * 0.4
        evidence.append(f"Correlation strength: {strength:.0%}")

        # Effect size component
        if effect_size > 2.0:
            score += 0.3
            evidence.append(f"Large effect size: {effect_size:.2f}")
        elif effect_size > 1.5:
            score += 0.2
            evidence.append(f"Medium effect size: {effect_size:.2f}")
        elif effect_size > 1.0:
            score += 0.1
            evidence.append(f"Small effect size: {effect_size:.2f}")
        else:
            concerns.append(f"Small effect size: {effect_size:.2f}")

        # Statistical significance component
        if p_value is not None:
            if p_value < 0.001:
                score += 0.3
                evidence.append(f"Highly significant: p < 0.001")
            elif p_value < 0.01:
                score += 0.2
                evidence.append(f"Significant: p < 0.01")
            elif p_value < 0.05:
                score += 0.15
                evidence.append(f"Significant: p < 0.05")
            else:
                concerns.append(f"Not statistically significant: p = {p_value:.3f}")

        return CriterionAssessment(
            criterion="strength",
            score=min(1.0, score),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["strength"],
        )

    def _assess_consistency(
        self,
        correlation: "TemporalCorrelationResult",
        additional_evidence: Dict[str, Any]
    ) -> CriterionAssessment:
        """Assess consistency (repeated observation)."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate

        # Sample size / occurrences
        occurrences = correlation.co_occurrences or 0
        if occurrences >= 20:
            score += 0.3
            evidence.append(f"Many observations: {occurrences}")
        elif occurrences >= 10:
            score += 0.2
            evidence.append(f"Moderate observations: {occurrences}")
        elif occurrences >= 5:
            score += 0.1
            evidence.append(f"Some observations: {occurrences}")
        else:
            score -= 0.2
            concerns.append(f"Few observations: {occurrences}")

        # Gap consistency
        gap_consistency = correlation.gap_consistency or 0
        if gap_consistency > 0.7:
            score += 0.2
            evidence.append(f"Consistent timing: {gap_consistency:.0%}")
        elif gap_consistency < 0.3:
            score -= 0.1
            concerns.append(f"Inconsistent timing: {gap_consistency:.0%}")

        # External evidence
        if additional_evidence.get("replicated_studies"):
            score += 0.2
            evidence.append("Pattern replicated in other contexts")

        return CriterionAssessment(
            criterion="consistency",
            score=min(1.0, max(0.0, score)),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["consistency"],
        )

    def _assess_specificity(
        self,
        correlation: "TemporalCorrelationResult",
        additional_evidence: Dict[str, Any]
    ) -> CriterionAssessment:
        """Assess specificity (specific cause, specific effect)."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate

        # Check if effect has many other causes
        other_causes = additional_evidence.get("other_causes", [])
        if len(other_causes) > 5:
            score -= 0.2
            concerns.append(f"Effect has many other known causes: {len(other_causes)}")
        elif len(other_causes) > 2:
            score -= 0.1
            concerns.append(f"Effect has some other causes: {len(other_causes)}")
        else:
            score += 0.1
            evidence.append("Relatively specific effect")

        # Check if cause has many other effects
        other_effects = additional_evidence.get("other_effects", [])
        if len(other_effects) > 5:
            score -= 0.1
            concerns.append(f"Cause has many other effects: {len(other_effects)}")

        # Specificity is often considered the weakest criterion
        # Modern interpretation: can still be causal even if not specific
        evidence.append("Note: Non-specificity doesn't rule out causation")

        return CriterionAssessment(
            criterion="specificity",
            score=min(1.0, max(0.0, score)),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["specificity"],
        )

    def _assess_temporality(
        self,
        correlation: "TemporalCorrelationResult"
    ) -> CriterionAssessment:
        """Assess temporality (cause precedes effect)."""
        evidence = []
        concerns = []

        avg_gap = correlation.avg_gap_days or 0
        min_gap = correlation.min_gap_days or 0

        if avg_gap > 0 and min_gap >= 0:
            score = 1.0  # Cause always precedes effect
            evidence.append(f"Cause consistently precedes effect by {avg_gap:.1f} days")
            evidence.append("Temporal order clearly established")
        elif avg_gap > 0:
            score = 0.8
            evidence.append(f"Cause typically precedes effect (avg: {avg_gap:.1f} days)")
            concerns.append("Some instances with unclear temporal order")
        else:
            score = 0.3
            concerns.append("Temporal relationship unclear or reversed")

        return CriterionAssessment(
            criterion="temporality",
            score=score,
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["temporality"],
        )

    def _assess_gradient(
        self,
        correlation: "TemporalCorrelationResult",
        additional_evidence: Dict[str, Any]
    ) -> CriterionAssessment:
        """Assess biological gradient (dose-response)."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate

        # Check for dose-response data
        dose_response = additional_evidence.get("dose_response_data")
        if dose_response:
            # Analyze dose-response relationship
            if dose_response.get("monotonic"):
                score += 0.3
                evidence.append("Monotonic dose-response relationship observed")
            if dose_response.get("linear"):
                score += 0.2
                evidence.append("Linear dose-response relationship")
            score = min(1.0, score)
        else:
            concerns.append("No dose-response data available")
            evidence.append("Dose-response assessment limited by available data")

        # Effect size can proxy for gradient
        effect_size = correlation.effect_size
        if effect_size and effect_size > 1.5:
            score += 0.1
            evidence.append("Strong effect size suggests gradient")

        return CriterionAssessment(
            criterion="gradient",
            score=max(0.0, score),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["gradient"],
        )

    async def _assess_plausibility(
        self,
        correlation: "TemporalCorrelationResult",
        hypothesis: Optional["CausalHypothesis"],
        additional_evidence: Dict[str, Any]
    ) -> CriterionAssessment:
        """Assess plausibility (mechanism makes sense)."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate

        # Use hypothesis mechanism if available
        if hypothesis and hypothesis.mechanism_description:
            score += 0.2
            evidence.append(f"Proposed mechanism: {hypothesis.mechanism_description[:100]}...")
            if hypothesis.mechanism_strength > 0.6:
                score += 0.1
                evidence.append("Strong mechanism plausibility")

        # Use LLM for plausibility assessment if available
        if self.llm_client:
            llm_score = await self._llm_plausibility_check(
                correlation.event_type_a,
                correlation.event_type_b,
                hypothesis
            )
            score = (score + llm_score) / 2
            evidence.append(f"LLM plausibility assessment: {llm_score:.0%}")

        # Check for known mechanisms
        known_mechanisms = additional_evidence.get("known_mechanisms", [])
        if known_mechanisms:
            score += 0.2
            evidence.append(f"Known mechanisms: {', '.join(known_mechanisms[:3])}")

        return CriterionAssessment(
            criterion="plausibility",
            score=min(1.0, max(0.0, score)),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["plausibility"],
        )

    async def _llm_plausibility_check(
        self,
        cause: str,
        effect: str,
        hypothesis: Optional["CausalHypothesis"]
    ) -> float:
        """Use LLM to assess plausibility."""
        prompt = f"""Assess the plausibility of this causal relationship:

Cause: {cause}
Effect: {effect}
"""
        if hypothesis and hypothesis.mechanism_description:
            prompt += f"\nProposed mechanism: {hypothesis.mechanism_description}"

        prompt += """

Rate the plausibility from 0.0 to 1.0:
- 0.0-0.3: Highly implausible, no reasonable mechanism
- 0.3-0.5: Somewhat plausible but mechanism unclear
- 0.5-0.7: Moderately plausible with reasonable mechanism
- 0.7-1.0: Highly plausible with strong mechanistic support

Respond with just a number between 0.0 and 1.0."""

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
            else:
                return 0.5  # Default

            # Parse score from response
            import re
            numbers = re.findall(r"0\.\d+|1\.0|0|1", response)
            if numbers:
                return float(numbers[0])
        except Exception as e:
            logger.warning(f"LLM plausibility check failed: {e}")

        return 0.5

    def _assess_coherence(
        self,
        correlation: "TemporalCorrelationResult",
        hypothesis: Optional["CausalHypothesis"],
        additional_evidence: Dict[str, Any]
    ) -> CriterionAssessment:
        """Assess coherence (fits with known facts)."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate

        # Check for contradictions
        contradictions = additional_evidence.get("contradictions", [])
        if contradictions:
            score -= 0.3
            concerns.append(f"Contradicts known facts: {contradictions[0]}")
        else:
            score += 0.1
            evidence.append("No known contradictions")

        # Check data consistency
        if hypothesis:
            score += hypothesis.data_support * 0.2
            evidence.append(f"Data support: {hypothesis.data_support:.0%}")

        # Check for supporting facts
        supporting_facts = additional_evidence.get("supporting_facts", [])
        if supporting_facts:
            score += 0.2
            evidence.append(f"Supporting evidence: {len(supporting_facts)} facts")

        return CriterionAssessment(
            criterion="coherence",
            score=min(1.0, max(0.0, score)),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["coherence"],
        )

    def _assess_experiment(
        self,
        additional_evidence: Dict[str, Any]
    ) -> CriterionAssessment:
        """Assess experimental evidence."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate (often not available)

        # Check for experimental data
        experiments = additional_evidence.get("experimental_evidence", [])
        if experiments:
            positive = sum(1 for e in experiments if e.get("supports_causation"))
            negative = len(experiments) - positive
            if positive > negative:
                score = 0.8
                evidence.append(f"Experimental support: {positive}/{len(experiments)} studies")
            elif positive == negative:
                score = 0.5
                evidence.append("Mixed experimental results")
            else:
                score = 0.3
                concerns.append("Experimental evidence contradicts")
        else:
            concerns.append("No experimental evidence available")
            evidence.append("Assessment based on observational data only")

        return CriterionAssessment(
            criterion="experiment",
            score=score,
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["experiment"],
            applicable=bool(experiments),  # Only applicable if experiments exist
        )

    def _assess_analogy(
        self,
        correlation: "TemporalCorrelationResult"
    ) -> CriterionAssessment:
        """Assess analogy (similar relationships are causal)."""
        evidence = []
        concerns = []
        score = 0.5  # Default moderate

        cause = correlation.event_type_a.lower()
        effect = correlation.event_type_b.lower()

        # Check analogies database
        for pattern, analogies in self.analogies_database.items():
            if pattern in cause or pattern in effect:
                score += 0.1
                evidence.append(f"Similar patterns known: {pattern}")

        # Common causal patterns
        common_patterns = [
            ("sleep", ["mood", "energy", "performance", "health"]),
            ("exercise", ["mood", "sleep", "health", "energy"]),
            ("stress", ["sleep", "health", "mood", "performance"]),
            ("diet", ["energy", "health", "weight", "mood"]),
            ("work", ["stress", "sleep", "mood"]),
        ]

        for cause_pattern, effects in common_patterns:
            if cause_pattern in cause:
                for e in effects:
                    if e in effect:
                        score += 0.2
                        evidence.append(f"Known causal pattern: {cause_pattern} → {e}")
                        break

        if not evidence:
            concerns.append("No known analogous causal relationships")

        return CriterionAssessment(
            criterion="analogy",
            score=min(1.0, max(0.0, score)),
            strength=self._score_to_strength(score),
            evidence=evidence,
            concerns=concerns,
            weight=self.weights["analogy"],
        )

    def _score_to_strength(self, score: float) -> CriterionStrength:
        """Convert numeric score to strength level."""
        for strength, threshold in self.STRENGTH_THRESHOLDS.items():
            if score >= threshold:
                return strength
        return CriterionStrength.VERY_WEAK

    def _calculate_overall_score(
        self,
        assessments: Dict[str, CriterionAssessment]
    ) -> float:
        """Calculate weighted overall score."""
        total_weight = sum(
            a.weight for a in assessments.values() if a.applicable
        )
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(
            a.weighted_score for a in assessments.values()
        )

        return weighted_sum / total_weight

    def _determine_verdict(
        self,
        overall_score: float,
        assessments: Dict[str, CriterionAssessment]
    ) -> CausalityVerdict:
        """Determine overall causality verdict."""
        # Must have temporality for causation
        temporality = assessments.get("temporality")
        if temporality and temporality.score < 0.5:
            return CausalityVerdict.LIKELY_NOT_CAUSAL

        # Check against thresholds
        for verdict, threshold in self.VERDICT_THRESHOLDS.items():
            if overall_score >= threshold:
                return verdict

        return CausalityVerdict.LIKELY_NOT_CAUSAL

    def _generate_summary(
        self,
        cause: str,
        effect: str,
        assessments: Dict[str, CriterionAssessment],
        verdict: CausalityVerdict
    ) -> str:
        """Generate human-readable summary."""
        strong = [name for name, a in assessments.items()
                  if a.strength in [CriterionStrength.STRONG, CriterionStrength.VERY_STRONG]]
        weak = [name for name, a in assessments.items()
                if a.strength in [CriterionStrength.WEAK, CriterionStrength.VERY_WEAK]]

        summary_parts = [
            f"Bradford-Hill assessment of '{cause}' → '{effect}': {verdict.value.replace('_', ' ').title()}."
        ]

        if strong:
            summary_parts.append(f"Strong support from: {', '.join(strong)}.")
        if weak:
            summary_parts.append(f"Weak support from: {', '.join(weak)}.")

        return " ".join(summary_parts)

    def _generate_recommendations(
        self,
        assessments: Dict[str, CriterionAssessment]
    ) -> List[str]:
        """Generate recommendations for strengthening evidence."""
        recommendations = []

        for name, assessment in assessments.items():
            if assessment.strength in [CriterionStrength.WEAK, CriterionStrength.VERY_WEAK]:
                if name == "consistency":
                    recommendations.append("Collect more observations to strengthen consistency")
                elif name == "gradient":
                    recommendations.append("Investigate dose-response relationship")
                elif name == "plausibility":
                    recommendations.append("Research potential causal mechanisms")
                elif name == "experiment":
                    recommendations.append("Consider controlled experiments if feasible")
                elif name == "analogy":
                    recommendations.append("Research similar causal relationships in literature")

        if not recommendations:
            recommendations.append("Evidence base is strong; continue monitoring for consistency")

        return recommendations

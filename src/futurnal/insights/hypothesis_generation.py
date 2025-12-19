"""
LLM Hypothesis Generation Pipeline.

Bridges temporal correlation detection to causal hypothesis generation.
Uses LLM to generate structured causal hypotheses from statistical correlations.

Research Foundation:
- LeSR (2501.01246v1): LLM-enhanced symbolic reasoning
- ZIA (2502.16124v1): Proactive hypothesis generation
- ICDA: Interactive causal discovery validation

Pipeline Flow:
TemporalCorrelationResult → HypothesisGenerator → CausalHypothesis → ICDA

Option B Compliance:
- Uses LLM for hypothesis generation (not training)
- No model parameter updates
- Hypotheses stored as natural language priors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
from uuid import uuid4
import json

if TYPE_CHECKING:
    from futurnal.search.temporal.results import TemporalCorrelationResult
    from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent

logger = logging.getLogger(__name__)


class HypothesisType(str, Enum):
    """Types of causal hypotheses."""
    DIRECT_CAUSE = "direct_cause"  # A directly causes B
    INDIRECT_CAUSE = "indirect_cause"  # A causes B through mediator
    BIDIRECTIONAL = "bidirectional"  # A and B influence each other
    COMMON_CAUSE = "common_cause"  # C causes both A and B
    CONDITIONAL = "conditional"  # A causes B under condition C


class MechanismCategory(str, Enum):
    """Categories of causal mechanisms."""
    BEHAVIORAL = "behavioral"  # Human behavior patterns
    PHYSIOLOGICAL = "physiological"  # Body/health processes
    ENVIRONMENTAL = "environmental"  # External factors
    PSYCHOLOGICAL = "psychological"  # Mental/emotional
    SOCIAL = "social"  # Interpersonal dynamics
    TEMPORAL = "temporal"  # Time-based mechanisms
    UNKNOWN = "unknown"


@dataclass
class CausalMechanism:
    """A proposed mechanism for causal relationship."""
    description: str
    category: MechanismCategory = MechanismCategory.UNKNOWN
    plausibility: float = 0.5  # 0-1 scale
    supporting_evidence: List[str] = field(default_factory=list)
    required_conditions: List[str] = field(default_factory=list)


@dataclass
class AlternativeExplanation:
    """An alternative explanation for observed correlation."""
    explanation: str
    explanation_type: str  # "confounder", "reverse_causation", "coincidence", "mediator"
    likelihood: float = 0.5
    potential_confounder: Optional[str] = None


@dataclass
class CausalHypothesis:
    """A structured causal hypothesis generated from correlation.

    This is the bridge between statistical correlation detection
    and interactive causal discovery.
    """
    hypothesis_id: str = field(default_factory=lambda: str(uuid4()))

    # Core relationship
    cause_event: str = ""
    effect_event: str = ""
    hypothesis_type: HypothesisType = HypothesisType.DIRECT_CAUSE

    # Natural language components
    hypothesis_statement: str = ""  # "X causes Y because..."
    mechanism_description: str = ""  # How the causation works
    testable_prediction: str = ""  # What would we expect if true

    # Proposed mechanisms
    mechanisms: List[CausalMechanism] = field(default_factory=list)

    # Alternative explanations
    alternatives: List[AlternativeExplanation] = field(default_factory=list)

    # Confidence assessment
    prior_confidence: float = 0.5  # Before validation
    mechanism_strength: float = 0.5  # How strong the proposed mechanism is
    data_support: float = 0.5  # How well data supports hypothesis

    # Statistical grounding
    correlation_strength: float = 0.0
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    temporal_consistency: float = 0.0  # Gap consistency
    sample_size: int = 0

    # Bradford-Hill criteria scores (0-1)
    bradford_hill: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_correlation_id: Optional[str] = None
    domain_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "cause_event": self.cause_event,
            "effect_event": self.effect_event,
            "hypothesis_type": self.hypothesis_type.value,
            "hypothesis_statement": self.hypothesis_statement,
            "mechanism_description": self.mechanism_description,
            "testable_prediction": self.testable_prediction,
            "prior_confidence": self.prior_confidence,
            "mechanism_strength": self.mechanism_strength,
            "data_support": self.data_support,
            "bradford_hill": self.bradford_hill,
            "created_at": self.created_at.isoformat(),
        }

    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        weights = {
            "prior": 0.3,
            "mechanism": 0.3,
            "data": 0.4,
        }
        return (
            weights["prior"] * self.prior_confidence +
            weights["mechanism"] * self.mechanism_strength +
            weights["data"] * self.data_support
        )


class HypothesisGenerator:
    """
    Generates causal hypotheses from temporal correlations using LLM.

    This is the core bridge between statistical pattern detection
    and structured causal reasoning.

    Workflow:
    1. Receive correlation results from TemporalCorrelationDetector
    2. Use LLM to generate plausible causal mechanisms
    3. Identify alternative explanations
    4. Score hypothesis using Bradford-Hill criteria
    5. Create structured CausalHypothesis for ICDA validation
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        min_confidence_threshold: float = 0.3,
        domain_context: Optional[str] = None
    ):
        """Initialize hypothesis generator.

        Args:
            llm_client: LLM client for generation (optional)
            min_confidence_threshold: Minimum confidence to generate hypothesis
            domain_context: Domain context for better hypotheses (e.g., "personal health")
        """
        self.llm_client = llm_client
        self.min_confidence_threshold = min_confidence_threshold
        self.domain_context = domain_context or "personal knowledge management"

        # Cache for generated hypotheses
        self._hypothesis_cache: Dict[str, CausalHypothesis] = {}

    async def generate_hypothesis(
        self,
        correlation: "TemporalCorrelationResult",
        additional_context: Optional[str] = None
    ) -> Optional[CausalHypothesis]:
        """Generate causal hypothesis from correlation.

        Args:
            correlation: Temporal correlation result
            additional_context: Additional context for LLM

        Returns:
            CausalHypothesis if generation successful, None otherwise
        """
        # Check if correlation is suitable
        if not self._is_suitable_correlation(correlation):
            logger.debug(f"Correlation not suitable for hypothesis: {correlation.event_type_a} -> {correlation.event_type_b}")
            return None

        # Build hypothesis components
        cause = correlation.event_type_a
        effect = correlation.event_type_b

        # Generate using LLM if available, otherwise use rule-based
        if self.llm_client:
            hypothesis = await self._generate_with_llm(correlation, additional_context)
        else:
            hypothesis = self._generate_rule_based(correlation)

        if hypothesis:
            # Score with Bradford-Hill criteria
            hypothesis.bradford_hill = self._score_bradford_hill(correlation, hypothesis)

            # Update confidence based on Bradford-Hill scores
            bh_score = sum(hypothesis.bradford_hill.values()) / len(hypothesis.bradford_hill) if hypothesis.bradford_hill else 0.5
            hypothesis.prior_confidence = (hypothesis.prior_confidence + bh_score) / 2

            # Cache
            cache_key = f"{cause}|{effect}"
            self._hypothesis_cache[cache_key] = hypothesis

            logger.info(
                f"Generated hypothesis: {cause} -> {effect} "
                f"(confidence: {hypothesis.overall_confidence:.2f})"
            )

        return hypothesis

    def generate_hypothesis_sync(
        self,
        correlation: "TemporalCorrelationResult",
        additional_context: Optional[str] = None
    ) -> Optional[CausalHypothesis]:
        """Synchronous version of generate_hypothesis."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_hypothesis(correlation, additional_context)
        )

    async def generate_batch(
        self,
        correlations: List["TemporalCorrelationResult"],
        max_hypotheses: int = 10
    ) -> List[CausalHypothesis]:
        """Generate hypotheses for multiple correlations.

        Args:
            correlations: List of correlation results
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            List of generated hypotheses
        """
        hypotheses = []

        # Sort by potential importance
        sorted_correlations = sorted(
            correlations,
            key=lambda c: (c.correlation_strength or 0) * (c.co_occurrences or 0),
            reverse=True
        )

        for correlation in sorted_correlations[:max_hypotheses * 2]:  # Try more than needed
            hypothesis = await self.generate_hypothesis(correlation)
            if hypothesis:
                hypotheses.append(hypothesis)
                if len(hypotheses) >= max_hypotheses:
                    break

        return hypotheses

    def _is_suitable_correlation(self, correlation: "TemporalCorrelationResult") -> bool:
        """Check if correlation is suitable for hypothesis generation."""
        if not correlation.correlation_found:
            return False

        if not correlation.is_causal_candidate:
            return False

        strength = correlation.correlation_strength or 0
        if strength < self.min_confidence_threshold:
            return False

        occurrences = correlation.co_occurrences or 0
        if occurrences < 3:  # Minimum data points
            return False

        return True

    async def _generate_with_llm(
        self,
        correlation: "TemporalCorrelationResult",
        additional_context: Optional[str]
    ) -> Optional[CausalHypothesis]:
        """Generate hypothesis using LLM."""
        prompt = self._build_generation_prompt(correlation, additional_context)

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = await self.llm_client.chat([
                    {"role": "user", "content": prompt}
                ])
                response = response.get("content", "")
            else:
                return self._generate_rule_based(correlation)

            return self._parse_llm_response(correlation, response)

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(correlation)

    def _build_generation_prompt(
        self,
        correlation: "TemporalCorrelationResult",
        additional_context: Optional[str]
    ) -> str:
        """Build prompt for LLM hypothesis generation."""
        cause = correlation.event_type_a
        effect = correlation.event_type_b
        gap = correlation.avg_gap_days or 0
        occurrences = correlation.co_occurrences or 0
        strength = correlation.correlation_strength or 0

        prompt = f"""Generate a causal hypothesis for the following observed correlation in {self.domain_context}:

## Observed Pattern
- Event A: "{cause}"
- Event B: "{effect}"
- Temporal relationship: A typically precedes B by {gap:.1f} days
- Occurrences: {occurrences} times
- Correlation strength: {strength:.0%}
"""

        if correlation.p_value:
            prompt += f"- Statistical significance: p={correlation.p_value:.4f}\n"

        if additional_context:
            prompt += f"\n## Additional Context\n{additional_context}\n"

        prompt += """
## Required Output
Please provide a structured hypothesis with:

1. HYPOTHESIS_STATEMENT: A clear statement of the causal relationship (1-2 sentences)

2. MECHANISM: How does the first event causally lead to the second? What is the process?

3. MECHANISM_CATEGORY: One of: behavioral, physiological, environmental, psychological, social, temporal

4. TESTABLE_PREDICTION: What would we expect to observe if this hypothesis is true?

5. ALTERNATIVE_EXPLANATIONS: List 2-3 alternative explanations for this correlation:
   - Potential confounders (third factors causing both)
   - Reverse causation possibility
   - Coincidence factors

6. PLAUSIBILITY_SCORE: 0.0-1.0 rating of how plausible this mechanism is

7. HYPOTHESIS_TYPE: One of: direct_cause, indirect_cause, bidirectional, common_cause, conditional

Format your response with clear section headers."""

        return prompt

    def _parse_llm_response(
        self,
        correlation: "TemporalCorrelationResult",
        response: str
    ) -> Optional[CausalHypothesis]:
        """Parse LLM response into CausalHypothesis."""
        cause = correlation.event_type_a
        effect = correlation.event_type_b

        # Initialize hypothesis
        hypothesis = CausalHypothesis(
            cause_event=cause,
            effect_event=effect,
            correlation_strength=correlation.correlation_strength or 0,
            p_value=correlation.p_value,
            temporal_consistency=correlation.gap_consistency or 0,
            sample_size=correlation.co_occurrences or 0,
        )

        # Parse sections from response
        lines = response.strip().split("\n")
        current_section = None
        section_content = []

        for line in lines:
            line_upper = line.upper().strip()

            if "HYPOTHESIS_STATEMENT" in line_upper:
                current_section = "statement"
                section_content = []
            elif "MECHANISM:" in line_upper or "MECHANISM_DESCRIPTION" in line_upper:
                if current_section == "statement":
                    hypothesis.hypothesis_statement = " ".join(section_content).strip()
                current_section = "mechanism"
                section_content = []
            elif "MECHANISM_CATEGORY" in line_upper:
                if current_section == "mechanism":
                    hypothesis.mechanism_description = " ".join(section_content).strip()
                current_section = "category"
                section_content = []
            elif "TESTABLE_PREDICTION" in line_upper:
                current_section = "prediction"
                section_content = []
            elif "ALTERNATIVE" in line_upper:
                if current_section == "prediction":
                    hypothesis.testable_prediction = " ".join(section_content).strip()
                current_section = "alternatives"
                section_content = []
            elif "PLAUSIBILITY" in line_upper:
                current_section = "plausibility"
                section_content = []
            elif "HYPOTHESIS_TYPE" in line_upper:
                current_section = "type"
                section_content = []
            elif current_section:
                section_content.append(line)

        # Handle last section
        if current_section and section_content:
            content = " ".join(section_content).strip()
            if current_section == "statement":
                hypothesis.hypothesis_statement = content
            elif current_section == "mechanism":
                hypothesis.mechanism_description = content
            elif current_section == "prediction":
                hypothesis.testable_prediction = content
            elif current_section == "plausibility":
                try:
                    score = float(content.split()[0])
                    hypothesis.mechanism_strength = min(1.0, max(0.0, score))
                except (ValueError, IndexError):
                    pass
            elif current_section == "type":
                content_lower = content.lower()
                for ht in HypothesisType:
                    if ht.value in content_lower:
                        hypothesis.hypothesis_type = ht
                        break

        # Parse alternatives
        hypothesis.alternatives = self._parse_alternatives(response)

        # Generate fallback statements if missing
        if not hypothesis.hypothesis_statement:
            hypothesis.hypothesis_statement = f"{cause} may cause {effect} based on observed temporal patterns."
        if not hypothesis.testable_prediction:
            hypothesis.testable_prediction = f"If {cause} causes {effect}, reducing {cause} should reduce {effect}."

        # Calculate data support
        hypothesis.data_support = self._calculate_data_support(correlation)

        return hypothesis

    def _parse_alternatives(self, response: str) -> List[AlternativeExplanation]:
        """Parse alternative explanations from LLM response."""
        alternatives = []
        response_lower = response.lower()

        # Look for confounder mentions
        if "confounder" in response_lower or "third factor" in response_lower:
            alternatives.append(AlternativeExplanation(
                explanation="A third factor may cause both events",
                explanation_type="confounder",
                likelihood=0.3
            ))

        # Look for reverse causation
        if "reverse" in response_lower:
            alternatives.append(AlternativeExplanation(
                explanation="The causation may be reversed",
                explanation_type="reverse_causation",
                likelihood=0.2
            ))

        # Look for coincidence
        if "coincidence" in response_lower or "chance" in response_lower:
            alternatives.append(AlternativeExplanation(
                explanation="The correlation may be coincidental",
                explanation_type="coincidence",
                likelihood=0.2
            ))

        return alternatives

    def _generate_rule_based(
        self,
        correlation: "TemporalCorrelationResult"
    ) -> CausalHypothesis:
        """Generate hypothesis using rule-based approach (no LLM)."""
        cause = correlation.event_type_a
        effect = correlation.event_type_b
        gap = correlation.avg_gap_days or 0
        occurrences = correlation.co_occurrences or 0
        strength = correlation.correlation_strength or 0

        # Generate statement
        if gap < 1:
            time_phrase = "shortly after"
        elif gap < 7:
            time_phrase = f"about {gap:.0f} days after"
        else:
            time_phrase = f"approximately {gap:.0f} days after"

        hypothesis_statement = (
            f"The occurrence of '{cause}' may causally contribute to '{effect}' "
            f"occurring {time_phrase}. This pattern was observed {occurrences} times "
            f"with {strength:.0%} correlation strength."
        )

        # Generate mechanism based on common patterns
        mechanism = self._infer_mechanism(cause, effect)

        # Generate prediction
        testable_prediction = (
            f"If this hypothesis is correct, interventions that reduce '{cause}' "
            f"should lead to corresponding reductions in '{effect}'."
        )

        # Generate alternatives
        alternatives = [
            AlternativeExplanation(
                explanation=f"An unmeasured factor may cause both '{cause}' and '{effect}'",
                explanation_type="confounder",
                likelihood=0.3
            ),
            AlternativeExplanation(
                explanation=f"'{effect}' might actually lead to '{cause}' (reverse causation)",
                explanation_type="reverse_causation",
                likelihood=0.2
            ),
        ]

        # Calculate confidence scores
        data_support = self._calculate_data_support(correlation)
        mechanism_strength = mechanism.plausibility

        return CausalHypothesis(
            cause_event=cause,
            effect_event=effect,
            hypothesis_statement=hypothesis_statement,
            mechanism_description=mechanism.description,
            testable_prediction=testable_prediction,
            mechanisms=[mechanism],
            alternatives=alternatives,
            prior_confidence=0.4 + (strength * 0.2),
            mechanism_strength=mechanism_strength,
            data_support=data_support,
            correlation_strength=strength,
            p_value=correlation.p_value,
            temporal_consistency=correlation.gap_consistency or 0,
            sample_size=occurrences,
        )

    def _infer_mechanism(self, cause: str, effect: str) -> CausalMechanism:
        """Infer causal mechanism from event types."""
        cause_lower = cause.lower()
        effect_lower = effect.lower()

        # Pattern matching for common mechanisms
        if any(w in cause_lower for w in ["sleep", "rest", "tired"]):
            return CausalMechanism(
                description=f"Sleep/rest quality affects subsequent {effect} through physiological recovery processes",
                category=MechanismCategory.PHYSIOLOGICAL,
                plausibility=0.6
            )

        if any(w in cause_lower for w in ["exercise", "workout", "activity"]):
            return CausalMechanism(
                description=f"Physical activity influences {effect} through metabolic and hormonal changes",
                category=MechanismCategory.PHYSIOLOGICAL,
                plausibility=0.7
            )

        if any(w in cause_lower for w in ["stress", "anxiety", "worry"]):
            return CausalMechanism(
                description=f"Stress/emotional state affects {effect} through psychological and physiological pathways",
                category=MechanismCategory.PSYCHOLOGICAL,
                plausibility=0.6
            )

        if any(w in cause_lower for w in ["meeting", "work", "project"]):
            return CausalMechanism(
                description=f"Work activities influence {effect} through behavioral and scheduling patterns",
                category=MechanismCategory.BEHAVIORAL,
                plausibility=0.5
            )

        if any(w in cause_lower for w in ["social", "friend", "family"]):
            return CausalMechanism(
                description=f"Social interactions influence {effect} through interpersonal dynamics",
                category=MechanismCategory.SOCIAL,
                plausibility=0.5
            )

        # Default mechanism
        return CausalMechanism(
            description=f"The occurrence of {cause} creates conditions that lead to {effect}",
            category=MechanismCategory.UNKNOWN,
            plausibility=0.4
        )

    def _calculate_data_support(
        self,
        correlation: "TemporalCorrelationResult"
    ) -> float:
        """Calculate data support score from correlation statistics."""
        score = 0.3  # Base score

        # Boost from correlation strength
        strength = correlation.correlation_strength or 0
        score += strength * 0.3

        # Boost from statistical significance
        if correlation.p_value is not None:
            if correlation.p_value < 0.01:
                score += 0.2
            elif correlation.p_value < 0.05:
                score += 0.15
            elif correlation.p_value < 0.1:
                score += 0.1

        # Boost from sample size
        occurrences = correlation.co_occurrences or 0
        if occurrences >= 20:
            score += 0.1
        elif occurrences >= 10:
            score += 0.05

        # Boost from gap consistency
        if correlation.gap_consistency:
            score += correlation.gap_consistency * 0.1

        return min(1.0, score)

    def _score_bradford_hill(
        self,
        correlation: "TemporalCorrelationResult",
        hypothesis: CausalHypothesis
    ) -> Dict[str, float]:
        """Score hypothesis against Bradford-Hill criteria.

        Returns dict with scores for each criterion.
        """
        scores = {}

        # 1. Strength of association
        strength = correlation.correlation_strength or 0
        scores["strength"] = strength

        # 2. Consistency (temporal consistency)
        scores["consistency"] = correlation.gap_consistency or 0.5

        # 3. Specificity (inverse of alternatives)
        num_alternatives = len(hypothesis.alternatives)
        scores["specificity"] = max(0.3, 1.0 - (num_alternatives * 0.2))

        # 4. Temporality (A precedes B - always 1.0 for temporal correlations)
        scores["temporality"] = 1.0 if (correlation.avg_gap_days or 0) > 0 else 0.5

        # 5. Biological gradient / Dose-response (proxy from effect size)
        effect_size = correlation.effect_size or 0.5
        scores["gradient"] = min(1.0, effect_size)

        # 6. Plausibility (from mechanism)
        scores["plausibility"] = hypothesis.mechanism_strength

        # 7. Coherence (from data support)
        scores["coherence"] = hypothesis.data_support

        # 8. Experiment (not applicable for observational data)
        scores["experiment"] = 0.5  # Neutral

        # 9. Analogy (future: check similar patterns)
        scores["analogy"] = 0.5  # Neutral for now

        return scores


class HypothesisPipeline:
    """
    Complete pipeline connecting correlation detection to causal validation.

    Orchestrates:
    1. Correlation detection
    2. Hypothesis generation
    3. ICDA validation
    4. Knowledge storage
    """

    def __init__(
        self,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
        icda_agent: Optional["InteractiveCausalDiscoveryAgent"] = None,
        llm_client: Optional[Any] = None
    ):
        self.generator = hypothesis_generator or HypothesisGenerator(llm_client=llm_client)
        self.icda_agent = icda_agent

        # Pipeline statistics
        self.stats = {
            "correlations_processed": 0,
            "hypotheses_generated": 0,
            "hypotheses_validated": 0,
            "causal_confirmed": 0,
            "causal_rejected": 0,
        }

    async def process_correlations(
        self,
        correlations: List["TemporalCorrelationResult"],
        max_hypotheses: int = 10
    ) -> List[CausalHypothesis]:
        """Process correlations through the full pipeline.

        Args:
            correlations: List of correlation results
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            List of generated hypotheses
        """
        self.stats["correlations_processed"] += len(correlations)

        # Generate hypotheses
        hypotheses = await self.generator.generate_batch(correlations, max_hypotheses)
        self.stats["hypotheses_generated"] += len(hypotheses)

        # Submit to ICDA for validation if available
        if self.icda_agent:
            for hypothesis in hypotheses:
                # Create a mock correlation result for ICDA
                # (ICDA expects TemporalCorrelationResult interface)
                self._submit_to_icda(hypothesis)

        return hypotheses

    def _submit_to_icda(self, hypothesis: CausalHypothesis):
        """Submit hypothesis to ICDA for validation."""
        if not self.icda_agent:
            return

        try:
            # Create a mock correlation-like object that ICDA can process
            class MockCorrelation:
                def __init__(self, h: CausalHypothesis):
                    self.event_type_a = h.cause_event
                    self.event_type_b = h.effect_event
                    self.avg_gap_days = 1.0  # Placeholder
                    self.co_occurrences = h.sample_size
                    self.correlation_strength = h.overall_confidence
                    self.p_value = h.p_value
                    self.is_causal_candidate = True

            mock = MockCorrelation(hypothesis)
            self.icda_agent.add_candidate_from_correlation(mock)

        except Exception as e:
            logger.warning(f"Failed to submit hypothesis to ICDA: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary statistics."""
        return {
            **self.stats,
            "generator_cache_size": len(self.generator._hypothesis_cache),
            "icda_pending": (
                len(self.icda_agent._questions) if self.icda_agent else 0
            ),
        }

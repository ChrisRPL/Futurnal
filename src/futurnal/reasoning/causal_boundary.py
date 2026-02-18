"""
Explicit Causal Boundary Abstraction Layer.

Research Foundation:
- arxiv:2510.07231 - LLM causal benchmarks (best: 57.6% accuracy)
- arxiv:2503.00237 - Agentic AI Needs Systems Theory (2025)

This module enforces the critical separation:
- LLMs identify CORRELATIONS (pattern matching - their strength)
- Structured algorithms validate CAUSATION (Bradford-Hill, temporal ordering)
- Humans provide final verification (domain knowledge)

Three-Layer Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    FUTURNAL CAUSAL ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: Pattern Detection (LLM-Powered)                       │
│  ├── Correlation identification in experiential data            │
│  ├── Hypothesis generation from temporal patterns               │
│  └── Natural language insight formulation                       │
│                           ↓                                      │
│  LAYER 2: Causal Validation (Structured Algorithms)             │
│  ├── Bradford Hill criteria scoring (9 criteria)                │
│  ├── Statistical temporal tests                                  │
│  └── Confounding factor detection                               │
│                           ↓                                      │
│  LAYER 3: Human Verification (User-in-Loop)                     │
│  ├── ICDA conversational exploration                            │
│  ├── Evidence presentation with confidence levels               │
│  └── Final judgment preserved for user                          │
└─────────────────────────────────────────────────────────────────┘

Option B Compliance:
- Ghost model FROZEN
- No parameter updates
- Outputs stored as token priors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid4

if TYPE_CHECKING:
    from futurnal.search.temporal.results import TemporalCorrelationResult
    from futurnal.search.causal.bradford_hill import BradfordHillValidator, BradfordHillReport
    from futurnal.insights.hypothesis_generation import HypothesisGenerator, CausalHypothesis
    from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent

logger = logging.getLogger(__name__)


class ConfidenceType(str, Enum):
    """Types of confidence scores.

    Critical distinction:
    - CORRELATION: LLM-derived, pattern-based (Layer 1)
    - CAUSAL: Algorithm-validated, Bradford-Hill (Layer 2)
    - VERIFIED: Human-verified through ICDA (Layer 3)
    """
    CORRELATION = "correlation"  # LLM-derived pattern confidence
    CAUSAL = "causal"  # Algorithm-validated causal confidence
    VERIFIED = "verified"  # Human-verified confidence


@dataclass
class CausalBoundaryResult:
    """Result from CausalBoundary processing.

    Maintains explicit separation between correlation and causal confidence.
    This is the core data structure that enforces the LLM/causal boundary.

    Research Basis:
    - arxiv:2510.07231: LLMs achieve only 57.6% on causal benchmarks
    - Correlation confidence != Causal confidence

    Option B Compliance:
    - All fields are natural language or simple values
    - Can be exported as token priors
    - No model parameter updates
    """
    result_id: str = field(default_factory=lambda: str(uuid4()))

    # Source data
    cause_event: str = ""
    effect_event: str = ""

    # Layer 1: Correlation confidence (LLM-derived)
    correlation_confidence: float = 0.0
    correlation_evidence: List[str] = field(default_factory=list)
    hypothesis_statement: Optional[str] = None
    mechanism_description: Optional[str] = None

    # Layer 2: Causal confidence (Algorithm-validated)
    causal_confidence: float = 0.0
    bradford_hill_score: float = 0.0
    bradford_hill_verdict: Optional[str] = None
    bradford_hill_report: Optional[Dict[str, Any]] = None

    # Layer 3: Verification status (Human-verified)
    verification_status: str = "pending"  # pending, verified_causal, verified_non_causal
    verified_confidence: Optional[float] = None
    user_explanation: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None

    def get_display_confidence(self) -> Tuple[float, ConfidenceType]:
        """Get the most authoritative confidence for display.

        Priority: verified > causal > correlation

        Returns:
            Tuple of (confidence_value, confidence_type)
        """
        if self.verified_confidence is not None:
            return (self.verified_confidence, ConfidenceType.VERIFIED)
        elif self.causal_confidence > 0:
            return (self.causal_confidence, ConfidenceType.CAUSAL)
        else:
            return (self.correlation_confidence, ConfidenceType.CORRELATION)

    def to_natural_language(self) -> str:
        """Convert to natural language for token priors.

        Option B Compliance: Knowledge as text, not weights.
        """
        conf, conf_type = self.get_display_confidence()

        lines = [
            f"Relationship: {self.cause_event} -> {self.effect_event}",
            f"Correlation confidence: {self.correlation_confidence:.0%} (LLM pattern detection)",
            f"Causal confidence: {self.causal_confidence:.0%} (Bradford-Hill validation)",
        ]

        if self.bradford_hill_verdict:
            lines.append(f"Bradford-Hill verdict: {self.bradford_hill_verdict}")

        if self.verified_confidence is not None:
            lines.append(f"Verified confidence: {self.verified_confidence:.0%} (human verified)")

        if self.hypothesis_statement:
            lines.append(f"Hypothesis: {self.hypothesis_statement}")

        if self.mechanism_description:
            lines.append(f"Mechanism: {self.mechanism_description}")

        lines.append(f"Display confidence: {conf:.0%} ({conf_type.value})")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "result_id": self.result_id,
            "cause_event": self.cause_event,
            "effect_event": self.effect_event,
            "correlation_confidence": self.correlation_confidence,
            "correlation_evidence": self.correlation_evidence,
            "hypothesis_statement": self.hypothesis_statement,
            "mechanism_description": self.mechanism_description,
            "causal_confidence": self.causal_confidence,
            "bradford_hill_score": self.bradford_hill_score,
            "bradford_hill_verdict": self.bradford_hill_verdict,
            "verification_status": self.verification_status,
            "verified_confidence": self.verified_confidence,
            "created_at": self.created_at.isoformat(),
        }


class CausalBoundary:
    """Explicit separation between LLM pattern detection and causal validation.

    This is the core abstraction that addresses the fundamental limitation
    that LLMs cannot perform genuine causal reasoning.

    Research Basis:
    - arxiv:2510.07231 (LLM causal benchmarks 2025): Best LLM achieves 57.6%
    - arxiv:2503.00237 (Agentic AI Systems Theory): Structured validation needed

    The CausalBoundary enforces the three-layer architecture:

    Layer 1: Pattern Detection (LLM-Powered)
    - Uses LLM for hypothesis generation (what it's good at)
    - Produces correlation_confidence (NOT causal confidence)

    Layer 2: Causal Validation (Structured Algorithms)
    - Bradford-Hill criteria scoring (9 criteria)
    - Temporal ordering validation
    - Produces causal_confidence

    Layer 3: Human Verification (User-in-Loop)
    - ICDA conversational exploration
    - User confirms or rejects hypothesis
    - Produces verified_confidence

    Option B Compliance:
    - Ghost model FROZEN
    - Learning through token priors only
    - No parameter updates

    Example:
        >>> boundary = create_causal_boundary(llm_client=client)
        >>> result = await boundary.process_correlation(correlation)
        >>> print(f"Correlation: {result.correlation_confidence:.0%}")
        >>> print(f"Causal: {result.causal_confidence:.0%}")
    """

    def __init__(
        self,
        hypothesis_generator: Optional["HypothesisGenerator"] = None,
        bradford_hill_validator: Optional["BradfordHillValidator"] = None,
        icda_agent: Optional["InteractiveCausalDiscoveryAgent"] = None,
    ):
        """Initialize CausalBoundary.

        Args:
            hypothesis_generator: LLM-powered hypothesis generator (Layer 1)
            bradford_hill_validator: Bradford-Hill criteria validator (Layer 2)
            icda_agent: Interactive Causal Discovery Agent (Layer 3)
        """
        self._hypothesis_generator = hypothesis_generator
        self._bradford_hill_validator = bradford_hill_validator
        self._icda_agent = icda_agent

        # Results cache
        self._results_cache: Dict[str, CausalBoundaryResult] = {}

        # Statistics
        self._correlations_processed = 0
        self._causal_validated = 0
        self._submitted_for_verification = 0

        logger.info("CausalBoundary initialized - enforcing LLM/causal separation")

    async def process_correlation(
        self,
        correlation: "TemporalCorrelationResult",
        submit_for_verification: bool = False,
    ) -> CausalBoundaryResult:
        """Process correlation through all three layers.

        This is the main entry point for the CausalBoundary.

        Flow:
        1. Layer 1: LLM generates hypothesis -> correlation_confidence
        2. Layer 2: Bradford-Hill validates -> causal_confidence
        3. Layer 3: Optionally submit to ICDA -> pending verification

        Args:
            correlation: Detected temporal correlation
            submit_for_verification: Whether to submit to ICDA for human verification

        Returns:
            CausalBoundaryResult with both correlation and causal confidence
        """
        import time
        start_time = time.time()

        cause = correlation.event_type_a
        effect = correlation.event_type_b
        cache_key = f"{cause}|{effect}"

        # Initialize result
        result = CausalBoundaryResult(
            cause_event=cause,
            effect_event=effect,
        )

        # LAYER 1: LLM Pattern Detection (Correlation)
        result.correlation_confidence = await self._layer1_pattern_detection(
            correlation, result
        )

        # LAYER 2: Causal Validation (Bradford-Hill)
        result.causal_confidence, result.bradford_hill_score = await self._layer2_causal_validation(
            correlation, result
        )

        # LAYER 3: Human Verification (Optional)
        if submit_for_verification:
            self._layer3_submit_for_verification(correlation, result)

        # Record timing
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Cache result
        self._results_cache[cache_key] = result
        self._correlations_processed += 1

        logger.info(
            f"CausalBoundary processed {cause} -> {effect}: "
            f"correlation={result.correlation_confidence:.2f}, "
            f"causal={result.causal_confidence:.2f}"
        )

        return result

    async def _layer1_pattern_detection(
        self,
        correlation: "TemporalCorrelationResult",
        result: CausalBoundaryResult,
    ) -> float:
        """Layer 1: LLM-powered pattern detection.

        LLMs are GOOD at:
        - Identifying co-occurrence patterns
        - Generating natural language hypotheses
        - Semantic similarity matching

        LLMs are BAD at:
        - Distinguishing correlation from causation
        - Accounting for confounders
        - Formal causal reasoning

        This layer produces CORRELATION confidence, not causal confidence.

        Returns:
            Correlation confidence (0-1)
        """
        # Use statistical correlation strength as base
        correlation_confidence = correlation.correlation_strength or 0.5

        # Add statistical evidence
        if correlation.p_value is not None:
            result.correlation_evidence.append(f"p-value: {correlation.p_value:.4f}")
            # Boost confidence for significant results
            if correlation.p_value < 0.01:
                correlation_confidence = min(1.0, correlation_confidence + 0.1)
            elif correlation.p_value < 0.05:
                correlation_confidence = min(1.0, correlation_confidence + 0.05)

        if correlation.effect_size is not None:
            result.correlation_evidence.append(f"Effect size: {correlation.effect_size:.2f}")
            # Boost for large effect sizes
            if correlation.effect_size > 2.0:
                correlation_confidence = min(1.0, correlation_confidence + 0.1)

        if correlation.co_occurrences is not None:
            result.correlation_evidence.append(f"Co-occurrences: {correlation.co_occurrences}")

        # Generate hypothesis if generator available
        if self._hypothesis_generator:
            try:
                hypothesis = await self._hypothesis_generator.generate_hypothesis(correlation)
                if hypothesis:
                    # Use hypothesis confidence but label it as CORRELATION
                    correlation_confidence = max(correlation_confidence, hypothesis.overall_confidence)
                    result.hypothesis_statement = hypothesis.hypothesis_statement
                    result.mechanism_description = hypothesis.mechanism_description
                    result.correlation_evidence.append(
                        f"Hypothesis generated: {hypothesis.hypothesis_statement[:100]}..."
                    )
            except Exception as e:
                logger.warning(f"Hypothesis generation failed: {e}")

        return min(1.0, correlation_confidence)

    async def _layer2_causal_validation(
        self,
        correlation: "TemporalCorrelationResult",
        result: CausalBoundaryResult,
    ) -> Tuple[float, float]:
        """Layer 2: Algorithm-based causal validation.

        Structured algorithms are GOOD at:
        - Bradford-Hill criteria scoring
        - Temporal ordering validation
        - Statistical significance testing

        This layer produces CAUSAL confidence through structured validation.

        Returns:
            Tuple of (causal_confidence, bradford_hill_score)
        """
        causal_confidence = 0.0
        bradford_hill_score = 0.0

        if self._bradford_hill_validator:
            try:
                report = await self._bradford_hill_validator.validate(correlation)
                bradford_hill_score = report.overall_score
                causal_confidence = bradford_hill_score

                # Store report details
                result.bradford_hill_verdict = report.verdict.value
                result.bradford_hill_report = {
                    "overall_score": report.overall_score,
                    "verdict": report.verdict.value,
                    "summary": report.summary,
                    "recommendations": report.recommendations,
                }

                # Apply verdict-based adjustment
                if report.verdict.value == "likely_causal":
                    causal_confidence = min(0.9, causal_confidence + 0.1)
                elif report.verdict.value == "likely_not_causal":
                    causal_confidence = max(0.1, causal_confidence - 0.2)

                self._causal_validated += 1

            except Exception as e:
                logger.warning(f"Bradford-Hill validation failed: {e}")
                # Fall back to correlation confidence with penalty
                # Correlation != Causation, so apply significant discount
                causal_confidence = result.correlation_confidence * 0.4
        else:
            # Without validator, use conservative estimate
            # Correlation != Causation, so apply significant discount
            causal_confidence = result.correlation_confidence * 0.4
            result.bradford_hill_verdict = "not_validated"

        return (causal_confidence, bradford_hill_score)

    def _layer3_submit_for_verification(
        self,
        correlation: "TemporalCorrelationResult",
        result: CausalBoundaryResult,
    ) -> None:
        """Layer 3: Submit to ICDA for human verification.

        The user has the final say on causal relationships.
        """
        if not self._icda_agent:
            logger.warning("ICDA agent not available for verification")
            return

        try:
            candidate = self._icda_agent.add_candidate_from_correlation(correlation)
            if candidate:
                result.verification_status = "pending"
                self._submitted_for_verification += 1
                logger.info(f"Submitted for verification: {result.cause_event} -> {result.effect_event}")
        except Exception as e:
            logger.warning(f"Failed to submit for verification: {e}")

    def compute_causal_discount(
        self,
        correlation_confidence: float,
        is_causal_candidate: bool = False,
        co_occurrences: int = 0,
        effect_size: Optional[float] = None,
    ) -> float:
        """Compute discounted causal confidence from correlation confidence.

        This is a synchronous method for use in insight generation.
        The full async process_correlation() provides more rigorous validation.

        Key principle (arxiv:2510.07231):
        LLMs achieve only 57.6% on causal benchmarks, so we discount
        correlation confidence when claiming causation.

        Discount factors:
        - Base discount: 0.4 (correlation != causation)
        - Causal candidate bonus: +0.2 (passed statistical tests)
        - High co-occurrences bonus: +0.1 (more evidence)
        - Large effect size bonus: +0.1 (stronger effect)

        Args:
            correlation_confidence: LLM-derived correlation confidence
            is_causal_candidate: Whether it passed causal candidate tests
            co_occurrences: Number of observed co-occurrences
            effect_size: Cohen's d or similar effect size metric

        Returns:
            Discounted causal confidence (always <= correlation_confidence)
        """
        # Base discount: 0.4 (correlation != causation)
        discount_factor = 0.4

        # Bonus for passing causal candidate tests
        if is_causal_candidate:
            discount_factor += 0.2

        # Bonus for high co-occurrences (more evidence)
        if co_occurrences >= 10:
            discount_factor += 0.1
        elif co_occurrences >= 5:
            discount_factor += 0.05

        # Bonus for large effect size
        if effect_size is not None:
            if effect_size >= 0.8:  # Large effect
                discount_factor += 0.1
            elif effect_size >= 0.5:  # Medium effect
                discount_factor += 0.05

        # Apply discount (cap at 0.8 to always leave room for human validation)
        causal_confidence = correlation_confidence * min(discount_factor, 0.8)

        return min(causal_confidence, correlation_confidence)  # Never exceed correlation

    def get_cached_result(self, cause: str, effect: str) -> Optional[CausalBoundaryResult]:
        """Get cached result for a cause-effect pair."""
        return self._results_cache.get(f"{cause}|{effect}")

    def get_all_results(self) -> List[CausalBoundaryResult]:
        """Get all cached results."""
        return list(self._results_cache.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "correlations_processed": self._correlations_processed,
            "causal_validated": self._causal_validated,
            "submitted_for_verification": self._submitted_for_verification,
            "cached_results": len(self._results_cache),
        }

    def export_for_token_priors(self) -> str:
        """Export results as natural language for token priors.

        Option B Compliance: Knowledge as text, not weights.
        """
        lines = [
            "## Causal Boundary Analysis Results",
            "",
            "The following relationships have been analyzed for causality.",
            "Correlation confidence is LLM-derived; causal confidence is algorithm-validated.",
            "",
        ]

        for key, result in self._results_cache.items():
            lines.append(result.to_natural_language())
            lines.append("")

        return "\n".join(lines)

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()
        logger.info("CausalBoundary cache cleared")


def create_causal_boundary(
    llm_client: Optional[Any] = None,
    pkg_queries: Optional[Any] = None,
) -> CausalBoundary:
    """Create CausalBoundary with default components.

    Factory function for easy initialization.

    Args:
        llm_client: Optional LLM client for hypothesis generation
        pkg_queries: Optional PKG queries for validation

    Returns:
        Configured CausalBoundary instance
    """
    hypothesis_generator = None
    bradford_hill_validator = None
    icda_agent = None

    try:
        from futurnal.insights.hypothesis_generation import HypothesisGenerator
        hypothesis_generator = HypothesisGenerator(llm_client=llm_client)
    except ImportError:
        logger.warning("HypothesisGenerator not available")

    try:
        from futurnal.search.causal.bradford_hill import BradfordHillValidator
        bradford_hill_validator = BradfordHillValidator(llm_client=llm_client)
    except ImportError:
        logger.warning("BradfordHillValidator not available")

    try:
        from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent
        icda_agent = InteractiveCausalDiscoveryAgent()
    except ImportError:
        logger.warning("InteractiveCausalDiscoveryAgent not available")

    return CausalBoundary(
        hypothesis_generator=hypothesis_generator,
        bradford_hill_validator=bradford_hill_validator,
        icda_agent=icda_agent,
    )

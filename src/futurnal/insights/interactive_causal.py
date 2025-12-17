"""Interactive Causal Discovery Agent (ICDA).

AGI Phase 7: Enables user verification of low-confidence causal hypotheses
through interactive dialogue.

Research Foundation:
- ICDA (2024): Interactive Causal Discovery Agent
- ACCESS (2025): Causal validation metrics
- Training-Free GRPO (2510.08191v1): Learning from human feedback

Key Innovation:
Unlike fully automated causal inference (which often fails on observational
data), ICDA leverages human domain knowledge to:
1. Validate ambiguous causal hypotheses
2. Provide counterexamples and confounders
3. Refine confidence estimates based on feedback
4. Build personalized causal understanding over time

Example Question:
> "I noticed you slept poorly after late coding sessions.
>  Do you think the late coding caused the poor sleep?"

User Response:
> "Yes, the bright screen affects my sleep quality"
> OR
> "No, I usually code late when I'm stressed, and stress causes poor sleep"

The second response reveals a confounder (stress), improving causal model.

Option B Compliance:
- No model parameter updates
- User feedback stored as natural language priors
- Ghost model FROZEN throughout
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
    from futurnal.learning.token_priors import TokenPriorStore

logger = logging.getLogger(__name__)


class CausalResponse(str, Enum):
    """User response types to causal questions."""

    YES_CAUSAL = "yes_causal"  # User confirms causal relationship
    NO_CORRELATION_ONLY = "no_correlation"  # Correlation but not causal
    NO_REVERSE_CAUSATION = "reverse_causation"  # B causes A, not A causes B
    NO_CONFOUNDER = "confounder"  # Third factor causes both
    UNCERTAIN = "uncertain"  # User is not sure
    SKIP = "skip"  # User doesn't want to answer


class VerificationStatus(str, Enum):
    """Status of causal verification."""

    PENDING = "pending"  # Waiting for user response
    VERIFIED_CAUSAL = "verified_causal"  # User confirmed causal
    VERIFIED_NON_CAUSAL = "verified_non_causal"  # User rejected causal
    UNCERTAIN = "uncertain"  # User uncertain
    SKIPPED = "skipped"  # User skipped


@dataclass
class CausalCandidate:
    """A candidate causal relationship for verification.

    Represents a potential causal relationship detected from temporal
    correlations that needs user validation.
    """

    candidate_id: str = field(default_factory=lambda: str(uuid4()))

    # Relationship details
    cause_event: str = ""
    effect_event: str = ""
    avg_gap_days: float = 0.0
    co_occurrences: int = 0

    # Statistical confidence
    correlation_strength: float = 0.0
    p_value: Optional[float] = None
    initial_confidence: float = 0.0

    # Verification status
    status: VerificationStatus = VerificationStatus.PENDING
    user_response: Optional[CausalResponse] = None
    user_explanation: Optional[str] = None
    verified_at: Optional[datetime] = None

    # Updated confidence after verification
    final_confidence: float = 0.0
    confidence_delta: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    related_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "cause_event": self.cause_event,
            "effect_event": self.effect_event,
            "avg_gap_days": self.avg_gap_days,
            "co_occurrences": self.co_occurrences,
            "correlation_strength": self.correlation_strength,
            "p_value": self.p_value,
            "initial_confidence": self.initial_confidence,
            "status": self.status.value,
            "user_response": self.user_response.value if self.user_response else None,
            "user_explanation": self.user_explanation,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "final_confidence": self.final_confidence,
            "confidence_delta": self.confidence_delta,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CausalVerificationQuestion:
    """A question to present to the user for verification.

    Structures the verification dialogue in a user-friendly way.
    """

    question_id: str = field(default_factory=lambda: str(uuid4()))
    candidate: CausalCandidate = field(default_factory=CausalCandidate)

    # Question text
    main_question: str = ""
    context: str = ""
    evidence_summary: str = ""

    # Response options
    response_options: List[Tuple[CausalResponse, str]] = field(default_factory=list)

    # Follow-up prompts based on response
    followup_prompts: Dict[CausalResponse, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "question_id": self.question_id,
            "candidate_id": self.candidate.candidate_id,
            "main_question": self.main_question,
            "context": self.context,
            "evidence_summary": self.evidence_summary,
            "response_options": [
                {"value": opt[0].value, "label": opt[1]}
                for opt in self.response_options
            ],
        }


class InteractiveCausalDiscoveryAgent:
    """ICDA-style agent for interactive causal hypothesis verification.

    AGI Phase 7 core component that engages users in validating
    causal hypotheses through structured dialogue.

    Workflow:
    1. Identify low-confidence causal candidates from correlations
    2. Generate verification questions
    3. Present questions to user
    4. Process user responses
    5. Update confidence estimates and store as priors

    Example Usage:
        agent = InteractiveCausalDiscoveryAgent(token_store=store)

        # Add candidate from correlation
        candidate = agent.add_candidate_from_correlation(correlation)

        # Get pending verifications
        questions = agent.get_pending_verifications()

        # Process user response
        updated = agent.process_user_response(
            question.question_id,
            CausalResponse.YES_CAUSAL,
            "The late screen time definitely disrupts my sleep"
        )
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # No verification needed
    LOW_CONFIDENCE_THRESHOLD = 0.4  # Needs verification
    MIN_OCCURRENCES_FOR_QUESTION = 3  # Minimum data points

    # Confidence adjustments based on response
    CONFIDENCE_ADJUSTMENTS = {
        CausalResponse.YES_CAUSAL: 0.25,  # Increase confidence
        CausalResponse.NO_CORRELATION_ONLY: -0.4,  # Strong decrease
        CausalResponse.NO_REVERSE_CAUSATION: -0.5,  # Very strong decrease
        CausalResponse.NO_CONFOUNDER: -0.3,  # Moderate decrease
        CausalResponse.UNCERTAIN: 0.0,  # No change
        CausalResponse.SKIP: 0.0,  # No change
    }

    def __init__(
        self,
        token_prior_store: Optional["TokenPriorStore"] = None,
        max_pending: int = 10,
    ):
        """Initialize ICDA agent.

        Args:
            token_prior_store: For storing verified causal knowledge
            max_pending: Maximum pending verifications at once
        """
        self._token_prior_store = token_prior_store
        self._max_pending = max_pending

        # Candidate storage
        self._candidates: Dict[str, CausalCandidate] = {}
        self._questions: Dict[str, CausalVerificationQuestion] = {}

        # History
        self._verified_history: List[CausalCandidate] = []
        self._max_history = 100

        logger.info("InteractiveCausalDiscoveryAgent initialized")

    def add_candidate_from_correlation(
        self,
        correlation: "TemporalCorrelationResult",
    ) -> Optional[CausalCandidate]:
        """Create causal candidate from temporal correlation.

        Only creates candidates for correlations with:
        - Sufficient occurrences
        - Low-to-medium confidence (needs verification)
        - Causal candidate flag set

        Args:
            correlation: Temporal correlation result

        Returns:
            CausalCandidate if created, None if not suitable
        """
        # Check if correlation is suitable for verification
        if not self._is_suitable_for_verification(correlation):
            return None

        # Extract data
        cause = getattr(correlation, "event_type_a", "")
        effect = getattr(correlation, "event_type_b", "")
        gap_days = getattr(correlation, "avg_gap_days", 0) or 0
        co_occurrences = getattr(correlation, "co_occurrences", 0) or 0
        strength = getattr(correlation, "correlation_strength", 0.5) or 0.5
        p_value = getattr(correlation, "p_value", None)

        # Calculate initial confidence
        initial_confidence = self._calculate_initial_confidence(correlation)

        candidate = CausalCandidate(
            cause_event=cause,
            effect_event=effect,
            avg_gap_days=gap_days,
            co_occurrences=co_occurrences,
            correlation_strength=strength,
            p_value=p_value,
            initial_confidence=initial_confidence,
            final_confidence=initial_confidence,
        )

        # Store candidate
        self._candidates[candidate.candidate_id] = candidate

        # Generate verification question
        question = self.generate_verification_question(candidate)
        self._questions[question.question_id] = question

        logger.info(
            f"Added causal candidate: {cause} -> {effect} "
            f"(initial confidence: {initial_confidence:.2f})"
        )

        return candidate

    def _is_suitable_for_verification(
        self,
        correlation: "TemporalCorrelationResult",
    ) -> bool:
        """Check if correlation is suitable for user verification."""
        # Must have causal candidate flag
        is_causal_candidate = getattr(correlation, "is_causal_candidate", False)
        if not is_causal_candidate:
            return False

        # Must have minimum occurrences
        co_occurrences = getattr(correlation, "co_occurrences", 0) or 0
        if co_occurrences < self.MIN_OCCURRENCES_FOR_QUESTION:
            return False

        # Must not already be high confidence
        strength = getattr(correlation, "correlation_strength", 0.5) or 0.5
        if strength >= self.HIGH_CONFIDENCE_THRESHOLD:
            return False

        # Check if we have too many pending
        pending_count = sum(
            1 for c in self._candidates.values()
            if c.status == VerificationStatus.PENDING
        )
        if pending_count >= self._max_pending:
            return False

        return True

    def _calculate_initial_confidence(
        self,
        correlation: "TemporalCorrelationResult",
    ) -> float:
        """Calculate initial causal confidence from correlation."""
        confidence = 0.3  # Base

        # Boost from correlation strength
        strength = getattr(correlation, "correlation_strength", 0.5) or 0.5
        confidence += strength * 0.3

        # Boost from statistical significance
        p_value = getattr(correlation, "p_value", None)
        if p_value is not None and p_value < 0.05:
            confidence += 0.2
        elif p_value is not None and p_value < 0.1:
            confidence += 0.1

        # Boost from co-occurrences
        co_occurrences = getattr(correlation, "co_occurrences", 0) or 0
        if co_occurrences >= 10:
            confidence += 0.1
        elif co_occurrences >= 5:
            confidence += 0.05

        return min(0.9, confidence)

    def generate_verification_question(
        self,
        candidate: CausalCandidate,
    ) -> CausalVerificationQuestion:
        """Generate user-friendly verification question.

        Creates a structured question that helps users understand
        and validate the causal hypothesis.

        Args:
            candidate: The causal candidate to verify

        Returns:
            Formatted verification question
        """
        cause = candidate.cause_event
        effect = candidate.effect_event
        gap_days = candidate.avg_gap_days
        count = candidate.co_occurrences

        # Generate main question
        if gap_days < 1:
            time_phrase = "on the same day"
        elif gap_days < 2:
            time_phrase = f"about {gap_days:.0f} day later"
        else:
            time_phrase = f"about {gap_days:.0f} days later"

        main_question = (
            f"I've noticed a pattern: when '{cause}' happens, "
            f"'{effect}' tends to follow {time_phrase}. "
            f"Do you think {cause.lower()} actually causes {effect.lower()}?"
        )

        # Generate context
        context = (
            f"This pattern was observed {count} times in your data. "
            f"The correlation strength is {candidate.correlation_strength:.0%}."
        )

        # Evidence summary
        evidence_lines = [
            f"Pattern: {cause} → {effect}",
            f"Average gap: {gap_days:.1f} days",
            f"Occurrences: {count}",
            f"Correlation: {candidate.correlation_strength:.0%}",
        ]
        if candidate.p_value:
            evidence_lines.append(f"Statistical significance: p={candidate.p_value:.3f}")

        evidence_summary = "\n".join(evidence_lines)

        # Response options with descriptions
        response_options = [
            (CausalResponse.YES_CAUSAL, f"Yes, {cause} causes {effect}"),
            (CausalResponse.NO_CORRELATION_ONLY, "No, they just happen together by coincidence"),
            (CausalResponse.NO_REVERSE_CAUSATION, f"No, actually {effect} causes {cause}"),
            (CausalResponse.NO_CONFOUNDER, "No, something else causes both"),
            (CausalResponse.UNCERTAIN, "I'm not sure"),
            (CausalResponse.SKIP, "Skip this question"),
        ]

        # Follow-up prompts
        followup_prompts = {
            CausalResponse.YES_CAUSAL: "Can you explain how this works?",
            CausalResponse.NO_CORRELATION_ONLY: "What makes you think they're unrelated?",
            CausalResponse.NO_REVERSE_CAUSATION: "How does the reverse causation work?",
            CausalResponse.NO_CONFOUNDER: "What factor might be causing both?",
            CausalResponse.UNCERTAIN: "What would help you decide?",
        }

        return CausalVerificationQuestion(
            candidate=candidate,
            main_question=main_question,
            context=context,
            evidence_summary=evidence_summary,
            response_options=response_options,
            followup_prompts=followup_prompts,
        )

    def process_user_response(
        self,
        question_id: str,
        response: CausalResponse,
        explanation: Optional[str] = None,
    ) -> CausalCandidate:
        """Process user's response to verification question.

        Updates candidate confidence and stores feedback as priors.

        Args:
            question_id: The question being answered
            response: User's response type
            explanation: Optional user explanation

        Returns:
            Updated CausalCandidate
        """
        if question_id not in self._questions:
            raise ValueError(f"Unknown question: {question_id}")

        question = self._questions[question_id]
        candidate = question.candidate

        # Update candidate
        candidate.user_response = response
        candidate.user_explanation = explanation
        candidate.verified_at = datetime.utcnow()

        # Update status based on response
        if response == CausalResponse.YES_CAUSAL:
            candidate.status = VerificationStatus.VERIFIED_CAUSAL
        elif response in [
            CausalResponse.NO_CORRELATION_ONLY,
            CausalResponse.NO_REVERSE_CAUSATION,
            CausalResponse.NO_CONFOUNDER,
        ]:
            candidate.status = VerificationStatus.VERIFIED_NON_CAUSAL
        elif response == CausalResponse.UNCERTAIN:
            candidate.status = VerificationStatus.UNCERTAIN
        else:
            candidate.status = VerificationStatus.SKIPPED

        # Calculate confidence adjustment
        adjustment = self.CONFIDENCE_ADJUSTMENTS.get(response, 0.0)
        candidate.confidence_delta = adjustment
        candidate.final_confidence = max(0.0, min(1.0,
            candidate.initial_confidence + adjustment
        ))

        # Store as token prior if verified
        if candidate.status in [
            VerificationStatus.VERIFIED_CAUSAL,
            VerificationStatus.VERIFIED_NON_CAUSAL,
        ]:
            self._store_verified_knowledge(candidate)

        # Move to history
        self._verified_history.append(candidate)
        if len(self._verified_history) > self._max_history:
            self._verified_history = self._verified_history[-self._max_history:]

        # Clean up
        del self._questions[question_id]

        logger.info(
            f"Processed response for {candidate.cause_event} -> {candidate.effect_event}: "
            f"{response.value} (confidence: {candidate.initial_confidence:.2f} -> {candidate.final_confidence:.2f})"
        )

        return candidate

    def _store_verified_knowledge(self, candidate: CausalCandidate):
        """Store verified causal knowledge as token priors.

        Args:
            candidate: The verified candidate
        """
        if not self._token_prior_store:
            return

        try:
            from futurnal.learning.token_priors import TemporalPatternPrior

            # Generate description based on verification
            if candidate.status == VerificationStatus.VERIFIED_CAUSAL:
                description = (
                    f"Verified causal relationship: {candidate.cause_event} "
                    f"causes {candidate.effect_event} (confirmed by user)"
                )
            else:
                description = (
                    f"Non-causal correlation: {candidate.cause_event} and "
                    f"{candidate.effect_event} correlate but don't have causal relationship"
                )

            if candidate.user_explanation:
                description += f". User explanation: {candidate.user_explanation}"

            prior = TemporalPatternPrior(
                pattern_type="verified_causal",
                description=description,
                learned_weights=str({
                    "cause": candidate.cause_event,
                    "effect": candidate.effect_event,
                    "is_causal": candidate.status == VerificationStatus.VERIFIED_CAUSAL,
                    "confidence": candidate.final_confidence,
                }),
                confidence=candidate.final_confidence,
                observation_count=candidate.co_occurrences,
            )

            self._token_prior_store.add_temporal_pattern(prior)

            logger.info("Stored verified causal knowledge as token prior")

        except Exception as e:
            logger.warning(f"Failed to store causal knowledge: {e}")

    def get_pending_verifications(
        self,
        max_items: int = 5,
    ) -> List[CausalVerificationQuestion]:
        """Get pending verification questions for user.

        Args:
            max_items: Maximum questions to return

        Returns:
            List of pending verification questions
        """
        # Filter to pending questions only
        pending = [
            q for q in self._questions.values()
            if q.candidate.status == VerificationStatus.PENDING
        ]

        # Sort by initial confidence (lower first - needs more verification)
        pending.sort(key=lambda q: q.candidate.initial_confidence)

        return pending[:max_items]

    def get_verification_history(
        self,
        status: Optional[VerificationStatus] = None,
        limit: int = 20,
    ) -> List[CausalCandidate]:
        """Get verification history.

        Args:
            status: Filter by status (optional)
            limit: Maximum results

        Returns:
            List of verified candidates
        """
        results = self._verified_history

        if status:
            results = [c for c in results if c.status == status]

        return results[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics.

        Returns:
            Dictionary with verification statistics
        """
        total = len(self._verified_history)
        verified_causal = sum(
            1 for c in self._verified_history
            if c.status == VerificationStatus.VERIFIED_CAUSAL
        )
        verified_non_causal = sum(
            1 for c in self._verified_history
            if c.status == VerificationStatus.VERIFIED_NON_CAUSAL
        )
        uncertain = sum(
            1 for c in self._verified_history
            if c.status == VerificationStatus.UNCERTAIN
        )
        skipped = sum(
            1 for c in self._verified_history
            if c.status == VerificationStatus.SKIPPED
        )

        pending = sum(
            1 for c in self._candidates.values()
            if c.status == VerificationStatus.PENDING
        )

        # Calculate average confidence improvement
        improvements = [
            c.confidence_delta for c in self._verified_history
            if c.confidence_delta != 0
        ]
        avg_improvement = (
            sum(improvements) / len(improvements)
            if improvements else 0
        )

        return {
            "total_verifications": total,
            "verified_causal": verified_causal,
            "verified_non_causal": verified_non_causal,
            "uncertain": uncertain,
            "skipped": skipped,
            "pending": pending,
            "avg_confidence_improvement": avg_improvement,
            "causal_rate": verified_causal / total if total > 0 else 0,
        }

    def export_for_token_priors(self) -> str:
        """Export all verified knowledge as natural language.

        Returns:
            Natural language summary for token priors
        """
        lines = ["Verified causal relationships:"]

        for candidate in self._verified_history:
            if candidate.status == VerificationStatus.VERIFIED_CAUSAL:
                lines.append(
                    f"- {candidate.cause_event} causes {candidate.effect_event} "
                    f"(confidence: {candidate.final_confidence:.0%})"
                )
            elif candidate.status == VerificationStatus.VERIFIED_NON_CAUSAL:
                lines.append(
                    f"- {candidate.cause_event} does NOT cause {candidate.effect_event} "
                    f"(correlation only)"
                )

        return "\n".join(lines)

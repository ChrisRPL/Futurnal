"""Correlation Planner for AgentFlow Analysis.

Phase 2E: AgentFlow Architecture - Step 13

Generates and prioritizes correlation hypotheses for investigation.
Uses PKG structure and event patterns to identify potential causal
relationships worth exploring.

Research Foundation:
- Event-CausNet (2025): Causal feature extraction from events
- ICDA (2024): Interactive Causal Discovery
- Bradford Hill criteria for causal inference

Option B Compliance:
- No model parameter updates
- Hypotheses expressed as natural language
- Ghost model FROZEN
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from futurnal.agents.memory_buffer import (
    EvolvingMemoryBuffer,
    MemoryEntry,
    MemoryEntryType,
    MemoryPriority,
    get_memory_buffer,
)

logger = logging.getLogger(__name__)


class HypothesisType(str, Enum):
    """Types of correlation hypotheses."""

    TEMPORAL_SEQUENCE = "temporal_sequence"  # A precedes B
    CO_OCCURRENCE = "co_occurrence"  # A and B appear together
    PERIODIC_PATTERN = "periodic_pattern"  # Recurring rhythm
    CAUSAL_CHAIN = "causal_chain"  # A -> B -> C
    CONFOUNDED = "confounded"  # A and B share cause C


class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""

    PROPOSED = "proposed"  # Newly generated
    INVESTIGATING = "investigating"  # Currently being tested
    NEEDS_EVIDENCE = "needs_evidence"  # Requires more data
    CONFIRMED = "confirmed"  # Verified as likely causal
    REFUTED = "refuted"  # Evidence against causality
    INCONCLUSIVE = "inconclusive"  # Cannot determine


@dataclass
class CorrelationHypothesis:
    """A hypothesis about potential correlation or causation.

    Attributes:
        hypothesis_id: Unique identifier
        hypothesis_type: Type of correlation
        description: Natural language description
        event_type_a: First event type
        event_type_b: Second event type
        confidence: Initial confidence (0-1)
        evidence_for: Supporting evidence points
        evidence_against: Contradicting evidence points
        status: Current status
        created_at: When hypothesis was generated
        last_updated: When hypothesis was last modified
        metadata: Additional structured data
    """

    hypothesis_id: str = field(default_factory=lambda: str(uuid4()))
    hypothesis_type: HypothesisType = HypothesisType.TEMPORAL_SEQUENCE
    description: str = ""
    event_type_a: str = ""
    event_type_b: str = ""
    confidence: float = 0.5
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_type": self.hypothesis_type.value,
            "description": self.description,
            "event_type_a": self.event_type_a,
            "event_type_b": self.event_type_b,
            "confidence": self.confidence,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrelationHypothesis":
        """Create from dictionary."""
        return cls(
            hypothesis_id=data.get("hypothesis_id", str(uuid4())),
            hypothesis_type=HypothesisType(data.get("hypothesis_type", "temporal_sequence")),
            description=data.get("description", ""),
            event_type_a=data.get("event_type_a", ""),
            event_type_b=data.get("event_type_b", ""),
            confidence=data.get("confidence", 0.5),
            evidence_for=data.get("evidence_for", []),
            evidence_against=data.get("evidence_against", []),
            status=HypothesisStatus(data.get("status", "proposed")),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            last_updated=datetime.fromisoformat(
                data.get("last_updated", datetime.utcnow().isoformat())
            ),
            metadata=data.get("metadata", {}),
        )

    def to_natural_language(self) -> str:
        """Convert to natural language for token priors."""
        lines = [
            f"Hypothesis: {self.description}",
            f"Type: {self.hypothesis_type.value}",
            f"Events: {self.event_type_a} -> {self.event_type_b}",
            f"Confidence: {self.confidence:.0%}",
            f"Status: {self.status.value}",
        ]

        if self.evidence_for:
            lines.append(f"Supporting evidence: {len(self.evidence_for)} points")
        if self.evidence_against:
            lines.append(f"Contradicting evidence: {len(self.evidence_against)} points")

        return "\n".join(lines)


@dataclass
class QueryPlan:
    """Plan for investigating a hypothesis.

    Attributes:
        hypothesis_id: Hypothesis being investigated
        queries: List of PKG queries to execute
        expected_results: What we expect to find if hypothesis is true
        completion_criteria: When to stop investigating
    """

    hypothesis_id: str
    queries: List[Dict[str, Any]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    completion_criteria: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "queries": self.queries,
            "expected_results": self.expected_results,
            "completion_criteria": self.completion_criteria,
        }


class CorrelationPlanner:
    """Plans and prioritizes correlation investigations.

    AgentFlow Module 1: Generates hypotheses about potential correlations
    in the user's data and prioritizes which to investigate first.

    Key Capabilities:
    1. Generate hypotheses from PKG event patterns
    2. Prioritize hypotheses by potential value
    3. Design query strategies for investigation
    4. Track investigation progress in memory buffer

    Option B Compliance:
    - No model updates
    - Hypotheses as natural language
    - Ghost model FROZEN

    Usage:
        planner = CorrelationPlanner()
        hypotheses = planner.generate_hypotheses(pkg_summary)
        top_hypothesis = planner.prioritize_investigation(hypotheses)
        query_plan = planner.design_query_strategy(top_hypothesis)
    """

    # Hypothesis generation templates
    TEMPORAL_TEMPLATES = [
        "Events of type '{event_a}' tend to precede '{event_b}' events",
        "'{event_a}' and '{event_b}' show temporal correlation",
        "'{event_a}' activities may influence later '{event_b}' outcomes",
    ]

    PERIODIC_TEMPLATES = [
        "'{event_a}' shows a weekly rhythm pattern",
        "'{event_a}' tends to cluster at specific times",
        "'{event_a}' follows a recurring cycle",
    ]

    def __init__(
        self,
        memory_buffer: Optional[EvolvingMemoryBuffer] = None,
        pkg_queries: Optional[Any] = None,
    ):
        """Initialize correlation planner.

        Args:
            memory_buffer: Memory buffer for tracking state
            pkg_queries: PKG query interface
        """
        self.memory = memory_buffer or get_memory_buffer()
        self.pkg_queries = pkg_queries

        # Active hypotheses
        self._hypotheses: Dict[str, CorrelationHypothesis] = {}

        logger.info("CorrelationPlanner initialized")

    def generate_hypotheses(
        self,
        event_types: List[str],
        event_counts: Optional[Dict[str, int]] = None,
        co_occurrences: Optional[List[Tuple[str, str, int]]] = None,
    ) -> List[CorrelationHypothesis]:
        """Generate correlation hypotheses from event data.

        Args:
            event_types: List of distinct event types in PKG
            event_counts: Optional counts per event type
            co_occurrences: Optional list of (type_a, type_b, count) tuples

        Returns:
            List of generated hypotheses
        """
        hypotheses: List[CorrelationHypothesis] = []

        # Generate temporal sequence hypotheses for event pairs
        for i, event_a in enumerate(event_types):
            for event_b in event_types[i + 1 :]:
                # Skip if same type
                if event_a == event_b:
                    continue

                # Check if we already have a hypothesis for this pair
                existing = self._find_existing_hypothesis(event_a, event_b)
                if existing:
                    continue

                hypothesis = CorrelationHypothesis(
                    hypothesis_type=HypothesisType.TEMPORAL_SEQUENCE,
                    description=self.TEMPORAL_TEMPLATES[0].format(
                        event_a=event_a, event_b=event_b
                    ),
                    event_type_a=event_a,
                    event_type_b=event_b,
                    confidence=0.3,  # Low initial confidence
                    metadata={
                        "source": "event_pair_analysis",
                        "event_count_a": event_counts.get(event_a, 0) if event_counts else 0,
                        "event_count_b": event_counts.get(event_b, 0) if event_counts else 0,
                    },
                )

                hypotheses.append(hypothesis)
                self._hypotheses[hypothesis.hypothesis_id] = hypothesis

        # Boost confidence for pairs with co-occurrences
        if co_occurrences:
            for event_a, event_b, count in co_occurrences:
                for h in hypotheses:
                    if h.event_type_a == event_a and h.event_type_b == event_b:
                        # Boost confidence based on co-occurrence count
                        boost = min(0.3, count * 0.02)
                        h.confidence = min(0.9, h.confidence + boost)
                        h.metadata["co_occurrence_count"] = count

        # Generate periodic pattern hypotheses
        if event_counts:
            for event_type, count in event_counts.items():
                if count >= 10:  # Need enough events for pattern
                    hypothesis = CorrelationHypothesis(
                        hypothesis_type=HypothesisType.PERIODIC_PATTERN,
                        description=self.PERIODIC_TEMPLATES[0].format(event_a=event_type),
                        event_type_a=event_type,
                        event_type_b="",
                        confidence=0.4,
                        metadata={
                            "source": "periodic_analysis",
                            "event_count": count,
                        },
                    )
                    hypotheses.append(hypothesis)
                    self._hypotheses[hypothesis.hypothesis_id] = hypothesis

        # Store in memory buffer
        for h in hypotheses:
            self.memory.add_entry(
                MemoryEntry(
                    entry_type=MemoryEntryType.HYPOTHESIS,
                    content=h.to_natural_language(),
                    priority=MemoryPriority.NORMAL,
                    metadata={"hypothesis_id": h.hypothesis_id},
                )
            )

        logger.info(f"Generated {len(hypotheses)} correlation hypotheses")
        return hypotheses

    def _find_existing_hypothesis(
        self, event_a: str, event_b: str
    ) -> Optional[CorrelationHypothesis]:
        """Find existing hypothesis for event pair."""
        for h in self._hypotheses.values():
            if h.event_type_a == event_a and h.event_type_b == event_b:
                return h
            if h.event_type_a == event_b and h.event_type_b == event_a:
                return h
        return None

    def prioritize_investigation(
        self,
        hypotheses: List[CorrelationHypothesis],
        user_interests: Optional[List[str]] = None,
    ) -> Optional[CorrelationHypothesis]:
        """Prioritize which hypothesis to investigate next.

        Scoring factors:
        - Confidence (medium is best - not too certain, not too unlikely)
        - Event frequency (more data = better investigation)
        - User interest alignment
        - Novelty (haven't investigated similar before)

        Args:
            hypotheses: List of hypotheses to prioritize
            user_interests: Optional list of user interest keywords

        Returns:
            Highest priority hypothesis, or None if empty
        """
        if not hypotheses:
            return None

        scored: List[Tuple[float, CorrelationHypothesis]] = []

        for h in hypotheses:
            # Skip already investigated
            if h.status in (HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED):
                continue

            score = 0.0

            # Medium confidence is most valuable (0.4-0.7 range)
            if 0.4 <= h.confidence <= 0.7:
                score += 0.3
            elif h.confidence < 0.3:
                score += 0.1
            else:
                score += 0.2

            # Event count bonus
            event_count = h.metadata.get("event_count_a", 0) + h.metadata.get(
                "event_count_b", 0
            )
            score += min(0.3, event_count / 100)

            # User interest alignment
            if user_interests:
                for interest in user_interests:
                    if interest.lower() in h.description.lower():
                        score += 0.2
                        break

            # Novelty bonus (not already in memory as investigated)
            related_memories = self.memory.get_relevant_context(
                h.description,
                max_entries=3,
                entry_types=[MemoryEntryType.INVESTIGATION],
            )
            if not related_memories:
                score += 0.2

            scored.append((score, h))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def design_query_strategy(
        self, hypothesis: CorrelationHypothesis
    ) -> QueryPlan:
        """Design a query strategy to investigate the hypothesis.

        Args:
            hypothesis: The hypothesis to investigate

        Returns:
            QueryPlan with PKG queries and expected results
        """
        queries = []
        expected_results = []

        if hypothesis.hypothesis_type == HypothesisType.TEMPORAL_SEQUENCE:
            # Query 1: Find co-occurrences within time windows
            queries.append({
                "type": "temporal_correlation",
                "event_a": hypothesis.event_type_a,
                "event_b": hypothesis.event_type_b,
                "max_lag_hours": 72,
                "description": f"Find {hypothesis.event_type_a} events followed by {hypothesis.event_type_b}",
            })

            expected_results.append(
                f"If causal: >60% of {hypothesis.event_type_a} events are followed by {hypothesis.event_type_b}"
            )

            # Query 2: Check reverse direction
            queries.append({
                "type": "temporal_correlation",
                "event_a": hypothesis.event_type_b,
                "event_b": hypothesis.event_type_a,
                "max_lag_hours": 72,
                "description": f"Check reverse: {hypothesis.event_type_b} followed by {hypothesis.event_type_a}",
            })

            expected_results.append(
                "If truly causal: reverse direction should have lower correlation"
            )

        elif hypothesis.hypothesis_type == HypothesisType.PERIODIC_PATTERN:
            # Query: Day-of-week distribution
            queries.append({
                "type": "day_of_week_distribution",
                "event_type": hypothesis.event_type_a,
                "description": f"Analyze day-of-week pattern for {hypothesis.event_type_a}",
            })

            expected_results.append(
                "If periodic: significant deviation from uniform distribution"
            )

        elif hypothesis.hypothesis_type == HypothesisType.CO_OCCURRENCE:
            # Query: Same-day co-occurrence
            queries.append({
                "type": "co_occurrence",
                "event_a": hypothesis.event_type_a,
                "event_b": hypothesis.event_type_b,
                "same_day": True,
                "description": f"Count same-day co-occurrences of {hypothesis.event_type_a} and {hypothesis.event_type_b}",
            })

            expected_results.append(
                "If correlated: co-occurrence rate significantly above random chance"
            )

        plan = QueryPlan(
            hypothesis_id=hypothesis.hypothesis_id,
            queries=queries,
            expected_results=expected_results,
            completion_criteria=(
                f"Investigation complete when: "
                f"1) All {len(queries)} queries executed, "
                f"2) Statistical significance determined, "
                f"3) Confidence updated above 0.7 (confirmed) or below 0.3 (refuted)"
            ),
        )

        # Store investigation start in memory
        hypothesis.status = HypothesisStatus.INVESTIGATING
        hypothesis.last_updated = datetime.utcnow()

        self.memory.add_entry(
            MemoryEntry(
                entry_type=MemoryEntryType.INVESTIGATION,
                content=f"Started investigating: {hypothesis.description}",
                priority=MemoryPriority.NORMAL,
                metadata={
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "query_count": len(queries),
                },
            )
        )

        logger.info(
            f"Designed query plan for hypothesis {hypothesis.hypothesis_id}: "
            f"{len(queries)} queries"
        )

        return plan

    def update_hypothesis(
        self,
        hypothesis_id: str,
        evidence: Optional[str] = None,
        is_supporting: bool = True,
        confidence_delta: float = 0.0,
        new_status: Optional[HypothesisStatus] = None,
    ) -> Optional[CorrelationHypothesis]:
        """Update a hypothesis with new evidence.

        Args:
            hypothesis_id: Hypothesis to update
            evidence: Evidence description
            is_supporting: True if evidence supports hypothesis
            confidence_delta: Amount to adjust confidence
            new_status: New status (if changing)

        Returns:
            Updated hypothesis, or None if not found
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            return None

        if evidence:
            if is_supporting:
                hypothesis.evidence_for.append(evidence)
            else:
                hypothesis.evidence_against.append(evidence)

        hypothesis.confidence = max(0.0, min(1.0, hypothesis.confidence + confidence_delta))

        if new_status:
            hypothesis.status = new_status

        hypothesis.last_updated = datetime.utcnow()

        # Update memory
        self.memory.add_entry(
            MemoryEntry(
                entry_type=MemoryEntryType.EVIDENCE,
                content=f"Evidence for hypothesis '{hypothesis.description[:50]}...': {evidence}",
                priority=MemoryPriority.NORMAL if is_supporting else MemoryPriority.LOW,
                metadata={
                    "hypothesis_id": hypothesis_id,
                    "is_supporting": is_supporting,
                },
            )
        )

        return hypothesis

    def get_active_hypotheses(self) -> List[CorrelationHypothesis]:
        """Get all active (non-resolved) hypotheses."""
        return [
            h
            for h in self._hypotheses.values()
            if h.status not in (HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED)
        ]

    def get_hypothesis(self, hypothesis_id: str) -> Optional[CorrelationHypothesis]:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def export_for_token_priors(self) -> str:
        """Export planner state as natural language for token priors."""
        lines = ["Correlation Investigation State:"]

        active = self.get_active_hypotheses()
        lines.append(f"\nActive Hypotheses: {len(active)}")

        for h in active[:5]:
            lines.append(f"\n{h.to_natural_language()}")

        confirmed = [
            h for h in self._hypotheses.values() if h.status == HypothesisStatus.CONFIRMED
        ]
        if confirmed:
            lines.append(f"\nConfirmed Correlations: {len(confirmed)}")
            for h in confirmed[:3]:
                lines.append(f"- {h.description}")

        return "\n".join(lines)


# Global instance
_correlation_planner: Optional[CorrelationPlanner] = None


def get_correlation_planner() -> CorrelationPlanner:
    """Get the default correlation planner singleton."""
    global _correlation_planner
    if _correlation_planner is None:
        _correlation_planner = CorrelationPlanner()
    return _correlation_planner

"""Emergent Insights Generation for AGI-Level Intelligence.

AGI Phase 5: Transforms detected patterns and correlations into
human-readable insights that enable proactive self-discovery.

Research Foundation:
- Event-CausNet (2025): Causal feature extraction from events
- ACCESS (2025): Causal validation metrics
- Training-Free GRPO (2510.08191v1): Natural language learning

Key Innovation:
Unlike standard reporting systems, the InsightGenerator:
1. Transforms statistical patterns into natural language narratives
2. Ranks insights by relevance to user's aspirations
3. Generates actionable suggestions
4. Maintains insight quality through confidence thresholds

Example Emergent Insights:
- "A pattern has been detected: 75% of your project proposals written on a
  Monday are accepted, compared to 30% for those written on a Friday. This
  pattern has held for the past 18 months."
- "Your journal entries show a recurring theme: 'fatigued' appears 4x more
  frequently on Fridays and precedes low productivity days by 24-48 hours."
- "Knowledge pattern: You have 15 notes referencing 'Project Titan' but none
  are linked to your stated aspiration of 'Lead high-impact projects'—this
  gap might indicate a documentation blind spot or potential misalignment."

Option B Compliance:
- No model parameter updates
- Insights expressed as natural language for token priors
- Ghost model FROZEN throughout
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid4

if TYPE_CHECKING:
    from futurnal.search.temporal.results import TemporalCorrelationResult
    from futurnal.insights.curiosity_engine import KnowledgeGap

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """Categories of emergent insights."""

    TEMPORAL_CORRELATION = "temporal_correlation"  # A precedes B pattern
    BEHAVIORAL_PATTERN = "behavioral_pattern"  # Recurring behavior
    KNOWLEDGE_GAP = "knowledge_gap"  # Missing synthesis/connections
    ASPIRATION_MISALIGNMENT = "aspiration_misalignment"  # Goal-knowledge disconnect
    PRODUCTIVITY_PATTERN = "productivity_pattern"  # Work effectiveness
    CAUSAL_HYPOTHESIS = "causal_hypothesis"  # Potential causal relationship
    WEEKLY_RHYTHM = "weekly_rhythm"  # Day-of-week patterns
    SEQUENCE_PATTERN = "sequence_pattern"  # Multi-step event sequences


class InsightConfidence(str, Enum):
    """Confidence levels for insights."""

    HIGH = "high"  # >0.8 confidence, statistically significant
    MEDIUM = "medium"  # 0.5-0.8 confidence, likely valid
    LOW = "low"  # <0.5 confidence, needs validation


@dataclass
class EmergentInsight:
    """Represents a system-generated insight from experiential data analysis.

    Insights are autonomous discoveries the Ghost makes by analyzing the PKG.
    They are presented to users with supporting evidence and actionable context.

    Attributes:
        insight_id: Unique identifier
        insight_type: Category of insight
        title: Short, attention-grabbing summary
        description: Detailed explanation of the pattern or discovery
        confidence: Confidence in the insight (0.0 to 1.0)
        confidence_level: Categorical confidence (HIGH/MEDIUM/LOW)
        relevance_score: Estimated relevance to user (0.0 to 1.0)
        supporting_evidence: List of event IDs or data points supporting this insight
        suggested_actions: Optional recommendations for user
        related_aspirations: Links to user's Aspirational Self if applicable
        statistical_summary: Summary of statistical support
        created_at: When insight was generated
        dismissed: Whether user has dismissed this insight
        validated: Whether user has validated this insight
    """

    insight_id: str = field(default_factory=lambda: str(uuid4()))
    insight_type: InsightType = InsightType.TEMPORAL_CORRELATION
    title: str = ""
    description: str = ""
    confidence: float = 0.0
    confidence_level: InsightConfidence = InsightConfidence.LOW
    relevance_score: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    related_aspirations: List[str] = field(default_factory=list)
    statistical_summary: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    dismissed: bool = False
    validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "relevance_score": self.relevance_score,
            "supporting_evidence": self.supporting_evidence,
            "suggested_actions": self.suggested_actions,
            "related_aspirations": self.related_aspirations,
            "statistical_summary": self.statistical_summary,
            "created_at": self.created_at.isoformat(),
            "dismissed": self.dismissed,
            "validated": self.validated,
        }

    @property
    def priority_score(self) -> float:
        """Combined priority score for ranking insights."""
        validation_bonus = 0.1 if self.validated else 0
        dismissal_penalty = -0.5 if self.dismissed else 0
        return (
            0.35 * self.confidence +
            0.35 * self.relevance_score +
            0.20 * (1.0 if self.confidence_level == InsightConfidence.HIGH else 0.5) +
            validation_bonus +
            dismissal_penalty
        )

    def to_natural_language(self) -> str:
        """Convert insight to natural language for token priors."""
        lines = [
            f"Insight: {self.title}",
            f"Type: {self.insight_type.value}",
            f"Description: {self.description}",
            f"Confidence: {self.confidence:.0%} ({self.confidence_level.value})",
        ]

        if self.statistical_summary:
            lines.append(f"Statistics: {self.statistical_summary}")

        if self.suggested_actions:
            lines.append("Suggested Actions:")
            for action in self.suggested_actions[:3]:
                lines.append(f"  - {action}")

        return "\n".join(lines)


class InsightGenerator:
    """Generates emergent insights from detected patterns and correlations.

    AGI Phase 5 implementation that transforms statistical patterns into
    actionable, human-readable insights.

    Key Capabilities:
    1. Convert temporal correlations into natural language insights
    2. Generate contextual explanations with supporting evidence
    3. Rank insights by relevance and novelty
    4. Link insights to user's Aspirational Self where applicable
    5. Suggest actionable next steps based on detected patterns
    6. Filter low-confidence insights using statistical thresholds

    Example Usage:
        generator = InsightGenerator()
        correlations = detector.detect_correlations(stream)
        patterns = recognizer.recognize_patterns(stream)
        insights = generator.generate_insights(correlations, patterns)
        for insight in insights:
            if insight.priority_score > 0.7:
                display_to_user(insight)

    Option B Compliance:
    - No model updates, outputs natural language for token priors
    - Ghost model FROZEN
    """

    # Configuration
    MIN_CONFIDENCE = 0.5  # Minimum confidence to generate insight
    MIN_STATISTICAL_SIGNIFICANCE = 0.05  # p-value threshold
    MIN_OCCURRENCES = 3  # Minimum pattern occurrences
    MAX_INSIGHTS_PER_TYPE = 5  # Limit per category

    # Natural language templates
    CORRELATION_TEMPLATES = [
        "{event_a} typically precedes {event_b} by {gap_days:.1f} days (observed {count} times, p={p_value:.3f})",
        "A pattern has been detected: {event_a} is followed by {event_b} within {gap_days:.1f} days in {percentage:.0%} of cases",
        "Statistical correlation found: {event_a} → {event_b} (avg gap: {gap_days:.1f} days, strength: {strength:.0%})",
    ]

    WEEKLY_TEMPLATES = [
        "{event_type} occurs {multiplier:.1f}x more frequently on {day_name}s",
        "Your {event_type} shows a weekly rhythm: most active on {day_name}s ({percentage:.0%} of occurrences)",
        "Weekly pattern detected: {event_type} peaks on {day_name} and dips on {low_day}",
    ]

    BEHAVIORAL_TEMPLATES = [
        "Recurring pattern: {pattern_description} (observed {count} times over {span_days} days)",
        "You tend to {behavior} approximately every {interval_days:.0f} days",
        "A behavioral rhythm detected: {pattern_description}",
    ]

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        min_significance: float = MIN_STATISTICAL_SIGNIFICANCE,
        storage_path: Optional[str] = None,
        ranking_model: Optional[Any] = None,
    ):
        """Initialize insight generator.

        Args:
            min_confidence: Minimum confidence threshold for generating insights
            min_significance: Maximum p-value for statistical significance
            storage_path: Path to persist insights (default: ~/.futurnal/insights/emergent.json)
            ranking_model: Optional RankingModel for personalized scoring
        """
        import os
        import json
        from pathlib import Path

        self.min_confidence = min_confidence
        self.min_significance = min_significance

        # Phase 2C: Personalized ranking model
        self._ranking_model = ranking_model
        if self._ranking_model is None:
            try:
                from futurnal.insights.feedback import get_ranking_model
                self._ranking_model = get_ranking_model()
            except ImportError:
                logger.debug("Feedback module not available, using default ranking")

        # Persistent storage for emergent insights
        self._storage_path = Path(
            storage_path or os.path.expanduser("~/.futurnal/insights/emergent.json")
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load cached insights from storage
        self._cached_insights: List[Dict[str, Any]] = []
        if self._storage_path.exists():
            try:
                self._cached_insights = json.loads(self._storage_path.read_text())
                logger.info(f"Loaded {len(self._cached_insights)} cached insights")
            except Exception as e:
                logger.warning(f"Could not load cached insights: {e}")

        logger.info(
            f"InsightGenerator initialized "
            f"(min_confidence={min_confidence}, min_significance={min_significance}, "
            f"personalized_ranking={'enabled' if self._ranking_model else 'disabled'})"
        )

    def generate_insights(
        self,
        correlations: List[Any],
        patterns: Optional[List[Any]] = None,
        knowledge_gaps: Optional[List[Any]] = None,
        aspirations: Optional[List[Any]] = None,
    ) -> List[EmergentInsight]:
        """Generate emergent insights from detected patterns.

        AGI Phase 5 main entry point. Transforms statistical patterns
        into human-readable insights ranked by priority.

        Args:
            correlations: Detected temporal correlations (TemporalCorrelationResult)
            patterns: Detected behavioral patterns (RecurringPattern)
            knowledge_gaps: Knowledge gaps from CuriosityEngine
            aspirations: User's aspirations for relevance scoring

        Returns:
            List of generated insights ranked by priority
        """
        all_insights: List[EmergentInsight] = []

        # 1. Generate insights from correlations
        if correlations:
            correlation_insights = self._generate_from_correlations(
                correlations,
                aspirations,
            )
            all_insights.extend(correlation_insights[:self.MAX_INSIGHTS_PER_TYPE])

        # 2. Generate insights from patterns
        if patterns:
            pattern_insights = self._generate_from_patterns(
                patterns,
                aspirations,
            )
            all_insights.extend(pattern_insights[:self.MAX_INSIGHTS_PER_TYPE])

        # 3. Generate insights from knowledge gaps
        if knowledge_gaps:
            gap_insights = self._generate_from_gaps(
                knowledge_gaps,
                aspirations,
            )
            all_insights.extend(gap_insights[:self.MAX_INSIGHTS_PER_TYPE])

        # Filter by confidence
        filtered = [
            i for i in all_insights
            if i.confidence >= self.min_confidence
        ]

        # Calculate relevance scores
        for insight in filtered:
            if aspirations and not insight.relevance_score:
                insight.relevance_score = self.calculate_relevance_score(
                    insight,
                    {},  # user_context
                    aspirations,
                )

        # Sort by priority
        filtered.sort(key=lambda i: -i.priority_score)

        # Cache and persist new insights
        if filtered:
            self._cache_insights(filtered)

        logger.info(
            f"Generated {len(filtered)} insights "
            f"(from {len(all_insights)} candidates)"
        )

        return filtered

    def _cache_insights(self, insights: List[EmergentInsight]) -> None:
        """Cache and persist insights to storage."""
        import json

        # Convert insights to dictionaries for storage
        new_entries = []
        for insight in insights:
            entry = {
                "insightId": insight.insight_id,
                "insightType": insight.insight_type.value if hasattr(insight.insight_type, 'value') else str(insight.insight_type),
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence,
                "relevance": insight.relevance_score,
                "priority": "high" if insight.priority_score >= 0.7 else "medium" if insight.priority_score >= 0.4 else "low",
                "sourceEvents": insight.supporting_evidence or [],
                "suggestedActions": insight.suggested_actions or [],
                "createdAt": insight.created_at.isoformat() if hasattr(insight.created_at, 'isoformat') else str(insight.created_at),
                "expiresAt": None,
                "isRead": False,
            }
            new_entries.append(entry)

        # Add to cache (avoid duplicates by insight_id)
        existing_ids = {i.get("insightId") for i in self._cached_insights}
        for entry in new_entries:
            if entry["insightId"] not in existing_ids:
                self._cached_insights.append(entry)

        # Limit cache size (keep most recent 100)
        if len(self._cached_insights) > 100:
            self._cached_insights = self._cached_insights[-100:]

        # Persist to storage
        try:
            self._storage_path.write_text(json.dumps(self._cached_insights, indent=2))
            logger.debug(f"Persisted {len(self._cached_insights)} insights to {self._storage_path}")
        except Exception as e:
            logger.warning(f"Failed to persist insights: {e}")

    def _generate_from_correlations(
        self,
        correlations: List[Any],
        aspirations: Optional[List[Any]] = None,
    ) -> List[EmergentInsight]:
        """Generate insights from temporal correlations."""
        insights: List[EmergentInsight] = []

        for corr in correlations:
            # Check if correlation is significant
            if not self._is_significant_correlation(corr):
                continue

            insight = self.generate_insight_from_correlation(corr, aspirations)
            if insight.confidence >= self.min_confidence:
                insights.append(insight)

        return insights

    def _generate_from_patterns(
        self,
        patterns: List[Any],
        aspirations: Optional[List[Any]] = None,
    ) -> List[EmergentInsight]:
        """Generate insights from recurring patterns."""
        insights: List[EmergentInsight] = []

        for pattern in patterns:
            insight = self._generate_from_pattern(pattern, aspirations)
            if insight and insight.confidence >= self.min_confidence:
                insights.append(insight)

        return insights

    def _generate_from_gaps(
        self,
        gaps: List[Any],
        aspirations: Optional[List[Any]] = None,
    ) -> List[EmergentInsight]:
        """Generate insights from knowledge gaps."""
        insights: List[EmergentInsight] = []

        for gap in gaps:
            insight = self._generate_from_gap(gap, aspirations)
            if insight and insight.confidence >= self.min_confidence:
                insights.append(insight)

        return insights

    def _is_significant_correlation(self, corr: Any) -> bool:
        """Check if correlation meets significance thresholds."""
        # Check for correlation_found attribute
        if hasattr(corr, "correlation_found") and not corr.correlation_found:
            return False

        # Check statistical significance
        if hasattr(corr, "is_statistically_significant"):
            if corr.is_statistically_significant:
                return True

        # Check p-value if available
        if hasattr(corr, "p_value") and corr.p_value is not None:
            if corr.p_value <= self.min_significance:
                return True

        # Check co-occurrences minimum
        if hasattr(corr, "co_occurrences"):
            if corr.co_occurrences is not None and corr.co_occurrences >= self.MIN_OCCURRENCES:
                return True

        return False

    def generate_insight_from_correlation(
        self,
        correlation: Any,
        aspirations: Optional[List[Any]] = None,
    ) -> EmergentInsight:
        """Generate insight from a temporal correlation.

        Transforms statistical correlation data into human-readable insight
        with appropriate confidence levels and suggestions.

        Args:
            correlation: TemporalCorrelationResult from correlation detector
            aspirations: User aspirations for relevance scoring

        Returns:
            EmergentInsight with natural language description
        """
        # Extract correlation data
        event_a = getattr(correlation, "event_type_a", "Event A")
        event_b = getattr(correlation, "event_type_b", "Event B")
        gap_days = getattr(correlation, "avg_gap_days", 0) or 0
        co_occurrences = getattr(correlation, "co_occurrences", 0) or 0
        p_value = getattr(correlation, "p_value", None)
        strength = getattr(correlation, "correlation_strength", 0.5) or 0.5
        is_causal = getattr(correlation, "is_causal_candidate", False)
        effect_size = getattr(correlation, "effect_size", None)
        effect_interp = getattr(correlation, "effect_interpretation", None)

        # Determine insight type
        if is_causal:
            insight_type = InsightType.CAUSAL_HYPOTHESIS
        else:
            insight_type = InsightType.TEMPORAL_CORRELATION

        # Generate title
        title = self._generate_correlation_title(event_a, event_b, gap_days)

        # Generate description using template
        description = self._generate_correlation_description(
            event_a,
            event_b,
            gap_days,
            co_occurrences,
            p_value,
            strength,
        )

        # Calculate confidence
        confidence = self._calculate_correlation_confidence(
            co_occurrences,
            p_value,
            strength,
        )

        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence)

        # Generate statistical summary
        statistical_summary = self._generate_statistical_summary(
            co_occurrences,
            p_value,
            effect_size,
            effect_interp,
        )

        # Generate suggested actions
        actions = self._generate_correlation_actions(
            event_a,
            event_b,
            is_causal,
            confidence_level,
        )

        # Find related aspirations
        related = self._find_related_aspirations(
            event_a,
            event_b,
            aspirations,
        )

        # Build evidence list
        evidence = []
        if hasattr(correlation, "examples"):
            for example in correlation.examples[:5]:
                if isinstance(example, tuple) and len(example) >= 3:
                    evidence.append(f"{example[0]} -> {example[1]} ({example[2]:.1f}d)")

        return EmergentInsight(
            insight_type=insight_type,
            title=title,
            description=description,
            confidence=confidence,
            confidence_level=confidence_level,
            relevance_score=0.5 + 0.1 * len(related),  # Base + aspiration bonus
            supporting_evidence=evidence,
            suggested_actions=actions,
            related_aspirations=related,
            statistical_summary=statistical_summary,
        )

    def _generate_from_pattern(
        self,
        pattern: Any,
        aspirations: Optional[List[Any]] = None,
    ) -> Optional[EmergentInsight]:
        """Generate insight from a recurring pattern."""
        try:
            # Extract pattern data
            pattern_desc = getattr(pattern, "pattern_description", str(pattern))
            occurrences = getattr(pattern, "occurrences", 0)
            significance = getattr(pattern, "statistical_significance", 0.5)

            if occurrences < self.MIN_OCCURRENCES:
                return None

            # Generate insight
            title = f"Recurring pattern: {pattern_desc[:50]}"
            description = self.BEHAVIORAL_TEMPLATES[0].format(
                pattern_description=pattern_desc,
                count=occurrences,
                span_days=90,  # Default span
            )

            return EmergentInsight(
                insight_type=InsightType.BEHAVIORAL_PATTERN,
                title=title,
                description=description,
                confidence=significance,
                confidence_level=self._get_confidence_level(significance),
                relevance_score=0.5,
                statistical_summary=f"{occurrences} occurrences, significance: {significance:.0%}",
            )

        except Exception as e:
            logger.debug(f"Error generating pattern insight: {e}")
            return None

    def _generate_from_gap(
        self,
        gap: Any,
        aspirations: Optional[List[Any]] = None,
    ) -> Optional[EmergentInsight]:
        """Generate insight from a knowledge gap."""
        try:
            # Extract gap data
            gap_type = getattr(gap, "gap_type", None)
            title = getattr(gap, "title", "Knowledge gap")
            description = getattr(gap, "description", "")
            severity = getattr(gap, "severity", 0.5)
            info_gain = getattr(gap, "information_gain", 0.5)
            prompts = getattr(gap, "suggested_exploration", [])

            # Map gap type to insight type
            insight_type = InsightType.KNOWLEDGE_GAP
            if gap_type and "aspiration" in str(gap_type).lower():
                insight_type = InsightType.ASPIRATION_MISALIGNMENT

            # Confidence from gap severity and info gain
            confidence = (severity + info_gain) / 2

            return EmergentInsight(
                insight_type=insight_type,
                title=f"Gap detected: {title[:50]}",
                description=description,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                relevance_score=severity,
                suggested_actions=prompts[:3],
                related_aspirations=getattr(gap, "related_aspirations", []),
            )

        except Exception as e:
            logger.debug(f"Error generating gap insight: {e}")
            return None

    def _generate_correlation_title(
        self,
        event_a: str,
        event_b: str,
        gap_days: float,
    ) -> str:
        """Generate title for correlation insight."""
        if gap_days < 1:
            return f"{event_a} → {event_b} (same day)"
        elif gap_days < 7:
            return f"{event_a} → {event_b} ({gap_days:.0f} days later)"
        else:
            return f"{event_a} precedes {event_b} (by ~{gap_days:.0f} days)"

    def _generate_correlation_description(
        self,
        event_a: str,
        event_b: str,
        gap_days: float,
        count: int,
        p_value: Optional[float],
        strength: float,
    ) -> str:
        """Generate natural language description for correlation."""
        # Choose template based on available data
        if p_value is not None:
            template = self.CORRELATION_TEMPLATES[0]
            return template.format(
                event_a=event_a,
                event_b=event_b,
                gap_days=gap_days,
                count=count,
                p_value=p_value,
            )
        else:
            template = self.CORRELATION_TEMPLATES[2]
            return template.format(
                event_a=event_a,
                event_b=event_b,
                gap_days=gap_days,
                strength=strength,
            )

    def _calculate_correlation_confidence(
        self,
        count: int,
        p_value: Optional[float],
        strength: float,
    ) -> float:
        """Calculate overall confidence for correlation insight."""
        confidence = 0.5  # Base

        # Boost from count
        if count >= 10:
            confidence += 0.2
        elif count >= 5:
            confidence += 0.1

        # Boost from statistical significance
        if p_value is not None:
            if p_value <= 0.01:
                confidence += 0.3
            elif p_value <= 0.05:
                confidence += 0.2
            elif p_value <= 0.1:
                confidence += 0.1

        # Boost from strength
        confidence += strength * 0.2

        return min(1.0, confidence)

    def _get_confidence_level(self, confidence: float) -> InsightConfidence:
        """Map numerical confidence to categorical level."""
        if confidence >= 0.8:
            return InsightConfidence.HIGH
        elif confidence >= 0.5:
            return InsightConfidence.MEDIUM
        else:
            return InsightConfidence.LOW

    def _generate_statistical_summary(
        self,
        count: int,
        p_value: Optional[float],
        effect_size: Optional[float],
        effect_interp: Optional[str],
    ) -> str:
        """Generate statistical summary text."""
        parts = [f"n={count}"]

        if p_value is not None:
            parts.append(f"p={p_value:.4f}")

        if effect_size is not None:
            effect_text = f"effect size={effect_size:.2f}"
            if effect_interp:
                effect_text += f" ({effect_interp})"
            parts.append(effect_text)

        return ", ".join(parts)

    def _generate_correlation_actions(
        self,
        event_a: str,
        event_b: str,
        is_causal: bool,
        confidence_level: InsightConfidence,
    ) -> List[str]:
        """Generate suggested actions for correlation insight."""
        actions = []

        if is_causal and confidence_level == InsightConfidence.HIGH:
            actions.append(
                f"Consider whether {event_a} might be causing {event_b}"
            )
            actions.append(
                f"Look for confounding factors that might explain both"
            )
        elif is_causal:
            actions.append(
                f"This might be causal - consider gathering more data"
            )

        if confidence_level in [InsightConfidence.HIGH, InsightConfidence.MEDIUM]:
            actions.append(
                f"Track how changes to {event_a} affect {event_b}"
            )

        return actions[:3]

    def _find_related_aspirations(
        self,
        event_a: str,
        event_b: str,
        aspirations: Optional[List[Any]],
    ) -> List[str]:
        """Find aspirations related to the events."""
        if not aspirations:
            return []

        related = []
        event_text = f"{event_a} {event_b}".lower()

        for asp in aspirations:
            asp_name = str(asp.name if hasattr(asp, "name") else asp).lower()
            asp_id = str(asp.id if hasattr(asp, "id") else asp)

            # Simple keyword matching
            if any(word in event_text for word in asp_name.split()):
                related.append(asp_id)

        return related[:3]

    def calculate_relevance_score(
        self,
        insight: EmergentInsight,
        user_context: Dict[str, Any],
        aspirations: Optional[List[Any]] = None,
    ) -> float:
        """Calculate how relevant an insight is to the user.

        Phase 2C: Uses RankingModel for personalized scoring when available.

        Considers:
        - Alignment with user's Aspirational Self
        - Novelty (new pattern vs previously known)
        - Actionability (can user do something about it?)
        - Statistical strength of underlying pattern
        - User feedback history (Phase 2C)

        Args:
            insight: The insight to score
            user_context: User context information
            aspirations: User's aspirations

        Returns:
            Relevance score (0-1)
        """
        # Phase 2C: Use personalized ranking model if available
        if self._ranking_model is not None:
            try:
                # Calculate aspiration alignment
                aspiration_alignment = 0.15 * len(insight.related_aspirations) if insight.related_aspirations else 0.0

                # Calculate age in days
                age_days = 0
                if hasattr(insight, 'created_at') and insight.created_at:
                    age_days = (datetime.utcnow() - insight.created_at).days

                # Get personalized score from ranking model
                score = self._ranking_model.compute_relevance_score(
                    insight_type=insight.insight_type.value if hasattr(insight.insight_type, 'value') else str(insight.insight_type),
                    confidence=insight.confidence,
                    base_relevance=0.5 + (0.1 if insight.suggested_actions else 0.0),
                    aspiration_alignment=aspiration_alignment,
                    age_days=age_days,
                )

                # Apply dismissed penalty
                if insight.dismissed:
                    score -= 0.3

                return max(0.0, min(1.0, score))

            except Exception as e:
                logger.debug(f"Personalized scoring failed, using default: {e}")

        # Fallback: Default scoring logic
        score = 0.5  # Base relevance

        # Boost for aspiration alignment
        if insight.related_aspirations:
            score += 0.15 * len(insight.related_aspirations)

        # Boost for actionability
        if insight.suggested_actions:
            score += 0.1

        # Boost for high confidence
        if insight.confidence_level == InsightConfidence.HIGH:
            score += 0.1

        # Boost for causal insights (more actionable)
        if insight.insight_type == InsightType.CAUSAL_HYPOTHESIS:
            score += 0.1

        # Penalty for dismissed insights
        if insight.dismissed:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def suggest_actions(
        self,
        insight: EmergentInsight,
        user_context: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable suggestions based on the insight.

        Example suggestions:
        - "Consider scheduling important proposals on Mondays"
        - "Link Project Titan notes to your 'Lead projects' aspiration"
        - "Journal when feeling fatigued to track energy patterns"

        Args:
            insight: The insight
            user_context: User context

        Returns:
            List of action suggestions
        """
        actions = list(insight.suggested_actions)

        # Add generic actions based on insight type
        if insight.insight_type == InsightType.TEMPORAL_CORRELATION:
            if "productivity" in insight.title.lower():
                actions.append("Track this correlation over the next 2 weeks")

        elif insight.insight_type == InsightType.KNOWLEDGE_GAP:
            actions.append("Set aside time to explore this gap")

        elif insight.insight_type == InsightType.ASPIRATION_MISALIGNMENT:
            actions.append("Review how your current work aligns with this goal")

        return actions[:5]

    def export_for_token_priors(
        self,
        insights: List[EmergentInsight],
    ) -> str:
        """Export insights as natural language for token prior storage.

        Option B Compliance: Learning through natural language priors.

        Args:
            insights: List of insights to export

        Returns:
            Natural language summary for token priors
        """
        lines = ["Discovered patterns and insights:"]

        for insight in insights[:10]:  # Limit to top 10
            lines.append("")
            lines.append(insight.to_natural_language())

        return "\n".join(lines)

"""User Feedback System for Insight Ranking.

Phase 2C: Intelligent Ranking System

This module implements user feedback collection and adaptive ranking
based on user preferences. Feedback is stored locally and used to
personalize insight prioritization.

Research Foundation:
- Training-Free GRPO (2510.08191v1): Natural language learning
- Human feedback integration patterns from RLHF literature

Option B Compliance:
- No model parameter updates
- Feedback stored as JSON for token prior context
- Ghost model FROZEN
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class FeedbackRating(str, Enum):
    """User feedback rating for an insight."""

    VALUABLE = "valuable"  # User found this insight helpful
    NOT_VALUABLE = "not_valuable"  # Insight was not useful
    DISMISS = "dismiss"  # User wants to hide this insight
    NEUTRAL = "neutral"  # No strong opinion


@dataclass
class InsightFeedback:
    """Represents user feedback on an insight.

    Attributes:
        feedback_id: Unique identifier
        insight_id: The insight being rated
        rating: User's rating (valuable, not_valuable, dismiss)
        timestamp: When feedback was submitted
        context: Optional explanation from user
        insight_type: Type of insight (for learning patterns)
        insight_confidence: Original insight confidence
    """

    feedback_id: str = field(default_factory=lambda: str(uuid4()))
    insight_id: str = ""
    rating: FeedbackRating = FeedbackRating.NEUTRAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Optional[str] = None
    insight_type: Optional[str] = None
    insight_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "insight_id": self.insight_id,
            "rating": self.rating.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "insight_type": self.insight_type,
            "insight_confidence": self.insight_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsightFeedback":
        """Create from dictionary."""
        return cls(
            feedback_id=data.get("feedback_id", str(uuid4())),
            insight_id=data.get("insight_id", ""),
            rating=FeedbackRating(data.get("rating", "neutral")),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            context=data.get("context"),
            insight_type=data.get("insight_type"),
            insight_confidence=data.get("insight_confidence"),
        )


class FeedbackStore:
    """Persistent storage for user feedback.

    Stores feedback in a JSON file at ~/.futurnal/insights/feedback.json
    and provides methods for querying feedback history.
    """

    DEFAULT_FEEDBACK_PATH = "~/.futurnal/insights/feedback.json"

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize feedback store.

        Args:
            storage_path: Path to feedback JSON file
        """
        self._storage_path = Path(
            os.path.expanduser(storage_path or self.DEFAULT_FEEDBACK_PATH)
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._feedback: List[InsightFeedback] = []
        self._load()

        logger.info(
            f"FeedbackStore initialized with {len(self._feedback)} entries "
            f"(path={self._storage_path})"
        )

    def _load(self) -> None:
        """Load feedback from storage."""
        if not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            self._feedback = [InsightFeedback.from_dict(f) for f in data]
        except Exception as e:
            logger.warning(f"Failed to load feedback: {e}")

    def _save(self) -> None:
        """Save feedback to storage."""
        try:
            data = [f.to_dict() for f in self._feedback]
            self._storage_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self._feedback)} feedback entries")
        except Exception as e:
            logger.warning(f"Failed to save feedback: {e}")

    def add_feedback(self, feedback: InsightFeedback) -> None:
        """Add new feedback entry.

        Args:
            feedback: The feedback to add
        """
        # Check for existing feedback on same insight
        for i, existing in enumerate(self._feedback):
            if existing.insight_id == feedback.insight_id:
                # Update existing feedback
                self._feedback[i] = feedback
                self._save()
                logger.info(f"Updated feedback for insight {feedback.insight_id}: {feedback.rating.value}")
                return

        self._feedback.append(feedback)
        self._save()
        logger.info(f"Added feedback for insight {feedback.insight_id}: {feedback.rating.value}")

    def get_feedback_for_insight(self, insight_id: str) -> Optional[InsightFeedback]:
        """Get feedback for a specific insight.

        Args:
            insight_id: The insight ID

        Returns:
            InsightFeedback if exists, None otherwise
        """
        for feedback in self._feedback:
            if feedback.insight_id == insight_id:
                return feedback
        return None

    def get_all_feedback(self) -> List[InsightFeedback]:
        """Get all feedback entries."""
        return list(self._feedback)

    def get_feedback_by_rating(self, rating: FeedbackRating) -> List[InsightFeedback]:
        """Get feedback entries with a specific rating.

        Args:
            rating: The rating to filter by

        Returns:
            List of feedback entries with that rating
        """
        return [f for f in self._feedback if f.rating == rating]

    def get_feedback_stats(self) -> Dict[str, int]:
        """Get aggregate feedback statistics.

        Returns:
            Dictionary with counts per rating
        """
        stats = {r.value: 0 for r in FeedbackRating}
        for feedback in self._feedback:
            stats[feedback.rating.value] += 1
        return stats

    def get_type_preferences(self) -> Dict[str, float]:
        """Calculate preference scores by insight type.

        Returns:
            Dictionary mapping insight_type to preference score (-1 to 1)
        """
        type_scores: Dict[str, List[float]] = {}

        for feedback in self._feedback:
            if not feedback.insight_type:
                continue

            if feedback.insight_type not in type_scores:
                type_scores[feedback.insight_type] = []

            # Map rating to score
            score_map = {
                FeedbackRating.VALUABLE: 1.0,
                FeedbackRating.NOT_VALUABLE: -0.5,
                FeedbackRating.DISMISS: -1.0,
                FeedbackRating.NEUTRAL: 0.0,
            }
            type_scores[feedback.insight_type].append(score_map[feedback.rating])

        # Calculate average scores
        return {
            insight_type: sum(scores) / len(scores)
            for insight_type, scores in type_scores.items()
            if scores
        }


class RankingModel:
    """Adaptive ranking model based on user feedback.

    Uses feedback history to personalize insight ranking by:
    1. Boosting insight types the user finds valuable
    2. Penalizing insight types the user dismisses
    3. Adjusting confidence thresholds based on preferences

    Option B Compliance:
    - No model weight updates
    - All adjustments via stored preferences
    - Ghost model remains FROZEN
    """

    # Base weights for ranking factors
    DEFAULT_WEIGHTS = {
        "confidence": 0.30,
        "relevance": 0.25,
        "type_preference": 0.20,
        "recency": 0.15,
        "aspiration_alignment": 0.10,
    }

    def __init__(self, feedback_store: Optional[FeedbackStore] = None):
        """Initialize ranking model.

        Args:
            feedback_store: Feedback storage (creates default if None)
        """
        self.feedback_store = feedback_store or FeedbackStore()
        self._weights = dict(self.DEFAULT_WEIGHTS)

        # Adjust weights based on feedback history
        self._update_weights_from_feedback()

        logger.info(f"RankingModel initialized (weights={self._weights})")

    def _update_weights_from_feedback(self) -> None:
        """Update ranking weights based on feedback patterns."""
        stats = self.feedback_store.get_feedback_stats()
        total = sum(stats.values())

        if total < 5:
            # Not enough feedback to adjust
            return

        valuable_ratio = stats.get("valuable", 0) / total
        dismiss_ratio = stats.get("dismiss", 0) / total

        # If user values many insights, boost confidence weight
        if valuable_ratio > 0.6:
            self._weights["confidence"] = min(0.40, self._weights["confidence"] + 0.05)

        # If user dismisses many, boost type_preference weight
        if dismiss_ratio > 0.3:
            self._weights["type_preference"] = min(0.30, self._weights["type_preference"] + 0.05)

        # Normalize weights
        total_weight = sum(self._weights.values())
        self._weights = {k: v / total_weight for k, v in self._weights.items()}

        logger.debug(f"Updated ranking weights from {total} feedback entries")

    def update_from_feedback(self, feedback: InsightFeedback) -> None:
        """Update ranking model with new feedback.

        Args:
            feedback: New feedback to incorporate
        """
        self.feedback_store.add_feedback(feedback)
        self._update_weights_from_feedback()

    def compute_relevance_score(
        self,
        insight_type: str,
        confidence: float,
        base_relevance: float,
        aspiration_alignment: float = 0.0,
        age_days: int = 0,
    ) -> float:
        """Compute personalized relevance score for an insight.

        Args:
            insight_type: Type of insight (e.g., "temporal_correlation")
            confidence: Original confidence score (0-1)
            base_relevance: Base relevance score (0-1)
            aspiration_alignment: Alignment with user aspirations (0-1)
            age_days: Days since insight was generated

        Returns:
            Personalized relevance score (0-1)
        """
        # Get type preference from feedback history
        type_prefs = self.feedback_store.get_type_preferences()
        type_pref = type_prefs.get(insight_type, 0.0)

        # Normalize type preference to 0-1 range
        type_pref_normalized = (type_pref + 1.0) / 2.0  # -1..1 -> 0..1

        # Calculate recency score (exponential decay)
        recency_score = max(0.0, 1.0 - (age_days / 30.0) * 0.5)

        # Weighted combination
        score = (
            self._weights["confidence"] * confidence +
            self._weights["relevance"] * base_relevance +
            self._weights["type_preference"] * type_pref_normalized +
            self._weights["recency"] * recency_score +
            self._weights["aspiration_alignment"] * aspiration_alignment
        )

        return min(1.0, max(0.0, score))

    def get_personalized_weights(self) -> Dict[str, float]:
        """Get current personalized ranking weights.

        Returns:
            Dictionary of weight factors
        """
        return dict(self._weights)

    def should_show_insight(
        self,
        insight_type: str,
        confidence: float,
    ) -> bool:
        """Determine if insight should be shown based on user preferences.

        Args:
            insight_type: Type of insight
            confidence: Insight confidence

        Returns:
            True if insight should be shown
        """
        type_prefs = self.feedback_store.get_type_preferences()
        type_pref = type_prefs.get(insight_type, 0.0)

        # If user consistently dismisses this type, raise threshold
        if type_pref < -0.5:
            return confidence >= 0.8  # Higher threshold for disliked types
        elif type_pref > 0.5:
            return confidence >= 0.4  # Lower threshold for liked types
        else:
            return confidence >= 0.5  # Default threshold

    def export_for_token_priors(self) -> str:
        """Export feedback patterns as natural language for token priors.

        Option B Compliance: Learning through natural language context.

        Returns:
            Natural language summary of user preferences
        """
        stats = self.feedback_store.get_feedback_stats()
        type_prefs = self.feedback_store.get_type_preferences()

        lines = ["User insight preferences:"]

        total = sum(stats.values())
        if total > 0:
            lines.append(f"- Total feedback: {total} ratings")
            lines.append(f"- Valuable: {stats.get('valuable', 0)} ({stats.get('valuable', 0)/total:.0%})")
            lines.append(f"- Not valuable: {stats.get('not_valuable', 0)} ({stats.get('not_valuable', 0)/total:.0%})")

        if type_prefs:
            lines.append("- Type preferences:")
            for insight_type, score in sorted(type_prefs.items(), key=lambda x: -x[1]):
                preference = "liked" if score > 0.3 else "disliked" if score < -0.3 else "neutral"
                lines.append(f"  * {insight_type}: {preference} ({score:+.2f})")

        return "\n".join(lines)


# Global instances
_feedback_store: Optional[FeedbackStore] = None
_ranking_model: Optional[RankingModel] = None


def get_feedback_store() -> FeedbackStore:
    """Get the default feedback store singleton."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store


def get_ranking_model() -> RankingModel:
    """Get the default ranking model singleton."""
    global _ranking_model
    if _ranking_model is None:
        _ranking_model = RankingModel(get_feedback_store())
    return _ranking_model

"""Aspirational Self models for AI alignment and reward signals.

The Aspirational Self represents the user's goals, habits, and values—serving as
the reward signal that guides the Animal's development in Phase 3. By explicitly
defining aspirations, users give the AI a framework for understanding what matters
and detecting misalignments between stated goals and observed behavior.

Phase 3 (Guide) will use these models for:
- Reward Signal Dashboard showing alignment with aspirations
- Misalignment detection (e.g., "90% of reading unrelated to stated goal")
- Guided causal exploration toward aspiration fulfillment
- Progress tracking against personal growth objectives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class AspirationCategory(str, Enum):
    """Categories of user aspirations.

    These categories help the Animal understand the type of guidance needed:
    - HABIT: Recurring behaviors to develop/maintain
    - SKILL: Capabilities to acquire or improve
    - VALUE: Core principles guiding decisions
    - OUTCOME: Specific results to achieve
    - RELATIONSHIP: Interpersonal goals
    """

    HABIT = "habit"
    SKILL = "skill"
    VALUE = "value"
    OUTCOME = "outcome"
    RELATIONSHIP = "relationship"


@dataclass
class Aspiration:
    """Represents a user's goal, habit, value, or aspiration.

    Aspirations serve as the Animal's reward signal in Phase 3, helping the AI
    understand what matters to the user and detect patterns of alignment or
    misalignment with stated goals.

    Example aspirations:
    - "Develop expertise in causal inference" (SKILL)
    - "Maintain peak creative output" (HABIT)
    - "Lead high-impact projects" (OUTCOME)
    - "Prioritize deep work over shallow tasks" (VALUE)

    Attributes:
        aspiration_id: Unique identifier
        category: Type of aspiration (habit, skill, value, outcome, relationship)
        title: Short description (e.g., "Learn Causal Inference")
        description: Detailed explanation of the aspiration
        priority: User-assigned priority (1-10, higher = more important)
        created_at: When this aspiration was defined
        target_date: Optional deadline or milestone date
        progress_metrics: Custom metrics for tracking progress
        related_events: Links to experiential events supporting this aspiration
        alignment_score: Current alignment with behavior (0.0-1.0, Phase 3)
    """

    aspiration_id: str = field(default_factory=lambda: str(uuid4()))
    category: AspirationCategory = AspirationCategory.OUTCOME
    title: str = ""
    description: str = ""
    priority: int = field(default=5)  # 1-10 scale
    created_at: datetime = field(default_factory=datetime.utcnow)
    target_date: Optional[datetime] = None
    progress_metrics: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)

    # Phase 3: Alignment tracking
    alignment_score: float = 0.0  # 0.0 = no alignment, 1.0 = perfect alignment

    def __post_init__(self):
        """Validate fields."""
        if not self.title:
            raise ValueError("title is required for Aspiration")
        if not 1 <= self.priority <= 10:
            raise ValueError("priority must be between 1 and 10")
        if not 0.0 <= self.alignment_score <= 1.0:
            raise ValueError("alignment_score must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "aspiration_id": self.aspiration_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "progress_metrics": self.progress_metrics,
            "related_events": self.related_events,
            "alignment_score": self.alignment_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Aspiration:
        """Create from dictionary."""
        data_copy = data.copy()
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        if "target_date" in data_copy and data_copy["target_date"]:
            data_copy["target_date"] = datetime.fromisoformat(data_copy["target_date"])
        if "category" in data_copy and isinstance(data_copy["category"], str):
            data_copy["category"] = AspirationCategory(data_copy["category"])
        return cls(**data_copy)


@dataclass
class AspirationAlignment:
    """Links aspirations to experiential events for alignment tracking.

    Phase 3 (Guide) uses these links to:
    - Calculate alignment scores showing progress toward aspirations
    - Detect misalignments (e.g., "reading unrelated to goal")
    - Generate Reward Signal Dashboard metrics
    - Guide users toward aspiration-aligned behaviors

    Attributes:
        alignment_id: Unique identifier
        aspiration_id: Reference to the aspiration
        event_id: Reference to the experiential event
        alignment_type: How this event relates to the aspiration
        contribution_score: How much this event advances the aspiration (-1.0 to 1.0)
        detected_at: When this alignment was detected
        confidence: Confidence in the alignment assessment (0.0-1.0)
        explanation: Human-readable explanation of the alignment
    """

    alignment_id: str = field(default_factory=lambda: str(uuid4()))
    aspiration_id: str = ""
    event_id: str = ""
    alignment_type: str = "supports"  # 'supports', 'opposes', 'neutral'
    contribution_score: float = 0.0  # -1.0 (opposes) to 1.0 (strongly supports)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0  # 0.0 to 1.0
    explanation: str = ""

    def __post_init__(self):
        """Validate fields."""
        if not self.aspiration_id:
            raise ValueError("aspiration_id is required")
        if not self.event_id:
            raise ValueError("event_id is required")
        if not -1.0 <= self.contribution_score <= 1.0:
            raise ValueError("contribution_score must be between -1.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "alignment_id": self.alignment_id,
            "aspiration_id": self.aspiration_id,
            "event_id": self.event_id,
            "alignment_type": self.alignment_type,
            "contribution_score": self.contribution_score,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AspirationAlignment:
        """Create from dictionary."""
        data_copy = data.copy()
        if "detected_at" in data_copy and isinstance(data_copy["detected_at"], str):
            data_copy["detected_at"] = datetime.fromisoformat(data_copy["detected_at"])
        return cls(**data_copy)

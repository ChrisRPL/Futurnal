"""Experiential data models for Ghost-to-Animal evolution.

This module defines data structures for experiential learning and AI personalization,
representing the conceptual foundation for transforming generic AI (Ghost) into
personalized experiential intelligence (Animal).

Key Concepts:
- **Experiential Events**: Timestamped occurrences in the user's data stream that
  the Ghost learns from, emphasizing temporal context over static documents.
- **Aspirations**: User's goals, habits, and values that serve as the reward signal
  guiding the Animal's development (Phase 3).

Phase Roadmap:
- Phase 1 (Current): Establish vocabulary and data structures
- Phase 2 (Analyst): Use ExperientialEvent for temporal correlation detection
- Phase 3 (Guide): Use Aspiration for alignment tracking and reward signals
"""

from .experiential import ExperientialEvent, ExperientialStream
from .aspirational import Aspiration, AspirationAlignment, AspirationCategory

__all__ = [
    "ExperientialEvent",
    "ExperientialStream",
    "Aspiration",
    "AspirationAlignment",
    "AspirationCategory",
]

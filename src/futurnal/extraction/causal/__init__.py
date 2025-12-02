"""Causal Structure Extraction for Phase 1 Module 05.

This package implements causal structure preparation for Phase 3 causal inference:
- Event extraction with temporal grounding
- Event-event relationship detection as causal candidates
- Bradford Hill criteria preparation
- Integration with schema evolution and experiential learning

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Option B Compliance:
- Ghost model frozen (no parameter updates)
- Experiential knowledge as token priors
- Temporal-first design (events must have timestamps)
- Autonomous schema evolution (event types discoverable)
"""

from futurnal.extraction.causal.bradford_hill_prep import BradfordHillPreparation
from futurnal.extraction.causal.event_extractor import EventExtractor
from futurnal.extraction.causal.models import (
    BradfordHillCriteria,
    CausalCandidate,
    CausalRelationshipType,
    EventType,
)
from futurnal.extraction.causal.relationship_detector import CausalRelationshipDetector

__all__ = [
    "EventType",
    "CausalRelationshipType",
    "CausalCandidate",
    "BradfordHillCriteria",
    "EventExtractor",
    "CausalRelationshipDetector",
    "BradfordHillPreparation",
]

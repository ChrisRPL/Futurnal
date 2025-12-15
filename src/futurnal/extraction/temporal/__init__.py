"""Temporal extraction module for Module 01.

This module implements comprehensive temporal reasoning capabilities including:
- Temporal marker extraction (explicit timestamps, relative expressions)
- Temporal relationship detection (Allen's Interval Algebra + Causal)
- Event extraction with temporal grounding
- Training-Free GRPO for experiential learning
- TOTAL thought templates for evolving reasoning patterns
- AgentFlow architecture for temporal analysis

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/01-temporal-extraction.md

Option B Compliance:
- Ghost model frozen (no parameter updates)
- Experiential knowledge as token priors
- TOTAL thought templates with textual gradients
- AgentFlow 4-module architecture
- Temporal-first design (>85% accuracy target)
"""

from futurnal.extraction.temporal.models import (
    # Enums
    TemporalRelationshipType,
    TemporalSourceType,
    # Core data structures
    TemporalSource,
    TemporalMark,
    ChunkReference,
    TemporalTriple,
    Event,
    TemporalEntity,
    TemporalRelationship,
    # GRPO models
    ExperientialTemporalKnowledge,
    # TOTAL models
    ThoughtTemplate,
    # Result models
    TemporalExtractionResult,
    ValidationResult,
)

from futurnal.extraction.temporal.consistency import (
    TemporalConsistencyValidator,
    TemporalInconsistency,
    ViolationType,
    Severity,
    validate_temporal_consistency,
)

from futurnal.extraction.temporal.enricher import (
    TemporalEnricher,
    enrich_with_temporal,
)

__all__ = [
    # Enums
    "TemporalRelationshipType",
    "TemporalSourceType",
    # Core data structures
    "TemporalSource",
    "TemporalMark",
    "ChunkReference",
    "TemporalTriple",
    "Event",
    "TemporalEntity",
    "TemporalRelationship",
    # GRPO models
    "ExperientialTemporalKnowledge",
    # TOTAL models
    "ThoughtTemplate",
    # Result models
    "TemporalExtractionResult",
    "ValidationResult",
    # Consistency validation (Step 04)
    "TemporalConsistencyValidator",
    "TemporalInconsistency",
    "ViolationType",
    "Severity",
    "validate_temporal_consistency",
    # Pipeline integration (Step 04)
    "TemporalEnricher",
    "enrich_with_temporal",
]

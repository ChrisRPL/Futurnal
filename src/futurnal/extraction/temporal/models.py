"""Temporal extraction data models for Module 01.

This module defines all data structures for temporal extraction including:
- Temporal relationship types (Allen's Interval Algebra + Causal extensions)
- Temporal markers and triples
- Events with temporal grounding
- Experiential knowledge for Training-Free GRPO
- Thought templates for TOTAL framework

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/01-temporal-extraction.md

Option B Compliance:
- Ghost model frozen (no parameter updates)
- Experiential knowledge as token priors (not weights)
- TOTAL thought templates with textual gradients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TemporalRelationshipType(Enum):
    """Allen's Interval Algebra + Causal Extensions.

    Implements comprehensive temporal relationship types for:
    - Core temporal relationships (Allen's 13 interval relations)
    - Causal temporal relationships (Phase 3 preparation)
    - Concurrent relationships

    Reference: Allen, J. F. (1983). "Maintaining knowledge about temporal intervals"
    """

    # Core temporal relationships (Allen's Interval Algebra)
    BEFORE = "before"           # A finishes before B starts
    AFTER = "after"             # A starts after B finishes
    DURING = "during"           # A occurs within B's timespan
    CONTAINS = "contains"       # B occurs within A's timespan
    OVERLAPS = "overlaps"       # A and B overlap partially
    MEETS = "meets"             # A finishes exactly when B starts
    STARTS = "starts"           # A and B start together
    FINISHES = "finishes"       # A and B finish together
    EQUALS = "equals"           # A and B have identical timespan

    # Causal temporal relationships (Phase 3 preparation)
    CAUSES = "causes"           # A temporally precedes and causes B
    ENABLES = "enables"         # A creates conditions for B
    PREVENTS = "prevents"       # A blocks B from occurring
    TRIGGERS = "triggers"       # A directly initiates B

    # Concurrent relationships
    SIMULTANEOUS = "simultaneous"  # A and B occur at same time
    PARALLEL = "parallel"          # A and B occur in parallel


class TemporalSourceType(Enum):
    """Type of temporal information source."""

    EXPLICIT = "explicit"                    # Explicit timestamp in text
    INFERRED = "inferred"                    # Inferred from context
    DOCUMENT_METADATA = "document_metadata"  # From document metadata
    RELATIVE = "relative"                    # Relative expression parsed


# ---------------------------------------------------------------------------
# Core Temporal Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TemporalSource:
    """Track origin of temporal information.

    Provides provenance tracking for temporal metadata, enabling:
    - Confidence calibration based on source type
    - Debugging temporal extraction issues
    - Auditing temporal inference decisions
    """

    source_type: TemporalSourceType
    evidence: str                        # Original text that led to temporal inference
    inference_method: Optional[str] = None  # Method used for inference (e.g., "rule_based", "llm_inference")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_type": self.source_type.value,
            "evidence": self.evidence,
            "inference_method": self.inference_method,
        }


@dataclass
class TemporalMark:
    """Individual temporal marker extracted from text.

    Represents a single temporal reference (timestamp, date, time expression)
    with metadata about its origin and confidence.
    """

    text: str                           # Original text (e.g., "January 15, 2024")
    timestamp: Optional[datetime]       # Parsed timestamp (if absolute)
    temporal_type: TemporalSourceType   # How was this extracted?
    confidence: float = 1.0             # Confidence in extraction (0.0-1.0)
    span_start: Optional[int] = None    # Character offset in source text
    span_end: Optional[int] = None      # Character offset end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "temporal_type": self.temporal_type.value,
            "confidence": self.confidence,
            "span_start": self.span_start,
            "span_end": self.span_end,
        }


@dataclass
class ChunkReference:
    """Reference to source chunk in document.

    Enables precise provenance tracking for temporal triples.
    """

    document_id: str
    chunk_id: Optional[str] = None
    element_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "element_id": self.element_id,
        }


@dataclass
class TemporalTriple:
    """Enhanced triple with comprehensive temporal metadata.

    Extends the semantic triple concept with temporal information:
    - When did this relationship exist?
    - How long did it last?
    - What temporal relationship type is this?
    - Where did the temporal information come from?

    This is the core data structure for temporal knowledge graph construction.
    """

    # Core triple
    subject: str
    predicate: str
    object: str

    # Temporal metadata (REQUIRED for temporal triples)
    timestamp: Optional[datetime] = None          # When did this occur?
    duration: Optional[timedelta] = None          # How long did it last?
    temporal_type: Optional[TemporalRelationshipType] = None  # BEFORE/AFTER/DURING/CAUSES/etc.
    valid_from: Optional[datetime] = None         # When did this become true?
    valid_to: Optional[datetime] = None           # When did this stop being true?

    # Provenance
    temporal_source: Optional[TemporalSource] = None  # Where did temporal info come from?
    provenance: Optional[ChunkReference] = None       # Source chunk in document

    # Confidence
    confidence: float = 1.0                      # Overall confidence (0.0-1.0)
    temporal_confidence: float = 1.0             # Confidence in temporal information

    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration": self.duration.total_seconds() if self.duration else None,
            "temporal_type": self.temporal_type.value if self.temporal_type else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "temporal_source": self.temporal_source.to_dict() if self.temporal_source else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "confidence": self.confidence,
            "temporal_confidence": self.temporal_confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Event:
    """Represents an event with temporal grounding.

    Events are distinct from static entities:
    - Events MUST have temporal grounding (timestamp)
    - Events have participants and relationships
    - Events are the foundation for Phase 3 causal inference

    Example events:
    - "meeting with team on 2024-01-15"
    - "email sent at 2024-01-15T14:30:00Z"
    - "decision made after reviewing proposal"
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    event_type: str = ""

    # REQUIRED temporal grounding
    timestamp: Optional[datetime] = None           # When did this occur?
    duration: Optional[timedelta] = None           # How long did it last?

    # Context
    participants: List[str] = field(default_factory=list)
    temporal_relationships: Dict[str, TemporalRelationshipType] = field(default_factory=dict)

    # Provenance
    source_document: str = ""
    extraction_confidence: float = 1.0

    def __post_init__(self):
        """Validate event has required fields."""
        if not self.name:
            raise ValueError("Event name is required")
        if not self.event_type:
            raise ValueError("Event type is required")
        if not self.source_document:
            raise ValueError("Event source_document is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "name": self.name,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration": self.duration.total_seconds() if self.duration else None,
            "participants": self.participants,
            "temporal_relationships": {
                k: v.value for k, v in self.temporal_relationships.items()
            },
            "source_document": self.source_document,
            "extraction_confidence": self.extraction_confidence,
        }


@dataclass
class TemporalEntity:
    """Entity with temporal context for relationship detection.

    Lightweight structure for passing entities through relationship detection.
    """

    entity_id: str
    name: str
    context: str                    # Surrounding text
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "context": self.context,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class TemporalRelationship:
    """Relationship between temporal entities.

    Represents a detected temporal relationship between two entities/events.
    Used during relationship detection before converting to TemporalTriple.
    """

    entity1_id: str
    entity2_id: str
    relationship_type: TemporalRelationshipType
    confidence: float
    evidence: str = ""              # Text evidence for this relationship

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entity1_id": self.entity1_id,
            "entity2_id": self.entity2_id,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Training-Free GRPO Models
# ---------------------------------------------------------------------------


@dataclass
class ExperientialTemporalKnowledge:
    """Experiential knowledge for temporal extraction (token priors).

    Implements Training-Free GRPO paradigm:
    - Experiential knowledge stored as natural language patterns
    - Token priors, NOT parameter updates
    - Ghost model remains frozen
    - Knowledge evolves via semantic advantages

    Reference: Training-Free GRPO (ArXiv 2510.08191v1)
    """

    pattern: str                           # Natural language pattern description
    guidance: str                          # Extraction guidance for this pattern
    success_count: int = 0                 # Times this pattern succeeded
    failure_count: int = 0                 # Times this pattern failed
    confidence: float = 0.5                # Pattern reliability (0.0-1.0)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_success(self):
        """Update after successful application."""
        self.success_count += 1
        self.confidence = self.success_count / (self.success_count + self.failure_count)
        self.updated_at = datetime.utcnow()

    def update_failure(self):
        """Update after failed application."""
        self.failure_count += 1
        self.confidence = self.success_count / (self.success_count + self.failure_count)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern": self.pattern,
            "guidance": self.guidance,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# TOTAL Thought Template Models
# ---------------------------------------------------------------------------


@dataclass
class ThoughtTemplate:
    """Thought template for temporal reasoning (TOTAL framework).

    Implements TOTAL paradigm:
    - Templates as natural language reasoning patterns
    - Textual gradients (KEEP/FIX/DISCARD) refine templates
    - Template composition for complex reasoning
    - Human-readable and inspectable

    Reference: TOTAL - Thought Templates (ArXiv 2510.07499v1)
    """

    template_id: str
    name: str
    content: str                           # Template text (reasoning instructions)
    textual_gradient: str = ""             # KEEP/FIX/DISCARD feedback
    composition_rules: List[str] = field(default_factory=list)  # How to combine with other templates
    success_rate: float = 0.0              # Template effectiveness (0.0-1.0)
    use_count: int = 0                     # Times this template was used
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def record_use(self, success: bool):
        """Record template usage."""
        self.use_count += 1
        # Update success rate with exponential moving average
        alpha = 0.2  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        self.updated_at = datetime.utcnow()

    def apply_textual_gradient(self, gradient: str):
        """Apply textual gradient to update template."""
        self.textual_gradient = gradient
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "content": self.content,
            "textual_gradient": self.textual_gradient,
            "composition_rules": self.composition_rules,
            "success_rate": self.success_rate,
            "use_count": self.use_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


@dataclass
class TemporalExtractionResult:
    """Result of temporal extraction pipeline.

    Contains all extracted temporal information:
    - Temporal markers found in text
    - Temporal relationships detected
    - Events extracted with temporal grounding
    - Temporal triples ready for PKG storage
    """

    temporal_markers: List[TemporalMark] = field(default_factory=list)
    temporal_relationships: List[TemporalRelationship] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    temporal_triples: List[TemporalTriple] = field(default_factory=list)
    summary: str = ""                      # Human-readable summary (from Generator)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "temporal_markers": [m.to_dict() for m in self.temporal_markers],
            "temporal_relationships": [r.to_dict() for r in self.temporal_relationships],
            "events": [e.to_dict() for e in self.events],
            "temporal_triples": [t.to_dict() for t in self.temporal_triples],
            "summary": self.summary,
        }


@dataclass
class ValidationResult:
    """Result of temporal consistency validation."""

    valid: bool
    relationships: List[TemporalRelationship] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "valid": self.valid,
            "relationships": [r.to_dict() for r in self.relationships],
            "errors": self.errors,
            "warnings": self.warnings,
        }

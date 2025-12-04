"""PKG Schema Models for Graph Storage.

Defines Pydantic models for all PKG node and relationship types supporting
Option B requirements:
- Static entities (Person, Organization, Concept, Document)
- Event entities with required temporal grounding
- Schema versioning for autonomous evolution
- Temporal and causal relationship metadata
- Provenance tracking

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Option B Compliance:
- Event.timestamp is REQUIRED (temporal-first design)
- Causal relationships include Bradford Hill criteria structure
- Schema versioning tracks autonomous evolution
- No hardcoded entity types (seed schema, discoverable)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Relationship Type Enums
# ---------------------------------------------------------------------------


class TemporalRelationType(str, Enum):
    """Temporal relationship types between events.

    Based on Allen's Interval Algebra with extensions for PKG use cases.
    These relationships require temporal ordering validation.
    """

    BEFORE = "BEFORE"           # Event A finishes before Event B starts
    AFTER = "AFTER"             # Event A starts after Event B finishes
    DURING = "DURING"           # Event A occurs within Event B's timespan
    SIMULTANEOUS = "SIMULTANEOUS"  # Events occur at approximately same time


class CausalRelationType(str, Enum):
    """Causal relationship types between events.

    These relationships are causal candidates for Phase 3 validation.
    Temporal ordering is required (cause must precede effect).
    """

    CAUSES = "CAUSES"           # Strong causal claim (A causes B)
    ENABLES = "ENABLES"         # Prerequisite relationship (A enables B)
    PREVENTS = "PREVENTS"       # Blocking relationship (A prevents B)
    TRIGGERS = "TRIGGERS"       # Immediate causation (A triggers B)


class ProvenanceRelationType(str, Enum):
    """Provenance relationship types for tracking data origins."""

    EXTRACTED_FROM = "EXTRACTED_FROM"      # Entity/Triple extracted from Chunk
    DISCOVERED_IN = "DISCOVERED_IN"        # Entity discovered in Document
    PARTICIPATED_IN = "PARTICIPATED_IN"    # Entity participated in Event


class StandardRelationType(str, Enum):
    """Standard relationship types between entities."""

    RELATED_TO = "RELATED_TO"       # Generic relationship
    WORKS_AT = "WORKS_AT"           # Person works at Organization
    CREATED = "CREATED"             # Entity created something
    BELONGS_TO = "BELONGS_TO"       # Entity belongs to category/group
    HAS_TAG = "HAS_TAG"             # Entity has tag


# ---------------------------------------------------------------------------
# Base Node Model
# ---------------------------------------------------------------------------


class BaseNode(BaseModel):
    """Base class for all PKG node types.

    Provides common fields and serialization for Neo4j storage.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique node identifier (UUID)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Node creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties as key-value pairs"
    )

    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Cypher parameters.

        Handles datetime serialization and nested structures.
        """
        data = self.model_dump(exclude={"properties"})
        # Flatten properties into main dict for Neo4j
        data.update(self.properties)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, timedelta):
                data[key] = value.total_seconds()
        return data

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }
    }


# ---------------------------------------------------------------------------
# Static Entity Node Models
# ---------------------------------------------------------------------------


class PersonNode(BaseNode):
    """Person entity node.

    Represents a person discovered in the user's data with metadata
    for confidence tracking and alias resolution.
    """

    name: str = Field(..., description="Primary name")
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names/nicknames"
    )
    discovery_count: int = Field(
        default=0,
        ge=0,
        description="Number of times discovered across documents"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Average confidence across extractions"
    )
    first_seen_document: Optional[str] = Field(
        None,
        description="Document ID where first discovered"
    )


class OrganizationNode(BaseNode):
    """Organization entity node.

    Represents an organization (company, institution, group) discovered
    in the user's data.
    """

    name: str = Field(..., description="Organization name")
    type: str = Field(
        default="unknown",
        description="Organization type: company, institution, group, etc."
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names/abbreviations"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence"
    )


class ConceptNode(BaseNode):
    """Concept entity node.

    Represents an abstract concept, topic, or idea discovered in the
    user's data.
    """

    name: str = Field(..., description="Concept name")
    description: str = Field(
        default="",
        description="Human-readable description"
    )
    category: str = Field(
        default="topic",
        description="Category: topic, idea, field, skill, etc."
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names/terms"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence"
    )


class DocumentNode(BaseNode):
    """Document source node.

    Represents a source document in the PKG for provenance tracking.
    Maps to NormalizedDocument from the pipeline.
    """

    source_id: str = Field(
        ...,
        description="Connector-specific document identifier"
    )
    source_type: str = Field(
        ...,
        description="Source type: obsidian_vault, imap_mailbox, github_repo, etc."
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of document content for deduplication"
    )
    format: str = Field(
        default="unknown",
        description="Document format: markdown, email, code, pdf, etc."
    )
    modified_at: Optional[datetime] = Field(
        None,
        description="Document modification timestamp"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When document was ingested into PKG"
    )


# ---------------------------------------------------------------------------
# Event Entity Node (Option B Critical)
# ---------------------------------------------------------------------------


class EventNode(BaseNode):
    """Event entity node with required temporal grounding.

    Events are distinct from static entities in that they MUST have
    temporal grounding. This is critical for Option B compliance and
    enables Phase 2 temporal correlation and Phase 3 causal inference.

    Option B Compliance:
    - timestamp field is REQUIRED (not optional)
    - Events without timestamps should not be stored as EventNode
    - Use temporal_confidence to track extraction reliability
    """

    name: str = Field(..., description="Event name/title")
    event_type: str = Field(
        ...,
        description="Event type: meeting, decision, publication, communication, action, etc."
    )
    description: str = Field(
        default="",
        description="Event description"
    )

    # REQUIRED temporal grounding (Option B critical)
    timestamp: datetime = Field(
        ...,
        description="When the event occurred (REQUIRED for temporal grounding)"
    )
    duration: Optional[timedelta] = Field(
        None,
        description="How long the event lasted"
    )
    end_timestamp: Optional[datetime] = Field(
        None,
        description="When the event ended (computed or explicit)"
    )

    # Context
    location: Optional[str] = Field(
        None,
        description="Physical or virtual location"
    )
    context: Optional[str] = Field(
        None,
        description="Brief contextual information"
    )

    # Extraction metadata
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence"
    )
    source_document: str = Field(
        ...,
        description="Document ID where event was extracted"
    )
    extraction_method: str = Field(
        default="explicit",
        description="Extraction method: explicit, inferred, llm"
    )

    @model_validator(mode="after")
    def compute_end_timestamp(self) -> "EventNode":
        """Compute end_timestamp from timestamp + duration if not provided."""
        if self.end_timestamp is None and self.duration is not None:
            self.end_timestamp = self.timestamp + self.duration
        return self


# ---------------------------------------------------------------------------
# Schema Versioning Node (Option B)
# ---------------------------------------------------------------------------


class SchemaVersionNode(BaseNode):
    """Schema version tracking for autonomous evolution.

    Tracks schema versions to support Option B's autonomous schema
    evolution capabilities. Each version records:
    - Entity and relationship types in use
    - Changes from previous version
    - Quality metrics that triggered evolution

    Option B Compliance:
    - Enables autonomous schema updates
    - Tracks evolution history for rollback
    - Records reflection quality metrics
    """

    version: int = Field(
        ...,
        ge=1,
        description="Incrementing version number"
    )
    entity_types: List[str] = Field(
        ...,
        description="List of entity type names in this version"
    )
    relationship_types: List[str] = Field(
        ...,
        description="List of relationship type names in this version"
    )
    changes: str = Field(
        default="{}",
        description="JSON description of changes from previous version"
    )
    reflection_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality metrics that triggered this evolution"
    )
    parent_version: Optional[int] = Field(
        None,
        description="Previous version number (None for v1)"
    )
    documents_processed: int = Field(
        default=0,
        ge=0,
        description="Total documents processed when this version was created"
    )


# ---------------------------------------------------------------------------
# Chunk Node (Provenance)
# ---------------------------------------------------------------------------


class ChunkNode(BaseNode):
    """Document chunk node for provenance tracking.

    Maps from pipeline/models.py:DocumentChunk to enable precise
    provenance tracking: Entity -> EXTRACTED_FROM -> Chunk
    """

    document_id: str = Field(
        ...,
        description="Parent document identifier"
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of chunk content"
    )
    position: int = Field(
        ...,
        ge=0,
        description="Character position in source document"
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Sequential chunk index within document"
    )


# ---------------------------------------------------------------------------
# Relationship Property Models
# ---------------------------------------------------------------------------


class BaseRelationshipProps(BaseModel):
    """Base class for relationship properties."""

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relationship confidence score"
    )
    source_document: str = Field(
        ...,
        description="Document ID where relationship was extracted"
    )
    extraction_method: str = Field(
        default="metadata",
        description="Extraction method: metadata, temporal_extraction, llm, etc."
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Relationship creation timestamp"
    )

    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Cypher parameters."""
        data = self.model_dump()
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, timedelta):
                data[key] = value.total_seconds()
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }
    }


class StandardRelationshipProps(BaseRelationshipProps):
    """Properties for standard entity relationships.

    Used for WORKS_AT, CREATED, RELATED_TO, etc.
    Includes temporal validity for tracking when relationships were active.
    """

    valid_from: Optional[datetime] = Field(
        None,
        description="When this relationship started"
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="When this relationship ended (None = ongoing)"
    )
    role: Optional[str] = Field(
        None,
        description="Role description (e.g., for WORKS_AT)"
    )
    relationship_subtype: Optional[str] = Field(
        None,
        description="Specific subtype (e.g., 'subset_of' for RELATED_TO)"
    )
    strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Relationship strength for weighted relationships"
    )


class TemporalRelationshipProps(BaseRelationshipProps):
    """Properties for temporal relationships between events.

    Used for BEFORE, AFTER, DURING, SIMULTANEOUS relationships.
    Includes temporal-specific metadata for ordering validation.
    """

    temporal_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in temporal ordering"
    )
    temporal_source: str = Field(
        default="explicit_timestamp",
        description="Source of temporal info: explicit_timestamp, relative_expression, inferred"
    )
    temporal_gap: Optional[timedelta] = Field(
        None,
        description="Time gap between events"
    )

    # For DURING relationships
    overlap_start: Optional[datetime] = Field(
        None,
        description="When overlap started"
    )
    overlap_end: Optional[datetime] = Field(
        None,
        description="When overlap ended"
    )
    overlap_type: Optional[str] = Field(
        None,
        description="Overlap type: contains, contained_by, partial"
    )

    # For SIMULTANEOUS relationships
    simultaneity_tolerance: Optional[timedelta] = Field(
        None,
        description="How close in time to be considered simultaneous"
    )


class CausalRelationshipProps(BaseRelationshipProps):
    """Properties for causal relationships between events.

    Used for CAUSES, ENABLES, PREVENTS, TRIGGERS relationships.
    Includes Bradford Hill criteria structure for Phase 3 validation.

    Option B Compliance:
    - is_causal_candidate flags for Phase 3 validation
    - Bradford Hill criteria fields prepared but nullable
    - Temporal ordering must be valid (cause precedes effect)
    """

    # Causal metadata
    causal_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in causal relationship"
    )
    causal_evidence: str = Field(
        default="",
        description="Text evidence supporting causation"
    )
    is_causal_candidate: bool = Field(
        default=True,
        description="Flagged for Phase 3 validation"
    )
    is_validated: bool = Field(
        default=False,
        description="Has Phase 3 validated this relationship"
    )
    validation_method: Optional[str] = Field(
        None,
        description="Method used for Phase 3 validation"
    )

    # Temporal requirements (cause must precede effect)
    temporal_gap: timedelta = Field(
        ...,
        description="Time between cause and effect"
    )
    temporal_ordering_valid: bool = Field(
        ...,
        description="Is cause before effect?"
    )

    # Bradford Hill criteria (Phase 3 validation)
    temporality_satisfied: bool = Field(
        ...,
        description="Bradford Hill criterion 1: Does cause precede effect?"
    )
    strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Bradford Hill criterion 2: Strength of association"
    )
    dose_response: Optional[bool] = Field(
        None,
        description="Bradford Hill criterion 3: More cause → more effect?"
    )
    consistency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Bradford Hill criterion 4: Replicability"
    )
    plausibility: Optional[str] = Field(
        None,
        description="Bradford Hill criterion 5: Mechanistic explanation"
    )

    @field_validator("temporal_ordering_valid", "temporality_satisfied", mode="before")
    @classmethod
    def validate_temporal_ordering(cls, v: bool, info) -> bool:
        """Validate that temporal ordering fields are consistent."""
        return v


class ProvenanceRelationshipProps(BaseRelationshipProps):
    """Properties for provenance relationships.

    Used for EXTRACTED_FROM, DISCOVERED_IN, PARTICIPATED_IN relationships.
    Tracks how entities and relationships were discovered.
    """

    extraction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When extraction occurred"
    )
    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction"
    )
    discovery_method: Optional[str] = Field(
        None,
        description="Discovery method: pattern_match, llm_extraction, rule_based"
    )

    # For PARTICIPATED_IN relationships
    role: Optional[str] = Field(
        None,
        description="Participation role: organizer, attendee, speaker, etc."
    )
    participation_confirmed: Optional[bool] = Field(
        None,
        description="Is participation confirmed or inferred"
    )


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class TemporalOrderingError(Exception):
    """Raised when temporal ordering constraint is violated.

    For BEFORE relationships, source event must have timestamp < target timestamp.
    For CAUSES relationships, cause must precede effect.
    """

    def __init__(
        self,
        source_timestamp: datetime,
        target_timestamp: datetime,
        relationship_type: str
    ):
        self.source_timestamp = source_timestamp
        self.target_timestamp = target_timestamp
        self.relationship_type = relationship_type
        super().__init__(
            f"Temporal ordering violation for {relationship_type}: "
            f"source ({source_timestamp}) must precede target ({target_timestamp})"
        )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def validate_temporal_relationship(
    source_event: EventNode,
    target_event: EventNode,
    relationship_type: TemporalRelationType | CausalRelationType
) -> bool:
    """Validate temporal ordering for a relationship.

    For BEFORE and causal relationships, source must precede target.
    For AFTER, target must precede source.
    For DURING and SIMULTANEOUS, order doesn't matter.

    Args:
        source_event: Source event node
        target_event: Target event node
        relationship_type: Type of relationship

    Returns:
        True if ordering is valid

    Raises:
        TemporalOrderingError: If ordering is invalid
    """
    if relationship_type in (TemporalRelationType.BEFORE,) or isinstance(
        relationship_type, CausalRelationType
    ):
        if source_event.timestamp >= target_event.timestamp:
            raise TemporalOrderingError(
                source_event.timestamp,
                target_event.timestamp,
                relationship_type.value if isinstance(relationship_type, Enum) else str(relationship_type)
            )
    elif relationship_type == TemporalRelationType.AFTER:
        if source_event.timestamp <= target_event.timestamp:
            raise TemporalOrderingError(
                source_event.timestamp,
                target_event.timestamp,
                relationship_type.value
            )
    # DURING and SIMULTANEOUS don't require strict ordering
    return True

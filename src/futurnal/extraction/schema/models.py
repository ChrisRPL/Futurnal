"""
Schema Evolution Data Models

Pydantic models for schema versioning, entity/relationship types,
and discovery tracking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractionPhase(str, Enum):
    """Multi-phase extraction stages following AutoSchemaKG."""
    
    ENTITY_ENTITY = "entity_entity"  # Phase 1: Person → Organization
    ENTITY_EVENT = "entity_event"    # Phase 2: Person → Meeting
    EVENT_EVENT = "event_event"      # Phase 3: Meeting → Decision


class SchemaElement(BaseModel):
    """Base class for schema elements with discovery tracking."""
    
    name: str = Field(..., description="Element name")
    description: str = Field(..., description="Human-readable description")
    examples: List[str] = Field(default_factory=list, description="Example instances")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    discovery_count: int = Field(default=0, description="Number of times discovered")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last discovery timestamp")


class EntityType(SchemaElement):
    """Discovered or seed entity type definition."""
    
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Property name to type mapping"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names for this entity type"
    )
    parent_type: Optional[str] = Field(
        None,
        description="Parent type for hierarchical relationships"
    )


class RelationshipType(SchemaElement):
    """Discovered or seed relationship type definition."""
    
    subject_types: List[str] = Field(
        ...,
        description="Valid subject entity types"
    )
    object_types: List[str] = Field(
        ...,
        description="Valid object entity types"
    )
    temporal: bool = Field(
        default=False,
        description="Whether this relationship has temporal grounding"
    )
    causal: bool = Field(
        default=False,
        description="Whether this is a causal relationship candidate"
    )
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Relationship property name to type mapping"
    )


class SchemaVersion(BaseModel):
    """Versioned schema snapshot with quality metrics."""
    
    version: int = Field(..., ge=1, description="Schema version number")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Version creation timestamp"
    )
    entity_types: Dict[str, EntityType] = Field(
        ...,
        description="Entity type name to definition mapping"
    )
    relationship_types: Dict[str, RelationshipType] = Field(
        ...,
        description="Relationship type name to definition mapping"
    )
    current_phase: ExtractionPhase = Field(
        ...,
        description="Current extraction phase"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality assessment metrics"
    )
    changes_from_previous: Optional[str] = Field(
        None,
        description="Changelog describing changes from previous version"
    )


class SchemaDiscovery(BaseModel):
    """Discovered schema element pending validation."""
    
    element_type: str = Field(
        ...,
        description="Type of element: 'entity' or 'relationship'"
    )
    name: str = Field(..., description="Proposed element name")
    description: str = Field(..., description="Element description")
    examples: List[str] = Field(
        default_factory=list,
        description="Example instances from documents"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Discovery confidence score"
    )
    source_documents: List[str] = Field(
        default_factory=list,
        description="Document IDs where this pattern was observed"
    )


class ExperientialKnowledge(BaseModel):
    """Natural language patterns learned from experience."""
    
    pattern_id: str = Field(..., description="Unique identifier for this pattern")
    description: str = Field(..., description="Description of the better approach")
    context: str = Field(..., description="Context/reasoning for why this is better")
    success_count: int = Field(default=0, description="Number of times this pattern succeeded")
    failure_count: int = Field(default=0, description="Number of times this pattern failed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this pattern")
    examples: List[str] = Field(default_factory=list, description="Examples of this pattern")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class SemanticAdvantage(BaseModel):
    """Advantage signal from LLM introspection."""
    
    better_approach: str = Field(..., description="Description of the better approach")
    worse_approach: str = Field(..., description="Description of the worse approach")
    reasoning: str = Field(..., description="Reasoning for the advantage")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this advantage")

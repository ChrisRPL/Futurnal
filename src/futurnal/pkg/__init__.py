"""PKG (Personal Knowledge Graph) Storage Layer.

This module provides the graph database storage layer for Futurnal's PKG,
supporting Option B requirements including:
- Temporal metadata on all entities and relationships
- Causal relationship structure with Bradford Hill criteria
- Autonomous schema evolution tracking
- Comprehensive provenance tracking

Module Structure:
- schema/: Schema definitions, constraints, and migration support
- repository/: Data access layer (Module 03)
- temporal/: Temporal query support (Module 04)

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/

Option B Compliance:
- Event nodes require temporal grounding (timestamp)
- Schema versioning for autonomous evolution
- Causal candidates flagged for Phase 3 validation
- No hardcoded entity types (seed schema only)
"""

from futurnal.pkg.schema import (
    # Node models
    PersonNode,
    OrganizationNode,
    ConceptNode,
    DocumentNode,
    EventNode,
    SchemaVersionNode,
    ChunkNode,
    # Relationship models
    TemporalRelationshipProps,
    CausalRelationshipProps,
    ProvenanceRelationshipProps,
    StandardRelationshipProps,
    # Enums
    TemporalRelationType,
    CausalRelationType,
    ProvenanceRelationType,
    StandardRelationType,
    # Constraints
    init_schema,
    validate_schema,
    # Migration
    SchemaVersionManager,
)

__all__ = [
    # Node models
    "PersonNode",
    "OrganizationNode",
    "ConceptNode",
    "DocumentNode",
    "EventNode",
    "SchemaVersionNode",
    "ChunkNode",
    # Relationship models
    "TemporalRelationshipProps",
    "CausalRelationshipProps",
    "ProvenanceRelationshipProps",
    "StandardRelationshipProps",
    # Enums
    "TemporalRelationType",
    "CausalRelationType",
    "ProvenanceRelationType",
    "StandardRelationType",
    # Constraints
    "init_schema",
    "validate_schema",
    # Migration
    "SchemaVersionManager",
]

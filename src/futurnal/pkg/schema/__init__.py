"""PKG Schema Module.

Defines the canonical graph schema for the Personal Knowledge Graph supporting
Option B requirements including temporal metadata, causal relationships,
schema evolution, and provenance tracking.

This schema serves as the contract between:
- Extraction pipeline (upstream)
- Vector embeddings (downstream)
- Hybrid search (downstream)
- Phase 2/3 analytics (downstream)

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md
"""

from futurnal.pkg.schema.models import (
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
)

from futurnal.pkg.schema.constraints import (
    CONSTRAINT_DEFINITIONS,
    INDEX_DEFINITIONS,
    init_schema,
    validate_schema,
)

from futurnal.pkg.schema.migration import (
    SchemaVersionManager,
    MigrationStep,
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
    "CONSTRAINT_DEFINITIONS",
    "INDEX_DEFINITIONS",
    "init_schema",
    "validate_schema",
    # Migration
    "SchemaVersionManager",
    "MigrationStep",
]

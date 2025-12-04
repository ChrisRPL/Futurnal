"""PKG (Personal Knowledge Graph) Storage Layer.

This module provides the graph database storage layer for Futurnal's PKG,
supporting Option B requirements including:
- Temporal metadata on all entities and relationships
- Causal relationship structure with Bradford Hill criteria
- Autonomous schema evolution tracking
- Comprehensive provenance tracking

Module Structure:
- schema/: Schema definitions, constraints, and migration support (Module 01)
- database/: Database setup, configuration, and lifecycle (Module 02)
- repository/: Data access layer (Module 03)
- queries/: Temporal query support (Module 04)
- sync/: PKG ↔ Vector store synchronization (Module 05)
- validation/: Production readiness validation (Module 05)

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/

Option B Compliance:
- Event nodes require temporal grounding (timestamp)
- Schema versioning for autonomous evolution
- Causal candidates flagged for Phase 3 validation
- No hardcoded entity types (seed schema only)
"""

from futurnal.pkg.database import (
    # Configuration
    PKGDatabaseConfig,
    # Manager
    PKGDatabaseManager,
    # Backup
    PKGBackupManager,
    # Exceptions
    PKGDatabaseError,
    PKGConnectionError,
    PKGBackupError,
    PKGRestoreError,
    PKGSchemaInitializationError,
    PKGHealthCheckError,
)
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
from futurnal.pkg.queries import (
    # Main service (Module 04)
    TemporalGraphQueries,
    # Result models
    CausalPath,
    CausalChainResult,
    TemporalNeighborhood,
    TemporalQueryResult,
    # Exceptions
    TemporalQueryError,
    InvalidTimeRangeError,
    EventNotFoundError,
    CausalChainDepthError,
)
from futurnal.pkg.sync import (
    # Sync events (Module 05)
    SyncEvent,
    SyncEventCapture,
    SyncEventType,
    SyncStatus,
)
from futurnal.pkg.validation import (
    # Production readiness (Module 05)
    ProductionReadinessValidator,
    ProductionReadinessReport,
    ValidationResult,
)

__all__ = [
    # Database (Module 02)
    "PKGDatabaseConfig",
    "PKGDatabaseManager",
    "PKGBackupManager",
    "PKGDatabaseError",
    "PKGConnectionError",
    "PKGBackupError",
    "PKGRestoreError",
    "PKGSchemaInitializationError",
    "PKGHealthCheckError",
    # Node models (Module 01)
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
    # Temporal Queries (Module 04)
    "TemporalGraphQueries",
    "CausalPath",
    "CausalChainResult",
    "TemporalNeighborhood",
    "TemporalQueryResult",
    "TemporalQueryError",
    "InvalidTimeRangeError",
    "EventNotFoundError",
    "CausalChainDepthError",
    # Sync Events (Module 05)
    "SyncEvent",
    "SyncEventCapture",
    "SyncEventType",
    "SyncStatus",
    # Production Readiness (Module 05)
    "ProductionReadinessValidator",
    "ProductionReadinessReport",
    "ValidationResult",
]

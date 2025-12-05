"""Embedding Consistency Validator for PKG ↔ Vector Store Synchronization.

Validates and repairs consistency between PKG graph storage and the
Vector Embedding Store, ensuring embeddings remain synchronized.

The validator detects:
1. Missing embeddings: Entities in PKG without embeddings
2. Orphaned embeddings: Embeddings without corresponding PKG entities
3. Outdated embeddings: Embeddings with old schema versions

Success Metrics:
- >99.9% consistency between PKG and embeddings
- Automatic repair of detected inconsistencies
- Zero data loss during sync operations

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/04-pkg-synchronization.md

Example Usage:
    >>> from futurnal.embeddings.consistency_validator import EmbeddingConsistencyValidator
    >>>
    >>> # Create validator
    >>> validator = EmbeddingConsistencyValidator(
    ...     entity_repo=entity_repo,
    ...     embedding_store=embedding_store,
    ...     sync_handler=sync_handler,
    ... )
    >>>
    >>> # Run validation
    >>> report = validator.validate_consistency()
    >>> print(f"Consistency: {report.consistency_ratio:.2%}")
    >>>
    >>> # Repair if needed
    >>> if not report.is_consistent:
    ...     validator.repair_inconsistencies(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from uuid import uuid4

from futurnal.pkg.sync.events import PKGEvent, SyncEventType

if TYPE_CHECKING:
    from futurnal.pkg.repository.entities import EntityRepository
    from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
    from futurnal.embeddings.sync_handler import PKGSyncHandler

logger = logging.getLogger(__name__)


# Consistency threshold (>99.9%)
CONSISTENCY_THRESHOLD = 0.999


@dataclass
class ConsistencyReport:
    """Report of PKG ↔ Embedding consistency validation.

    Attributes:
        missing_embeddings: Entity IDs in PKG but not in embeddings
        orphaned_embeddings: Embedding entity IDs not in PKG
        outdated_embeddings: Embeddings with schema version < current
        total_pkg_entities: Total entities in PKG
        total_embeddings: Total embeddings in store
        validation_timestamp: When validation was performed
    """
    missing_embeddings: List[str] = field(default_factory=list)
    orphaned_embeddings: List[str] = field(default_factory=list)
    outdated_embeddings: List[str] = field(default_factory=list)
    total_pkg_entities: int = 0
    total_embeddings: int = 0
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def consistency_ratio(self) -> float:
        """Calculate consistency ratio (0.0 to 1.0).

        Consistency = (entities with valid embeddings) / (total entities)
        """
        if self.total_pkg_entities == 0:
            return 1.0

        valid_embeddings = (
            self.total_pkg_entities
            - len(self.missing_embeddings)
            - len(self.outdated_embeddings)
        )
        return max(0.0, valid_embeddings / self.total_pkg_entities)

    @property
    def is_consistent(self) -> bool:
        """Check if consistency meets threshold (>99.9%)."""
        return self.consistency_ratio >= CONSISTENCY_THRESHOLD

    @property
    def issues_count(self) -> int:
        """Total number of issues found."""
        return (
            len(self.missing_embeddings)
            + len(self.orphaned_embeddings)
            + len(self.outdated_embeddings)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "missing_embeddings": self.missing_embeddings,
            "orphaned_embeddings": self.orphaned_embeddings,
            "outdated_embeddings": self.outdated_embeddings,
            "total_pkg_entities": self.total_pkg_entities,
            "total_embeddings": self.total_embeddings,
            "consistency_ratio": self.consistency_ratio,
            "is_consistent": self.is_consistent,
            "issues_count": self.issues_count,
            "validation_timestamp": self.validation_timestamp.isoformat(),
        }


@dataclass
class RepairResult:
    """Result of consistency repair operation.

    Attributes:
        missing_created: Number of missing embeddings created
        orphaned_deleted: Number of orphaned embeddings deleted
        outdated_marked: Number of outdated embeddings marked for re-embedding
        errors: List of errors encountered during repair
    """
    missing_created: int = 0
    orphaned_deleted: int = 0
    outdated_marked: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_repaired(self) -> int:
        """Total number of items repaired."""
        return self.missing_created + self.orphaned_deleted + self.outdated_marked

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "missing_created": self.missing_created,
            "orphaned_deleted": self.orphaned_deleted,
            "outdated_marked": self.outdated_marked,
            "total_repaired": self.total_repaired,
            "has_errors": self.has_errors,
            "errors": self.errors,
        }


class EmbeddingConsistencyValidator:
    """Validates and repairs PKG ↔ Embedding consistency.

    Performs comprehensive validation by:
    1. Querying all entity IDs from PKG
    2. Querying all entity IDs from embedding store
    3. Computing set differences to find missing/orphaned
    4. Checking schema versions for outdated embeddings

    Thread Safety:
        This class is NOT thread-safe. Do not run concurrent validations.

    Attributes:
        entity_repo: EntityRepository for PKG access
        embedding_store: SchemaVersionedEmbeddingStore for embedding access
        sync_handler: PKGSyncHandler for creating missing embeddings

    Example:
        >>> validator = EmbeddingConsistencyValidator(
        ...     entity_repo=entity_repo,
        ...     embedding_store=embedding_store,
        ...     sync_handler=sync_handler,
        ... )
        >>> report = validator.validate_consistency()
    """

    def __init__(
        self,
        entity_repo: "EntityRepository",
        embedding_store: "SchemaVersionedEmbeddingStore",
        sync_handler: "PKGSyncHandler",
    ) -> None:
        """Initialize the consistency validator.

        Args:
            entity_repo: EntityRepository for PKG entity access
            embedding_store: SchemaVersionedEmbeddingStore for embedding access
            sync_handler: PKGSyncHandler for creating missing embeddings
        """
        self._entity_repo = entity_repo
        self._store = embedding_store
        self._sync_handler = sync_handler

        logger.info("Initialized EmbeddingConsistencyValidator")

    def validate_consistency(
        self,
        entity_types: Optional[List[str]] = None,
    ) -> ConsistencyReport:
        """Run comprehensive consistency validation.

        Args:
            entity_types: Optional list of entity types to check.
                         If None, checks all types.

        Returns:
            ConsistencyReport with validation results
        """
        logger.info("Starting consistency validation")
        start_time = datetime.utcnow()

        # Get all PKG entity IDs
        pkg_entity_ids = self._get_all_pkg_entity_ids(entity_types)

        # Get all embedding entity IDs
        embedding_entity_ids = self._get_all_embedding_entity_ids()

        # Calculate differences
        missing = pkg_entity_ids - embedding_entity_ids
        orphaned = embedding_entity_ids - pkg_entity_ids

        # Check for outdated embeddings
        current_schema = self._store.current_schema_version
        outdated = self._get_outdated_embeddings(current_schema)

        report = ConsistencyReport(
            missing_embeddings=list(missing),
            orphaned_embeddings=list(orphaned),
            outdated_embeddings=outdated,
            total_pkg_entities=len(pkg_entity_ids),
            total_embeddings=len(embedding_entity_ids),
            validation_timestamp=start_time,
        )

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"Consistency validation complete in {duration:.2f}s: "
            f"consistency={report.consistency_ratio:.2%}, "
            f"issues={report.issues_count}"
        )

        return report

    def repair_inconsistencies(
        self,
        report: ConsistencyReport,
        create_missing: bool = True,
        delete_orphaned: bool = True,
        mark_outdated: bool = True,
    ) -> RepairResult:
        """Repair detected inconsistencies.

        Args:
            report: ConsistencyReport from validate_consistency()
            create_missing: Create embeddings for missing entities
            delete_orphaned: Delete orphaned embeddings
            mark_outdated: Mark outdated embeddings for re-embedding

        Returns:
            RepairResult with repair statistics
        """
        logger.info(f"Starting repair of {report.issues_count} inconsistencies")
        result = RepairResult()

        # Create missing embeddings
        if create_missing and report.missing_embeddings:
            result.missing_created = self._create_missing_embeddings(
                report.missing_embeddings,
                result.errors,
            )

        # Delete orphaned embeddings
        if delete_orphaned and report.orphaned_embeddings:
            result.orphaned_deleted = self._delete_orphaned_embeddings(
                report.orphaned_embeddings,
                result.errors,
            )

        # Mark outdated for re-embedding
        if mark_outdated and report.outdated_embeddings:
            result.outdated_marked = self._mark_outdated_embeddings(
                report.outdated_embeddings,
            )

        logger.info(
            f"Repair complete: created={result.missing_created}, "
            f"deleted={result.orphaned_deleted}, "
            f"marked={result.outdated_marked}, "
            f"errors={len(result.errors)}"
        )

        return result

    def _get_all_pkg_entity_ids(
        self,
        entity_types: Optional[List[str]] = None,
    ) -> Set[str]:
        """Get all entity IDs from PKG.

        Args:
            entity_types: Optional entity types to filter

        Returns:
            Set of entity IDs
        """
        entity_ids: Set[str] = set()

        # Default entity types if not specified
        if entity_types is None:
            entity_types = ["Person", "Organization", "Concept", "Event", "Document"]

        for entity_type in entity_types:
            try:
                # Stream entities to handle large datasets
                for entity in self._entity_repo.stream_entities(entity_type):
                    if entity.id:
                        entity_ids.add(entity.id)
            except Exception as e:
                logger.warning(f"Error streaming {entity_type} entities: {e}")

        logger.debug(f"Found {len(entity_ids)} PKG entities")
        return entity_ids

    def _get_all_embedding_entity_ids(self) -> Set[str]:
        """Get all entity IDs from embedding store.

        Returns:
            Set of entity IDs that have embeddings
        """
        entity_ids: Set[str] = set()

        # Query both collections
        for collection in [
            self._store._writer._events_collection,
            self._store._writer._entities_collection,
        ]:
            try:
                # Get all embeddings (ChromaDB limitation: no efficient streaming)
                results = collection.get(
                    include=["metadatas"],
                    limit=100000,  # Adjust based on expected size
                )

                for metadata in results.get("metadatas", []):
                    entity_id = metadata.get("entity_id")
                    if entity_id:
                        entity_ids.add(entity_id)

            except Exception as e:
                logger.warning(f"Error getting embedding entity IDs: {e}")

        logger.debug(f"Found {len(entity_ids)} embedding entity IDs")
        return entity_ids

    def _get_outdated_embeddings(self, current_schema: int) -> List[str]:
        """Get entity IDs with outdated schema versions.

        Args:
            current_schema: Current PKG schema version

        Returns:
            List of entity IDs with outdated embeddings
        """
        outdated: List[str] = []

        for collection in [
            self._store._writer._events_collection,
            self._store._writer._entities_collection,
        ]:
            try:
                # Query embeddings with old schema version
                results = collection.get(
                    where={"schema_version": {"$lt": current_schema}},
                    include=["metadatas"],
                    limit=10000,
                )

                for metadata in results.get("metadatas", []):
                    entity_id = metadata.get("entity_id")
                    if entity_id and entity_id not in outdated:
                        outdated.append(entity_id)

            except Exception as e:
                logger.warning(f"Error getting outdated embeddings: {e}")

        logger.debug(f"Found {len(outdated)} outdated embeddings")
        return outdated

    def _create_missing_embeddings(
        self,
        entity_ids: List[str],
        errors: List[Dict[str, Any]],
    ) -> int:
        """Create embeddings for missing entities.

        Args:
            entity_ids: Entity IDs needing embeddings
            errors: List to append errors to

        Returns:
            Number of embeddings created
        """
        created = 0

        for entity_id in entity_ids:
            try:
                # Fetch entity from PKG
                entity = self._entity_repo.get_entity(entity_id)
                if entity is None:
                    # Entity was deleted between validation and repair
                    continue

                # Create PKGEvent for sync handler
                entity_type = type(entity).__name__
                if entity_type.endswith("Node"):
                    entity_type = entity_type[:-4]

                event = PKGEvent(
                    event_id=f"repair_{entity_id}_{uuid4().hex[:8]}",
                    event_type=SyncEventType.ENTITY_CREATED,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    timestamp=datetime.utcnow(),
                    new_data=entity.to_cypher_properties(),
                    schema_version=self._store.current_schema_version,
                )

                # Use sync handler to create embedding
                if self._sync_handler.handle_event(event):
                    created += 1
                else:
                    errors.append({
                        "entity_id": entity_id,
                        "operation": "create_missing",
                        "error": "Sync handler returned False",
                    })

            except Exception as e:
                logger.error(f"Failed to create embedding for {entity_id}: {e}")
                errors.append({
                    "entity_id": entity_id,
                    "operation": "create_missing",
                    "error": str(e),
                })

        return created

    def _delete_orphaned_embeddings(
        self,
        entity_ids: List[str],
        errors: List[Dict[str, Any]],
    ) -> int:
        """Delete orphaned embeddings.

        Args:
            entity_ids: Entity IDs to delete embeddings for
            errors: List to append errors to

        Returns:
            Number of embeddings deleted
        """
        deleted = 0

        for entity_id in entity_ids:
            try:
                count = self._store.delete_embedding_by_entity_id(entity_id)
                if count > 0:
                    deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete orphaned embedding {entity_id}: {e}")
                errors.append({
                    "entity_id": entity_id,
                    "operation": "delete_orphaned",
                    "error": str(e),
                })

        return deleted

    def _mark_outdated_embeddings(self, entity_ids: List[str]) -> int:
        """Mark outdated embeddings for re-embedding.

        Args:
            entity_ids: Entity IDs to mark

        Returns:
            Number of embeddings marked
        """
        return self._store.mark_for_reembedding(
            entity_ids=entity_ids,
            reason="schema_version_outdated",
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get quick health status without full validation.

        Returns:
            Dict with basic health metrics
        """
        embedding_count = self._store.get_embedding_count()

        return {
            "embedding_store": {
                "events_count": embedding_count["events"],
                "entities_count": embedding_count["entities"],
                "total_count": embedding_count["total"],
            },
            "current_schema_version": self._store.current_schema_version,
            "status": "healthy",
        }

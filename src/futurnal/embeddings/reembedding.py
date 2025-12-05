"""Re-embedding Service for schema evolution support.

Monitors schema changes and triggers re-embedding when needed.
Provides batch processing capabilities for efficient re-embedding
of large numbers of entities when PKG schema evolves.

Option B Compliance:
- Uses MultiModelEmbeddingService (frozen Ghost models)
- Fetches entities from PKG for re-embedding
- Batch processing for efficiency
- Progress tracking and telemetry

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from futurnal.embeddings.exceptions import ReembeddingError
from futurnal.embeddings.models import TemporalEmbeddingContext

if TYPE_CHECKING:
    from neo4j import Driver
    from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
    from futurnal.embeddings.service import MultiModelEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SchemaChangeDetection:
    """Result of schema change detection between two versions.

    Contains details about what changed between schema versions and
    whether re-embedding is required.

    Attributes:
        old_version: Source schema version number
        new_version: Target schema version number
        new_entity_types: Entity types added in new version
        removed_entity_types: Entity types removed from new version
        new_relationship_types: Relationship types added
        removed_relationship_types: Relationship types removed
        requires_reembedding: Whether changes require re-embedding
    """

    old_version: int
    new_version: int
    new_entity_types: List[str] = field(default_factory=list)
    removed_entity_types: List[str] = field(default_factory=list)
    new_relationship_types: List[str] = field(default_factory=list)
    removed_relationship_types: List[str] = field(default_factory=list)
    requires_reembedding: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "new_entity_types": self.new_entity_types,
            "removed_entity_types": self.removed_entity_types,
            "new_relationship_types": self.new_relationship_types,
            "removed_relationship_types": self.removed_relationship_types,
            "requires_reembedding": self.requires_reembedding,
        }

    @property
    def has_entity_changes(self) -> bool:
        """Check if entity types changed."""
        return bool(self.new_entity_types or self.removed_entity_types)

    @property
    def has_relationship_changes(self) -> bool:
        """Check if relationship types changed."""
        return bool(self.new_relationship_types or self.removed_relationship_types)

    def __str__(self) -> str:
        """Human-readable summary of changes."""
        parts = [f"Schema v{self.old_version} -> v{self.new_version}"]
        if self.new_entity_types:
            parts.append(f"+entities: {self.new_entity_types}")
        if self.removed_entity_types:
            parts.append(f"-entities: {self.removed_entity_types}")
        if self.new_relationship_types:
            parts.append(f"+relationships: {self.new_relationship_types}")
        if self.removed_relationship_types:
            parts.append(f"-relationships: {self.removed_relationship_types}")
        parts.append(f"requires_reembedding={self.requires_reembedding}")
        return ", ".join(parts)


@dataclass
class ReembeddingProgress:
    """Tracks re-embedding batch operation progress.

    Provides real-time progress information and success metrics
    for batch re-embedding operations.

    Attributes:
        total: Total number of embeddings to process
        processed: Number of embeddings processed so far
        succeeded: Number successfully re-embedded
        failed: Number that failed
        started_at: When processing started
        completed_at: When processing completed (None if in progress)
    """

    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if all items have been processed."""
        return self.processed >= self.total

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)."""
        if self.processed == 0:
            return 0.0
        return self.succeeded / self.processed

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def embeddings_per_second(self) -> Optional[float]:
        """Calculate processing rate."""
        duration = self.duration_seconds
        if duration is None or duration == 0:
            return None
        return self.processed / duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": self.total,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "is_complete": self.is_complete,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "embeddings_per_second": self.embeddings_per_second,
            "error_count": len(self.errors),
        }


class ReembeddingService:
    """Service to re-embed entities when schema evolves.

    Monitors schema changes and triggers re-embedding as needed.
    Uses batch processing for efficiency with progress tracking.

    Example:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))

        service = ReembeddingService(
            store=schema_versioned_store,
            embedding_service=multi_model_service,
            neo4j_driver=driver,
        )

        # Detect changes between schema versions
        changes = service.detect_schema_changes(old_version=1, new_version=2)

        if changes.requires_reembedding:
            # Trigger re-embedding with progress tracking
            progress = service.trigger_reembedding(schema_version=1)
            print(f"Re-embedded {progress.succeeded}/{progress.total}")

    Option B Compliance:
    - Uses MultiModelEmbeddingService (frozen models)
    - Fetches entities from PKG for re-embedding
    - Batch processing with configurable size
    - Progress tracking and error handling
    """

    def __init__(
        self,
        store: "SchemaVersionedEmbeddingStore",
        embedding_service: "MultiModelEmbeddingService",
        neo4j_driver: "Driver",
        neo4j_database: Optional[str] = None,
    ) -> None:
        """Initialize re-embedding service.

        Args:
            store: Schema-versioned embedding store
            embedding_service: Multi-model embedding service
            neo4j_driver: Neo4j driver for PKG access
            neo4j_database: Optional Neo4j database name
        """
        self._store = store
        self._embedding_service = embedding_service
        self._driver = neo4j_driver
        self._database = neo4j_database

        # Import here to avoid circular dependency
        from futurnal.pkg.schema.migration import SchemaVersionManager

        self._schema_manager = SchemaVersionManager(
            driver=neo4j_driver,
            database=neo4j_database,
        )

        logger.info("Initialized ReembeddingService")

    def detect_schema_changes(
        self,
        old_version: int,
        new_version: int,
    ) -> SchemaChangeDetection:
        """Detect changes between schema versions.

        Compares entity types and relationship types between two
        schema versions to determine what changed.

        Args:
            old_version: Source version number
            new_version: Target version number

        Returns:
            SchemaChangeDetection with detailed change information

        Note:
            requires_reembedding is True when new entity types are added,
            as existing documents may contain entities of the new types
            that were not previously extracted.
        """
        old_schema = self._schema_manager.get_version(old_version)
        new_schema = self._schema_manager.get_version(new_version)

        if old_schema is None or new_schema is None:
            logger.warning(
                f"Could not find schema versions: old={old_version}, new={new_version}"
            )
            return SchemaChangeDetection(
                old_version=old_version,
                new_version=new_version,
                requires_reembedding=False,
            )

        # Import helper from migration module
        from futurnal.pkg.schema.migration import get_schema_diff

        diff = get_schema_diff(old_schema, new_schema)

        # Determine if re-embedding is needed:
        # - New entity types require re-embedding (existing docs may have new entities)
        # - Removed types don't require re-embedding (old embeddings still valid)
        requires_reembedding = len(diff["added_entity_types"]) > 0

        result = SchemaChangeDetection(
            old_version=old_version,
            new_version=new_version,
            new_entity_types=diff["added_entity_types"],
            removed_entity_types=diff["removed_entity_types"],
            new_relationship_types=diff["added_relationship_types"],
            removed_relationship_types=diff["removed_relationship_types"],
            requires_reembedding=requires_reembedding,
        )

        logger.info(f"Schema change detection: {result}")
        return result

    def trigger_reembedding(
        self,
        schema_version: Optional[int] = None,
        entity_ids: Optional[List[str]] = None,
        batch_size: int = 100,
        max_total: Optional[int] = None,
    ) -> ReembeddingProgress:
        """Trigger re-embedding for entities.

        Marks embeddings for re-embedding and processes them in batches.
        Can re-embed by schema version (all embeddings from that version)
        or by specific entity IDs.

        Args:
            schema_version: Re-embed all from this schema version
            entity_ids: Specific entity IDs to re-embed
            batch_size: Batch size for processing (default 100)
            max_total: Maximum total embeddings to process (None = all)

        Returns:
            ReembeddingProgress with results and statistics

        Raises:
            ReembeddingError: If re-embedding fails fatally
        """
        progress = ReembeddingProgress()
        progress.started_at = datetime.utcnow()

        try:
            # Mark for re-embedding
            marked_count = self._store.mark_for_reembedding(
                entity_ids=entity_ids,
                schema_version=schema_version,
                reason="schema_evolution",
            )

            logger.info(f"Marked {marked_count} embeddings for re-embedding")

            # Determine limit
            limit = max_total or (batch_size * 100)  # Default max 10000

            # Get embeddings needing re-embedding
            to_reembed = self._store.get_embeddings_needing_reembedding(limit=limit)
            progress.total = len(to_reembed)

            if progress.total == 0:
                logger.info("No embeddings to re-embed")
                progress.completed_at = datetime.utcnow()
                return progress

            logger.info(f"Starting re-embedding of {progress.total} embeddings")

            # Process in batches
            for i in range(0, len(to_reembed), batch_size):
                batch = to_reembed[i : i + batch_size]
                self._process_reembedding_batch(batch, progress)

                # Log progress
                if progress.processed % (batch_size * 5) == 0:
                    logger.info(
                        f"Re-embedding progress: {progress.processed}/{progress.total} "
                        f"({progress.success_rate:.1%} success)"
                    )

            progress.completed_at = datetime.utcnow()

            logger.info(
                f"Re-embedding complete: {progress.succeeded}/{progress.total} succeeded, "
                f"{progress.failed} failed in {progress.duration_seconds:.1f}s"
            )

            return progress

        except Exception as e:
            progress.completed_at = datetime.utcnow()
            logger.error(f"Re-embedding failed: {e}")
            raise ReembeddingError(
                f"Re-embedding failed after processing {progress.processed} embeddings: {e}",
                cause=e,
            ) from e

    def _process_reembedding_batch(
        self,
        batch: List[Dict[str, Any]],
        progress: ReembeddingProgress,
    ) -> None:
        """Process a batch of embeddings for re-embedding.

        Args:
            batch: List of embedding metadata dicts
            progress: Progress tracker to update
        """
        # Fetch entities from PKG
        entity_ids = [m["entity_id"] for m in batch]
        entities = self._fetch_entities_from_pkg(entity_ids)

        # Re-embed each entity
        for metadata in batch:
            entity_id = metadata["entity_id"]
            embedding_id = metadata.get("embedding_id")

            if entity_id not in entities:
                logger.warning(f"Entity {entity_id} not found in PKG, skipping")
                progress.failed += 1
                progress.processed += 1
                progress.errors.append({
                    "entity_id": entity_id,
                    "error": "Entity not found in PKG",
                })
                continue

            entity = entities[entity_id]

            try:
                # Generate new embedding using MultiModelEmbeddingService
                result = self._embedding_service.embed(
                    entity_type=entity["type"],
                    content=entity["content"],
                    temporal_context=entity.get("temporal_context"),
                    entity_id=entity_id,
                )

                # Store with updated schema version
                new_embedding_id = self._store.store_embedding(
                    entity_id=entity_id,
                    entity_type=entity["type"],
                    embedding=list(result.embedding),
                    model_id=metadata.get("model_id", "unknown"),
                    model_version=result.model_version,
                    extraction_confidence=metadata.get("extraction_confidence", 1.0),
                    source_document_id=metadata.get("source_document_id", "unknown"),
                    temporal_context=entity.get("temporal_context"),
                )

                # Delete old embedding if different ID
                if embedding_id and embedding_id != new_embedding_id:
                    self._store.delete_embedding(embedding_id)

                progress.succeeded += 1

            except Exception as e:
                logger.error(f"Failed to re-embed {entity_id}: {e}")
                progress.failed += 1
                progress.errors.append({
                    "entity_id": entity_id,
                    "error": str(e),
                })

            progress.processed += 1

    def _fetch_entities_from_pkg(
        self,
        entity_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch entity data from PKG for re-embedding.

        Queries Neo4j to get entity properties needed for embedding.

        Args:
            entity_ids: Entity IDs to fetch

        Returns:
            Dict mapping entity_id to entity data with keys:
            - type: Entity type (label)
            - content: Formatted content for embedding
            - temporal_context: TemporalEmbeddingContext (for Events)
            - properties: Raw node properties
        """
        results: Dict[str, Dict[str, Any]] = {}

        with self._driver.session(database=self._database) as session:
            for entity_id in entity_ids:
                result = session.run(
                    """
                    MATCH (e {id: $entity_id})
                    RETURN e, labels(e) as labels
                    """,
                    entity_id=entity_id,
                )

                record = result.single()
                if record:
                    node = record["e"]
                    labels = record["labels"]

                    # Determine entity type from labels (first non-internal label)
                    entity_type = "Unknown"
                    for label in labels:
                        if not label.startswith("_"):
                            entity_type = label
                            break

                    # Build content for embedding
                    content = self._format_entity_content(dict(node), entity_type)

                    # Extract temporal context for events
                    temporal_context = None
                    if entity_type == "Event" and "timestamp" in node:
                        temporal_context = self._extract_temporal_context(dict(node))

                    results[entity_id] = {
                        "type": entity_type,
                        "content": content,
                        "temporal_context": temporal_context,
                        "properties": dict(node),
                    }

        return results

    def _format_entity_content(
        self,
        properties: Dict[str, Any],
        entity_type: str,
    ) -> str:
        """Format entity properties for embedding.

        Creates a text representation suitable for the embedding model.

        Args:
            properties: Node properties
            entity_type: Entity type

        Returns:
            Formatted content string
        """
        name = properties.get("name", "")
        description = properties.get("description", "")

        if description:
            return f"{name}: {description}"
        return name or str(properties.get("id", ""))

    def _extract_temporal_context(
        self,
        properties: Dict[str, Any],
    ) -> Optional[TemporalEmbeddingContext]:
        """Extract temporal context from Event node properties.

        Args:
            properties: Node properties

        Returns:
            TemporalEmbeddingContext if timestamp present, else None
        """
        timestamp = properties.get("timestamp")
        if timestamp is None:
            return None

        # Parse timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        from datetime import timedelta

        duration = None
        if "duration" in properties:
            duration_val = properties["duration"]
            if isinstance(duration_val, (int, float)):
                duration = timedelta(seconds=duration_val)

        return TemporalEmbeddingContext(
            timestamp=timestamp,
            duration=duration,
            temporal_type=properties.get("temporal_type"),
        )

    def get_reembedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings needing re-embedding.

        Returns:
            Dict with counts by reason and schema version
        """
        pending = self._store.get_embeddings_needing_reembedding(limit=10000)

        by_reason: Dict[str, int] = {}
        by_schema_version: Dict[int, int] = {}

        for metadata in pending:
            reason = metadata.get("reembedding_reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1

            version = metadata.get("schema_version", 0)
            by_schema_version[version] = by_schema_version.get(version, 0) + 1

        return {
            "total_pending": len(pending),
            "by_reason": by_reason,
            "by_schema_version": by_schema_version,
        }

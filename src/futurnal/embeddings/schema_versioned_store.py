"""Schema-Versioned Embedding Store.

Wraps TemporalAwareVectorWriter with PKG schema version tracking,
enabling re-embedding when schema evolves.

Option B Compliance:
- Schema version from PKG (int), not static config (str)
- Schema hash for precise change detection
- Re-embedding triggers on schema evolution
- Temporal-first design preserved (timestamps required for events)

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.exceptions import SchemaVersionError, StorageError
from futurnal.embeddings.integration import TemporalAwareVectorWriter
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingMetadata,
    SimilarityResult,
    TemporalEmbeddingContext,
)

if TYPE_CHECKING:
    from neo4j import Driver
    from futurnal.pkg.schema.migration import SchemaVersionManager
    from futurnal.pkg.schema.models import SchemaVersionNode

logger = logging.getLogger(__name__)


class SchemaVersionedEmbeddingStore:
    """Embedding store with PKG schema version tracking.

    Wraps TemporalAwareVectorWriter to add:
    - Schema version tracking from PKG via SchemaVersionManager
    - Schema hash computation for precise change detection
    - Re-embedding flag management
    - Schema-version-aware queries

    The store fetches the current schema version from the PKG on initialization
    and caches it. Call refresh_schema_cache() after schema evolution to update.

    Example:
        from neo4j import GraphDatabase
        from futurnal.embeddings import EmbeddingServiceConfig

        driver = GraphDatabase.driver(uri, auth=(user, password))
        config = EmbeddingServiceConfig()

        store = SchemaVersionedEmbeddingStore(
            config=config,
            neo4j_driver=driver,
        )

        # Store with schema versioning
        embedding_id = store.store_embedding(
            entity_id="person_123",
            entity_type="Person",
            embedding=result.embedding,
            model_id="instructor-large",
            extraction_confidence=0.95,
            source_document_id="doc_456",
        )

        # Query with schema version filtering
        results = store.query_embeddings(
            query_vector=query_embedding,
            min_schema_version=2,
        )

    Option B Compliance:
    - Uses PKG schema version (int) as source of truth
    - Schema hash enables precise change detection
    - Re-embedding triggers on schema evolution
    - Temporal context preserved for events
    """

    def __init__(
        self,
        config: Optional[EmbeddingServiceConfig] = None,
        neo4j_driver: Optional["Driver"] = None,
        neo4j_database: Optional[str] = None,
        persist_directory: Optional[Path] = None,
    ) -> None:
        """Initialize schema-versioned embedding store.

        Args:
            config: Embedding service configuration
            neo4j_driver: Neo4j driver for PKG access (optional)
            neo4j_database: Optional Neo4j database name
            persist_directory: Override ChromaDB persist directory
        """
        self._config = config or EmbeddingServiceConfig()
        self._neo4j_driver = neo4j_driver
        self._neo4j_database = neo4j_database

        # Initialize wrapped writer
        self._writer = TemporalAwareVectorWriter(
            config=self._config,
            persist_directory=persist_directory,
        )

        # Initialize schema version manager if driver provided
        self._schema_manager: Optional["SchemaVersionManager"] = None
        if neo4j_driver is not None:
            from futurnal.pkg.schema.migration import SchemaVersionManager

            self._schema_manager = SchemaVersionManager(
                driver=neo4j_driver,
                database=neo4j_database,
            )

        # Cache current schema version
        self._current_schema_version: Optional[int] = None
        self._current_schema_hash: Optional[str] = None
        self._current_schema_node: Optional["SchemaVersionNode"] = None

        logger.info("Initialized SchemaVersionedEmbeddingStore")

    # -------------------------------------------------------------------------
    # Schema Version Management
    # -------------------------------------------------------------------------

    def _get_current_schema_version(self) -> int:
        """Get current schema version from PKG.

        Returns cached version if available, otherwise queries PKG.
        If no PKG connection, returns default version 1.

        Returns:
            Current schema version (int)

        Raises:
            SchemaVersionError: If PKG query fails
        """
        if self._current_schema_version is not None:
            return self._current_schema_version

        if self._schema_manager is None:
            logger.warning(
                "No Neo4j driver configured, using default schema version 1"
            )
            self._current_schema_version = 1
            return 1

        try:
            version_node = self._schema_manager.get_current_version()
            if version_node is None:
                logger.warning("No schema version in PKG, using default version 1")
                self._current_schema_version = 1
                return 1

            self._current_schema_version = version_node.version
            self._current_schema_node = version_node
            return self._current_schema_version

        except Exception as e:
            raise SchemaVersionError(
                f"Failed to get current schema version from PKG: {e}"
            ) from e

    def _compute_schema_hash(self) -> str:
        """Compute SHA-256 hash of current schema structure.

        Used to detect schema changes that require re-embedding.
        Hash is computed from sorted entity_types and relationship_types
        for deterministic output.

        Returns:
            SHA-256 hex digest of schema structure
        """
        if self._current_schema_hash is not None:
            return self._current_schema_hash

        if self._schema_manager is None:
            # No PKG connection, return empty hash
            self._current_schema_hash = hashlib.sha256(b"").hexdigest()
            return self._current_schema_hash

        try:
            version_node = self._current_schema_node
            if version_node is None:
                version_node = self._schema_manager.get_current_version()

            if version_node is None:
                self._current_schema_hash = hashlib.sha256(b"").hexdigest()
                return self._current_schema_hash

            # Hash entity types and relationship types (sorted for determinism)
            schema_data = {
                "version": version_node.version,
                "entity_types": sorted(version_node.entity_types),
                "relationship_types": sorted(version_node.relationship_types),
            }
            schema_json = json.dumps(schema_data, sort_keys=True)
            self._current_schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

            return self._current_schema_hash

        except Exception as e:
            logger.warning(f"Failed to compute schema hash: {e}")
            self._current_schema_hash = hashlib.sha256(b"").hexdigest()
            return self._current_schema_hash

    def refresh_schema_cache(self) -> None:
        """Refresh cached schema version and hash.

        Call after schema evolution to update cached values.
        This should be called when PKG schema changes to ensure
        new embeddings use the updated version.
        """
        self._current_schema_version = None
        self._current_schema_hash = None
        self._current_schema_node = None

        # Re-fetch
        version = self._get_current_schema_version()
        schema_hash = self._compute_schema_hash()

        logger.info(
            f"Refreshed schema cache: version={version}, "
            f"hash={schema_hash[:16]}..."
        )

    # -------------------------------------------------------------------------
    # Storage Operations
    # -------------------------------------------------------------------------

    def store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: List[float],
        model_id: str,
        extraction_confidence: float,
        source_document_id: str,
        model_version: Optional[str] = None,
        template_version: Optional[str] = None,
        temporal_context: Optional[TemporalEmbeddingContext] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store embedding with schema version metadata.

        Creates an EmbeddingMetadata record with current PKG schema version
        and stores the embedding in the appropriate ChromaDB collection.

        Args:
            entity_id: PKG entity ID
            entity_type: Entity type (Person, Event, Organization, etc.)
            embedding: Embedding vector
            model_id: Embedding model ID
            extraction_confidence: Confidence from extraction (0.0-1.0)
            source_document_id: Source document ID for provenance
            model_version: Optional model version string
            template_version: Optional TOTAL framework template version
            temporal_context: Optional temporal context (REQUIRED for Events)
            additional_metadata: Optional extra metadata to store

        Returns:
            embedding_id (UUID string)

        Raises:
            StorageError: If storage operation fails
        """
        embedding_id = str(uuid4())

        # Create metadata with schema version tracking
        metadata = EmbeddingMetadata(
            embedding_id=embedding_id,
            entity_id=entity_id,
            entity_type=entity_type,
            model_id=model_id,
            model_version=model_version or f"unknown:{model_id}",
            schema_version=self._get_current_schema_version(),
            schema_hash=self._compute_schema_hash(),
            extraction_template_version=template_version,
            extraction_confidence=extraction_confidence,
            source_document_id=source_document_id,
        )

        # Convert to ChromaDB-compatible format
        chroma_metadata = metadata.to_chromadb_metadata()

        # Add any additional metadata (must be primitive types)
        if additional_metadata:
            for key, value in additional_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_metadata[key] = value

        # Add temporal metadata if provided
        if temporal_context:
            chroma_metadata["timestamp"] = temporal_context.timestamp.isoformat()
            if temporal_context.duration:
                chroma_metadata["duration_seconds"] = (
                    temporal_context.duration.total_seconds()
                )
            if temporal_context.temporal_type:
                chroma_metadata["temporal_type"] = temporal_context.temporal_type

        # Store based on entity type
        try:
            if entity_type == "Event" and temporal_context:
                self._writer._events_collection.upsert(
                    ids=[embedding_id],
                    embeddings=[list(embedding)],
                    metadatas=[chroma_metadata],
                    documents=[entity_id],
                )
            else:
                self._writer._entities_collection.upsert(
                    ids=[embedding_id],
                    embeddings=[list(embedding)],
                    metadatas=[chroma_metadata],
                    documents=[entity_id],
                )

            logger.debug(
                f"Stored embedding {embedding_id} for {entity_type} "
                f"with schema version {metadata.schema_version}"
            )
            return embedding_id

        except Exception as e:
            raise StorageError(
                f"Failed to store embedding for {entity_id}: {e}"
            ) from e

    def query_embeddings(
        self,
        query_vector: List[float],
        top_k: int = 10,
        entity_type: Optional[str] = None,
        min_schema_version: Optional[int] = None,
        exclude_needs_reembedding: bool = False,
    ) -> List[SimilarityResult]:
        """Query embeddings with schema version filtering.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            entity_type: Optional entity type filter
            min_schema_version: Optional minimum schema version filter
            exclude_needs_reembedding: Exclude embeddings flagged for re-embedding

        Returns:
            List of SimilarityResult with metadata including schema version

        Raises:
            StorageError: If query fails
        """
        # Build where filter - ChromaDB requires $and for multiple conditions
        conditions: List[Dict[str, Any]] = []

        if entity_type:
            conditions.append({"entity_type": entity_type})

        if min_schema_version:
            conditions.append({"schema_version": {"$gte": min_schema_version}})

        if exclude_needs_reembedding:
            conditions.append({"needs_reembedding": {"$eq": False}})

        # Build final filter
        if len(conditions) == 0:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        # Query appropriate collection
        try:
            if entity_type == "Event":
                results = self._writer._events_collection.query(
                    query_embeddings=[list(query_vector)],
                    n_results=top_k,
                    where=where_filter,
                    include=["embeddings", "metadatas", "distances"],
                )
            else:
                results = self._writer._entities_collection.query(
                    query_embeddings=[list(query_vector)],
                    n_results=top_k,
                    where=where_filter,
                    include=["embeddings", "metadatas", "distances"],
                )

            return self._parse_query_results(results)

        except Exception as e:
            raise StorageError(f"Failed to query embeddings: {e}") from e

    def encode_query(self, query: str) -> List[float]:
        """Encode a query string into an embedding vector.

        Uses the same model as the collection's embedding function to ensure
        dimension compatibility. This is critical for hybrid search.

        Args:
            query: Natural language query string

        Returns:
            Query embedding vector (384-dim for all-MiniLM-L6-v2)

        Raises:
            StorageError: If encoding fails
        """
        try:
            # Use sentence-transformers directly for consistent 384-dim embeddings
            # This matches the model used by KnowledgeGraphIndexer
            from sentence_transformers import SentenceTransformer

            # Cache the model for reuse
            if not hasattr(self, "_query_encoder"):
                self._query_encoder = SentenceTransformer("all-MiniLM-L6-v2")

            embedding = self._query_encoder.encode(query)
            # Convert np.float32 to Python float for ChromaDB compatibility
            return [float(x) for x in embedding]

        except ImportError:
            raise StorageError(
                "sentence-transformers package not installed. "
                "Please install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise StorageError(f"Failed to encode query: {e}") from e

    def _parse_query_results(
        self,
        results: Dict[str, Any],
    ) -> List[SimilarityResult]:
        """Parse ChromaDB query results into SimilarityResult objects."""
        similarity_results = []

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, emb_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            # Convert L2 distance to similarity score
            similarity = 1.0 / (1.0 + distance)

            metadata = metadatas[i] if i < len(metadatas) else {}

            # Determine entity type for result
            entity_type_str = metadata.get("entity_type", "static_entity")
            try:
                entity_type = EmbeddingEntityType(entity_type_str)
            except ValueError:
                entity_type = EmbeddingEntityType.STATIC_ENTITY

            similarity_results.append(
                SimilarityResult(
                    entity_id=metadata.get("entity_id", emb_id),
                    similarity_score=similarity,
                    entity_type=entity_type,
                    metadata=metadata,
                )
            )

        return similarity_results

    # -------------------------------------------------------------------------
    # Re-embedding Management
    # -------------------------------------------------------------------------

    def mark_for_reembedding(
        self,
        entity_ids: Optional[List[str]] = None,
        schema_version: Optional[int] = None,
        reason: str = "schema_evolution",
    ) -> int:
        """Mark embeddings for re-embedding.

        Can mark specific entities or all embeddings from a schema version.
        Marked embeddings can be retrieved via get_embeddings_needing_reembedding().

        Args:
            entity_ids: Specific entity IDs to mark
            schema_version: Mark all embeddings from this schema version
            reason: Reason for re-embedding (schema_evolution, quality, model_update)

        Returns:
            Number of embeddings marked

        Raises:
            StorageError: If marking operation fails
        """
        marked_count = 0

        if entity_ids:
            for entity_id in entity_ids:
                marked_count += self._mark_entity_for_reembedding(entity_id, reason)

        elif schema_version is not None:
            # Mark all embeddings from schema version
            for collection in [
                self._writer._events_collection,
                self._writer._entities_collection,
            ]:
                try:
                    results = collection.get(
                        where={"schema_version": schema_version},
                        include=["metadatas"],
                    )

                    for emb_id, metadata in zip(
                        results.get("ids", []),
                        results.get("metadatas", []),
                    ):
                        # Update metadata
                        updated_metadata = dict(metadata)
                        updated_metadata["needs_reembedding"] = True
                        updated_metadata["reembedding_reason"] = reason

                        collection.update(
                            ids=[emb_id],
                            metadatas=[updated_metadata],
                        )
                        marked_count += 1

                except Exception as e:
                    logger.warning(f"Error marking embeddings in collection: {e}")

        logger.info(f"Marked {marked_count} embeddings for re-embedding: {reason}")
        return marked_count

    def _mark_entity_for_reembedding(
        self,
        entity_id: str,
        reason: str,
    ) -> int:
        """Mark a single entity's embeddings for re-embedding.

        Args:
            entity_id: Entity ID to mark
            reason: Reason for re-embedding

        Returns:
            Number of embeddings marked (0 or 1+)
        """
        marked = 0

        for collection in [
            self._writer._events_collection,
            self._writer._entities_collection,
        ]:
            try:
                results = collection.get(
                    where={"entity_id": entity_id},
                    include=["metadatas"],
                )

                for emb_id, metadata in zip(
                    results.get("ids", []),
                    results.get("metadatas", []),
                ):
                    updated_metadata = dict(metadata)
                    updated_metadata["needs_reembedding"] = True
                    updated_metadata["reembedding_reason"] = reason

                    collection.update(
                        ids=[emb_id],
                        metadatas=[updated_metadata],
                    )
                    marked += 1

            except Exception:
                pass

        return marked

    def get_embeddings_needing_reembedding(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get embeddings marked for re-embedding.

        Args:
            limit: Maximum number to return

        Returns:
            List of embedding metadata dicts with embedding_id key
        """
        results = []

        for collection in [
            self._writer._events_collection,
            self._writer._entities_collection,
        ]:
            if len(results) >= limit:
                break

            try:
                collection_results = collection.get(
                    where={"needs_reembedding": True},
                    include=["metadatas"],
                    limit=limit - len(results),
                )

                for emb_id, metadata in zip(
                    collection_results.get("ids", []),
                    collection_results.get("metadatas", []),
                ):
                    results.append({
                        "embedding_id": emb_id,
                        **metadata,
                    })

                    if len(results) >= limit:
                        break

            except Exception as e:
                logger.warning(f"Error getting embeddings for re-embedding: {e}")

        return results

    def clear_reembedding_flag(
        self,
        embedding_ids: List[str],
    ) -> int:
        """Clear re-embedding flag after successful re-embedding.

        Args:
            embedding_ids: Embedding IDs to clear flags for

        Returns:
            Number of flags cleared
        """
        cleared = 0

        for collection in [
            self._writer._events_collection,
            self._writer._entities_collection,
        ]:
            try:
                for emb_id in embedding_ids:
                    results = collection.get(
                        ids=[emb_id],
                        include=["metadatas"],
                    )

                    if results.get("ids"):
                        metadata = results["metadatas"][0]
                        updated_metadata = dict(metadata)
                        updated_metadata["needs_reembedding"] = False
                        updated_metadata["reembedding_reason"] = ""
                        updated_metadata["last_validated"] = datetime.utcnow().isoformat()

                        collection.update(
                            ids=[emb_id],
                            metadatas=[updated_metadata],
                        )
                        cleared += 1

            except Exception as e:
                logger.warning(f"Error clearing re-embedding flag: {e}")

        return cleared

    def delete_embedding(
        self,
        embedding_id: str,
    ) -> bool:
        """Delete an embedding by ID.

        Args:
            embedding_id: Embedding ID to delete

        Returns:
            True if deleted, False if not found
        """
        for collection in [
            self._writer._events_collection,
            self._writer._entities_collection,
        ]:
            try:
                collection.delete(ids=[embedding_id])
                logger.debug(f"Deleted embedding: {embedding_id}")
                return True
            except Exception:
                pass

        return False

    def delete_embedding_by_entity_id(
        self,
        entity_id: str,
    ) -> int:
        """Delete all embeddings for an entity by entity_id.

        Used by PKGSyncHandler when an entity is deleted from PKG.

        Args:
            entity_id: PKG entity ID whose embeddings should be deleted

        Returns:
            Number of embeddings deleted
        """
        deleted_count = 0

        for collection in [
            self._writer._events_collection,
            self._writer._entities_collection,
        ]:
            try:
                # Query embeddings for this entity
                results = collection.get(
                    where={"entity_id": entity_id},
                    include=["metadatas"],
                )

                embedding_ids = results.get("ids", [])
                if embedding_ids:
                    collection.delete(ids=embedding_ids)
                    deleted_count += len(embedding_ids)
                    logger.debug(
                        f"Deleted {len(embedding_ids)} embedding(s) for entity: {entity_id}"
                    )

            except Exception as e:
                logger.warning(f"Error deleting embeddings for entity {entity_id}: {e}")

        return deleted_count

    # -------------------------------------------------------------------------
    # Properties and Utilities
    # -------------------------------------------------------------------------

    @property
    def current_schema_version(self) -> int:
        """Current PKG schema version."""
        return self._get_current_schema_version()

    @property
    def current_schema_hash(self) -> str:
        """Current schema hash (SHA-256)."""
        return self._compute_schema_hash()

    @property
    def writer(self) -> TemporalAwareVectorWriter:
        """Access underlying writer for advanced operations."""
        return self._writer

    @property
    def schema_manager(self) -> Optional["SchemaVersionManager"]:
        """Access schema manager if available."""
        return self._schema_manager

    def get_embedding_count(self) -> Dict[str, int]:
        """Get count of embeddings in each collection.

        Returns:
            Dict with 'events', 'entities', 'total' counts
        """
        events_count = self._writer._events_collection.count()
        entities_count = self._writer._entities_collection.count()

        return {
            "events": events_count,
            "entities": entities_count,
            "total": events_count + entities_count,
        }

    def close(self) -> None:
        """Clean up resources."""
        self._writer.close()
        logger.info("Closed SchemaVersionedEmbeddingStore")

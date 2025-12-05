"""Integration with ChromaDB for temporal-aware embedding storage.

Provides:
- TemporalAwareVectorWriter: Extended ChromaDB writer with temporal awareness
- Separate collections for events, entities, and sequences
- Schema version tracking in metadata

Maintains backward compatibility with existing ChromaVectorWriter while
adding temporal embedding capabilities.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.event_sequence import EventSequenceEmbedder
from futurnal.embeddings.exceptions import StorageError
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingResult,
    SimilarityResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.static_entity import StaticEntityEmbedder
from futurnal.embeddings.temporal_event import TemporalEventEmbedder

logger = logging.getLogger(__name__)


class TemporalAwareVectorWriter:
    """Extended vector writer with temporal embedding awareness.

    Manages three separate ChromaDB collections:
    - Events: Temporal event embeddings with timestamp metadata
    - Entities: Static entity embeddings
    - Sequences: Event sequence embeddings for pattern matching

    All embeddings include schema version tracking for re-embedding support.

    Example:
        config = EmbeddingServiceConfig()
        writer = TemporalAwareVectorWriter(config)

        # Embed and store a temporal event
        context = TemporalEmbeddingContext(timestamp=datetime.now())
        result = writer.embed_and_store_event(
            event_id="evt_123",
            event_name="Team Meeting",
            event_description="Quarterly planning",
            temporal_context=context,
        )

        # Search for similar events
        similar = writer.search_similar_events(
            query_embedding=result.embedding,
            top_k=10,
        )
    """

    def __init__(
        self,
        config: Optional[EmbeddingServiceConfig] = None,
        persist_directory: Optional[Path] = None,
    ) -> None:
        """Initialize the temporal-aware vector writer.

        Args:
            config: Embedding service configuration
            persist_directory: Override persist directory from config
        """
        self._config = config or EmbeddingServiceConfig()
        self._persist_dir = persist_directory or self._config.get_persist_path()

        # Initialize model manager and embedders
        self._model_manager = ModelManager(self._config)
        self._temporal_embedder = TemporalEventEmbedder(self._model_manager)
        self._static_embedder = StaticEntityEmbedder(self._model_manager)
        self._sequence_embedder = EventSequenceEmbedder(
            self._model_manager, self._temporal_embedder
        )

        # Initialize ChromaDB collections
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialize ChromaDB collections for each entity type."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=str(self._persist_dir))

            # Create embedding function (used only for text-based queries)
            # Precomputed embeddings are stored directly
            default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Events collection
            self._events_collection = self._client.get_or_create_collection(
                name=self._config.get_collection_name("events"),
                metadata={
                    "description": "Temporal event embeddings",
                    "schema_version": self._config.schema_version,
                    "entity_type": EmbeddingEntityType.TEMPORAL_EVENT.value,
                },
                embedding_function=default_ef,
            )

            # Entities collection
            self._entities_collection = self._client.get_or_create_collection(
                name=self._config.get_collection_name("entities"),
                metadata={
                    "description": "Static entity embeddings",
                    "schema_version": self._config.schema_version,
                    "entity_type": EmbeddingEntityType.STATIC_ENTITY.value,
                },
                embedding_function=default_ef,
            )

            # Sequences collection
            self._sequences_collection = self._client.get_or_create_collection(
                name=self._config.get_collection_name("sequences"),
                metadata={
                    "description": "Event sequence embeddings",
                    "schema_version": self._config.schema_version,
                    "entity_type": "sequence",
                },
                embedding_function=default_ef,
            )

            logger.info(
                f"Initialized ChromaDB collections at {self._persist_dir} "
                f"with schema version {self._config.schema_version}"
            )

        except ImportError as e:
            raise StorageError(
                "chromadb package not installed. "
                "Please install it with: pip install chromadb"
            ) from e
        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB: {e}") from e

    # -------------------------------------------------------------------------
    # Event Operations
    # -------------------------------------------------------------------------

    def embed_and_store_event(
        self,
        event_id: str,
        event_name: str,
        event_description: str,
        temporal_context: TemporalEmbeddingContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Embed a temporal event and store in ChromaDB.

        Args:
            event_id: Unique identifier for the event
            event_name: Name/title of the event
            event_description: Description of the event
            temporal_context: REQUIRED temporal context
            metadata: Additional metadata to store

        Returns:
            EmbeddingResult from embedding generation
        """
        # Generate embedding
        result = self._temporal_embedder.embed(
            event_name=event_name,
            event_description=event_description,
            temporal_context=temporal_context,
        )

        # Store in ChromaDB
        self._store_event_embedding(event_id, result, temporal_context, metadata)

        return result

    def store_event_embedding(
        self,
        event_id: str,
        embedding_result: EmbeddingResult,
        temporal_context: TemporalEmbeddingContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a pre-computed event embedding.

        Args:
            event_id: Unique identifier for the event
            embedding_result: Pre-computed embedding result
            temporal_context: Temporal context for metadata
            metadata: Additional metadata to store
        """
        self._store_event_embedding(event_id, embedding_result, temporal_context, metadata)

    def _store_event_embedding(
        self,
        event_id: str,
        result: EmbeddingResult,
        temporal_context: TemporalEmbeddingContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal method to store event embedding."""
        try:
            # Build metadata
            chroma_metadata = {
                "entity_type": result.entity_type.value,
                "model_version": result.model_version,
                "schema_version": self._config.schema_version,
                "temporal_context_encoded": result.temporal_context_encoded,
                "causal_context_encoded": result.causal_context_encoded,
                "timestamp": temporal_context.timestamp.isoformat(),
                "embedded_at": datetime.utcnow().isoformat(),
            }

            if temporal_context.duration:
                chroma_metadata["duration_seconds"] = temporal_context.duration.total_seconds()

            if temporal_context.temporal_type:
                chroma_metadata["temporal_type"] = temporal_context.temporal_type

            if metadata:
                # Add user metadata (ChromaDB requires string/int/float/bool values)
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value

            # Upsert to collection
            self._events_collection.upsert(
                ids=[event_id],
                embeddings=[list(result.embedding)],
                metadatas=[chroma_metadata],
                documents=[result.metadata.get("event_name", "")],
            )

            logger.debug(f"Stored event embedding: {event_id}")

        except Exception as e:
            raise StorageError(f"Failed to store event embedding {event_id}: {e}") from e

    def search_similar_events(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        timestamp_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SimilarityResult]:
        """Search for similar events.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            timestamp_filter: Optional ChromaDB where clause for timestamp

        Returns:
            List of SimilarityResult
        """
        try:
            results = self._events_collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                where=timestamp_filter,
                include=["embeddings", "metadatas", "distances"],
            )

            return self._parse_search_results(results, EmbeddingEntityType.TEMPORAL_EVENT)

        except Exception as e:
            raise StorageError(f"Failed to search events: {e}") from e

    # -------------------------------------------------------------------------
    # Entity Operations
    # -------------------------------------------------------------------------

    def embed_and_store_entity(
        self,
        entity_id: str,
        entity_type: str,
        entity_name: str,
        entity_description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Embed a static entity and store in ChromaDB.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type (Person, Organization, Concept)
            entity_name: Name of the entity
            entity_description: Optional description
            properties: Entity properties to include in embedding
            metadata: Additional metadata to store

        Returns:
            EmbeddingResult from embedding generation
        """
        # Generate embedding
        result = self._static_embedder.embed(
            entity_type=entity_type,
            entity_name=entity_name,
            entity_description=entity_description,
            properties=properties,
        )

        # Store in ChromaDB
        self._store_entity_embedding(entity_id, result, entity_type, entity_name, metadata)

        return result

    def _store_entity_embedding(
        self,
        entity_id: str,
        result: EmbeddingResult,
        entity_type: str,
        entity_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal method to store entity embedding."""
        try:
            chroma_metadata = {
                "entity_type_category": entity_type,
                "entity_name": entity_name,
                "model_version": result.model_version,
                "schema_version": self._config.schema_version,
                "embedded_at": datetime.utcnow().isoformat(),
            }

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value

            self._entities_collection.upsert(
                ids=[entity_id],
                embeddings=[list(result.embedding)],
                metadatas=[chroma_metadata],
                documents=[entity_name],
            )

            logger.debug(f"Stored entity embedding: {entity_id}")

        except Exception as e:
            raise StorageError(f"Failed to store entity embedding {entity_id}: {e}") from e

    def search_similar_entities(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        entity_type_filter: Optional[str] = None,
    ) -> List[SimilarityResult]:
        """Search for similar entities.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            entity_type_filter: Optional filter by entity type

        Returns:
            List of SimilarityResult
        """
        try:
            where_clause = None
            if entity_type_filter:
                where_clause = {"entity_type_category": entity_type_filter}

            results = self._entities_collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                where=where_clause,
                include=["embeddings", "metadatas", "distances"],
            )

            return self._parse_search_results(results, EmbeddingEntityType.STATIC_ENTITY)

        except Exception as e:
            raise StorageError(f"Failed to search entities: {e}") from e

    # -------------------------------------------------------------------------
    # Sequence Operations
    # -------------------------------------------------------------------------

    def embed_and_store_sequence(
        self,
        sequence_id: str,
        events: List[Any],
        temporal_contexts: List[TemporalEmbeddingContext],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Embed an event sequence and store in ChromaDB.

        Args:
            sequence_id: Unique identifier for the sequence
            events: List of events in temporal order
            temporal_contexts: Corresponding temporal contexts
            metadata: Additional metadata to store

        Returns:
            EmbeddingResult from embedding generation
        """
        result = self._sequence_embedder.embed(events, temporal_contexts)
        self._store_sequence_embedding(sequence_id, result, metadata)
        return result

    def _store_sequence_embedding(
        self,
        sequence_id: str,
        result: EmbeddingResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal method to store sequence embedding."""
        try:
            chroma_metadata = {
                "sequence_length": result.metadata.get("sequence_length", 0),
                "event_pattern": result.metadata.get("event_pattern", ""),
                "model_version": result.model_version,
                "schema_version": self._config.schema_version,
                "embedded_at": datetime.utcnow().isoformat(),
            }

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_metadata[key] = value

            self._sequences_collection.upsert(
                ids=[sequence_id],
                embeddings=[list(result.embedding)],
                metadatas=[chroma_metadata],
                documents=[result.metadata.get("event_pattern", "")],
            )

            logger.debug(f"Stored sequence embedding: {sequence_id}")

        except Exception as e:
            raise StorageError(f"Failed to store sequence embedding {sequence_id}: {e}") from e

    def search_similar_sequences(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[SimilarityResult]:
        """Search for similar event sequences.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of SimilarityResult
        """
        try:
            results = self._sequences_collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                include=["embeddings", "metadatas", "distances"],
            )

            return self._parse_search_results(results, EmbeddingEntityType.TEMPORAL_EVENT)

        except Exception as e:
            raise StorageError(f"Failed to search sequences: {e}") from e

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _parse_search_results(
        self,
        results: Dict[str, Any],
        entity_type: EmbeddingEntityType,
    ) -> List[SimilarityResult]:
        """Parse ChromaDB search results into SimilarityResult objects."""
        similarity_results = []

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, entity_id in enumerate(ids):
            # Convert distance to similarity (ChromaDB returns L2 distance)
            distance = distances[i] if i < len(distances) else 0.0
            similarity = 1.0 / (1.0 + distance)  # Convert to similarity

            metadata = metadatas[i] if i < len(metadatas) else {}

            similarity_results.append(
                SimilarityResult(
                    entity_id=entity_id,
                    similarity_score=similarity,
                    entity_type=entity_type,
                    metadata=metadata,
                )
            )

        return similarity_results

    def delete_event(self, event_id: str) -> None:
        """Delete an event embedding."""
        try:
            self._events_collection.delete(ids=[event_id])
            logger.debug(f"Deleted event embedding: {event_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete event {event_id}: {e}") from e

    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity embedding."""
        try:
            self._entities_collection.delete(ids=[entity_id])
            logger.debug(f"Deleted entity embedding: {entity_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete entity {entity_id}: {e}") from e

    def delete_sequence(self, sequence_id: str) -> None:
        """Delete a sequence embedding."""
        try:
            self._sequences_collection.delete(ids=[sequence_id])
            logger.debug(f"Deleted sequence embedding: {sequence_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete sequence {sequence_id}: {e}") from e

    @property
    def temporal_embedder(self) -> TemporalEventEmbedder:
        """Access the temporal event embedder."""
        return self._temporal_embedder

    @property
    def static_embedder(self) -> StaticEntityEmbedder:
        """Access the static entity embedder."""
        return self._static_embedder

    @property
    def sequence_embedder(self) -> EventSequenceEmbedder:
        """Access the event sequence embedder."""
        return self._sequence_embedder

    @property
    def schema_version(self) -> str:
        """Current schema version."""
        return self._config.schema_version

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_model_manager"):
            self._model_manager.unload_all()
        logger.info("Closed TemporalAwareVectorWriter")

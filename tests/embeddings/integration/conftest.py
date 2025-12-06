"""Integration test fixtures for Vector Embedding Service Module 06.

Provides fixtures combining MultiModelEmbeddingService + SchemaVersionedEmbeddingStore
+ PKGSyncHandler + QualityMetricsTracker for end-to-end testing.

Option B Compliance:
- Uses frozen models (no parameter updates)
- Temporal context required for Events
- Schema version tracking enabled
- Quality metrics integration

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/06-integration-testing.md
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.request import EmbeddingRequest

from futurnal.pkg.sync.events import (
    PKGEvent,
    SyncEvent,
    SyncEventCapture,
    SyncEventType,
    SyncStatus,
    SourceOperation,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Test Entity Factory
# -----------------------------------------------------------------------------


@dataclass
class SampleEntity:
    """Sample entity for pipeline testing.

    Named 'SampleEntity' instead of 'TestEntity' to avoid pytest collection.
    """

    id: str
    type: str
    content: str
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Optional[TemporalEmbeddingContext] = None

    @classmethod
    def person(cls, name: str, description: str = "") -> "SampleEntity":
        """Create a Person test entity."""
        content = f"{name}: {description}" if description else name
        return cls(
            id=f"person_{uuid.uuid4().hex[:8]}",
            type="Person",
            content=content,
            name=name,
            metadata={"description": description},
        )

    @classmethod
    def event(
        cls,
        name: str,
        description: str = "",
        timestamp: Optional[datetime] = None,
        duration: Optional[timedelta] = None,
    ) -> "SampleEntity":
        """Create an Event test entity (with required temporal context)."""
        ts = timestamp or datetime.utcnow()
        content = f"{name}: {description}" if description else name
        return cls(
            id=f"event_{uuid.uuid4().hex[:8]}",
            type="Event",
            content=content,
            name=name,
            metadata={
                "description": description,
                "timestamp": ts.isoformat(),
                "duration": duration.total_seconds() if duration else None,
            },
            temporal_context=TemporalEmbeddingContext(
                timestamp=ts,
                duration=duration,
            ),
        )

    @classmethod
    def organization(cls, name: str, description: str = "") -> "SampleEntity":
        """Create an Organization test entity."""
        content = f"{name}: {description}" if description else name
        return cls(
            id=f"org_{uuid.uuid4().hex[:8]}",
            type="Organization",
            content=content,
            name=name,
            metadata={"description": description},
        )

    @classmethod
    def concept(cls, name: str, description: str = "") -> "SampleEntity":
        """Create a Concept test entity."""
        content = f"{name}: {description}" if description else name
        return cls(
            id=f"concept_{uuid.uuid4().hex[:8]}",
            type="Concept",
            content=content,
            name=name,
            metadata={"description": description},
        )


def create_test_entity(
    entity_type: str,
    name: str,
    description: str = "",
    **kwargs: Any,
) -> SampleEntity:
    """Factory function to create test entities."""
    if entity_type == "Person":
        return SampleEntity.person(name, description)
    elif entity_type == "Event":
        return SampleEntity.event(name, description, **kwargs)
    elif entity_type == "Organization":
        return SampleEntity.organization(name, description)
    elif entity_type == "Concept":
        return SampleEntity.concept(name, description)
    else:
        content = f"{name}: {description}" if description else name
        return SampleEntity(
            id=f"{entity_type.lower()}_{uuid.uuid4().hex[:8]}",
            type=entity_type,
            content=content,
            name=name,
            metadata={"description": description},
        )


def create_test_events(
    count: int,
    with_temporal_context: bool = True,
    base_date: Optional[datetime] = None,
) -> List[SampleEntity]:
    """Create multiple test events."""
    base = base_date or datetime(2024, 1, 1, 9, 0)
    events = []
    for i in range(count):
        timestamp = base + timedelta(hours=i)
        events.append(
            SampleEntity.event(
                name=f"Event {i}",
                description=f"Test event number {i}",
                timestamp=timestamp if with_temporal_context else None,
                duration=timedelta(hours=1) if with_temporal_context else None,
            )
        )
    return events


def create_temporal_context(
    timestamp: Optional[datetime] = None,
    duration: Optional[timedelta] = None,
    temporal_type: Optional[str] = None,
    causal_chain: Optional[List[str]] = None,
) -> TemporalEmbeddingContext:
    """Create a temporal embedding context."""
    return TemporalEmbeddingContext(
        timestamp=timestamp or datetime.utcnow(),
        duration=duration,
        temporal_type=temporal_type,
        causal_chain=causal_chain or [],
    )


# -----------------------------------------------------------------------------
# Mock Embedding Generation
# -----------------------------------------------------------------------------


def generate_mock_embedding(
    content: str,
    dimension: int = 768,
    temporal_context: Optional[TemporalEmbeddingContext] = None,
) -> List[float]:
    """Generate deterministic mock embedding based on content hash."""
    # Include temporal context in seed if provided
    seed_str = content
    if temporal_context and temporal_context.timestamp:
        seed_str += temporal_context.timestamp.isoformat()

    seed = hash(seed_str) % (2**32)
    rng = np.random.default_rng(seed)
    embedding = rng.random(dimension).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def create_mock_embedding_result(
    entity_type: str,
    content: str,
    temporal_context: Optional[TemporalEmbeddingContext] = None,
    model_id: str = "mock-model",
    schema_version: str = "test-1.0",
) -> EmbeddingResult:
    """Create a mock embedding result."""
    embedding = generate_mock_embedding(content, temporal_context=temporal_context)

    is_temporal = entity_type == "Event"

    return EmbeddingResult(
        embedding=embedding,
        entity_type=EmbeddingEntityType.TEMPORAL_EVENT if is_temporal else EmbeddingEntityType.STATIC_ENTITY,
        model_version=model_id,
        embedding_dimension=768,
        generation_time_ms=10.0,
        temporal_context_encoded=is_temporal and temporal_context is not None,
        causal_context_encoded=False,
        metadata={
            "model_id": model_id,
            "schema_version": schema_version,
        },
    )


# -----------------------------------------------------------------------------
# Mock Services
# -----------------------------------------------------------------------------


class MockEmbeddingService:
    """Mock embedding service for integration testing.

    Provides deterministic embeddings without loading real models.
    """

    def __init__(self, config: Optional[EmbeddingServiceConfig] = None):
        self._config = config or EmbeddingServiceConfig(schema_version="test-1.0")
        self._embed_count = 0
        self._batch_count = 0

    def embed(
        self,
        entity_type: str,
        content: str,
        temporal_context: Optional[TemporalEmbeddingContext] = None,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embedding for a single entity."""
        self._embed_count += 1
        return create_mock_embedding_result(
            entity_type=entity_type,
            content=content,
            temporal_context=temporal_context,
            schema_version=self._config.schema_version,
        )

    def embed_batch(
        self,
        requests: List[EmbeddingRequest],
    ) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of entities."""
        self._batch_count += 1
        results = []
        for request in requests:
            result = create_mock_embedding_result(
                entity_type=request.entity_type,
                content=request.content,
                temporal_context=request.temporal_context,
                schema_version=self._config.schema_version,
            )
            results.append(result)
        return results

    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types."""
        return ["Person", "Organization", "Event", "Concept", "Document"]

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "total_embeddings": self._embed_count,
            "total_batches": self._batch_count,
        }

    def close(self) -> None:
        """Clean up resources."""
        pass


class MockEmbeddingStore:
    """Mock embedding store for integration testing.

    Provides in-memory storage with schema version tracking.
    """

    def __init__(
        self,
        config: Optional[EmbeddingServiceConfig] = None,
        current_schema_version: int = 1,
    ):
        self._config = config or EmbeddingServiceConfig(schema_version="test-1.0")
        self._embeddings: Dict[str, Dict[str, Any]] = {}
        self._current_schema_version = current_schema_version
        self._reembedding_flags: Dict[str, str] = {}

    @property
    def current_schema_version(self) -> int:
        return self._current_schema_version

    @current_schema_version.setter
    def current_schema_version(self, value: int) -> None:
        self._current_schema_version = value

    def store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: Sequence[float],
        model_id: str,
        extraction_confidence: float,
        source_document_id: str,
        temporal_context: Optional[TemporalEmbeddingContext] = None,
    ) -> str:
        """Store embedding with metadata."""
        embedding_id = f"emb_{uuid.uuid4().hex[:12]}"

        self._embeddings[embedding_id] = {
            "embedding_id": embedding_id,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "embedding": list(embedding),
            "model_id": model_id,
            "extraction_confidence": extraction_confidence,
            "source_document_id": source_document_id,
            "schema_version": self._current_schema_version,
            "temporal_context": temporal_context,
            "created_at": datetime.utcnow().isoformat(),
        }

        return embedding_id

    def query_embeddings(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        entity_type: Optional[str] = None,
        min_schema_version: Optional[int] = None,
    ) -> List["MockSimilarityResult"]:
        """Query embeddings by similarity."""
        results = []
        query_np = np.array(query_vector)

        for emb_id, data in self._embeddings.items():
            # Skip if entity_id is flagged for reembedding (deleted)
            if data["entity_id"] in self._reembedding_flags and self._reembedding_flags[data["entity_id"]] == "deleted":
                continue

            # Filter by entity type
            if entity_type and data["entity_type"] != entity_type:
                continue

            # Filter by schema version
            if min_schema_version and data["schema_version"] < min_schema_version:
                continue

            # Compute similarity
            emb_np = np.array(data["embedding"])
            similarity = float(np.dot(query_np, emb_np))

            results.append(MockSimilarityResult(
                embedding_id=emb_id,
                entity_id=data["entity_id"],
                entity_type=EmbeddingEntityType.TEMPORAL_EVENT if data["entity_type"] == "Event" else EmbeddingEntityType.STATIC_ENTITY,
                similarity=similarity,
                metadata={
                    "schema_version": data["schema_version"],
                    "entity_id": data["entity_id"],
                    "model_id": data["model_id"],
                },
            ))

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def delete_embedding_by_entity_id(self, entity_id: str) -> int:
        """Delete embeddings for an entity."""
        to_delete = [
            emb_id for emb_id, data in self._embeddings.items()
            if data["entity_id"] == entity_id
        ]
        for emb_id in to_delete:
            del self._embeddings[emb_id]

        # Mark as deleted
        self._reembedding_flags[entity_id] = "deleted"
        return len(to_delete)

    def mark_for_reembedding(
        self,
        entity_ids: List[str],
        reason: str = "manual",
    ) -> int:
        """Mark embeddings for re-embedding."""
        count = 0
        for entity_id in entity_ids:
            # Check if entity has embeddings
            for data in self._embeddings.values():
                if data["entity_id"] == entity_id:
                    self._reembedding_flags[entity_id] = reason
                    count += 1
                    break
        return count

    def get_embeddings_needing_reembedding(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get embeddings that need re-embedding."""
        results = []
        for entity_id, reason in self._reembedding_flags.items():
            if reason == "deleted":
                continue
            for emb_id, data in self._embeddings.items():
                if data["entity_id"] == entity_id:
                    results.append({
                        "embedding_id": emb_id,
                        "entity_id": entity_id,
                        "reason": reason,
                    })
                    break
            if len(results) >= limit:
                break
        return results

    def clear_reembedding_flag(self, embedding_ids: List[str]) -> int:
        """Clear re-embedding flag for embeddings."""
        count = 0
        for emb_id in embedding_ids:
            if emb_id in self._embeddings:
                entity_id = self._embeddings[emb_id]["entity_id"]
                if entity_id in self._reembedding_flags:
                    del self._reembedding_flags[entity_id]
                    count += 1
        return count

    def close(self) -> None:
        """Clean up resources."""
        pass


@dataclass
class MockSimilarityResult:
    """Mock similarity result from embedding query."""

    embedding_id: str
    entity_id: str
    entity_type: EmbeddingEntityType
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockSyncHandler:
    """Mock sync handler for integration testing."""

    def __init__(
        self,
        embedding_service: MockEmbeddingService,
        embedding_store: MockEmbeddingStore,
        sync_event_capture: Optional[SyncEventCapture] = None,
    ):
        self._embedding_service = embedding_service
        self._embedding_store = embedding_store
        self._sync_capture = sync_event_capture or SyncEventCapture()
        self._statistics = {
            "events_processed": 0,
            "events_succeeded": 0,
            "events_failed": 0,
        }

    def handle_event(self, event: PKGEvent) -> bool:
        """Handle a PKG event."""
        self._statistics["events_processed"] += 1

        try:
            event_type = event.event_type
            if isinstance(event_type, str):
                event_type = SyncEventType(event_type)

            if event_type == SyncEventType.ENTITY_CREATED:
                return self._handle_entity_created(event)
            elif event_type == SyncEventType.ENTITY_UPDATED:
                return self._handle_entity_updated(event)
            elif event_type == SyncEventType.ENTITY_DELETED:
                return self._handle_entity_deleted(event)
            else:
                self._statistics["events_succeeded"] += 1
                return True

        except Exception as e:
            self._statistics["events_failed"] += 1
            logger.error(f"Failed to handle event: {e}")
            return False

    def _handle_entity_created(self, event: PKGEvent) -> bool:
        """Handle entity created event."""
        # Option B: Event entities require temporal context
        if event.entity_type == "Event":
            new_data = event.new_data or {}
            if "timestamp" not in new_data:
                self._statistics["events_failed"] += 1
                self._capture_sync_event(event, SyncStatus.FAILED, "Missing temporal context")
                return False

        # Format content
        new_data = event.new_data or {}
        content = self._format_entity_content(event.entity_type, new_data)

        # Extract temporal context if Event
        temporal_context = None
        if event.entity_type == "Event" and "timestamp" in new_data:
            timestamp = new_data["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            temporal_context = TemporalEmbeddingContext(
                timestamp=timestamp,
                duration=timedelta(seconds=new_data.get("duration", 0)) if new_data.get("duration") else None,
            )

        # Generate embedding
        result = self._embedding_service.embed(
            entity_type=event.entity_type,
            content=content,
            temporal_context=temporal_context,
            entity_id=event.entity_id,
        )

        # Store embedding
        self._embedding_store.store_embedding(
            entity_id=event.entity_id,
            entity_type=event.entity_type,
            embedding=result.embedding,
            model_id=result.model_version,
            extraction_confidence=event.extraction_confidence or 0.85,
            source_document_id=event.source_document_id or "unknown",
            temporal_context=temporal_context,
        )

        self._statistics["events_succeeded"] += 1
        self._capture_sync_event(event, SyncStatus.COMPLETED)
        return True

    def _handle_entity_updated(self, event: PKGEvent) -> bool:
        """Handle entity updated event."""
        # Similar to create, just update the embedding
        return self._handle_entity_created(event)

    def _handle_entity_deleted(self, event: PKGEvent) -> bool:
        """Handle entity deleted event."""
        self._embedding_store.delete_embedding_by_entity_id(event.entity_id)
        self._statistics["events_succeeded"] += 1
        self._capture_sync_event(event, SyncStatus.COMPLETED)
        return True

    def _format_entity_content(
        self,
        entity_type: str,
        data: Dict[str, Any],
    ) -> str:
        """Format entity data into content string."""
        name = data.get("name", "")
        description = data.get("description", "")
        if description:
            return f"{name}: {description}"
        return name

    def _capture_sync_event(
        self,
        event: PKGEvent,
        status: SyncStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Capture sync event for monitoring."""
        sync_event = event.to_sync_event(
            sync_status=status,
            error_message=error_message,
        )
        self._sync_capture.capture(sync_event)

    def get_statistics(self) -> Dict[str, Any]:
        """Get sync handler statistics."""
        processed = self._statistics["events_processed"]
        succeeded = self._statistics["events_succeeded"]
        return {
            "events_processed": processed,
            "events_succeeded": succeeded,
            "events_failed": self._statistics["events_failed"],
            "success_rate": succeeded / processed if processed > 0 else 0.0,
        }


# -----------------------------------------------------------------------------
# Embedding Pipeline Container
# -----------------------------------------------------------------------------


@dataclass
class EmbeddingPipeline:
    """Container for embedding pipeline components."""

    embedding_service: MockEmbeddingService
    store: MockEmbeddingStore
    sync_handler: MockSyncHandler
    sync_capture: SyncEventCapture
    config: EmbeddingServiceConfig
    quality_tracker: Optional[Any] = None

    def close(self) -> None:
        """Clean up resources."""
        self.embedding_service.close()
        self.store.close()


def create_embedding_pipeline(
    config: Optional[EmbeddingServiceConfig] = None,
    with_quality_tracking: bool = True,
    current_schema_version: int = 1,
) -> EmbeddingPipeline:
    """Create a complete embedding pipeline for integration testing.

    Args:
        config: Optional custom configuration
        with_quality_tracking: Enable quality metrics tracking
        current_schema_version: Initial schema version

    Returns:
        EmbeddingPipeline with all components wired together
    """
    config = config or EmbeddingServiceConfig(schema_version="test-1.0")
    sync_capture = SyncEventCapture()

    # Create embedding service
    embedding_service = MockEmbeddingService(config=config)

    # Create store
    store = MockEmbeddingStore(
        config=config,
        current_schema_version=current_schema_version,
    )

    # Create sync handler
    sync_handler = MockSyncHandler(
        embedding_service=embedding_service,
        embedding_store=store,
        sync_event_capture=sync_capture,
    )

    # Quality tracker (optional - mock for now)
    quality_tracker = None
    if with_quality_tracking:
        quality_tracker = MagicMock()
        quality_tracker.record_embedding_quality = MagicMock(return_value=MagicMock(
            overall_quality_score=0.85,
        ))
        quality_tracker.get_statistics = MagicMock(return_value={
            "total_tracked": 0,
            "avg_quality": 0.85,
        })

    return EmbeddingPipeline(
        embedding_service=embedding_service,
        store=store,
        sync_handler=sync_handler,
        sync_capture=sync_capture,
        config=config,
        quality_tracker=quality_tracker,
    )


# -----------------------------------------------------------------------------
# Mock PKG Client
# -----------------------------------------------------------------------------


class MockPKGClient:
    """Mock PKG client for consistency validation tests."""

    def __init__(self) -> None:
        self.entities: Dict[str, Dict[str, Any]] = {}

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
    ) -> None:
        """Add entity to mock PKG."""
        self.entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "properties": properties,
        }

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity from mock PKG."""
        return self.entities.get(entity_id)

    def list_entity_ids(self) -> List[str]:
        """List all entity IDs."""
        return list(self.entities.keys())

    def delete_entity(self, entity_id: str) -> None:
        """Delete entity from mock PKG."""
        self.entities.pop(entity_id, None)


def create_mock_pkg_client() -> MockPKGClient:
    """Create a mock PKG client."""
    return MockPKGClient()


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def embedding_pipeline(tmp_path: Path) -> Generator[EmbeddingPipeline, None, None]:
    """Provide a complete embedding pipeline for integration tests."""
    pipeline = create_embedding_pipeline(
        with_quality_tracking=True,
    )
    yield pipeline
    pipeline.close()


@pytest.fixture
def sample_test_entities() -> Dict[str, SampleEntity]:
    """Provide sample test entities of different types."""
    return {
        "person": create_test_entity("Person", "John Doe", "Software Engineer"),
        "org": create_test_entity("Organization", "Acme Corp", "Technology company"),
        "event": create_test_entity(
            "Event",
            "Team Meeting",
            "Quarterly planning",
            timestamp=datetime(2024, 1, 15, 10, 0),
            duration=timedelta(hours=2),
        ),
        "concept": create_test_entity("Concept", "Machine Learning", "AI technique"),
    }


@pytest.fixture
def batch_test_entities() -> List[SampleEntity]:
    """Provide a batch of test entities for throughput testing."""
    entities = []
    for i in range(100):
        if i % 4 == 0:
            entities.append(create_test_entity("Person", f"Person {i}"))
        elif i % 4 == 1:
            entities.append(create_test_entity("Organization", f"Org {i}"))
        elif i % 4 == 2:
            entities.append(
                create_test_entity(
                    "Event",
                    f"Event {i}",
                    timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                )
            )
        else:
            entities.append(create_test_entity("Concept", f"Concept {i}"))
    return entities


@pytest.fixture
def sync_event_capture() -> SyncEventCapture:
    """Provide sync event capture for monitoring."""
    return SyncEventCapture()


@pytest.fixture
def mock_pkg_client() -> MockPKGClient:
    """Provide a mock PKG client."""
    return MockPKGClient()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.linalg.norm(a_np - b_np))

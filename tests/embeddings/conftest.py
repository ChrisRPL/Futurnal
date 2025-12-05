"""Test fixtures for the embeddings module.

Provides:
- Mock embedding models that return deterministic embeddings
- Test data factories for events, entities, and contexts
- ChromaDB test collection fixtures
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from futurnal.embeddings.config import EmbeddingServiceConfig, ModelConfig, ModelType
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    FusionWeights,
    TemporalEmbeddingContext,
)


# -----------------------------------------------------------------------------
# Mock Embedding Model
# -----------------------------------------------------------------------------


class MockEmbeddingModel:
    """Mock embedding model that returns deterministic embeddings.

    Uses hash of input text to generate reproducible embeddings.
    """

    def __init__(
        self,
        dimension: int = 384,
        model_name: str = "mock-model",
    ) -> None:
        self.dimension = dimension
        self.model_name = model_name

    def encode(
        self,
        sentences: Any,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate deterministic embeddings based on input hash."""
        if isinstance(sentences, str):
            return self._encode_single(sentences, instruction)
        else:
            return np.array([
                self._encode_single(s, instruction)
                for s in sentences
            ])

    def _encode_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Generate a single embedding."""
        # Use hash for reproducibility
        combined = f"{instruction or ''}{text}"
        seed = hash(combined) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.random(self.dimension)
        # Normalize
        return embedding / np.linalg.norm(embedding)


class MockInstructorModel:
    """Mock Instructor model wrapper."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self._base = MockEmbeddingModel(dimension)

    def encode(
        self,
        sentences: Any,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode with instruction support."""
        if isinstance(sentences, list) and len(sentences) > 0:
            if isinstance(sentences[0], list):
                # Instructor format: [[instruction, text], ...]
                texts = [s[1] for s in sentences]
                instructions = [s[0] for s in sentences]
                return np.array([
                    self._base._encode_single(t, i)
                    for t, i in zip(texts, instructions)
                ])
        return self._base.encode(sentences, batch_size, show_progress_bar)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_model() -> MockEmbeddingModel:
    """Create a mock embedding model."""
    return MockEmbeddingModel(dimension=384)


@pytest.fixture
def mock_instructor_model() -> MockInstructorModel:
    """Create a mock Instructor model."""
    return MockInstructorModel(dimension=768)


@pytest.fixture
def mock_model_manager(
    mock_embedding_model: MockEmbeddingModel,
    mock_instructor_model: MockInstructorModel,
) -> ModelManager:
    """Create a ModelManager with mocked models.

    Patches the model loading to return mock models instead of
    loading real models from HuggingFace.
    """
    config = EmbeddingServiceConfig()

    with patch.object(ModelManager, "_load_model") as mock_load:
        manager = ModelManager(config)

        # Pre-populate with mock models
        manager._models["content-instructor"] = mock_instructor_model
        manager._models["temporal-minilm"] = mock_embedding_model
        manager._model_versions["content-instructor"] = "mock:instructor-large"
        manager._model_versions["temporal-minilm"] = "mock:all-MiniLM-L6-v2"

        return manager


@pytest.fixture
def embedding_config() -> EmbeddingServiceConfig:
    """Create a test embedding configuration."""
    return EmbeddingServiceConfig(
        schema_version="test-1.0",
    )


@pytest.fixture
def default_fusion_weights() -> FusionWeights:
    """Create default fusion weights (60/30/10)."""
    return FusionWeights()


@pytest.fixture
def temporal_heavy_weights() -> FusionWeights:
    """Create temporal-heavy fusion weights."""
    return FusionWeights(
        content_weight=0.4,
        temporal_weight=0.5,
        causal_weight=0.1,
    )


# -----------------------------------------------------------------------------
# Test Data Factories
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_temporal_context() -> TemporalEmbeddingContext:
    """Create a sample temporal context."""
    return TemporalEmbeddingContext(
        timestamp=datetime(2024, 1, 15, 14, 30, 0),
        duration=timedelta(hours=2),
        temporal_type="DURING",
    )


@pytest.fixture
def sample_temporal_context_with_causal() -> TemporalEmbeddingContext:
    """Create a temporal context with causal chain."""
    return TemporalEmbeddingContext(
        timestamp=datetime(2024, 1, 15, 14, 30, 0),
        duration=timedelta(hours=2),
        temporal_type="CAUSES",
        causal_chain=["Meeting", "Discussion", "Decision"],
    )


@pytest.fixture
def january_temporal_context() -> TemporalEmbeddingContext:
    """Create a January temporal context."""
    return TemporalEmbeddingContext(
        timestamp=datetime(2024, 1, 15, 14, 30, 0),
    )


@pytest.fixture
def february_temporal_context() -> TemporalEmbeddingContext:
    """Create a February temporal context (1 month later)."""
    return TemporalEmbeddingContext(
        timestamp=datetime(2024, 2, 15, 14, 30, 0),
    )


@pytest.fixture
def june_temporal_context() -> TemporalEmbeddingContext:
    """Create a June temporal context (5 months later)."""
    return TemporalEmbeddingContext(
        timestamp=datetime(2024, 6, 15, 14, 30, 0),
    )


def create_temporal_context(
    timestamp: datetime,
    duration: Optional[timedelta] = None,
    temporal_type: Optional[str] = None,
    causal_chain: Optional[List[str]] = None,
) -> TemporalEmbeddingContext:
    """Factory function to create temporal contexts."""
    return TemporalEmbeddingContext(
        timestamp=timestamp,
        duration=duration,
        temporal_type=temporal_type,
        causal_chain=causal_chain or [],
    )


# -----------------------------------------------------------------------------
# Event Test Data
# -----------------------------------------------------------------------------


class SimpleEvent:
    """Simple event class for testing."""

    def __init__(
        self,
        name: str,
        event_type: str = "test_event",
        description: str = "",
    ) -> None:
        self.name = name
        self.event_type = event_type
        self.description = description


@pytest.fixture
def sample_event() -> SimpleEvent:
    """Create a sample event."""
    return SimpleEvent(
        name="Team Meeting",
        event_type="meeting",
        description="Quarterly planning discussion",
    )


@pytest.fixture
def meeting_event() -> SimpleEvent:
    """Create a meeting event."""
    return SimpleEvent(
        name="Meeting",
        event_type="meeting",
        description="Team sync meeting",
    )


@pytest.fixture
def decision_event() -> SimpleEvent:
    """Create a decision event."""
    return SimpleEvent(
        name="Decision",
        event_type="decision",
        description="Project direction decided",
    )


@pytest.fixture
def publication_event() -> SimpleEvent:
    """Create a publication event."""
    return SimpleEvent(
        name="Publication",
        event_type="publication",
        description="Article published",
    )


@pytest.fixture
def event_sequence(
    meeting_event: SimpleEvent,
    decision_event: SimpleEvent,
    publication_event: SimpleEvent,
) -> List[SimpleEvent]:
    """Create a typical event sequence."""
    return [meeting_event, decision_event, publication_event]


@pytest.fixture
def event_sequence_contexts() -> List[TemporalEmbeddingContext]:
    """Create temporal contexts for event sequence."""
    return [
        TemporalEmbeddingContext(timestamp=datetime(2024, 1, 1, 10, 0, 0)),
        TemporalEmbeddingContext(timestamp=datetime(2024, 1, 2, 14, 0, 0)),
        TemporalEmbeddingContext(timestamp=datetime(2024, 1, 5, 9, 0, 0)),
    ]


# -----------------------------------------------------------------------------
# ChromaDB Mock Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client."""
    client = MagicMock()

    # Mock collection
    collection = MagicMock()
    collection.upsert = MagicMock()
    collection.delete = MagicMock()
    collection.query = MagicMock(return_value={
        "ids": [["test_id"]],
        "distances": [[0.1]],
        "metadatas": [[{"entity_type": "temporal_event"}]],
    })

    client.get_or_create_collection = MagicMock(return_value=collection)

    return client


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))

"""Test fixtures for schema-aware hybrid retrieval.

Provides mocks and fixtures for testing:
- QueryEmbeddingRouter
- SchemaAwareRetrieval
- EntityTypeRetrievalStrategy
- SchemaVersionCompatibility
- ResultFusion

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.types import (
    GraphSearchResult,
    HybridSearchResult,
    QueryEmbeddingType,
    SchemaCompatibilityResult,
    TemporalQueryContext,
    VectorSearchResult,
)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hybrid_config() -> HybridSearchConfig:
    """Default HybridSearchConfig for testing."""
    return HybridSearchConfig()


@pytest.fixture
def custom_hybrid_config() -> HybridSearchConfig:
    """HybridSearchConfig with custom values for testing."""
    return HybridSearchConfig(
        default_vector_weight=0.6,
        default_graph_weight=0.4,
        event_vector_weight=0.3,
        event_graph_weight=0.7,
        schema_drift_threshold=2,
        target_latency_ms=500.0,
    )


# ---------------------------------------------------------------------------
# Mock Embedding Service
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock MultiModelEmbeddingService for testing."""
    service = MagicMock()

    # Mock embed method to return a result with embedding
    mock_result = MagicMock()
    mock_result.embedding = [0.1] * 768
    mock_result.embedding_dimension = 768
    service.embed.return_value = mock_result

    # Mock router access
    service.router = MagicMock()

    return service


# ---------------------------------------------------------------------------
# Mock Embedding Store
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_store() -> MagicMock:
    """Mock SchemaVersionedEmbeddingStore for testing."""
    store = MagicMock()

    # Mock schema version
    store._get_current_schema_version.return_value = 5

    # Mock query_embeddings
    store.query_embeddings.return_value = [
        MagicMock(
            entity_id="entity_1",
            entity_type="Event",
            document="Test event content",
            similarity_score=0.9,
            metadata={"schema_version": 5},
        ),
        MagicMock(
            entity_id="entity_2",
            entity_type="Person",
            document="Test person content",
            similarity_score=0.8,
            metadata={"schema_version": 4},
        ),
        MagicMock(
            entity_id="entity_3",
            entity_type="Document",
            document="Test document content",
            similarity_score=0.7,
            metadata={"schema_version": 5},
        ),
    ]

    return store


# ---------------------------------------------------------------------------
# Mock PKG Queries
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pkg_queries() -> MagicMock:
    """Mock TemporalGraphQueries for testing."""
    pkg = MagicMock()

    # Mock temporal neighborhood
    neighborhood = MagicMock()
    neighborhood.event_neighbors = [
        MagicMock(id="neighbor_event_1", name="Neighbor Event 1"),
        MagicMock(id="neighbor_event_2", name="Neighbor Event 2"),
    ]
    neighborhood.entity_neighbors = [
        MagicMock(id="neighbor_entity_1", type="Person"),
    ]
    pkg.query_temporal_neighborhood.return_value = neighborhood

    return pkg


# ---------------------------------------------------------------------------
# Mock Temporal Engine
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_temporal_engine() -> MagicMock:
    """Mock TemporalQueryEngine for testing."""
    engine = MagicMock()

    # Mock query_temporal_neighborhood
    neighborhood = MagicMock()
    neighborhood.event_neighbors = [
        MagicMock(id="temp_event_1", name="Temporal Event 1"),
        MagicMock(id="temp_event_2", name="Temporal Event 2"),
    ]
    neighborhood.entity_neighbors = []
    engine.query_temporal_neighborhood.return_value = neighborhood

    return engine


# ---------------------------------------------------------------------------
# Mock Causal Retrieval
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_causal_retrieval() -> MagicMock:
    """Mock CausalChainRetrieval for testing."""
    retrieval = MagicMock()

    # Mock find_effects
    effects_result = MagicMock()
    effects_result.effects = [
        MagicMock(effect_id="effect_1", aggregate_confidence=0.85),
        MagicMock(effect_id="effect_2", aggregate_confidence=0.75),
    ]
    retrieval.find_effects.return_value = effects_result

    # Mock find_causes
    causes_result = MagicMock()
    causes_result.causes = [
        MagicMock(cause_id="cause_1", aggregate_confidence=0.9),
    ]
    retrieval.find_causes.return_value = causes_result

    return retrieval


# ---------------------------------------------------------------------------
# Test Data Factories
# ---------------------------------------------------------------------------


def create_vector_result(
    entity_id: str,
    entity_type: str = "Event",
    content: str = "Test content",
    similarity_score: float = 0.8,
    schema_version: int = 5,
) -> VectorSearchResult:
    """Create a VectorSearchResult for testing."""
    return VectorSearchResult(
        entity_id=entity_id,
        entity_type=entity_type,
        content=content,
        similarity_score=similarity_score,
        schema_version=schema_version,
        metadata={},
    )


def create_graph_result(
    entity_id: str,
    entity_type: str = "Event",
    path_score: float = 0.7,
    relationship_types: Optional[List[str]] = None,
) -> GraphSearchResult:
    """Create a GraphSearchResult for testing."""
    return GraphSearchResult(
        entity_id=entity_id,
        entity_type=entity_type,
        path_from_seed=[entity_id],
        path_score=path_score,
        relationship_types=relationship_types or ["CAUSES"],
        metadata={},
    )


def create_hybrid_result(
    entity_id: str,
    entity_type: str = "Event",
    vector_score: float = 0.8,
    graph_score: float = 0.6,
    combined_score: float = 0.7,
) -> HybridSearchResult:
    """Create a HybridSearchResult for testing."""
    return HybridSearchResult(
        entity_id=entity_id,
        entity_type=entity_type,
        vector_score=vector_score,
        graph_score=graph_score,
        combined_score=combined_score,
        source="hybrid",
        content="Test content",
        schema_version=5,
        metadata={},
    )


def create_temporal_context(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    reference: Optional[datetime] = None,
    relation: Optional[str] = None,
) -> TemporalQueryContext:
    """Create a TemporalQueryContext for testing."""
    return TemporalQueryContext(
        time_range_start=start,
        time_range_end=end,
        reference_timestamp=reference,
        temporal_relation=relation,
    )


# ---------------------------------------------------------------------------
# Batch Test Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_vector_results() -> List[VectorSearchResult]:
    """Sample vector results for testing fusion."""
    return [
        create_vector_result("entity_1", "Event", similarity_score=0.95),
        create_vector_result("entity_2", "Person", similarity_score=0.85),
        create_vector_result("entity_3", "Document", similarity_score=0.75),
        create_vector_result("entity_4", "Concept", similarity_score=0.65),
    ]


@pytest.fixture
def sample_graph_results() -> List[GraphSearchResult]:
    """Sample graph results for testing fusion."""
    return [
        create_graph_result("entity_1", "Event", path_score=0.8),
        create_graph_result("entity_5", "Event", path_score=0.7),
        create_graph_result("entity_6", "Person", path_score=0.6),
    ]


@pytest.fixture
def sample_temporal_context() -> TemporalQueryContext:
    """Sample temporal context for testing."""
    return create_temporal_context(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 3, 31),
        relation="DURING",
    )

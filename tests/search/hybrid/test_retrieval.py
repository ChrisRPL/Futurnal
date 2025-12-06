"""Tests for SchemaAwareRetrieval.

Tests the main hybrid retrieval engine.
"""

import pytest
from unittest.mock import MagicMock, patch

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import (
    GraphExpansionError,
    HybridSearchError,
    InvalidHybridQueryError,
    VectorSearchError,
)
from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
from futurnal.search.hybrid.types import HybridSearchQuery

from tests.search.hybrid.conftest import create_temporal_context


class TestSchemaAwareRetrieval:
    """Test SchemaAwareRetrieval functionality."""

    @pytest.fixture
    def retrieval(
        self,
        mock_pkg_queries,
        mock_embedding_store,
        mock_temporal_engine,
        mock_causal_retrieval,
        mock_embedding_service,
        hybrid_config,
    ):
        """Create SchemaAwareRetrieval with all mocks."""
        router = QueryEmbeddingRouter(mock_embedding_service, hybrid_config)
        return SchemaAwareRetrieval(
            pkg_queries=mock_pkg_queries,
            embedding_store=mock_embedding_store,
            temporal_engine=mock_temporal_engine,
            causal_retrieval=mock_causal_retrieval,
            embedding_router=router,
            config=hybrid_config,
        )

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_minimal(self, mock_pkg_queries, mock_embedding_store):
        """Retrieval initializes with minimal dependencies."""
        retrieval = SchemaAwareRetrieval(
            pkg_queries=mock_pkg_queries,
            embedding_store=mock_embedding_store,
        )
        assert retrieval.config is not None

    def test_init_full(
        self,
        mock_pkg_queries,
        mock_embedding_store,
        mock_temporal_engine,
        mock_causal_retrieval,
        mock_embedding_service,
        hybrid_config,
    ):
        """Retrieval initializes with all dependencies."""
        router = QueryEmbeddingRouter(mock_embedding_service, hybrid_config)
        retrieval = SchemaAwareRetrieval(
            pkg_queries=mock_pkg_queries,
            embedding_store=mock_embedding_store,
            temporal_engine=mock_temporal_engine,
            causal_retrieval=mock_causal_retrieval,
            embedding_router=router,
            config=hybrid_config,
        )
        assert retrieval._temporal is not None
        assert retrieval._causal is not None
        assert retrieval._router is not None

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_empty_query(self, retrieval):
        """Empty query raises InvalidHybridQueryError."""
        with pytest.raises(InvalidHybridQueryError, match="cannot be empty"):
            retrieval.hybrid_search(query="", intent="exploratory")

    def test_validate_invalid_intent(self, retrieval):
        """Invalid intent raises InvalidHybridQueryError."""
        with pytest.raises(InvalidHybridQueryError, match="Invalid intent"):
            retrieval.hybrid_search(query="test", intent="invalid")

    def test_validate_invalid_top_k(self, retrieval):
        """Invalid top_k raises InvalidHybridQueryError."""
        with pytest.raises(InvalidHybridQueryError, match="top_k"):
            retrieval.hybrid_search(query="test", intent="exploratory", top_k=0)

        with pytest.raises(InvalidHybridQueryError, match="top_k"):
            retrieval.hybrid_search(query="test", intent="exploratory", top_k=200)

    def test_validate_invalid_weights(self, retrieval):
        """Invalid weights raise InvalidHybridQueryError."""
        with pytest.raises(InvalidHybridQueryError, match="vector_weight"):
            retrieval.hybrid_search(
                query="test",
                intent="exploratory",
                vector_weight=1.5,
            )

    # -------------------------------------------------------------------------
    # Basic Search Tests
    # -------------------------------------------------------------------------

    def test_hybrid_search_returns_results(self, retrieval):
        """Basic hybrid search returns results."""
        results = retrieval.hybrid_search(
            query="What happened in the project?",
            intent="exploratory",
            top_k=10,
        )

        assert results is not None
        assert isinstance(results, list)

    def test_hybrid_search_with_temporal_intent(self, retrieval):
        """Temporal intent triggers temporal expansion."""
        results = retrieval.hybrid_search(
            query="What happened yesterday?",
            intent="temporal",
            top_k=10,
        )

        assert results is not None

    def test_hybrid_search_with_causal_intent(self, retrieval, mock_causal_retrieval):
        """Causal intent triggers causal expansion."""
        results = retrieval.hybrid_search(
            query="What caused the delay?",
            intent="causal",
            top_k=10,
        )

        assert results is not None
        # Causal retrieval should have been called
        mock_causal_retrieval.find_effects.assert_called()

    def test_hybrid_search_with_temporal_context(
        self, retrieval, sample_temporal_context
    ):
        """Search with temporal context."""
        results = retrieval.hybrid_search(
            query="Events in Q1",
            intent="temporal",
            top_k=10,
            temporal_context=sample_temporal_context,
        )

        assert results is not None

    # -------------------------------------------------------------------------
    # Query Model Tests
    # -------------------------------------------------------------------------

    def test_hybrid_search_with_query_model(self, retrieval):
        """Search using HybridSearchQuery model."""
        query = HybridSearchQuery(
            query_text="What happened?",
            intent="exploratory",
            top_k=5,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        results = retrieval.hybrid_search_with_query(query)

        assert results is not None
        assert isinstance(results, list)

    # -------------------------------------------------------------------------
    # Weight Adjustment Tests
    # -------------------------------------------------------------------------

    def test_adjust_weights_temporal_intent(self, retrieval):
        """Temporal intent increases graph weight."""
        weights = retrieval._adjust_weights(
            intent="temporal",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=10,
            graph_result_count=10,
        )

        assert weights["graph"] > weights["vector"]

    def test_adjust_weights_causal_intent(self, retrieval):
        """Causal intent increases graph weight more."""
        weights = retrieval._adjust_weights(
            intent="causal",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=10,
            graph_result_count=10,
        )

        assert weights["graph"] > weights["vector"]

    def test_adjust_weights_lookup_intent(self, retrieval):
        """Lookup intent increases vector weight."""
        weights = retrieval._adjust_weights(
            intent="lookup",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=10,
            graph_result_count=10,
        )

        assert weights["vector"] > weights["graph"]

    def test_adjust_weights_result_count_adjustment(self, retrieval):
        """Weights adjust based on result counts."""
        # Few vector results, many graph results
        weights = retrieval._adjust_weights(
            intent="exploratory",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=2,
            graph_result_count=20,
        )

        assert weights["graph"] > weights["vector"]

    def test_weights_normalized(self, retrieval):
        """Weights are normalized to sum to 1."""
        weights = retrieval._adjust_weights(
            intent="exploratory",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=10,
            graph_result_count=10,
        )

        assert abs(weights["vector"] + weights["graph"] - 1.0) < 0.01


class TestSchemaAwareRetrievalExpansion:
    """Test graph expansion strategies."""

    @pytest.fixture
    def retrieval(
        self,
        mock_pkg_queries,
        mock_embedding_store,
        mock_temporal_engine,
        mock_causal_retrieval,
        mock_embedding_service,
        hybrid_config,
    ):
        """Create SchemaAwareRetrieval with all mocks."""
        router = QueryEmbeddingRouter(mock_embedding_service, hybrid_config)
        return SchemaAwareRetrieval(
            pkg_queries=mock_pkg_queries,
            embedding_store=mock_embedding_store,
            temporal_engine=mock_temporal_engine,
            causal_retrieval=mock_causal_retrieval,
            embedding_router=router,
            config=hybrid_config,
        )

    def test_causal_expansion(self, retrieval, mock_causal_retrieval):
        """Causal expansion uses CausalChainRetrieval."""
        results = retrieval._causal_expansion(["event_1", "event_2"])

        assert results is not None
        mock_causal_retrieval.find_effects.assert_called()
        mock_causal_retrieval.find_causes.assert_called()

    def test_temporal_expansion(self, retrieval, mock_temporal_engine):
        """Temporal expansion uses TemporalQueryEngine."""
        results = retrieval._temporal_expansion(["event_1"])

        assert results is not None
        mock_temporal_engine.query_temporal_neighborhood.assert_called()

    def test_neighborhood_expansion(self, retrieval, mock_pkg_queries):
        """Neighborhood expansion uses PKG queries."""
        results = retrieval._neighborhood_expansion(["entity_1"])

        assert results is not None
        mock_pkg_queries.query_temporal_neighborhood.assert_called()

    def test_expansion_handles_empty_seeds(self, retrieval):
        """Expansion with empty seeds returns empty list."""
        results = retrieval._graph_expansion(
            seed_entities=[],
            intent="exploratory",
        )

        assert results == []


class TestSchemaAwareRetrievalErrors:
    """Test error handling."""

    def test_vector_search_error(
        self, mock_pkg_queries, mock_embedding_store, hybrid_config
    ):
        """VectorSearchError raised on embedding store failure."""
        mock_embedding_store.query_embeddings.side_effect = Exception("DB error")

        retrieval = SchemaAwareRetrieval(
            pkg_queries=mock_pkg_queries,
            embedding_store=mock_embedding_store,
            config=hybrid_config,
        )

        # Need to mock the query embedding method
        with patch.object(
            retrieval, "_get_query_embedding", return_value=[0.1] * 768
        ):
            with pytest.raises(VectorSearchError, match="Vector search failed"):
                retrieval.hybrid_search(query="test", intent="exploratory")

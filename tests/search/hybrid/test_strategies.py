"""Tests for EntityTypeRetrievalStrategy.

Tests entity-type-specific retrieval strategies.
"""

import pytest
from unittest.mock import MagicMock

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.strategies import EntityTypeRetrievalStrategy
from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval

from tests.search.hybrid.conftest import create_hybrid_result, create_temporal_context


class TestEntityTypeRetrievalStrategy:
    """Test EntityTypeRetrievalStrategy functionality."""

    @pytest.fixture
    def mock_retrieval(self, hybrid_config):
        """Create mock SchemaAwareRetrieval."""
        retrieval = MagicMock(spec=SchemaAwareRetrieval)
        retrieval.config = hybrid_config

        # Mock hybrid_search to return results
        retrieval.hybrid_search.return_value = [
            create_hybrid_result("event_1", "Event"),
            create_hybrid_result("person_1", "Person"),
            create_hybrid_result("event_2", "Event"),
            create_hybrid_result("doc_1", "Document"),
        ]

        return retrieval

    @pytest.fixture
    def strategy(self, mock_retrieval, hybrid_config):
        """Create EntityTypeRetrievalStrategy."""
        return EntityTypeRetrievalStrategy(
            retrieval=mock_retrieval,
            config=hybrid_config,
        )

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init(self, mock_retrieval, hybrid_config):
        """Strategy initializes correctly."""
        strategy = EntityTypeRetrievalStrategy(
            retrieval=mock_retrieval,
            config=hybrid_config,
        )
        assert strategy.config == hybrid_config

    # -------------------------------------------------------------------------
    # Weight Retrieval Tests
    # -------------------------------------------------------------------------

    def test_get_weights_for_event(self, strategy, hybrid_config):
        """Event gets temporal-optimized weights."""
        weights = strategy.get_weights_for_type("Event")

        assert weights["vector"] == hybrid_config.event_vector_weight
        assert weights["graph"] == hybrid_config.event_graph_weight
        # Events should favor graph
        assert weights["graph"] > weights["vector"]

    def test_get_weights_for_code(self, strategy, hybrid_config):
        """CodeEntity gets code-optimized weights."""
        weights = strategy.get_weights_for_type("CodeEntity")

        assert weights["vector"] == hybrid_config.code_vector_weight
        assert weights["graph"] == hybrid_config.code_graph_weight
        # Code should favor vector
        assert weights["vector"] > weights["graph"]

    def test_get_weights_for_document(self, strategy, hybrid_config):
        """Document gets document-optimized weights."""
        weights = strategy.get_weights_for_type("Document")

        assert weights["vector"] == hybrid_config.document_vector_weight
        assert weights["graph"] == hybrid_config.document_graph_weight
        # Documents should slightly favor vector
        assert weights["vector"] > weights["graph"]

    def test_get_weights_for_static_entity(self, strategy, hybrid_config):
        """Static entities get balanced weights."""
        for entity_type in ["Person", "Organization", "Concept"]:
            weights = strategy.get_weights_for_type(entity_type)
            assert weights["vector"] == 0.5
            assert weights["graph"] == 0.5

    def test_get_weights_for_unknown(self, strategy, hybrid_config):
        """Unknown types get default weights."""
        weights = strategy.get_weights_for_type("UnknownType")

        assert weights["vector"] == hybrid_config.default_vector_weight
        assert weights["graph"] == hybrid_config.default_graph_weight

    # -------------------------------------------------------------------------
    # Intent Mapping Tests
    # -------------------------------------------------------------------------

    def test_get_intent_for_event(self, strategy):
        """Event type maps to temporal intent."""
        intent = strategy.get_intent_for_type("Event")
        assert intent == "temporal"

    def test_get_intent_for_code(self, strategy):
        """CodeEntity maps to lookup intent."""
        intent = strategy.get_intent_for_type("CodeEntity")
        assert intent == "lookup"

    def test_get_intent_for_document(self, strategy):
        """Document maps to exploratory intent."""
        intent = strategy.get_intent_for_type("Document")
        assert intent == "exploratory"

    def test_get_intent_for_unknown(self, strategy):
        """Unknown types map to exploratory."""
        intent = strategy.get_intent_for_type("UnknownType")
        assert intent == "exploratory"

    # -------------------------------------------------------------------------
    # Search By Entity Type Tests
    # -------------------------------------------------------------------------

    def test_search_by_entity_type_filters_results(self, strategy, mock_retrieval):
        """Results are filtered to target type."""
        results = strategy.search_by_entity_type(
            query="Find events",
            target_entity_type="Event",
            top_k=10,
        )

        # Should only return Event results
        assert all(r.entity_type == "Event" for r in results)
        assert len(results) == 2  # event_1 and event_2

    def test_search_by_entity_type_uses_correct_weights(
        self, strategy, mock_retrieval, hybrid_config
    ):
        """Correct weights are passed to retrieval."""
        strategy.search_by_entity_type(
            query="Find events",
            target_entity_type="Event",
            top_k=10,
        )

        # Check that hybrid_search was called with Event weights
        call_kwargs = mock_retrieval.hybrid_search.call_args.kwargs
        assert call_kwargs["vector_weight"] == hybrid_config.event_vector_weight
        assert call_kwargs["graph_weight"] == hybrid_config.event_graph_weight
        assert call_kwargs["intent"] == "temporal"

    def test_search_by_entity_type_with_temporal_context(
        self, strategy, mock_retrieval, sample_temporal_context
    ):
        """Temporal context is passed through."""
        strategy.search_by_entity_type(
            query="Find events",
            target_entity_type="Event",
            top_k=10,
            temporal_context=sample_temporal_context,
        )

        call_kwargs = mock_retrieval.hybrid_search.call_args.kwargs
        assert call_kwargs["temporal_context"] == sample_temporal_context

    # -------------------------------------------------------------------------
    # Convenience Method Tests
    # -------------------------------------------------------------------------

    def test_search_events(self, strategy, mock_retrieval):
        """search_events convenience method."""
        results = strategy.search_events(
            query="Project meetings",
            top_k=5,
        )

        assert all(r.entity_type == "Event" for r in results)
        call_kwargs = mock_retrieval.hybrid_search.call_args.kwargs
        assert call_kwargs["intent"] == "temporal"

    def test_search_code(self, strategy, mock_retrieval):
        """search_code convenience method."""
        # Update mock to return code results
        mock_retrieval.hybrid_search.return_value = [
            create_hybrid_result("code_1", "CodeEntity"),
        ]

        results = strategy.search_code(
            query="Find authentication handler",
            top_k=5,
        )

        call_kwargs = mock_retrieval.hybrid_search.call_args.kwargs
        assert call_kwargs["intent"] == "lookup"

    def test_search_documents(self, strategy, mock_retrieval):
        """search_documents convenience method."""
        # Update mock to return document results
        mock_retrieval.hybrid_search.return_value = [
            create_hybrid_result("doc_1", "Document"),
        ]

        results = strategy.search_documents(
            query="Architecture documentation",
            top_k=5,
        )

        call_kwargs = mock_retrieval.hybrid_search.call_args.kwargs
        assert call_kwargs["intent"] == "exploratory"

    def test_search_entities(self, strategy, mock_retrieval):
        """search_entities for static entities."""
        # Update mock to return person results
        mock_retrieval.hybrid_search.return_value = [
            create_hybrid_result("person_1", "Person"),
        ]

        results = strategy.search_entities(
            query="Find John",
            entity_type="Person",
            top_k=5,
        )

        call_kwargs = mock_retrieval.hybrid_search.call_args.kwargs
        assert call_kwargs["intent"] == "lookup"


class TestEntityWeightsConfiguration:
    """Test entity weight configuration from HybridSearchConfig."""

    def test_config_weights_override_defaults(self):
        """Config weights override class defaults."""
        custom_config = HybridSearchConfig(
            event_vector_weight=0.2,
            event_graph_weight=0.8,
        )

        mock_retrieval = MagicMock(spec=SchemaAwareRetrieval)
        mock_retrieval.config = custom_config
        strategy = EntityTypeRetrievalStrategy(
            retrieval=mock_retrieval,
            config=custom_config,
        )

        weights = strategy.get_weights_for_type("Event")
        assert weights["vector"] == 0.2
        assert weights["graph"] == 0.8

    def test_all_entity_types_have_weights(self):
        """All known entity types have weight configurations."""
        expected_types = [
            "Event",
            "CodeEntity",
            "Document",
            "Person",
            "Organization",
            "Concept",
            "Chunk",
        ]

        for entity_type in expected_types:
            assert entity_type in EntityTypeRetrievalStrategy.ENTITY_WEIGHTS

"""Tests for QueryEmbeddingRouter.

Tests intent-based and content-based query routing.
"""

import pytest

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import QueryRoutingError
from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
from futurnal.search.hybrid.types import QueryEmbeddingType, TemporalQueryContext

from tests.search.hybrid.conftest import create_temporal_context


class TestQueryEmbeddingRouter:
    """Test QueryEmbeddingRouter functionality."""

    def test_init(self, mock_embedding_service, hybrid_config):
        """Router initializes correctly."""
        router = QueryEmbeddingRouter(
            embedding_service=mock_embedding_service,
            config=hybrid_config,
        )
        assert router.config == hybrid_config

    # -------------------------------------------------------------------------
    # Intent-Based Routing Tests
    # -------------------------------------------------------------------------

    def test_intent_based_routing_temporal(self, mock_embedding_service):
        """Temporal intent routes to TEMPORAL."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        query_type = router.determine_query_type(
            query="What happened?",
            intent="temporal",
        )
        assert query_type == QueryEmbeddingType.TEMPORAL

    def test_intent_based_routing_causal(self, mock_embedding_service):
        """Causal intent routes to TEMPORAL (causal needs temporal context)."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        query_type = router.determine_query_type(
            query="What caused the delay?",
            intent="causal",
        )
        assert query_type == QueryEmbeddingType.TEMPORAL

    def test_intent_based_routing_exploratory(self, mock_embedding_service):
        """Exploratory intent uses content-based detection."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Without code/document keywords, should be GENERAL
        query_type = router.determine_query_type(
            query="Tell me about the project",
            intent="exploratory",
        )
        assert query_type == QueryEmbeddingType.GENERAL

    # -------------------------------------------------------------------------
    # Content-Based Routing Tests
    # -------------------------------------------------------------------------

    def test_content_based_routing_code(self, mock_embedding_service):
        """Code keywords trigger CODE embedding type."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Various code keywords
        code_queries = [
            "Find the authentication function",
            "Show me the class definition",
            "Where is the bug in the method",
            "Implement the new feature",
        ]

        for query in code_queries:
            query_type = router.determine_query_type(
                query=query,
                intent="exploratory",
            )
            assert query_type == QueryEmbeddingType.CODE, f"Failed for: {query}"

    def test_content_based_routing_document(self, mock_embedding_service):
        """Document keywords trigger DOCUMENT embedding type."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Various document keywords
        doc_queries = [
            "Search my meeting notes",
            "Find the document about architecture",
            "Show the article on design patterns",
            "Get the paper on machine learning",
        ]

        for query in doc_queries:
            query_type = router.determine_query_type(
                query=query,
                intent="exploratory",
            )
            assert query_type == QueryEmbeddingType.DOCUMENT, f"Failed for: {query}"

    def test_content_based_routing_temporal(self, mock_embedding_service):
        """Temporal keywords in content trigger TEMPORAL."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Temporal keywords (when intent is not already temporal)
        temporal_queries = [
            "When did this happen",
            "Events before the deadline",
            "What happened yesterday",
        ]

        for query in temporal_queries:
            query_type = router.determine_query_type(
                query=query,
                intent="lookup",  # Not temporal intent
            )
            assert query_type == QueryEmbeddingType.TEMPORAL, f"Failed for: {query}"

    def test_intent_overrides_content(self, mock_embedding_service):
        """Intent-based routing takes precedence over content."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Query has code keywords but temporal intent
        query_type = router.determine_query_type(
            query="Find the function implementation",
            intent="temporal",
        )
        # Temporal intent should override code detection
        assert query_type == QueryEmbeddingType.TEMPORAL

    # -------------------------------------------------------------------------
    # Embedding Generation Tests
    # -------------------------------------------------------------------------

    def test_embed_query_general(self, mock_embedding_service):
        """General query embedding works."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        embedding = router.embed_query(
            query="Test query",
            query_type=QueryEmbeddingType.GENERAL,
        )

        assert embedding is not None
        assert len(embedding) == 768
        mock_embedding_service.embed.assert_called_once()

    def test_embed_query_temporal(self, mock_embedding_service):
        """Temporal query embedding includes temporal context."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        temporal_ctx = create_temporal_context(relation="BEFORE")

        embedding = router.embed_query(
            query="What happened before the meeting?",
            query_type=QueryEmbeddingType.TEMPORAL,
            temporal_context=temporal_ctx,
        )

        assert embedding is not None
        # Embedding service should be called with Event type
        call_kwargs = mock_embedding_service.embed.call_args.kwargs
        assert call_kwargs["entity_type"] == "Event"

    def test_embed_query_code(self, mock_embedding_service):
        """Code query embedding uses CodeEntity type."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        embedding = router.embed_query(
            query="Find the login handler",
            query_type=QueryEmbeddingType.CODE,
        )

        assert embedding is not None
        call_kwargs = mock_embedding_service.embed.call_args.kwargs
        assert call_kwargs["entity_type"] == "CodeEntity"

    def test_embed_query_with_type_detection(self, mock_embedding_service):
        """embed_query_with_type_detection combines routing and embedding."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        embedding, query_type = router.embed_query_with_type_detection(
            query="What caused the error?",
            intent="causal",
        )

        assert embedding is not None
        assert query_type == QueryEmbeddingType.TEMPORAL  # Causal -> Temporal

    def test_embed_query_failure_handling(self, mock_embedding_service):
        """QueryRoutingError raised on embedding failure."""
        mock_embedding_service.embed.side_effect = Exception("Embedding failed")

        router = QueryEmbeddingRouter(mock_embedding_service)

        with pytest.raises(QueryRoutingError, match="Failed to generate"):
            router.embed_query(
                query="Test",
                query_type=QueryEmbeddingType.GENERAL,
            )


class TestQueryEmbeddingRouterEntityTypeMapping:
    """Test entity type mapping for query types."""

    def test_query_to_entity_type_mapping(self, mock_embedding_service):
        """Verify query type to entity type mapping."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        expected_mappings = {
            QueryEmbeddingType.GENERAL: "Concept",
            QueryEmbeddingType.TEMPORAL: "Event",
            QueryEmbeddingType.CAUSAL: "Event",
            QueryEmbeddingType.CODE: "CodeEntity",
            QueryEmbeddingType.DOCUMENT: "Document",
        }

        for query_type, expected_entity_type in expected_mappings.items():
            assert router.QUERY_TO_ENTITY_TYPE[query_type] == expected_entity_type

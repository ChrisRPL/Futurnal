"""Test fixtures for Query Routing & Orchestration tests.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from futurnal.search.hybrid.types import (
    HybridSearchResult,
    QueryIntent,
    QueryPlan,
    QueryResult,
)
from futurnal.search.hybrid.routing.config import (
    LLMBackendType,
    QueryRouterLLMConfig,
)
from futurnal.search.hybrid.routing.feedback import (
    SearchQualityFeedback,
    SearchQualitySignal,
)
from futurnal.search.hybrid.routing.templates import (
    QueryTemplate,
    QueryTemplateDatabase,
)


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> QueryRouterLLMConfig:
    """Default LLM configuration for testing."""
    return QueryRouterLLMConfig(
        backend=LLMBackendType.OLLAMA,
        model="phi3:mini",  # Use fast model for tests
        timeout_seconds=2,
    )


@pytest.fixture
def hf_config() -> QueryRouterLLMConfig:
    """HuggingFace configuration for testing."""
    return QueryRouterLLMConfig(
        backend=LLMBackendType.HUGGINGFACE,
        model="microsoft/Phi-3-mini-4k-instruct",
        timeout_seconds=5,
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_ollama_available():
    """Mock Ollama availability check to return True."""
    with patch(
        "futurnal.search.hybrid.routing.intent_classifier.requests.get"
    ) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_ollama_unavailable():
    """Mock Ollama availability check to return False."""
    with patch(
        "futurnal.search.hybrid.routing.intent_classifier.requests.get"
    ) as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        yield mock_get


@pytest.fixture
def mock_ollama_generate():
    """Mock Ollama generate endpoint."""
    with patch(
        "futurnal.search.hybrid.routing.intent_classifier.requests.post"
    ) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"intent": "temporal", "confidence": 0.9, "reasoning": "Contains when keyword"}'
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_intent_classifier():
    """Mock intent classifier for testing."""
    classifier = MagicMock()
    classifier.classify_intent.return_value = {
        "intent": "temporal",
        "confidence": 0.9,
        "reasoning": "Contains temporal keywords",
    }
    return classifier


@pytest.fixture
def mock_temporal_engine():
    """Mock TemporalQueryEngine for testing."""
    engine = MagicMock()
    mock_result = MagicMock()
    mock_result.events = []
    engine.query.return_value = mock_result
    return engine


@pytest.fixture
def mock_causal_retrieval():
    """Mock CausalChainRetrieval for testing."""
    retrieval = MagicMock()
    mock_result = MagicMock()
    mock_result.causes = []
    retrieval.query.return_value = mock_result
    return retrieval


@pytest.fixture
def mock_schema_retrieval():
    """Mock SchemaAwareRetrieval for testing."""
    retrieval = MagicMock()
    retrieval.hybrid_search.return_value = [
        HybridSearchResult(
            entity_id="entity-1",
            entity_type="Document",
            vector_score=0.8,
            graph_score=0.7,
            combined_score=0.75,
            source="hybrid",
            content="Test content",
        ),
    ]
    return retrieval


@pytest.fixture
def mock_grpo_engine():
    """Mock TrainingFreeGRPO for testing."""
    grpo = MagicMock()
    grpo.update_experiential_knowledge = MagicMock()
    return grpo


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_temporal_queries() -> List[str]:
    """Sample temporal queries for testing."""
    return [
        "What happened between January and March 2024?",
        "Events before the product launch",
        "What was I doing last week?",
        "When did we start the project?",
        "Show me everything from yesterday",
    ]


@pytest.fixture
def sample_causal_queries() -> List[str]:
    """Sample causal queries for testing."""
    return [
        "What caused the project delay?",
        "Why did the meeting lead to this decision?",
        "What led to the product launch?",
        "Why did the system fail?",
        "What were the consequences of the change?",
    ]


@pytest.fixture
def sample_lookup_queries() -> List[str]:
    """Sample lookup queries for testing."""
    return [
        "What is machine learning?",
        "Who is John Smith?",
        "Find the project documentation",
        "Get the API endpoint details",
        "What does the config file contain?",
    ]


@pytest.fixture
def sample_exploratory_queries() -> List[str]:
    """Sample exploratory queries for testing."""
    return [
        "Tell me about the project",
        "What do I know about cloud computing?",
        "Explore the customer feedback",
        "Show me related topics",
        "What connections exist?",
    ]


@pytest.fixture
def sample_polish_queries() -> List[str]:
    """Sample Polish language queries for testing."""
    return [
        "Co to jest machine learning?",
        "Kiedy było spotkanie projektowe?",
        "Jak działa ten algorytm?",
        "Dlaczego projekt się opóźnił?",
        "Gdzie jest dokumentacja?",
    ]


@pytest.fixture
def sample_query_plan() -> QueryPlan:
    """Sample QueryPlan for testing."""
    return QueryPlan(
        query_id="test-query-123",
        original_query="What happened in January 2024?",
        intent=QueryIntent.TEMPORAL,
        intent_confidence=0.9,
        primary_strategy="temporal_query",
        secondary_strategy="hybrid_retrieval",
        weights={"temporal": 0.7, "hybrid": 0.3},
        estimated_latency_ms=300,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_query_result() -> QueryResult:
    """Sample QueryResult for testing."""
    return QueryResult(
        result_id="result-123",
        query_id="test-query-123",
        entities=[
            {
                "id": "entity-1",
                "type": "Event",
                "content": "Project kickoff",
                "score": 0.85,
            },
            {
                "id": "entity-2",
                "type": "Event",
                "content": "Team meeting",
                "score": 0.75,
            },
        ],
        relationships=[],
        temporal_context={"query_type": "temporal"},
        relevance_scores={"combined_relevance": 0.80},
        provenance=["doc-1", "doc-2"],
        latency_ms=150,
    )


@pytest.fixture
def sample_quality_signals() -> List[SearchQualitySignal]:
    """Sample quality signals for testing."""
    return [
        SearchQualitySignal(
            query_id="query-1",
            signal_type="click",
            signal_value=1.0,
            context={"entity_id": "entity-1"},
        ),
        SearchQualitySignal(
            query_id="query-1",
            signal_type="click",
            signal_value=1.0,
            context={"entity_id": "entity-2"},
        ),
        SearchQualitySignal(
            query_id="query-2",
            signal_type="refinement",
            signal_value=-0.5,
            context={"new_query": "more specific query"},
        ),
        SearchQualitySignal(
            query_id="query-3",
            signal_type="no_results",
            signal_value=-1.0,
            context={},
        ),
    ]


# =============================================================================
# Template Fixtures
# =============================================================================


@pytest.fixture
def template_database() -> QueryTemplateDatabase:
    """Initialized template database for testing."""
    return QueryTemplateDatabase()


@pytest.fixture
def sample_template() -> QueryTemplate:
    """Sample QueryTemplate for testing."""
    return QueryTemplate(
        template_id="test_template_v1",
        name="Test Template",
        intent_type=QueryIntent.TEMPORAL,
        pattern="# Test Pattern\n\n1. Step one\n2. Step two",
        version=1,
        success_rate=0.8,
    )


# =============================================================================
# Integration Fixtures
# =============================================================================


@pytest.fixture
def feedback_with_grpo(mock_grpo_engine) -> SearchQualityFeedback:
    """SearchQualityFeedback with mock GRPO engine."""
    return SearchQualityFeedback(grpo_engine=mock_grpo_engine)


@pytest.fixture
def feedback_without_grpo() -> SearchQualityFeedback:
    """SearchQualityFeedback without GRPO engine."""
    return SearchQualityFeedback(grpo_engine=None)

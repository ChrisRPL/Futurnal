"""Integration tests for Chat pipeline.

Step 03: Chat Interface & Conversational AI
Production Plan Reference:
docs/phase-1/implementation-steps/03-chat-interface-conversational.md

Research Foundation:
- ProPerSim (2509.21730v1): Multi-turn context carryover
- Causal-Copilot (2504.13263v2): Confidence scoring

These tests verify the full pipeline from user query through search
to answer generation, with mocked external services (Ollama, Neo4j, ChromaDB).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.chat.models import ChatMessage, ChatSession, SessionStorage
from futurnal.chat.service import ChatService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollama_pool() -> MagicMock:
    """Mock OllamaConnectionPool for integration tests."""
    pool = MagicMock()
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    pool.is_initialized = True
    pool.is_healthy = True

    # Realistic response simulation
    pool.generate = AsyncMock(
        return_value=(
            "Based on your knowledge graph, Python is a high-level programming "
            "language created by Guido van Rossum. It is known for its clean syntax "
            "and extensive standard library. [Source: docs/python-guide.md]"
        )
    )

    async def mock_stream_generate(*args, **kwargs):
        """Simulate token-by-token streaming."""
        response = (
            "Python is a versatile programming language used for "
            "web development, data science, and automation."
        )
        # Simulate realistic streaming delay
        for word in response.split():
            await asyncio.sleep(0.01)  # Small delay between tokens
            yield word + " "

    pool.stream_generate = mock_stream_generate

    return pool


@pytest.fixture
def mock_search_results() -> List[Dict[str, Any]]:
    """Realistic search results for integration tests."""
    return [
        {
            "content": "Python is a high-level, general-purpose programming language. "
                       "Its design philosophy emphasizes code readability.",
            "metadata": {
                "source": "docs/python-guide.md",
                "label": "Python Guide",
                "entity_id": "entity_python_001",
            },
            "score": 0.92,
            "confidence": 0.9,
            "entity_id": "entity_python_001",
            "graph_context": {
                "related_entities": [
                    {"id": "entity_guido", "name": "Guido van Rossum"},
                    {"id": "entity_programming", "name": "Programming Languages"},
                ],
            },
        },
        {
            "content": "Guido van Rossum began working on Python in the late 1980s. "
                       "The language was first released in 1991.",
            "metadata": {
                "source": "notes/tech-history.md",
                "entity_id": "entity_history_001",
            },
            "score": 0.85,
            "confidence": 0.82,
        },
        {
            "content": "Python supports multiple programming paradigms including "
                       "procedural, object-oriented, and functional programming.",
            "metadata": {
                "source": "docs/python-guide.md",
                "label": "Python Guide",
            },
            "score": 0.78,
            "confidence": 0.75,
        },
    ]


@pytest.fixture
def mock_search_api(mock_search_results: List[Dict[str, Any]]) -> MagicMock:
    """Mock HybridSearchAPI for integration tests."""
    api = MagicMock()
    api.search = AsyncMock(return_value=mock_search_results)
    return api


@pytest.fixture
def integration_chat_service(
    mock_search_api: MagicMock,
    mock_ollama_pool: MagicMock,
    tmp_path,
) -> ChatService:
    """Create ChatService configured for integration testing."""
    from pathlib import Path

    # Create mock answer generator with the pool
    mock_generator = MagicMock()
    mock_generator.initialize = AsyncMock()
    mock_generator.close = AsyncMock()
    mock_generator.config = MagicMock()
    mock_generator.config.model_name = "llama3.1:8b-instruct-q4_0"
    mock_generator.config.temperature = 0.4
    mock_generator.config.max_tokens = 600
    mock_generator._pool = mock_ollama_pool

    # Use temp directory for session storage
    storage = SessionStorage(base_path=tmp_path / "test_sessions")

    return ChatService(
        search_api=mock_search_api,
        answer_generator=mock_generator,
        storage=storage,
    )


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


class TestChatPipelineIntegration:
    """End-to-end integration tests for chat pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_chat(
        self,
        integration_chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test full pipeline: query → search → answer → response."""
        await integration_chat_service.initialize()

        response = await integration_chat_service.chat(
            session_id="integration-test-1",
            message="What is Python and who created it?",
        )

        # Verify complete pipeline
        assert isinstance(response, ChatMessage)
        assert response.role == "assistant"
        assert len(response.content) > 50  # Substantive response
        assert len(response.sources) > 0  # Has sources
        assert response.confidence > 0.5  # Reasonable confidence

        # Verify search was called with query
        mock_search_api.search.assert_called_once()
        call_args = mock_search_api.search.call_args
        assert "Python" in call_args.kwargs.get("query", "")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_pipeline(
        self,
        integration_chat_service: ChatService,
    ):
        """Test ProPerSim: full multi-turn conversation pipeline."""
        await integration_chat_service.initialize()
        session_id = "multi-turn-test"

        # Turn 1: Initial question
        response1 = await integration_chat_service.chat(
            session_id=session_id,
            message="Tell me about Python programming.",
        )

        assert response1.role == "assistant"

        # Turn 2: Follow-up question
        response2 = await integration_chat_service.chat(
            session_id=session_id,
            message="Who created it?",
        )

        assert response2.role == "assistant"

        # Turn 3: Another follow-up
        response3 = await integration_chat_service.chat(
            session_id=session_id,
            message="When was it first released?",
        )

        assert response3.role == "assistant"

        # Verify session has full history
        session = integration_chat_service.get_session(session_id)
        assert len(session.messages) == 6  # 3 user + 3 assistant

    @pytest.mark.asyncio
    async def test_context_entity_focus_pipeline(
        self,
        integration_chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test 'Ask about this' pipeline with entity context."""
        await integration_chat_service.initialize()

        response = await integration_chat_service.chat(
            session_id="entity-focus-test",
            message="Tell me more about this entity.",
            context_entity_id="entity_python_001",
        )

        # Verify entity filter was passed to search
        call_kwargs = mock_search_api.search.call_args.kwargs
        assert call_kwargs.get("filters") == {"entity_id": "entity_python_001"}

        # Response should still be generated
        assert isinstance(response, ChatMessage)
        assert len(response.content) > 0


# ---------------------------------------------------------------------------
# Response Quality Tests
# ---------------------------------------------------------------------------


class TestResponseQuality:
    """Tests for response quality and accuracy."""

    @pytest.mark.asyncio
    async def test_response_includes_sources(
        self,
        integration_chat_service: ChatService,
    ):
        """Test that responses include source citations."""
        await integration_chat_service.initialize()

        response = await integration_chat_service.chat(
            session_id="source-test",
            message="What is Python?",
        )

        # Should have multiple sources from search results
        assert len(response.sources) >= 1
        # At least one labeled source
        source_names = [s.lower() for s in response.sources]
        assert any("python" in s or "guide" in s for s in source_names)

    @pytest.mark.asyncio
    async def test_response_includes_entity_refs(
        self,
        integration_chat_service: ChatService,
    ):
        """Test that responses include entity references."""
        await integration_chat_service.initialize()

        response = await integration_chat_service.chat(
            session_id="entity-ref-test",
            message="Tell me about Python.",
        )

        # Should have entity references from graph context
        assert len(response.entity_refs) > 0

    @pytest.mark.asyncio
    async def test_confidence_reflects_result_quality(
        self,
        integration_chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test Causal-Copilot: confidence scoring accuracy."""
        await integration_chat_service.initialize()

        # Good search results = high confidence
        response_good = await integration_chat_service.chat(
            session_id="confidence-test-good",
            message="What is Python?",
        )
        assert response_good.confidence >= 0.7

        # Empty search results = low confidence
        mock_search_api.search.return_value = []
        response_empty = await integration_chat_service.chat(
            session_id="confidence-test-empty",
            message="What is XYZ123?",
        )
        assert response_empty.confidence < 0.5


# ---------------------------------------------------------------------------
# Performance Tests (Quality Gates)
# ---------------------------------------------------------------------------


class TestPerformanceQualityGates:
    """Performance tests as defined in quality gates."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_response_latency_under_3s(
        self,
        integration_chat_service: ChatService,
    ):
        """Quality Gate: Full response under 3 seconds."""
        await integration_chat_service.initialize()

        start_time = time.time()

        await integration_chat_service.chat(
            session_id="latency-test",
            message="What is Python?",
        )

        elapsed = time.time() - start_time

        # Quality gate: <3s for full response
        assert elapsed < 3.0, f"Response took {elapsed:.2f}s, expected <3s"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_streaming_first_token_latency(
        self,
        integration_chat_service: ChatService,
    ):
        """Quality Gate: First token under 500ms for streaming."""
        await integration_chat_service.initialize()

        start_time = time.time()
        first_token_time = None

        async for token in integration_chat_service.stream_chat(
            session_id="stream-latency-test",
            message="What is Python?",
        ):
            if first_token_time is None:
                first_token_time = time.time() - start_time
            break  # Only need first token

        # Quality gate: <500ms for first token
        assert first_token_time is not None
        assert first_token_time < 0.5, (
            f"First token at {first_token_time*1000:.0f}ms, expected <500ms"
        )


# ---------------------------------------------------------------------------
# Session Persistence Tests
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    """Tests for session persistence across service restarts."""

    @pytest.mark.asyncio
    async def test_session_survives_service_restart(
        self,
        mock_search_api: MagicMock,
        mock_ollama_pool: MagicMock,
        tmp_path,
    ):
        """Test that sessions persist across service instances."""
        from pathlib import Path

        # Use pytest tmp_path for session storage
        session_path = tmp_path / "persist_sessions"

        # First service instance
        mock_generator = MagicMock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator.config = MagicMock()
        mock_generator.config.model_name = "test-model"
        mock_generator.config.temperature = 0.4
        mock_generator.config.max_tokens = 600
        mock_generator._pool = mock_ollama_pool

        storage = SessionStorage(base_path=session_path)
        service1 = ChatService(
            search_api=mock_search_api,
            answer_generator=mock_generator,
            storage=storage,
        )

        await service1.initialize()
        await service1.chat("persist-test", "Hello, this is message 1")
        await service1.close()

        # Second service instance (simulating restart)
        storage2 = SessionStorage(base_path=session_path)
        service2 = ChatService(
            search_api=mock_search_api,
            answer_generator=mock_generator,
            storage=storage2,
        )

        await service2.initialize()

        # Session should be loaded from storage
        session = service2.get_session("persist-test")

        assert session is not None
        assert len(session.messages) == 2  # 1 user + 1 assistant
        assert "message 1" in session.messages[0].content

        await service2.close()


# ---------------------------------------------------------------------------
# Streaming Pipeline Tests
# ---------------------------------------------------------------------------


class TestStreamingPipeline:
    """Tests for streaming chat pipeline."""

    @pytest.mark.asyncio
    async def test_streaming_full_pipeline(
        self,
        integration_chat_service: ChatService,
    ):
        """Test full streaming pipeline from query to tokens."""
        await integration_chat_service.initialize()

        tokens = []
        async for token in integration_chat_service.stream_chat(
            session_id="stream-pipeline-test",
            message="What is Python?",
        ):
            tokens.append(token)

        # Should have multiple tokens
        assert len(tokens) > 5

        # Combined should form coherent response
        full_response = "".join(tokens)
        assert len(full_response) > 20

    @pytest.mark.asyncio
    async def test_streaming_saves_complete_message(
        self,
        integration_chat_service: ChatService,
    ):
        """Test that streamed message is fully saved to session."""
        await integration_chat_service.initialize()
        session_id = "stream-save-test"

        # Consume all tokens
        full_response = []
        async for token in integration_chat_service.stream_chat(
            session_id=session_id,
            message="Tell me about Python.",
        ):
            full_response.append(token)

        # Verify session was updated with complete message
        session = integration_chat_service.get_session(session_id)
        assert len(session.messages) == 2  # user + assistant

        assistant_msg = session.messages[1]
        assert assistant_msg.role == "assistant"
        # Saved message should match streamed content
        assert "".join(full_response).strip() == assistant_msg.content.strip()


# ---------------------------------------------------------------------------
# Error Recovery Tests
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """Tests for error handling and recovery in pipeline."""

    @pytest.mark.asyncio
    async def test_recovers_from_search_failure(
        self,
        integration_chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test graceful recovery when search fails."""
        await integration_chat_service.initialize()

        # Make search fail
        mock_search_api.search = AsyncMock(side_effect=Exception("Network error"))

        response = await integration_chat_service.chat(
            session_id="search-failure-test",
            message="What is Python?",
        )

        # Should still return a response
        assert isinstance(response, ChatMessage)
        assert response.sources == []  # No sources from failed search

    @pytest.mark.asyncio
    async def test_recovers_from_generation_failure(
        self,
        integration_chat_service: ChatService,
        mock_ollama_pool: MagicMock,
    ):
        """Test graceful recovery when generation fails."""
        await integration_chat_service.initialize()

        # Make generation fail
        mock_ollama_pool.generate = AsyncMock(
            side_effect=Exception("Ollama unavailable")
        )

        response = await integration_chat_service.chat(
            session_id="gen-failure-test",
            message="What is Python?",
        )

        # Should return error message, not crash
        assert isinstance(response, ChatMessage)
        assert "couldn't" in response.content.lower()
        assert response.confidence == 0.0

"""Tests for ChatService module.

Step 03: Chat Interface & Conversational AI
Production Plan Reference:
docs/phase-1/implementation-steps/03-chat-interface-conversational.md

Research Foundation:
- ProPerSim (2509.21730v1): Proactive + personalized AI, multi-turn context
- Causal-Copilot (2504.13263v2): Natural language exploration, confidence scoring
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.chat.models import ChatMessage, ChatSession, SessionStorage
from futurnal.chat.service import ChatService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_api() -> MagicMock:
    """Mock HybridSearchAPI for testing."""
    api = MagicMock()
    api.search = AsyncMock(return_value=[
        {
            "content": "Python is a programming language known for its simplicity.",
            "metadata": {"source": "docs/python.md", "label": "Python Guide"},
            "score": 0.92,
            "confidence": 0.9,
        },
        {
            "content": "Python was created by Guido van Rossum in 1991.",
            "metadata": {"source": "docs/history.md"},
            "score": 0.85,
            "confidence": 0.8,
        },
    ])
    return api


@pytest.fixture
def mock_answer_generator() -> MagicMock:
    """Mock AnswerGenerator for testing."""
    generator = MagicMock()
    generator.initialize = AsyncMock()
    generator.close = AsyncMock()
    generator.config = MagicMock()
    generator.config.model_name = "llama3.1:8b-instruct-q4_0"
    generator.config.temperature = 0.4
    generator.config.max_tokens = 600

    # Mock the pool for direct generation
    mock_pool = MagicMock()
    mock_pool.generate = AsyncMock(
        return_value="Python is a programming language. [Source: docs/python.md]"
    )

    async def mock_stream_generate(*args, **kwargs):
        tokens = ["Python ", "is ", "a ", "programming ", "language."]
        for token in tokens:
            yield token

    mock_pool.stream_generate = mock_stream_generate
    generator._pool = mock_pool

    return generator


@pytest.fixture
def mock_storage() -> MagicMock:
    """Mock SessionStorage for testing."""
    storage = MagicMock(spec=SessionStorage)
    storage.load.return_value = None
    storage.save.return_value = None
    storage.delete.return_value = True
    storage.get_all_sessions.return_value = []
    return storage


@pytest.fixture
def chat_service(
    mock_search_api: MagicMock,
    mock_answer_generator: MagicMock,
    mock_storage: MagicMock,
) -> ChatService:
    """Create ChatService with mocked dependencies."""
    return ChatService(
        search_api=mock_search_api,
        answer_generator=mock_answer_generator,
        storage=mock_storage,
        context_window_size=10,
    )


# ---------------------------------------------------------------------------
# ChatService Initialization Tests
# ---------------------------------------------------------------------------


class TestChatServiceInitialization:
    """Tests for ChatService initialization."""

    @pytest.mark.asyncio
    async def test_initialization_success(
        self,
        chat_service: ChatService,
        mock_answer_generator: MagicMock,
    ):
        """Test successful initialization."""
        await chat_service.initialize()

        assert chat_service._initialized is True
        mock_answer_generator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialization_noop(
        self,
        chat_service: ChatService,
        mock_answer_generator: MagicMock,
    ):
        """Test that double initialization is a no-op."""
        await chat_service.initialize()
        await chat_service.initialize()

        # Should only initialize once
        mock_answer_generator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_saves_sessions(
        self,
        chat_service: ChatService,
        mock_storage: MagicMock,
    ):
        """Test that close saves all active sessions."""
        await chat_service.initialize()

        # Create some sessions
        chat_service._sessions["session-1"] = ChatSession(id="session-1")
        chat_service._sessions["session-2"] = ChatSession(id="session-2")

        await chat_service.close()

        assert chat_service._initialized is False
        assert mock_storage.save.call_count == 2


# ---------------------------------------------------------------------------
# Chat Method Tests - ProPerSim Multi-turn Context
# ---------------------------------------------------------------------------


class TestChatMethod:
    """Tests for the main chat method."""

    @pytest.mark.asyncio
    async def test_basic_chat_response(
        self,
        chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test basic chat generates response."""
        await chat_service.initialize()

        response = await chat_service.chat("session-1", "What is Python?")

        assert isinstance(response, ChatMessage)
        assert response.role == "assistant"
        assert len(response.content) > 0
        mock_search_api.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_sources(
        self,
        chat_service: ChatService,
    ):
        """Test that chat returns source citations."""
        await chat_service.initialize()

        response = await chat_service.chat("session-1", "What is Python?")

        # Should have sources from search results
        assert len(response.sources) > 0
        assert "Python Guide" in response.sources

    @pytest.mark.asyncio
    async def test_chat_returns_confidence(
        self,
        chat_service: ChatService,
    ):
        """Test Causal-Copilot: confidence scoring in responses."""
        await chat_service.initialize()

        response = await chat_service.chat("session-1", "What is Python?")

        # Should have confidence between 0 and 1
        assert 0.0 <= response.confidence <= 1.0
        # With good search results, should have reasonable confidence
        assert response.confidence > 0.5

    @pytest.mark.asyncio
    async def test_multi_turn_context_carryover(
        self,
        chat_service: ChatService,
    ):
        """Test ProPerSim: conversation history influences responses."""
        await chat_service.initialize()

        # First message
        await chat_service.chat("session-1", "Tell me about Python.")

        # Second message - should have context from first
        await chat_service.chat("session-1", "What about its history?")

        # Check session has both messages
        session = chat_service.get_session("session-1")
        assert session is not None
        assert len(session.messages) == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_context_entity_focus(
        self,
        chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test 'Ask about this' feature with entity context."""
        await chat_service.initialize()

        response = await chat_service.chat(
            session_id="session-1",
            message="Tell me more about this.",
            context_entity_id="entity_123",
        )

        # Should have called search with filter
        call_kwargs = mock_search_api.search.call_args.kwargs
        assert call_kwargs.get("filters") == {"entity_id": "entity_123"}

    @pytest.mark.asyncio
    async def test_chat_auto_initializes(
        self,
        chat_service: ChatService,
    ):
        """Test that chat auto-initializes if needed."""
        # Don't call initialize explicitly
        response = await chat_service.chat("session-1", "Hello")

        assert chat_service._initialized is True
        assert isinstance(response, ChatMessage)


# ---------------------------------------------------------------------------
# Context Building Tests
# ---------------------------------------------------------------------------


class TestContextBuilding:
    """Tests for conversation context building."""

    def test_build_conversation_context_empty_session(
        self,
        chat_service: ChatService,
    ):
        """Test context building with empty session."""
        session = ChatSession(id="test")
        session.add_message(ChatMessage(role="user", content="Hello"))

        context = chat_service._build_conversation_context(session)

        # Should return empty with only one message
        assert context == ""

    def test_build_conversation_context_with_history(
        self,
        chat_service: ChatService,
    ):
        """Test context building with conversation history."""
        session = ChatSession(id="test")
        session.add_message(ChatMessage(role="user", content="What is Python?"))
        session.add_message(ChatMessage(
            role="assistant",
            content="Python is a programming language.",
        ))
        session.add_message(ChatMessage(role="user", content="Tell me more."))

        context = chat_service._build_conversation_context(session)

        assert "Previous conversation:" in context
        assert "User:" in context
        assert "Assistant:" in context
        assert "Python" in context

    def test_context_truncates_long_messages(
        self,
        chat_service: ChatService,
    ):
        """Test that long messages are truncated in context."""
        session = ChatSession(id="test")
        long_content = "A" * 500  # 500 chars
        session.add_message(ChatMessage(role="user", content=long_content))
        session.add_message(ChatMessage(role="assistant", content="Response"))
        session.add_message(ChatMessage(role="user", content="Next"))

        context = chat_service._build_conversation_context(session)

        # Should be truncated to 300 chars with "..."
        assert "..." in context


# ---------------------------------------------------------------------------
# Source and Entity Extraction Tests
# ---------------------------------------------------------------------------


class TestExtraction:
    """Tests for source and entity extraction."""

    def test_extract_sources_from_results(
        self,
        chat_service: ChatService,
    ):
        """Test source extraction from search results."""
        results = [
            {"metadata": {"source": "doc1.md", "label": "Document 1"}},
            {"metadata": {"source": "doc2.md"}},  # No label
            {"metadata": {"source": "doc1.md", "label": "Document 1"}},  # Duplicate
        ]

        sources = chat_service._extract_sources_from_results(results)

        assert len(sources) == 2  # Unique sources
        assert "Document 1" in sources
        assert "doc2.md" in sources

    def test_extract_entity_refs(
        self,
        chat_service: ChatService,
    ):
        """Test entity reference extraction."""
        results = [
            {"entity_id": "entity_1"},
            {"metadata": {"entity_id": "entity_2"}},
            {
                "graph_context": {
                    "related_entities": [
                        {"id": "entity_3", "name": "Python"},
                    ]
                }
            },
        ]

        refs = chat_service._extract_entity_refs(results)

        assert "entity_1" in refs
        assert "entity_2" in refs
        assert "entity_3" in refs


# ---------------------------------------------------------------------------
# Confidence Calculation Tests - Causal-Copilot
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """Tests for confidence scoring per Causal-Copilot research."""

    def test_no_results_low_confidence(
        self,
        chat_service: ChatService,
    ):
        """Test that empty results give low confidence."""
        confidence = chat_service._calculate_confidence([])

        assert confidence == 0.3  # Low confidence

    def test_high_score_results_high_confidence(
        self,
        chat_service: ChatService,
    ):
        """Test that high-scoring results give high confidence."""
        results = [
            {"score": 0.95, "confidence": 0.95},
            {"score": 0.90, "confidence": 0.90},
            {"score": 0.85, "confidence": 0.85},
        ]

        confidence = chat_service._calculate_confidence(results)

        assert confidence > 0.8  # High confidence

    def test_confidence_bounded(
        self,
        chat_service: ChatService,
    ):
        """Test that confidence is always between 0 and 1."""
        # Very low scores
        low_results = [{"score": 0.1, "confidence": 0.1}]
        low_conf = chat_service._calculate_confidence(low_results)
        assert 0.0 <= low_conf <= 1.0

        # Very high scores
        high_results = [{"score": 1.0, "confidence": 1.0}]
        high_conf = chat_service._calculate_confidence(high_results)
        assert 0.0 <= high_conf <= 1.0


# ---------------------------------------------------------------------------
# Session Management Tests
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Tests for session management functionality."""

    @pytest.mark.asyncio
    async def test_get_session_from_cache(
        self,
        chat_service: ChatService,
    ):
        """Test getting session from memory cache."""
        session = ChatSession(id="test-session")
        chat_service._sessions["test-session"] = session

        result = chat_service.get_session("test-session")

        assert result is session

    def test_get_session_from_storage(
        self,
        chat_service: ChatService,
        mock_storage: MagicMock,
    ):
        """Test getting session from storage."""
        stored_session = ChatSession(id="stored-session")
        mock_storage.load.return_value = stored_session

        result = chat_service.get_session("stored-session")

        assert result is stored_session
        mock_storage.load.assert_called_with("stored-session")

    def test_get_session_history(
        self,
        chat_service: ChatService,
    ):
        """Test getting message history for session."""
        session = ChatSession(id="test")
        session.add_message(ChatMessage(role="user", content="Hello"))
        session.add_message(ChatMessage(role="assistant", content="Hi there!"))
        chat_service._sessions["test"] = session

        history = chat_service.get_session_history("test")

        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"

    def test_get_session_history_nonexistent(
        self,
        chat_service: ChatService,
    ):
        """Test getting history for nonexistent session."""
        history = chat_service.get_session_history("nonexistent")

        assert history == []

    def test_list_sessions(
        self,
        chat_service: ChatService,
        mock_storage: MagicMock,
    ):
        """Test listing all sessions."""
        sessions = [
            ChatSession(id="session-1"),
            ChatSession(id="session-2"),
        ]
        sessions[0].add_message(ChatMessage(role="user", content="Hello"))
        mock_storage.get_all_sessions.return_value = sessions

        result = chat_service.list_sessions()

        assert len(result) == 2
        assert result[0]["id"] == "session-1"
        assert "createdAt" in result[0]

    def test_delete_session(
        self,
        chat_service: ChatService,
        mock_storage: MagicMock,
    ):
        """Test deleting a session."""
        chat_service._sessions["to-delete"] = ChatSession(id="to-delete")

        result = chat_service.delete_session("to-delete")

        assert result is True
        assert "to-delete" not in chat_service._sessions
        mock_storage.delete.assert_called_with("to-delete")

    @pytest.mark.asyncio
    async def test_clear_session(
        self,
        chat_service: ChatService,
        mock_storage: MagicMock,
    ):
        """Test clearing session messages."""
        await chat_service.initialize()
        await chat_service.chat("clear-test", "Hello")

        chat_service.clear_session("clear-test")

        session = chat_service.get_session("clear-test")
        assert len(session.messages) == 0


# ---------------------------------------------------------------------------
# Stream Chat Tests
# ---------------------------------------------------------------------------


class TestStreamChat:
    """Tests for streaming chat responses."""

    @pytest.mark.asyncio
    async def test_stream_chat_yields_tokens(
        self,
        chat_service: ChatService,
    ):
        """Test that stream_chat yields individual tokens."""
        await chat_service.initialize()

        tokens = []
        async for token in chat_service.stream_chat("session-1", "What is Python?"):
            tokens.append(token)

        assert len(tokens) > 0
        # Tokens should combine to form the response
        full_response = "".join(tokens)
        assert "Python" in full_response

    @pytest.mark.asyncio
    async def test_stream_chat_saves_message(
        self,
        chat_service: ChatService,
        mock_storage: MagicMock,
    ):
        """Test that streamed response is saved to session."""
        await chat_service.initialize()

        # Consume all tokens
        async for _ in chat_service.stream_chat("session-stream", "Hello"):
            pass

        # Session should have been saved
        mock_storage.save.assert_called()


# ---------------------------------------------------------------------------
# Option B Compliance Tests
# ---------------------------------------------------------------------------


class TestOptionBCompliance:
    """Tests to verify Option B compliance - Ghost model FROZEN."""

    @pytest.mark.asyncio
    async def test_ghost_model_frozen_no_parameter_updates(
        self,
        chat_service: ChatService,
        mock_answer_generator: MagicMock,
    ):
        """Option B: Verify no model parameter updates during chat."""
        await chat_service.initialize()

        # Multiple chat interactions
        for i in range(3):
            await chat_service.chat("session-1", f"Question {i}")

        # Option B Compliance: Ghost model FROZEN
        # Verify only inference methods were called, not training methods
        # The pool.generate should be called for inference
        assert mock_answer_generator._pool.generate.called

        # Verify the service only uses inference, not training
        # This test ensures the architecture doesn't include training paths
        service_methods = [m for m in dir(chat_service) if not m.startswith('_')]
        training_methods = ['train', 'fine_tune', 'update_weights', 'learn']
        for method in training_methods:
            assert method not in service_methods, f"Service should not have {method} method"

    @pytest.mark.asyncio
    async def test_learning_via_context_not_weights(
        self,
        chat_service: ChatService,
    ):
        """Option B: Learning happens via context (token priors), not weight updates."""
        await chat_service.initialize()

        # First interaction
        await chat_service.chat("session-1", "My favorite programming language is Python.")

        # Second interaction should use context from first
        # but NOT update model weights
        session = chat_service.get_session("session-1")

        # Verify learning is via conversation context, not model updates
        assert len(session.messages) == 2  # Context stored in session
        context = chat_service._build_conversation_context(session)
        assert "Python" in context  # Knowledge preserved in context


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in ChatService."""

    @pytest.mark.asyncio
    async def test_search_api_unavailable(
        self,
        mock_answer_generator: MagicMock,
        mock_storage: MagicMock,
    ):
        """Test graceful handling when search API is unavailable."""
        service = ChatService(
            search_api=None,
            answer_generator=mock_answer_generator,
            storage=mock_storage,
        )
        await service.initialize()

        response = await service.chat("session-1", "Hello")

        # Should still return a response
        assert isinstance(response, ChatMessage)

    @pytest.mark.asyncio
    async def test_search_error_handled(
        self,
        chat_service: ChatService,
        mock_search_api: MagicMock,
    ):
        """Test handling of search errors."""
        await chat_service.initialize()
        mock_search_api.search = AsyncMock(side_effect=Exception("Search failed"))

        response = await chat_service.chat("session-1", "Hello")

        # Should still return a response (with empty sources)
        assert isinstance(response, ChatMessage)
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_generation_error_handled(
        self,
        chat_service: ChatService,
        mock_answer_generator: MagicMock,
    ):
        """Test handling of generation errors."""
        await chat_service.initialize()
        mock_answer_generator._pool.generate = AsyncMock(
            side_effect=Exception("Generation failed")
        )

        response = await chat_service.chat("session-1", "Hello")

        # Should return error message
        assert "couldn't generate" in response.content.lower()
        assert response.confidence == 0.0

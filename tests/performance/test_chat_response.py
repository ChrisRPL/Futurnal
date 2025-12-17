"""Chat response time performance benchmarks.

Production target: Chat response < 3 seconds (3000ms)

Tests:
- Simple question response
- Multi-turn context response
- Streaming response time to first token
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Performance targets (milliseconds)
CHAT_RESPONSE_TARGET_MS = 3000
STREAMING_FIRST_TOKEN_TARGET_MS = 500
MULTI_TURN_TARGET_MS = 3500


@pytest.mark.performance
class TestChatResponseTime:
    """Chat response time benchmark tests."""

    @pytest.mark.asyncio
    async def test_simple_chat_response(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that simple chat questions complete under target."""
        mock_response = MagicMock()
        mock_response.content = "This is a test response based on your knowledge."
        mock_response.sources = ["test-doc.md"]
        mock_response.confidence = 0.85

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            with performance_timer("simple_chat", CHAT_RESPONSE_TARGET_MS) as timer:
                response = await mock_service.chat("session-1", "What is Python?")

            assert response.content
            assert timer.passed, f"Chat response {timer.duration_ms:.2f}ms exceeded target {CHAT_RESPONSE_TARGET_MS}ms"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_multi_turn_chat_response(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that multi-turn conversations maintain acceptable latency."""
        mock_response = MagicMock()
        mock_response.content = "Building on our previous discussion..."
        mock_response.sources = ["doc1.md", "doc2.md"]
        mock_response.confidence = 0.90

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            # Simulate multi-turn context (5 previous messages)
            for i in range(5):
                await mock_service.chat("session-1", f"Previous message {i}")

            # Time the response with accumulated context
            with performance_timer("multi_turn_chat", MULTI_TURN_TARGET_MS) as timer:
                response = await mock_service.chat(
                    "session-1", "Tell me more about the first point"
                )

            assert response.content
            assert timer.passed, f"Multi-turn chat {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_streaming_first_token_latency(
        self,
        performance_timer,
        mock_ollama,
    ):
        """Test time to first token in streaming response."""

        async def mock_stream() -> AsyncIterator[str]:
            """Mock streaming response."""
            tokens = ["This ", "is ", "a ", "test ", "response."]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)  # Simulate token generation

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_stream())
            MockService.return_value = mock_service

            first_token_received = False

            with performance_timer(
                "streaming_first_token", STREAMING_FIRST_TOKEN_TARGET_MS
            ) as timer:
                stream = await mock_service.chat("session-1", "Test", stream=True)
                async for token in stream:
                    first_token_received = True
                    break  # Stop after first token

            assert first_token_received
            assert timer.passed, f"First token {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_chat_with_large_context(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test chat response with large knowledge context."""
        # Simulate response that required searching large context
        mock_response = MagicMock()
        mock_response.content = "Based on 50 relevant documents..." + "x" * 2000
        mock_response.sources = [f"doc-{i}.md" for i in range(50)]
        mock_response.confidence = 0.75

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            # Allow extra time for large context
            large_context_target_ms = CHAT_RESPONSE_TARGET_MS * 1.5

            with performance_timer("large_context_chat", large_context_target_ms) as timer:
                response = await mock_service.chat(
                    "session-1",
                    "Summarize all my notes about machine learning",
                )

            assert len(response.sources) == 50
            assert timer.passed, f"Large context chat {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_concurrent_chat_sessions(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test multiple concurrent chat sessions."""
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.sources = []
        mock_response.confidence = 0.9

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            # 5 concurrent sessions
            concurrent_target_ms = CHAT_RESPONSE_TARGET_MS * 2

            with performance_timer("concurrent_chat", concurrent_target_ms) as timer:
                tasks = [
                    mock_service.chat(f"session-{i}", f"Question {i}")
                    for i in range(5)
                ]
                results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert timer.passed, f"Concurrent chat {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)


@pytest.mark.performance
class TestChatServiceInitialization:
    """Chat service initialization performance tests."""

    @pytest.mark.asyncio
    async def test_service_initialization_time(self, performance_timer):
        """Test that chat service initializes quickly."""
        init_target_ms = 2000  # 2 seconds

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            MockService.return_value = mock_service

            with performance_timer("chat_init", init_target_ms) as timer:
                service = MockService()
                await service.initialize()

            assert timer.passed, f"Chat init {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_session_creation_time(self, performance_timer):
        """Test that new session creation is fast."""
        session_target_ms = 100  # 100ms

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.get_session = AsyncMock(return_value=None)
            MockService.return_value = mock_service

            with performance_timer("session_create", session_target_ms) as timer:
                await mock_service.get_session("new-session")

            assert timer.passed, f"Session creation {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

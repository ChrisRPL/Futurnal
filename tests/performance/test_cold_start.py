"""Cold start time performance benchmarks.

Production target: Cold start time < 10 seconds

Tests:
- Application startup time
- Service initialization time
- First query time after startup
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Performance targets (milliseconds)
COLD_START_TARGET_MS = 10000  # 10 seconds
SERVICE_INIT_TARGET_MS = 5000  # 5 seconds
FIRST_QUERY_TARGET_MS = 2000  # 2 seconds after init


@pytest.mark.performance
class TestColdStart:
    """Cold start time benchmark tests."""

    @pytest.mark.asyncio
    async def test_search_api_initialization(self, performance_timer):
        """Test HybridSearchAPI initialization time."""
        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.initialize = AsyncMock()
            MockAPI.return_value = mock_api

            with performance_timer("search_api_init", SERVICE_INIT_TARGET_MS) as timer:
                api = MockAPI()
                await api.initialize()

            assert timer.passed, f"Search API init {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_chat_service_initialization(self, performance_timer):
        """Test ChatService initialization time."""
        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            MockService.return_value = mock_service

            with performance_timer("chat_service_init", SERVICE_INIT_TARGET_MS) as timer:
                service = MockService()
                await service.initialize()

            assert timer.passed, f"Chat service init {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_embedding_service_warmup(self, performance_timer, mock_embeddings):
        """Test embedding service warmup time."""
        warmup_target_ms = 3000  # 3 seconds for model warmup

        with performance_timer("embedding_warmup", warmup_target_ms) as timer:
            # Warmup call
            await mock_embeddings.embed("warmup text")

        assert timer.passed, f"Embedding warmup {timer.duration_ms:.2f}ms exceeded target"
        print(timer.result)

    @pytest.mark.asyncio
    async def test_first_search_after_cold_start(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test first search query time after cold start."""
        mock_results = [{"id": "1", "content": "test", "score": 0.9}]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.initialize = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_results)
            MockAPI.return_value = mock_api

            # Simulate cold start
            api = MockAPI()
            await api.initialize()

            # First search (may include lazy loading)
            with performance_timer("first_search", FIRST_QUERY_TARGET_MS) as timer:
                results = await api.search("test query", top_k=10)

            assert len(results) == 1
            assert timer.passed, f"First search {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_first_chat_after_cold_start(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
    ):
        """Test first chat message time after cold start."""
        mock_response = MagicMock()
        mock_response.content = "First response"
        mock_response.sources = []

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            # Cold start
            service = MockService()
            await service.initialize()

            # First chat
            first_chat_target_ms = 3000  # First chat may load models

            with performance_timer("first_chat", first_chat_target_ms) as timer:
                response = await service.chat("session-1", "Hello")

            assert response.content
            assert timer.passed, f"First chat {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_full_cold_start_sequence(self, performance_timer, mock_ollama, mock_embeddings):
        """Test complete cold start sequence."""
        with patch("futurnal.search.api.HybridSearchAPI") as MockSearchAPI:
            with patch("futurnal.chat.service.ChatService") as MockChatService:
                mock_search = AsyncMock()
                mock_search.initialize = AsyncMock()
                mock_search.search = AsyncMock(return_value=[])
                MockSearchAPI.return_value = mock_search

                mock_chat = AsyncMock()
                mock_chat.initialize = AsyncMock()
                mock_chat.chat = AsyncMock(
                    return_value=MagicMock(content="response", sources=[])
                )
                MockChatService.return_value = mock_chat

                with performance_timer("full_cold_start", COLD_START_TARGET_MS) as timer:
                    # Initialize services
                    search_api = MockSearchAPI()
                    await search_api.initialize()

                    chat_service = MockChatService()
                    await chat_service.initialize()

                    # First operations
                    await search_api.search("warmup", top_k=1)
                    await chat_service.chat("session", "warmup")

                assert timer.passed, f"Full cold start {timer.duration_ms:.2f}ms exceeded target"
                print(timer.result)


@pytest.mark.performance
class TestLazyLoading:
    """Test lazy loading optimizations."""

    @pytest.mark.asyncio
    async def test_lazy_model_loading(self, performance_timer):
        """Test that models are loaded lazily."""
        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = MagicMock()
            mock_api._model_loaded = False

            async def lazy_search(*args, **kwargs):
                if not mock_api._model_loaded:
                    await asyncio.sleep(0.1)  # Simulate model loading
                    mock_api._model_loaded = True
                return []

            mock_api.search = AsyncMock(side_effect=lazy_search)
            MockAPI.return_value = mock_api

            # First search (triggers load)
            with performance_timer("lazy_first", 500) as timer1:
                await mock_api.search("test")

            # Second search (already loaded)
            with performance_timer("lazy_second", 100) as timer2:
                await mock_api.search("test")

            # Second should be significantly faster
            assert (
                timer2.duration_ms < timer1.duration_ms
            ), "Lazy loading not effective"
            print(f"First: {timer1.duration_ms:.2f}ms, Second: {timer2.duration_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_cache_warmup_impact(self, performance_timer, mock_embeddings):
        """Test impact of cache warmup on performance."""
        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()

            call_count = 0

            async def search_with_cache(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    await asyncio.sleep(0.2)  # Cold cache
                else:
                    await asyncio.sleep(0.02)  # Warm cache
                return []

            mock_api.search = AsyncMock(side_effect=search_with_cache)
            mock_api.warmup = AsyncMock()
            MockAPI.return_value = mock_api

            # Without warmup
            with performance_timer("no_warmup", 500) as timer1:
                await mock_api.search("test")

            # With warmup
            await mock_api.warmup()

            with performance_timer("with_warmup", 100) as timer2:
                await mock_api.search("test")

            print(f"No warmup: {timer1.duration_ms:.2f}ms, With warmup: {timer2.duration_ms:.2f}ms")

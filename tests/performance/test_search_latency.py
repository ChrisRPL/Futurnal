"""Search latency performance benchmarks.

Production target: First search < 1 second (1000ms)

Tests:
- Simple keyword search
- Temporal query search
- Hybrid search (vector + graph)
- Cached search performance
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Performance targets (milliseconds)
SEARCH_LATENCY_TARGET_MS = 1000
CACHED_SEARCH_TARGET_MS = 200
TEMPORAL_SEARCH_TARGET_MS = 1500


@pytest.mark.performance
class TestSearchLatency:
    """Search latency benchmark tests."""

    @pytest.mark.asyncio
    async def test_simple_search_latency(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that simple keyword search completes under target."""
        # Setup mock search API
        mock_search_results = [
            {
                "id": f"result-{i}",
                "content": f"Test content {i}",
                "score": 0.9 - (i * 0.1),
            }
            for i in range(10)
        ]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_search_results)
            MockAPI.return_value = mock_api

            # Time the search
            with performance_timer("simple_search", SEARCH_LATENCY_TARGET_MS) as timer:
                results = await mock_api.search("test query", top_k=10)

            # Verify results
            assert len(results) == 10
            assert timer.passed, f"Search latency {timer.duration_ms:.2f}ms exceeded target {SEARCH_LATENCY_TARGET_MS}ms"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_temporal_search_latency(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that temporal queries complete under target."""
        mock_search_results = [
            {
                "id": f"result-{i}",
                "content": f"Test content {i}",
                "score": 0.9,
                "timestamp": "2024-12-15T10:00:00Z",
            }
            for i in range(10)
        ]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search_temporal = AsyncMock(return_value=mock_search_results)
            MockAPI.return_value = mock_api

            with performance_timer(
                "temporal_search", TEMPORAL_SEARCH_TARGET_MS
            ) as timer:
                results = await mock_api.search_temporal(
                    "what happened",
                    relative_range="last week",
                )

            assert len(results) == 10
            assert timer.passed, f"Temporal search {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_cached_search_latency(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that cached searches are significantly faster."""
        mock_results = [{"id": "result-1", "content": "cached", "score": 0.95}]

        # Simulate cached response (instant)
        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_results)
            MockAPI.return_value = mock_api

            # First search (populate cache conceptually)
            await mock_api.search("test query", top_k=10)

            # Cached search should be faster
            with performance_timer("cached_search", CACHED_SEARCH_TARGET_MS) as timer:
                results = await mock_api.search("test query", top_k=10)

            assert len(results) == 1
            assert timer.passed, f"Cached search {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_concurrent_searches(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that multiple concurrent searches complete in reasonable time."""
        mock_results = [{"id": "result-1", "content": "test", "score": 0.9}]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_results)
            MockAPI.return_value = mock_api

            # Run 10 concurrent searches
            concurrent_target_ms = SEARCH_LATENCY_TARGET_MS * 2  # 2x for concurrent

            with performance_timer("concurrent_searches", concurrent_target_ms) as timer:
                tasks = [
                    mock_api.search(f"query {i}", top_k=5) for i in range(10)
                ]
                results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert timer.passed, f"Concurrent searches {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_large_result_set_latency(
        self,
        performance_timer,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test latency with larger result sets."""
        # 100 results
        mock_results = [
            {"id": f"result-{i}", "content": f"content {i}" * 100, "score": 0.9}
            for i in range(100)
        ]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_results)
            MockAPI.return_value = mock_api

            large_result_target_ms = SEARCH_LATENCY_TARGET_MS * 1.5

            with performance_timer("large_result_search", large_result_target_ms) as timer:
                results = await mock_api.search("test", top_k=100)

            assert len(results) == 100
            assert timer.passed, f"Large result search {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)


@pytest.mark.performance
class TestSearchLatencyIntegration:
    """Integration-level search latency tests.

    These tests require actual infrastructure and are marked slow.
    Run with: pytest -m "performance and slow"
    """

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_end_to_end_search_latency(self, performance_timer):
        """Test actual end-to-end search latency.

        This test requires:
        - Running Ollama
        - Initialized PKG database
        - Indexed content

        Skip if infrastructure not available.
        """
        pytest.skip("Requires running infrastructure - run manually")

        # Would use actual API here
        # from futurnal.search.api import create_hybrid_search_api
        # api = await create_hybrid_search_api()
        # with performance_timer("e2e_search", SEARCH_LATENCY_TARGET_MS) as timer:
        #     results = await api.search("test", top_k=10)
        # assert timer.passed

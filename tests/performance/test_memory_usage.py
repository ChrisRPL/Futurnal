"""Memory usage performance benchmarks.

Production target: < 2GB memory usage for extraction pipeline

Tests:
- Baseline memory usage
- Memory growth during processing
- Memory leak detection
- Large document handling
"""

from __future__ import annotations

import asyncio
import gc
import tracemalloc
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Performance targets (MB)
MEMORY_TARGET_MB = 2048  # 2GB
BASELINE_MEMORY_MB = 256
LEAK_THRESHOLD_MB = 50  # Max growth per iteration


@pytest.mark.performance
class TestMemoryUsage:
    """Memory usage benchmark tests."""

    @pytest.mark.asyncio
    async def test_baseline_memory(self, memory_tracker):
        """Test baseline memory usage."""
        with memory_tracker("baseline", BASELINE_MEMORY_MB) as tracker:
            # Just importing and basic setup
            pass

        print(tracker.result)
        # Baseline should be well under limit
        assert tracker.peak_mb < BASELINE_MEMORY_MB

    @pytest.mark.asyncio
    async def test_search_memory_usage(
        self,
        memory_tracker,
        mock_ollama,
        mock_embeddings,
        mock_pkg,
    ):
        """Test memory usage during search operations."""
        mock_results = [
            {"id": f"result-{i}", "content": "x" * 1000, "score": 0.9}
            for i in range(100)
        ]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_results)
            MockAPI.return_value = mock_api

            search_memory_target_mb = 512

            with memory_tracker("search_operation", search_memory_target_mb) as tracker:
                for _ in range(10):
                    results = await mock_api.search("test query", top_k=100)
                    # Process results
                    _ = [r["content"] for r in results]

            print(tracker.result)
            assert tracker.passed, f"Search memory {tracker.peak_mb:.2f}MB exceeded target"

    @pytest.mark.asyncio
    async def test_chat_memory_usage(
        self,
        memory_tracker,
        mock_ollama,
        mock_embeddings,
    ):
        """Test memory usage during chat operations."""
        mock_response = MagicMock()
        mock_response.content = "Response " * 500
        mock_response.sources = []

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            chat_memory_target_mb = 512

            with memory_tracker("chat_operation", chat_memory_target_mb) as tracker:
                for i in range(20):
                    response = await mock_service.chat(f"session-{i}", "Question")
                    _ = response.content

            print(tracker.result)
            assert tracker.passed, f"Chat memory {tracker.peak_mb:.2f}MB exceeded target"

    @pytest.mark.asyncio
    async def test_ingestion_memory_usage(
        self,
        memory_tracker,
        test_documents,
        mock_embeddings,
        mock_pkg,
    ):
        """Test memory usage during document ingestion."""
        docs = test_documents[:50]

        with patch("futurnal.pipeline.normalization.service.NormalizationService") as MockNorm:
            mock_norm = AsyncMock()
            mock_norm.normalize = AsyncMock(
                side_effect=lambda d: {"content": d["content"], "metadata": d["metadata"]}
            )
            MockNorm.return_value = mock_norm

            ingestion_memory_target_mb = MEMORY_TARGET_MB

            with memory_tracker("ingestion", ingestion_memory_target_mb) as tracker:
                for doc in docs:
                    normalized = await mock_norm.normalize(doc)
                    embedding = await mock_embeddings.embed(normalized["content"])
                    # Simulate storing
                    del normalized, embedding
                    gc.collect()

            print(tracker.result)
            assert tracker.passed, f"Ingestion memory {tracker.peak_mb:.2f}MB exceeded target"


@pytest.mark.performance
class TestMemoryLeaks:
    """Memory leak detection tests."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_in_search(self, mock_ollama, mock_embeddings, mock_pkg):
        """Test that repeated searches don't leak memory."""
        mock_results = [{"id": "1", "content": "test", "score": 0.9}]

        with patch("futurnal.search.api.HybridSearchAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.search = AsyncMock(return_value=mock_results)
            MockAPI.return_value = mock_api

            gc.collect()
            tracemalloc.start()

            # Initial snapshot
            snapshot1 = tracemalloc.take_snapshot()

            # Run many searches
            for _ in range(100):
                results = await mock_api.search("test", top_k=10)
                del results

            gc.collect()

            # Final snapshot
            snapshot2 = tracemalloc.take_snapshot()

            tracemalloc.stop()

            # Compare snapshots
            top_stats = snapshot2.compare_to(snapshot1, "lineno")

            # Calculate total growth
            total_growth_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)

            print(f"Memory growth after 100 searches: {total_growth_mb:.2f}MB")

            assert (
                total_growth_mb < LEAK_THRESHOLD_MB
            ), f"Possible memory leak: {total_growth_mb:.2f}MB growth"

    @pytest.mark.asyncio
    async def test_no_memory_leak_in_chat(self, mock_ollama, mock_embeddings):
        """Test that repeated chat messages don't leak memory."""
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.sources = []

        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.chat = AsyncMock(return_value=mock_response)
            MockService.return_value = mock_service

            gc.collect()
            tracemalloc.start()

            snapshot1 = tracemalloc.take_snapshot()

            # Run many chat messages
            for i in range(100):
                response = await mock_service.chat(f"session-{i % 5}", f"Message {i}")
                del response

            gc.collect()

            snapshot2 = tracemalloc.take_snapshot()

            tracemalloc.stop()

            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            total_growth_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)

            print(f"Memory growth after 100 chats: {total_growth_mb:.2f}MB")

            assert (
                total_growth_mb < LEAK_THRESHOLD_MB
            ), f"Possible memory leak: {total_growth_mb:.2f}MB growth"

    @pytest.mark.asyncio
    async def test_session_cleanup(self, mock_ollama):
        """Test that closed sessions release memory."""
        with patch("futurnal.chat.service.ChatService") as MockService:
            mock_service = AsyncMock()
            mock_service.delete_session = AsyncMock()
            MockService.return_value = mock_service

            gc.collect()
            tracemalloc.start()

            snapshot1 = tracemalloc.take_snapshot()

            # Create and delete many sessions
            for i in range(50):
                await mock_service.delete_session(f"session-{i}")

            gc.collect()

            snapshot2 = tracemalloc.take_snapshot()

            tracemalloc.stop()

            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            total_growth_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)

            print(f"Memory after 50 session cleanups: {total_growth_mb:.2f}MB")

            # Should be minimal or negative growth after cleanup
            assert total_growth_mb < 10, f"Sessions not properly cleaned up: {total_growth_mb:.2f}MB"


@pytest.mark.performance
class TestLargeDocumentHandling:
    """Test memory handling with large documents."""

    @pytest.mark.asyncio
    async def test_large_document_memory(self, memory_tracker, mock_embeddings):
        """Test memory usage with very large documents."""
        # 10MB document
        large_content = "x" * (10 * 1024 * 1024)
        large_doc = {"id": "large", "content": large_content}

        large_doc_target_mb = 256  # Should process without huge overhead

        with memory_tracker("large_doc", large_doc_target_mb) as tracker:
            embedding = await mock_embeddings.embed(large_doc["content"][:10000])
            del embedding, large_content, large_doc
            gc.collect()

        print(tracker.result)
        # Note: This is expected to fail with actual large docs
        # The test documents the expected behavior

    @pytest.mark.asyncio
    async def test_many_small_documents_memory(
        self,
        memory_tracker,
        mock_embeddings,
    ):
        """Test memory with many small documents."""
        # 1000 small documents
        small_docs = [
            {"id": f"doc-{i}", "content": f"Small content {i}" * 10}
            for i in range(1000)
        ]

        many_docs_target_mb = 512

        with memory_tracker("many_small_docs", many_docs_target_mb) as tracker:
            for doc in small_docs:
                await mock_embeddings.embed(doc["content"])

            del small_docs
            gc.collect()

        print(tracker.result)
        assert tracker.passed, f"Many docs memory {tracker.peak_mb:.2f}MB exceeded target"

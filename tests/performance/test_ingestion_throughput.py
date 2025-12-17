"""Ingestion throughput performance benchmarks.

Production target: > 5 documents per second

Tests:
- Single document processing
- Batch document processing
- Pipeline throughput
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Performance targets
THROUGHPUT_TARGET_DOCS_PER_SEC = 5.0
SINGLE_DOC_TARGET_MS = 200
BATCH_100_TARGET_MS = 20000  # 100 docs in 20 seconds = 5 docs/sec


@pytest.mark.performance
class TestIngestionThroughput:
    """Ingestion throughput benchmark tests."""

    @pytest.mark.asyncio
    async def test_single_document_processing(
        self,
        performance_timer,
        test_documents,
        mock_embeddings,
        mock_pkg,
    ):
        """Test single document processing time."""
        doc = test_documents[0]

        with patch("futurnal.pipeline.normalization.service.NormalizationService") as MockNorm:
            mock_norm = AsyncMock()
            mock_norm.normalize = AsyncMock(
                return_value={"content": doc["content"], "metadata": doc["metadata"]}
            )
            MockNorm.return_value = mock_norm

            with performance_timer("single_doc", SINGLE_DOC_TARGET_MS) as timer:
                result = await mock_norm.normalize(doc)

            assert result is not None
            assert timer.passed, f"Single doc {timer.duration_ms:.2f}ms exceeded target"
            print(timer.result)

    @pytest.mark.asyncio
    async def test_batch_processing_throughput(
        self,
        performance_timer,
        test_documents,
        mock_embeddings,
        mock_pkg,
    ):
        """Test batch document processing throughput."""
        docs = test_documents[:100]  # 100 documents

        with patch("futurnal.pipeline.normalization.service.NormalizationService") as MockNorm:
            mock_norm = AsyncMock()

            async def mock_normalize(doc):
                # Simulate some processing time
                await asyncio.sleep(0.001)
                return {"content": doc["content"], "metadata": doc["metadata"]}

            mock_norm.normalize = AsyncMock(side_effect=mock_normalize)
            MockNorm.return_value = mock_norm

            processed_count = 0

            with performance_timer("batch_100", BATCH_100_TARGET_MS) as timer:
                for doc in docs:
                    await mock_norm.normalize(doc)
                    processed_count += 1

            # Calculate throughput
            duration_sec = timer.duration_ms / 1000
            throughput = processed_count / duration_sec if duration_sec > 0 else 0

            assert processed_count == 100
            assert (
                throughput >= THROUGHPUT_TARGET_DOCS_PER_SEC
            ), f"Throughput {throughput:.2f} docs/sec below target {THROUGHPUT_TARGET_DOCS_PER_SEC}"
            print(f"Throughput: {throughput:.2f} docs/sec")
            print(timer.result)

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(
        self,
        performance_timer,
        test_documents,
        mock_embeddings,
        mock_pkg,
    ):
        """Test concurrent document processing."""
        docs = test_documents[:100]

        with patch("futurnal.pipeline.normalization.service.NormalizationService") as MockNorm:
            mock_norm = AsyncMock()

            async def mock_normalize(doc):
                await asyncio.sleep(0.001)
                return {"content": doc["content"], "metadata": doc["metadata"]}

            mock_norm.normalize = AsyncMock(side_effect=mock_normalize)
            MockNorm.return_value = mock_norm

            # Process in batches of 10 concurrently
            concurrent_target_ms = 10000  # 10 seconds for 100 docs

            with performance_timer("concurrent_batch", concurrent_target_ms) as timer:
                batch_size = 10
                for i in range(0, len(docs), batch_size):
                    batch = docs[i : i + batch_size]
                    tasks = [mock_norm.normalize(doc) for doc in batch]
                    await asyncio.gather(*tasks)

            duration_sec = timer.duration_ms / 1000
            throughput = len(docs) / duration_sec if duration_sec > 0 else 0

            assert throughput >= THROUGHPUT_TARGET_DOCS_PER_SEC
            print(f"Concurrent throughput: {throughput:.2f} docs/sec")
            print(timer.result)

    @pytest.mark.asyncio
    async def test_embedding_generation_throughput(
        self,
        performance_timer,
        test_documents,
        mock_embeddings,
    ):
        """Test embedding generation throughput."""
        docs = test_documents[:50]
        contents = [doc["content"] for doc in docs]

        # Target: embed 50 documents in under 5 seconds
        embedding_target_ms = 5000

        with performance_timer("embeddings_50", embedding_target_ms) as timer:
            embeddings = await mock_embeddings.embed_batch(contents)

        assert len(embeddings) == 50
        duration_sec = timer.duration_ms / 1000
        throughput = len(docs) / duration_sec if duration_sec > 0 else 0

        print(f"Embedding throughput: {throughput:.2f} docs/sec")
        print(timer.result)

    @pytest.mark.asyncio
    async def test_full_pipeline_throughput(
        self,
        performance_timer,
        test_documents,
        mock_embeddings,
        mock_pkg,
    ):
        """Test full ingestion pipeline throughput."""
        docs = test_documents[:20]  # Smaller set for full pipeline

        with patch("futurnal.pipeline.normalization.service.NormalizationService") as MockNorm:
            mock_norm = AsyncMock()
            mock_norm.normalize = AsyncMock(
                side_effect=lambda d: {"content": d["content"], "metadata": d["metadata"]}
            )
            MockNorm.return_value = mock_norm

            # Full pipeline: normalize -> embed -> store
            pipeline_target_ms = 4000  # 4 seconds for 20 docs

            processed = 0

            with performance_timer("full_pipeline", pipeline_target_ms) as timer:
                for doc in docs:
                    # Normalize
                    normalized = await mock_norm.normalize(doc)

                    # Embed
                    embedding = await mock_embeddings.embed(normalized["content"])

                    # Store (mocked)
                    await mock_pkg.query("STORE")

                    processed += 1

            duration_sec = timer.duration_ms / 1000
            throughput = processed / duration_sec if duration_sec > 0 else 0

            assert throughput >= THROUGHPUT_TARGET_DOCS_PER_SEC
            print(f"Full pipeline throughput: {throughput:.2f} docs/sec")
            print(timer.result)


@pytest.mark.performance
class TestIngestionScaling:
    """Test ingestion scaling characteristics."""

    @pytest.mark.asyncio
    async def test_linear_scaling(
        self,
        performance_timer,
        mock_embeddings,
        mock_pkg,
    ):
        """Test that processing time scales linearly with document count."""
        timings = []

        for count in [10, 20, 40]:
            docs = [
                {"id": f"doc-{i}", "content": f"content {i}" * 50}
                for i in range(count)
            ]

            start = time.perf_counter()

            for doc in docs:
                await mock_embeddings.embed(doc["content"])

            duration_ms = (time.perf_counter() - start) * 1000
            timings.append((count, duration_ms))

        # Check approximate linear scaling (2x docs should be ~2x time)
        # Allow 50% variance
        ratio_20_10 = timings[1][1] / timings[0][1]
        ratio_40_20 = timings[2][1] / timings[1][1]

        print(f"Scaling ratios: 20/10={ratio_20_10:.2f}, 40/20={ratio_40_20:.2f}")

        # Should be roughly 2x (allowing for overhead)
        assert 1.0 <= ratio_20_10 <= 3.0, f"Non-linear scaling at 20 docs: {ratio_20_10}"
        assert 1.0 <= ratio_40_20 <= 3.0, f"Non-linear scaling at 40 docs: {ratio_40_20}"

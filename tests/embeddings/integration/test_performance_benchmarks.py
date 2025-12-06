"""Performance and scalability benchmarks.

Validates:
- Single embedding latency <2s
- Batch throughput >100 embeddings/minute
- Memory usage <2GB
- Concurrent embedding performance

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/06-integration-testing.md
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import List

import pytest

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from tests.embeddings.integration.conftest import (
    EmbeddingPipeline,
    SampleEntity,
    create_test_entity,
    create_embedding_pipeline,
)
from futurnal.embeddings.request import EmbeddingRequest


class TestPerformanceBenchmarks:
    """Performance and scalability tests."""

    @pytest.mark.performance
    def test_single_embedding_latency(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate single embedding latency <2s."""
        pipeline = embedding_pipeline

        entity = create_test_entity(
            "Person",
            "John Doe",
            "Software Engineer with 10 years experience in distributed systems",
        )

        start = time.time()
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )
        elapsed = time.time() - start

        assert result.embedding is not None
        assert elapsed < 2.0, f"Latency {elapsed:.2f}s exceeds 2s target"

    @pytest.mark.performance
    def test_batch_throughput(
        self,
        embedding_pipeline: EmbeddingPipeline,
        batch_test_entities: List[SampleEntity],
    ) -> None:
        """Validate batch throughput >100 embeddings/minute."""
        pipeline = embedding_pipeline

        # Create embedding requests
        requests = [
            EmbeddingRequest(
                entity_type=entity.type,
                content=entity.content,
                entity_id=entity.id,
                temporal_context=entity.temporal_context,
            )
            for entity in batch_test_entities[:100]
        ]

        start = time.time()
        results = pipeline.embedding_service.embed_batch(requests)
        elapsed = time.time() - start

        throughput_per_minute = (len(results) / elapsed) * 60

        assert throughput_per_minute > 100, (
            f"Throughput {throughput_per_minute:.0f}/min below 100/min target"
        )

    @pytest.mark.performance
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_memory_usage(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate memory usage <2GB."""
        import os

        pipeline = embedding_pipeline
        process = psutil.Process(os.getpid())

        memory_before = process.memory_info().rss / (1024**3)  # GB

        # Generate many embeddings
        for i in range(100):
            entity = create_test_entity("Person", f"Person {i}")
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            # Store embedding
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id="test-model",
                extraction_confidence=0.9,
                source_document_id=f"doc_{i}",
            )

        memory_after = process.memory_info().rss / (1024**3)  # GB

        assert memory_after < 2.0, f"Memory usage {memory_after:.2f}GB exceeds 2GB target"

    @pytest.mark.performance
    def test_concurrent_embedding_performance(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate concurrent embedding requests."""
        pipeline = embedding_pipeline

        def embed_entity(i: int) -> bool:
            entity = create_test_entity("Person", f"Concurrent Person {i}")
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            return result.embedding is not None

        # Run 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(embed_entity, i) for i in range(50)]
            results = [f.result() for f in futures]

        assert len(results) == 50
        assert all(results)

    @pytest.mark.performance
    def test_latency_distribution(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Measure P50, P95, P99 latencies."""
        pipeline = embedding_pipeline

        latencies = []
        for i in range(50):
            entity = create_test_entity("Person", f"Latency Test {i}")

            start = time.time()
            pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            latencies.append(time.time() - start)

        # Calculate percentiles
        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.5)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]

        # P99 should be under 2s (with mock models, should be much faster)
        assert p99 < 2.0, f"P99 latency {p99:.3f}s too high"

        # Log for reporting
        print(f"\nLatency distribution: P50={p50:.4f}s, P95={p95:.4f}s, P99={p99:.4f}s")

    @pytest.mark.performance
    def test_store_and_query_latency(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Measure store and query latencies."""
        pipeline = embedding_pipeline

        # First populate the store
        for i in range(100):
            entity = create_test_entity("Person", f"Store Test {i}")
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id="test-model",
                extraction_confidence=0.9,
                source_document_id=f"doc_{i}",
            )

        # Measure query latency
        import numpy as np

        query_latencies = []
        for _ in range(20):
            query_vector = np.random.rand(768).tolist()
            start = time.time()
            pipeline.store.query_embeddings(
                query_vector=query_vector,
                top_k=10,
            )
            query_latencies.append(time.time() - start)

        avg_query_latency = sum(query_latencies) / len(query_latencies)
        assert avg_query_latency < 1.0, f"Avg query latency {avg_query_latency:.3f}s too high"

    @pytest.mark.performance
    def test_batch_store_performance(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Measure batch store performance."""
        pipeline = embedding_pipeline

        # Generate embeddings
        entities = [create_test_entity("Person", f"Batch Store {i}") for i in range(100)]
        results = []
        for entity in entities:
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            results.append((entity, result))

        # Measure batch store time
        start = time.time()
        for entity, result in results:
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id="test-model",
                extraction_confidence=0.9,
                source_document_id="batch_doc",
            )
        elapsed = time.time() - start

        store_rate = len(results) / elapsed
        assert store_rate > 100, f"Store rate {store_rate:.0f}/s too slow"

    @pytest.mark.performance
    def test_sustained_throughput(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Test sustained throughput over multiple batches."""
        pipeline = embedding_pipeline

        total_embeddings = 0
        total_time = 0

        # Run 5 batches
        for batch in range(5):
            requests = [
                EmbeddingRequest(
                    entity_type="Person",
                    content=f"Batch {batch} Person {i}",
                )
                for i in range(20)
            ]

            start = time.time()
            results = pipeline.embedding_service.embed_batch(requests)
            elapsed = time.time() - start

            total_embeddings += len(results)
            total_time += elapsed

        throughput_per_minute = (total_embeddings / total_time) * 60
        assert throughput_per_minute > 100, (
            f"Sustained throughput {throughput_per_minute:.0f}/min below target"
        )

    @pytest.mark.performance
    def test_mixed_workload_performance(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Test performance with mixed read/write workload."""
        import numpy as np

        pipeline = embedding_pipeline

        operations = 0
        start = time.time()

        for i in range(50):
            # Write operation
            entity = create_test_entity("Person", f"Mixed {i}")
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id="test-model",
                extraction_confidence=0.9,
                source_document_id="mixed_doc",
            )
            operations += 1

            # Read operation
            if i > 0:
                query_vector = np.random.rand(768).tolist()
                pipeline.store.query_embeddings(
                    query_vector=query_vector,
                    top_k=5,
                )
                operations += 1

        elapsed = time.time() - start
        ops_per_second = operations / elapsed

        assert ops_per_second > 50, f"Mixed workload {ops_per_second:.0f} ops/s too slow"

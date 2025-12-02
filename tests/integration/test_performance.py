"""Performance and scalability benchmarks.

Validates:
- Throughput > 5 docs/sec
- Memory usage < 2GB
- Latency distribution (P50, P95, P99)
- Scalability with corpus size
"""

import pytest
import time
import statistics
from typing import List

from tests.integration.fixtures.pipelines import create_pipeline_for_performance_testing
from tests.integration.fixtures.corpus import CorpusLoader
from tests.integration.fixtures.metrics import measure_throughput, get_memory_usage


class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.fixture
    def corpus_loader(self):
        return CorpusLoader()
        
    def test_throughput_target(self, corpus_loader):
        """Validate >5 documents/second throughput."""
        # Load a substantial corpus
        docs = corpus_loader.load_test_corpus(100)
        pipeline = create_pipeline_for_performance_testing()
        
        throughput = measure_throughput(pipeline, docs)
        
        # Production target: >5 docs/sec
        # Note: Mock pipeline will be very fast, real one slower
        target = 5.0
        assert throughput > target, f"Throughput {throughput} below target {target} docs/sec"

    def test_memory_usage(self, corpus_loader):
        """Validate memory usage <2GB."""
        # Load large document or batch
        docs = corpus_loader.load_test_corpus(50)
        pipeline = create_pipeline_for_performance_testing()
        
        # Measure baseline
        memory_before = get_memory_usage()
        
        # Process
        for doc in docs:
            pipeline.process(doc)
            
        # Measure peak/current
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before
        
        # Check absolute usage (more relevant for OOM prevention)
        # Production target: <2GB
        target_gb = 2.0
        assert memory_after < target_gb, f"Total memory {memory_after}GB exceeds limit {target_gb}GB"

    def test_latency_distribution(self, corpus_loader):
        """Measure P50, P95, P99 latencies."""
        docs = corpus_loader.load_test_corpus(50)
        pipeline = create_pipeline_for_performance_testing()
        
        latencies = []
        for doc in docs:
            start = time.time()
            pipeline.process(doc)
            latencies.append(time.time() - start)
            
        # Calculate percentiles
        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.5)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]
        
        # Assertions on latency (e.g., P99 < 1s)
        # These targets might need adjustment based on hardware
        assert p99 < 1.0, f"P99 latency {p99}s too high"
        
        # Log for report (in real test runner)
        print(f"Latencies: P50={p50:.4f}s, P95={p95:.4f}s, P99={p99:.4f}s")

    def test_scalability_with_corpus_size(self, corpus_loader):
        """Validate performance doesn't degrade with corpus size."""
        pipeline = create_pipeline_for_performance_testing()
        
        # Small batch
        docs_small = corpus_loader.load_test_corpus(10)
        throughput_small = measure_throughput(pipeline, docs_small)
        
        # Large batch
        docs_large = corpus_loader.load_test_corpus(100)
        throughput_large = measure_throughput(pipeline, docs_large)
        
        # Throughput shouldn't drop significantly (allow 20% variance)
        assert throughput_large >= throughput_small * 0.8, \
            f"Significant performance degradation: {throughput_small} -> {throughput_large}"

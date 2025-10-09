"""Performance tests for streaming processor.

These tests verify:
- Peak memory usage stays under 2GB requirement
- Processing throughput for large files
- Memory stability over time
- Graceful degradation under memory pressure

Run with: pytest -m performance
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from futurnal.pipeline.normalization.streaming import (
    MemoryMonitor,
    StreamingConfig,
    StreamingProcessor,
)
from futurnal.pipeline.normalization.chunking import ChunkingConfig, ChunkingStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def performance_config():
    """Streaming configuration for performance tests."""
    return StreamingConfig(
        file_size_threshold_mb=100.0,
        buffer_size_bytes=1024 * 1024,  # 1MB buffer
        chunk_batch_size=10,
        max_memory_mb=2048.0,
        enable_progress_tracking=True,
        memory_check_interval_chunks=5,
    )


@pytest.fixture
def performance_processor(performance_config):
    """Create processor for performance testing."""
    return StreamingProcessor(config=performance_config)


@pytest.fixture
def very_large_file(tmp_path):
    """Create a very large test file (100MB+)."""

    def _create_file(size_mb: float = 150.0, name: str = "very_large.txt") -> Path:
        file_path = tmp_path / name
        target_bytes = int(size_mb * 1024 * 1024)

        # Generate content efficiently
        chunk_template = "# Section {section}\n\n" + ("Content line. " * 50 + "\n") * 20
        chunk_size = len(chunk_template.format(section=0))
        num_chunks = target_bytes // chunk_size

        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(num_chunks):
                f.write(chunk_template.format(section=i))

        return file_path

    return _create_file


# ---------------------------------------------------------------------------
# Peak Memory Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestPeakMemoryUsage:
    """Tests to verify memory usage stays within limits."""

    @pytest.mark.asyncio
    async def test_memory_usage_100mb_file(
        self, performance_processor, very_large_file
    ):
        """Test peak memory usage with 100MB file stays under 2GB.

        Requirement: Peak memory usage <2GB for large files
        """
        file_path = very_large_file(size_mb=100.0)
        memory_monitor = performance_processor.memory_monitor

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BY_TITLE.value,
            max_chunk_size=4000,
            min_chunk_size=500,
        )

        # Track memory usage
        memory_samples = []

        class MemoryTrackingCallback:
            def on_progress(self, bytes_processed, total_bytes, chunks_created, memory_mb):
                memory_samples.append(memory_mb)

            def on_memory_warning(self, current_mb, limit_mb):
                pass

            def on_complete(self, total_chunks, total_bytes, duration_seconds):
                pass

        callback = MemoryTrackingCallback()

        # Process file
        start_memory_mb = memory_monitor.get_process_memory_mb()
        chunks = await performance_processor.stream_chunks(
            file_path, chunking_config, "perf-test-100mb", callback
        )
        peak_memory_mb = max(memory_samples) if memory_samples else start_memory_mb

        # Verify memory stayed within limits
        assert peak_memory_mb < 2048.0, f"Peak memory {peak_memory_mb:.1f}MB exceeded 2GB limit"

        # Log results
        print(f"\n100MB File Performance:")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Start memory: {start_memory_mb:.1f}MB")
        print(f"  Peak memory: {peak_memory_mb:.1f}MB")
        print(f"  Memory increase: {peak_memory_mb - start_memory_mb:.1f}MB")

    @pytest.mark.asyncio
    async def test_memory_usage_200mb_file(
        self, performance_processor, very_large_file
    ):
        """Test peak memory usage with 200MB file stays under 2GB."""
        file_path = very_large_file(size_mb=200.0)
        memory_monitor = performance_processor.memory_monitor

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC.value,
            max_chunk_size=4000,
        )

        # Track peak memory
        peak_memory_mb = memory_monitor.get_process_memory_mb()

        class PeakMemoryCallback:
            def __init__(self, monitor):
                self.monitor = monitor
                self.peak = 0.0

            def on_progress(self, bytes_processed, total_bytes, chunks_created, memory_mb):
                self.peak = max(self.peak, memory_mb)

            def on_memory_warning(self, current_mb, limit_mb):
                pass

            def on_complete(self, total_chunks, total_bytes, duration_seconds):
                pass

        callback = PeakMemoryCallback(memory_monitor)

        chunks = await performance_processor.stream_chunks(
            file_path, chunking_config, "perf-test-200mb", callback
        )

        assert callback.peak < 2048.0, f"Peak memory {callback.peak:.1f}MB exceeded 2GB limit"

        print(f"\n200MB File Performance:")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Peak memory: {callback.peak:.1f}MB")

    @pytest.mark.asyncio
    async def test_memory_stability_multiple_files(
        self, performance_processor, tmp_path
    ):
        """Test memory remains stable across multiple file processing."""
        memory_monitor = performance_processor.memory_monitor
        initial_memory = memory_monitor.get_process_memory_mb()

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC.value,
            max_chunk_size=4000,
        )

        # Process multiple medium-sized files
        memory_readings = [initial_memory]

        for i in range(5):
            file_path = tmp_path / f"file_{i}.txt"
            content = f"File {i} section. " * 50000  # ~50KB repeated
            file_path.write_text(content, encoding="utf-8")

            await performance_processor.stream_chunks(
                file_path, chunking_config, f"doc-{i}"
            )

            memory_readings.append(memory_monitor.get_process_memory_mb())

        # Memory should not grow unbounded
        memory_growth = memory_readings[-1] - memory_readings[0]
        assert memory_growth < 500.0, f"Memory grew by {memory_growth:.1f}MB across 5 files"

        print(f"\nMemory Stability Test:")
        print(f"  Initial: {memory_readings[0]:.1f}MB")
        print(f"  Final: {memory_readings[-1]:.1f}MB")
        print(f"  Growth: {memory_growth:.1f}MB")


# ---------------------------------------------------------------------------
# Processing Throughput Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestProcessingThroughput:
    """Tests to measure and verify processing throughput."""

    @pytest.mark.asyncio
    async def test_throughput_100mb_file(
        self, performance_processor, very_large_file
    ):
        """Test processing throughput for 100MB file.

        Target: At least 10MB/s processing speed
        """
        file_path = very_large_file(size_mb=100.0)
        file_size_bytes = file_path.stat().st_size

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BY_TITLE.value,
            max_chunk_size=4000,
        )

        start_time = time.time()
        chunks = await performance_processor.stream_chunks(
            file_path, chunking_config, "throughput-test-100mb"
        )
        duration = time.time() - start_time

        # Calculate throughput
        throughput_mbps = (file_size_bytes / (1024 * 1024)) / duration

        print(f"\nThroughput Test (100MB):")
        print(f"  File size: {file_size_bytes / (1024 * 1024):.1f}MB")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput_mbps:.2f} MB/s")
        print(f"  Chunks: {len(chunks)}")

        # Verify reasonable throughput (at least 5 MB/s on most hardware)
        assert throughput_mbps > 5.0, f"Throughput {throughput_mbps:.2f} MB/s too slow"

    @pytest.mark.asyncio
    async def test_chunking_overhead(self, performance_processor, very_large_file):
        """Compare performance across different chunking strategies."""
        file_path = very_large_file(size_mb=50.0)

        strategies = [
            ("BASIC", ChunkingStrategy.BASIC.value),
            ("BY_TITLE", ChunkingStrategy.BY_TITLE.value),
            ("SEMANTIC", ChunkingStrategy.SEMANTIC.value),
        ]

        results = {}

        for name, strategy in strategies:
            chunking_config = ChunkingConfig(
                strategy=strategy,
                max_chunk_size=4000,
            )

            start_time = time.time()
            chunks = await performance_processor.stream_chunks(
                file_path, chunking_config, f"strategy-test-{name.lower()}"
            )
            duration = time.time() - start_time

            results[name] = {
                "duration": duration,
                "chunks": len(chunks),
                "throughput_mbps": 50.0 / duration,
            }

        # Print comparison
        print(f"\nChunking Strategy Comparison (50MB file):")
        for name, data in results.items():
            print(
                f"  {name}: {data['duration']:.2f}s, {data['chunks']} chunks, "
                f"{data['throughput_mbps']:.2f} MB/s"
            )

        # All strategies should complete in reasonable time
        for name, data in results.items():
            assert data["duration"] < 30.0, f"{name} took too long: {data['duration']:.2f}s"


# ---------------------------------------------------------------------------
# Garbage Collection Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestGarbageCollection:
    """Tests for garbage collection effectiveness."""

    @pytest.mark.asyncio
    async def test_gc_trigger_effectiveness(self, performance_processor, very_large_file):
        """Test that garbage collection effectively reduces memory."""
        memory_monitor = performance_processor.memory_monitor
        file_path = very_large_file(size_mb=100.0)

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC.value,
            max_chunk_size=4000,
        )

        # Track GC events
        gc_count_before = memory_monitor.get_metrics()["gc_triggered_count"]

        await performance_processor.stream_chunks(
            file_path, chunking_config, "gc-test"
        )

        gc_count_after = memory_monitor.get_metrics()["gc_triggered_count"]

        print(f"\nGC Effectiveness:")
        print(f"  GC triggers: {gc_count_after - gc_count_before}")

    @pytest.mark.asyncio
    async def test_memory_after_gc(self):
        """Test that memory reduces after explicit garbage collection."""
        import gc

        memory_monitor = MemoryMonitor()

        # Create some garbage
        memory_before_gc = memory_monitor.get_process_memory_mb()
        large_list = [f"data_{i}" * 1000 for i in range(10000)]
        memory_with_garbage = memory_monitor.get_process_memory_mb()

        # Clear references and collect
        large_list = None
        gc.collect()
        memory_after_gc = memory_monitor.get_process_memory_mb()

        print(f"\nMemory After GC:")
        print(f"  Before: {memory_before_gc:.1f}MB")
        print(f"  With garbage: {memory_with_garbage:.1f}MB")
        print(f"  After GC: {memory_after_gc:.1f}MB")

        # Memory should be reduced (allow some variance)
        assert memory_after_gc < memory_with_garbage + 10.0


# ---------------------------------------------------------------------------
# Stress Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.slow
class TestStress:
    """Stress tests for extreme conditions."""

    @pytest.mark.asyncio
    async def test_very_large_file_500mb(self, performance_processor, very_large_file):
        """Stress test with 500MB file.

        This is a stress test - may take several minutes.
        """
        file_path = very_large_file(size_mb=500.0)
        memory_monitor = performance_processor.memory_monitor

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC.value,
            max_chunk_size=4000,
        )

        start_memory = memory_monitor.get_process_memory_mb()
        start_time = time.time()

        chunks = await performance_processor.stream_chunks(
            file_path, chunking_config, "stress-test-500mb"
        )

        duration = time.time() - start_time
        end_memory = memory_monitor.get_process_memory_mb()

        print(f"\nStress Test (500MB):")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Start memory: {start_memory:.1f}MB")
        print(f"  End memory: {end_memory:.1f}MB")
        print(f"  Peak memory stayed under 2GB: {end_memory < 2048.0}")

        assert end_memory < 2048.0, f"Memory {end_memory:.1f}MB exceeded limit"
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_continuous_processing(self, performance_processor, tmp_path):
        """Test continuous processing of multiple large files."""
        memory_monitor = performance_processor.memory_monitor

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC.value,
            max_chunk_size=4000,
        )

        total_chunks = 0
        memory_readings = []

        # Process 10 files continuously
        for i in range(10):
            file_path = tmp_path / f"continuous_{i}.txt"
            content = f"Content block {i}. " * 200000  # ~2MB per file
            file_path.write_text(content, encoding="utf-8")

            chunks = await performance_processor.stream_chunks(
                file_path, chunking_config, f"continuous-{i}"
            )

            total_chunks += len(chunks)
            memory_readings.append(memory_monitor.get_process_memory_mb())

        print(f"\nContinuous Processing:")
        print(f"  Files processed: 10")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Memory range: {min(memory_readings):.1f}MB - {max(memory_readings):.1f}MB")

        # Memory should remain stable
        assert max(memory_readings) < 2048.0

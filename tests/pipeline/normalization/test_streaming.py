"""Unit and integration tests for streaming processor.

Tests cover:
- MemoryMonitor functionality
- StreamingProcessor unit tests
- Buffer management and boundary detection
- Progress tracking and callbacks
- Memory limit enforcement
- Error handling and graceful degradation
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from futurnal.pipeline.normalization.streaming import (
    LoggingProgressCallback,
    MemoryMonitor,
    ProgressCallback,
    StreamingConfig,
    StreamingProcessor,
)
from futurnal.pipeline.normalization.chunking import ChunkingConfig, ChunkingStrategy
from futurnal.pipeline.models import DocumentChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def streaming_config():
    """Default streaming configuration for testing."""
    return StreamingConfig(
        file_size_threshold_mb=1.0,  # Low threshold for testing
        buffer_size_bytes=1024,  # Small buffer for testing
        chunk_batch_size=5,
        max_memory_mb=1024.0,
        enable_progress_tracking=True,
        memory_check_interval_chunks=2,
    )


@pytest.fixture
def chunking_config():
    """Default chunking configuration for testing."""
    return ChunkingConfig(
        strategy=ChunkingStrategy.BASIC.value,
        max_chunk_size=500,
        min_chunk_size=100,
        overlap_size=50,
    )


@pytest.fixture
def memory_monitor():
    """Create memory monitor instance."""
    return MemoryMonitor()


@pytest.fixture
def streaming_processor(streaming_config, memory_monitor):
    """Create streaming processor instance."""
    return StreamingProcessor(config=streaming_config, memory_monitor=memory_monitor)


@pytest.fixture
def large_test_file(tmp_path):
    """Create a large test file for streaming."""

    def _create_file(size_mb: float = 2.0, name: str = "large_test.txt") -> Path:
        file_path = tmp_path / name
        # Generate content with headings and paragraphs
        content_lines = []
        bytes_written = 0
        target_bytes = int(size_mb * 1024 * 1024)

        section_num = 0
        while bytes_written < target_bytes:
            section_num += 1
            heading = f"# Section {section_num}\n\n"
            content_lines.append(heading)
            bytes_written += len(heading)

            # Add paragraphs
            for para_num in range(5):
                paragraph = f"This is paragraph {para_num} in section {section_num}. " * 10
                paragraph += "\n\n"
                content_lines.append(paragraph)
                bytes_written += len(paragraph)

                if bytes_written >= target_bytes:
                    break

        file_path.write_text("".join(content_lines), encoding="utf-8")
        return file_path

    return _create_file


@pytest.fixture
def mock_progress_callback():
    """Create mock progress callback."""

    class MockCallback:
        def __init__(self):
            self.progress_calls = []
            self.warning_calls = []
            self.complete_calls = []

        def on_progress(self, bytes_processed, total_bytes, chunks_created, memory_mb):
            self.progress_calls.append(
                {
                    "bytes_processed": bytes_processed,
                    "total_bytes": total_bytes,
                    "chunks_created": chunks_created,
                    "memory_mb": memory_mb,
                }
            )

        def on_memory_warning(self, current_mb, limit_mb):
            self.warning_calls.append({"current_mb": current_mb, "limit_mb": limit_mb})

        def on_complete(self, total_chunks, total_bytes, duration_seconds):
            self.complete_calls.append(
                {
                    "total_chunks": total_chunks,
                    "total_bytes": total_bytes,
                    "duration_seconds": duration_seconds,
                }
            )

    return MockCallback()


# ---------------------------------------------------------------------------
# MemoryMonitor Tests
# ---------------------------------------------------------------------------


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_get_process_memory_mb(self, memory_monitor):
        """Test getting process memory usage."""
        memory_mb = memory_monitor.get_process_memory_mb()

        assert isinstance(memory_mb, float)
        assert memory_mb > 0  # Should have some memory usage
        assert memory_mb < 10000  # Sanity check (< 10GB)

    def test_get_system_memory_available_mb(self, memory_monitor):
        """Test getting available system memory."""
        available_mb = memory_monitor.get_system_memory_available_mb()

        assert isinstance(available_mb, float)
        assert available_mb > 0  # Should have some available memory

    def test_is_memory_pressure(self, memory_monitor):
        """Test memory pressure detection."""
        current_mb = memory_monitor.get_process_memory_mb()

        # Should not have pressure with very high threshold
        assert not memory_monitor.is_memory_pressure(current_mb + 1000)

        # Should have pressure with very low threshold
        assert memory_monitor.is_memory_pressure(current_mb - 10)

    def test_trigger_gc_if_needed(self, memory_monitor):
        """Test garbage collection triggering."""
        # Should not trigger GC with plenty of headroom
        triggered = memory_monitor.trigger_gc_if_needed(100.0, 1000.0, threshold_pct=0.80)
        assert not triggered

        # Should trigger GC when near threshold
        triggered = memory_monitor.trigger_gc_if_needed(850.0, 1000.0, threshold_pct=0.80)
        assert triggered

    def test_get_metrics(self, memory_monitor):
        """Test getting memory metrics."""
        metrics = memory_monitor.get_metrics()

        assert "process_memory_mb" in metrics
        assert "system_memory_available_mb" in metrics
        assert "gc_triggered_count" in metrics
        assert isinstance(metrics["gc_triggered_count"], int)


# ---------------------------------------------------------------------------
# StreamingConfig Tests
# ---------------------------------------------------------------------------


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.file_size_threshold_mb == 100.0
        assert config.buffer_size_bytes == 1024 * 1024
        assert config.chunk_batch_size == 10
        assert config.max_memory_mb == 2048.0
        assert config.enable_progress_tracking is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StreamingConfig(
            file_size_threshold_mb=50.0,
            buffer_size_bytes=512 * 1024,
            max_memory_mb=1024.0,
        )

        assert config.file_size_threshold_mb == 50.0
        assert config.buffer_size_bytes == 512 * 1024
        assert config.max_memory_mb == 1024.0


# ---------------------------------------------------------------------------
# StreamingProcessor Tests
# ---------------------------------------------------------------------------


class TestStreamingProcessor:
    """Tests for StreamingProcessor class."""

    @pytest.mark.asyncio
    async def test_should_stream_large_file(self, streaming_processor, large_test_file):
        """Test that large files trigger streaming."""
        file_path = large_test_file(size_mb=2.0)
        should_stream = await streaming_processor.should_stream(file_path)

        assert should_stream is True

    @pytest.mark.asyncio
    async def test_should_not_stream_small_file(self, streaming_processor, tmp_path):
        """Test that small files don't trigger streaming."""
        small_file = tmp_path / "small.txt"
        small_file.write_text("Small content", encoding="utf-8")

        should_stream = await streaming_processor.should_stream(small_file)

        assert should_stream is False

    @pytest.mark.asyncio
    async def test_should_stream_nonexistent_file(self, streaming_processor, tmp_path):
        """Test handling of nonexistent file."""
        nonexistent = tmp_path / "nonexistent.txt"
        should_stream = await streaming_processor.should_stream(nonexistent)

        assert should_stream is False

    @pytest.mark.asyncio
    async def test_stream_chunks_basic(
        self, streaming_processor, chunking_config, large_test_file
    ):
        """Test basic chunk streaming."""
        file_path = large_test_file(size_mb=0.1)  # Small file for testing
        parent_document_id = "test-doc-123"

        chunks = await streaming_processor.stream_chunks(
            file_path, chunking_config, parent_document_id
        )

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Verify chunk properties
        for idx, chunk in enumerate(chunks):
            assert isinstance(chunk, DocumentChunk)
            assert chunk.parent_document_id == parent_document_id
            assert chunk.chunk_index == idx
            assert chunk.content
            assert chunk.content_hash
            assert chunk.character_count == len(chunk.content)
            assert chunk.word_count > 0

    @pytest.mark.asyncio
    async def test_stream_chunks_by_title(
        self, streaming_processor, large_test_file
    ):
        """Test streaming with BY_TITLE chunking strategy."""
        file_path = large_test_file(size_mb=0.1)
        parent_document_id = "test-doc-title"

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BY_TITLE.value,
            max_chunk_size=2000,
            min_chunk_size=100,
        )

        chunks = await streaming_processor.stream_chunks(
            file_path, chunking_config, parent_document_id
        )

        assert len(chunks) > 0

        # Verify some chunks have section titles
        chunks_with_titles = [c for c in chunks if c.section_title]
        assert len(chunks_with_titles) > 0

    @pytest.mark.asyncio
    async def test_stream_chunks_semantic(
        self, streaming_processor, tmp_path
    ):
        """Test streaming with SEMANTIC chunking strategy."""
        # Create file with clear paragraph boundaries
        content = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(10)])
        file_path = tmp_path / "semantic_test.txt"
        file_path.write_text(content, encoding="utf-8")

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC.value,
            max_chunk_size=500,
            min_chunk_size=100,
        )

        chunks = await streaming_processor.stream_chunks(
            file_path, chunking_config, "test-doc-semantic"
        )

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_chunks_with_progress(
        self, streaming_processor, chunking_config, large_test_file, mock_progress_callback
    ):
        """Test streaming with progress callbacks."""
        file_path = large_test_file(size_mb=0.1)

        chunks = await streaming_processor.stream_chunks(
            file_path, chunking_config, "test-doc-progress", mock_progress_callback
        )

        assert len(chunks) > 0

        # Verify progress callbacks were called
        assert len(mock_progress_callback.progress_calls) > 0
        assert len(mock_progress_callback.complete_calls) == 1

        # Verify completion callback has correct data
        completion = mock_progress_callback.complete_calls[0]
        assert completion["total_chunks"] == len(chunks)
        assert completion["total_bytes"] > 0
        assert completion["duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_stream_file_content(self, streaming_processor, tmp_path):
        """Test file content streaming."""
        content = "Test content " * 1000
        file_path = tmp_path / "stream_test.txt"
        file_path.write_text(content, encoding="utf-8")

        buffers = []
        async for buffer in streaming_processor._stream_file_content(file_path):
            buffers.append(buffer)

        assert len(buffers) > 0
        reconstructed = "".join(buffers)
        assert reconstructed == content

    @pytest.mark.asyncio
    async def test_memory_pressure_monitoring(
        self, streaming_processor, chunking_config, large_test_file
    ):
        """Test that memory pressure is monitored during streaming."""
        file_path = large_test_file(size_mb=0.1)

        # Mock memory monitor to simulate high memory usage
        with patch.object(
            streaming_processor.memory_monitor, "get_process_memory_mb", return_value=1800.0
        ):
            # Should still complete but trigger memory checks
            chunks = await streaming_processor.stream_chunks(
                file_path, chunking_config, "test-doc-memory"
            )

            assert len(chunks) > 0

    def test_get_metrics(self, streaming_processor):
        """Test getting streaming metrics."""
        metrics = streaming_processor.get_metrics()

        assert "files_streamed" in metrics
        assert "total_bytes_streamed" in metrics
        assert "total_chunks_created" in metrics
        assert "average_file_size_mb" in metrics
        assert "average_chunks_per_file" in metrics
        assert "process_memory_mb" in metrics

    @pytest.mark.asyncio
    async def test_metrics_update_after_streaming(
        self, streaming_processor, chunking_config, large_test_file
    ):
        """Test that metrics are updated after streaming."""
        initial_metrics = streaming_processor.get_metrics()
        assert initial_metrics["files_streamed"] == 0

        file_path = large_test_file(size_mb=0.1)
        await streaming_processor.stream_chunks(file_path, chunking_config, "test-doc-metrics")

        final_metrics = streaming_processor.get_metrics()
        assert final_metrics["files_streamed"] == 1
        assert final_metrics["total_bytes_streamed"] > 0
        assert final_metrics["total_chunks_created"] > 0


# ---------------------------------------------------------------------------
# Progress Callback Tests
# ---------------------------------------------------------------------------


class TestLoggingProgressCallback:
    """Tests for LoggingProgressCallback."""

    def test_on_progress(self, caplog):
        """Test progress logging."""
        import logging
        caplog.set_level(logging.INFO, logger="futurnal.pipeline.normalization.streaming")

        callback = LoggingProgressCallback()
        callback.on_progress(50000, 100000, 10, 128.5)

        # Check that log message was generated
        assert len(caplog.records) > 0
        assert "Streaming progress" in caplog.text

    def test_on_memory_warning(self, caplog):
        """Test memory warning logging."""
        import logging
        caplog.set_level(logging.WARNING, logger="futurnal.pipeline.normalization.streaming")

        callback = LoggingProgressCallback()
        callback.on_memory_warning(1800.0, 2048.0)

        # Check that warning was logged
        assert any("Memory pressure" in record.message for record in caplog.records)

    def test_on_complete(self, caplog):
        """Test completion logging."""
        import logging
        caplog.set_level(logging.INFO, logger="futurnal.pipeline.normalization.streaming")

        callback = LoggingProgressCallback()
        callback.on_complete(50, 100000, 5.5)

        # Check that completion was logged
        assert len(caplog.records) > 0
        assert "Streaming complete" in caplog.text


# ---------------------------------------------------------------------------
# Chunk Creation Tests
# ---------------------------------------------------------------------------


class TestChunkCreation:
    """Tests for chunk creation functionality."""

    def test_create_document_chunk(self, streaming_processor):
        """Test creating a document chunk."""
        content = "Test chunk content with multiple words."
        chunk = streaming_processor._create_document_chunk(
            content=content,
            chunk_index=0,
            parent_document_id="parent-123",
            start_char=0,
            section_title="Test Section",
            heading_hierarchy=["Chapter 1", "Test Section"],
        )

        assert isinstance(chunk, DocumentChunk)
        assert chunk.content == content
        assert chunk.chunk_index == 0
        assert chunk.parent_document_id == "parent-123"
        assert chunk.section_title == "Test Section"
        assert chunk.heading_hierarchy == ["Chapter 1", "Test Section"]
        assert chunk.character_count == len(content)
        assert chunk.word_count > 0
        assert chunk.start_char == 0
        assert chunk.end_char == len(content)

    def test_chunk_hash_deterministic(self, streaming_processor):
        """Test that chunk hashes are deterministic."""
        content = "Same content"
        parent_id = "parent-123"

        chunk1 = streaming_processor._create_document_chunk(
            content, 0, parent_id, 0
        )
        chunk2 = streaming_processor._create_document_chunk(
            content, 0, parent_id, 0
        )

        assert chunk1.content_hash == chunk2.content_hash
        assert chunk1.chunk_id == chunk2.chunk_id


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_file(self, streaming_processor, chunking_config, tmp_path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        chunks = await streaming_processor.stream_chunks(
            empty_file, chunking_config, "test-doc-empty"
        )

        # Should handle gracefully - may produce empty or minimal chunks
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_unicode_content(self, streaming_processor, chunking_config, tmp_path):
        """Test handling of Unicode content."""
        unicode_content = "Hello 世界 🌍 Привет мир"
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text(unicode_content * 100, encoding="utf-8")

        chunks = await streaming_processor.stream_chunks(
            unicode_file, chunking_config, "test-doc-unicode"
        )

        assert len(chunks) > 0
        # Verify Unicode is preserved
        full_content = "".join(c.content for c in chunks)
        assert "世界" in full_content
        assert "🌍" in full_content

    @pytest.mark.asyncio
    async def test_very_long_lines(self, streaming_processor, chunking_config, tmp_path):
        """Test handling of very long lines."""
        long_line = "x" * 10000  # 10KB single line
        file_path = tmp_path / "long_lines.txt"
        file_path.write_text(long_line * 10, encoding="utf-8")

        chunks = await streaming_processor.stream_chunks(
            file_path, chunking_config, "test-doc-longlines"
        )

        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestStreamingIntegration:
    """Integration tests for complete streaming workflow."""

    @pytest.mark.asyncio
    async def test_full_streaming_pipeline(
        self, streaming_processor, large_test_file, mock_progress_callback
    ):
        """Test complete streaming pipeline with all components."""
        file_path = large_test_file(size_mb=0.5)

        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.BY_TITLE.value,
            max_chunk_size=2000,
            min_chunk_size=200,
            overlap_size=100,
        )

        # Stream with all features enabled
        chunks = await streaming_processor.stream_chunks(
            file_path, chunking_config, "test-doc-integration", mock_progress_callback
        )

        # Verify chunks
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

        # Verify chunk indices are sequential
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

        # Verify progress tracking
        assert len(mock_progress_callback.progress_calls) > 0
        assert len(mock_progress_callback.complete_calls) == 1

        # Verify metrics
        metrics = streaming_processor.get_metrics()
        assert metrics["files_streamed"] > 0
        assert metrics["total_chunks_created"] == len(chunks)

    @pytest.mark.asyncio
    async def test_multiple_files_sequential(
        self, streaming_processor, chunking_config, tmp_path
    ):
        """Test streaming multiple files sequentially."""
        files = []
        for i in range(3):
            file_path = tmp_path / f"file_{i}.txt"
            content = f"File {i} content. " * 1000
            file_path.write_text(content, encoding="utf-8")
            files.append(file_path)

        all_chunks = []
        for file_path in files:
            chunks = await streaming_processor.stream_chunks(
                file_path, chunking_config, f"doc-{file_path.stem}"
            )
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0

        # Verify metrics reflect all files
        metrics = streaming_processor.get_metrics()
        assert metrics["files_streamed"] == 3

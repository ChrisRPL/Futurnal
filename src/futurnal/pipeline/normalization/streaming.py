"""Streaming processor for efficient handling of large documents.

This module provides a streaming document processor that handles large files (>100MB)
efficiently with minimal memory footprint. The processor uses iterative parsing and
chunked processing to avoid loading entire documents into memory, enabling normalization
of arbitrarily large files on consumer hardware.

Key Features:
- Streaming file reader with configurable buffer size
- Iterative element processing
- Chunked normalization pipeline
- Memory monitoring and limits
- Progress tracking for long-running operations
- Graceful degradation when memory pressure detected

Design Philosophy:
- Peak memory usage <2GB for large files
- Process files incrementally without full load
- Monitor and respond to memory pressure
- Support cancellation and progress tracking
"""

from __future__ import annotations

import asyncio
import gc
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional, Protocol, List

import psutil

from ..models import DocumentChunk, ChunkingStrategy, compute_chunk_hash, generate_chunk_id
from .chunking import ChunkingConfig, IntermediateChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StreamingConfig:
    """Configuration for streaming processor.

    Attributes:
        file_size_threshold_mb: Stream files larger than this threshold
        buffer_size_bytes: Size of read buffer for file streaming
        chunk_batch_size: Number of chunks to process before memory check
        max_memory_mb: Hard memory limit (triggers graceful degradation)
        enable_progress_tracking: Whether to enable progress callbacks
        memory_check_interval_chunks: Check memory every N chunks
        memory_warning_threshold_pct: Warn at this % of max memory
        memory_gc_threshold_pct: Trigger GC at this % of max memory
        memory_pause_threshold_pct: Pause processing at this % of max memory
    """

    file_size_threshold_mb: float = 100.0  # Stream files >100MB
    buffer_size_bytes: int = 1024 * 1024  # 1MB buffer
    chunk_batch_size: int = 10  # Process N chunks at a time
    max_memory_mb: float = 2048.0  # Hard limit
    enable_progress_tracking: bool = True

    # Memory management thresholds
    memory_check_interval_chunks: int = 5  # Check every 5 chunks
    memory_warning_threshold_pct: float = 0.75  # Warn at 75%
    memory_gc_threshold_pct: float = 0.80  # GC at 80%
    memory_pause_threshold_pct: float = 0.95  # Pause at 95%


# ---------------------------------------------------------------------------
# Progress Tracking
# ---------------------------------------------------------------------------


class ProgressCallback(Protocol):
    """Protocol for streaming progress callbacks.

    Implementations can track progress, log updates, or update UI.
    """

    def on_progress(
        self,
        bytes_processed: int,
        total_bytes: int,
        chunks_created: int,
        memory_mb: float,
    ) -> None:
        """Called periodically with progress updates.

        Args:
            bytes_processed: Bytes processed so far
            total_bytes: Total bytes in file
            chunks_created: Number of chunks created
            memory_mb: Current memory usage in MB
        """
        ...

    def on_memory_warning(self, current_mb: float, limit_mb: float) -> None:
        """Called when memory usage exceeds warning threshold.

        Args:
            current_mb: Current memory usage
            limit_mb: Configured memory limit
        """
        ...

    def on_complete(self, total_chunks: int, total_bytes: int, duration_seconds: float) -> None:
        """Called when streaming completes successfully.

        Args:
            total_chunks: Total number of chunks created
            total_bytes: Total bytes processed
            duration_seconds: Processing duration
        """
        ...


class LoggingProgressCallback:
    """Simple logging-based progress callback."""

    def on_progress(
        self, bytes_processed: int, total_bytes: int, chunks_created: int, memory_mb: float
    ) -> None:
        """Log progress updates."""
        pct = (bytes_processed / total_bytes * 100) if total_bytes > 0 else 0
        logger.info(
            f"Streaming progress: {pct:.1f}% ({bytes_processed}/{total_bytes} bytes), "
            f"{chunks_created} chunks, {memory_mb:.1f}MB memory"
        )

    def on_memory_warning(self, current_mb: float, limit_mb: float) -> None:
        """Log memory warnings."""
        pct = (current_mb / limit_mb * 100) if limit_mb > 0 else 0
        logger.warning(
            f"Memory pressure detected: {current_mb:.1f}MB / {limit_mb:.1f}MB ({pct:.1f}%)"
        )

    def on_complete(self, total_chunks: int, total_bytes: int, duration_seconds: float) -> None:
        """Log completion."""
        throughput_mbps = (total_bytes / (1024 * 1024)) / duration_seconds if duration_seconds > 0 else 0
        logger.info(
            f"Streaming complete: {total_chunks} chunks, {total_bytes} bytes in "
            f"{duration_seconds:.2f}s ({throughput_mbps:.2f} MB/s)"
        )


# ---------------------------------------------------------------------------
# Memory Monitoring
# ---------------------------------------------------------------------------


class MemoryMonitor:
    """Monitor memory usage with psutil.

    Tracks both process memory and system memory availability.
    Provides methods to check memory pressure and trigger garbage collection.
    """

    def __init__(self):
        self._process = psutil.Process()
        self._gc_triggered_count = 0

    def get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            logger.warning(f"Failed to get process memory: {e}")
            return 0.0

    def get_system_memory_available_mb(self) -> float:
        """Get available system memory in MB.

        Returns:
            Available memory in megabytes
        """
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            logger.warning(f"Failed to get system memory: {e}")
            return 0.0

    def is_memory_pressure(self, threshold_mb: float) -> bool:
        """Check if current memory usage exceeds threshold.

        Args:
            threshold_mb: Memory threshold in MB

        Returns:
            True if memory usage exceeds threshold
        """
        current_mb = self.get_process_memory_mb()
        return current_mb > threshold_mb

    def trigger_gc_if_needed(self, current_mb: float, limit_mb: float, threshold_pct: float = 0.80) -> bool:
        """Trigger garbage collection if memory usage exceeds threshold.

        Args:
            current_mb: Current memory usage in MB
            limit_mb: Memory limit in MB
            threshold_pct: Threshold as percentage of limit (0.0 to 1.0)

        Returns:
            True if GC was triggered
        """
        if current_mb > (limit_mb * threshold_pct):
            logger.debug(f"Triggering garbage collection (memory: {current_mb:.1f}MB / {limit_mb:.1f}MB)")
            gc.collect()
            self._gc_triggered_count += 1
            return True
        return False

    def get_metrics(self) -> dict:
        """Get memory monitoring metrics.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "process_memory_mb": self.get_process_memory_mb(),
            "system_memory_available_mb": self.get_system_memory_available_mb(),
            "gc_triggered_count": self._gc_triggered_count,
        }


# ---------------------------------------------------------------------------
# Streaming Processor
# ---------------------------------------------------------------------------


class StreamingProcessor:
    """Streaming processor for large documents.

    Processes large files iteratively to maintain minimal memory footprint.
    Monitors memory usage and adjusts processing strategy dynamically.

    Example:
        >>> processor = StreamingProcessor(config=StreamingConfig())
        >>> async for chunk in processor.stream_chunks(large_file, chunking_config, "doc-123"):
        ...     await process_chunk(chunk)
    """

    def __init__(
        self,
        config: StreamingConfig,
        memory_monitor: Optional[MemoryMonitor] = None,
    ):
        """Initialize streaming processor.

        Args:
            config: Streaming configuration
            memory_monitor: Optional memory monitor (creates one if not provided)
        """
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()

        # Metrics
        self.files_streamed = 0
        self.total_bytes_streamed = 0
        self.total_chunks_created = 0

    async def should_stream(self, file_path: Path) -> bool:
        """Determine if file should be processed via streaming.

        Args:
            file_path: Path to file

        Returns:
            True if file size exceeds threshold
        """
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            should_stream = file_size_mb > self.config.file_size_threshold_mb

            if should_stream:
                logger.info(
                    f"File {file_path.name} ({file_size_mb:.1f}MB) exceeds threshold "
                    f"({self.config.file_size_threshold_mb}MB), will use streaming"
                )

            return should_stream
        except Exception as e:
            logger.warning(f"Failed to check file size for {file_path}: {e}")
            return False

    async def stream_chunks(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig,
        parent_document_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[DocumentChunk]:
        """Stream document chunks without loading full file into memory.

        Yields chunks as they are produced, enabling downstream processing
        without waiting for complete document normalization.

        Args:
            file_path: Path to file to stream
            chunking_config: Chunking configuration
            parent_document_id: Parent document ID for chunk references
            progress_callback: Optional progress callback

        Returns:
            List of document chunks
        """
        import time
        start_time = time.time()

        chunks: List[DocumentChunk] = []
        file_size_bytes = file_path.stat().st_size

        # Use logging callback if none provided and progress tracking enabled
        if progress_callback is None and self.config.enable_progress_tracking:
            progress_callback = LoggingProgressCallback()

        async for chunk in self._iterative_chunk(
            file_path, chunking_config, parent_document_id, file_size_bytes, progress_callback
        ):
            chunks.append(chunk)

        # Update metrics
        self.files_streamed += 1
        self.total_bytes_streamed += file_size_bytes
        self.total_chunks_created += len(chunks)

        # Report completion
        if progress_callback:
            duration = time.time() - start_time
            progress_callback.on_complete(len(chunks), file_size_bytes, duration)

        logger.info(
            f"Streaming complete for {file_path.name}: {len(chunks)} chunks, "
            f"{file_size_bytes / (1024 * 1024):.1f}MB"
        )

        return chunks

    async def _iterative_chunk(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig,
        parent_document_id: str,
        file_size_bytes: int,
        progress_callback: Optional[ProgressCallback],
    ) -> AsyncIterator[DocumentChunk]:
        """Iteratively read and chunk file.

        Args:
            file_path: Path to file
            chunking_config: Chunking configuration
            parent_document_id: Parent document ID
            file_size_bytes: Total file size in bytes
            progress_callback: Optional progress callback

        Yields:
            DocumentChunk objects as they are created
        """
        bytes_processed = 0
        chunk_index = 0

        # Select chunking strategy
        if chunking_config.strategy == ChunkingStrategy.BY_TITLE.value:
            async for chunk in self._stream_by_title(file_path, chunking_config, parent_document_id):
                chunk_index = await self._yield_chunk_with_monitoring(
                    chunk, chunk_index, bytes_processed, file_size_bytes, progress_callback
                )
                bytes_processed += len(chunk.content)
                yield chunk

        elif chunking_config.strategy == ChunkingStrategy.SEMANTIC.value:
            async for chunk in self._stream_semantic(file_path, chunking_config, parent_document_id):
                chunk_index = await self._yield_chunk_with_monitoring(
                    chunk, chunk_index, bytes_processed, file_size_bytes, progress_callback
                )
                bytes_processed += len(chunk.content)
                yield chunk

        else:  # BASIC or default
            async for chunk in self._stream_basic(file_path, chunking_config, parent_document_id):
                chunk_index = await self._yield_chunk_with_monitoring(
                    chunk, chunk_index, bytes_processed, file_size_bytes, progress_callback
                )
                bytes_processed += len(chunk.content)
                yield chunk

    async def _yield_chunk_with_monitoring(
        self,
        chunk: DocumentChunk,
        chunk_index: int,
        bytes_processed: int,
        file_size_bytes: int,
        progress_callback: Optional[ProgressCallback],
    ) -> int:
        """Yield chunk with memory monitoring and progress tracking.

        Args:
            chunk: Chunk to yield
            chunk_index: Current chunk index
            bytes_processed: Bytes processed so far
            file_size_bytes: Total file size
            progress_callback: Optional progress callback

        Returns:
            Updated chunk index
        """
        # Check memory periodically
        if chunk_index % self.config.memory_check_interval_chunks == 0:
            await self._check_memory_pressure()

        # Report progress
        if progress_callback and self.config.enable_progress_tracking:
            current_memory_mb = self.memory_monitor.get_process_memory_mb()
            progress_callback.on_progress(
                bytes_processed, file_size_bytes, chunk_index + 1, current_memory_mb
            )

        return chunk_index + 1

    async def _check_memory_pressure(self) -> None:
        """Check memory pressure and take appropriate action.

        Implements graceful degradation:
        1. Warning threshold (75%): Log warning
        2. GC threshold (80%): Trigger garbage collection
        3. Pause threshold (95%): Pause briefly to allow memory to free
        """
        current_mb = self.memory_monitor.get_process_memory_mb()
        limit_mb = self.config.max_memory_mb
        usage_pct = current_mb / limit_mb if limit_mb > 0 else 0

        # Warning threshold
        if usage_pct >= self.config.memory_warning_threshold_pct:
            logger.warning(
                f"Memory usage high: {current_mb:.1f}MB / {limit_mb:.1f}MB ({usage_pct * 100:.1f}%)"
            )

        # GC threshold
        if usage_pct >= self.config.memory_gc_threshold_pct:
            logger.info(f"Triggering garbage collection due to memory pressure ({usage_pct * 100:.1f}%)")
            self.memory_monitor.trigger_gc_if_needed(current_mb, limit_mb, self.config.memory_gc_threshold_pct)

        # Pause threshold
        if usage_pct >= self.config.memory_pause_threshold_pct:
            logger.warning(
                f"Memory usage critical ({usage_pct * 100:.1f}%), pausing briefly to allow memory to free"
            )
            await asyncio.sleep(0.5)  # Brief pause
            gc.collect()  # Force collection

    async def _stream_by_title(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig,
        parent_document_id: str,
    ) -> AsyncIterator[DocumentChunk]:
        """Stream chunks by title/heading boundaries.

        Args:
            file_path: Path to file
            chunking_config: Chunking configuration
            parent_document_id: Parent document ID

        Yields:
            DocumentChunk objects
        """
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        current_chunk_lines: List[str] = []
        current_title: Optional[str] = None
        current_hierarchy: List[str] = []
        chunk_index = 0
        start_char = 0

        async for buffer in self._stream_file_content(file_path):
            lines = buffer.split("\n")

            for line in lines:
                match = heading_pattern.match(line)

                if match:
                    # Found heading - finalize current chunk if exists
                    if current_chunk_lines and len("\n".join(current_chunk_lines)) >= chunking_config.min_chunk_size:
                        chunk_content = "\n".join(current_chunk_lines)
                        yield self._create_document_chunk(
                            chunk_content,
                            chunk_index,
                            parent_document_id,
                            start_char,
                            current_title,
                            current_hierarchy.copy(),
                        )
                        chunk_index += 1
                        start_char += len(chunk_content) + 1
                        current_chunk_lines = []

                    # Update hierarchy
                    heading_level = len(match.group(1))
                    heading_text = match.group(2).strip()

                    if heading_level <= len(current_hierarchy):
                        current_hierarchy = current_hierarchy[: heading_level - 1]

                    current_hierarchy.append(heading_text)
                    current_title = heading_text

                current_chunk_lines.append(line)

                # Check if chunk exceeds max size
                if len("\n".join(current_chunk_lines)) > chunking_config.max_chunk_size:
                    chunk_content = "\n".join(current_chunk_lines)
                    yield self._create_document_chunk(
                        chunk_content,
                        chunk_index,
                        parent_document_id,
                        start_char,
                        current_title,
                        current_hierarchy.copy(),
                    )
                    chunk_index += 1
                    start_char += len(chunk_content) + 1
                    current_chunk_lines = []

        # Yield final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            yield self._create_document_chunk(
                chunk_content,
                chunk_index,
                parent_document_id,
                start_char,
                current_title,
                current_hierarchy.copy(),
            )

    async def _stream_semantic(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig,
        parent_document_id: str,
    ) -> AsyncIterator[DocumentChunk]:
        """Stream chunks by semantic boundaries (paragraphs).

        Args:
            file_path: Path to file
            chunking_config: Chunking configuration
            parent_document_id: Parent document ID

        Yields:
            DocumentChunk objects
        """
        current_chunk_paras: List[str] = []
        current_size = 0
        chunk_index = 0
        start_char = 0

        async for buffer in self._stream_file_content(file_path):
            # Split by paragraph boundaries
            paragraphs = re.split(r"\n\s*\n", buffer)

            for para in paragraphs:
                para_size = len(para)

                # Check if adding paragraph would exceed max size
                if current_size + para_size > chunking_config.max_chunk_size and current_chunk_paras:
                    # Finalize current chunk
                    chunk_content = "\n\n".join(current_chunk_paras)
                    yield self._create_document_chunk(
                        chunk_content, chunk_index, parent_document_id, start_char
                    )
                    chunk_index += 1
                    start_char += len(chunk_content) + 2
                    current_chunk_paras = []
                    current_size = 0

                current_chunk_paras.append(para)
                current_size += para_size

        # Yield final chunk
        if current_chunk_paras:
            chunk_content = "\n\n".join(current_chunk_paras)
            yield self._create_document_chunk(
                chunk_content, chunk_index, parent_document_id, start_char
            )

    async def _stream_basic(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig,
        parent_document_id: str,
    ) -> AsyncIterator[DocumentChunk]:
        """Stream chunks with fixed-size chunking.

        Args:
            file_path: Path to file
            chunking_config: Chunking configuration
            parent_document_id: Parent document ID

        Yields:
            DocumentChunk objects
        """
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        chunk_size = chunking_config.max_chunk_size

        async for buffer in self._stream_file_content(file_path):
            current_chunk += buffer

            # Create chunks when we have enough content
            while len(current_chunk) >= chunk_size:
                chunk_content = current_chunk[:chunk_size]
                yield self._create_document_chunk(
                    chunk_content, chunk_index, parent_document_id, start_char
                )

                # Handle overlap
                overlap_size = chunking_config.overlap_size
                current_chunk = current_chunk[chunk_size - overlap_size :]
                chunk_index += 1
                start_char += chunk_size - overlap_size

        # Yield final chunk
        if current_chunk and len(current_chunk) >= chunking_config.min_chunk_size:
            yield self._create_document_chunk(
                current_chunk, chunk_index, parent_document_id, start_char
            )

    async def _stream_file_content(self, file_path: Path) -> AsyncIterator[str]:
        """Stream file content in buffers.

        Args:
            file_path: Path to file

        Yields:
            Buffer-sized chunks of file content
        """
        try:
            import aiofiles

            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                while True:
                    buffer = await f.read(self.config.buffer_size_bytes)
                    if not buffer:
                        break
                    yield buffer

        except ImportError:
            # Fallback to synchronous reading if aiofiles not available
            logger.warning("aiofiles not available, using synchronous file reading")
            with open(file_path, "r", encoding="utf-8") as f:
                while True:
                    buffer = f.read(self.config.buffer_size_bytes)
                    if not buffer:
                        break
                    yield buffer

    def _create_document_chunk(
        self,
        content: str,
        chunk_index: int,
        parent_document_id: str,
        start_char: int,
        section_title: Optional[str] = None,
        heading_hierarchy: Optional[List[str]] = None,
    ) -> DocumentChunk:
        """Create DocumentChunk from content.

        Args:
            content: Chunk content
            chunk_index: Chunk index
            parent_document_id: Parent document ID
            start_char: Start character offset
            section_title: Optional section title
            heading_hierarchy: Optional heading hierarchy

        Returns:
            DocumentChunk instance
        """
        chunk_id = generate_chunk_id(parent_document_id, chunk_index)
        chunk_hash = compute_chunk_hash(content, parent_document_id, chunk_index)
        word_count = len(re.findall(r"\b\w+\b", content))

        return DocumentChunk(
            chunk_id=chunk_id,
            parent_document_id=parent_document_id,
            chunk_index=chunk_index,
            content=content,
            content_hash=chunk_hash,
            start_char=start_char,
            end_char=start_char + len(content),
            section_title=section_title,
            heading_hierarchy=heading_hierarchy or [],
            character_count=len(content),
            word_count=word_count,
        )

    def get_metrics(self) -> dict:
        """Get streaming metrics.

        Returns:
            Dictionary with streaming statistics
        """
        avg_file_size_mb = (
            self.total_bytes_streamed / self.files_streamed / (1024 * 1024)
            if self.files_streamed > 0
            else 0
        )

        avg_chunks_per_file = (
            self.total_chunks_created / self.files_streamed if self.files_streamed > 0 else 0
        )

        return {
            "files_streamed": self.files_streamed,
            "total_bytes_streamed": self.total_bytes_streamed,
            "total_chunks_created": self.total_chunks_created,
            "average_file_size_mb": avg_file_size_mb,
            "average_chunks_per_file": avg_chunks_per_file,
            **self.memory_monitor.get_metrics(),
        }

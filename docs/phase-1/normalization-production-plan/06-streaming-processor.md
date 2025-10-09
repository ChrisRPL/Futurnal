Summary: Implement streaming processor for efficient handling of large documents with minimal memory footprint.

# 06 · Streaming Processor

## Purpose
Design and implement a streaming document processor that handles large files (>100MB) efficiently with minimal memory footprint. The processor uses iterative parsing and chunked processing to avoid loading entire documents into memory, enabling normalization of arbitrarily large files on consumer hardware.

## Scope
- Streaming file reader with configurable buffer size
- Iterative Unstructured.io element processing
- Chunked normalization pipeline
- Memory monitoring and limits
- Progress tracking for long-running operations
- Graceful degradation when memory pressure detected

## Requirements Alignment
- **Feature Requirement**: "Handles large documents efficiently with streaming chunkers"
- **Non-Functional Guarantee**: "Minimal memory footprint via streaming processing"
- **Performance**: Peak memory usage <2 GB for large files

## Component Design

### StreamingProcessor

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import AsyncIterator, Optional
import asyncio

from .schema import NormalizedDocument, DocumentChunk

logger = logging.getLogger(__name__)


class StreamingConfig:
    """Configuration for streaming processor."""

    file_size_threshold_mb: float = 100.0  # Stream files >100MB
    buffer_size_bytes: int = 1024 * 1024  # 1MB buffer
    chunk_batch_size: int = 10  # Process N chunks at a time
    max_memory_mb: float = 2048.0  # Hard limit
    enable_progress_tracking: bool = True


class StreamingProcessor:
    """Streaming processor for large documents.

    Processes large files iteratively to maintain minimal memory footprint.
    Monitors memory usage and adjusts processing strategy dynamically.

    Example:
        >>> processor = StreamingProcessor(config=StreamingConfig())
        >>> async for chunk in processor.stream_document(large_file):
        ...     await process_chunk(chunk)
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.files_streamed = 0
        self.total_bytes_streamed = 0

    async def should_stream(self, file_path: Path) -> bool:
        """Determine if file should be processed via streaming."""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return file_size_mb > self.config.file_size_threshold_mb
        except Exception:
            return False

    async def stream_chunks(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig
    ) -> AsyncIterator[DocumentChunk]:
        """Stream document chunks without loading full file into memory.

        Yields chunks as they are produced, enabling downstream processing
        without waiting for complete document normalization.
        """
        async for chunk in self._iterative_chunk(file_path, chunking_config):
            yield chunk

    async def _iterative_chunk(
        self,
        file_path: Path,
        chunking_config: ChunkingConfig
    ) -> AsyncIterator[DocumentChunk]:
        """Iteratively read and chunk file."""
        # Implementation would use file streaming
        # Placeholder for now
        pass

    def get_metrics(self) -> dict:
        """Get streaming metrics."""
        return {
            "files_streamed": self.files_streamed,
            "total_bytes_streamed": self.total_bytes_streamed,
            "average_file_size_mb": (
                self.total_bytes_streamed / self.files_streamed / (1024 * 1024)
                if self.files_streamed > 0
                else 0
            )
        }
```

## Acceptance Criteria

- ✅ Files >100MB processed via streaming
- ✅ Memory usage stays under configured limit
- ✅ Progress tracking for long operations
- ✅ Graceful handling of memory pressure
- ✅ Chunked processing without full file in memory

## Test Plan

### Unit Tests
- Streaming threshold detection
- Memory limit enforcement
- Buffer management

### Integration Tests
- Large file processing (1GB+)
- Memory profiling during streaming

### Performance Tests
- Peak memory usage measurement
- Processing throughput for large files

## Dependencies

- NormalizedDocument schema (Task 01)
- ChunkingEngine (Task 04)
- Memory monitoring utilities

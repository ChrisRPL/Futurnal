"""Configurable chunking engine for document segmentation.

This module provides multiple chunking strategies for segmenting documents into
semantically coherent chunks suitable for embedding and retrieval. Integrates with
Unstructured.io chunking functions while adding custom semantic boundary detection.

Key Features:
- Multiple strategies: BY_TITLE, BY_PAGE, BASIC, SEMANTIC
- Integration with Unstructured.io chunking
- Custom semantic boundary detection
- Chunk size optimization for embeddings
- Overlap configuration for context preservation
- Parent-child relationship tracking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..models import DocumentChunk, ChunkingStrategy as ModelChunkingStrategy, compute_chunk_hash, generate_chunk_id

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    BY_TITLE = "by_title"  # Preserve section boundaries
    BY_PAGE = "by_page"  # Preserve page boundaries
    BASIC = "basic"  # Simple size-based chunking
    SEMANTIC = "semantic"  # Semantic boundary detection
    NONE = "none"  # No chunking


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        strategy: Chunking strategy to use
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters to avoid tiny chunks
        overlap_size: Character overlap between adjacent chunks
        hard_max_size: Never exceed this size (for embedding limits)
        combine_short_chunks: Merge chunks below min_size
        preserve_section_boundaries: For BY_TITLE strategy
        respect_page_breaks: For BY_PAGE strategy
        semantic_similarity_threshold: For SEMANTIC chunking
        enable_streaming: For very large documents
        chunk_buffer_size: Number of chunks to buffer when streaming
    """

    strategy: str = ChunkingStrategy.BY_TITLE.value
    max_chunk_size: int = 4000  # Characters
    min_chunk_size: int = 500  # Avoid tiny chunks
    overlap_size: int = 200  # Character overlap between chunks
    hard_max_size: int = 8000  # Never exceed (for embedding limits)

    # Strategy-specific options
    combine_short_chunks: bool = True  # Merge chunks below min_size
    preserve_section_boundaries: bool = True  # For by_title
    respect_page_breaks: bool = True  # For by_page
    semantic_similarity_threshold: float = 0.75  # For semantic chunking

    # Performance
    enable_streaming: bool = False  # For very large documents
    chunk_buffer_size: int = 10  # Number of chunks to buffer


@dataclass
class IntermediateChunk:
    """Intermediate chunk representation during processing.

    Used internally before finalizing to DocumentChunk.
    """

    content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_title: Optional[str] = None
    heading_hierarchy: List[str] = None

    def __post_init__(self):
        if self.heading_hierarchy is None:
            self.heading_hierarchy = []


class ChunkingEngine:
    """Document chunking engine with configurable strategies.

    Provides multiple chunking strategies optimized for different document
    types and use cases. Integrates with Unstructured.io while adding custom
    semantic boundary detection.

    Example:
        >>> engine = ChunkingEngine()
        >>> config = ChunkingConfig(strategy=ChunkingStrategy.BY_TITLE.value)
        >>> chunks = await engine.chunk_document(
        ...     content=document_content,
        ...     config=config,
        ...     parent_document_id="doc-123"
        ... )
    """

    def __init__(self):
        self.chunks_created = 0
        self.bytes_chunked = 0

    async def chunk_document(
        self,
        *,
        content: str,
        config: ChunkingConfig,
        elements: Optional[List[dict]] = None,
        parent_document_id: str,
    ) -> List[DocumentChunk]:
        """Chunk document using specified strategy.

        Args:
            content: Full document content
            config: Chunking configuration
            elements: Unstructured.io elements (if available)
            parent_document_id: Parent document ID for chunk references

        Returns:
            List of DocumentChunk objects
        """
        if config.strategy == ChunkingStrategy.NONE.value:
            return []

        # Select chunking strategy
        if config.strategy == ChunkingStrategy.BY_TITLE.value:
            chunks = await self._chunk_by_title(content, config, elements)
        elif config.strategy == ChunkingStrategy.BY_PAGE.value:
            chunks = await self._chunk_by_page(content, config, elements)
        elif config.strategy == ChunkingStrategy.SEMANTIC.value:
            chunks = await self._chunk_semantic(content, config)
        else:  # BASIC
            chunks = await self._chunk_basic(content, config)

        # Post-process chunks
        chunks = self._apply_size_constraints(chunks, config)

        if config.overlap_size > 0:
            chunks = self._add_overlap(chunks, content, config)

        # Finalize chunks with IDs and metadata
        finalized_chunks = self._finalize_chunks(chunks, content, parent_document_id)

        # Update metrics
        self.chunks_created += len(finalized_chunks)
        self.bytes_chunked += len(content)

        logger.debug(
            f"Chunked document into {len(finalized_chunks)} chunks "
            f"using {config.strategy} strategy"
        )

        return finalized_chunks

    async def _chunk_by_title(
        self,
        content: str,
        config: ChunkingConfig,
        elements: Optional[List[dict]],
    ) -> List[IntermediateChunk]:
        """Chunk by section titles/headings.

        Args:
            content: Document content
            config: Chunking configuration
            elements: Unstructured.io elements

        Returns:
            List of intermediate chunks
        """
        chunks: List[IntermediateChunk] = []

        # Try to use Unstructured.io elements if available
        if elements:
            chunks = self._chunk_by_title_from_elements(elements, config)
        else:
            # Fallback: detect markdown-style headings
            chunks = self._chunk_by_title_from_markdown(content, config)

        return chunks

    def _chunk_by_title_from_elements(
        self, elements: List[dict], config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Chunk by titles using Unstructured.io elements.

        Args:
            elements: Unstructured.io elements
            config: Chunking configuration

        Returns:
            List of intermediate chunks
        """
        chunks: List[IntermediateChunk] = []
        current_chunk_elements: List[dict] = []
        current_title: Optional[str] = None
        current_hierarchy: List[str] = []

        for element in elements:
            element_type = element.get("type", "")

            # Check if this is a title/heading
            if "Title" in element_type or "Heading" in element_type:
                # Start new chunk if we have content
                if current_chunk_elements:
                    chunk_content = "\n\n".join(
                        el.get("text", "") for el in current_chunk_elements
                    )
                    chunks.append(
                        IntermediateChunk(
                            content=chunk_content,
                            section_title=current_title,
                            heading_hierarchy=current_hierarchy.copy(),
                        )
                    )
                    current_chunk_elements = []

                # Update title and hierarchy
                current_title = element.get("text", "")
                current_hierarchy.append(current_title)

            # Add element to current chunk
            current_chunk_elements.append(element)

        # Add final chunk
        if current_chunk_elements:
            chunk_content = "\n\n".join(
                el.get("text", "") for el in current_chunk_elements
            )
            chunks.append(
                IntermediateChunk(
                    content=chunk_content,
                    section_title=current_title,
                    heading_hierarchy=current_hierarchy.copy(),
                )
            )

        return chunks

    def _chunk_by_title_from_markdown(
        self, content: str, config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Chunk by markdown-style headings.

        Args:
            content: Document content
            config: Chunking configuration

        Returns:
            List of intermediate chunks
        """
        chunks: List[IntermediateChunk] = []

        # Split by markdown headings (# Title, ## Title, etc.)
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        lines = content.split("\n")

        current_chunk_lines: List[str] = []
        current_title: Optional[str] = None
        current_hierarchy: List[str] = []
        start_char = 0

        for line_idx, line in enumerate(lines):
            match = re.match(heading_pattern, line)

            if match:
                # Found a heading - start new chunk
                if current_chunk_lines:
                    chunk_content = "\n".join(current_chunk_lines)
                    end_char = start_char + len(chunk_content)

                    chunks.append(
                        IntermediateChunk(
                            content=chunk_content,
                            section_title=current_title,
                            heading_hierarchy=current_hierarchy.copy(),
                            start_char=start_char,
                            end_char=end_char,
                        )
                    )

                    start_char = end_char + 1  # +1 for newline
                    current_chunk_lines = []

                # Update hierarchy based on heading level
                heading_level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Maintain hierarchy depth
                if heading_level <= len(current_hierarchy):
                    current_hierarchy = current_hierarchy[: heading_level - 1]

                current_hierarchy.append(heading_text)
                current_title = heading_text

            current_chunk_lines.append(line)

        # Add final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                IntermediateChunk(
                    content=chunk_content,
                    section_title=current_title,
                    heading_hierarchy=current_hierarchy.copy(),
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                )
            )

        return chunks

    async def _chunk_by_page(
        self,
        content: str,
        config: ChunkingConfig,
        elements: Optional[List[dict]],
    ) -> List[IntermediateChunk]:
        """Chunk by page boundaries.

        Args:
            content: Document content
            config: Chunking configuration
            elements: Unstructured.io elements

        Returns:
            List of intermediate chunks
        """
        chunks: List[IntermediateChunk] = []

        if elements:
            # Group elements by page number
            pages: dict[int, List[dict]] = {}
            for element in elements:
                page_num = element.get("metadata", {}).get("page_number", 1)
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(element)

            # Create chunk per page
            for page_num in sorted(pages.keys()):
                page_elements = pages[page_num]
                page_content = "\n\n".join(el.get("text", "") for el in page_elements)

                chunks.append(
                    IntermediateChunk(
                        content=page_content,
                        start_page=page_num,
                        end_page=page_num,
                    )
                )
        else:
            # Fallback: simple size-based chunking
            chunks = await self._chunk_basic(content, config)

        return chunks

    async def _chunk_basic(
        self, content: str, config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Basic fixed-size chunking.

        Args:
            content: Document content
            config: Chunking configuration

        Returns:
            List of intermediate chunks
        """
        chunks: List[IntermediateChunk] = []
        chunk_size = config.max_chunk_size

        # Split into chunks of max_chunk_size
        for start in range(0, len(content), chunk_size):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]

            chunks.append(
                IntermediateChunk(
                    content=chunk_content, start_char=start, end_char=end
                )
            )

        return chunks

    async def _chunk_semantic(
        self, content: str, config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Semantic boundary detection chunking.

        Uses paragraph boundaries as semantic units.

        Args:
            content: Document content
            config: Chunking configuration

        Returns:
            List of intermediate chunks
        """
        chunks: List[IntermediateChunk] = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)

        current_chunk_paras: List[str] = []
        current_size = 0
        start_char = 0

        for para in paragraphs:
            para_size = len(para)

            # Check if adding this paragraph would exceed max size
            if current_size + para_size > config.max_chunk_size and current_chunk_paras:
                # Finalize current chunk
                chunk_content = "\n\n".join(current_chunk_paras)
                chunks.append(
                    IntermediateChunk(
                        content=chunk_content,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                    )
                )

                start_char += len(chunk_content) + 2  # +2 for \n\n
                current_chunk_paras = []
                current_size = 0

            current_chunk_paras.append(para)
            current_size += para_size

        # Add final chunk
        if current_chunk_paras:
            chunk_content = "\n\n".join(current_chunk_paras)
            chunks.append(
                IntermediateChunk(
                    content=chunk_content,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                )
            )

        return chunks

    def _apply_size_constraints(
        self, chunks: List[IntermediateChunk], config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Apply size constraints to chunks.

        Args:
            chunks: Intermediate chunks
            config: Chunking configuration

        Returns:
            Chunks with size constraints applied
        """
        constrained_chunks: List[IntermediateChunk] = []

        for chunk in chunks:
            # Skip empty chunks
            if not chunk.content.strip():
                continue

            # If chunk exceeds hard max, split it
            if len(chunk.content) > config.hard_max_size:
                sub_chunks = self._split_oversized_chunk(chunk, config)
                constrained_chunks.extend(sub_chunks)
            else:
                constrained_chunks.append(chunk)

        # Combine short chunks if enabled
        if config.combine_short_chunks:
            constrained_chunks = self._combine_short_chunks(constrained_chunks, config)

        return constrained_chunks

    def _split_oversized_chunk(
        self, chunk: IntermediateChunk, config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Split oversized chunk into smaller chunks.

        Args:
            chunk: Oversized chunk
            config: Chunking configuration

        Returns:
            List of smaller chunks
        """
        sub_chunks: List[IntermediateChunk] = []
        content = chunk.content
        chunk_size = config.hard_max_size

        for start in range(0, len(content), chunk_size):
            end = min(start + chunk_size, len(content))
            sub_content = content[start:end]

            sub_chunks.append(
                IntermediateChunk(
                    content=sub_content,
                    section_title=chunk.section_title,
                    heading_hierarchy=chunk.heading_hierarchy,
                )
            )

        return sub_chunks

    def _combine_short_chunks(
        self, chunks: List[IntermediateChunk], config: ChunkingConfig
    ) -> List[IntermediateChunk]:
        """Combine chunks that are below minimum size.

        Args:
            chunks: Intermediate chunks
            config: Chunking configuration

        Returns:
            Chunks with short ones combined
        """
        if not chunks:
            return chunks

        combined: List[IntermediateChunk] = []
        current_combined: Optional[IntermediateChunk] = None

        for chunk in chunks:
            if len(chunk.content) < config.min_chunk_size:
                # Combine with current
                if current_combined is None:
                    current_combined = chunk
                else:
                    # Merge content
                    current_combined.content += "\n\n" + chunk.content
            else:
                # Add previous combined if exists
                if current_combined:
                    combined.append(current_combined)
                    current_combined = None

                # Add current chunk
                combined.append(chunk)

        # Add final combined chunk
        if current_combined:
            combined.append(current_combined)

        return combined

    def _add_overlap(
        self,
        chunks: List[IntermediateChunk],
        full_content: str,
        config: ChunkingConfig,
    ) -> List[IntermediateChunk]:
        """Add overlap between adjacent chunks.

        Args:
            chunks: Intermediate chunks
            full_content: Full document content
            config: Chunking configuration

        Returns:
            Chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks: List[IntermediateChunk] = []

        for idx, chunk in enumerate(chunks):
            new_chunk = IntermediateChunk(
                content=chunk.content,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_title=chunk.section_title,
                heading_hierarchy=chunk.heading_hierarchy,
            )

            # Add overlap from previous chunk
            if idx > 0 and chunk.start_char is not None:
                overlap_start = max(0, chunk.start_char - config.overlap_size)
                overlap_text = full_content[overlap_start : chunk.start_char]
                new_chunk.content = overlap_text + new_chunk.content
                new_chunk.start_char = overlap_start

            overlapped_chunks.append(new_chunk)

        return overlapped_chunks

    def _finalize_chunks(
        self,
        chunks: List[IntermediateChunk],
        full_content: str,
        parent_document_id: str,
    ) -> List[DocumentChunk]:
        """Finalize intermediate chunks to DocumentChunk objects.

        Args:
            chunks: Intermediate chunks
            full_content: Full document content
            parent_document_id: Parent document ID

        Returns:
            List of DocumentChunk objects
        """
        finalized: List[DocumentChunk] = []

        for idx, chunk in enumerate(chunks):
            # Generate stable chunk ID
            chunk_id = generate_chunk_id(parent_document_id, idx)

            # Compute chunk hash
            chunk_hash = compute_chunk_hash(chunk.content, parent_document_id, idx)

            # Count words
            word_count = len(re.findall(r"\b\w+\b", chunk.content))

            # Create DocumentChunk
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                parent_document_id=parent_document_id,
                chunk_index=idx,
                content=chunk.content,
                content_hash=chunk_hash,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_title=chunk.section_title,
                heading_hierarchy=chunk.heading_hierarchy or [],
                character_count=len(chunk.content),
                word_count=word_count,
            )

            finalized.append(doc_chunk)

        return finalized

    def get_metrics(self) -> dict:
        """Get chunking metrics for telemetry.

        Returns:
            Dictionary with chunking statistics
        """
        return {
            "chunks_created": self.chunks_created,
            "bytes_chunked": self.bytes_chunked,
        }

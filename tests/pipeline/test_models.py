"""Tests for normalized document schema models.

Tests cover:
- DocumentFormat, ChunkingStrategy, SchemaVersion enums
- NormalizedMetadata validation and defaults
- DocumentChunk validation and relationships
- NormalizedDocument validation and properties
- Helper functions (hashing, chunk ID generation)
- to_sink_format() backward compatibility
- Size limit validators
- Datetime timezone handling
- Round-trip serialization
- Edge cases and error handling
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from futurnal.pipeline.models import (
    ChunkingStrategy,
    DocumentChunk,
    DocumentFormat,
    NormalizedDocument,
    NormalizedDocumentV1,
    NormalizedMetadata,
    SchemaVersion,
    compute_chunk_hash,
    compute_content_hash,
    generate_chunk_id,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


def test_document_format_enum():
    """Test DocumentFormat enum values."""
    assert DocumentFormat.MARKDOWN.value == "markdown"
    assert DocumentFormat.PDF.value == "pdf"
    assert DocumentFormat.HTML.value == "html"
    assert DocumentFormat.EMAIL.value == "email"
    assert DocumentFormat.UNKNOWN.value == "unknown"
    # Verify total count
    assert len(DocumentFormat) == 16


def test_chunking_strategy_enum():
    """Test ChunkingStrategy enum values."""
    assert ChunkingStrategy.BY_TITLE.value == "by_title"
    assert ChunkingStrategy.BY_PAGE.value == "by_page"
    assert ChunkingStrategy.BASIC.value == "basic"
    assert ChunkingStrategy.SEMANTIC.value == "semantic"
    assert ChunkingStrategy.NONE.value == "none"
    assert len(ChunkingStrategy) == 5


def test_schema_version_enum():
    """Test SchemaVersion enum values."""
    assert SchemaVersion.V1_0.value == "1.0"


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


def test_compute_content_hash():
    """Test content hash computation."""
    content = "Hello, world!"
    hash1 = compute_content_hash(content)

    # Should be valid SHA-256 hex
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Should be deterministic
    hash2 = compute_content_hash(content)
    assert hash1 == hash2

    # Different content should produce different hash
    hash3 = compute_content_hash("Different content")
    assert hash1 != hash3


def test_compute_content_hash_unicode():
    """Test content hash with Unicode characters."""
    content = "Hello, 世界! 🌍"
    hash_result = compute_content_hash(content)

    assert len(hash_result) == 64
    assert all(c in "0123456789abcdef" for c in hash_result)


def test_compute_chunk_hash():
    """Test chunk hash computation with position."""
    chunk_content = "This is chunk content"
    parent_hash = "a" * 64
    chunk_index = 0

    hash1 = compute_chunk_hash(chunk_content, parent_hash, chunk_index)

    # Should be valid SHA-256 hex
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Should be deterministic
    hash2 = compute_chunk_hash(chunk_content, parent_hash, chunk_index)
    assert hash1 == hash2

    # Different index should produce different hash
    hash3 = compute_chunk_hash(chunk_content, parent_hash, 1)
    assert hash1 != hash3

    # Different parent should produce different hash
    hash4 = compute_chunk_hash(chunk_content, "b" * 64, chunk_index)
    assert hash1 != hash4


def test_generate_chunk_id():
    """Test stable chunk ID generation."""
    parent_id = "doc123"
    chunk_index = 0

    chunk_id1 = generate_chunk_id(parent_id, chunk_index)

    # Should be valid UUID string
    assert len(chunk_id1) == 36
    assert chunk_id1.count("-") == 4

    # Should be deterministic (UUID5)
    chunk_id2 = generate_chunk_id(parent_id, chunk_index)
    assert chunk_id1 == chunk_id2

    # Different index should produce different ID
    chunk_id3 = generate_chunk_id(parent_id, 1)
    assert chunk_id1 != chunk_id3

    # Different parent should produce different ID
    chunk_id4 = generate_chunk_id("doc456", chunk_index)
    assert chunk_id1 != chunk_id4


# ---------------------------------------------------------------------------
# NormalizedMetadata Tests
# ---------------------------------------------------------------------------


def test_normalized_metadata_basic():
    """Test basic NormalizedMetadata creation."""
    metadata = NormalizedMetadata(
        source_path="/path/to/file.md",
        source_id="file123",
        source_type="local_files",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=1000,
        word_count=200,
        line_count=50,
        content_hash="a" * 64,
    )

    assert metadata.source_path == "/path/to/file.md"
    assert metadata.source_id == "file123"
    assert metadata.source_type == "local_files"
    assert metadata.format == DocumentFormat.MARKDOWN
    assert metadata.content_type == "text/markdown"
    assert metadata.character_count == 1000
    assert metadata.word_count == 200
    assert metadata.line_count == 50


def test_normalized_metadata_defaults():
    """Test NormalizedMetadata default values."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="b" * 64,
    )

    assert metadata.language is None
    assert metadata.language_confidence is None
    assert metadata.created_at is None
    assert metadata.modified_at is None
    assert metadata.ingested_at is not None  # Should have default
    assert metadata.file_size_bytes is None
    assert metadata.parent_hash is None
    assert metadata.is_chunked is False
    assert metadata.chunk_strategy is None
    assert metadata.total_chunks is None
    assert metadata.extra == {}
    assert metadata.frontmatter is None
    assert metadata.tags == []
    assert metadata.aliases == []
    assert metadata.normalization_version == "1.0"
    assert metadata.processing_duration_ms is None


def test_normalized_metadata_language_validation():
    """Test language code validation."""
    # Valid language code
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="c" * 64,
        language="EN",  # Should be normalized to lowercase
    )
    assert metadata.language == "en"

    # Invalid language code (too long)
    with pytest.raises(ValueError, match="Invalid ISO 639-1 language code"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="d" * 64,
            language="ENG",
        )


def test_normalized_metadata_confidence_validation():
    """Test language confidence validation."""
    # Valid confidence
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="e" * 64,
        language_confidence=0.95,
    )
    assert metadata.language_confidence == 0.95

    # Invalid confidence (too high)
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="f" * 64,
            language_confidence=1.5,
        )

    # Invalid confidence (negative)
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="1" * 64,
            language_confidence=-0.1,
        )


def test_normalized_metadata_hash_validation():
    """Test content hash validation."""
    # Valid hash
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="A" * 64,  # Should be normalized to lowercase
    )
    assert metadata.content_hash == "a" * 64

    # Invalid hash (wrong length)
    with pytest.raises(ValueError, match="must be a valid SHA-256 hex string"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="abc123",
        )

    # Invalid hash (non-hex characters)
    with pytest.raises(ValueError, match="must be a valid SHA-256 hex string"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="g" * 64,
        )


def test_normalized_metadata_extra_size_validation():
    """Test extra field size limit (10KB)."""
    # Valid extra field
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="2" * 64,
        extra={"key": "value"},
    )
    assert metadata.extra == {"key": "value"}

    # Extra field too large (>10KB)
    large_extra = {"data": "x" * 11_000}
    with pytest.raises(ValueError, match="extra field exceeds 10KB limit"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="3" * 64,
            extra=large_extra,
        )


def test_normalized_metadata_frontmatter_size_validation():
    """Test frontmatter size limit (50KB)."""
    # Valid frontmatter
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="4" * 64,
        frontmatter={"title": "Test Document"},
    )
    assert metadata.frontmatter == {"title": "Test Document"}

    # Frontmatter too large (>50KB)
    large_frontmatter = {"content": "x" * 52_000}
    with pytest.raises(ValueError, match="frontmatter field exceeds 50KB limit"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="5" * 64,
            frontmatter=large_frontmatter,
        )


def test_normalized_metadata_tags_count_validation():
    """Test tags count limit (100)."""
    # Valid tags
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="6" * 64,
        tags=["tag1", "tag2", "tag3"],
    )
    assert len(metadata.tags) == 3

    # Too many tags (>100)
    too_many_tags = [f"tag{i}" for i in range(101)]
    with pytest.raises(ValueError, match="tags list exceeds 100 entries limit"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="7" * 64,
            tags=too_many_tags,
        )


def test_normalized_metadata_aliases_count_validation():
    """Test aliases count limit (50)."""
    # Valid aliases
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="8" * 64,
        aliases=["alias1", "alias2"],
    )
    assert len(metadata.aliases) == 2

    # Too many aliases (>50)
    too_many_aliases = [f"alias{i}" for i in range(51)]
    with pytest.raises(ValueError, match="aliases list exceeds 50 entries limit"):
        NormalizedMetadata(
            source_path="/test.md",
            source_id="test",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=100,
            word_count=20,
            line_count=5,
            content_hash="9" * 64,
            aliases=too_many_aliases,
        )


def test_normalized_metadata_datetime_timezone():
    """Test datetime fields are timezone-aware."""
    # Naive datetime should be converted to UTC
    naive_dt = datetime(2025, 1, 1, 12, 0, 0)
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="0" * 64,
        created_at=naive_dt,
        modified_at=naive_dt,
    )

    assert metadata.created_at.tzinfo == timezone.utc
    assert metadata.modified_at.tzinfo == timezone.utc
    assert metadata.ingested_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# DocumentChunk Tests
# ---------------------------------------------------------------------------


def test_document_chunk_basic():
    """Test basic DocumentChunk creation."""
    chunk = DocumentChunk(
        chunk_id="chunk-123",
        parent_document_id="doc-456",
        chunk_index=0,
        content="This is chunk content",
        content_hash="abc" + "0" * 61,
        character_count=21,
        word_count=4,
    )

    assert chunk.chunk_id == "chunk-123"
    assert chunk.parent_document_id == "doc-456"
    assert chunk.chunk_index == 0
    assert chunk.content == "This is chunk content"
    assert chunk.character_count == 21
    assert chunk.word_count == 4


def test_document_chunk_defaults():
    """Test DocumentChunk default values."""
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc-1",
        chunk_index=0,
        content="Content",
        content_hash="def" + "0" * 61,
        character_count=7,
        word_count=1,
    )

    assert chunk.start_char is None
    assert chunk.end_char is None
    assert chunk.start_page is None
    assert chunk.end_page is None
    assert chunk.section_title is None
    assert chunk.heading_hierarchy == []
    assert chunk.metadata == {}


def test_document_chunk_hash_validation():
    """Test chunk content hash validation."""
    # Valid hash
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc-1",
        chunk_index=0,
        content="Content",
        content_hash="ABC" + "0" * 61,  # Should be normalized to lowercase
        character_count=7,
        word_count=1,
    )
    assert chunk.content_hash == "abc" + "0" * 61

    # Invalid hash
    with pytest.raises(ValueError, match="must be a valid SHA-256 hex string"):
        DocumentChunk(
            chunk_id="chunk-1",
            parent_document_id="doc-1",
            chunk_index=0,
            content="Content",
            content_hash="invalid",
            character_count=7,
            word_count=1,
        )


def test_document_chunk_char_range_validation():
    """Test chunk character range validation."""
    # Valid range
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc-1",
        chunk_index=0,
        content="Content",
        content_hash="1" * 64,
        character_count=7,
        word_count=1,
        start_char=0,
        end_char=100,
    )
    assert chunk.start_char == 0
    assert chunk.end_char == 100

    # Invalid range (end < start)
    with pytest.raises(ValueError, match="end_char must be >= start_char"):
        DocumentChunk(
            chunk_id="chunk-1",
            parent_document_id="doc-1",
            chunk_index=0,
            content="Content",
            content_hash="2" * 64,
            character_count=7,
            word_count=1,
            start_char=100,
            end_char=50,
        )


def test_document_chunk_page_range_validation():
    """Test chunk page range validation."""
    # Valid range
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc-1",
        chunk_index=0,
        content="Content",
        content_hash="3" * 64,
        character_count=7,
        word_count=1,
        start_page=1,
        end_page=3,
    )
    assert chunk.start_page == 1
    assert chunk.end_page == 3

    # Invalid range (end < start)
    with pytest.raises(ValueError, match="end_page must be >= start_page"):
        DocumentChunk(
            chunk_id="chunk-1",
            parent_document_id="doc-1",
            chunk_index=0,
            content="Content",
            content_hash="4" * 64,
            character_count=7,
            word_count=1,
            start_page=5,
            end_page=3,
        )


def test_document_chunk_with_structure():
    """Test DocumentChunk with structural metadata."""
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc-1",
        chunk_index=0,
        content="Section content",
        content_hash="5" * 64,
        character_count=15,
        word_count=2,
        section_title="Introduction",
        heading_hierarchy=["Chapter 1", "Section 1.1", "Introduction"],
        metadata={"importance": "high"},
    )

    assert chunk.section_title == "Introduction"
    assert len(chunk.heading_hierarchy) == 3
    assert chunk.heading_hierarchy[0] == "Chapter 1"
    assert chunk.metadata["importance"] == "high"


# ---------------------------------------------------------------------------
# NormalizedDocument Tests
# ---------------------------------------------------------------------------


def test_normalized_document_non_chunked():
    """Test non-chunked NormalizedDocument."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="6" * 64,
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="6" * 64,
        content="This is the full document content.",
        metadata=metadata,
    )

    assert doc.document_id == "doc123"
    assert doc.sha256 == "6" * 64
    assert doc.content == "This is the full document content."
    assert len(doc.chunks) == 0
    assert doc.is_chunked is False
    assert doc.total_content_length == 34


def test_normalized_document_chunked():
    """Test chunked NormalizedDocument."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="7" * 64,
        is_chunked=True,
        chunk_strategy=ChunkingStrategy.BY_TITLE,
        total_chunks=2,
    )

    chunk1 = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc123",
        chunk_index=0,
        content="First chunk content",
        content_hash="8" * 64,
        character_count=19,
        word_count=3,
    )

    chunk2 = DocumentChunk(
        chunk_id="chunk-2",
        parent_document_id="doc123",
        chunk_index=1,
        content="Second chunk content",
        content_hash="9" * 64,
        character_count=20,
        word_count=3,
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="7" * 64,
        chunks=[chunk1, chunk2],
        metadata=metadata,
    )

    assert doc.is_chunked is True
    assert len(doc.chunks) == 2
    assert doc.chunks[0].chunk_index == 0
    assert doc.chunks[1].chunk_index == 1
    assert doc.total_content_length == 39  # 19 + 20


def test_normalized_document_defaults():
    """Test NormalizedDocument default values."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="a" * 64,
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="a" * 64,
        content="Content",
        metadata=metadata,
    )

    assert doc.chunks == []
    assert doc.elements == []
    assert doc.normalized_at is not None
    assert doc.normalization_errors == []


def test_normalized_document_elements_size_validation():
    """Test elements size validation (1MB per element)."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="b" * 64,
    )

    # Valid element
    doc = NormalizedDocument(
        document_id="doc123",
        sha256="b" * 64,
        content="Content",
        metadata=metadata,
        elements=[{"type": "text", "content": "Small element"}],
    )
    assert len(doc.elements) == 1

    # Element too large (>1MB)
    large_element = {"type": "text", "content": "x" * (1_048_577)}
    with pytest.raises(ValueError, match="exceeds 1MB limit"):
        NormalizedDocument(
            document_id="doc123",
            sha256="c" * 64,
            content="Content",
            metadata=metadata,
            elements=[large_element],
        )


def test_normalized_document_to_sink_format_non_chunked():
    """Test to_sink_format() for non-chunked document."""
    metadata = NormalizedMetadata(
        source_path="/vault/note.md",
        source_id="note123",
        source_type="obsidian_vault",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="d" * 64,
        language="en",
        created_at=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        modified_at=datetime(2025, 1, 2, 11, 0, 0, tzinfo=timezone.utc),
        tags=["important", "work"],
        aliases=["My Note"],
        frontmatter={"title": "Test Note"},
        extra={"custom_field": "value"},
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="d" * 64,
        content="Full document content here.",
        metadata=metadata,
    )

    sink_format = doc.to_sink_format()

    # Verify structure
    assert sink_format["sha256"] == "d" * 64
    assert sink_format["path"] == "/vault/note.md"
    assert sink_format["source"] == "obsidian_vault"
    assert sink_format["text"] == "Full document content here."

    # Verify metadata
    assert sink_format["metadata"]["format"] == "markdown"
    assert sink_format["metadata"]["content_type"] == "text/markdown"
    assert sink_format["metadata"]["language"] == "en"
    assert sink_format["metadata"]["character_count"] == 100
    assert sink_format["metadata"]["word_count"] == 20
    assert sink_format["metadata"]["content_hash"] == "d" * 64
    assert sink_format["metadata"]["is_chunked"] is False
    assert sink_format["metadata"]["total_chunks"] is None
    assert sink_format["metadata"]["tags"] == ["important", "work"]
    assert sink_format["metadata"]["aliases"] == ["My Note"]
    assert sink_format["metadata"]["frontmatter"] == {"title": "Test Note"}
    assert sink_format["metadata"]["custom_field"] == "value"  # From extra

    # Verify datetime serialization
    assert sink_format["metadata"]["created_at"] == "2025-01-01T10:00:00+00:00"
    assert sink_format["metadata"]["modified_at"] == "2025-01-02T11:00:00+00:00"

    # Should not have chunks key
    assert "chunks" not in sink_format


def test_normalized_document_to_sink_format_chunked():
    """Test to_sink_format() for chunked document."""
    metadata = NormalizedMetadata(
        source_path="/docs/paper.pdf",
        source_id="paper123",
        source_type="local_files",
        format=DocumentFormat.PDF,
        content_type="application/pdf",
        character_count=5000,
        word_count=800,
        line_count=100,
        content_hash="e" * 64,
        is_chunked=True,
        chunk_strategy=ChunkingStrategy.BY_PAGE,
        total_chunks=3,
    )

    chunk1 = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc123",
        chunk_index=0,
        content="Page 1 content",
        content_hash="f" * 64,
        character_count=14,
        word_count=3,
        section_title="Introduction",
        heading_hierarchy=["Introduction"],
    )

    chunk2 = DocumentChunk(
        chunk_id="chunk-2",
        parent_document_id="doc123",
        chunk_index=1,
        content="Page 2 content",
        content_hash="1" * 64,
        character_count=14,
        word_count=3,
        section_title="Methods",
        heading_hierarchy=["Methods"],
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="e" * 64,
        chunks=[chunk1, chunk2],
        metadata=metadata,
    )

    sink_format = doc.to_sink_format()

    # Verify structure
    assert sink_format["sha256"] == "e" * 64
    assert sink_format["path"] == "/docs/paper.pdf"
    assert sink_format["source"] == "local_files"

    # Verify metadata
    assert sink_format["metadata"]["is_chunked"] is True
    assert sink_format["metadata"]["total_chunks"] == 2

    # Verify chunks array
    assert "chunks" in sink_format
    assert len(sink_format["chunks"]) == 2
    assert sink_format["chunks"][0]["chunk_id"] == "chunk-1"
    assert sink_format["chunks"][0]["chunk_index"] == 0
    assert sink_format["chunks"][0]["content"] == "Page 1 content"
    assert sink_format["chunks"][0]["section_title"] == "Introduction"
    assert sink_format["chunks"][1]["chunk_id"] == "chunk-2"

    # Verify text is concatenation
    assert sink_format["text"] == "Page 1 content\n\nPage 2 content"


def test_normalized_document_v1_versioning():
    """Test NormalizedDocumentV1 includes schema version."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="2" * 64,
    )

    doc = NormalizedDocumentV1(
        document_id="doc123",
        sha256="2" * 64,
        content="Content",
        metadata=metadata,
    )

    assert doc.schema_version == "1.0"


# ---------------------------------------------------------------------------
# Round-Trip Serialization Tests
# ---------------------------------------------------------------------------


def test_normalized_document_round_trip_json():
    """Test round-trip serialization through JSON."""
    metadata = NormalizedMetadata(
        source_path="/test.md",
        source_id="test",
        source_type="test",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=100,
        word_count=20,
        line_count=5,
        content_hash="3" * 64,
        language="en",
        tags=["tag1", "tag2"],
    )

    original = NormalizedDocument(
        document_id="doc123",
        sha256="3" * 64,
        content="Original content",
        metadata=metadata,
    )

    # Convert to dict
    as_dict = original.model_dump()

    # Convert to JSON string
    as_json = json.dumps(as_dict, default=str)

    # Parse back
    parsed_dict = json.loads(as_json)

    # Reconstruct model
    reconstructed = NormalizedDocument(**parsed_dict)

    # Verify equality
    assert reconstructed.document_id == original.document_id
    assert reconstructed.sha256 == original.sha256
    assert reconstructed.content == original.content
    assert reconstructed.metadata.source_path == original.metadata.source_path
    assert reconstructed.metadata.format == original.metadata.format
    assert reconstructed.metadata.tags == original.metadata.tags


def test_document_chunk_round_trip_json():
    """Test chunk round-trip serialization."""
    original = DocumentChunk(
        chunk_id="chunk-1",
        parent_document_id="doc-1",
        chunk_index=0,
        content="Chunk content",
        content_hash="4" * 64,
        character_count=13,
        word_count=2,
        section_title="Section 1",
        heading_hierarchy=["Chapter 1", "Section 1"],
    )

    # Round-trip
    as_dict = original.model_dump()
    as_json = json.dumps(as_dict, default=str)
    parsed = json.loads(as_json)
    reconstructed = DocumentChunk(**parsed)

    assert reconstructed.chunk_id == original.chunk_id
    assert reconstructed.content == original.content
    assert reconstructed.section_title == original.section_title
    assert reconstructed.heading_hierarchy == original.heading_hierarchy


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


def test_normalized_document_empty_content():
    """Test document with empty content."""
    metadata = NormalizedMetadata(
        source_path="/empty.txt",
        source_id="empty",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=0,
        word_count=0,
        line_count=0,
        content_hash="5" * 64,
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="5" * 64,
        content="",
        metadata=metadata,
    )

    assert doc.content == ""
    assert doc.total_content_length == 0
    assert doc.is_chunked is False


def test_normalized_document_unicode_content():
    """Test document with Unicode content."""
    metadata = NormalizedMetadata(
        source_path="/unicode.txt",
        source_id="unicode",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain; charset=utf-8",
        character_count=50,
        word_count=10,
        line_count=3,
        content_hash="6" * 64,
    )

    unicode_content = "Hello, 世界! 🌍 Emoji and CJK characters: 你好"

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="6" * 64,
        content=unicode_content,
        metadata=metadata,
    )

    assert doc.content == unicode_content
    assert len(doc.content) > 0

    # Verify to_sink_format handles Unicode
    sink_format = doc.to_sink_format()
    assert sink_format["text"] == unicode_content


def test_normalized_document_many_chunks():
    """Test document with many chunks."""
    metadata = NormalizedMetadata(
        source_path="/large.pdf",
        source_id="large",
        source_type="test",
        format=DocumentFormat.PDF,
        content_type="application/pdf",
        character_count=100000,
        word_count=15000,
        line_count=2000,
        content_hash="7" * 64,
        is_chunked=True,
        chunk_strategy=ChunkingStrategy.BY_PAGE,
        total_chunks=100,
    )

    # Create 100 chunks
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{i}",
            parent_document_id="doc123",
            chunk_index=i,
            content=f"Page {i+1} content",
            content_hash=compute_chunk_hash(f"Page {i+1} content", "7" * 64, i),
            character_count=len(f"Page {i+1} content"),
            word_count=3,
        )
        for i in range(100)
    ]

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="7" * 64,
        chunks=chunks,
        metadata=metadata,
    )

    assert doc.is_chunked is True
    assert len(doc.chunks) == 100
    assert doc.chunks[0].chunk_index == 0
    assert doc.chunks[99].chunk_index == 99

    # Verify to_sink_format handles many chunks
    sink_format = doc.to_sink_format()
    assert len(sink_format["chunks"]) == 100


def test_normalized_document_no_optional_fields():
    """Test document with minimal required fields only."""
    metadata = NormalizedMetadata(
        source_path="/minimal.txt",
        source_id="minimal",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=10,
        word_count=2,
        line_count=1,
        content_hash="8" * 64,
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256="8" * 64,
        content="Minimal doc",
        metadata=metadata,
    )

    # Verify all optional fields have defaults
    assert doc.chunks == []
    assert doc.elements == []
    assert doc.normalization_errors == []
    assert metadata.language is None
    assert metadata.created_at is None
    assert metadata.frontmatter is None
    assert metadata.tags == []


def test_hash_collision_theoretical():
    """Test handling of theoretical hash collision scenario."""
    # While SHA-256 collisions are computationally infeasible,
    # verify system handles duplicate hashes gracefully
    metadata1 = NormalizedMetadata(
        source_path="/file1.txt",
        source_id="file1",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=10,
        word_count=2,
        line_count=1,
        content_hash="9" * 64,
    )

    metadata2 = NormalizedMetadata(
        source_path="/file2.txt",
        source_id="file2",
        source_type="test",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        character_count=10,
        word_count=2,
        line_count=1,
        content_hash="9" * 64,  # Same hash
    )

    doc1 = NormalizedDocument(
        document_id="doc1",
        sha256="9" * 64,
        content="Content A",
        metadata=metadata1,
    )

    doc2 = NormalizedDocument(
        document_id="doc2",
        sha256="9" * 64,  # Same SHA256
        content="Content B",
        metadata=metadata2,
    )

    # Both documents are valid
    assert doc1.sha256 == doc2.sha256
    assert doc1.document_id != doc2.document_id  # Different IDs prevent confusion

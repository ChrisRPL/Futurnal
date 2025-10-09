Summary: Define standardized schema for normalized documents with chunking support and provenance tracking.

# 01 · Normalized Document Schema

## Purpose
Establish the canonical schema for normalized documents that serves as the contract between normalization pipeline and downstream components (PKG storage, vector embeddings, semantic triple extraction). The schema must support chunked documents, version tracking, and rich metadata while maintaining alignment with existing `NormalizationSink` expectations.

## Scope
- Define `NormalizedDocument` schema with required and optional fields
- Design `DocumentChunk` schema for multi-chunk documents
- Specify metadata structure for enrichment pipeline outputs
- Align with existing PKG/vector store schemas
- Support for parent-child relationships in chunked documents
- Provenance and versioning metadata

## Requirements Alignment
- **Feature Requirement**: "Outputs chunked documents with standardized schema: source ID, content, metadata (timestamps, tags), provenance hash"
- **System Architecture**: Aligns with PKG expectations and vector store integration
- **Privacy-First**: Metadata-only audit logging, no content exposure

## Component Design

### NormalizedDocument Model

```python
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentFormat(str, Enum):
    """Supported document formats."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    EMAIL = "email"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    CODE = "code"
    TEXT = "text"
    JUPYTER = "jupyter"
    XML = "xml"
    RTF = "rtf"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    """Chunking strategies applied to documents."""
    BY_TITLE = "by_title"
    BY_PAGE = "by_page"
    BASIC = "basic"
    SEMANTIC = "semantic"
    NONE = "none"


class NormalizedMetadata(BaseModel):
    """Metadata extracted and enriched during normalization."""

    # Source information
    source_path: str  # Original file path
    source_id: str  # Connector-specific source ID
    source_type: str  # e.g., "obsidian_vault", "imap_mailbox", "github_repo"

    # Format and type
    format: DocumentFormat
    content_type: str  # MIME type
    language: Optional[str] = None  # ISO 639-1 code (e.g., "en", "fr")
    language_confidence: Optional[float] = None  # 0.0 to 1.0

    # Temporal metadata
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    # Size and complexity
    file_size_bytes: Optional[int] = None
    character_count: int
    word_count: int
    line_count: int

    # Provenance
    content_hash: str  # SHA-256 of content
    parent_hash: Optional[str] = None  # For tracking versions

    # Chunking metadata
    is_chunked: bool = False
    chunk_strategy: Optional[ChunkingStrategy] = None
    total_chunks: Optional[int] = None

    # Format-specific metadata (extensible)
    extra: Dict[str, Any] = Field(default_factory=dict)

    # Frontmatter/structured metadata from source
    frontmatter: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)

    # Processing metadata
    normalization_version: str = "1.0"
    processing_duration_ms: Optional[float] = None


class DocumentChunk(BaseModel):
    """Individual chunk of a larger document."""

    # Identity
    chunk_id: str  # Unique identifier for this chunk
    parent_document_id: str  # Reference to parent NormalizedDocument
    chunk_index: int  # 0-based index in document

    # Content
    content: str
    content_hash: str  # SHA-256 of chunk content

    # Positioning
    start_char: Optional[int] = None  # Character offset in original document
    end_char: Optional[int] = None
    start_page: Optional[int] = None  # For paginated documents
    end_page: Optional[int] = None

    # Structural context
    section_title: Optional[str] = None  # For by_title chunking
    heading_hierarchy: List[str] = Field(default_factory=list)

    # Size metadata
    character_count: int
    word_count: int

    # Chunk-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    """Standardized normalized document ready for PKG ingestion."""

    # Core identity
    document_id: str  # SHA-256 hash or connector-specific stable ID
    sha256: str  # Content hash for idempotency

    # Content (for non-chunked documents)
    content: Optional[str] = None  # Full content if not chunked

    # Chunks (for chunked documents)
    chunks: List[DocumentChunk] = Field(default_factory=list)

    # Metadata
    metadata: NormalizedMetadata

    # Elements from Unstructured.io (if applicable)
    elements: List[Dict[str, Any]] = Field(default_factory=list)

    # Processing status
    normalized_at: datetime = Field(default_factory=datetime.utcnow)
    normalization_errors: List[str] = Field(default_factory=list)

    @property
    def is_chunked(self) -> bool:
        """Check if document is chunked."""
        return len(self.chunks) > 0

    @property
    def total_content_length(self) -> int:
        """Get total content length."""
        if self.content:
            return len(self.content)
        return sum(len(chunk.content) for chunk in self.chunks)

    def to_sink_format(self) -> Dict[str, Any]:
        """Convert to NormalizationSink expected format.

        Returns format compatible with existing PKG/vector pipeline:
        {
            "sha256": ...,
            "path": ...,
            "source": ...,
            "metadata": {...},
            "text": ...,
            "chunks": [...]  # If chunked
        }
        """
        sink_payload = {
            "sha256": self.sha256,
            "path": self.metadata.source_path,
            "source": self.metadata.source_type,
            "metadata": {
                "format": self.metadata.format.value,
                "content_type": self.metadata.content_type,
                "language": self.metadata.language,
                "created_at": self.metadata.created_at.isoformat() if self.metadata.created_at else None,
                "modified_at": self.metadata.modified_at.isoformat() if self.metadata.modified_at else None,
                "ingested_at": self.metadata.ingested_at.isoformat(),
                "file_size": self.metadata.file_size_bytes,
                "character_count": self.metadata.character_count,
                "word_count": self.metadata.word_count,
                "content_hash": self.metadata.content_hash,
                "is_chunked": self.is_chunked,
                "total_chunks": len(self.chunks) if self.is_chunked else None,
                "tags": self.metadata.tags,
                "aliases": self.metadata.aliases,
                "frontmatter": self.metadata.frontmatter,
                **self.metadata.extra
            }
        }

        if self.is_chunked:
            sink_payload["chunks"] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "section_title": chunk.section_title,
                    "heading_hierarchy": chunk.heading_hierarchy,
                    "character_count": chunk.character_count,
                    "word_count": chunk.word_count,
                    "metadata": chunk.metadata
                }
                for chunk in self.chunks
            ]
            # For chunked documents, text is concatenation for backward compat
            sink_payload["text"] = "\n\n".join(chunk.content for chunk in self.chunks)
        else:
            sink_payload["text"] = self.content

        return sink_payload
```

### Schema Evolution Support

```python
class SchemaVersion(str, Enum):
    """Schema version tracking for backward compatibility."""
    V1_0 = "1.0"  # Initial schema
    # Future versions for non-breaking extensions


class NormalizedDocumentV1(NormalizedDocument):
    """Version 1.0 of normalized document schema.

    Provides explicit versioning for future schema migrations.
    """
    schema_version: str = Field(default=SchemaVersion.V1_0.value, const=True)
```

## Acceptance Criteria

- ✅ Schema supports both chunked and non-chunked documents
- ✅ Backward compatible with existing NormalizationSink format
- ✅ All required metadata fields captured per feature requirements
- ✅ Provenance tracking via content hashing
- ✅ Extensible metadata via `extra` field
- ✅ Parent-child relationships for chunks preserved
- ✅ Temporal metadata for versioning support
- ✅ Format-specific metadata accommodated
- ✅ Privacy-safe audit logging (metadata-only)
- ✅ Type hints for IDE support and validation

## Test Plan

### Unit Tests
- Schema validation for required/optional fields
- Content hash calculation and verification
- Chunk indexing and ordering
- `to_sink_format()` conversion accuracy
- Metadata enrichment field validation
- Language code validation
- DateTime handling and serialization

### Integration Tests
- Round-trip serialization (JSON, dict, back to model)
- Compatibility with existing NormalizationSink
- PKG storage schema alignment
- Vector store metadata alignment
- Multi-chunk document assembly

### Edge Case Tests
- Empty documents
- Very large documents (>1GB content)
- Documents with no metadata
- Malformed frontmatter handling
- Unicode and special characters
- Chunked documents with missing chunks
- Hash collision handling (theoretical)

## Implementation Notes

### Content Hashing Strategy

```python
import hashlib

def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for provenance tracking."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def compute_chunk_hash(chunk_content: str, parent_hash: str, chunk_index: int) -> str:
    """Compute deterministic hash for chunk including position."""
    combined = f"{parent_hash}:{chunk_index}:{chunk_content}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
```

### Chunk ID Generation

```python
import uuid

def generate_chunk_id(parent_document_id: str, chunk_index: int) -> str:
    """Generate stable chunk ID based on parent and index."""
    # Deterministic ID for reproducibility
    namespace = uuid.NAMESPACE_DNS
    name = f"{parent_document_id}:chunk:{chunk_index}"
    return str(uuid.uuid5(namespace, name))
```

### Metadata Size Limits

To prevent metadata bloat:
- `extra` field limited to 10KB serialized JSON
- `frontmatter` limited to 50KB
- `tags` limited to 100 entries
- `aliases` limited to 50 entries
- Individual `elements` limited to 1MB each

## Open Questions

- Should we support nested chunking (chunks of chunks)?
- How to handle documents that partially fail enrichment?
- Should language detection be optional or always run?
- What's the maximum reasonable chunk count per document?
- Should we preserve original Unstructured.io elements or transform completely?
- How to version the schema for future breaking changes?
- Should chunk boundaries be adjustable post-normalization?

## Dependencies

- Pydantic for schema validation
- SHA-256 hashing from hashlib
- Existing NormalizationSink interface
- PKG storage schema requirements
- Vector store metadata requirements
- Semantic triple extraction expectations



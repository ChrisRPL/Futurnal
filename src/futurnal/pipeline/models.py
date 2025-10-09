"""Normalized document schema for the normalization pipeline.

This module defines the canonical schema for normalized documents that serves as
the contract between the normalization pipeline and downstream components (PKG storage,
vector embeddings, semantic triple extraction). The schema supports chunked documents,
version tracking, and rich metadata while maintaining alignment with existing
NormalizationSink expectations.

Key Features:
- Support for both chunked and non-chunked documents
- Backward compatible with existing NormalizationSink format
- Provenance tracking via SHA-256 content hashing
- Extensible metadata via 'extra' field
- Parent-child relationships for chunks
- Temporal metadata for versioning
- Format-specific metadata accommodation
- Privacy-safe design (metadata-only)
- Comprehensive type hints for IDE support
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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


class SchemaVersion(str, Enum):
    """Schema version tracking for backward compatibility."""

    V1_0 = "1.0"  # Initial schema
    # Future versions for non-breaking extensions


# ---------------------------------------------------------------------------
# Metadata Models
# ---------------------------------------------------------------------------


class NormalizedMetadata(BaseModel):
    """Metadata extracted and enriched during normalization.

    Contains source information, format details, temporal metadata, provenance,
    and extensible fields for format-specific data.
    """

    # Source information
    source_path: str = Field(..., description="Original file path")
    source_id: str = Field(..., description="Connector-specific source ID")
    source_type: str = Field(
        ...,
        description="Source type (e.g., 'obsidian_vault', 'imap_mailbox', 'github_repo')",
    )

    # Format and type
    format: DocumentFormat = Field(..., description="Document format")
    content_type: str = Field(..., description="MIME type")
    language: Optional[str] = Field(
        default=None, description="ISO 639-1 language code (e.g., 'en', 'fr')"
    )
    language_confidence: Optional[float] = Field(
        default=None, description="Language detection confidence (0.0 to 1.0)"
    )

    # Temporal metadata
    created_at: Optional[datetime] = Field(
        default=None, description="Document creation timestamp"
    )
    modified_at: Optional[datetime] = Field(
        default=None, description="Last modification timestamp"
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=__import__('datetime').timezone.utc),
        description="Ingestion timestamp"
    )

    # Size and complexity
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes")
    character_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    line_count: int = Field(..., description="Line count")

    # Provenance
    content_hash: str = Field(..., description="SHA-256 hash of content")
    parent_hash: Optional[str] = Field(
        default=None, description="Parent hash for version tracking"
    )

    # Chunking metadata
    is_chunked: bool = Field(default=False, description="Whether document is chunked")
    chunk_strategy: Optional[ChunkingStrategy] = Field(
        default=None, description="Chunking strategy used"
    )
    total_chunks: Optional[int] = Field(default=None, description="Total number of chunks")

    # Format-specific metadata (extensible)
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Format-specific metadata (max 10KB)"
    )

    # Frontmatter/structured metadata from source
    frontmatter: Optional[Dict[str, Any]] = Field(
        default=None, description="Frontmatter metadata (max 50KB)"
    )
    tags: List[str] = Field(default_factory=list, description="Document tags (max 100)")
    aliases: List[str] = Field(
        default_factory=list, description="Document aliases (max 50)"
    )

    # Processing metadata
    normalization_version: str = Field(default="1.0", description="Normalization version")
    processing_duration_ms: Optional[float] = Field(
        default=None, description="Processing duration in milliseconds"
    )

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISO 639-1 language code format."""
        if v is None:
            return v
        if len(v) != 2 or not v.isalpha():
            raise ValueError(f"Invalid ISO 639-1 language code: {v}")
        return v.lower()

    @field_validator("language_confidence")
    @classmethod
    def validate_confidence(cls, v: Optional[float]) -> Optional[float]:
        """Validate confidence is between 0.0 and 1.0."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("language_confidence must be between 0.0 and 1.0")
        return v

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, v: str) -> str:
        """Validate content hash is valid SHA-256 hex."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("content_hash must be a valid SHA-256 hex string")
        return v.lower()

    @field_validator("extra")
    @classmethod
    def validate_extra_size(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extra field doesn't exceed 10KB."""
        if v:
            serialized = json.dumps(v, ensure_ascii=False)
            if len(serialized) > 10_240:  # 10KB
                raise ValueError("extra field exceeds 10KB limit")
        return v

    @field_validator("frontmatter")
    @classmethod
    def validate_frontmatter_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate frontmatter doesn't exceed 50KB."""
        if v:
            serialized = json.dumps(v, ensure_ascii=False)
            if len(serialized) > 51_200:  # 50KB
                raise ValueError("frontmatter field exceeds 50KB limit")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags_count(cls, v: List[str]) -> List[str]:
        """Validate tags list doesn't exceed 100 entries."""
        if len(v) > 100:
            raise ValueError("tags list exceeds 100 entries limit")
        return v

    @field_validator("aliases")
    @classmethod
    def validate_aliases_count(cls, v: List[str]) -> List[str]:
        """Validate aliases list doesn't exceed 50 entries."""
        if len(v) > 50:
            raise ValueError("aliases list exceeds 50 entries limit")
        return v

    @field_validator("created_at", "modified_at", "ingested_at")
    @classmethod
    def ensure_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime fields are timezone-aware (UTC)."""
        if v is None:
            return v
        if v.tzinfo is None:
            # Assume UTC for naive datetimes
            from datetime import timezone
            return v.replace(tzinfo=timezone.utc)
        return v


# ---------------------------------------------------------------------------
# Chunk Model
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """Individual chunk of a larger document.

    Represents a single chunk with identity, content, positioning information,
    and structural context.
    """

    # Identity
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    parent_document_id: str = Field(..., description="Reference to parent NormalizedDocument")
    chunk_index: int = Field(..., description="0-based index in document", ge=0)

    # Content
    content: str = Field(..., description="Chunk content")
    content_hash: str = Field(..., description="SHA-256 hash of chunk content")

    # Positioning
    start_char: Optional[int] = Field(
        default=None, description="Character offset in original document", ge=0
    )
    end_char: Optional[int] = Field(
        default=None, description="End character offset in original document", ge=0
    )
    start_page: Optional[int] = Field(
        default=None, description="Start page for paginated documents", ge=1
    )
    end_page: Optional[int] = Field(
        default=None, description="End page for paginated documents", ge=1
    )

    # Structural context
    section_title: Optional[str] = Field(
        default=None, description="Section title for by_title chunking"
    )
    heading_hierarchy: List[str] = Field(
        default_factory=list, description="Heading hierarchy for context"
    )

    # Size metadata
    character_count: int = Field(..., description="Character count", ge=0)
    word_count: int = Field(..., description="Word count", ge=0)

    # Chunk-specific metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Chunk-specific metadata"
    )

    @field_validator("content_hash")
    @classmethod
    def validate_chunk_hash(cls, v: str) -> str:
        """Validate chunk hash is valid SHA-256 hex."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("content_hash must be a valid SHA-256 hex string")
        return v.lower()

    @field_validator("end_char")
    @classmethod
    def validate_char_range(cls, v: Optional[int], info) -> Optional[int]:
        """Validate end_char is after start_char."""
        if v is not None and "start_char" in info.data:
            start_char = info.data["start_char"]
            if start_char is not None and v < start_char:
                raise ValueError("end_char must be >= start_char")
        return v

    @field_validator("end_page")
    @classmethod
    def validate_page_range(cls, v: Optional[int], info) -> Optional[int]:
        """Validate end_page is >= start_page."""
        if v is not None and "start_page" in info.data:
            start_page = info.data["start_page"]
            if start_page is not None and v < start_page:
                raise ValueError("end_page must be >= start_page")
        return v


# ---------------------------------------------------------------------------
# Main Document Model
# ---------------------------------------------------------------------------


class NormalizedDocument(BaseModel):
    """Standardized normalized document ready for PKG ingestion.

    Represents a fully normalized document with content (or chunks), metadata,
    Unstructured.io elements, and processing status.
    """

    # Core identity
    document_id: str = Field(..., description="SHA-256 hash or connector-specific stable ID")
    sha256: str = Field(..., description="Content hash for idempotency")

    # Content (for non-chunked documents)
    content: Optional[str] = Field(default=None, description="Full content if not chunked")

    # Chunks (for chunked documents)
    chunks: List[DocumentChunk] = Field(
        default_factory=list, description="Document chunks if chunked"
    )

    # Metadata
    metadata: NormalizedMetadata = Field(..., description="Structured metadata")

    # Elements from Unstructured.io (if applicable)
    elements: List[Dict[str, Any]] = Field(
        default_factory=list, description="Unstructured.io elements"
    )

    # Processing status
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=__import__('datetime').timezone.utc),
        description="Normalization timestamp"
    )
    normalization_errors: List[str] = Field(
        default_factory=list, description="Non-fatal errors during normalization"
    )

    @field_validator("sha256", "document_id")
    @classmethod
    def validate_hash_format(cls, v: str) -> str:
        """Validate hash fields are valid SHA-256 hex."""
        if len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower()):
            return v.lower()
        # document_id can be other formats, but if it's hash-like, validate
        return v

    @field_validator("elements")
    @classmethod
    def validate_elements_size(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate individual elements don't exceed 1MB."""
        for idx, element in enumerate(v):
            serialized = json.dumps(element, ensure_ascii=False)
            if len(serialized) > 1_048_576:  # 1MB
                raise ValueError(f"Element at index {idx} exceeds 1MB limit")
        return v

    @field_validator("normalized_at")
    @classmethod
    def ensure_normalized_at_timezone(cls, v: datetime) -> datetime:
        """Ensure normalized_at is timezone-aware (UTC)."""
        if v.tzinfo is None:
            from datetime import timezone
            return v.replace(tzinfo=timezone.utc)
        return v

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

        Returns:
            Dictionary compatible with NormalizationSink.handle() expectations
        """
        sink_payload = {
            "sha256": self.sha256,
            "path": self.metadata.source_path,
            "source": self.metadata.source_type,
            "metadata": {
                "format": self.metadata.format.value,
                "content_type": self.metadata.content_type,
                "language": self.metadata.language,
                "created_at": (
                    self.metadata.created_at.isoformat()
                    if self.metadata.created_at
                    else None
                ),
                "modified_at": (
                    self.metadata.modified_at.isoformat()
                    if self.metadata.modified_at
                    else None
                ),
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
                **self.metadata.extra,
            },
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
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks
            ]
            # For chunked documents, text is concatenation for backward compat
            sink_payload["text"] = "\n\n".join(chunk.content for chunk in self.chunks)
        else:
            sink_payload["text"] = self.content

        return sink_payload


# ---------------------------------------------------------------------------
# Versioned Document Model
# ---------------------------------------------------------------------------


class NormalizedDocumentV1(NormalizedDocument):
    """Version 1.0 of normalized document schema.

    Provides explicit versioning for future schema migrations.
    """

    schema_version: str = Field(
        default=SchemaVersion.V1_0.value, description="Schema version"
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for provenance tracking.

    Args:
        content: Text content to hash

    Returns:
        SHA-256 hash as lowercase hex string
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_chunk_hash(chunk_content: str, parent_hash: str, chunk_index: int) -> str:
    """Compute deterministic hash for chunk including position.

    Args:
        chunk_content: Content of the chunk
        parent_hash: SHA-256 hash of parent document
        chunk_index: 0-based index of chunk in document

    Returns:
        SHA-256 hash as lowercase hex string
    """
    combined = f"{parent_hash}:{chunk_index}:{chunk_content}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def generate_chunk_id(parent_document_id: str, chunk_index: int) -> str:
    """Generate stable chunk ID based on parent and index.

    Uses UUID5 (SHA-1 based) with DNS namespace for deterministic,
    reproducible chunk identifiers.

    Args:
        parent_document_id: Document ID of parent document
        chunk_index: 0-based index of chunk in document

    Returns:
        UUID5 as string
    """
    namespace = uuid.NAMESPACE_DNS
    name = f"{parent_document_id}:chunk:{chunk_index}"
    return str(uuid.uuid5(namespace, name))

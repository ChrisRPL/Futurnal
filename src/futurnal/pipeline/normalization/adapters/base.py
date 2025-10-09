"""Base adapter helper class for format-specific normalizers.

Provides common functionality and utilities for adapter implementations.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ...models import DocumentFormat, NormalizedDocument, NormalizedMetadata

logger = logging.getLogger(__name__)


class BaseAdapter:
    """Base helper class for format adapters.

    Provides common functionality for creating NormalizedDocument instances
    and extracting file metadata.

    Not required but helpful for reducing boilerplate in adapters.
    """

    def __init__(self, name: str, supported_formats: List[DocumentFormat]):
        self.name = name
        self.supported_formats = supported_formats

    async def validate(self, file_path: Path) -> bool:
        """Default validation: check file exists and has supported extension.

        Args:
            file_path: Path to validate

        Returns:
            True if file exists and has valid extension
        """
        if not file_path.exists():
            return False

        # Check extension matches supported formats
        extension = file_path.suffix.lower()
        format_extensions = self._get_format_extensions()

        return extension in format_extensions

    def _get_format_extensions(self) -> set[str]:
        """Get file extensions for supported formats.

        Returns:
            Set of file extensions (with leading dot)
        """
        extension_map = {
            DocumentFormat.MARKDOWN: {".md", ".markdown"},
            DocumentFormat.PDF: {".pdf"},
            DocumentFormat.HTML: {".html", ".htm"},
            DocumentFormat.EMAIL: {".eml", ".msg"},
            DocumentFormat.DOCX: {".docx"},
            DocumentFormat.PPTX: {".pptx"},
            DocumentFormat.XLSX: {".xlsx"},
            DocumentFormat.CSV: {".csv"},
            DocumentFormat.JSON: {".json"},
            DocumentFormat.YAML: {".yaml", ".yml"},
            DocumentFormat.CODE: {".py", ".js", ".java", ".go", ".rs", ".cpp", ".c"},
            DocumentFormat.TEXT: {".txt"},
            DocumentFormat.JUPYTER: {".ipynb"},
            DocumentFormat.XML: {".xml"},
            DocumentFormat.RTF: {".rtf"},
        }

        extensions = set()
        for fmt in self.supported_formats:
            extensions.update(extension_map.get(fmt, set()))

        return extensions

    def create_normalized_document(
        self,
        *,
        content: str,
        file_path: Path,
        source_id: str,
        source_type: str,
        format: DocumentFormat,
        source_metadata: Optional[Dict] = None,
        frontmatter: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> NormalizedDocument:
        """Create a NormalizedDocument with standard metadata.

        Args:
            content: Document content
            file_path: Path to source file
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "local_files")
            format: Document format
            source_metadata: Optional source metadata
            frontmatter: Optional frontmatter dictionary
            tags: Optional list of tags
            aliases: Optional list of aliases

        Returns:
            NormalizedDocument instance
        """
        # Extract temporal metadata from file
        created_at, modified_at = self._extract_temporal_metadata(file_path)

        # Compute preliminary content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Get file size
        file_size_bytes = file_path.stat().st_size if file_path.exists() else None

        # Create metadata
        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=format,
            content_type=self._get_content_type(format),
            created_at=created_at,
            modified_at=modified_at,
            ingested_at=datetime.now(timezone.utc),
            file_size_bytes=file_size_bytes,
            character_count=len(content),
            word_count=len(content.split()),
            line_count=content.count("\n") + 1,
            content_hash=content_hash,
            frontmatter=frontmatter,
            tags=tags or [],
            aliases=aliases or [],
        )

        # Merge source metadata into extra
        if source_metadata:
            metadata.extra.update(source_metadata)

        # Create document
        document = NormalizedDocument(
            document_id=content_hash,
            sha256=content_hash,
            content=content,
            metadata=metadata,
            normalized_at=datetime.now(timezone.utc),
        )

        return document

    def _extract_temporal_metadata(
        self, file_path: Path
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract created/modified timestamps from file.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (created_at, modified_at) as timezone-aware datetimes
        """
        try:
            stat = file_path.stat()
            # Use timezone-aware UTC timestamps
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            # Try to get creation time (platform-specific)
            try:
                created_at = datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
            except AttributeError:
                # st_birthtime not available on all platforms
                created_at = modified_at

            return created_at, modified_at

        except Exception as e:
            logger.debug(f"Failed to extract temporal metadata: {e}")
            return None, None

    def _get_content_type(self, format: DocumentFormat) -> str:
        """Get MIME type for document format.

        Args:
            format: Document format

        Returns:
            MIME type string
        """
        mime_map = {
            DocumentFormat.MARKDOWN: "text/markdown",
            DocumentFormat.PDF: "application/pdf",
            DocumentFormat.HTML: "text/html",
            DocumentFormat.EMAIL: "message/rfc822",
            DocumentFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DocumentFormat.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            DocumentFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            DocumentFormat.CSV: "text/csv",
            DocumentFormat.JSON: "application/json",
            DocumentFormat.YAML: "application/x-yaml",
            DocumentFormat.CODE: "text/plain",
            DocumentFormat.TEXT: "text/plain",
            DocumentFormat.JUPYTER: "application/x-ipynb+json",
            DocumentFormat.XML: "application/xml",
            DocumentFormat.RTF: "application/rtf",
        }

        return mime_map.get(format, "application/octet-stream")

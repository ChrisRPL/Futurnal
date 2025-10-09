"""PDF format adapter.

Processes PDF documents using Unstructured.io with HI_RES strategy for
accurate text extraction, table detection, and OCR when needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class PDFAdapter(BaseAdapter):
    """Adapter for PDF documents.

    Requires Unstructured.io processing with HI_RES strategy for optimal
    accuracy including table extraction and OCR capabilities.

    Example:
        >>> adapter = PDFAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("document.pdf"),
        ...     source_id="doc-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        super().__init__(
            name="PDFAdapter",
            supported_formats=[
                DocumentFormat.PDF,
                DocumentFormat.DOCX,
                DocumentFormat.PPTX,
            ],
        )
        self.requires_unstructured_processing = True

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize PDF document.

        Note: This returns a preliminary document with basic metadata.
        The NormalizationService will call Unstructured.io processing
        to extract content since requires_unstructured_processing=True.

        Args:
            file_path: Path to PDF file
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            Preliminary NormalizedDocument (content will be added by Unstructured.io)

        Raises:
            AdapterError: If PDF validation fails
        """
        try:
            # Validate file exists
            if not file_path.exists():
                from ..registry import AdapterError

                raise AdapterError(f"PDF file not found: {file_path}")

            # Detect format from extension
            extension = file_path.suffix.lower()
            format_map = {
                ".pdf": DocumentFormat.PDF,
                ".docx": DocumentFormat.DOCX,
                ".pptx": DocumentFormat.PPTX,
            }
            format = format_map.get(extension, DocumentFormat.PDF)

            # Create preliminary document with empty content
            # Content will be filled by UnstructuredBridge
            document = self.create_normalized_document(
                content="",  # Will be populated by Unstructured.io
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=format,
                source_metadata=source_metadata,
            )

            # Add PDF-specific metadata flags
            document.metadata.extra["pdf"] = {
                "requires_ocr": self._might_require_ocr(file_path),
                "format_specific": format.value,
            }

            logger.debug(
                f"Created preliminary normalized document for {format.value}: {file_path.name}"
            )

            return document

        except Exception as e:
            logger.error(f"PDF normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(f"Failed to normalize PDF document: {str(e)}") from e

    def _might_require_ocr(self, file_path: Path) -> bool:
        """Check if PDF might require OCR.

        This is a heuristic check. Actual OCR decision is made by Unstructured.io.

        Args:
            file_path: Path to PDF

        Returns:
            True if file might need OCR (heuristic)
        """
        # Simple heuristic: if file is large relative to pages, likely has images
        # This is a rough estimate and actual OCR need is determined by Unstructured.io
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            # If PDF is >5MB, it might contain scanned images
            return file_size_mb > 5.0
        except Exception:
            return False

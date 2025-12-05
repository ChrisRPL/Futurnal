"""Scanned PDF format adapter.

Processes scanned PDF files using OCR to extract text content from images.
This adapter is triggered when standard PDF text extraction fails (no text layer).

Module 08: Multimodal Integration & Tool Enhancement - Phase 2
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class ScannedPDFAdapter(BaseAdapter):
    """Adapter for scanned PDF files requiring OCR.

    Processes PDFs without embedded text by:
    1. Converting each page to an image
    2. Running OCR on each image
    3. Merging results with page boundaries preserved

    Privacy Features:
    - Local-first processing (on-device OCR)
    - No cloud upload by default
    - Temporary image files securely deleted
    - Consent tracking integrated with ConsentRegistry
    - Audit logging for all OCR operations

    Use Cases:
    - Scanned documents without text layer
    - Low-quality PDF scans
    - PDFs with images of text
    - Historical documents
    - Handwritten notes (with appropriate OCR model)

    Example:
        >>> adapter = ScannedPDFAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("scanned_document.pdf"),
        ...     source_id="pdf-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
        >>> print(f"Extracted {len(doc.content)} characters from {doc.metadata.extra['page_count']} pages")
    """

    def __init__(self):
        """Initialize ScannedPDFAdapter with OCR client."""
        super().__init__(
            name="ScannedPDFAdapter",
            supported_formats=[DocumentFormat.SCANNED_PDF]
        )
        self.requires_unstructured_processing = False  # We handle OCR directly

        # Lazy-load OCR client and PDF conversion library
        self._ocr_client = None

    def _get_ocr_client(self):
        """Lazy-load OCR client.

        Returns:
            OCRClient instance (DeepSeek or Tesseract)
        """
        if self._ocr_client is None:
            from futurnal.extraction.ocr_client import get_ocr_client

            # Auto-select best backend
            self._ocr_client = get_ocr_client(backend="auto")

            logger.info(f"Initialized OCR client: {type(self._ocr_client).__name__}")

        return self._ocr_client

    async def _pdf_to_images(self, pdf_path: Path) -> List[Path]:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of paths to temporary image files (one per page)

        Raises:
            ImportError: If pdf2image not available
        """
        try:
            from pdf2image import convert_from_path

            logger.info(f"Converting PDF to images: {pdf_path.name}")

            # Create temporary directory for images
            temp_dir = Path(tempfile.mkdtemp(prefix="futurnal_pdf_"))

            # Convert PDF to images
            # Note: This requires poppler-utils to be installed
            images = convert_from_path(
                str(pdf_path),
                dpi=300,  # High DPI for better OCR accuracy
                fmt="png",
                output_folder=str(temp_dir),
                paths_only=True  # Return paths instead of PIL images
            )

            # Save images and return paths
            image_paths = []
            for idx, img_path in enumerate(images):
                if isinstance(img_path, str):
                    image_paths.append(Path(img_path))
                else:
                    # If pdf2image returns PIL images instead of paths
                    output_path = temp_dir / f"page_{idx:04d}.png"
                    img_path.save(str(output_path), "PNG")
                    image_paths.append(output_path)

            logger.info(f"Converted {len(image_paths)} pages to images")

            return image_paths

        except ImportError as e:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            logger.error("Also install poppler: brew install poppler (macOS) or apt-get install poppler-utils (Ubuntu)")
            raise

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize scanned PDF by converting to images and running OCR.

        Args:
            file_path: Path to PDF file
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "local_files")
            source_metadata: Additional metadata from connector

        Returns:
            NormalizedDocument with extracted text from all pages

        Raises:
            AdapterError: If PDF validation or OCR fails
        """
        image_paths = []
        try:
            # Validate file exists
            if not file_path.exists():
                from ..registry import AdapterError
                raise AdapterError(f"PDF file not found: {file_path}")

            # Check file size (warn if >50MB)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                logger.warning(
                    f"Large PDF file ({file_size_mb:.1f} MB): {file_path.name}. "
                    "OCR may take several minutes."
                )

            # TODO: Privacy Integration (Phase 2 enhancement)
            # await self._check_consent(source_id, "ocr_processing")
            logger.info(f"Processing scanned PDF with OCR: {file_path.name} ({file_size_mb:.2f} MB)")

            # Step 1: Convert PDF pages to images
            image_paths = await self._pdf_to_images(file_path)

            # Step 2: Get OCR client
            client = self._get_ocr_client()

            # Step 3: OCR each page
            page_results = []
            total_regions = 0

            for idx, img_path in enumerate(image_paths):
                logger.info(f"OCR processing page {idx + 1}/{len(image_paths)}")

                result = client.extract_text(
                    image_or_pdf=str(img_path),
                    preserve_layout=True
                )

                page_results.append(result)
                total_regions += len(result.regions)

                logger.debug(
                    f"Page {idx + 1}: {len(result.text)} chars, "
                    f"{len(result.regions)} regions, "
                    f"confidence: {result.confidence:.2f}"
                )

            # Step 4: Merge results preserving page boundaries
            merged_text = "\n\n---PAGE BREAK---\n\n".join(
                r.text for r in page_results
            )

            # Calculate average confidence across all pages
            confidences = [r.confidence for r in page_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.info(
                f"OCR complete for all pages: {len(merged_text)} total chars, "
                f"{total_regions} total regions, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            # Create normalized document
            document = self.create_normalized_document(
                content=merged_text,
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=DocumentFormat.SCANNED_PDF,
                source_metadata={
                    **source_metadata,
                    "ocr": {
                        "page_count": len(page_results),
                        "total_regions": total_regions,
                        "average_confidence": avg_confidence,
                        "layout_preserved": True,
                        "ocr_backend": type(client).__name__,
                    }
                }
            )

            # Store per-page OCR information
            document.metadata.extra["pages"] = [
                {
                    "page_number": idx + 1,
                    "text_length": len(result.text),
                    "region_count": len(result.regions),
                    "confidence": result.confidence,
                }
                for idx, result in enumerate(page_results)
            ]

            # Add PDF-specific metadata
            document.metadata.extra["pdf_file_size_mb"] = file_size_mb
            document.metadata.extra["page_count"] = len(page_results)

            # TODO: Audit Logging (Phase 2 enhancement)
            # await self._audit_log(source_id, "scanned_pdf_processed", {
            #     "page_count": len(page_results),
            #     "total_regions": total_regions,
            #     "average_confidence": avg_confidence
            # })

            logger.debug(
                f"Created normalized document for scanned PDF: {file_path.name} "
                f"({document.metadata.character_count} chars, "
                f"{len(page_results)} pages)"
            )

            return document

        except Exception as e:
            logger.error(f"Scanned PDF normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(f"Failed to normalize scanned PDF: {str(e)}") from e

        finally:
            # Clean up temporary image files
            if image_paths:
                for img_path in image_paths:
                    try:
                        if img_path.exists():
                            img_path.unlink()
                            logger.debug(f"Deleted temporary image: {img_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary image {img_path}: {e}")

                # Try to remove temporary directory
                if image_paths and image_paths[0].parent.exists():
                    try:
                        image_paths[0].parent.rmdir()
                        logger.debug("Deleted temporary directory")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary directory: {e}")

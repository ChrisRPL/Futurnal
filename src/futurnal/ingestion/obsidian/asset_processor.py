"""Asset text extraction using Unstructured.io for images and PDFs.

This module provides text extraction capabilities for embedded assets using
Unstructured.io's partition functions with OCR support for images and PDFs.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .assets import ObsidianAsset
from .security import SecurityError

logger = logging.getLogger(__name__)

# Try to import unstructured components with graceful fallback
try:
    from unstructured.partition.image import partition_image
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unstructured.io not available: {e}")
    UNSTRUCTURED_AVAILABLE = False
    partition_image = None
    partition_pdf = None


@dataclass
class AssetTextExtraction:
    """Represents extracted text from an asset with metadata."""

    asset: ObsidianAsset
    extracted_text: str
    element_count: int
    extraction_method: str  # 'ocr', 'pdf_parse', 'embedded_text'
    processing_time_ms: int
    success: bool = True
    error_message: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "asset_target": self.asset.target,
            "asset_path": str(self.asset.resolved_path) if self.asset.resolved_path else None,
            "content_hash": self.asset.content_hash,
            "extracted_text": self.extracted_text,
            "text_length": len(self.extracted_text),
            "element_count": self.element_count,
            "extraction_method": self.extraction_method,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error_message": self.error_message,
            "extracted_at": self.extracted_at.isoformat(),
            "file_size": self.asset.file_size,
            "mime_type": self.asset.mime_type
        }


class AssetTextExtractorConfig:
    """Configuration for asset text extraction."""

    def __init__(
        self,
        *,
        enable_image_ocr: bool = True,
        enable_pdf_extraction: bool = True,
        ocr_languages: str = "eng",
        max_file_size_mb: int = 50,
        processing_timeout_seconds: int = 60,
        strategy: str = "fast",  # "fast", "hi_res", "ocr_only"
        skip_single_word_elements: bool = True,
        include_metadata: bool = True
    ):
        self.enable_image_ocr = enable_image_ocr
        self.enable_pdf_extraction = enable_pdf_extraction
        self.ocr_languages = ocr_languages
        self.max_file_size_mb = max_file_size_mb
        self.processing_timeout_seconds = processing_timeout_seconds
        self.strategy = strategy
        self.skip_single_word_elements = skip_single_word_elements
        self.include_metadata = include_metadata

        # File size limit in bytes
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024


class AssetTextExtractor:
    """Extracts text from assets using Unstructured.io with configurable options."""

    def __init__(self, config: Optional[AssetTextExtractorConfig] = None):
        """Initialize asset text extractor.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AssetTextExtractorConfig()

        # Check if Unstructured.io is available
        if not UNSTRUCTURED_AVAILABLE:
            logger.warning("Unstructured.io not available - text extraction will be disabled")

        # Statistics
        self.images_processed = 0
        self.pdfs_processed = 0
        self.total_text_extracted = 0
        self.total_processing_time_ms = 0
        self.errors_encountered = 0

    def extract_text(self, asset: ObsidianAsset) -> Optional[AssetTextExtraction]:
        """Extract text from an asset file.

        Args:
            asset: Asset to extract text from

        Returns:
            AssetTextExtraction object or None if extraction not possible/configured
        """
        if not UNSTRUCTURED_AVAILABLE:
            logger.debug(f"Skipping text extraction for {asset.target}: Unstructured.io not available")
            return None

        if not asset.resolved_path or asset.is_broken:
            logger.debug(f"Skipping text extraction for {asset.target}: Asset broken or unresolved")
            return None

        if not asset.is_processable:
            logger.debug(f"Skipping text extraction for {asset.target}: Asset type not processable")
            return None

        # Check file size limits
        if asset.file_size and asset.file_size > self.config.max_file_size_bytes:
            logger.warning(
                f"Skipping {asset.target}: File size {asset.file_size} bytes exceeds limit "
                f"{self.config.max_file_size_bytes} bytes"
            )
            return self._create_error_result(
                asset,
                f"File size {asset.file_size} bytes exceeds limit {self.config.max_file_size_bytes} bytes"
            )

        try:
            if asset.is_image and self.config.enable_image_ocr:
                return self._extract_from_image(asset)
            elif asset.is_pdf and self.config.enable_pdf_extraction:
                return self._extract_from_pdf(asset)
            else:
                logger.debug(f"Skipping {asset.target}: Processing disabled for this asset type")
                return None

        except Exception as e:
            logger.error(f"Failed to extract text from {asset.target}: {e}")
            self.errors_encountered += 1
            return self._create_error_result(asset, str(e))

    def _extract_from_image(self, asset: ObsidianAsset) -> AssetTextExtraction:
        """Extract text from image using OCR."""
        start_time = datetime.utcnow()

        try:
            # Use partition_image with OCR
            elements = partition_image(
                filename=str(asset.resolved_path),
                ocr_languages=self.config.ocr_languages,
                include_metadata=self.config.include_metadata,
                skip_single_word_elements=self.config.skip_single_word_elements
            )

            # Extract text from all elements
            extracted_text = "\n".join(str(element) for element in elements if str(element).strip())

            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Update statistics
            self.images_processed += 1
            self.total_text_extracted += len(extracted_text)
            self.total_processing_time_ms += processing_time

            logger.debug(
                f"Extracted {len(extracted_text)} characters from image {asset.target} "
                f"in {processing_time}ms using OCR"
            )

            return AssetTextExtraction(
                asset=asset,
                extracted_text=extracted_text,
                element_count=len(elements),
                extraction_method="ocr",
                processing_time_ms=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.errors_encountered += 1

            logger.error(f"OCR failed for image {asset.target}: {e}")
            return self._create_error_result(asset, f"OCR failed: {e}", processing_time)

    def _extract_from_pdf(self, asset: ObsidianAsset) -> AssetTextExtraction:
        """Extract text from PDF."""
        start_time = datetime.utcnow()

        try:
            # Use partition_pdf with optional OCR for scanned PDFs
            elements = partition_pdf(
                filename=str(asset.resolved_path),
                strategy=self.config.strategy,
                ocr_languages=self.config.ocr_languages,
                include_metadata=self.config.include_metadata,
                skip_single_word_elements=self.config.skip_single_word_elements
            )

            # Extract text from all elements
            extracted_text = "\n".join(str(element) for element in elements if str(element).strip())

            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Determine extraction method based on strategy
            extraction_method = "pdf_parse" if self.config.strategy == "fast" else "pdf_ocr"

            # Update statistics
            self.pdfs_processed += 1
            self.total_text_extracted += len(extracted_text)
            self.total_processing_time_ms += processing_time

            logger.debug(
                f"Extracted {len(extracted_text)} characters from PDF {asset.target} "
                f"in {processing_time}ms using {extraction_method}"
            )

            return AssetTextExtraction(
                asset=asset,
                extracted_text=extracted_text,
                element_count=len(elements),
                extraction_method=extraction_method,
                processing_time_ms=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.errors_encountered += 1

            logger.error(f"PDF extraction failed for {asset.target}: {e}")
            return self._create_error_result(asset, f"PDF extraction failed: {e}", processing_time)

    def _create_error_result(
        self,
        asset: ObsidianAsset,
        error_message: str,
        processing_time_ms: int = 0
    ) -> AssetTextExtraction:
        """Create an error result for failed extraction."""
        return AssetTextExtraction(
            asset=asset,
            extracted_text="",
            element_count=0,
            extraction_method="error",
            processing_time_ms=processing_time_ms,
            success=False,
            error_message=error_message
        )

    def extract_batch(self, assets: List[ObsidianAsset]) -> List[AssetTextExtraction]:
        """Extract text from multiple assets.

        Args:
            assets: List of assets to process

        Returns:
            List of extraction results (successful and failed)
        """
        results = []

        for asset in assets:
            result = self.extract_text(asset)
            if result:  # Only include results where extraction was attempted
                results.append(result)

        logger.info(
            f"Batch extraction completed: {len(results)} assets processed, "
            f"{sum(1 for r in results if r.success)} successful, "
            f"{sum(1 for r in results if not r.success)} failed"
        )

        return results

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get extraction statistics.

        Returns:
            Dictionary with processing statistics
        """
        total_processed = self.images_processed + self.pdfs_processed

        return {
            "images_processed": self.images_processed,
            "pdfs_processed": self.pdfs_processed,
            "total_processed": total_processed,
            "total_text_extracted": self.total_text_extracted,
            "total_processing_time_ms": self.total_processing_time_ms,
            "average_processing_time_ms": (
                self.total_processing_time_ms / total_processed if total_processed > 0 else 0
            ),
            "errors_encountered": self.errors_encountered,
            "success_rate": (
                (total_processed - self.errors_encountered) / total_processed
                if total_processed > 0 else 0.0
            ),
            "average_text_per_asset": (
                self.total_text_extracted / total_processed if total_processed > 0 else 0
            )
        }

    def clear_statistics(self) -> None:
        """Clear processing statistics."""
        self.images_processed = 0
        self.pdfs_processed = 0
        self.total_text_extracted = 0
        self.total_processing_time_ms = 0
        self.errors_encountered = 0


class AssetProcessingPipeline:
    """Complete asset processing pipeline combining detection, resolution, and text extraction."""

    def __init__(
        self,
        vault_root: Path,
        vault_id: str,
        extractor_config: Optional[AssetTextExtractorConfig] = None
    ):
        """Initialize asset processing pipeline.

        Args:
            vault_root: Root directory of the vault
            vault_id: Unique identifier for the vault
            extractor_config: Optional text extraction configuration
        """
        from .assets import AssetPipeline  # Import here to avoid circular imports

        self.vault_root = Path(vault_root)
        self.vault_id = vault_id

        # Initialize components
        self.asset_pipeline = AssetPipeline(vault_root, vault_id)
        self.text_extractor = AssetTextExtractor(extractor_config)

    def process_document_assets(
        self,
        content: str,
        source_file_path: Path,
        include_text_extraction: bool = True
    ) -> Dict[str, Any]:
        """Process all assets in a document with optional text extraction.

        Args:
            content: Markdown content containing asset embeds
            source_file_path: Path to the source markdown file
            include_text_extraction: Whether to extract text from processable assets

        Returns:
            Dictionary with processed assets and extraction results
        """
        # Step 1: Process assets (detection, resolution, hashing)
        assets = self.asset_pipeline.process_assets(content, source_file_path)

        # Step 2: Extract text from processable assets if enabled
        text_extractions = []
        if include_text_extraction and assets:
            processable_assets = [asset for asset in assets if asset.is_processable and not asset.is_broken]
            if processable_assets:
                text_extractions = self.text_extractor.extract_batch(processable_assets)

        # Step 3: Combine results
        return {
            "assets": [asset.__dict__ for asset in assets],  # Convert to dict for serialization
            "text_extractions": [extraction.to_dict() for extraction in text_extractions],
            "statistics": {
                "total_assets": len(assets),
                "broken_assets": sum(1 for asset in assets if asset.is_broken),
                "processable_assets": sum(1 for asset in assets if asset.is_processable),
                "text_extracted_from": len(text_extractions),
                "successful_extractions": sum(1 for ext in text_extractions if ext.success),
                "failed_extractions": sum(1 for ext in text_extractions if not ext.success),
                "total_extracted_chars": sum(len(ext.extracted_text) for ext in text_extractions),
                "asset_pipeline_stats": self.asset_pipeline.get_statistics(),
                "text_extractor_stats": self.text_extractor.get_statistics()
            }
        }

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all pipeline components.

        Returns:
            Dictionary with detailed statistics
        """
        return {
            "asset_pipeline": self.asset_pipeline.get_statistics(),
            "text_extractor": self.text_extractor.get_statistics(),
            "unstructured_available": UNSTRUCTURED_AVAILABLE
        }
"""OCR Content Processor for search optimization.

Processes DeepSeek-OCR output for optimal retrieval, including:
- Layout detection and metadata extraction
- Fuzzy variant generation for OCR error tolerance
- Quality tier classification

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Option B Compliance:
- Ghost model frozen (OCR model used for extraction only)
- Local-first processing
- Quality target: >80% OCR content relevance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
    SourceMetadata,
)


class OCRLayoutType(str, Enum):
    """Document layout types from OCR processing.

    Helps determine retrieval strategy based on document structure.
    """

    SIMPLE_TEXT = "simple_text"  # Paragraphs, simple structure
    MULTI_COLUMN = "multi_column"  # Newspaper-style columns
    TABLE_HEAVY = "table_heavy"  # Many tables
    FORM = "form"  # Form fields
    MIXED = "mixed"  # Mixed content types
    HANDWRITTEN = "handwritten"  # Handwritten notes


@dataclass
class OCRContentMetadata:
    """Metadata specific to OCR-extracted content.

    Captures OCR-specific information for retrieval optimization.
    """

    source_file: str
    page_number: Optional[int]
    layout_type: OCRLayoutType
    confidence_score: float
    character_error_rate: float
    detected_language: str
    has_tables: bool
    has_images: bool
    has_handwriting: bool
    extraction_model: str
    bounding_boxes_preserved: bool


class OCRContentProcessor:
    """Processes OCR-extracted content for optimal retrieval.

    Transforms DeepSeek-OCR output into indexed format with:
    - Source metadata for confidence-weighted ranking
    - Fuzzy variants for OCR error tolerance
    - Layout information for specialized retrieval

    Integration Points:
    - MultimodalQueryHandler: OCR-specific search strategies
    - PKGClient: Stores OCR metadata with content

    Attributes:
        OCR_ERROR_PATTERNS: Common OCR character confusions
        min_confidence: Minimum confidence to include text blocks
    """

    # Common OCR error patterns for fuzzy matching
    # Maps correct character to common OCR misreadings
    OCR_ERROR_PATTERNS: Dict[str, List[str]] = {
        "l": ["1", "I", "|"],
        "I": ["1", "l", "|"],
        "O": ["0", "Q"],
        "0": ["O", "Q"],
        "m": ["rn", "nn"],
        "w": ["vv"],
        "cl": ["d"],
        "d": ["cl"],
        "rn": ["m"],
        "ri": ["n"],
        "ii": ["u"],
        " ": [""],  # Missing spaces
    }

    def __init__(self, min_confidence: float = 0.6) -> None:
        """Initialize OCR content processor.

        Args:
            min_confidence: Minimum confidence for text blocks (0.0-1.0)
        """
        self.min_confidence = min_confidence

    def process_ocr_result(self, ocr_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process DeepSeek-OCR output for PKG storage.

        Transforms raw OCR output into indexed format with metadata
        for source-aware retrieval.

        Args:
            ocr_output: Raw output from DeepSeek-OCR or OCRClient
                Expected keys: text, confidence, source_file, layout, regions

        Returns:
            Processed content with metadata for PKG storage:
            - content: Extracted text
            - source_metadata: SourceMetadata dict
            - ocr_metadata: OCR-specific metadata
            - fuzzy_variants: Error-tolerant search variants
            - bounding_boxes: Spatial information (if available)
        """
        # Extract text content
        text_content = self._extract_text(ocr_output)

        # Build metadata
        metadata = self._build_metadata(ocr_output)

        # Generate fuzzy variants for error tolerance
        fuzzy_variants = self._generate_fuzzy_variants(text_content)

        # Build source metadata for retrieval
        source_metadata = SourceMetadata(
            source_type=self._determine_source_type(ocr_output),
            extraction_confidence=metadata.confidence_score,
            extraction_quality=self._quality_tier(metadata.confidence_score),
            extractor_version=metadata.extraction_model,
            extraction_timestamp=datetime.utcnow(),
            original_format=self._detect_format(ocr_output),
            language_detected=metadata.detected_language,
            character_error_rate=metadata.character_error_rate,
            layout_complexity=metadata.layout_type.value,
        )

        return {
            "content": text_content,
            "source_metadata": source_metadata.to_dict(),
            "ocr_metadata": {
                "layout_type": metadata.layout_type.value,
                "has_tables": metadata.has_tables,
                "has_images": metadata.has_images,
                "has_handwriting": metadata.has_handwriting,
                "page_number": metadata.page_number,
                "bounding_boxes_preserved": metadata.bounding_boxes_preserved,
            },
            "fuzzy_variants": fuzzy_variants,
            "bounding_boxes": ocr_output.get("boxes", []),
        }

    def _extract_text(self, ocr_output: Dict[str, Any]) -> str:
        """Extract clean text from OCR output.

        Handles both direct text and block-based output formats.

        Args:
            ocr_output: Raw OCR output

        Returns:
            Extracted text content
        """
        # Direct text field
        if "text" in ocr_output and ocr_output["text"]:
            return ocr_output["text"]

        # Reconstruct from blocks/regions if needed
        blocks = ocr_output.get("blocks", ocr_output.get("regions", []))
        text_parts: List[str] = []

        for block in blocks:
            confidence = block.get("confidence", 1.0)
            if confidence >= self.min_confidence:
                text = block.get("text", "")
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    def _build_metadata(self, ocr_output: Dict[str, Any]) -> OCRContentMetadata:
        """Build OCR-specific metadata from output.

        Args:
            ocr_output: Raw OCR output

        Returns:
            OCRContentMetadata with extracted information
        """
        layout_info = ocr_output.get("layout", {})

        return OCRContentMetadata(
            source_file=ocr_output.get("source_file", ""),
            page_number=ocr_output.get("page", ocr_output.get("page_number")),
            layout_type=self._detect_layout_type(ocr_output),
            confidence_score=ocr_output.get("confidence", 0.0),
            character_error_rate=self._estimate_cer(ocr_output),
            detected_language=ocr_output.get("language", "en"),
            has_tables=self._has_tables(ocr_output),
            has_images=self._has_images(ocr_output),
            has_handwriting=layout_info.get("has_handwriting", False),
            extraction_model=ocr_output.get("model", "deepseek-ocr-v2"),
            bounding_boxes_preserved="boxes" in ocr_output or "regions" in ocr_output,
        )

    def _detect_layout_type(self, ocr_output: Dict[str, Any]) -> OCRLayoutType:
        """Detect document layout type from OCR output.

        Args:
            ocr_output: Raw OCR output

        Returns:
            Detected OCRLayoutType
        """
        layout_info = ocr_output.get("layout", {})

        if layout_info.get("is_handwritten", False):
            return OCRLayoutType.HANDWRITTEN

        if layout_info.get("table_count", 0) > 2:
            return OCRLayoutType.TABLE_HEAVY

        if layout_info.get("column_count", 1) > 1:
            return OCRLayoutType.MULTI_COLUMN

        if layout_info.get("is_form", False):
            return OCRLayoutType.FORM

        if layout_info.get("is_mixed", False):
            return OCRLayoutType.MIXED

        return OCRLayoutType.SIMPLE_TEXT

    def _estimate_cer(self, ocr_output: Dict[str, Any]) -> float:
        """Estimate Character Error Rate from OCR output.

        Args:
            ocr_output: Raw OCR output

        Returns:
            Estimated CER (0.0-1.0)
        """
        # Use provided CER if available
        if "cer" in ocr_output:
            return float(ocr_output["cer"])

        # Estimate from confidence: CER ≈ (1 - confidence) * 0.5
        confidence = ocr_output.get("confidence", 0.9)
        return (1 - confidence) * 0.5

    def _has_tables(self, ocr_output: Dict[str, Any]) -> bool:
        """Check if document contains tables."""
        layout = ocr_output.get("layout", {})
        return layout.get("table_count", 0) > 0

    def _has_images(self, ocr_output: Dict[str, Any]) -> bool:
        """Check if document contains images."""
        layout = ocr_output.get("layout", {})
        return layout.get("image_count", 0) > 0

    def _quality_tier(self, confidence: float) -> ExtractionQuality:
        """Map confidence to quality tier.

        Args:
            confidence: Extraction confidence (0.0-1.0)

        Returns:
            ExtractionQuality tier
        """
        return ExtractionQuality.from_confidence(confidence)

    def _determine_source_type(self, ocr_output: Dict[str, Any]) -> ContentSource:
        """Determine content source type from OCR output.

        Args:
            ocr_output: Raw OCR output

        Returns:
            ContentSource (OCR_DOCUMENT or OCR_IMAGE)
        """
        source_file = ocr_output.get("source_file", "").lower()

        if source_file.endswith(".pdf"):
            return ContentSource.OCR_DOCUMENT
        elif source_file.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
            return ContentSource.OCR_IMAGE

        # Check if explicitly marked
        source_type = ocr_output.get("source_type", "")
        if source_type == "image":
            return ContentSource.OCR_IMAGE

        # Default to document
        return ContentSource.OCR_DOCUMENT

    def _detect_format(self, ocr_output: Dict[str, Any]) -> str:
        """Detect original file format from OCR output.

        Args:
            ocr_output: Raw OCR output

        Returns:
            Format string (e.g., "pdf", "image")
        """
        source = ocr_output.get("source_file", "")
        source_lower = source.lower()

        if source_lower.endswith(".pdf"):
            return "pdf"
        elif source_lower.endswith((".png", ".jpg", ".jpeg")):
            return "image"
        elif source_lower.endswith(".tiff"):
            return "tiff"

        return ocr_output.get("format", "unknown")

    def _generate_fuzzy_variants(self, text: str) -> List[str]:
        """Generate common OCR error variants for fuzzy matching.

        Creates variants based on common OCR character confusions
        to improve search recall for OCR-extracted content.

        Args:
            text: Original text content

        Returns:
            List of variant strings (max 10 to limit explosion)
        """
        if not text:
            return []

        variants: List[str] = []
        text_lower = text.lower()

        # Generate variants based on common OCR errors
        for original, replacements in self.OCR_ERROR_PATTERNS.items():
            if original in text_lower:
                for replacement in replacements:
                    # Create one variant per replacement
                    variant = text_lower.replace(original, replacement, 1)
                    if variant != text_lower and variant not in variants:
                        variants.append(variant)

                        # Limit to prevent explosion
                        if len(variants) >= 10:
                            return variants

        return variants

    def create_searchable_content(self, text: str) -> str:
        """Create search-optimized content from OCR text.

        Normalizes text for better search matching while preserving
        important structure.

        Args:
            text: Original OCR text

        Returns:
            Normalized searchable content
        """
        if not text:
            return ""

        # Normalize whitespace
        import re

        normalized = re.sub(r"\s+", " ", text)

        # Remove common OCR artifacts
        normalized = re.sub(r"[|]{2,}", "", normalized)  # Multiple pipes
        normalized = re.sub(r"\.{4,}", "...", normalized)  # Excessive dots

        return normalized.strip()

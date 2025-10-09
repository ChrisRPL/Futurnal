"""Metadata enrichment pipeline for normalized documents.

This module provides on-device metadata extraction and enrichment including
language detection, content classification, hashing, and statistical analysis.
All processing is privacy-preserving with no external API calls.

Key Features:
- Fast language detection using fasttext-langdetect (80x faster)
- Content type classification
- SHA-256 content hashing for provenance
- Word/character/line count statistics
- Temporal metadata extraction
- Privacy-preserving error handling
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from ..models import NormalizedDocument, NormalizedMetadata, DocumentFormat

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Fast language detection using fasttext-langdetect.

    Provides 80x faster language detection compared to traditional langdetect
    with 95% accuracy for 175+ languages.

    Example:
        >>> detector = LanguageDetector()
        >>> language, confidence = await detector.detect("Hello, world!")
        >>> print(f"Language: {language} (confidence: {confidence:.2f})")
    """

    def __init__(self):
        self._detector = None
        self._ensure_detector_available()

    def _ensure_detector_available(self) -> None:
        """Load fasttext-langdetect detector.

        Falls back to simple detection if library not available.
        """
        try:
            from ftlangdetect import detect as ft_detect

            self._detector = ft_detect
            logger.debug("fasttext-langdetect loaded successfully")
        except ImportError:
            logger.warning(
                "fasttext-langdetect not available, using fallback detector. "
                "Install with: pip install fasttext-langdetect"
            )
            self._detector = None

    async def detect(self, content: str) -> Tuple[Optional[str], Optional[float]]:
        """Detect language from content.

        Args:
            content: Text content to analyze

        Returns:
            Tuple of (language_code, confidence) or (None, None) if detection fails
        """
        if not content or len(content.strip()) < 10:
            return None, None

        try:
            if self._detector:
                # Use fasttext-langdetect
                result = self._detector(text=content[:10000])  # Use first 10K chars
                if result and "lang" in result:
                    # Extract language code and confidence
                    lang = result["lang"][:2]  # ISO 639-1 (2-letter)
                    confidence = result.get("score", 0.0)
                    return lang, confidence
            else:
                # Fallback: simple heuristic detection
                return self._fallback_detect(content)

        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return None, None

        return None, None

    def _fallback_detect(self, content: str) -> Tuple[Optional[str], Optional[float]]:
        """Fallback language detection using simple heuristics.

        Args:
            content: Text content

        Returns:
            Tuple of (language_code, confidence)
        """
        # Count ASCII vs non-ASCII ratio
        ascii_chars = sum(1 for c in content if ord(c) < 128)
        total_chars = len(content)

        if total_chars == 0:
            return None, None

        ascii_ratio = ascii_chars / total_chars

        # Simple heuristic: if >90% ASCII, assume English
        if ascii_ratio > 0.9:
            return "en", 0.7

        # Otherwise, unknown
        return None, 0.0


class ContentClassifier:
    """Content type classifier using MIME types and heuristics.

    Classifies document content type based on format, structure, and patterns.

    Example:
        >>> classifier = ContentClassifier()
        >>> content_type = await classifier.classify(content, DocumentFormat.PDF)
        >>> print(f"Content type: {content_type}")
    """

    def __init__(self):
        self.classifications_performed = 0

    async def classify(
        self, content: str, format: DocumentFormat
    ) -> str:
        """Classify content type.

        Args:
            content: Document content
            format: Document format

        Returns:
            MIME type string
        """
        self.classifications_performed += 1

        # Map document format to MIME type
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


class MetadataEnrichmentPipeline:
    """Pipeline for enriching document metadata.

    Performs on-device metadata extraction and enrichment without external
    dependencies. Designed for privacy and determinism.

    Example:
        >>> pipeline = MetadataEnrichmentPipeline()
        >>> enriched_metadata = await pipeline.enrich(
        ...     document=normalized_doc,
        ...     enable_language_detection=True,
        ...     compute_hash=True
        ... )
    """

    def __init__(
        self,
        *,
        language_detector: Optional[LanguageDetector] = None,
        content_classifier: Optional[ContentClassifier] = None,
    ):
        self.language_detector = language_detector or LanguageDetector()
        self.content_classifier = content_classifier or ContentClassifier()

        # Metrics
        self.documents_enriched = 0
        self.enrichment_errors = 0

    async def enrich(
        self,
        *,
        document: NormalizedDocument,
        enable_language_detection: bool = True,
        enable_classification: bool = True,
        compute_hash: bool = True,
    ) -> NormalizedMetadata:
        """Enrich document metadata.

        Args:
            document: Normalized document to enrich
            enable_language_detection: Run language detection
            enable_classification: Run content classification
            compute_hash: Compute content hash

        Returns:
            Enriched NormalizedMetadata
        """
        metadata = document.metadata

        try:
            # Get content for analysis
            content = self._get_content_for_analysis(document)

            if not content:
                logger.warning("No content available for metadata enrichment")
                return metadata

            # Language detection
            if enable_language_detection:
                language, confidence = await self.language_detector.detect(content)
                if language:
                    metadata.language = language
                    metadata.language_confidence = confidence

            # Content classification
            if enable_classification:
                content_type = await self.content_classifier.classify(
                    content, metadata.format
                )
                if content_type:
                    metadata.content_type = content_type

            # Content hashing
            if compute_hash:
                content_hash = self._compute_content_hash(content)
                metadata.content_hash = content_hash
                document.sha256 = content_hash

            # Statistics
            metadata.character_count = len(content)
            metadata.word_count = self._count_words(content)
            metadata.line_count = content.count("\n") + 1

            # Temporal metadata (if not already set)
            if not metadata.ingested_at:
                metadata.ingested_at = datetime.now(timezone.utc)

            self.documents_enriched += 1

        except Exception as e:
            logger.error(f"Metadata enrichment failed: {type(e).__name__}")
            self.enrichment_errors += 1
            # Don't fail completely, return partially enriched metadata
            metadata.extra["enrichment_error"] = str(e)[:500]

        return metadata

    def _get_content_for_analysis(self, document: NormalizedDocument) -> str:
        """Get content string for analysis from document.

        Args:
            document: Normalized document

        Returns:
            Content string
        """
        if document.content:
            return document.content
        elif document.chunks:
            # Concatenate chunks for analysis
            return "\n\n".join(chunk.content for chunk in document.chunks)
        return ""

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hex digest
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _count_words(self, content: str) -> int:
        """Count words in content.

        Args:
            content: Text content

        Returns:
            Word count
        """
        # Split on whitespace and punctuation, filter empty strings
        words = re.findall(r"\b\w+\b", content)
        return len(words)

    def extract_temporal_metadata(self, file_path: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
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

    def get_metrics(self) -> dict:
        """Get enrichment metrics for telemetry.

        Returns:
            Dictionary with enrichment statistics
        """
        return {
            "documents_enriched": self.documents_enriched,
            "enrichment_errors": self.enrichment_errors,
            "success_rate": (
                self.documents_enriched / (self.documents_enriched + self.enrichment_errors)
                if (self.documents_enriched + self.enrichment_errors) > 0
                else 0.0
            ),
        }

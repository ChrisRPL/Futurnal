"""Metadata enrichment pipeline for normalized documents.

This module provides on-device metadata extraction and enrichment including
language detection, content classification, hashing, and statistical analysis.
All processing is privacy-preserving with no external API calls.

Key Features:
- Language detection using langdetect (offline-capable, deterministic)
- Content type classification with format-specific detection
- SHA-256 content hashing for provenance
- Word/character/line count statistics
- Temporal metadata and permissions extraction from filesystem
- Privacy-preserving error handling
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from ..models import NormalizedDocument, NormalizedMetadata, DocumentFormat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filesystem Metadata Extraction
# ---------------------------------------------------------------------------


class FilesystemMetadataExtractor:
    """Extract filesystem-level metadata from files.

    Provides static methods for extracting temporal metadata (creation/modification times)
    and file permissions. All operations are privacy-preserving and handle platform
    differences gracefully.

    Example:
        >>> created, modified, size = FilesystemMetadataExtractor.extract_temporal_metadata(
        ...     Path("document.pdf")
        ... )
        >>> permissions = FilesystemMetadataExtractor.extract_permissions(
        ...     Path("document.pdf")
        ... )
    """

    @staticmethod
    def extract_temporal_metadata(
        file_path: Path,
    ) -> Tuple[Optional[datetime], Optional[datetime], Optional[int]]:
        """Extract creation time, modification time, and file size from file.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (created_at, modified_at, file_size_bytes) as timezone-aware datetimes
            Returns (None, None, None) if file doesn't exist or error occurs
        """
        try:
            if not file_path.exists():
                logger.debug(f"File does not exist for temporal metadata extraction")
                return None, None, None

            stat = file_path.stat()

            # Modified time (available on all platforms)
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            # Creation time (platform-specific)
            try:
                # macOS and Windows
                created_at = datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
            except AttributeError:
                # Linux doesn't have st_birthtime, use st_ctime (metadata change time)
                # This is the best approximation available
                created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)

            # File size in bytes
            file_size = stat.st_size

            return created_at, modified_at, file_size

        except Exception as e:
            logger.debug(f"Failed to extract temporal metadata: {e}")
            return None, None, None

    @staticmethod
    def extract_permissions(file_path: Path) -> dict:
        """Extract file permissions metadata.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with permission metadata:
            - mode: Octal file mode (e.g., '0o100644')
            - is_readable: Boolean indicating read permission
            - is_writable: Boolean indicating write permission
            - is_executable: Boolean indicating execute permission
        """
        try:
            if not file_path.exists():
                logger.debug(f"File does not exist for permission extraction")
                return {}

            stat = file_path.stat()

            return {
                "mode": oct(stat.st_mode),
                "is_readable": os.access(file_path, os.R_OK),
                "is_writable": os.access(file_path, os.W_OK),
                "is_executable": os.access(file_path, os.X_OK),
            }

        except Exception as e:
            logger.debug(f"Failed to extract permissions: {e}")
            return {}


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
        """Load language detector.

        Uses langdetect library for reliable, production-ready language detection.
        Falls back to simple heuristic detection if library not available.
        """
        # Use langdetect (widely compatible, production-ready)
        try:
            from langdetect import detect as ld_detect, detect_langs, DetectorFactory
            # Set seed for consistent/deterministic results
            DetectorFactory.seed = 0

            def langdetect_wrapper(text, **kwargs):
                """Wrapper to provide consistent interface with confidence scores."""
                try:
                    # Get language with confidence
                    results = detect_langs(text)
                    if results:
                        # Return top result
                        top = results[0]
                        return {"lang": top.lang, "score": top.prob}
                    return None
                except Exception as e:
                    logger.debug(f"langdetect detection failed: {e}")
                    return None

            self._detector = langdetect_wrapper
            logger.debug("langdetect loaded successfully")
            return
        except ImportError as e:
            logger.debug(f"langdetect not available: {e}")

        # No detection library available
        logger.warning(
            "langdetect not available, using fallback detector. "
            "Install with: pip install langdetect"
        )
        self._detector = None

    async def detect(
        self, content: str, min_confidence: float = 0.5
    ) -> Tuple[Optional[str], Optional[float]]:
        """Detect language from content.

        Args:
            content: Text content to analyze
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            Tuple of (language_code, confidence) or (None, None) if detection fails
            or confidence is below threshold
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

                    # Apply confidence threshold
                    if confidence >= min_confidence:
                        return lang, confidence
                    else:
                        return None, None
            else:
                # Fallback: simple heuristic detection
                return self._fallback_detect(content, min_confidence)

        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return None, None

        return None, None

    def _fallback_detect(
        self, content: str, min_confidence: float = 0.5
    ) -> Tuple[Optional[str], Optional[float]]:
        """Fallback language detection using simple heuristics.

        Args:
            content: Text content
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (language_code, confidence)
        """
        # Count ASCII vs non-ASCII ratio
        ascii_chars = sum(1 for c in content if ord(c) < 128)
        total_chars = len(content)

        if total_chars == 0:
            return None, None

        ascii_ratio = ascii_chars / total_chars

        # Common English words for detection
        common_english = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "for"]
        content_lower = content.lower()

        # Count occurrences of common words
        english_word_count = sum(
            content_lower.count(f" {word} ") for word in common_english
        )

        # Heuristic: high ASCII ratio + common English words
        if ascii_ratio > 0.9 and english_word_count > 5:
            confidence = 0.7
            if confidence >= min_confidence:
                return "en", confidence

        # Otherwise, no confident detection
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
        """Classify content type with format-specific refinement.

        Args:
            content: Document content
            format: Document format

        Returns:
            MIME type string, potentially refined with format-specific variants
        """
        self.classifications_performed += 1

        # Format-specific classification for enhanced detection
        if format == DocumentFormat.MARKDOWN:
            return self._classify_markdown(content)
        elif format == DocumentFormat.CODE:
            return self._classify_code(content)
        elif format == DocumentFormat.HTML:
            return "text/html"
        elif format == DocumentFormat.PDF:
            return "application/pdf"
        elif format == DocumentFormat.EMAIL:
            return "message/rfc822"
        elif format == DocumentFormat.DOCX:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif format == DocumentFormat.PPTX:
            return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif format == DocumentFormat.XLSX:
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif format == DocumentFormat.CSV:
            return "text/csv"
        elif format == DocumentFormat.JSON:
            return "application/json"
        elif format == DocumentFormat.YAML:
            return "application/x-yaml"
        elif format == DocumentFormat.TEXT:
            return "text/plain"
        elif format == DocumentFormat.JUPYTER:
            return "application/x-ipynb+json"
        elif format == DocumentFormat.XML:
            return "application/xml"
        elif format == DocumentFormat.RTF:
            return "application/rtf"
        else:
            return "application/octet-stream"

    def _classify_markdown(self, content: str) -> str:
        """Classify markdown document with variant detection.

        Detects special markdown variants:
        - Mermaid diagrams: text/markdown+mermaid
        - Math notation: text/markdown+math
        - Standard: text/markdown

        Args:
            content: Markdown content

        Returns:
            MIME type string with variant suffix
        """
        # Check for mermaid diagrams
        if "```mermaid" in content or "```merm" in content:
            return "text/markdown+mermaid"

        # Check for math notation (LaTeX-style)
        # $$...$$ blocks or \[...\] or \(...\)
        if any(marker in content for marker in ["$$", "\\[", "\\(", "\\begin{equation}"]):
            return "text/markdown+math"

        # Standard markdown
        return "text/markdown"

    def _classify_code(self, content: str) -> str:
        """Classify code document by programming language.

        Uses heuristic pattern matching to detect:
        - Python: text/x-python
        - JavaScript/TypeScript: text/javascript
        - Java: text/x-java
        - Go: text/x-go
        - Rust: text/x-rust
        - C/C++: text/x-c or text/x-c++
        - Shell: text/x-shellscript
        - Generic: text/plain

        Args:
            content: Code content

        Returns:
            MIME type string for detected language
        """
        content_sample = content[:2000].lower()  # Analyze first 2K chars

        # Python detection
        if ("import " in content_sample and "def " in content_sample) or (
            "from " in content_sample and "import " in content_sample
        ):
            return "text/x-python"

        # JavaScript/TypeScript detection
        if any(
            keyword in content_sample
            for keyword in ["function ", "const ", "let ", "var ", "=>"]
        ):
            if "interface " in content_sample or "type " in content_sample:
                return "text/typescript"
            return "text/javascript"

        # Java detection
        if "public class " in content_sample or "public static void main" in content_sample:
            return "text/x-java"

        # Go detection
        if "package main" in content_sample or "func " in content_sample:
            return "text/x-go"

        # Rust detection
        if "fn main()" in content_sample or "pub fn " in content_sample:
            return "text/x-rust"

        # C++ detection
        if any(
            keyword in content_sample
            for keyword in ["#include <iostream>", "std::", "namespace "]
        ):
            return "text/x-c++"

        # C detection
        if "#include <stdio.h>" in content_sample or "#include <stdlib.h>" in content_sample:
            return "text/x-c"

        # Shell script detection
        if content_sample.startswith("#!/bin/bash") or content_sample.startswith("#!/bin/sh"):
            return "text/x-shellscript"

        # Default to plain text for code
        return "text/plain"


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
        file_path: Optional[Path] = None,
    ) -> NormalizedMetadata:
        """Enrich document metadata.

        Args:
            document: Normalized document to enrich
            enable_language_detection: Run language detection
            enable_classification: Run content classification
            compute_hash: Compute content hash
            file_path: Optional path to source file for temporal metadata extraction

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

            # Temporal metadata extraction from filesystem (if file_path provided)
            if file_path:
                created_at, modified_at, file_size = (
                    FilesystemMetadataExtractor.extract_temporal_metadata(file_path)
                )
                if created_at:
                    metadata.created_at = created_at
                if modified_at:
                    metadata.modified_at = modified_at
                if file_size is not None:
                    metadata.file_size_bytes = file_size

                # Extract permissions metadata (optional, stored in extra)
                permissions = FilesystemMetadataExtractor.extract_permissions(file_path)
                if permissions:
                    metadata.extra["file_permissions"] = permissions

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

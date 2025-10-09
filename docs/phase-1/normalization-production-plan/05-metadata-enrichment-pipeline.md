Summary: Implement metadata enrichment with language detection, content classification, and provenance hashing.

# 05 · Metadata Enrichment Pipeline

## Purpose
Design and implement the metadata enrichment pipeline that augments normalized documents with derived metadata: language detection, content classification, content hashing, and custom metadata extraction. This pipeline operates on-device, preserves privacy, and provides deterministic outputs for idempotency.

## Scope
- Language detection using fasttext-langdetect (80x faster)
- Content type classification
- SHA-256 content hashing for provenance
- Temporal metadata extraction (created/modified dates)
- Word count, character count, line count statistics
- Format-specific metadata extraction
- Privacy-preserving metadata only (no content in logs)

## Requirements Alignment
- **Feature Requirement**: "Metadata enrichment (language detection, content type, content hash)"
- **Implementation Guide**: "Integrate language detection, sentiment tags (optional) while staying on-device"
- **Privacy-First**: All processing on-device, no external API calls

## Component Design

### MetadataEnrichmentPipeline

```python
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schema import NormalizedDocument, NormalizedMetadata

logger = logging.getLogger(__name__)


class MetadataEnrichmentPipeline:
    """Pipeline for enriching document metadata.

    Performs on-device metadata extraction and enrichment without
    external dependencies. Designed for privacy and determinism.

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
            metadata.line_count = content.count('\n') + 1

            # Temporal metadata (if not already set)
            if not metadata.ingested_at:
                metadata.ingested_at = datetime.utcnow()

            self.documents_enriched += 1

        except Exception as e:
            logger.error(f"Metadata enrichment failed: {e}")
            self.enrichment_errors += 1
            # Don't fail completely, return partially enriched metadata
            metadata.extra["enrichment_error"] = str(e)

        return metadata

    def _get_content_for_analysis(self, document: NormalizedDocument) -> str:
        """Get content string for analysis from document."""
        if document.content:
            return document.content
        elif document.chunks:
            # Concatenate chunks for analysis
            return "\n\n".join(chunk.content for chunk in document.chunks)
        return ""

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _count_words(self, content: str) -> int:
        """Count words in content."""
        # Simple word counting (can be enhanced with language-specific tokenization)
        return len(content.split())

    def get_metrics(self) -> dict:
        """Get enrichment metrics for telemetry."""
        return {
            "documents_enriched": self.documents_enriched,
            "enrichment_errors": self.enrichment_errors,
            "success_rate": (
                self.documents_enriched / (self.documents_enriched + self.enrichment_errors)
                if (self.documents_enriched + self.enrichment_errors) > 0
                else 0
            )
        }
```

### LanguageDetector

```python
class LanguageDetector:
    """On-device language detection using fasttext-langdetect.

    Uses fasttext model for 80x faster language detection with 95% accuracy.
    Operates completely offline with no external dependencies.
    """

    def __init__(self):
        self._detector = None
        self._load_detector()

    def _load_detector(self):
        """Load fasttext language detection model."""
        try:
            from fasttext_langdetect import detect as ft_detect
            self._detector = ft_detect
            logger.info("Loaded fasttext-langdetect model")
        except ImportError:
            logger.warning(
                "fasttext-langdetect not available, falling back to simple detection"
            )
            self._detector = None

    async def detect(self, content: str, min_confidence: float = 0.5) -> tuple[Optional[str], Optional[float]]:
        """Detect language of content.

        Args:
            content: Text content to analyze
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (language_code, confidence) or (None, None)
        """
        if not content or len(content) < 50:
            # Too short for reliable detection
            return None, None

        try:
            if self._detector:
                # Use fasttext-langdetect
                result = self._detector(content[:1000])  # Sample first 1000 chars
                language = result.get('lang')
                confidence = result.get('score', 0.0)

                if confidence >= min_confidence:
                    return language, confidence

            else:
                # Fallback: simple heuristic detection
                language = self._simple_detect(content)
                return language, 0.8  # Moderate confidence for heuristic

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

        return None, None

    def _simple_detect(self, content: str) -> str:
        """Simple heuristic language detection as fallback."""
        # Count common English words as proxy
        english_common = ['the', 'and', 'is', 'in', 'to', 'of', 'a']
        content_lower = content.lower()

        english_score = sum(
            content_lower.count(f' {word} ') for word in english_common
        )

        # Simple heuristic: assume English if high score
        if english_score > 10:
            return 'en'

        return 'unknown'
```

### ContentClassifier

```python
from .schema import DocumentFormat


class ContentClassifier:
    """Content type classifier for refined MIME type detection."""

    def __init__(self):
        self.classifications = 0

    async def classify(
        self,
        content: str,
        document_format: DocumentFormat
    ) -> Optional[str]:
        """Classify content type more specifically.

        Args:
            content: Document content
            document_format: Detected document format

        Returns:
            Refined MIME type or None
        """
        self.classifications += 1

        # Format-specific classification
        if document_format == DocumentFormat.MARKDOWN:
            return self._classify_markdown(content)
        elif document_format == DocumentFormat.CODE:
            return self._classify_code(content)
        elif document_format == DocumentFormat.HTML:
            return "text/html"
        elif document_format == DocumentFormat.PDF:
            return "application/pdf"
        elif document_format == DocumentFormat.EMAIL:
            return "message/rfc822"
        else:
            return "text/plain"

    def _classify_markdown(self, content: str) -> str:
        """Classify markdown document subtype."""
        # Check for specific markdown variants
        if '```mermaid' in content:
            return "text/markdown+mermaid"
        elif any(x in content for x in ['$$', '\\[', '\\(']):
            return "text/markdown+math"
        else:
            return "text/markdown"

    def _classify_code(self, content: str) -> str:
        """Classify code document by language."""
        # Simple heuristics for language detection
        if 'import ' in content and 'def ' in content:
            return "text/x-python"
        elif 'function ' in content or 'const ' in content:
            return "text/javascript"
        elif 'public class ' in content:
            return "text/x-java"
        elif 'package main' in content:
            return "text/x-go"
        else:
            return "text/plain"
```

### FilesystemMetadataExtractor

```python
import os
from datetime import datetime
from pathlib import Path


class FilesystemMetadataExtractor:
    """Extract filesystem-level metadata from files."""

    @staticmethod
    def extract_temporal_metadata(file_path: Path) -> dict:
        """Extract creation and modification times.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with created_at and modified_at timestamps
        """
        try:
            stat = file_path.stat()

            return {
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "file_size_bytes": stat.st_size
            }
        except Exception as e:
            logger.warning(f"Failed to extract filesystem metadata: {e}")
            return {}

    @staticmethod
    def extract_permissions(file_path: Path) -> dict:
        """Extract file permissions metadata.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with permission metadata
        """
        try:
            stat = file_path.stat()

            return {
                "mode": oct(stat.st_mode),
                "is_readable": os.access(file_path, os.R_OK),
                "is_writable": os.access(file_path, os.W_OK),
            }
        except Exception:
            return {}
```

## Acceptance Criteria

- ✅ Language detection working with 95% accuracy
- ✅ Content classification per format type
- ✅ SHA-256 hashing for all documents
- ✅ Statistics (word/char/line count) computed
- ✅ Temporal metadata extracted from filesystem
- ✅ On-device processing (no external APIs)
- ✅ Deterministic outputs for idempotency
- ✅ Privacy-preserving (no content logging)
- ✅ Graceful degradation when fasttext unavailable
- ✅ Metrics tracking for telemetry

## Test Plan

### Unit Tests
- Language detection accuracy (test corpus)
- Content hash stability
- Word/character counting accuracy
- Classification per format
- Filesystem metadata extraction
- Error handling for missing models

### Integration Tests
- End-to-end enrichment pipeline
- Multi-language document handling
- Format-specific classification
- Metrics collection

### Performance Tests
- Language detection throughput
- Hash computation speed
- Memory usage for large documents

## Implementation Notes

### Installing fasttext-langdetect

```bash
pip install fasttext-langdetect
```

This provides 80x faster language detection than traditional langdetect while maintaining 95% accuracy.

### Alternative: Lingua (Most Accurate)

For highest accuracy (at cost of speed):
```bash
pip install lingua-language-detector
```

Lingua provides the most accurate language detection but is slower than fasttext.

### Privacy Considerations

- No document content is logged
- Only metadata (language code, hash) appears in audit logs
- All processing is on-device
- No network requests

### Determinism Guarantees

- Content hashing uses SHA-256 (cryptographically deterministic)
- Language detection on same content always returns same result
- Statistics computation is deterministic

## Open Questions

- Should we add sentiment analysis as optional metadata?
- How to handle multi-language documents?
- Should we extract entities during enrichment or separately?
- How to version metadata schema for backward compatibility?
- Should we cache language detection results?

## Dependencies

- NormalizedDocument schema (Task 01)
- fasttext-langdetect library
- SHA-256 from hashlib (standard library)



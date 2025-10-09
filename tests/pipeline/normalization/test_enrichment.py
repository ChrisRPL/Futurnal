"""Comprehensive unit tests for metadata enrichment pipeline.

Tests cover:
- FilesystemMetadataExtractor (temporal metadata, permissions)
- LanguageDetector (multiple languages, confidence thresholds, fallback)
- ContentClassifier (all formats, markdown/code variants)
- MetadataEnrichmentPipeline (full integration, metrics, error handling)

These tests validate all acceptance criteria from the specification.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.pipeline.models import (
    DocumentFormat,
    NormalizedDocument,
    NormalizedMetadata,
)
from futurnal.pipeline.normalization.enrichment import (
    ContentClassifier,
    FilesystemMetadataExtractor,
    LanguageDetector,
    MetadataEnrichmentPipeline,
)
from tests.pipeline.normalization.fixtures.language_samples import get_sample

# Valid SHA-256 hash for test placeholder
PLACEHOLDER_HASH = "4097889236a2af26c293033feb964c4cf118c0224e0d063fec0a89e9d0569ef2"


# ---------------------------------------------------------------------------
# FilesystemMetadataExtractor Tests
# ---------------------------------------------------------------------------


class TestFilesystemMetadataExtractor:
    """Tests for filesystem metadata extraction."""

    def test_extract_temporal_metadata_success(self, tmp_path):
        """Test successful temporal metadata extraction."""
        # Create a test file
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("Test content", encoding="utf-8")

        # Extract metadata
        created_at, modified_at, file_size = (
            FilesystemMetadataExtractor.extract_temporal_metadata(test_file)
        )

        # Verify results
        assert created_at is not None
        assert modified_at is not None
        assert file_size is not None
        assert file_size == len("Test content")
        assert isinstance(created_at, datetime)
        assert isinstance(modified_at, datetime)
        assert created_at.tzinfo is not None  # Should be timezone-aware
        assert modified_at.tzinfo is not None

    def test_extract_temporal_metadata_nonexistent_file(self, tmp_path):
        """Test temporal metadata extraction with nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.txt"

        created_at, modified_at, file_size = (
            FilesystemMetadataExtractor.extract_temporal_metadata(nonexistent)
        )

        # Should return None for all values
        assert created_at is None
        assert modified_at is None
        assert file_size is None

    def test_extract_permissions_success(self, tmp_path):
        """Test successful permissions extraction."""
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("Test content", encoding="utf-8")

        # Extract permissions
        permissions = FilesystemMetadataExtractor.extract_permissions(test_file)

        # Verify results
        assert "mode" in permissions
        assert "is_readable" in permissions
        assert "is_writable" in permissions
        assert "is_executable" in permissions
        assert isinstance(permissions["is_readable"], bool)
        assert isinstance(permissions["is_writable"], bool)
        assert isinstance(permissions["is_executable"], bool)
        assert permissions["is_readable"] is True  # File should be readable

    def test_extract_permissions_nonexistent_file(self, tmp_path):
        """Test permissions extraction with nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.txt"

        permissions = FilesystemMetadataExtractor.extract_permissions(nonexistent)

        # Should return empty dict
        assert permissions == {}

    @pytest.mark.skipif(os.name == "nt", reason="Unix permissions test")
    def test_extract_permissions_readonly_file(self, tmp_path):
        """Test permissions extraction with read-only file."""
        test_file = tmp_path / "readonly.txt"
        test_file.write_text("Test content", encoding="utf-8")

        # Make file read-only
        test_file.chmod(0o444)

        permissions = FilesystemMetadataExtractor.extract_permissions(test_file)

        assert permissions["is_readable"] is True
        assert permissions["is_writable"] is False


# ---------------------------------------------------------------------------
# LanguageDetector Tests
# ---------------------------------------------------------------------------


class TestLanguageDetector:
    """Tests for language detection."""

    @pytest.mark.asyncio
    async def test_detect_english(self):
        """Test English language detection."""
        detector = LanguageDetector()
        content = get_sample("en")

        language, confidence = await detector.detect(content)

        # Should detect English
        assert language == "en"
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_detect_spanish(self):
        """Test Spanish language detection."""
        detector = LanguageDetector()
        content = get_sample("es")

        language, confidence = await detector.detect(content)

        # Should detect Spanish
        assert language == "es"
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_detect_french(self):
        """Test French language detection."""
        detector = LanguageDetector()
        content = get_sample("fr")

        language, confidence = await detector.detect(content)

        # Should detect French
        assert language == "fr"
        assert confidence is not None

    @pytest.mark.asyncio
    async def test_detect_german(self):
        """Test German language detection."""
        detector = LanguageDetector()
        content = get_sample("de")

        language, confidence = await detector.detect(content)

        # Should detect German
        assert language == "de"
        assert confidence is not None

    @pytest.mark.asyncio
    async def test_detect_chinese(self):
        """Test Chinese language detection."""
        detector = LanguageDetector()
        content = get_sample("zh")

        language, confidence = await detector.detect(content)

        # Should detect Chinese
        assert language == "zh"
        assert confidence is not None

    @pytest.mark.asyncio
    async def test_detect_short_text(self):
        """Test detection with very short text (should return None)."""
        detector = LanguageDetector()
        content = get_sample("short")

        language, confidence = await detector.detect(content)

        # Too short for detection
        assert language is None
        assert confidence is None

    @pytest.mark.asyncio
    async def test_detect_empty_text(self):
        """Test detection with empty text."""
        detector = LanguageDetector()
        content = get_sample("empty")

        language, confidence = await detector.detect(content)

        # Empty content
        assert language is None
        assert confidence is None

    @pytest.mark.asyncio
    async def test_detect_with_confidence_threshold(self):
        """Test detection with custom confidence threshold."""
        detector = LanguageDetector()
        content = get_sample("en")

        # High confidence threshold
        language, confidence = await detector.detect(content, min_confidence=0.9)

        # With ftlangdetect, should still detect English with high confidence
        # If fallback is used, might return None due to lower confidence
        if language:
            assert confidence >= 0.9

    @pytest.mark.asyncio
    async def test_detect_with_low_confidence_threshold(self):
        """Test detection with low confidence threshold."""
        detector = LanguageDetector()
        content = get_sample("en")

        language, confidence = await detector.detect(content, min_confidence=0.3)

        # Should detect with low threshold
        assert language is not None
        assert confidence is not None
        assert confidence >= 0.3

    @pytest.mark.asyncio
    async def test_fallback_detection_when_ftlangdetect_unavailable(self, monkeypatch):
        """Test fallback detection when ftlangdetect is not available."""
        # Mock ftlangdetect import to fail
        import sys

        if "ftlangdetect" in sys.modules:
            del sys.modules["ftlangdetect"]

        # Create detector (should use fallback)
        with patch("futurnal.pipeline.normalization.enrichment.logger"):
            detector = LanguageDetector()
            detector._detector = None  # Force fallback

        content = get_sample("en")

        language, confidence = await detector.detect(content)

        # Fallback should detect English based on common words
        assert language == "en"
        assert confidence == 0.7  # Fallback confidence

    @pytest.mark.asyncio
    async def test_fallback_detection_non_english(self):
        """Test fallback detection with non-English text."""
        detector = LanguageDetector()
        detector._detector = None  # Force fallback

        content = get_sample("zh")  # Chinese content

        language, confidence = await detector.detect(content)

        # Fallback should return None for non-English
        assert language is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# ContentClassifier Tests
# ---------------------------------------------------------------------------


class TestContentClassifier:
    """Tests for content type classification."""

    @pytest.mark.asyncio
    async def test_classify_markdown_plain(self):
        """Test classification of plain markdown."""
        classifier = ContentClassifier()
        content = "# Heading\n\nSome plain markdown content."

        mime_type = await classifier.classify(content, DocumentFormat.MARKDOWN)

        assert mime_type == "text/markdown"

    @pytest.mark.asyncio
    async def test_classify_markdown_with_mermaid(self):
        """Test classification of markdown with mermaid diagrams."""
        classifier = ContentClassifier()
        content = """
# Document

```mermaid
graph TD
    A --> B
```
"""

        mime_type = await classifier.classify(content, DocumentFormat.MARKDOWN)

        assert mime_type == "text/markdown+mermaid"

    @pytest.mark.asyncio
    async def test_classify_markdown_with_math(self):
        """Test classification of markdown with math notation."""
        classifier = ContentClassifier()
        content = """
# Math Document

$$
E = mc^2
$$

Some text with inline math \\(a^2 + b^2 = c^2\\).
"""

        mime_type = await classifier.classify(content, DocumentFormat.MARKDOWN)

        assert mime_type == "text/markdown+math"

    @pytest.mark.asyncio
    async def test_classify_code_python(self):
        """Test classification of Python code."""
        classifier = ContentClassifier()
        content = """
import os
from pathlib import Path

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
"""

        mime_type = await classifier.classify(content, DocumentFormat.CODE)

        assert mime_type == "text/x-python"

    @pytest.mark.asyncio
    async def test_classify_code_javascript(self):
        """Test classification of JavaScript code."""
        classifier = ContentClassifier()
        content = """
const express = require('express');
const app = express();

function handleRequest(req, res) {
    res.send('Hello, World!');
}

app.get('/', handleRequest);
"""

        mime_type = await classifier.classify(content, DocumentFormat.CODE)

        assert mime_type == "text/javascript"

    @pytest.mark.asyncio
    async def test_classify_code_typescript(self):
        """Test classification of TypeScript code."""
        classifier = ContentClassifier()
        content = """
interface User {
    name: string;
    age: number;
}

const user: User = {
    name: "John",
    age: 30
};

function greet(user: User): string {
    return `Hello, ${user.name}`;
}
"""

        mime_type = await classifier.classify(content, DocumentFormat.CODE)

        assert mime_type == "text/typescript"

    @pytest.mark.asyncio
    async def test_classify_code_java(self):
        """Test classification of Java code."""
        classifier = ContentClassifier()
        content = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

        mime_type = await classifier.classify(content, DocumentFormat.CODE)

        assert mime_type == "text/x-java"

    @pytest.mark.asyncio
    async def test_classify_code_go(self):
        """Test classification of Go code."""
        classifier = ContentClassifier()
        content = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
"""

        mime_type = await classifier.classify(content, DocumentFormat.CODE)

        assert mime_type == "text/x-go"

    @pytest.mark.asyncio
    async def test_classify_code_rust(self):
        """Test classification of Rust code."""
        classifier = ContentClassifier()
        content = """
fn main() {
    println!("Hello, World!");
}

pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"""

        mime_type = await classifier.classify(content, DocumentFormat.CODE)

        assert mime_type == "text/x-rust"

    @pytest.mark.asyncio
    async def test_classify_pdf(self):
        """Test classification of PDF format."""
        classifier = ContentClassifier()

        mime_type = await classifier.classify("", DocumentFormat.PDF)

        assert mime_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_classify_email(self):
        """Test classification of email format."""
        classifier = ContentClassifier()

        mime_type = await classifier.classify("", DocumentFormat.EMAIL)

        assert mime_type == "message/rfc822"

    @pytest.mark.asyncio
    async def test_classify_html(self):
        """Test classification of HTML format."""
        classifier = ContentClassifier()

        mime_type = await classifier.classify("", DocumentFormat.HTML)

        assert mime_type == "text/html"

    @pytest.mark.asyncio
    async def test_classify_json(self):
        """Test classification of JSON format."""
        classifier = ContentClassifier()

        mime_type = await classifier.classify("", DocumentFormat.JSON)

        assert mime_type == "application/json"

    @pytest.mark.asyncio
    async def test_classify_unknown_format(self):
        """Test classification of unknown format."""
        classifier = ContentClassifier()

        mime_type = await classifier.classify("", DocumentFormat.UNKNOWN)

        assert mime_type == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_classifications_counter(self):
        """Test that classification counter is incremented."""
        classifier = ContentClassifier()

        assert classifier.classifications_performed == 0

        await classifier.classify("test", DocumentFormat.TEXT)
        assert classifier.classifications_performed == 1

        await classifier.classify("test", DocumentFormat.MARKDOWN)
        assert classifier.classifications_performed == 2


# ---------------------------------------------------------------------------
# MetadataEnrichmentPipeline Tests
# ---------------------------------------------------------------------------


class TestMetadataEnrichmentPipeline:
    """Tests for metadata enrichment pipeline."""

    @pytest.mark.asyncio
    async def test_enrich_with_all_features_enabled(self, tmp_path):
        """Test enrichment with all features enabled."""
        pipeline = MetadataEnrichmentPipeline()

        # Create test file
        test_file = tmp_path / "test.txt"
        content = get_sample("en")
        test_file.write_text(content, encoding="utf-8")

        # Create test document
        metadata = NormalizedMetadata(
            source_path=str(test_file),
            source_id="test-001",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        # Enrich
        enriched = await pipeline.enrich(
            document=document,
            enable_language_detection=True,
            enable_classification=True,
            compute_hash=True,
            file_path=test_file,
        )

        # Verify enrichment
        assert enriched.language == "en"
        assert enriched.language_confidence is not None
        assert enriched.content_type == "text/plain"
        assert enriched.content_hash is not None
        assert len(enriched.content_hash) == 64  # SHA-256
        assert enriched.character_count == len(content)
        assert enriched.word_count > 0
        assert enriched.line_count > 0
        assert enriched.created_at is not None
        assert enriched.modified_at is not None
        assert enriched.file_size_bytes == len(content.encode("utf-8"))
        assert "file_permissions" in enriched.extra

    @pytest.mark.asyncio
    async def test_enrich_language_detection_only(self):
        """Test enrichment with only language detection enabled."""
        pipeline = MetadataEnrichmentPipeline()

        content = get_sample("es")
        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-002",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        enriched = await pipeline.enrich(
            document=document,
            enable_language_detection=True,
            enable_classification=False,
            compute_hash=False,
        )

        # Should have language but not hash
        assert enriched.language == "es"
        assert enriched.content_hash == PLACEHOLDER_HASH  # Not changed

    @pytest.mark.asyncio
    async def test_enrich_hashing_only(self):
        """Test enrichment with only hashing enabled."""
        pipeline = MetadataEnrichmentPipeline()

        content = "Test content for hashing"
        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-003",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        enriched = await pipeline.enrich(
            document=document,
            enable_language_detection=False,
            enable_classification=False,
            compute_hash=True,
        )

        # Should have hash but not language
        assert enriched.language is None
        assert enriched.content_hash != PLACEHOLDER_HASH
        assert len(enriched.content_hash) == 64

    @pytest.mark.asyncio
    async def test_word_count_accuracy(self):
        """Test word counting accuracy."""
        pipeline = MetadataEnrichmentPipeline()

        content = "One two three four five. Six, seven; eight: nine! Ten?"
        expected_words = 10

        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-004",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        enriched = await pipeline.enrich(document=document)

        assert enriched.word_count == expected_words

    @pytest.mark.asyncio
    async def test_character_count_accuracy(self):
        """Test character counting accuracy."""
        pipeline = MetadataEnrichmentPipeline()

        content = "Hello, World!"
        expected_chars = len(content)

        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-005",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        enriched = await pipeline.enrich(document=document)

        assert enriched.character_count == expected_chars

    @pytest.mark.asyncio
    async def test_line_count_accuracy(self):
        """Test line counting accuracy."""
        pipeline = MetadataEnrichmentPipeline()

        content = "Line 1\nLine 2\nLine 3\nLine 4"
        expected_lines = 4

        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-006",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        enriched = await pipeline.enrich(document=document)

        assert enriched.line_count == expected_lines

    @pytest.mark.asyncio
    async def test_hash_determinism(self):
        """Test that content hashing is deterministic."""
        pipeline = MetadataEnrichmentPipeline()

        content = "Deterministic content for hashing"

        # First enrichment
        metadata1 = NormalizedMetadata(
            source_path="test1.txt",
            source_id="test-007a",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document1 = NormalizedDocument(
            document_id="test-doc-1",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata1,
        )

        enriched1 = await pipeline.enrich(document=document1, compute_hash=True)

        # Second enrichment with same content
        metadata2 = NormalizedMetadata(
            source_path="test2.txt",
            source_id="test-007b",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document2 = NormalizedDocument(
            document_id="test-doc-2",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata2,
        )

        enriched2 = await pipeline.enrich(document=document2, compute_hash=True)

        # Hashes should be identical
        assert enriched1.content_hash == enriched2.content_hash

    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self):
        """Test graceful degradation when enrichment fails."""
        pipeline = MetadataEnrichmentPipeline()

        # Create a document with problematic content that might cause errors
        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-008",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        # Document with no content
        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content="",
            metadata=metadata,
        )

        # Should not raise, should handle gracefully
        enriched = await pipeline.enrich(document=document)

        # Should return metadata even with empty content
        assert enriched is not None

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that enrichment metrics are tracked correctly."""
        pipeline = MetadataEnrichmentPipeline()

        content = "Test content"
        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-009",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        # Initial metrics
        assert pipeline.documents_enriched == 0
        assert pipeline.enrichment_errors == 0

        # Enrich document
        await pipeline.enrich(document=document)

        # Metrics should be updated
        assert pipeline.documents_enriched == 1
        assert pipeline.enrichment_errors == 0

        # Get metrics
        metrics = pipeline.get_metrics()
        assert metrics["documents_enriched"] == 1
        assert metrics["enrichment_errors"] == 0
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_enrich_without_file_path(self):
        """Test enrichment without file_path (no temporal metadata)."""
        pipeline = MetadataEnrichmentPipeline()

        content = "Test content"
        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-010",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=content,
            metadata=metadata,
        )

        # Enrich without file_path
        enriched = await pipeline.enrich(document=document)

        # Should not have temporal metadata
        assert enriched.created_at is None
        assert enriched.modified_at is None
        assert enriched.file_size_bytes is None

    @pytest.mark.asyncio
    async def test_enrich_chunked_document(self):
        """Test enrichment with chunked document."""
        from futurnal.pipeline.models import DocumentChunk

        pipeline = MetadataEnrichmentPipeline()

        # Create chunks
        # Generate valid hashes for chunks
        chunk1_hash = hashlib.sha256(b"First chunk content.").hexdigest()
        chunk2_hash = hashlib.sha256(b"Second chunk content.").hexdigest()

        chunk1 = DocumentChunk(
            chunk_id="chunk-1",
            parent_document_id="doc-1",
            chunk_index=0,
            content="First chunk content.",
            content_hash=chunk1_hash,
            character_count=20,
            word_count=3,
        )

        chunk2 = DocumentChunk(
            chunk_id="chunk-2",
            parent_document_id="doc-1",
            chunk_index=1,
            content="Second chunk content.",
            content_hash=chunk2_hash,
            character_count=21,
            word_count=3,
        )

        metadata = NormalizedMetadata(
            source_path="test.txt",
            source_id="test-011",
            source_type="test",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            character_count=0,
            word_count=0,
            line_count=0,
            content_hash=PLACEHOLDER_HASH,
        )

        document = NormalizedDocument(
            document_id="test-doc",
            sha256=PLACEHOLDER_HASH,
            content=None,  # No full content
            chunks=[chunk1, chunk2],
            metadata=metadata,
        )

        # Enrich
        enriched = await pipeline.enrich(document=document)

        # Should concatenate chunk content for analysis
        expected_content = "First chunk content.\n\nSecond chunk content."
        assert enriched.character_count == len(expected_content)

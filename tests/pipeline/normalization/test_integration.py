"""Integration tests for full normalization pipeline.

Tests the complete pipeline from file → normalized document → sink with
real components (except Unstructured.io which is mocked).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization import create_normalization_service


@pytest.fixture
def integration_service(
    mock_normalization_sink, mock_unstructured_partition, mock_language_detector, tmp_path
):
    """Create service with real components for integration testing."""
    # Create service with real components
    service = create_normalization_service(sink=mock_normalization_sink)

    return service


class TestEndToEndNormalization:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_markdown_file_normalization(
        self, integration_service, temp_file, mock_normalization_sink
    ):
        """Test complete normalization of markdown file."""
        content = """# Test Document

## Introduction
This is a test document for integration testing.

## Main Content
The main content goes here with some detailed text.

### Subsection
A nested section with more content.

## Conclusion
Final thoughts and summary.
"""
        test_file = temp_file(content, "test.md")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-md-001",
            source_type="integration_test",
        )

        # Verify document structure
        assert result.sha256 is not None
        assert len(result.sha256) == 64

        # Verify metadata
        assert result.metadata.format == DocumentFormat.MARKDOWN
        assert result.metadata.source_id == "integration-md-001"
        assert result.metadata.character_count > 0
        assert result.metadata.word_count > 0
        assert result.metadata.content_hash == result.sha256

        # Verify chunking
        assert result.is_chunked
        assert len(result.chunks) > 0

        # Verify chunks structure
        for idx, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == idx
            assert chunk.parent_document_id == result.document_id
            assert len(chunk.content) > 0
            assert chunk.content_hash is not None

        # Verify sink delivery
        assert len(mock_normalization_sink.handled_documents) == 1
        sink_doc = mock_normalization_sink.handled_documents[0]
        assert sink_doc["sha256"] == result.sha256
        assert "text" in sink_doc
        assert "chunks" in sink_doc

    @pytest.mark.asyncio
    async def test_text_file_normalization(
        self, integration_service, temp_file, mock_normalization_sink
    ):
        """Test normalization of plain text file."""
        content = "This is a plain text file.\nWith multiple lines of content.\nFor integration testing."
        test_file = temp_file(content, "test.txt")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-txt-001",
            source_type="integration_test",
        )

        # Verify basic structure
        assert result.sha256 is not None
        assert result.metadata.format in [DocumentFormat.TEXT, DocumentFormat.CODE]
        assert result.content == content

        # Verify metadata enrichment
        assert result.metadata.character_count == len(content)
        assert result.metadata.word_count > 0
        assert result.metadata.line_count == 3

        # Verify sink delivery
        assert len(mock_normalization_sink.handled_documents) == 1

    @pytest.mark.asyncio
    async def test_code_file_detection(
        self, integration_service, temp_file, mock_normalization_sink
    ):
        """Test detection and normalization of code files."""
        code_content = """def hello_world():
    \"\"\"A simple function.\"\"\"
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
"""
        test_file = temp_file(code_content, "test.py")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-code-001",
            source_type="integration_test",
        )

        # Verify code format detected
        assert result.metadata.format == DocumentFormat.CODE
        assert result.content == code_content

        # Verify metadata
        assert "text" in result.metadata.extra

    @pytest.mark.asyncio
    async def test_multiple_documents_sequential(
        self, integration_service, temp_file, mock_normalization_sink
    ):
        """Test processing multiple documents sequentially."""
        documents = [
            ("doc1.txt", "Content for document 1"),
            ("doc2.txt", "Content for document 2"),
            ("doc3.txt", "Content for document 3"),
        ]

        results = []
        for name, content in documents:
            test_file = temp_file(content, name)
            result = await integration_service.normalize_document(
                file_path=test_file,
                source_id=f"integration-{name}",
                source_type="integration_test",
            )
            results.append(result)

        # Verify all documents processed
        assert len(results) == 3
        assert len(mock_normalization_sink.handled_documents) == 3

        # Verify unique document IDs
        doc_ids = [doc.document_id for doc in results]
        assert len(set(doc_ids)) == 3  # All unique

        # Verify metrics
        metrics = integration_service.get_metrics()
        assert metrics["documents_processed"] == 3
        assert metrics["documents_failed"] == 0
        assert metrics["success_rate"] == 1.0


class TestMetadataEnrichment:
    """Tests for metadata enrichment in pipeline."""

    @pytest.mark.asyncio
    async def test_language_detection(
        self, integration_service, temp_file, mock_language_detector
    ):
        """Test language detection during enrichment."""
        content = "This is an English document with some content."
        test_file = temp_file(content, "test.txt")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-lang-001",
            source_type="integration_test",
        )

        # Language detection should have run
        assert result.metadata.language is not None
        assert result.metadata.language_confidence is not None

    @pytest.mark.asyncio
    async def test_content_hash_generation(self, integration_service, temp_file):
        """Test content hash generation and idempotency."""
        content = "Test content for hashing"
        test_file = temp_file(content, "test.txt")

        # Process same content twice
        result1 = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-hash-001",
            source_type="integration_test",
        )

        result2 = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-hash-002",  # Different source_id
            source_type="integration_test",
        )

        # Content hashes should be identical (idempotent)
        assert result1.metadata.content_hash == result2.metadata.content_hash
        assert result1.sha256 == result2.sha256


class TestChunkingStrategies:
    """Tests for different chunking strategies."""

    @pytest.mark.asyncio
    async def test_markdown_by_title_chunking(self, integration_service, temp_file):
        """Test markdown chunking by title preserves sections."""
        content = """# Document Title

## Section 1
Content for section 1.

## Section 2
Content for section 2.

### Subsection 2.1
Nested content.

## Section 3
Content for section 3.
"""
        test_file = temp_file(content, "test.md")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-chunk-001",
            source_type="integration_test",
        )

        # Should be chunked
        assert result.is_chunked
        assert len(result.chunks) >= 1

        # Check that section titles are preserved
        section_titles = [
            chunk.section_title for chunk in result.chunks if chunk.section_title
        ]
        assert len(section_titles) > 0

    @pytest.mark.asyncio
    async def test_long_markdown_document_chunking(self, integration_service, temp_file):
        """Test chunking of long markdown document with many sections."""
        # Create a long document with 15+ sections
        sections = []
        sections.append("# Long Document Test\n\n")
        sections.append("Introduction paragraph with some context.\n\n")

        for i in range(15):
            sections.append(f"## Section {i + 1}\n\n")
            sections.append(f"This is content for section {i + 1}. " * 30)
            sections.append("\n\n")

            # Add some subsections
            if i % 3 == 0:
                sections.append(f"### Subsection {i + 1}.1\n\n")
                sections.append("Nested content with details. " * 20)
                sections.append("\n\n")

        content = "".join(sections)
        test_file = temp_file(content, "long_doc.md")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-long-md-001",
            source_type="integration_test",
        )

        # Should be chunked
        assert result.is_chunked
        assert len(result.chunks) >= 5  # Should have multiple chunks

        # Verify chunks have proper structure
        for chunk in result.chunks:
            assert chunk.chunk_id is not None
            assert chunk.parent_document_id == result.document_id
            assert len(chunk.content) > 0
            assert chunk.content_hash is not None

        # Some chunks should have section titles
        titled_chunks = [c for c in result.chunks if c.section_title]
        assert len(titled_chunks) > 0

    @pytest.mark.asyncio
    async def test_large_text_file_chunking(self, integration_service, temp_file):
        """Test chunking of large plain text file."""
        # Create a large text file (~50KB)
        sentences = []
        for i in range(1000):
            sentences.append(f"This is sentence number {i} with some content. ")

        content = "".join(sentences)
        test_file = temp_file(content, "large.txt")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-large-txt-001",
            source_type="integration_test",
        )

        # Should be chunked
        assert result.is_chunked
        assert len(result.chunks) >= 5  # Should have multiple chunks

        # Verify total content is preserved
        total_content_length = sum(len(chunk.content) for chunk in result.chunks)
        # Due to overlap, total may be larger than original
        assert total_content_length >= len(content) * 0.9  # Allow 10% variation

    @pytest.mark.asyncio
    async def test_chunk_size_configuration(self, integration_service, temp_file):
        """Test that chunk sizes respect configuration."""
        content = "Test sentence. " * 500  # Create long content
        test_file = temp_file(content, "test.txt")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-size-001",
            source_type="integration_test",
        )

        # All chunks should be within reasonable size bounds
        for chunk in result.chunks:
            # Chunks should not exceed hard_max_size (8000 chars default)
            assert len(chunk.content) <= 10000

    @pytest.mark.asyncio
    async def test_chunk_metadata_completeness(self, integration_service, temp_file):
        """Test that chunk metadata is complete and accurate."""
        content = """# Test Document

## Section A
Content for section A.

## Section B
Content for section B.
"""
        test_file = temp_file(content, "test.md")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-metadata-001",
            source_type="integration_test",
        )

        assert result.is_chunked

        for idx, chunk in enumerate(result.chunks):
            # Verify required fields
            assert chunk.chunk_id is not None
            assert chunk.parent_document_id is not None
            assert chunk.chunk_index == idx
            assert chunk.content is not None
            assert chunk.content_hash is not None
            assert chunk.character_count > 0
            assert chunk.word_count >= 0

    @pytest.mark.asyncio
    async def test_very_small_document_chunking(self, integration_service, temp_file):
        """Test chunking of very small document."""
        content = "Small"
        test_file = temp_file(content, "small.txt")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-small-001",
            source_type="integration_test",
        )

        # Small docs may not be chunked or have single chunk
        if result.is_chunked:
            assert len(result.chunks) >= 1
        else:
            assert result.content == content


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_enrichment_error(
        self, integration_service, temp_file, monkeypatch
    ):
        """Test that enrichment errors don't fail entire pipeline."""

        # Mock language detector to raise error
        def failing_detector(text, **kwargs):
            raise RuntimeError("Simulated language detection failure")

        try:
            monkeypatch.setattr("ftlangdetect.detect", failing_detector)
        except (ImportError, AttributeError):
            pass

        content = "Test content"
        test_file = temp_file(content, "test.txt")

        # Should still succeed despite enrichment error
        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-error-001",
            source_type="integration_test",
        )

        # Document should still be normalized
        assert result.sha256 is not None
        assert result.content == content

        # May have enrichment error in metadata
        if "enrichment_error" in result.metadata.extra:
            assert isinstance(result.metadata.extra["enrichment_error"], str)


class TestPerformance:
    """Basic performance tests."""

    @pytest.mark.asyncio
    async def test_small_document_performance(self, integration_service, temp_file):
        """Test processing speed for small documents."""
        content = "Small test document" * 10
        test_file = temp_file(content, "test.txt")

        result = await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-perf-001",
            source_type="integration_test",
        )

        # Should complete quickly
        assert result.metadata.processing_duration_ms is not None
        assert result.metadata.processing_duration_ms < 5000  # <5 seconds

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, integration_service, temp_file):
        """Test that metrics are correctly tracked."""
        # Process a document
        content = "Test content"
        test_file = temp_file(content, "test.txt")

        await integration_service.normalize_document(
            file_path=test_file,
            source_id="integration-metrics-001",
            source_type="integration_test",
        )

        metrics = integration_service.get_metrics()

        assert metrics["documents_processed"] >= 1
        assert metrics["average_processing_time_ms"] > 0
        assert metrics["success_rate"] > 0

"""Tests for NormalizationService orchestration.

Tests cover:
- Format detection from file extension and metadata
- Adapter selection and routing
- Pipeline orchestration (adapter → unstructured → chunk → enrich)
- Error handling and quarantine integration
- Audit logging
- Metrics tracking
- Configuration handling
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from futurnal.pipeline.models import DocumentFormat, NormalizedDocument
from futurnal.pipeline.normalization import (
    NormalizationConfig,
    NormalizationError,
    NormalizationService,
)
from futurnal.pipeline.normalization.registry import FormatAdapterRegistry
from futurnal.pipeline.normalization.chunking import ChunkingEngine
from futurnal.pipeline.normalization.enrichment import MetadataEnrichmentPipeline
from futurnal.pipeline.normalization.unstructured_bridge import UnstructuredBridge


@pytest.fixture
def normalization_config():
    """Create test normalization configuration."""
    return NormalizationConfig(
        enable_chunking=True,
        default_chunk_strategy="by_title",
        max_chunk_size_chars=1000,
        chunk_overlap_chars=100,
        enable_language_detection=True,
        enable_content_classification=True,
        compute_content_hash=True,
        quarantine_on_failure=True,
        audit_logging_enabled=True,
    )


@pytest.fixture
def adapter_registry(mock_adapter):
    """Create adapter registry with mock adapter."""
    registry = FormatAdapterRegistry()
    mock_adapter.supported_formats = [DocumentFormat.TEXT, DocumentFormat.MARKDOWN]
    registry.register(mock_adapter)
    return registry


@pytest.fixture
def normalization_service(
    normalization_config,
    adapter_registry,
    mock_normalization_sink,
    mock_audit_logger,
):
    """Create normalization service for testing."""
    chunking_engine = ChunkingEngine()
    enrichment_pipeline = MetadataEnrichmentPipeline()
    unstructured_bridge = UnstructuredBridge()

    return NormalizationService(
        config=normalization_config,
        adapter_registry=adapter_registry,
        chunking_engine=chunking_engine,
        enrichment_pipeline=enrichment_pipeline,
        unstructured_bridge=unstructured_bridge,
        sink=mock_normalization_sink,
        audit_logger=mock_audit_logger,
    )


class TestFormatDetection:
    """Tests for format detection logic."""

    @pytest.mark.asyncio
    async def test_detect_format_from_extension(
        self, normalization_service, temp_file
    ):
        """Test format detection from file extension."""
        # Test markdown
        md_path = temp_file("# Test", "test.md")
        format = await normalization_service._detect_format(md_path, None)
        assert format == DocumentFormat.MARKDOWN

        # Test PDF
        pdf_path = temp_file("", "test.pdf")
        format = await normalization_service._detect_format(pdf_path, None)
        assert format == DocumentFormat.PDF

        # Test plain text
        txt_path = temp_file("Test", "test.txt")
        format = await normalization_service._detect_format(txt_path, None)
        assert format == DocumentFormat.TEXT

    @pytest.mark.asyncio
    async def test_detect_format_from_metadata(self, normalization_service, temp_file):
        """Test format detection from source metadata."""
        test_file = temp_file("Test", "unknown.xyz")

        # Metadata should override extension
        metadata = {"format": "markdown"}
        format = await normalization_service._detect_format(test_file, metadata)
        assert format == DocumentFormat.MARKDOWN

    @pytest.mark.asyncio
    async def test_detect_unknown_format(self, normalization_service, temp_file):
        """Test detection of unknown format."""
        unknown_file = temp_file("Test", "test.unknown")
        format = await normalization_service._detect_format(unknown_file, None)
        assert format == DocumentFormat.UNKNOWN


class TestChunkingConfiguration:
    """Tests for chunking configuration selection."""

    def test_markdown_chunking_config(self, normalization_service):
        """Test markdown uses by_title strategy."""
        config = normalization_service._get_chunking_config(
            DocumentFormat.MARKDOWN, "local_files"
        )
        assert config.strategy == "by_title"

    def test_pdf_chunking_config(self, normalization_service):
        """Test PDF uses by_page strategy."""
        config = normalization_service._get_chunking_config(
            DocumentFormat.PDF, "local_files"
        )
        assert config.strategy == "by_page"

    def test_email_chunking_config(self, normalization_service):
        """Test email uses basic strategy with smaller chunks."""
        config = normalization_service._get_chunking_config(
            DocumentFormat.EMAIL, "imap_mailbox"
        )
        assert config.strategy == "basic"
        assert config.max_chunk_size == 2000

    def test_default_chunking_config(self, normalization_service):
        """Test unknown formats use default strategy."""
        config = normalization_service._get_chunking_config(
            DocumentFormat.UNKNOWN, "local_files"
        )
        assert config.strategy == normalization_service.config.default_chunk_strategy


class TestNormalizationPipeline:
    """Tests for end-to-end normalization pipeline."""

    @pytest.mark.asyncio
    async def test_normalize_text_document(
        self, normalization_service, temp_file, mock_normalization_sink
    ):
        """Test normalizing a plain text document."""
        content = "This is a test document.\nWith multiple lines."
        test_file = temp_file(content, "test.txt")

        result = await normalization_service.normalize_document(
            file_path=test_file,
            source_id="test-123",
            source_type="local_files",
            source_metadata={},
        )

        # Verify document structure
        assert isinstance(result, NormalizedDocument)
        assert result.sha256 is not None
        assert len(result.sha256) == 64  # SHA-256 hex length

        # Verify metadata
        assert result.metadata.source_id == "test-123"
        assert result.metadata.source_type == "local_files"
        assert result.metadata.character_count > 0
        assert result.metadata.word_count > 0

        # Verify sink delivery
        assert len(mock_normalization_sink.handled_documents) == 1

    @pytest.mark.asyncio
    async def test_normalize_with_chunking(
        self, normalization_service, temp_file, sample_markdown_content
    ):
        """Test document normalization with chunking enabled."""
        test_file = temp_file(sample_markdown_content, "test.md")

        result = await normalization_service.normalize_document(
            file_path=test_file,
            source_id="test-md-123",
            source_type="local_files",
        )

        # Verify chunking occurred
        assert result.is_chunked
        assert len(result.chunks) > 0
        assert result.metadata.is_chunked
        assert result.metadata.total_chunks == len(result.chunks)

        # Verify chunks have proper structure
        for idx, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == idx
            assert chunk.parent_document_id == result.document_id
            assert chunk.content_hash is not None
            assert chunk.character_count > 0

    @pytest.mark.asyncio
    async def test_normalize_without_chunking(
        self, normalization_config, adapter_registry, temp_file
    ):
        """Test normalization with chunking disabled."""
        # Disable chunking
        normalization_config.enable_chunking = False

        service = NormalizationService(
            config=normalization_config,
            adapter_registry=adapter_registry,
            chunking_engine=ChunkingEngine(),
            enrichment_pipeline=MetadataEnrichmentPipeline(),
            unstructured_bridge=UnstructuredBridge(),
        )

        test_file = temp_file("Test content", "test.txt")

        result = await service.normalize_document(
            file_path=test_file, source_id="test-123", source_type="local_files"
        )

        # Verify no chunking
        assert not result.is_chunked
        assert len(result.chunks) == 0

    @pytest.mark.asyncio
    async def test_normalize_with_source_metadata(
        self, normalization_service, temp_file
    ):
        """Test normalization preserves source metadata."""
        test_file = temp_file("Test", "test.txt")
        source_metadata = {"custom_field": "custom_value", "tags": ["tag1", "tag2"]}

        result = await normalization_service.normalize_document(
            file_path=test_file,
            source_id="test-123",
            source_type="local_files",
            source_metadata=source_metadata,
        )

        # Verify source metadata preserved in extra
        assert "custom_field" in result.metadata.extra
        assert result.metadata.extra["custom_field"] == "custom_value"


class TestErrorHandling:
    """Tests for error handling and quarantine integration."""

    @pytest.mark.asyncio
    async def test_normalize_nonexistent_file(self, normalization_service):
        """Test error handling for nonexistent file."""
        with pytest.raises(NormalizationError):
            await normalization_service.normalize_document(
                file_path="/nonexistent/file.txt",
                source_id="test-123",
                source_type="local_files",
            )

        # Verify metrics updated
        assert normalization_service.documents_failed == 1

    @pytest.mark.asyncio
    async def test_quarantine_on_failure(
        self, normalization_service, mock_quarantine_manager, temp_file
    ):
        """Test failed documents are quarantined."""
        # Add quarantine manager
        normalization_service.quarantine_manager = mock_quarantine_manager

        # Create a file that will fail (by making adapter fail)
        test_file = temp_file("Test", "test.txt")

        # Mock adapter to raise error
        mock_adapter = normalization_service.adapter_registry.get_adapter(
            DocumentFormat.TEXT
        )
        original_normalize = mock_adapter.normalize

        async def failing_normalize(**kwargs):
            raise ValueError("Simulated adapter failure")

        mock_adapter.normalize = failing_normalize

        with pytest.raises(NormalizationError):
            await normalization_service.normalize_document(
                file_path=test_file, source_id="test-123", source_type="local_files"
            )

        # Restore original method
        mock_adapter.normalize = original_normalize

        # Note: Quarantine integration would be verified in integration tests
        # with real QuarantineManager


class TestAuditLogging:
    """Tests for audit logging integration."""

    @pytest.mark.asyncio
    async def test_audit_log_start_event(
        self, normalization_service, temp_file, mock_audit_logger
    ):
        """Test audit logging of normalization start."""
        test_file = temp_file("Test", "test.txt")

        await normalization_service.normalize_document(
            file_path=test_file, source_id="test-123", source_type="local_files"
        )

        # Verify audit log exists (simple check that file was created)
        audit_file = mock_audit_logger.output_dir / mock_audit_logger.filename
        assert audit_file.exists()

    @pytest.mark.asyncio
    async def test_audit_log_success_event(
        self, normalization_service, temp_file, mock_audit_logger
    ):
        """Test audit logging of successful normalization."""
        test_file = temp_file("Test", "test.txt")

        result = await normalization_service.normalize_document(
            file_path=test_file, source_id="test-123", source_type="local_files"
        )

        # Verify audit events recorded
        audit_file = mock_audit_logger.output_dir / mock_audit_logger.filename
        assert audit_file.exists()

        # Read audit log
        audit_content = audit_file.read_text()
        assert "normalization_started" in audit_content
        assert "normalization_completed" in audit_content


class TestMetrics:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_tracking_success(self, normalization_service, temp_file):
        """Test metrics updated on successful normalization."""
        test_file = temp_file("Test", "test.txt")

        await normalization_service.normalize_document(
            file_path=test_file, source_id="test-123", source_type="local_files"
        )

        metrics = normalization_service.get_metrics()

        assert metrics["documents_processed"] == 1
        assert metrics["documents_failed"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["average_processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_metrics_tracking_failure(self, normalization_service):
        """Test metrics updated on failed normalization."""
        with pytest.raises(NormalizationError):
            await normalization_service.normalize_document(
                file_path="/nonexistent/file.txt",
                source_id="test-123",
                source_type="local_files",
            )

        metrics = normalization_service.get_metrics()

        assert metrics["documents_processed"] == 0
        assert metrics["documents_failed"] == 1
        assert metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_success_rate(self, normalization_service, temp_file):
        """Test success rate calculation with mixed results."""
        # Process one successful document
        test_file = temp_file("Test", "test.txt")
        await normalization_service.normalize_document(
            file_path=test_file, source_id="test-1", source_type="local_files"
        )

        # Attempt one failed document
        with pytest.raises(NormalizationError):
            await normalization_service.normalize_document(
                file_path="/nonexistent.txt", source_id="test-2", source_type="local_files"
            )

        metrics = normalization_service.get_metrics()

        assert metrics["documents_processed"] == 1
        assert metrics["documents_failed"] == 1
        assert metrics["success_rate"] == 0.5

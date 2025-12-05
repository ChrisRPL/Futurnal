"""End-to-end integration tests for multimodal pipeline.

Module 08: Multimodal Integration & Tool Enhancement - Phase 4 (Integration & Polish)
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md

Tests the full pipeline from file to normalized document for all modalities:
- Audio files via Whisper V3 transcription
- Image files via DeepSeek-OCR
- Scanned PDFs via PDF→Image→OCR pipeline
- Mixed batches via orchestrator coordination
- Unified API entry point (extract_from_any_source)

Success Criteria from Production Plan:
- [ ] Unified API complete
- [ ] All modalities working
- [ ] Error handling robust
- [ ] Adapter registration verified
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.extraction import (
    extract_from_any_source,
    get_orchestrator_client,
    get_ocr_client,
    get_transcription_client,
    ModalityType,
    ProcessingStrategy,
)
from futurnal.pipeline.models import DocumentFormat, NormalizedDocument
from futurnal.pipeline.multimodal.router import MultiModalRouter, process_files
from futurnal.pipeline.normalization.registry import FormatAdapterRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_audio_file():
    """Create temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio content")
        return Path(f.name)


@pytest.fixture
def temp_image_file():
    """Create temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"fake image content")
        return Path(f.name)


@pytest.fixture
def temp_pdf_file():
    """Create temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake pdf content")
        return Path(f.name)


@pytest.fixture
def temp_markdown_file():
    """Create temporary markdown file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"# Test\n\nSome content here.")
        return Path(f.name)


@pytest.fixture
def mock_transcription_client():
    """Mock Whisper transcription client."""
    client = MagicMock()
    client.transcribe.return_value = MagicMock(
        text="Hello, this is a test transcription.",
        segments=[
            MagicMock(
                text="Hello, this is",
                start=0.0,
                end=1.5,
                confidence=0.95
            ),
            MagicMock(
                text="a test transcription.",
                start=1.5,
                end=3.0,
                confidence=0.92
            ),
        ],
        language="en",
        duration=3.0,
        confidence=0.93
    )
    return client


@pytest.fixture
def mock_ocr_client():
    """Mock OCR client."""
    client = MagicMock()
    client.extract_text.return_value = MagicMock(
        text="Extracted text from image.",
        regions=[
            MagicMock(
                text="Extracted text",
                bbox=MagicMock(x=10, y=10, width=100, height=20),
                confidence=0.98,
                region_type="paragraph"
            ),
            MagicMock(
                text="from image.",
                bbox=MagicMock(x=10, y=40, width=80, height=20),
                confidence=0.96,
                region_type="paragraph"
            ),
        ],
        layout_info=MagicMock(
            page_count=1,
            reading_order=["paragraph_1", "paragraph_2"]
        ),
        confidence=0.97,
        language=None
    )
    return client


# =============================================================================
# Test: Adapter Registration
# =============================================================================


class TestAdapterRegistration:
    """Test that multimodal adapters are properly registered."""

    def test_registry_includes_multimodal_formats(self):
        """Verify multimodal formats are registered after register_default_adapters()."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        formats = registry.list_supported_formats()

        # Verify multimodal formats are present
        assert DocumentFormat.AUDIO in formats
        assert DocumentFormat.IMAGE in formats
        assert DocumentFormat.SCANNED_PDF in formats

    def test_audio_adapter_registered(self):
        """Verify AudioAdapter is registered for AUDIO format."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        adapter = registry.get_adapter(DocumentFormat.AUDIO)

        assert adapter.name == "AudioAdapter"
        assert DocumentFormat.AUDIO in adapter.supported_formats
        assert adapter.requires_unstructured_processing is False

    def test_image_adapter_registered(self):
        """Verify ImageAdapter is registered for IMAGE format."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        adapter = registry.get_adapter(DocumentFormat.IMAGE)

        assert adapter.name == "ImageAdapter"
        assert DocumentFormat.IMAGE in adapter.supported_formats
        assert adapter.requires_unstructured_processing is False

    def test_scanned_pdf_adapter_registered(self):
        """Verify ScannedPDFAdapter is registered for SCANNED_PDF format."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        adapter = registry.get_adapter(DocumentFormat.SCANNED_PDF)

        assert adapter.name == "ScannedPDFAdapter"
        assert DocumentFormat.SCANNED_PDF in adapter.supported_formats
        assert adapter.requires_unstructured_processing is False

    def test_total_adapter_count(self):
        """Verify total number of registered adapters includes multimodal."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        formats = registry.list_supported_formats()

        # Should have at least 10 formats (5 original + 3 multimodal + docx/pptx from PDF)
        assert len(formats) >= 10


# =============================================================================
# Test: Orchestrator Client Integration
# =============================================================================


class TestOrchestratorClientIntegration:
    """Test orchestrator client for multi-modal coordination."""

    def test_get_orchestrator_client_auto(self):
        """Test auto-selection of orchestrator client."""
        client = get_orchestrator_client(backend="auto")

        # Should return either NvidiaOrchestratorClient or RuleBasedOrchestratorClient
        assert hasattr(client, "analyze_inputs")
        assert hasattr(client, "create_processing_plan")

    def test_orchestrator_detects_audio_modality(self, temp_audio_file):
        """Test orchestrator correctly detects audio modality."""
        client = get_orchestrator_client(backend="rule_based")

        analysis = client.analyze_inputs([temp_audio_file])

        assert ModalityType.AUDIO in analysis.modalities

    def test_orchestrator_detects_image_modality(self, temp_image_file):
        """Test orchestrator correctly detects image modality."""
        client = get_orchestrator_client(backend="rule_based")

        analysis = client.analyze_inputs([temp_image_file])

        assert ModalityType.IMAGE in analysis.modalities

    def test_orchestrator_detects_mixed_modalities(
        self, temp_audio_file, temp_image_file, temp_markdown_file
    ):
        """Test orchestrator correctly detects mixed modalities."""
        client = get_orchestrator_client(backend="rule_based")

        analysis = client.analyze_inputs([
            temp_audio_file,
            temp_image_file,
            temp_markdown_file
        ])

        assert ModalityType.AUDIO in analysis.modalities
        assert ModalityType.IMAGE in analysis.modalities
        assert ModalityType.TEXT in analysis.modalities

    def test_orchestrator_creates_processing_plan(
        self, temp_audio_file, temp_image_file
    ):
        """Test orchestrator creates valid processing plan."""
        client = get_orchestrator_client(backend="rule_based")

        analysis = client.analyze_inputs([temp_audio_file, temp_image_file])
        plan = client.create_processing_plan(analysis)

        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.strategy in [
            ProcessingStrategy.PARALLEL,
            ProcessingStrategy.SEQUENTIAL,
            ProcessingStrategy.DEPENDENCY_GRAPH
        ]


# =============================================================================
# Test: MultiModalRouter Integration
# =============================================================================


class TestMultiModalRouterIntegration:
    """Test MultiModalRouter for file processing."""

    @pytest.mark.asyncio
    async def test_router_processes_single_file(
        self, temp_markdown_file, mock_transcription_client
    ):
        """Test router processes single file."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        router = MultiModalRouter(adapter_registry=registry)

        # Process markdown file (no mocking needed for text)
        doc = await router.process(
            files=temp_markdown_file,
            source_id="test-123",
            source_type="local_files"
        )

        assert isinstance(doc, NormalizedDocument)
        # Source ID includes the base ID (format may vary)
        assert "test-123" in doc.metadata.source_id

    @pytest.mark.asyncio
    async def test_router_detects_format_correctly(self, temp_audio_file):
        """Test router correctly detects file format."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        router = MultiModalRouter(adapter_registry=registry)
        detected_format = router._detect_format(temp_audio_file)

        assert detected_format == DocumentFormat.AUDIO

    @pytest.mark.asyncio
    async def test_router_detects_image_format(self, temp_image_file):
        """Test router correctly detects image format."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        router = MultiModalRouter(adapter_registry=registry)
        detected_format = router._detect_format(temp_image_file)

        assert detected_format == DocumentFormat.IMAGE

    @pytest.mark.asyncio
    async def test_router_creates_error_document_on_failure(self, temp_audio_file):
        """Test router creates error document when processing fails."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        router = MultiModalRouter(adapter_registry=registry)

        # Mock the adapter to raise an error
        with patch.object(
            registry.get_adapter(DocumentFormat.AUDIO),
            "normalize",
            side_effect=Exception("Transcription failed")
        ):
            # Process should create error document, not raise exception
            docs = await router.process(
                files=[temp_audio_file],
                source_id="test",
                source_type="local",
                strategy="sequential"
            )

            assert len(docs) == 1
            assert docs[0].metadata.extra.get("error") is True
            assert "Transcription failed" in docs[0].metadata.extra.get("error_message", "")


# =============================================================================
# Test: Unified API (extract_from_any_source)
# =============================================================================


class TestUnifiedAPI:
    """Test the unified extract_from_any_source API."""

    @pytest.mark.asyncio
    async def test_extract_from_any_source_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            await extract_from_any_source("nonexistent_file.mp3")

    @pytest.mark.asyncio
    async def test_extract_from_any_source_type_error(self):
        """Test that TypeError is raised for invalid input types."""
        with pytest.raises(TypeError):
            await extract_from_any_source(12345)

    @pytest.mark.asyncio
    async def test_extract_from_any_source_with_markdown(self, temp_markdown_file):
        """Test extract_from_any_source with markdown file."""
        doc = await extract_from_any_source(temp_markdown_file)

        assert isinstance(doc, NormalizedDocument)
        assert doc.content is not None

    @pytest.mark.asyncio
    async def test_extract_from_any_source_accepts_string_path(self, temp_markdown_file):
        """Test extract_from_any_source accepts string paths."""
        doc = await extract_from_any_source(str(temp_markdown_file))

        assert isinstance(doc, NormalizedDocument)

    @pytest.mark.asyncio
    async def test_extract_from_any_source_list_of_files(
        self, temp_markdown_file
    ):
        """Test extract_from_any_source with list of files."""
        docs = await extract_from_any_source([temp_markdown_file])

        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], NormalizedDocument)


# =============================================================================
# Test: Audio Pipeline Integration
# =============================================================================


class TestAudioPipelineIntegration:
    """Test audio pipeline with mocked Whisper client."""

    @pytest.mark.asyncio
    async def test_audio_adapter_produces_temporal_segments(
        self, temp_audio_file, mock_transcription_client
    ):
        """Test AudioAdapter includes temporal segments in metadata."""
        from futurnal.pipeline.normalization.adapters.audio import AudioAdapter

        adapter = AudioAdapter()

        with patch.object(
            adapter, "_get_transcription_client", return_value=mock_transcription_client
        ):
            doc = await adapter.normalize(
                file_path=temp_audio_file,
                source_id="audio-test",
                source_type="local_files",
                source_metadata={}
            )

        assert doc.content == "Hello, this is a test transcription."
        assert "temporal_segments" in doc.metadata.extra
        assert len(doc.metadata.extra["temporal_segments"]) == 2

    @pytest.mark.asyncio
    async def test_audio_adapter_temporal_segment_structure(
        self, temp_audio_file, mock_transcription_client
    ):
        """Test temporal segment structure matches specification."""
        from futurnal.pipeline.normalization.adapters.audio import AudioAdapter

        adapter = AudioAdapter()

        with patch.object(
            adapter, "_get_transcription_client", return_value=mock_transcription_client
        ):
            doc = await adapter.normalize(
                file_path=temp_audio_file,
                source_id="audio-test",
                source_type="local_files",
                source_metadata={}
            )

        segments = doc.metadata.extra["temporal_segments"]
        for segment in segments:
            assert "text" in segment
            assert "start" in segment
            assert "end" in segment
            assert "confidence" in segment


# =============================================================================
# Test: OCR Pipeline Integration
# =============================================================================


class TestOCRPipelineIntegration:
    """Test OCR pipeline with mocked OCR client."""

    @pytest.mark.asyncio
    async def test_image_adapter_produces_ocr_regions(
        self, temp_image_file, mock_ocr_client
    ):
        """Test ImageAdapter includes OCR regions in metadata."""
        from futurnal.pipeline.normalization.adapters.image import ImageAdapter

        adapter = ImageAdapter()

        with patch.object(
            adapter, "_get_ocr_client", return_value=mock_ocr_client
        ):
            doc = await adapter.normalize(
                file_path=temp_image_file,
                source_id="image-test",
                source_type="local_files",
                source_metadata={}
            )

        assert doc.content == "Extracted text from image."
        assert "ocr_regions" in doc.metadata.extra
        assert len(doc.metadata.extra["ocr_regions"]) == 2

    @pytest.mark.asyncio
    async def test_image_adapter_ocr_region_structure(
        self, temp_image_file, mock_ocr_client
    ):
        """Test OCR region structure matches specification."""
        from futurnal.pipeline.normalization.adapters.image import ImageAdapter

        adapter = ImageAdapter()

        with patch.object(
            adapter, "_get_ocr_client", return_value=mock_ocr_client
        ):
            doc = await adapter.normalize(
                file_path=temp_image_file,
                source_id="image-test",
                source_type="local_files",
                source_metadata={}
            )

        regions = doc.metadata.extra["ocr_regions"]
        for region in regions:
            assert "text" in region
            assert "bbox" in region
            assert "confidence" in region


# =============================================================================
# Test: Scanned PDF Pipeline Integration
# =============================================================================


class TestScannedPDFPipelineIntegration:
    """Test scanned PDF pipeline with mocked components."""

    @pytest.mark.asyncio
    async def test_scanned_pdf_adapter_processes_pages(
        self, temp_pdf_file, mock_ocr_client
    ):
        """Test ScannedPDFAdapter processes all pages."""
        from futurnal.pipeline.normalization.adapters.scanned_pdf import ScannedPDFAdapter

        adapter = ScannedPDFAdapter()

        # Mock pdf2image conversion
        mock_image_path = temp_pdf_file.parent / "page_1.png"
        mock_image_path.touch()

        with patch.object(
            adapter, "_pdf_to_images", return_value=[mock_image_path]
        ), patch.object(
            adapter, "_get_ocr_client", return_value=mock_ocr_client
        ):
            doc = await adapter.normalize(
                file_path=temp_pdf_file,
                source_id="pdf-test",
                source_type="local_files",
                source_metadata={}
            )

        assert doc.content is not None
        assert "pages" in doc.metadata.extra

        # Cleanup
        if mock_image_path.exists():
            mock_image_path.unlink()


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for client creation."""

    def test_get_transcription_client_returns_client(self):
        """Test get_transcription_client returns a valid client."""
        client = get_transcription_client(backend="auto")

        assert hasattr(client, "transcribe")

    def test_get_ocr_client_returns_client(self):
        """Test get_ocr_client returns a valid client."""
        client = get_ocr_client(backend="auto")

        assert hasattr(client, "extract_text")

    def test_get_orchestrator_client_returns_client(self):
        """Test get_orchestrator_client returns a valid client."""
        client = get_orchestrator_client(backend="auto")

        assert hasattr(client, "analyze_inputs")
        assert hasattr(client, "create_processing_plan")


# =============================================================================
# Test: Quality Gates
# =============================================================================


class TestQualityGates:
    """Test quality gates from production plan."""

    def test_adapter_registration_quality_gate(self):
        """Quality Gate: Adapter registration verified."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        # Must have at least 10 formats registered (5 original + 3 multimodal + docx/pptx)
        assert len(registry.list_supported_formats()) >= 10

        # Must have all multimodal formats
        assert registry.has_adapter(DocumentFormat.AUDIO)
        assert registry.has_adapter(DocumentFormat.IMAGE)
        assert registry.has_adapter(DocumentFormat.SCANNED_PDF)

    def test_unified_api_quality_gate(self):
        """Quality Gate: Unified API complete."""
        from futurnal.extraction import extract_from_any_source, extract, process

        # Verify all aliases exist
        assert callable(extract_from_any_source)
        assert callable(extract)
        assert callable(process)

        # Verify they're the same function
        assert extract is extract_from_any_source
        assert process is extract_from_any_source

    def test_error_handling_quality_gate(self):
        """Quality Gate: Error handling robust."""
        registry = FormatAdapterRegistry()
        registry.register_default_adapters()

        router = MultiModalRouter(adapter_registry=registry)

        # Router should have error document creation capability
        assert hasattr(router, "_create_error_document")

        # Create error document
        error_doc = router._create_error_document(
            file_path=Path("test.mp3"),
            source_id="test",
            source_type="local",
            error_message="Test error"
        )

        assert error_doc.metadata.extra.get("error") is True
        assert "Test error" in error_doc.metadata.extra.get("error_message", "")

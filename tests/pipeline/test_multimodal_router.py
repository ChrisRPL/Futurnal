"""Tests for multi-modal router.

Module 08: Multimodal Integration & Tool Enhancement - Phase 3 Tests
Tests cover routing logic, parallel execution, sequential execution, and error handling.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from futurnal.pipeline.multimodal.router import (
    MultiModalRouter,
    process_files,
)
from futurnal.pipeline.models import (
    NormalizedDocument,
    NormalizedMetadata,
    DocumentFormat,
)
from futurnal.extraction.orchestrator_client import (
    ProcessingStrategy,
    ProcessingPlan,
    ProcessingStep,
    InputAnalysis,
    FileInfo,
    ModalityType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary test files."""
    files = {
        "text": tmp_path / "notes.md",
        "audio": tmp_path / "recording.wav",
        "image": tmp_path / "invoice.png",
        "pdf": tmp_path / "slides.pdf",
    }

    files["text"].write_text("# Notes\n\nTest content")
    files["audio"].write_bytes(b"RIFF" + b"\x00" * 1000)
    files["image"].write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000)
    files["pdf"].write_bytes(b"%PDF-1.4" + b"\x00" * 1000)

    return files


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_registry():
    """Mock adapter registry."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_adapter():
    """Mock format adapter."""
    mock_adapter = AsyncMock()
    mock_adapter.name = "TestAdapter"
    return mock_adapter


@pytest.fixture
def sample_document():
    """Create sample normalized document."""
    from datetime import datetime, timezone
    import hashlib

    content = "Test document content"
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    metadata = NormalizedMetadata(
        source_path="/tmp/test.txt",
        source_id="test-001",
        source_type="local_files",
        format=DocumentFormat.TEXT,
        content_type="text/plain",
        content_hash=content_hash,
        character_count=len(content),
        word_count=len(content.split()),
        line_count=1,
        ingested_at=datetime.now(timezone.utc),
    )

    return NormalizedDocument(
        document_id=content_hash,
        sha256=content_hash,
        content=content,
        metadata=metadata
    )


# =============================================================================
# Router Initialization Tests
# =============================================================================


class TestMultiModalRouterInit:
    """Tests for router initialization."""

    def test_initialization_with_defaults(self):
        """Test router initialization with default components."""
        router = MultiModalRouter()

        assert router.registry is not None
        assert router.orchestrator is not None

    def test_initialization_with_custom_orchestrator(self, mock_orchestrator, mock_registry):
        """Test router initialization with custom orchestrator."""
        router = MultiModalRouter(
            adapter_registry=mock_registry,
            orchestrator=mock_orchestrator
        )

        assert router.orchestrator is mock_orchestrator
        assert router.registry is mock_registry


# =============================================================================
# Format Detection Tests
# =============================================================================


class TestFormatDetection:
    """Tests for document format detection."""

    def test_detect_format_text(self):
        """Test text format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("test.txt")) == DocumentFormat.TEXT
        assert router._detect_format(Path("notes.md")) == DocumentFormat.TEXT
        assert router._detect_format(Path("doc.rst")) == DocumentFormat.TEXT

    def test_detect_format_html(self):
        """Test HTML format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("page.html")) == DocumentFormat.HTML
        assert router._detect_format(Path("page.htm")) == DocumentFormat.HTML

    def test_detect_format_audio(self):
        """Test audio format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("recording.mp3")) == DocumentFormat.AUDIO
        assert router._detect_format(Path("voice.wav")) == DocumentFormat.AUDIO
        assert router._detect_format(Path("song.m4a")) == DocumentFormat.AUDIO

    def test_detect_format_image(self):
        """Test image format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("photo.png")) == DocumentFormat.IMAGE
        assert router._detect_format(Path("scan.jpg")) == DocumentFormat.IMAGE
        assert router._detect_format(Path("diagram.jpeg")) == DocumentFormat.IMAGE

    def test_detect_format_pdf(self):
        """Test PDF format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("document.pdf")) == DocumentFormat.PDF

    def test_detect_format_office(self):
        """Test Office format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("doc.docx")) == DocumentFormat.DOCX
        assert router._detect_format(Path("spreadsheet.xlsx")) == DocumentFormat.XLSX
        assert router._detect_format(Path("slides.pptx")) == DocumentFormat.PPTX

    def test_detect_format_unknown(self):
        """Test unknown format detection."""
        router = MultiModalRouter()

        assert router._detect_format(Path("file.xyz")) == DocumentFormat.UNKNOWN


# =============================================================================
# Strategy Parsing Tests
# =============================================================================


class TestStrategyParsing:
    """Tests for strategy string parsing."""

    def test_parse_strategy_parallel(self):
        """Test parsing parallel strategy."""
        router = MultiModalRouter()

        assert router._parse_strategy("parallel") == ProcessingStrategy.PARALLEL
        assert router._parse_strategy("PARALLEL") == ProcessingStrategy.PARALLEL

    def test_parse_strategy_sequential(self):
        """Test parsing sequential strategy."""
        router = MultiModalRouter()

        assert router._parse_strategy("sequential") == ProcessingStrategy.SEQUENTIAL
        assert router._parse_strategy("SEQUENTIAL") == ProcessingStrategy.SEQUENTIAL

    def test_parse_strategy_dependency(self):
        """Test parsing dependency graph strategy."""
        router = MultiModalRouter()

        assert router._parse_strategy("dependency_graph") == ProcessingStrategy.DEPENDENCY_GRAPH
        assert router._parse_strategy("dependency") == ProcessingStrategy.DEPENDENCY_GRAPH

    def test_parse_strategy_auto(self):
        """Test parsing auto strategy returns None."""
        router = MultiModalRouter()

        assert router._parse_strategy("auto") is None
        assert router._parse_strategy("unknown") is None


# =============================================================================
# Single File Processing Tests
# =============================================================================


class TestProcessSingleFile:
    """Tests for single file processing."""

    @pytest.mark.asyncio
    async def test_process_single_file_success(self, temp_files, sample_document, mock_adapter, mock_registry):
        """Test successful single file processing."""
        mock_adapter.normalize.return_value = sample_document
        mock_registry.get_adapter.return_value = mock_adapter

        router = MultiModalRouter(adapter_registry=mock_registry)

        doc = await router._process_single_file(
            file_path=temp_files["text"],
            source_id="test-001",
            source_type="local_files",
            source_metadata={}
        )

        assert isinstance(doc, NormalizedDocument)
        assert doc.content == sample_document.content
        mock_adapter.normalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_file_no_adapter(self, temp_files, mock_registry):
        """Test single file processing when no adapter found."""
        mock_registry.get_adapter.return_value = None

        router = MultiModalRouter(adapter_registry=mock_registry)

        with pytest.raises(ValueError, match="No adapter found"):
            await router._process_single_file(
                file_path=temp_files["text"],
                source_id="test-001",
                source_type="local_files",
                source_metadata={}
            )


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Tests for parallel processing execution."""

    @pytest.mark.asyncio
    async def test_execute_parallel_success(self, temp_files, sample_document, mock_adapter, mock_registry, mock_orchestrator):
        """Test successful parallel execution."""
        # Setup
        mock_adapter.normalize.return_value = sample_document
        mock_registry.get_adapter.return_value = mock_adapter

        plan = ProcessingPlan(
            steps=[
                ProcessingStep(
                    step_id=0,
                    file_path=temp_files["text"],
                    modality=ModalityType.TEXT,
                    adapter_name="TextAdapter",
                    estimated_time=0.1
                ),
                ProcessingStep(
                    step_id=1,
                    file_path=temp_files["image"],
                    modality=ModalityType.IMAGE,
                    adapter_name="ImageAdapter",
                    estimated_time=2.0
                ),
            ],
            execution_order=[0, 1],
            dependencies={},
            strategy=ProcessingStrategy.PARALLEL,
            estimated_total_time=2.0
        )

        router = MultiModalRouter(adapter_registry=mock_registry, orchestrator=mock_orchestrator)

        documents = await router._execute_parallel(
            plan, "test", "local_files", {}
        )

        assert len(documents) == 2
        assert all(isinstance(doc, NormalizedDocument) for doc in documents)
        assert mock_adapter.normalize.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_parallel_with_error(self, temp_files, sample_document, mock_adapter, mock_registry, mock_orchestrator):
        """Test parallel execution with one file failing."""
        # First call succeeds, second raises exception
        mock_adapter.normalize.side_effect = [
            sample_document,
            Exception("Processing failed")
        ]
        mock_registry.get_adapter.return_value = mock_adapter

        plan = ProcessingPlan(
            steps=[
                ProcessingStep(0, temp_files["text"], ModalityType.TEXT, "TextAdapter", 0.1),
                ProcessingStep(1, temp_files["image"], ModalityType.IMAGE, "ImageAdapter", 2.0),
            ],
            execution_order=[0, 1],
            dependencies={},
            strategy=ProcessingStrategy.PARALLEL,
            estimated_total_time=2.0
        )

        router = MultiModalRouter(adapter_registry=mock_registry, orchestrator=mock_orchestrator)

        documents = await router._execute_parallel(plan, "test", "local_files", {})

        assert len(documents) == 2
        # First document succeeds
        assert documents[0].content == sample_document.content
        # Second document is error placeholder
        assert "[ERROR]" in documents[1].content
        assert documents[1].metadata.extra["error"] is True


# =============================================================================
# Sequential Execution Tests
# =============================================================================


class TestSequentialExecution:
    """Tests for sequential processing execution."""

    @pytest.mark.asyncio
    async def test_execute_sequential_success(self, temp_files, sample_document, mock_adapter, mock_registry, mock_orchestrator):
        """Test successful sequential execution."""
        mock_adapter.normalize.return_value = sample_document
        mock_registry.get_adapter.return_value = mock_adapter

        plan = ProcessingPlan(
            steps=[
                ProcessingStep(0, temp_files["text"], ModalityType.TEXT, "TextAdapter", 0.1),
                ProcessingStep(1, temp_files["image"], ModalityType.IMAGE, "ImageAdapter", 2.0),
            ],
            execution_order=[0, 1],
            dependencies={},
            strategy=ProcessingStrategy.SEQUENTIAL,
            estimated_total_time=2.1
        )

        router = MultiModalRouter(adapter_registry=mock_registry, orchestrator=mock_orchestrator)

        documents = await router._execute_sequential(plan, "test", "local_files", {})

        assert len(documents) == 2
        assert all(isinstance(doc, NormalizedDocument) for doc in documents)
        # Sequential execution should call normalize twice
        assert mock_adapter.normalize.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_sequential_with_error(self, temp_files, sample_document, mock_adapter, mock_registry, mock_orchestrator):
        """Test sequential execution with error."""
        mock_adapter.normalize.side_effect = Exception("Processing failed")
        mock_registry.get_adapter.return_value = mock_adapter

        plan = ProcessingPlan(
            steps=[
                ProcessingStep(0, temp_files["text"], ModalityType.TEXT, "TextAdapter", 0.1),
            ],
            execution_order=[0],
            dependencies={},
            strategy=ProcessingStrategy.SEQUENTIAL,
            estimated_total_time=0.1
        )

        router = MultiModalRouter(adapter_registry=mock_registry, orchestrator=mock_orchestrator)

        documents = await router._execute_sequential(plan, "test", "local_files", {})

        assert len(documents) == 1
        # Should create error document
        assert "[ERROR]" in documents[0].content
        assert documents[0].metadata.extra["error"] is True


# =============================================================================
# End-to-End Processing Tests
# =============================================================================


class TestEndToEndProcessing:
    """Tests for end-to-end file processing."""

    @pytest.mark.asyncio
    async def test_process_single_file(self, temp_files, sample_document, mock_adapter, mock_registry, mock_orchestrator):
        """Test processing single file."""
        # Setup mocks
        mock_adapter.normalize.return_value = sample_document
        mock_registry.get_adapter.return_value = mock_adapter

        mock_analysis = InputAnalysis(
            files=[FileInfo(temp_files["text"], ModalityType.TEXT, 100, 0.1)],
            modalities=[ModalityType.TEXT],
            total_size_bytes=100,
            estimated_total_time=0.1,
            recommended_strategy=ProcessingStrategy.SEQUENTIAL
        )
        mock_orchestrator.analyze_inputs.return_value = mock_analysis

        mock_plan = ProcessingPlan(
            steps=[ProcessingStep(0, temp_files["text"], ModalityType.TEXT, "TextAdapter", 0.1)],
            execution_order=[0],
            dependencies={},
            strategy=ProcessingStrategy.SEQUENTIAL,
            estimated_total_time=0.1,
            rationale="Single file"
        )
        mock_orchestrator.create_processing_plan.return_value = mock_plan

        router = MultiModalRouter(adapter_registry=mock_registry, orchestrator=mock_orchestrator)

        result = await router.process(
            files=temp_files["text"],
            source_id="test",
            source_type="local_files"
        )

        # Should return single document (not list)
        assert isinstance(result, NormalizedDocument)
        assert result.content == sample_document.content

    @pytest.mark.asyncio
    async def test_process_multiple_files(self, temp_files, sample_document, mock_adapter, mock_registry, mock_orchestrator):
        """Test processing multiple files."""
        # Setup mocks
        mock_adapter.normalize.return_value = sample_document
        mock_registry.get_adapter.return_value = mock_adapter

        files_list = [temp_files["text"], temp_files["image"]]
        mock_analysis = InputAnalysis(
            files=[
                FileInfo(files_list[0], ModalityType.TEXT, 100, 0.1),
                FileInfo(files_list[1], ModalityType.IMAGE, 1000, 2.0),
            ],
            modalities=[ModalityType.TEXT, ModalityType.IMAGE],
            total_size_bytes=1100,
            estimated_total_time=2.1,
            recommended_strategy=ProcessingStrategy.PARALLEL
        )
        mock_orchestrator.analyze_inputs.return_value = mock_analysis

        mock_plan = ProcessingPlan(
            steps=[
                ProcessingStep(0, files_list[0], ModalityType.TEXT, "TextAdapter", 0.1),
                ProcessingStep(1, files_list[1], ModalityType.IMAGE, "ImageAdapter", 2.0),
            ],
            execution_order=[0, 1],
            dependencies={},
            strategy=ProcessingStrategy.PARALLEL,
            estimated_total_time=2.0,
            rationale="Parallel processing"
        )
        mock_orchestrator.create_processing_plan.return_value = mock_plan

        router = MultiModalRouter(adapter_registry=mock_registry, orchestrator=mock_orchestrator)

        result = await router.process(
            files=files_list,
            source_id="test",
            source_type="local_files"
        )

        # Should return list of documents
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, NormalizedDocument) for doc in result)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_process_files_single(self, temp_files, sample_document, mock_adapter):
        """Test process_files convenience function with single file."""
        # Mock the router's process method
        with patch("futurnal.pipeline.multimodal.router.MultiModalRouter") as MockRouter:
            mock_router_instance = MockRouter.return_value
            mock_router_instance.process = AsyncMock(return_value=sample_document)

            result = await process_files(temp_files["text"])

            assert isinstance(result, NormalizedDocument)
            mock_router_instance.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_files_multiple(self, temp_files, sample_document):
        """Test process_files convenience function with multiple files."""
        with patch("futurnal.pipeline.multimodal.router.MultiModalRouter") as MockRouter:
            mock_router_instance = MockRouter.return_value
            mock_router_instance.process = AsyncMock(return_value=[sample_document, sample_document])

            files_list = [temp_files["text"], temp_files["image"]]
            result = await process_files(files_list, strategy="parallel")

            assert isinstance(result, list)
            assert len(result) == 2
            mock_router_instance.process.assert_called_once()


# =============================================================================
# Error Document Creation Tests
# =============================================================================


class TestErrorDocumentCreation:
    """Tests for error document creation."""

    def test_create_error_document(self, temp_files):
        """Test creating error placeholder document."""
        router = MultiModalRouter()

        error_doc = router._create_error_document(
            file_path=temp_files["text"],
            source_id="test-001",
            source_type="local_files",
            error_message="Test error"
        )

        assert isinstance(error_doc, NormalizedDocument)
        assert "[ERROR]" in error_doc.content
        assert "Test error" in error_doc.content
        assert error_doc.metadata.extra["error"] is True
        assert error_doc.metadata.extra["error_message"] == "Test error"
        assert error_doc.metadata.source_id == "test-001"
        assert error_doc.metadata.format == DocumentFormat.UNKNOWN

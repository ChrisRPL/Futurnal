"""Tests for orchestrator client.

Module 08: Multimodal Integration & Tool Enhancement - Phase 3 Tests
Tests cover orchestrator input analysis, plan generation, and strategy selection.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from futurnal.extraction.orchestrator_client import (
    NvidiaOrchestratorClient,
    RuleBasedOrchestratorClient,
    get_orchestrator_client,
    nvidia_orchestrator_available,
    ModalityType,
    ProcessingStrategy,
    FileInfo,
    InputAnalysis,
    ProcessingPlan,
    ProcessingStep,
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

    # Create files with some content
    files["text"].write_text("# Meeting Notes\n\nDiscussed Q4 strategy.")
    files["audio"].write_bytes(b"RIFF" + b"\x00" * 1000)  # Fake WAV
    files["image"].write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000)  # Fake PNG
    files["pdf"].write_bytes(b"%PDF-1.4" + b"\x00" * 1000)  # Fake PDF

    return files


@pytest.fixture
def mock_llm_client():
    """Mock Ollama LLM client."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "PARALLEL"
    return mock_client


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Tests for orchestrator data models."""

    def test_modality_type_enum(self):
        """Test ModalityType enum values."""
        assert ModalityType.TEXT == "text"
        assert ModalityType.AUDIO == "audio"
        assert ModalityType.IMAGE == "image"
        assert ModalityType.PDF == "pdf"

    def test_processing_strategy_enum(self):
        """Test ProcessingStrategy enum values."""
        assert ProcessingStrategy.PARALLEL == "parallel"
        assert ProcessingStrategy.SEQUENTIAL == "sequential"
        assert ProcessingStrategy.DEPENDENCY_GRAPH == "dependency_graph"

    def test_file_info_creation(self):
        """Test FileInfo dataclass."""
        info = FileInfo(
            path=Path("/tmp/test.txt"),
            modality=ModalityType.TEXT,
            size_bytes=1024,
            estimated_processing_time=0.5,
            metadata={"key": "value"}
        )

        assert info.path == Path("/tmp/test.txt")
        assert info.modality == ModalityType.TEXT
        assert info.size_bytes == 1024
        assert info.estimated_processing_time == 0.5
        assert info.metadata["key"] == "value"

    def test_processing_step_creation(self):
        """Test ProcessingStep dataclass."""
        step = ProcessingStep(
            step_id=0,
            file_path=Path("/tmp/test.txt"),
            modality=ModalityType.TEXT,
            adapter_name="TextAdapter",
            estimated_time=0.5,
            priority=5
        )

        assert step.step_id == 0
        assert step.adapter_name == "TextAdapter"
        assert step.priority == 5


# =============================================================================
# Rule-Based Orchestrator Tests
# =============================================================================


class TestRuleBasedOrchestratorClient:
    """Tests for rule-based orchestrator (no LLM dependency)."""

    def test_initialization(self):
        """Test rule-based orchestrator initialization."""
        client = RuleBasedOrchestratorClient()
        assert client is not None

    def test_detect_modality_text(self):
        """Test modality detection for text files."""
        client = RuleBasedOrchestratorClient()

        assert client._detect_modality(Path("test.txt")) == ModalityType.TEXT
        assert client._detect_modality(Path("notes.md")) == ModalityType.TEXT
        assert client._detect_modality(Path("doc.rst")) == ModalityType.TEXT
        assert client._detect_modality(Path("page.html")) == ModalityType.TEXT

    def test_detect_modality_audio(self):
        """Test modality detection for audio files."""
        client = RuleBasedOrchestratorClient()

        assert client._detect_modality(Path("recording.mp3")) == ModalityType.AUDIO
        assert client._detect_modality(Path("voice.wav")) == ModalityType.AUDIO
        assert client._detect_modality(Path("song.m4a")) == ModalityType.AUDIO
        assert client._detect_modality(Path("podcast.ogg")) == ModalityType.AUDIO

    def test_detect_modality_image(self):
        """Test modality detection for image files."""
        client = RuleBasedOrchestratorClient()

        assert client._detect_modality(Path("photo.png")) == ModalityType.IMAGE
        assert client._detect_modality(Path("scan.jpg")) == ModalityType.IMAGE
        assert client._detect_modality(Path("diagram.jpeg")) == ModalityType.IMAGE
        assert client._detect_modality(Path("screenshot.tiff")) == ModalityType.IMAGE

    def test_detect_modality_pdf(self):
        """Test modality detection for PDF files."""
        client = RuleBasedOrchestratorClient()

        assert client._detect_modality(Path("document.pdf")) == ModalityType.PDF

    def test_detect_modality_unknown(self):
        """Test modality detection for unknown files."""
        client = RuleBasedOrchestratorClient()

        assert client._detect_modality(Path("file.xyz")) == ModalityType.UNKNOWN

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        client = RuleBasedOrchestratorClient()

        # Text: 0.1s per MB
        assert client._estimate_processing_time(ModalityType.TEXT, 1024 * 1024) == pytest.approx(0.1, 0.01)

        # Audio: 30s per MB
        assert client._estimate_processing_time(ModalityType.AUDIO, 1024 * 1024) == pytest.approx(30, 0.01)

        # Image: 2s per MB
        assert client._estimate_processing_time(ModalityType.IMAGE, 1024 * 1024) == pytest.approx(2, 0.01)

        # PDF: 5s per MB
        assert client._estimate_processing_time(ModalityType.PDF, 1024 * 1024) == pytest.approx(5, 0.01)

    def test_analyze_inputs_single_file(self, temp_files):
        """Test input analysis with single file."""
        client = RuleBasedOrchestratorClient()

        analysis = client.analyze_inputs([temp_files["text"]])

        assert len(analysis.files) == 1
        assert analysis.files[0].modality == ModalityType.TEXT
        assert ModalityType.TEXT in analysis.modalities
        assert analysis.recommended_strategy == ProcessingStrategy.SEQUENTIAL
        assert len(analysis.insights) > 0

    def test_analyze_inputs_multiple_files(self, temp_files):
        """Test input analysis with multiple files."""
        client = RuleBasedOrchestratorClient()

        files = [temp_files["text"], temp_files["audio"], temp_files["image"]]
        analysis = client.analyze_inputs(files)

        assert len(analysis.files) == 3
        assert len(analysis.modalities) == 3
        assert ModalityType.TEXT in analysis.modalities
        assert ModalityType.AUDIO in analysis.modalities
        assert ModalityType.IMAGE in analysis.modalities
        assert analysis.total_size_bytes > 0
        assert analysis.estimated_total_time > 0

    def test_analyze_inputs_many_files(self, temp_files):
        """Test input analysis with many files (should recommend parallel)."""
        client = RuleBasedOrchestratorClient()

        # Create 5 files (>3 threshold for parallel)
        files = [temp_files["text"]] * 5
        analysis = client.analyze_inputs(files)

        assert len(analysis.files) == 5
        assert analysis.recommended_strategy == ProcessingStrategy.PARALLEL

    def test_analyze_inputs_nonexistent_file(self):
        """Test input analysis skips nonexistent files."""
        client = RuleBasedOrchestratorClient()

        analysis = client.analyze_inputs([Path("/nonexistent/file.txt")])

        assert len(analysis.files) == 0
        assert len(analysis.modalities) == 0

    def test_create_processing_plan_parallel(self, temp_files):
        """Test processing plan creation with parallel strategy."""
        client = RuleBasedOrchestratorClient()

        files = [temp_files["text"], temp_files["audio"]]
        analysis = client.analyze_inputs(files)
        plan = client.create_processing_plan(analysis, strategy=ProcessingStrategy.PARALLEL)

        assert plan.strategy == ProcessingStrategy.PARALLEL
        assert len(plan.steps) == 2
        assert len(plan.execution_order) == 2
        assert plan.execution_order == [0, 1]  # All steps in order
        assert plan.estimated_total_time > 0
        # Parallel time should be max of step times
        assert plan.estimated_total_time == max(s.estimated_time for s in plan.steps)

    def test_create_processing_plan_sequential(self, temp_files):
        """Test processing plan creation with sequential strategy."""
        client = RuleBasedOrchestratorClient()

        files = [temp_files["text"], temp_files["audio"]]
        analysis = client.analyze_inputs(files)
        plan = client.create_processing_plan(analysis, strategy=ProcessingStrategy.SEQUENTIAL)

        assert plan.strategy == ProcessingStrategy.SEQUENTIAL
        assert len(plan.steps) == 2
        assert len(plan.execution_order) == 2
        # Sequential time should be sum of step times
        assert plan.estimated_total_time == sum(s.estimated_time for s in plan.steps)

    def test_get_adapter_name(self):
        """Test adapter name mapping."""
        client = RuleBasedOrchestratorClient()

        assert client._get_adapter_name(ModalityType.TEXT) == "TextAdapter"
        assert client._get_adapter_name(ModalityType.AUDIO) == "AudioAdapter"
        assert client._get_adapter_name(ModalityType.IMAGE) == "ImageAdapter"
        assert client._get_adapter_name(ModalityType.PDF) == "PDFAdapter"


# =============================================================================
# NVIDIA Orchestrator Tests
# =============================================================================


class TestNvidiaOrchestratorClient:
    """Tests for NVIDIA Orchestrator-8B client."""

    def test_initialization(self):
        """Test NVIDIA orchestrator initialization."""
        client = NvidiaOrchestratorClient(
            model_name="nvidia/orchestrator-8b",
            base_url="http://localhost:11434"
        )

        assert client.model_name == "nvidia/orchestrator-8b"
        assert client.base_url == "http://localhost:11434"
        assert client._llm_client is None  # Lazy loading

    def test_get_llm_client_success(self, mock_llm_client):
        """Test LLM client lazy loading success."""
        client = NvidiaOrchestratorClient()

        with patch("futurnal.extraction.ollama_client.OllamaLLMClient", return_value=mock_llm_client):
            llm = client._get_llm_client()

            assert llm is not None
            assert client._llm_client is mock_llm_client

    def test_get_llm_client_failure(self):
        """Test LLM client loading failure."""
        client = NvidiaOrchestratorClient()

        # Mock the import to fail
        import sys
        with patch.dict(sys.modules, {"futurnal.extraction.ollama_client": None}):
            with pytest.raises(ImportError):
                client._get_llm_client()

    def test_recommend_strategy_via_llm_parallel(self, temp_files, mock_llm_client):
        """Test LLM strategy recommendation for parallel."""
        client = NvidiaOrchestratorClient()
        mock_llm_client.generate.return_value = "PARALLEL"

        file_infos = [
            FileInfo(temp_files["text"], ModalityType.TEXT, 1024, 0.1),
            FileInfo(temp_files["audio"], ModalityType.AUDIO, 2048, 30.0),
        ]

        strategy = client._recommend_strategy_via_llm(mock_llm_client, file_infos)

        assert strategy == ProcessingStrategy.PARALLEL
        assert mock_llm_client.generate.called

    def test_recommend_strategy_via_llm_sequential(self, temp_files, mock_llm_client):
        """Test LLM strategy recommendation for sequential."""
        client = NvidiaOrchestratorClient()
        mock_llm_client.generate.return_value = "SEQUENTIAL"

        file_infos = [FileInfo(temp_files["text"], ModalityType.TEXT, 1024, 0.1)]

        strategy = client._recommend_strategy_via_llm(mock_llm_client, file_infos)

        assert strategy == ProcessingStrategy.SEQUENTIAL

    def test_recommend_strategy_via_llm_dependency(self, temp_files, mock_llm_client):
        """Test LLM strategy recommendation for dependency graph."""
        client = NvidiaOrchestratorClient()
        mock_llm_client.generate.return_value = "DEPENDENCY_GRAPH"

        file_infos = [
            FileInfo(temp_files["audio"], ModalityType.AUDIO, 2048, 30.0),
            FileInfo(temp_files["pdf"], ModalityType.PDF, 3072, 15.0),
        ]

        strategy = client._recommend_strategy_via_llm(mock_llm_client, file_infos)

        assert strategy == ProcessingStrategy.DEPENDENCY_GRAPH

    def test_recommend_strategy_via_llm_unclear_response(self, temp_files, mock_llm_client):
        """Test LLM strategy recommendation with unclear response."""
        client = NvidiaOrchestratorClient()
        mock_llm_client.generate.return_value = "I'm not sure, maybe do something?"

        file_infos = [FileInfo(temp_files["text"], ModalityType.TEXT, 1024, 0.1)]

        strategy = client._recommend_strategy_via_llm(mock_llm_client, file_infos)

        # Should fall back to PARALLEL
        assert strategy == ProcessingStrategy.PARALLEL

    def test_recommend_strategy_via_llm_error(self, temp_files, mock_llm_client):
        """Test LLM strategy recommendation on error."""
        client = NvidiaOrchestratorClient()
        mock_llm_client.generate.side_effect = Exception("LLM error")

        file_infos = [FileInfo(temp_files["text"], ModalityType.TEXT, 1024, 0.1)]

        strategy = client._recommend_strategy_via_llm(mock_llm_client, file_infos)

        # Should fall back to PARALLEL
        assert strategy == ProcessingStrategy.PARALLEL

    def test_analyze_inputs_with_llm(self, temp_files, mock_llm_client):
        """Test input analysis with LLM orchestration."""
        client = NvidiaOrchestratorClient()

        with patch.object(client, "_get_llm_client", return_value=mock_llm_client):
            files = [temp_files["text"], temp_files["audio"]]
            analysis = client.analyze_inputs(files)

            assert len(analysis.files) == 2
            assert len(analysis.modalities) >= 1
            assert mock_llm_client.generate.called

    def test_calculate_priority(self):
        """Test priority calculation."""
        client = NvidiaOrchestratorClient()

        # Audio: high priority
        audio_info = FileInfo(Path("test.wav"), ModalityType.AUDIO, 1024, 30.0)
        assert client._calculate_priority(audio_info) == 10

        # Image: medium priority
        image_info = FileInfo(Path("test.png"), ModalityType.IMAGE, 1024, 2.0)
        assert client._calculate_priority(image_info) == 5

        # Text: low priority
        text_info = FileInfo(Path("test.txt"), ModalityType.TEXT, 1024, 0.1)
        assert client._calculate_priority(text_info) == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetOrchestratorClient:
    """Tests for orchestrator client factory function."""

    def test_get_orchestrator_explicit_nvidia(self):
        """Test explicit NVIDIA backend selection."""
        client = get_orchestrator_client(backend="nvidia")

        assert isinstance(client, NvidiaOrchestratorClient)

    def test_get_orchestrator_explicit_rule_based(self):
        """Test explicit rule-based backend selection."""
        client = get_orchestrator_client(backend="rule_based")

        assert isinstance(client, RuleBasedOrchestratorClient)

    def test_get_orchestrator_auto_with_nvidia(self, mock_llm_client):
        """Test auto-selection with NVIDIA available."""
        with patch("futurnal.extraction.ollama_client.OllamaLLMClient", return_value=mock_llm_client):
            client = get_orchestrator_client(backend="auto")

            assert isinstance(client, NvidiaOrchestratorClient)

    def test_get_orchestrator_auto_fallback(self):
        """Test auto-selection falls back to rule-based."""
        # Mock the import to fail
        import sys
        with patch.dict(sys.modules, {"futurnal.extraction.ollama_client": None}):
            client = get_orchestrator_client(backend="auto")

            assert isinstance(client, RuleBasedOrchestratorClient)

    def test_get_orchestrator_unknown_backend(self):
        """Test unknown backend falls back to rule-based."""
        client = get_orchestrator_client(backend="unknown_backend")

        assert isinstance(client, RuleBasedOrchestratorClient)

    def test_get_orchestrator_env_variable(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("FUTURNAL_ORCHESTRATOR_BACKEND", "rule_based")

        client = get_orchestrator_client(backend="auto")

        assert isinstance(client, RuleBasedOrchestratorClient)


# =============================================================================
# Availability Check Tests
# =============================================================================


class TestAvailabilityChecks:
    """Tests for backend availability checks."""

    def test_nvidia_orchestrator_available_true(self, mock_llm_client):
        """Test NVIDIA orchestrator availability check returns True."""
        with patch("futurnal.extraction.ollama_client.OllamaLLMClient", return_value=mock_llm_client):
            assert nvidia_orchestrator_available() is True

    def test_nvidia_orchestrator_available_false(self):
        """Test NVIDIA orchestrator availability check returns False."""
        import sys
        with patch.dict(sys.modules, {"futurnal.extraction.ollama_client": None}):
            assert nvidia_orchestrator_available() is False

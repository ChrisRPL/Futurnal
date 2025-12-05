"""Tests for Whisper transcription client infrastructure.

Module 08: Multimodal Integration & Tool Enhancement - Phase 1 Tests
Tests cover Ollama backend, HuggingFace fallback, auto-selection, and error handling.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests

from futurnal.extraction.whisper_client import (
    OllamaWhisperClient,
    LocalWhisperClient,
    get_transcription_client,
    whisper_available,
    get_whisper_models,
    TranscriptionResult,
    TimestampedSegment,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a temporary mock audio file."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"MOCK_AUDIO_DATA" * 100)  # ~1.5KB
    return audio_file


@pytest.fixture
def mock_transcription_result():
    """Mock TranscriptionResult for testing."""
    segments = [
        TimestampedSegment(
            text="Hello world",
            start=0.0,
            end=1.5,
            confidence=0.95
        ),
        TimestampedSegment(
            text="This is a test",
            start=1.5,
            end=3.2,
            confidence=0.98
        )
    ]

    return TranscriptionResult(
        text="Hello world This is a test",
        segments=segments,
        language="en",
        confidence=0.96
    )


# =============================================================================
# OllamaWhisperClient Tests
# =============================================================================


class TestOllamaWhisperClient:
    """Tests for Ollama-based Whisper client."""

    def test_initialization_success(self):
        """Test successful initialization when Ollama is running."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200

            client = OllamaWhisperClient(model_name="whisper:large-v3")

            assert client.model_name == "whisper:large-v3"
            assert client.base_url == "http://localhost:11434"
            assert client.timeout == 600

    def test_initialization_ollama_not_running(self):
        """Test initialization when Ollama is not running (warning logged)."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            client = OllamaWhisperClient()

            # Should still initialize but log warning
            assert client.model_name == "whisper:large-v3"

    def test_transcribe_success(self, mock_audio_file):
        """Test successful transcription via Ollama."""
        with patch("requests.post") as mock_post:
            # Mock Ollama response
            mock_response = {
                "text": "Hello world, this is a test",
                "segments": [
                    {
                        "text": "Hello world",
                        "start": 0.0,
                        "end": 1.2,
                        "confidence": 0.95
                    },
                    {
                        "text": "this is a test",
                        "start": 1.2,
                        "end": 2.8,
                        "confidence": 0.98
                    }
                ],
                "language": "en",
                "confidence": 0.96
            }
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = Mock()

            client = OllamaWhisperClient()
            result = client.transcribe(str(mock_audio_file))

            # Verify result structure
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello world, this is a test"
            assert len(result.segments) == 2
            assert result.language == "en"
            assert result.confidence == 0.96

            # Verify first segment
            assert result.segments[0].text == "Hello world"
            assert result.segments[0].start == 0.0
            assert result.segments[0].end == 1.2
            assert result.segments[0].confidence == 0.95

    def test_transcribe_with_language_specified(self, mock_audio_file):
        """Test transcription with explicit language specification."""
        with patch("requests.post") as mock_post:
            mock_response = {
                "text": "Bonjour le monde",
                "segments": [],
                "language": "fr",
                "confidence": 0.95
            }
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = Mock()

            client = OllamaWhisperClient()
            result = client.transcribe(str(mock_audio_file), language="fr")

            assert result.language == "fr"
            # Verify language was passed in request
            call_args = mock_post.call_args
            assert call_args[1]["json"]["options"]["language"] == "fr"

    def test_transcribe_file_not_found(self):
        """Test transcription with non-existent audio file."""
        client = OllamaWhisperClient()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            client.transcribe("/nonexistent/audio.wav")

    def test_transcribe_timeout(self, mock_audio_file):
        """Test transcription timeout handling."""
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            client = OllamaWhisperClient(timeout=30)

            with pytest.raises(requests.exceptions.Timeout):
                client.transcribe(str(mock_audio_file))

    def test_transcribe_request_error(self, mock_audio_file):
        """Test transcription request error handling."""
        with patch("requests.post", side_effect=requests.exceptions.RequestException("Network error")):
            client = OllamaWhisperClient()

            with pytest.raises(requests.exceptions.RequestException, match="Network error"):
                client.transcribe(str(mock_audio_file))


# =============================================================================
# LocalWhisperClient Tests (HuggingFace Fallback)
# =============================================================================


class TestLocalWhisperClient:
    """Tests for HuggingFace Whisper client fallback."""

    def test_initialization(self):
        """Test HuggingFace client initialization."""
        client = LocalWhisperClient(
            model_name="openai/whisper-large-v3",
            device="cpu"
        )

        assert client.model_name == "openai/whisper-large-v3"
        assert client.device == "cpu"
        assert client.model is None  # Lazy loading
        assert client.processor is None

    @pytest.mark.skip("Requires transformers/torch - heavy dependencies")
    def test_transcribe_success(self, mock_audio_file):
        """Test successful transcription via HuggingFace (requires transformers)."""
        # This test would require actual transformers/torch installation
        # Marked as skip for lightweight testing
        pass

    def test_transcribe_missing_dependencies(self, mock_audio_file):
        """Test transcription fails gracefully when dependencies missing."""
        client = LocalWhisperClient()

        # Patch inside _load_model where the import happens
        with patch.object(client, "_load_model", side_effect=ImportError("transformers not available")):
            with pytest.raises(ImportError):
                client.transcribe(str(mock_audio_file))


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions (whisper_available, get_whisper_models, factory)."""

    def test_whisper_available_true(self):
        """Test whisper_available returns True when Ollama + Whisper installed."""
        with patch("requests.get") as mock_get:
            mock_response = {
                "models": [
                    {"name": "whisper:large-v3"},
                    {"name": "llama3.1:8b"}
                ]
            }
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            assert whisper_available() is True

    def test_whisper_available_false_no_whisper_model(self):
        """Test whisper_available returns False when Whisper model not installed."""
        with patch("requests.get") as mock_get:
            mock_response = {
                "models": [
                    {"name": "llama3.1:8b"},
                    {"name": "phi3:mini"}
                ]
            }
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            assert whisper_available() is False

    def test_whisper_available_false_ollama_not_running(self):
        """Test whisper_available returns False when Ollama not running."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            assert whisper_available() is False

    def test_get_whisper_models_success(self):
        """Test get_whisper_models returns available Whisper models."""
        with patch("requests.get") as mock_get:
            mock_response = {
                "models": [
                    {"name": "whisper:large-v3"},
                    {"name": "whisper:medium"},
                    {"name": "llama3.1:8b"}  # Non-Whisper model
                ]
            }
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()

            models = get_whisper_models()

            assert len(models) == 2
            assert "whisper:large-v3" in models
            assert "whisper:medium" in models
            assert "llama3.1:8b" not in models

    def test_get_whisper_models_empty(self):
        """Test get_whisper_models returns empty list when Ollama not available."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            models = get_whisper_models()
            assert models == []


# =============================================================================
# Factory Function Tests (Auto-Selection)
# =============================================================================


class TestGetTranscriptionClient:
    """Tests for get_transcription_client factory function."""

    def test_auto_selection_ollama_available(self):
        """Test auto-selection chooses Ollama when available."""
        with patch("futurnal.extraction.whisper_client.whisper_available", return_value=True):
            client = get_transcription_client(backend="auto")

            assert isinstance(client, OllamaWhisperClient)
            assert client.model_name == "whisper:large-v3"

    def test_auto_selection_ollama_unavailable(self):
        """Test auto-selection falls back to HuggingFace when Ollama unavailable."""
        with patch("futurnal.extraction.whisper_client.whisper_available", return_value=False):
            client = get_transcription_client(backend="auto")

            assert isinstance(client, LocalWhisperClient)
            assert client.model_name == "openai/whisper-large-v3"

    def test_explicit_ollama_backend(self):
        """Test explicit Ollama backend selection."""
        client = get_transcription_client(backend="ollama")

        assert isinstance(client, OllamaWhisperClient)

    def test_explicit_hf_backend(self):
        """Test explicit HuggingFace backend selection."""
        client = get_transcription_client(backend="hf")

        assert isinstance(client, LocalWhisperClient)

    def test_explicit_huggingface_backend(self):
        """Test explicit HuggingFace backend selection (long form)."""
        client = get_transcription_client(backend="huggingface")

        assert isinstance(client, LocalWhisperClient)

    def test_environment_variable_override(self, monkeypatch):
        """Test environment variable FUTURNAL_AUDIO_BACKEND overrides default."""
        monkeypatch.setenv("FUTURNAL_AUDIO_BACKEND", "hf")

        client = get_transcription_client(backend="auto")

        assert isinstance(client, LocalWhisperClient)

    def test_custom_model_name(self):
        """Test custom model name specification."""
        client = get_transcription_client(
            backend="ollama",
            model_name="whisper:medium"
        )

        assert isinstance(client, OllamaWhisperClient)
        assert client.model_name == "whisper:medium"

    def test_unknown_backend_fallback(self):
        """Test unknown backend falls back to auto-selection."""
        with patch("futurnal.extraction.whisper_client.whisper_available", return_value=True):
            client = get_transcription_client(backend="invalid_backend")

            # Should fall back to auto-selection (Ollama in this case)
            assert isinstance(client, OllamaWhisperClient)


# =============================================================================
# Integration Tests
# =============================================================================


class TestWhisperClientIntegration:
    """Integration tests for Whisper client (requires Ollama running)."""

    @pytest.mark.integration
    @pytest.mark.skipif(not whisper_available(), reason="Requires Ollama with Whisper model")
    def test_real_ollama_transcription(self, mock_audio_file):
        """Test real transcription via Ollama (integration test)."""
        client = get_transcription_client(backend="ollama")

        # Note: This test requires Ollama running with whisper:large-v3
        # It will be skipped in CI/CD unless Ollama is set up
        result = client.transcribe(str(mock_audio_file))

        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert len(result.segments) >= 0
        assert result.language in ["en", "unknown"]  # Mock audio may not be transcribable

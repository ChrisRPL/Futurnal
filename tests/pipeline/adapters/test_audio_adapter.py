"""Tests for AudioAdapter.

Module 08: Multimodal Integration & Tool Enhancement - Phase 1 Tests
Tests cover audio normalization, privacy checks, audit logging, and temporal extraction.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from futurnal.pipeline.normalization.adapters.audio import AudioAdapter
from futurnal.pipeline.models import DocumentFormat, NormalizedDocument
from futurnal.extraction.whisper_client import TranscriptionResult, TimestampedSegment


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def audio_adapter():
    """Create AudioAdapter instance for testing."""
    return AudioAdapter()


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create temporary audio file for testing."""
    audio_file = tmp_path / "test_recording.mp3"
    # Write some mock audio data
    audio_file.write_bytes(b"MOCK_AUDIO_CONTENT" * 1000)  # ~18KB
    return audio_file


@pytest.fixture
def mock_transcription_result():
    """Mock transcription result from Whisper."""
    segments = [
        TimestampedSegment(
            text="Welcome to the meeting",
            start=0.0,
            end=2.5,
            confidence=0.95
        ),
        TimestampedSegment(
            text="Today we will discuss quarterly goals",
            start=2.5,
            end=6.8,
            confidence=0.98
        ),
        TimestampedSegment(
            text="Let's begin with the sales team update",
            start=6.8,
            end=10.2,
            confidence=0.96
        )
    ]

    return TranscriptionResult(
        text="Welcome to the meeting Today we will discuss quarterly goals Let's begin with the sales team update",
        segments=segments,
        language="en",
        confidence=0.96
    )


# =============================================================================
# AudioAdapter Basic Tests
# =============================================================================


class TestAudioAdapterBasics:
    """Test AudioAdapter initialization and basic properties."""

    def test_initialization(self, audio_adapter):
        """Test AudioAdapter initializes correctly."""
        assert audio_adapter.name == "AudioAdapter"
        assert DocumentFormat.AUDIO in audio_adapter.supported_formats
        assert audio_adapter.requires_unstructured_processing is False

    def test_supported_formats(self, audio_adapter):
        """Test AudioAdapter supports only AUDIO format."""
        assert len(audio_adapter.supported_formats) == 1
        assert audio_adapter.supported_formats[0] == DocumentFormat.AUDIO

    @pytest.mark.asyncio
    async def test_validate_existing_file(self, audio_adapter, mock_audio_file):
        """Test validate() accepts existing audio file."""
        is_valid = await audio_adapter.validate(mock_audio_file)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, audio_adapter, tmp_path):
        """Test validate() rejects non-existent file."""
        nonexistent = tmp_path / "nonexistent.mp3"
        is_valid = await audio_adapter.validate(nonexistent)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_wrong_extension(self, audio_adapter, tmp_path):
        """Test validate() rejects files with wrong extension."""
        wrong_ext = tmp_path / "test.txt"
        wrong_ext.write_text("not audio")
        is_valid = await audio_adapter.validate(wrong_ext)
        assert is_valid is False


# =============================================================================
# AudioAdapter Normalization Tests
# =============================================================================


class TestAudioAdapterNormalization:
    """Test AudioAdapter normalization functionality."""

    @pytest.mark.asyncio
    async def test_normalize_success(self, audio_adapter, mock_audio_file, mock_transcription_result):
        """Test successful audio normalization end-to-end."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            # Mock transcription client
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            # Normalize audio file
            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify result structure
            assert isinstance(result, NormalizedDocument)
            assert result.metadata.format == DocumentFormat.AUDIO
            assert result.metadata.source_type == "local_files"
            assert result.metadata.source_id == "audio-123"

            # Verify content is transcribed text
            assert result.content == mock_transcription_result.text
            assert len(result.content) > 0

            # Verify temporal segments stored in metadata
            assert "temporal_segments" in result.metadata.extra
            segments = result.metadata.extra["temporal_segments"]
            assert len(segments) == 3
            assert segments[0]["text"] == "Welcome to the meeting"
            assert segments[0]["start"] == 0.0
            assert segments[0]["end"] == 2.5
            assert segments[0]["confidence"] == 0.95

            # Verify audio metadata
            assert "audio" in result.metadata.extra
            assert result.metadata.extra["audio"]["language"] == "en"
            assert result.metadata.extra["audio"]["overall_confidence"] == 0.96
            assert result.metadata.extra["audio"]["segment_count"] == 3

            # Verify language metadata updated
            assert result.metadata.language == "en"
            assert result.metadata.language_confidence == 0.96

    @pytest.mark.asyncio
    async def test_normalize_nonexistent_file(self, audio_adapter, tmp_path):
        """Test normalization fails gracefully for non-existent file."""
        nonexistent = tmp_path / "missing.mp3"

        with pytest.raises(Exception, match="Audio file not found"):
            await audio_adapter.normalize(
                file_path=nonexistent,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

    @pytest.mark.asyncio
    async def test_normalize_large_file_warning(self, audio_adapter, tmp_path, mock_transcription_result):
        """Test warning logged for large audio files (>100MB)."""
        # Create large mock file (>100MB)
        large_audio = tmp_path / "large_recording.wav"
        large_audio.write_bytes(b"X" * (101 * 1024 * 1024))  # 101MB

        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            with patch("futurnal.pipeline.normalization.adapters.audio.logger") as mock_logger:
                await audio_adapter.normalize(
                    file_path=large_audio,
                    source_id="large-audio",
                    source_type="local_files",
                    source_metadata={}
                )

                # Verify warning was logged
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any("Large audio file" in str(call) for call in warning_calls)

    @pytest.mark.asyncio
    async def test_normalize_transcription_error(self, audio_adapter, mock_audio_file):
        """Test normalization handles transcription errors gracefully."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.side_effect = Exception("Transcription failed")
            mock_get_client.return_value = mock_client

            with pytest.raises(Exception, match="Failed to normalize audio file"):
                await audio_adapter.normalize(
                    file_path=mock_audio_file,
                    source_id="audio-123",
                    source_type="local_files",
                    source_metadata={}
                )


# =============================================================================
# Temporal Segment Tests
# =============================================================================


class TestAudioAdapterTemporalSegments:
    """Test temporal segment extraction and storage."""

    @pytest.mark.asyncio
    async def test_temporal_segments_structure(self, audio_adapter, mock_audio_file, mock_transcription_result):
        """Test temporal segments are correctly structured."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

            segments = result.metadata.extra["temporal_segments"]

            # Verify all segments have required fields
            for seg in segments:
                assert "text" in seg
                assert "start" in seg
                assert "end" in seg
                assert "confidence" in seg
                assert isinstance(seg["start"], (int, float))
                assert isinstance(seg["end"], (int, float))
                assert 0.0 <= seg["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_temporal_segments_ordering(self, audio_adapter, mock_audio_file, mock_transcription_result):
        """Test temporal segments maintain chronological order."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

            segments = result.metadata.extra["temporal_segments"]

            # Verify segments are in chronological order
            for i in range(len(segments) - 1):
                assert segments[i]["end"] <= segments[i + 1]["start"]

    @pytest.mark.asyncio
    async def test_audio_duration_calculation(self, audio_adapter, mock_audio_file, mock_transcription_result):
        """Test audio duration is calculated from last segment."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

            duration = result.metadata.extra["audio_duration_seconds"]

            # Duration should match last segment's end time
            expected_duration = mock_transcription_result.segments[-1].end
            assert duration == expected_duration
            assert duration == 10.2  # From fixture


# =============================================================================
# Metadata Tests
# =============================================================================


class TestAudioAdapterMetadata:
    """Test metadata extraction and enrichment."""

    @pytest.mark.asyncio
    async def test_audio_metadata_populated(self, audio_adapter, mock_audio_file, mock_transcription_result):
        """Test audio-specific metadata is populated."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={"custom_key": "custom_value"}
            )

            audio_meta = result.metadata.extra["audio"]

            assert audio_meta["language"] == "en"
            assert audio_meta["overall_confidence"] == 0.96
            assert audio_meta["segment_count"] == 3
            assert "transcription_backend" in audio_meta

            # Verify custom source metadata is preserved
            assert result.metadata.extra["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_file_size_metadata(self, audio_adapter, mock_audio_file, mock_transcription_result):
        """Test file size metadata is calculated."""
        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = mock_transcription_result
            mock_get_client.return_value = mock_client

            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify file size is recorded
            assert "audio_file_size_mb" in result.metadata.extra
            file_size_mb = result.metadata.extra["audio_file_size_mb"]
            assert file_size_mb > 0

            # Verify against actual file size
            actual_size_mb = mock_audio_file.stat().st_size / (1024 * 1024)
            assert abs(file_size_mb - actual_size_mb) < 0.01  # Within 10KB

    @pytest.mark.asyncio
    async def test_language_detection_updates_metadata(self, audio_adapter, mock_audio_file):
        """Test language detection updates document metadata."""
        # Create transcription result with non-English language
        french_result = TranscriptionResult(
            text="Bonjour le monde",
            segments=[
                TimestampedSegment(text="Bonjour le monde", start=0.0, end=2.0, confidence=0.95)
            ],
            language="fr",
            confidence=0.95
        )

        with patch.object(audio_adapter, "_get_transcription_client") as mock_get_client:
            mock_client = Mock()
            mock_client.transcribe.return_value = french_result
            mock_get_client.return_value = mock_client

            result = await audio_adapter.normalize(
                file_path=mock_audio_file,
                source_id="audio-123",
                source_type="local_files",
                source_metadata={}
            )

            assert result.metadata.language == "fr"
            assert result.metadata.language_confidence == 0.95


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestAudioAdapterIntegration:
    """Integration tests requiring real transcription backend."""

    @pytest.mark.skipif(True, reason="Requires Ollama with Whisper model - manual testing only")
    @pytest.mark.asyncio
    async def test_real_audio_transcription(self, audio_adapter, tmp_path):
        """Test real audio transcription (requires Ollama running)."""
        # This test requires:
        # 1. Ollama running: ollama serve
        # 2. Whisper model: ollama pull whisper:large-v3
        # 3. Real audio file

        # For manual testing only
        audio_file = tmp_path / "real_audio.wav"
        # User must provide real audio file

        if not audio_file.exists():
            pytest.skip("Real audio file not provided")

        result = await audio_adapter.normalize(
            file_path=audio_file,
            source_id="integration-test",
            source_type="local_files",
            source_metadata={}
        )

        assert isinstance(result, NormalizedDocument)
        assert len(result.content) > 0
        assert len(result.metadata.extra["temporal_segments"]) > 0

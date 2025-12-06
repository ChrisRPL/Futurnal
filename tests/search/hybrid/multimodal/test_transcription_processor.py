"""Tests for TranscriptionProcessor.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Tests cover:
- Whisper output processing
- Audio quality classification
- WER estimation
- Homophone expansion
- Speaker extraction
- Segment processing
- Source type detection
"""

import pytest
from datetime import datetime

from futurnal.search.hybrid.multimodal.transcription_processor import (
    TranscriptionProcessor,
    AudioQuality,
    TranscriptionMetadata,
)
from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
)


class TestAudioQuality:
    """Tests for AudioQuality enum."""

    def test_audio_quality_values(self):
        """Test all audio quality values exist."""
        assert AudioQuality.STUDIO == "studio"
        assert AudioQuality.CLEAN == "clean"
        assert AudioQuality.NOISY == "noisy"
        assert AudioQuality.MIXED == "mixed"
        assert AudioQuality.POOR == "poor"

    def test_audio_quality_is_string_enum(self):
        """Test AudioQuality is string enum."""
        assert isinstance(AudioQuality.STUDIO, str)
        assert AudioQuality.STUDIO.value == "studio"


class TestTranscriptionMetadata:
    """Tests for TranscriptionMetadata dataclass."""

    def test_metadata_creation(self):
        """Test metadata creation with all fields."""
        metadata = TranscriptionMetadata(
            source_file="recording.mp3",
            duration_seconds=120.5,
            audio_quality=AudioQuality.CLEAN,
            word_error_rate=0.05,
            speaker_count=2,
            language="en",
            model_version="whisper-v3-turbo",
            has_timestamps=True,
            has_speaker_labels=True,
            confidence_scores=[0.9, 0.85, 0.92],
        )
        assert metadata.source_file == "recording.mp3"
        assert metadata.duration_seconds == 120.5
        assert metadata.audio_quality == AudioQuality.CLEAN
        assert metadata.word_error_rate == 0.05
        assert metadata.speaker_count == 2

    def test_metadata_empty_confidence(self):
        """Test metadata with empty confidence scores."""
        metadata = TranscriptionMetadata(
            source_file="audio.wav",
            duration_seconds=60.0,
            audio_quality=AudioQuality.NOISY,
            word_error_rate=0.15,
            speaker_count=1,
            language="en",
            model_version="whisper-v3-turbo",
            has_timestamps=False,
            has_speaker_labels=False,
            confidence_scores=[],
        )
        assert metadata.confidence_scores == []


class TestTranscriptionProcessor:
    """Tests for TranscriptionProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_init_builds_homophone_map(self):
        """Test homophone map is built on init."""
        assert len(self.processor._homophone_map) > 0
        assert "their" in self.processor._homophone_map
        assert "there" in self.processor._homophone_map["their"]

    def test_homophone_groups_defined(self):
        """Test homophone groups are defined."""
        assert len(TranscriptionProcessor.HOMOPHONE_GROUPS) > 0
        # Check a known group
        assert any(
            "their" in group and "there" in group
            for group in TranscriptionProcessor.HOMOPHONE_GROUPS
        )

    def test_filler_words_defined(self):
        """Test filler words are defined."""
        assert "um" in TranscriptionProcessor.FILLER_WORDS
        assert "uh" in TranscriptionProcessor.FILLER_WORDS
        assert "like" in TranscriptionProcessor.FILLER_WORDS


class TestTextExtraction:
    """Tests for text extraction from Whisper output."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_extract_text_direct(self):
        """Test extraction from direct text field."""
        whisper_output = {
            "text": "Hello world",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["content"] == "Hello world"

    def test_extract_text_from_segments(self):
        """Test extraction from segments when no direct text."""
        whisper_output = {
            "segments": [
                {"text": "First segment", "start": 0, "end": 5},
                {"text": "Second segment", "start": 5, "end": 10},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        assert "First segment" in result["content"]
        assert "Second segment" in result["content"]


class TestTextCleaning:
    """Tests for transcription text cleaning."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_clean_repeated_fillers(self):
        """Test repeated filler words are reduced."""
        whisper_output = {
            "text": "So um um like the project",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        # Should not have "um um"
        assert "um um" not in result["content"].lower()

    def test_normalize_whitespace(self):
        """Test whitespace is normalized."""
        whisper_output = {
            "text": "Multiple   spaces   here",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        assert "  " not in result["content"]


class TestAudioQualityClassification:
    """Tests for audio quality classification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_studio_quality(self):
        """Test studio quality classification (SNR > 30)."""
        whisper_output = {
            "text": "Clear recording",
            "segments": [],
            "audio_quality": {"snr": 35},
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["audio_quality"] == "studio"

    def test_clean_quality(self):
        """Test clean quality classification (SNR 20-30)."""
        whisper_output = {
            "text": "Good recording",
            "segments": [],
            "audio_quality": {"snr": 25},
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["audio_quality"] == "clean"

    def test_noisy_quality(self):
        """Test noisy quality classification (SNR 10-20)."""
        whisper_output = {
            "text": "Noisy recording",
            "segments": [],
            "audio_quality": {"snr": 15},
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["audio_quality"] == "noisy"

    def test_poor_quality(self):
        """Test poor quality classification (SNR <= 10)."""
        whisper_output = {
            "text": "Poor recording",
            "segments": [],
            "audio_quality": {"snr": 5},
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["audio_quality"] == "poor"

    def test_mixed_quality(self):
        """Test mixed quality classification."""
        whisper_output = {
            "text": "Variable recording",
            "segments": [],
            "audio_quality": {"snr": 15, "variable": True},
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["audio_quality"] == "mixed"

    def test_default_clean_quality(self):
        """Test default to clean quality when no info."""
        whisper_output = {
            "text": "Unknown quality",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["audio_quality"] == "clean"


class TestWEREstimation:
    """Tests for Word Error Rate estimation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_wer_from_high_confidence(self):
        """Test WER estimation from high confidence."""
        whisper_output = {
            "text": "Clear text",
            "segments": [{"text": "Clear text", "confidence": 0.95}],
        }
        result = self.processor.process_transcription(whisper_output)
        # High confidence should give low WER
        assert result["source_metadata"]["word_error_rate"] < 0.1

    def test_wer_from_low_confidence(self):
        """Test WER estimation from low confidence."""
        whisper_output = {
            "text": "Unclear text",
            "segments": [{"text": "Unclear text", "confidence": 0.6}],
        }
        result = self.processor.process_transcription(whisper_output)
        # Low confidence should give higher WER
        assert result["source_metadata"]["word_error_rate"] > 0.1

    def test_wer_from_log_prob(self):
        """Test WER estimation from log probability."""
        whisper_output = {
            "text": "Text",
            "segments": [{"text": "Text", "avg_logprob": -0.3}],
        }
        result = self.processor.process_transcription(whisper_output)
        assert "word_error_rate" in result["source_metadata"]


class TestQualityTier:
    """Tests for quality tier mapping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_high_quality_tier(self):
        """Test high quality tier for low WER."""
        whisper_output = {
            "text": "Clear transcription",
            "segments": [{"text": "Clear transcription", "confidence": 0.98}],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["extraction_quality"] == "high"

    def test_medium_quality_tier(self):
        """Test medium quality tier for moderate WER."""
        whisper_output = {
            "text": "Moderate transcription",
            "segments": [{"text": "Moderate transcription", "confidence": 0.85}],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["extraction_quality"] == "medium"

    def test_low_quality_tier(self):
        """Test low quality tier for higher WER."""
        whisper_output = {
            "text": "Lower quality transcription",
            "segments": [{"text": "Lower quality", "confidence": 0.65}],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["extraction_quality"] in ("low", "medium")


class TestHomophoneExpansion:
    """Tests for homophone expansion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_expand_their_there(self):
        """Test their/there homophone expansion."""
        query = "check their work"
        expanded = self.processor.expand_query_for_transcription(query)
        assert "their" in expanded
        assert "(there)" in expanded or "(they're)" in expanded

    def test_expand_to_too(self):
        """Test to/too homophone expansion."""
        query = "go to the store"
        expanded = self.processor.expand_query_for_transcription(query)
        assert "to" in expanded
        assert "(too)" in expanded or "(two)" in expanded

    def test_no_expansion_for_regular_words(self):
        """Test no expansion for non-homophones."""
        query = "project deadline"
        expanded = self.processor.expand_query_for_transcription(query)
        assert "(" not in expanded

    def test_max_two_alternatives(self):
        """Test max two alternatives per homophone."""
        query = "write right rite"
        expanded = self.processor.expand_query_for_transcription(query)
        # Each word should have at most 2 alternatives
        # Count parentheses per original word
        words = expanded.split()
        parenthesized = [w for w in words if w.startswith("(")]
        # write has 2 homophones, right has 2, rite has 2
        # Should have max 6 alternatives total (2 per word)
        assert len(parenthesized) <= 6


class TestSearchableContent:
    """Tests for searchable content creation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_searchable_content_includes_homophones(self):
        """Test searchable content includes homophone variants."""
        whisper_output = {
            "text": "I know the way",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        searchable = result["searchable_content"]
        # Should contain "know" and "no" variant
        assert "know" in searchable
        assert "no" in searchable

    def test_searchable_content_lowercase(self):
        """Test searchable content is lowercase."""
        whisper_output = {
            "text": "HELLO World",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["searchable_content"] == result["searchable_content"].lower()


class TestHomophoneLookup:
    """Tests for homophone lookup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_get_homophones_their(self):
        """Test getting homophones for 'their'."""
        homophones = self.processor.get_homophones("their")
        assert "there" in homophones
        assert "they're" in homophones
        assert "their" not in homophones  # Should not include self

    def test_get_homophones_unknown(self):
        """Test getting homophones for unknown word."""
        homophones = self.processor.get_homophones("project")
        assert homophones == []

    def test_get_homophones_case_insensitive(self):
        """Test homophones lookup is case insensitive."""
        homophones = self.processor.get_homophones("THEIR")
        assert "there" in homophones


class TestSegmentExtraction:
    """Tests for segment extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_extract_segments_with_timestamps(self):
        """Test segment extraction with timestamps."""
        whisper_output = {
            "text": "Full text",
            "segments": [
                {"text": "First", "start": 0.0, "end": 2.5, "confidence": 0.9},
                {"text": "Second", "start": 2.5, "end": 5.0, "confidence": 0.85},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        segments = result["segments"]
        assert len(segments) == 2
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.5
        assert segments[0]["text"] == "First"

    def test_extract_segments_with_speaker(self):
        """Test segment extraction with speaker labels."""
        whisper_output = {
            "text": "Conversation",
            "segments": [
                {"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"},
                {"text": "Hi", "start": 1, "end": 2, "speaker": "SPEAKER_01"},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        segments = result["segments"]
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[1]["speaker"] == "SPEAKER_01"


class TestSpeakerExtraction:
    """Tests for speaker extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_extract_speakers(self):
        """Test speaker information extraction."""
        whisper_output = {
            "text": "Conversation",
            "segments": [
                {"text": "Hello", "start": 0, "end": 2, "speaker": "SPEAKER_00"},
                {"text": "Hi there", "start": 2, "end": 5, "speaker": "SPEAKER_01"},
                {"text": "How are you", "start": 5, "end": 8, "speaker": "SPEAKER_00"},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        speakers = result["speakers"]
        assert len(speakers) == 2
        # Find speaker 00
        speaker_00 = next(s for s in speakers if s["id"] == "SPEAKER_00")
        assert len(speaker_00["segments"]) == 2
        assert speaker_00["total_duration"] == 5.0  # 2 + 3 seconds

    def test_no_speakers(self):
        """Test handling of no speaker labels."""
        whisper_output = {
            "text": "Monologue",
            "segments": [
                {"text": "Just me talking", "start": 0, "end": 5},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["speakers"] == []

    def test_speaker_count_metadata(self):
        """Test speaker count in metadata."""
        whisper_output = {
            "text": "Meeting",
            "segments": [
                {"text": "A", "speaker": "S1"},
                {"text": "B", "speaker": "S2"},
                {"text": "C", "speaker": "S3"},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["speaker_count"] == 3


class TestSourceTypeDetection:
    """Tests for source type detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_detect_audio_source(self):
        """Test audio source detection."""
        whisper_output = {
            "text": "Audio content",
            "segments": [],
            "source_file": "recording.mp3",
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["source_type"] == "audio_transcription"

    def test_detect_video_source_mp4(self):
        """Test video source detection for MP4."""
        whisper_output = {
            "text": "Video content",
            "segments": [],
            "source_file": "video.mp4",
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["source_type"] == "video_transcription"

    def test_detect_video_source_mov(self):
        """Test video source detection for MOV."""
        whisper_output = {
            "text": "Video content",
            "segments": [],
            "source_file": "clip.mov",
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["source_type"] == "video_transcription"

    def test_default_audio_source(self):
        """Test default to audio for unknown source."""
        whisper_output = {
            "text": "Unknown source",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["source_type"] == "audio_transcription"


class TestFormatDetection:
    """Tests for audio format detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_detect_mp3_format(self):
        """Test MP3 format detection."""
        whisper_output = {
            "text": "Content",
            "segments": [],
            "source_file": "audio.mp3",
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["original_format"] == "mp3"

    def test_detect_wav_format(self):
        """Test WAV format detection."""
        whisper_output = {
            "text": "Content",
            "segments": [],
            "source_file": "audio.wav",
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["original_format"] == "wav"

    def test_detect_m4a_format(self):
        """Test M4A format detection."""
        whisper_output = {
            "text": "Content",
            "segments": [],
            "source_file": "voice.m4a",
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["original_format"] == "m4a"

    def test_default_audio_format(self):
        """Test default format for unknown extension."""
        whisper_output = {
            "text": "Content",
            "segments": [],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["source_metadata"]["original_format"] == "audio"


class TestDurationCalculation:
    """Tests for duration calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_duration_from_segments(self):
        """Test duration calculation from segments."""
        whisper_output = {
            "text": "Content",
            "segments": [
                {"text": "Start", "start": 0, "end": 30},
                {"text": "Middle", "start": 30, "end": 60},
                {"text": "End", "start": 60, "end": 90},
            ],
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["duration_seconds"] == 90

    def test_duration_from_field(self):
        """Test duration from duration field."""
        whisper_output = {
            "text": "Content",
            "segments": [],
            "duration": 120.5,
        }
        result = self.processor.process_transcription(whisper_output)
        assert result["transcription_metadata"]["duration_seconds"] == 120.5


class TestFullProcessing:
    """Integration tests for full transcription processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptionProcessor()

    def test_full_processing_audio(self):
        """Test full processing of audio transcription."""
        whisper_output = {
            "text": "Hello, this is a test recording about the project meeting.",
            "segments": [
                {
                    "text": "Hello, this is a test recording",
                    "start": 0.0,
                    "end": 3.5,
                    "confidence": 0.95,
                    "speaker": "SPEAKER_00",
                },
                {
                    "text": "about the project meeting.",
                    "start": 3.5,
                    "end": 6.0,
                    "confidence": 0.92,
                    "speaker": "SPEAKER_00",
                },
            ],
            "language": "en",
            "model": "whisper-v3-turbo",
            "source_file": "meeting.mp3",
            "audio_quality": {"snr": 25},
        }
        result = self.processor.process_transcription(whisper_output)

        # Verify structure
        assert "content" in result
        assert "searchable_content" in result
        assert "source_metadata" in result
        assert "transcription_metadata" in result
        assert "segments" in result
        assert "speakers" in result

        # Verify source metadata
        assert result["source_metadata"]["source_type"] == "audio_transcription"
        assert result["source_metadata"]["extraction_quality"] in ("high", "medium")
        assert result["source_metadata"]["original_format"] == "mp3"
        assert result["source_metadata"]["language_detected"] == "en"

        # Verify transcription metadata
        assert result["transcription_metadata"]["duration_seconds"] == 6.0
        assert result["transcription_metadata"]["speaker_count"] == 1
        assert result["transcription_metadata"]["has_timestamps"] is True
        assert result["transcription_metadata"]["has_speaker_labels"] is True
        assert result["transcription_metadata"]["audio_quality"] == "clean"

        # Verify segments
        assert len(result["segments"]) == 2
        assert result["segments"][0]["speaker"] == "SPEAKER_00"

        # Verify speakers
        assert len(result["speakers"]) == 1

    def test_full_processing_video(self):
        """Test full processing of video transcription."""
        whisper_output = {
            "text": "Welcome to the tutorial video",
            "segments": [
                {
                    "text": "Welcome to the tutorial video",
                    "start": 0,
                    "end": 4,
                    "avg_logprob": -0.15,
                },
            ],
            "source_file": "tutorial.mp4",
            "language": "en",
        }
        result = self.processor.process_transcription(whisper_output)

        assert result["source_metadata"]["source_type"] == "video_transcription"
        assert result["transcription_metadata"]["has_speaker_labels"] is False

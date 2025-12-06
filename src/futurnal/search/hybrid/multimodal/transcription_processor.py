"""Transcription Processor for audio search optimization.

Processes Whisper V3 transcription output for optimal retrieval, including:
- Homophone handling for transcription errors
- Filler word cleanup
- Segment-level metadata extraction

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Option B Compliance:
- Ghost model frozen (Whisper used for extraction only)
- Local-first processing
- Quality target: >75% audio content relevance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import re

from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
    SourceMetadata,
)


class AudioQuality(str, Enum):
    """Audio quality classification.

    Affects retrieval confidence weighting based on recording conditions.
    """

    STUDIO = "studio"  # Professional recording
    CLEAN = "clean"  # Good quality, minimal noise
    NOISY = "noisy"  # Background noise present
    MIXED = "mixed"  # Variable quality
    POOR = "poor"  # Significant quality issues


@dataclass
class TranscriptionMetadata:
    """Metadata for audio transcription content."""

    source_file: str
    duration_seconds: float
    audio_quality: AudioQuality
    word_error_rate: float
    speaker_count: int
    language: str
    model_version: str
    has_timestamps: bool
    has_speaker_labels: bool
    confidence_scores: List[float]


class TranscriptionProcessor:
    """Processes Whisper V3 transcriptions for optimal retrieval.

    Handles transcription-specific challenges:
    - Speaker diarization noise
    - Homophone confusion
    - Filler word handling
    - Timestamp alignment

    Integration Points:
    - MultimodalQueryHandler: Transcription-specific search strategies
    - PKGClient: Stores transcription metadata with content
    """

    # Common transcription homophone confusions
    # Each group contains words that sound similar
    HOMOPHONE_GROUPS: List[List[str]] = [
        ["their", "there", "they're"],
        ["to", "too", "two"],
        ["its", "it's"],
        ["your", "you're"],
        ["hear", "here"],
        ["know", "no"],
        ["write", "right", "rite"],
        ["weather", "whether"],
        ["where", "wear", "ware"],
        ["which", "witch"],
        ["peace", "piece"],
        ["weak", "week"],
        ["made", "maid"],
        ["whole", "hole"],
        ["would", "wood"],
        ["wait", "weight"],
        ["sea", "see"],
        ["meet", "meat"],
    ]

    # Filler words to handle specially
    FILLER_WORDS: List[str] = [
        "um",
        "uh",
        "like",
        "you know",
        "basically",
        "actually",
        "literally",
        "honestly",
        "right",
        "so",
        "well",
        "I mean",
        "kind of",
        "sort of",
    ]

    def __init__(self) -> None:
        """Initialize transcription processor."""
        self._homophone_map = self._build_homophone_map()

    def _build_homophone_map(self) -> Dict[str, List[str]]:
        """Build homophone lookup for search expansion.

        Returns:
            Dictionary mapping each word to its homophones
        """
        hmap: Dict[str, List[str]] = {}
        for group in self.HOMOPHONE_GROUPS:
            for word in group:
                hmap[word] = [w for w in group if w != word]
        return hmap

    def process_transcription(
        self, whisper_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process Whisper V3 output for PKG storage.

        Transforms raw Whisper output into indexed format with metadata
        for source-aware retrieval.

        Args:
            whisper_output: Raw output from Whisper/WhisperClient
                Expected keys: text, segments, language, confidence

        Returns:
            Processed content with metadata for PKG storage:
            - content: Cleaned transcription text
            - searchable_content: Expanded text with homophones
            - source_metadata: SourceMetadata dict
            - transcription_metadata: Audio-specific metadata
            - segments: Timestamped segments
            - speakers: Speaker information (if available)
        """
        # Extract and clean text
        text_content = self._extract_clean_text(whisper_output)

        # Build metadata
        metadata = self._build_metadata(whisper_output)

        # Generate search-optimized content
        searchable_content = self._create_searchable_content(
            text_content, whisper_output
        )

        # Build source metadata for retrieval
        source_metadata = SourceMetadata(
            source_type=self._determine_source_type(whisper_output),
            extraction_confidence=self._avg_confidence(whisper_output),
            extraction_quality=self._quality_tier(metadata.word_error_rate),
            extractor_version=metadata.model_version,
            extraction_timestamp=datetime.utcnow(),
            original_format=self._detect_format(whisper_output),
            language_detected=metadata.language,
            word_error_rate=metadata.word_error_rate,
            audio_quality=metadata.audio_quality.value,
        )

        return {
            "content": text_content,
            "searchable_content": searchable_content,
            "source_metadata": source_metadata.to_dict(),
            "transcription_metadata": {
                "duration_seconds": metadata.duration_seconds,
                "speaker_count": metadata.speaker_count,
                "has_timestamps": metadata.has_timestamps,
                "has_speaker_labels": metadata.has_speaker_labels,
                "language": metadata.language,
                "audio_quality": metadata.audio_quality.value,
            },
            "segments": self._extract_segments(whisper_output),
            "speakers": self._extract_speakers(whisper_output),
        }

    def _extract_clean_text(self, whisper_output: Dict[str, Any]) -> str:
        """Extract clean text from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            Cleaned transcription text
        """
        if "text" in whisper_output and whisper_output["text"]:
            text = whisper_output["text"]
        else:
            segments = whisper_output.get("segments", [])
            text = " ".join(s.get("text", "") for s in segments)

        # Clean up common transcription artifacts
        text = self._clean_transcription(text)

        return text

    def _clean_transcription(self, text: str) -> str:
        """Clean transcription artifacts.

        Removes repeated filler words and normalizes whitespace.

        Args:
            text: Raw transcription text

        Returns:
            Cleaned text
        """
        # Remove repeated filler words
        for filler in self.FILLER_WORDS:
            # Match repeated fillers (e.g., "um um" -> "um")
            pattern = rf"\b({re.escape(filler)})\s+\1\b"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _build_metadata(
        self, whisper_output: Dict[str, Any]
    ) -> TranscriptionMetadata:
        """Build transcription metadata from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            TranscriptionMetadata with extracted information
        """
        segments = whisper_output.get("segments", [])

        # Calculate duration
        if segments:
            duration = segments[-1].get("end", 0)
        else:
            duration = whisper_output.get("duration", 0)

        # Get confidence scores
        confidences = [
            s.get("confidence", s.get("avg_logprob", -0.5)) for s in segments
        ]

        # Estimate WER from confidence
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        estimated_wer = self._estimate_wer_from_confidence(avg_conf)

        # Count unique speakers
        speakers = set(s.get("speaker", "") for s in segments if s.get("speaker"))

        return TranscriptionMetadata(
            source_file=whisper_output.get("source_file", ""),
            duration_seconds=duration,
            audio_quality=self._classify_audio_quality(whisper_output),
            word_error_rate=estimated_wer,
            speaker_count=len(speakers),
            language=whisper_output.get("language", "en"),
            model_version=whisper_output.get("model", "whisper-v3-turbo"),
            has_timestamps=any("start" in s for s in segments),
            has_speaker_labels=any("speaker" in s for s in segments),
            confidence_scores=confidences,
        )

    def _estimate_wer_from_confidence(self, avg_confidence: float) -> float:
        """Estimate Word Error Rate from average confidence.

        Args:
            avg_confidence: Average segment confidence

        Returns:
            Estimated WER (0.0-1.0)
        """
        # Log prob to confidence mapping
        if avg_confidence < 0:  # Log prob format
            confidence = min(1.0, max(0.0, 1.0 + (avg_confidence / 4)))
        else:
            confidence = avg_confidence

        # WER estimation: lower confidence = higher WER
        return max(0.01, (1 - confidence) * 0.4)

    def _classify_audio_quality(
        self, whisper_output: Dict[str, Any]
    ) -> AudioQuality:
        """Classify audio quality from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            AudioQuality classification
        """
        quality_info = whisper_output.get("audio_quality", {})

        # No audio quality info - default to clean
        if not quality_info:
            return AudioQuality.CLEAN

        # Check variable quality first (mixed conditions)
        if quality_info.get("variable", False):
            return AudioQuality.MIXED

        snr = quality_info.get("snr", 20)

        if snr > 30:
            return AudioQuality.STUDIO
        elif snr >= 20:
            return AudioQuality.CLEAN
        elif snr > 10:
            return AudioQuality.NOISY
        else:
            return AudioQuality.POOR

    def _create_searchable_content(
        self, text: str, whisper_output: Dict[str, Any]
    ) -> str:
        """Create search-optimized content with homophone expansions.

        Expands words with their homophones to improve recall
        when searching transcriptions that may have homophone errors.

        Args:
            text: Cleaned transcription text
            whisper_output: Raw Whisper output

        Returns:
            Search-optimized content
        """
        words = text.lower().split()
        expanded_words: List[str] = []

        for word in words:
            expanded_words.append(word)
            # Add homophone variants
            if word in self._homophone_map:
                expanded_words.extend(self._homophone_map[word])

        return " ".join(expanded_words)

    def _extract_segments(
        self, whisper_output: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract timestamped segments from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            List of segment dictionaries
        """
        segments = whisper_output.get("segments", [])
        return [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("text", ""),
                "speaker": s.get("speaker"),
                "confidence": s.get("confidence", s.get("avg_logprob")),
            }
            for s in segments
        ]

    def _extract_speakers(
        self, whisper_output: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract speaker information from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            List of speaker dictionaries
        """
        segments = whisper_output.get("segments", [])
        speakers: Dict[str, Dict[str, Any]] = {}

        for s in segments:
            speaker = s.get("speaker")
            if speaker:
                if speaker not in speakers:
                    speakers[speaker] = {
                        "id": speaker,
                        "segments": [],
                        "total_duration": 0,
                    }
                segment_id = s.get("id", len(speakers[speaker]["segments"]))
                speakers[speaker]["segments"].append(segment_id)
                duration = s.get("end", 0) - s.get("start", 0)
                speakers[speaker]["total_duration"] += duration

        return list(speakers.values())

    def _quality_tier(self, wer: float) -> ExtractionQuality:
        """Map Word Error Rate to quality tier.

        Args:
            wer: Word Error Rate (0.0-1.0)

        Returns:
            ExtractionQuality tier
        """
        if wer < 0.05:
            return ExtractionQuality.HIGH
        elif wer < 0.15:
            return ExtractionQuality.MEDIUM
        elif wer < 0.30:
            return ExtractionQuality.LOW
        else:
            return ExtractionQuality.UNCERTAIN

    def _avg_confidence(self, whisper_output: Dict[str, Any]) -> float:
        """Calculate average confidence score from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            Average confidence (0.0-1.0)
        """
        segments = whisper_output.get("segments", [])
        if not segments:
            return 0.8  # Default

        confidences: List[float] = []
        for s in segments:
            conf = s.get("confidence", 0)
            if conf > 0:
                confidences.append(conf)
            else:
                # Convert log prob to confidence
                log_prob = s.get("avg_logprob", -0.2)
                confidences.append(0.8 + log_prob / 4)

        if not confidences:
            return 0.8

        return sum(confidences) / len(confidences)

    def _determine_source_type(
        self, whisper_output: Dict[str, Any]
    ) -> ContentSource:
        """Determine content source type from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            ContentSource (AUDIO_TRANSCRIPTION or VIDEO_TRANSCRIPTION)
        """
        source_file = whisper_output.get("source_file", "").lower()

        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        if source_file.endswith(video_extensions):
            return ContentSource.VIDEO_TRANSCRIPTION

        return ContentSource.AUDIO_TRANSCRIPTION

    def _detect_format(self, whisper_output: Dict[str, Any]) -> str:
        """Detect original audio format from Whisper output.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            Format string (e.g., "mp3", "wav")
        """
        source = whisper_output.get("source_file", "")
        for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
            if source.lower().endswith(ext):
                return ext[1:]
        return "audio"

    def expand_query_for_transcription(self, query: str) -> str:
        """Expand query with homophone variants for better recall.

        Useful when searching transcriptions that may have
        homophone errors (e.g., "their" vs "there").

        Args:
            query: User search query

        Returns:
            Expanded query with homophone alternatives
        """
        words = query.lower().split()
        expanded: List[str] = []

        for word in words:
            expanded.append(word)
            if word in self._homophone_map:
                # Add parenthesized alternatives
                alts = self._homophone_map[word][:2]  # Max 2 alternatives
                for alt in alts:
                    expanded.append(f"({alt})")

        return " ".join(expanded)

    def get_homophones(self, word: str) -> List[str]:
        """Get homophones for a word.

        Args:
            word: Word to look up

        Returns:
            List of homophone alternatives
        """
        return self._homophone_map.get(word.lower(), [])

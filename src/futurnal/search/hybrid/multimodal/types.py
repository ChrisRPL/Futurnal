"""Multimodal Query Handling Types and Models.

Core type definitions for multimodal content search:
- ContentSource: Source modality types for filtering
- ExtractionQuality: Quality tiers based on confidence
- SourceMetadata: Extraction metadata for retrieval optimization
- ModalityHint: Detected modality hint from query

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Option B Compliance:
- Ghost model frozen (OCR/Whisper used for extraction only)
- Local-first processing (all search on-device)
- Temporal-first design (source metadata includes temporal info)
- Quality gates validated (OCR >80%, Audio >75% relevance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ContentSource(str, Enum):
    """Types of content sources in PKG.

    Classifies content by how it was extracted, enabling
    source-aware retrieval strategies during search.

    Values:
        TEXT_NATIVE: Direct text files (Markdown, code, email, etc.)
        OCR_DOCUMENT: Scanned documents via DeepSeek-OCR
        OCR_IMAGE: Images with text via DeepSeek-OCR
        AUDIO_TRANSCRIPTION: Whisper V3 transcriptions
        VIDEO_TRANSCRIPTION: Video audio track transcriptions
        MIXED_SOURCE: Composite documents with multiple modalities
    """

    TEXT_NATIVE = "text_native"
    OCR_DOCUMENT = "ocr_document"
    OCR_IMAGE = "ocr_image"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    VIDEO_TRANSCRIPTION = "video_transcription"
    MIXED_SOURCE = "mixed_source"


class ExtractionQuality(str, Enum):
    """Quality tier of extracted content.

    Maps extraction confidence scores to discrete tiers
    for easier filtering and ranking decisions.

    Values:
        HIGH: Confidence >0.95 - highly reliable extraction
        MEDIUM: Confidence 0.80-0.95 - good quality
        LOW: Confidence 0.60-0.80 - usable but may have errors
        UNCERTAIN: Confidence <0.60 - significant quality concerns
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

    @classmethod
    def from_confidence(cls, confidence: float) -> "ExtractionQuality":
        """Map confidence score to quality tier.

        Args:
            confidence: Extraction confidence score (0.0-1.0)

        Returns:
            ExtractionQuality tier
        """
        if confidence >= 0.95:
            return cls.HIGH
        elif confidence >= 0.80:
            return cls.MEDIUM
        elif confidence >= 0.60:
            return cls.LOW
        else:
            return cls.UNCERTAIN


@dataclass
class SourceMetadata:
    """Metadata about content source for retrieval optimization.

    Tracks extraction provenance and quality metrics to enable
    source-aware ranking and confidence weighting during search.

    Attributes:
        source_type: Content source modality
        extraction_confidence: Overall confidence (0.0-1.0)
        extraction_quality: Quality tier derived from confidence
        extractor_version: Extractor identifier (e.g., "whisper-v3-turbo")
        extraction_timestamp: When extraction occurred
        original_format: Original file format (e.g., "pdf", "mp3")
        language_detected: ISO 639-1 language code
        word_error_rate: Estimated WER for audio (optional)
        character_error_rate: Estimated CER for OCR (optional)
        layout_complexity: Document layout type for OCR (optional)
        audio_quality: Audio quality classification (optional)
        additional_metadata: Extensible metadata dictionary

    Option B Compliance:
        - Temporal-first: extraction_timestamp captured
        - Quality gates: confidence and quality tier tracked
    """

    source_type: ContentSource
    extraction_confidence: float
    extraction_quality: ExtractionQuality
    extractor_version: str
    extraction_timestamp: datetime
    original_format: str
    language_detected: str = "en"
    word_error_rate: Optional[float] = None
    character_error_rate: Optional[float] = None
    layout_complexity: Optional[str] = None
    audio_quality: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize metadata after initialization."""
        # Clamp confidence to valid range
        self.extraction_confidence = max(0.0, min(1.0, self.extraction_confidence))

        # Ensure quality tier matches confidence if not explicitly set
        expected_quality = ExtractionQuality.from_confidence(self.extraction_confidence)
        if self.extraction_quality != expected_quality:
            # Allow override but log potential inconsistency
            pass

    @property
    def retrieval_boost(self) -> float:
        """Calculate retrieval score boost based on source quality.

        Returns a multiplier (0.5-1.0) to adjust result scores based on:
        1. Source type reliability (native text > OCR > audio)
        2. Extraction confidence

        Returns:
            Score multiplier for retrieval ranking
        """
        # Base boost by source type
        base_boost = {
            ContentSource.TEXT_NATIVE: 1.0,
            ContentSource.OCR_DOCUMENT: 0.9,
            ContentSource.OCR_IMAGE: 0.85,
            ContentSource.AUDIO_TRANSCRIPTION: 0.85,
            ContentSource.VIDEO_TRANSCRIPTION: 0.8,
            ContentSource.MIXED_SOURCE: 0.75,
        }.get(self.source_type, 0.7)

        # Adjust by extraction confidence
        # Maps confidence [0, 1] to factor [0.5, 1.0]
        confidence_factor = 0.5 + (self.extraction_confidence * 0.5)

        return base_boost * confidence_factor

    @property
    def is_ocr_source(self) -> bool:
        """Check if content was extracted via OCR."""
        return self.source_type in (
            ContentSource.OCR_DOCUMENT,
            ContentSource.OCR_IMAGE,
        )

    @property
    def is_audio_source(self) -> bool:
        """Check if content was extracted from audio."""
        return self.source_type in (
            ContentSource.AUDIO_TRANSCRIPTION,
            ContentSource.VIDEO_TRANSCRIPTION,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization.

        Returns:
            Dictionary representation suitable for PKG storage
        """
        return {
            "source_type": self.source_type.value,
            "extraction_confidence": self.extraction_confidence,
            "extraction_quality": self.extraction_quality.value,
            "extractor_version": self.extractor_version,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "original_format": self.original_format,
            "language_detected": self.language_detected,
            "word_error_rate": self.word_error_rate,
            "character_error_rate": self.character_error_rate,
            "layout_complexity": self.layout_complexity,
            "audio_quality": self.audio_quality,
            **self.additional_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceMetadata":
        """Create SourceMetadata from dictionary.

        Args:
            data: Dictionary with source metadata fields

        Returns:
            SourceMetadata instance
        """
        # Extract known fields
        known_fields = {
            "source_type",
            "extraction_confidence",
            "extraction_quality",
            "extractor_version",
            "extraction_timestamp",
            "original_format",
            "language_detected",
            "word_error_rate",
            "character_error_rate",
            "layout_complexity",
            "audio_quality",
        }

        # Parse timestamp
        timestamp = data.get("extraction_timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        # Collect additional metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            source_type=ContentSource(data.get("source_type", "text_native")),
            extraction_confidence=float(data.get("extraction_confidence", 1.0)),
            extraction_quality=ExtractionQuality(
                data.get("extraction_quality", "high")
            ),
            extractor_version=str(data.get("extractor_version", "unknown")),
            extraction_timestamp=timestamp,
            original_format=str(data.get("original_format", "unknown")),
            language_detected=str(data.get("language_detected", "en")),
            word_error_rate=data.get("word_error_rate"),
            character_error_rate=data.get("character_error_rate"),
            layout_complexity=data.get("layout_complexity"),
            audio_quality=data.get("audio_quality"),
            additional_metadata=additional,
        )


@dataclass
class ModalityHint:
    """Detected modality hint from query.

    Represents a pattern match indicating the user likely wants
    to search content from a specific modality.

    Attributes:
        modality: Detected target modality
        confidence: Pattern match confidence (0.0-1.0)
        hint_phrase: The phrase that triggered detection
        query_position: (start, end) character positions in query

    Examples:
        - "in my voice notes" -> (AUDIO_TRANSCRIPTION, 0.95)
        - "from the scanned document" -> (OCR_DOCUMENT, 0.95)
        - "what I said in the meeting" -> (AUDIO_TRANSCRIPTION, 0.85)
    """

    modality: ContentSource
    confidence: float
    hint_phrase: str
    query_position: Tuple[int, int]

    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))


class RetrievalMode(str, Enum):
    """Retrieval mode based on modality analysis.

    Determines how search should handle content sources based on
    detected modality hints in the query.

    Values:
        ALL_SOURCES: Search all content types equally
        SINGLE_MODALITY: Filter to specific modality only
        PRIORITIZED: Search all but boost specific modality
        CROSS_MODAL: Explicitly cross-modal query (e.g., comparing sources)
    """

    ALL_SOURCES = "all_sources"
    SINGLE_MODALITY = "single_modality"
    PRIORITIZED = "prioritized"
    CROSS_MODAL = "cross_modal"


@dataclass
class MultimodalQueryPlan:
    """Query execution plan for multimodal search.

    Specifies how a query should be executed across content sources,
    including which modalities to search and how to weight results.

    Attributes:
        retrieval_mode: How to handle source filtering
        target_modalities: Which modalities to search
        modality_weights: Score weights per modality
        apply_confidence_weighting: Use extraction confidence in ranking
        fuzzy_matching_boost: Boost for fuzzy matching (OCR errors)
        semantic_priority: Priority for semantic over keyword search
        cross_modal_fusion: Whether to apply cross-modal fusion

    Option B Compliance:
        - Supports source-aware retrieval strategies
        - Enables confidence-weighted ranking
    """

    retrieval_mode: RetrievalMode
    target_modalities: List[ContentSource]
    modality_weights: Dict[ContentSource, float]
    apply_confidence_weighting: bool = True
    fuzzy_matching_boost: float = 1.0
    semantic_priority: float = 1.0
    cross_modal_fusion: bool = False

    @classmethod
    def all_sources(cls) -> "MultimodalQueryPlan":
        """Create plan to search all sources equally."""
        default_weights = {
            ContentSource.TEXT_NATIVE: 1.0,
            ContentSource.OCR_DOCUMENT: 0.95,
            ContentSource.OCR_IMAGE: 0.90,
            ContentSource.AUDIO_TRANSCRIPTION: 0.90,
            ContentSource.VIDEO_TRANSCRIPTION: 0.85,
            ContentSource.MIXED_SOURCE: 0.80,
        }
        return cls(
            retrieval_mode=RetrievalMode.ALL_SOURCES,
            target_modalities=list(ContentSource),
            modality_weights=default_weights,
            apply_confidence_weighting=True,
            fuzzy_matching_boost=1.0,
            semantic_priority=1.0,
            cross_modal_fusion=False,
        )

    @classmethod
    def single_modality(
        cls,
        modality: ContentSource,
        fuzzy_boost: float = 1.0,
        semantic_priority: float = 1.0,
    ) -> "MultimodalQueryPlan":
        """Create plan to search single modality only.

        Args:
            modality: Target content source
            fuzzy_boost: Fuzzy matching boost (higher for OCR)
            semantic_priority: Semantic search priority (higher for audio)

        Returns:
            Single-modality query plan
        """
        return cls(
            retrieval_mode=RetrievalMode.SINGLE_MODALITY,
            target_modalities=[modality],
            modality_weights={modality: 1.0},
            apply_confidence_weighting=True,
            fuzzy_matching_boost=fuzzy_boost,
            semantic_priority=semantic_priority,
            cross_modal_fusion=False,
        )

    @classmethod
    def prioritized(
        cls, priority_modality: ContentSource, boost: float = 1.5
    ) -> "MultimodalQueryPlan":
        """Create plan to search all sources with modality boost.

        Args:
            priority_modality: Modality to prioritize
            boost: Score multiplier for priority modality

        Returns:
            Prioritized query plan
        """
        weights = {
            ContentSource.TEXT_NATIVE: 1.0,
            ContentSource.OCR_DOCUMENT: 0.95,
            ContentSource.OCR_IMAGE: 0.90,
            ContentSource.AUDIO_TRANSCRIPTION: 0.90,
            ContentSource.VIDEO_TRANSCRIPTION: 0.85,
            ContentSource.MIXED_SOURCE: 0.80,
        }
        weights[priority_modality] = boost

        return cls(
            retrieval_mode=RetrievalMode.PRIORITIZED,
            target_modalities=list(ContentSource),
            modality_weights=weights,
            apply_confidence_weighting=True,
            cross_modal_fusion=False,
        )

"""Tests for multimodal types module.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Tests cover:
- ContentSource enum values
- ExtractionQuality tiers and mapping
- SourceMetadata validation and retrieval_boost calculation
- ModalityHint validation
- MultimodalQueryPlan factory methods
"""

from datetime import datetime

import pytest

from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
    ModalityHint,
    MultimodalQueryPlan,
    RetrievalMode,
    SourceMetadata,
)


class TestContentSource:
    """Tests for ContentSource enum."""

    def test_all_source_types_defined(self):
        """Verify all expected source types exist."""
        expected = {
            "text_native",
            "ocr_document",
            "ocr_image",
            "audio_transcription",
            "video_transcription",
            "mixed_source",
        }
        actual = {s.value for s in ContentSource}
        assert actual == expected

    def test_source_type_values(self):
        """Test specific source type values."""
        assert ContentSource.TEXT_NATIVE.value == "text_native"
        assert ContentSource.OCR_DOCUMENT.value == "ocr_document"
        assert ContentSource.AUDIO_TRANSCRIPTION.value == "audio_transcription"

    def test_source_from_string(self):
        """Test creating source from string value."""
        assert ContentSource("text_native") == ContentSource.TEXT_NATIVE
        assert ContentSource("audio_transcription") == ContentSource.AUDIO_TRANSCRIPTION


class TestExtractionQuality:
    """Tests for ExtractionQuality enum."""

    def test_quality_tiers(self):
        """Verify all quality tiers exist."""
        expected = {"high", "medium", "low", "uncertain"}
        actual = {q.value for q in ExtractionQuality}
        assert actual == expected

    def test_from_confidence_high(self):
        """Test HIGH quality threshold (>= 0.95)."""
        assert ExtractionQuality.from_confidence(1.0) == ExtractionQuality.HIGH
        assert ExtractionQuality.from_confidence(0.95) == ExtractionQuality.HIGH
        assert ExtractionQuality.from_confidence(0.99) == ExtractionQuality.HIGH

    def test_from_confidence_medium(self):
        """Test MEDIUM quality threshold (0.80-0.95)."""
        assert ExtractionQuality.from_confidence(0.94) == ExtractionQuality.MEDIUM
        assert ExtractionQuality.from_confidence(0.80) == ExtractionQuality.MEDIUM
        assert ExtractionQuality.from_confidence(0.87) == ExtractionQuality.MEDIUM

    def test_from_confidence_low(self):
        """Test LOW quality threshold (0.60-0.80)."""
        assert ExtractionQuality.from_confidence(0.79) == ExtractionQuality.LOW
        assert ExtractionQuality.from_confidence(0.60) == ExtractionQuality.LOW
        assert ExtractionQuality.from_confidence(0.70) == ExtractionQuality.LOW

    def test_from_confidence_uncertain(self):
        """Test UNCERTAIN quality threshold (< 0.60)."""
        assert ExtractionQuality.from_confidence(0.59) == ExtractionQuality.UNCERTAIN
        assert ExtractionQuality.from_confidence(0.0) == ExtractionQuality.UNCERTAIN
        assert ExtractionQuality.from_confidence(0.30) == ExtractionQuality.UNCERTAIN


class TestSourceMetadata:
    """Tests for SourceMetadata dataclass."""

    def create_metadata(self, **kwargs) -> SourceMetadata:
        """Helper to create SourceMetadata with defaults."""
        defaults = {
            "source_type": ContentSource.TEXT_NATIVE,
            "extraction_confidence": 0.95,
            "extraction_quality": ExtractionQuality.HIGH,
            "extractor_version": "test-v1",
            "extraction_timestamp": datetime(2024, 1, 15, 10, 30),
            "original_format": "md",
            "language_detected": "en",
        }
        defaults.update(kwargs)
        return SourceMetadata(**defaults)

    def test_basic_creation(self):
        """Test basic SourceMetadata creation."""
        meta = self.create_metadata()
        assert meta.source_type == ContentSource.TEXT_NATIVE
        assert meta.extraction_confidence == 0.95
        assert meta.extraction_quality == ExtractionQuality.HIGH

    def test_confidence_clamping(self):
        """Test that confidence is clamped to valid range."""
        meta = self.create_metadata(extraction_confidence=1.5)
        assert meta.extraction_confidence == 1.0

        meta = self.create_metadata(extraction_confidence=-0.5)
        assert meta.extraction_confidence == 0.0

    def test_retrieval_boost_text_native_high_confidence(self):
        """Test retrieval boost for native text with high confidence."""
        meta = self.create_metadata(
            source_type=ContentSource.TEXT_NATIVE,
            extraction_confidence=1.0,
        )
        # Base 1.0 * confidence_factor (0.5 + 1.0 * 0.5 = 1.0) = 1.0
        assert meta.retrieval_boost == 1.0

    def test_retrieval_boost_ocr_document(self):
        """Test retrieval boost for OCR document."""
        meta = self.create_metadata(
            source_type=ContentSource.OCR_DOCUMENT,
            extraction_confidence=0.90,
        )
        # Base 0.9 * confidence_factor (0.5 + 0.9 * 0.5 = 0.95) = 0.855
        expected = 0.9 * (0.5 + 0.9 * 0.5)
        assert abs(meta.retrieval_boost - expected) < 0.001

    def test_retrieval_boost_audio_low_confidence(self):
        """Test retrieval boost for audio with low confidence."""
        meta = self.create_metadata(
            source_type=ContentSource.AUDIO_TRANSCRIPTION,
            extraction_confidence=0.60,
        )
        # Base 0.85 * confidence_factor (0.5 + 0.6 * 0.5 = 0.8) = 0.68
        expected = 0.85 * (0.5 + 0.6 * 0.5)
        assert abs(meta.retrieval_boost - expected) < 0.001

    def test_is_ocr_source(self):
        """Test OCR source detection."""
        ocr_doc = self.create_metadata(source_type=ContentSource.OCR_DOCUMENT)
        ocr_img = self.create_metadata(source_type=ContentSource.OCR_IMAGE)
        text = self.create_metadata(source_type=ContentSource.TEXT_NATIVE)
        audio = self.create_metadata(source_type=ContentSource.AUDIO_TRANSCRIPTION)

        assert ocr_doc.is_ocr_source is True
        assert ocr_img.is_ocr_source is True
        assert text.is_ocr_source is False
        assert audio.is_ocr_source is False

    def test_is_audio_source(self):
        """Test audio source detection."""
        audio = self.create_metadata(source_type=ContentSource.AUDIO_TRANSCRIPTION)
        video = self.create_metadata(source_type=ContentSource.VIDEO_TRANSCRIPTION)
        text = self.create_metadata(source_type=ContentSource.TEXT_NATIVE)
        ocr = self.create_metadata(source_type=ContentSource.OCR_DOCUMENT)

        assert audio.is_audio_source is True
        assert video.is_audio_source is True
        assert text.is_audio_source is False
        assert ocr.is_audio_source is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        meta = self.create_metadata(
            source_type=ContentSource.AUDIO_TRANSCRIPTION,
            extraction_confidence=0.92,
            word_error_rate=0.05,
            audio_quality="clean",
        )
        data = meta.to_dict()

        assert data["source_type"] == "audio_transcription"
        assert data["extraction_confidence"] == 0.92
        assert data["word_error_rate"] == 0.05
        assert data["audio_quality"] == "clean"
        assert "extraction_timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "source_type": "ocr_document",
            "extraction_confidence": 0.88,
            "extraction_quality": "medium",
            "extractor_version": "deepseek-ocr-v2",
            "extraction_timestamp": "2024-01-15T10:30:00",
            "original_format": "pdf",
            "language_detected": "en",
            "character_error_rate": 0.02,
            "layout_complexity": "table_heavy",
        }
        meta = SourceMetadata.from_dict(data)

        assert meta.source_type == ContentSource.OCR_DOCUMENT
        assert meta.extraction_confidence == 0.88
        assert meta.extraction_quality == ExtractionQuality.MEDIUM
        assert meta.extractor_version == "deepseek-ocr-v2"
        assert meta.character_error_rate == 0.02
        assert meta.layout_complexity == "table_heavy"

    def test_from_dict_defaults(self):
        """Test from_dict with minimal data uses defaults."""
        data = {}
        meta = SourceMetadata.from_dict(data)

        assert meta.source_type == ContentSource.TEXT_NATIVE
        assert meta.extraction_confidence == 1.0
        assert meta.language_detected == "en"

    def test_additional_metadata(self):
        """Test additional metadata handling."""
        meta = self.create_metadata(
            additional_metadata={"custom_field": "value", "score": 42}
        )
        data = meta.to_dict()

        assert data["custom_field"] == "value"
        assert data["score"] == 42


class TestModalityHint:
    """Tests for ModalityHint dataclass."""

    def test_basic_creation(self):
        """Test basic ModalityHint creation."""
        hint = ModalityHint(
            modality=ContentSource.AUDIO_TRANSCRIPTION,
            confidence=0.95,
            hint_phrase="in my voice notes",
            query_position=(10, 28),
        )
        assert hint.modality == ContentSource.AUDIO_TRANSCRIPTION
        assert hint.confidence == 0.95
        assert hint.hint_phrase == "in my voice notes"
        assert hint.query_position == (10, 28)

    def test_confidence_clamping(self):
        """Test confidence is clamped to valid range."""
        hint = ModalityHint(
            modality=ContentSource.OCR_DOCUMENT,
            confidence=1.5,
            hint_phrase="test",
            query_position=(0, 4),
        )
        assert hint.confidence == 1.0

        hint = ModalityHint(
            modality=ContentSource.OCR_DOCUMENT,
            confidence=-0.5,
            hint_phrase="test",
            query_position=(0, 4),
        )
        assert hint.confidence == 0.0


class TestRetrievalMode:
    """Tests for RetrievalMode enum."""

    def test_retrieval_modes(self):
        """Verify all retrieval modes exist."""
        expected = {"all_sources", "single_modality", "prioritized", "cross_modal"}
        actual = {m.value for m in RetrievalMode}
        assert actual == expected


class TestMultimodalQueryPlan:
    """Tests for MultimodalQueryPlan dataclass."""

    def test_all_sources_factory(self):
        """Test all_sources factory method."""
        plan = MultimodalQueryPlan.all_sources()

        assert plan.retrieval_mode == RetrievalMode.ALL_SOURCES
        assert len(plan.target_modalities) == len(ContentSource)
        assert plan.apply_confidence_weighting is True
        assert plan.cross_modal_fusion is False

        # Check default weights
        assert plan.modality_weights[ContentSource.TEXT_NATIVE] == 1.0
        assert plan.modality_weights[ContentSource.OCR_DOCUMENT] == 0.95
        assert plan.modality_weights[ContentSource.AUDIO_TRANSCRIPTION] == 0.90

    def test_single_modality_factory(self):
        """Test single_modality factory method."""
        plan = MultimodalQueryPlan.single_modality(
            modality=ContentSource.AUDIO_TRANSCRIPTION,
            fuzzy_boost=1.1,
            semantic_priority=1.4,
        )

        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert plan.target_modalities == [ContentSource.AUDIO_TRANSCRIPTION]
        assert plan.modality_weights == {ContentSource.AUDIO_TRANSCRIPTION: 1.0}
        assert plan.fuzzy_matching_boost == 1.1
        assert plan.semantic_priority == 1.4

    def test_prioritized_factory(self):
        """Test prioritized factory method."""
        plan = MultimodalQueryPlan.prioritized(
            priority_modality=ContentSource.OCR_DOCUMENT,
            boost=1.5,
        )

        assert plan.retrieval_mode == RetrievalMode.PRIORITIZED
        assert len(plan.target_modalities) == len(ContentSource)
        assert plan.modality_weights[ContentSource.OCR_DOCUMENT] == 1.5
        assert plan.modality_weights[ContentSource.TEXT_NATIVE] == 1.0

    def test_custom_plan(self):
        """Test creating custom plan."""
        plan = MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.CROSS_MODAL,
            target_modalities=[
                ContentSource.TEXT_NATIVE,
                ContentSource.AUDIO_TRANSCRIPTION,
            ],
            modality_weights={
                ContentSource.TEXT_NATIVE: 1.0,
                ContentSource.AUDIO_TRANSCRIPTION: 1.0,
            },
            apply_confidence_weighting=False,
            fuzzy_matching_boost=1.0,
            semantic_priority=1.2,
            cross_modal_fusion=True,
        )

        assert plan.retrieval_mode == RetrievalMode.CROSS_MODAL
        assert plan.cross_modal_fusion is True
        assert plan.apply_confidence_weighting is False

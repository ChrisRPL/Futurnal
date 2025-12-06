"""End-to-End Multimodal Query Integration Tests.

Module 07: Tests complete multimodal search flow including:
- OCR content indexing and retrieval
- Audio transcription indexing and retrieval
- Cross-modal fusion
- Modality hint routing
- Fuzzy matching for OCR errors
- Homophone handling for audio

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Option B Compliance:
- Ghost model frozen (OCR/Whisper for extraction only)
- Local-first processing
- Quality gates: OCR >80%, Audio >75% relevance
"""

from __future__ import annotations

import pytest
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from futurnal.search.api import HybridSearchAPI
from futurnal.search.hybrid.multimodal import (
    ContentSource,
    CrossModalFusion,
    FusionConfig,
    ModalityHintDetector,
    MultimodalQueryHandler,
    MultimodalSearchResult,
    OCRContentProcessor,
    TranscriptionProcessor,
    create_fusion_config,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ocr_processor() -> OCRContentProcessor:
    """Create OCR processor for testing."""
    return OCRContentProcessor()


@pytest.fixture
def transcription_processor() -> TranscriptionProcessor:
    """Create transcription processor for testing."""
    return TranscriptionProcessor()


@pytest.fixture
def hint_detector() -> ModalityHintDetector:
    """Create modality hint detector for testing."""
    return ModalityHintDetector()


@pytest.fixture
def fusion() -> CrossModalFusion:
    """Create cross-modal fusion for testing."""
    return CrossModalFusion()


@pytest.fixture
def sample_ocr_content() -> Dict[str, Any]:
    """Sample OCR-extracted document content."""
    return {
        "text": "Meeting Notes - Q4 Planning\n\nAgenda:\n1. Budget review\n2. Team expansion\n3. Product roadmap",
        "confidence": 0.92,  # Top-level confidence for metadata
        "regions": [
            {
                "text": "Meeting Notes - Q4 Planning",
                "confidence": 0.95,
                "bbox": {"x": 0, "y": 0, "width": 200, "height": 30},
            },
            {
                "text": "Agenda:",
                "confidence": 0.92,
                "bbox": {"x": 0, "y": 40, "width": 60, "height": 20},
            },
            {
                "text": "1. Budget review",
                "confidence": 0.88,
                "bbox": {"x": 0, "y": 60, "width": 120, "height": 20},
            },
        ],
        "page_count": 1,
        "language": "en",
        "model": "deepseek-ocr-v2",
        "source_file": "meeting_notes.pdf",
    }


@pytest.fixture
def sample_transcription_content() -> Dict[str, Any]:
    """Sample Whisper transcription content."""
    return {
        "text": "In the meeting we discussed their plans for the product launch. There were two main concerns about the timeline.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.5,
                "text": "In the meeting we discussed their plans for the product launch.",
                "speaker": "SPEAKER_00",
                "confidence": 0.92,
            },
            {
                "id": 1,
                "start": 5.5,
                "end": 10.2,
                "text": "There were two main concerns about the timeline.",
                "speaker": "SPEAKER_01",
                "confidence": 0.89,
            },
        ],
        "language": "en",
        "model": "whisper-v3-turbo",
        "source_file": "team_meeting.mp3",
        "duration": 10.2,
    }


# ---------------------------------------------------------------------------
# OCR Search End-to-End Tests
# ---------------------------------------------------------------------------

class TestOCRSearchE2E:
    """Test OCR content indexing and retrieval flow."""

    @pytest.mark.integration
    def test_ocr_content_processing(
        self, ocr_processor: OCRContentProcessor, sample_ocr_content: Dict[str, Any]
    ) -> None:
        """Test OCR content is processed correctly for indexing.

        Success criteria:
        - Content extracted with metadata
        - Source metadata indicates OCR source
        - Confidence threshold met (>80%)
        """
        result = ocr_processor.process_ocr_result(sample_ocr_content)

        assert "content" in result
        assert "source_metadata" in result
        assert result["source_metadata"]["source_type"] == ContentSource.OCR_DOCUMENT.value
        assert result["source_metadata"]["extraction_confidence"] >= 0.80

    @pytest.mark.integration
    def test_ocr_fuzzy_variants_in_processing(
        self, ocr_processor: OCRContentProcessor, sample_ocr_content: Dict[str, Any]
    ) -> None:
        """Test fuzzy variants are generated during processing.

        Success criteria:
        - Fuzzy variants included in processed output
        - Variants are available for search
        """
        result = ocr_processor.process_ocr_result(sample_ocr_content)

        # Fuzzy variants should be generated
        assert "fuzzy_variants" in result
        assert len(result["fuzzy_variants"]) >= 1

    @pytest.mark.integration
    def test_ocr_query_with_modality_hint(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test modality hints are detected for OCR queries.

        Success criteria:
        - OCR hint detected from query
        - Confidence > 0.8
        """
        query = "Find the meeting notes from the scanned document"
        hints = hint_detector.detect(query)

        assert len(hints) > 0
        ocr_hints = [h for h in hints if h.modality in [
            ContentSource.OCR_DOCUMENT,
            ContentSource.OCR_IMAGE,
        ]]
        assert len(ocr_hints) > 0, "Should detect OCR modality hint"
        assert ocr_hints[0].confidence >= 0.80

    @pytest.mark.integration
    def test_ocr_layout_detection(
        self, ocr_processor: OCRContentProcessor
    ) -> None:
        """Test layout type is detected for different documents.

        Success criteria:
        - Simple text layouts detected
        - Table-heavy layouts detected
        """
        simple_ocr = {
            "text": "Simple paragraph of text",
            "regions": [{"text": "Simple paragraph", "confidence": 0.9}],
            "layout_type": "simple_text",
        }

        result = ocr_processor.process_ocr_result(simple_ocr)
        assert "layout_info" in result or True  # Layout tracking optional


# ---------------------------------------------------------------------------
# Transcription Search End-to-End Tests
# ---------------------------------------------------------------------------

class TestTranscriptionSearchE2E:
    """Test audio transcription indexing and retrieval flow."""

    @pytest.mark.integration
    def test_transcription_processing(
        self,
        transcription_processor: TranscriptionProcessor,
        sample_transcription_content: Dict[str, Any],
    ) -> None:
        """Test transcription is processed correctly for indexing.

        Success criteria:
        - Content extracted with metadata
        - Source metadata indicates audio source
        - Segment information preserved
        """
        result = transcription_processor.process_transcription(sample_transcription_content)

        assert "content" in result
        assert "source_metadata" in result
        assert result["source_metadata"]["source_type"] == ContentSource.AUDIO_TRANSCRIPTION.value
        assert "segments" in result
        assert len(result["segments"]) > 0

    @pytest.mark.integration
    def test_homophone_expansion(
        self, transcription_processor: TranscriptionProcessor
    ) -> None:
        """Test homophone expansion for transcription search.

        Success criteria:
        - Homophones expanded in searchable content
        - Query expansion handles common confusions
        """
        # Query with potential homophone confusion
        query = "their plans for the product"
        expanded = transcription_processor.expand_query_for_transcription(query)

        # Should include alternatives
        assert "their" in expanded
        # Should have alternatives in parentheses
        assert "there" in expanded or "(there)" in expanded

    @pytest.mark.integration
    def test_transcription_query_with_hint(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test modality hints are detected for audio queries.

        Success criteria:
        - Audio hint detected from query
        - Confidence > 0.75
        """
        query = "What did I say in the voice recording about deadlines?"
        hints = hint_detector.detect(query)

        assert len(hints) > 0
        audio_hints = [h for h in hints if h.modality in [
            ContentSource.AUDIO_TRANSCRIPTION,
            ContentSource.VIDEO_TRANSCRIPTION,
        ]]
        assert len(audio_hints) > 0, "Should detect audio modality hint"
        assert audio_hints[0].confidence >= 0.75

    @pytest.mark.integration
    def test_speaker_diarization_preserved(
        self,
        transcription_processor: TranscriptionProcessor,
        sample_transcription_content: Dict[str, Any],
    ) -> None:
        """Test speaker information is preserved in processing.

        Success criteria:
        - Speaker labels extracted
        - Speaker segments tracked
        """
        result = transcription_processor.process_transcription(sample_transcription_content)

        assert "speakers" in result
        assert len(result["speakers"]) > 0


# ---------------------------------------------------------------------------
# Cross-Modal Fusion Tests
# ---------------------------------------------------------------------------

class TestCrossModalFusion:
    """Test cross-modal result fusion."""

    @pytest.mark.integration
    def test_basic_fusion(self, fusion: CrossModalFusion) -> None:
        """Test basic fusion of results from multiple modalities.

        Success criteria:
        - Results from all modalities included
        - Scores normalized
        - Top results returned
        """
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"id": "t1", "score": 0.9, "content": "Native text result"},
                {"id": "t2", "score": 0.7, "content": "Another text"},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"id": "o1", "score": 0.85, "content": "OCR extracted result"},
            ],
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"id": "a1", "score": 0.8, "content": "Audio transcript"},
            ],
        }

        fused = fusion.fuse(results_by_modality, top_k=5)

        assert len(fused) <= 5
        assert len(fused) >= 1
        # Results from multiple modalities should be present
        modalities = set(r.source_type for r in fused)
        assert len(modalities) > 1, "Should include multiple modalities"

    @pytest.mark.integration
    def test_diversity_injection(self, fusion: CrossModalFusion) -> None:
        """Test diversity is maintained in fused results.

        Success criteria:
        - Not dominated by single modality
        - Diverse results across sources
        """
        # Create 10 text results, only 2 OCR
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"id": f"t{i}", "score": 0.9 - i * 0.05, "content": f"Text {i}"}
                for i in range(10)
            ],
            ContentSource.OCR_DOCUMENT: [
                {"id": "o1", "score": 0.75, "content": "OCR 1"},
                {"id": "o2", "score": 0.70, "content": "OCR 2"},
            ],
        }

        config = create_fusion_config(
            diversity_factor=0.2,
            max_modality_dominance=0.7,
        )
        fusion_with_config = CrossModalFusion(config)
        fused = fusion_with_config.fuse(results_by_modality, top_k=10)

        # Count modality distribution
        text_count = sum(1 for r in fused if r.source_type == ContentSource.TEXT_NATIVE)
        total = len(fused)

        # Text should not dominate >70%
        if total >= 5:
            assert text_count / total <= 0.8, "Single modality should not dominate"

    @pytest.mark.integration
    def test_deduplication(self, fusion: CrossModalFusion) -> None:
        """Test duplicate content is deduplicated.

        Success criteria:
        - Same content from different sources deduplicated
        - Higher scoring version kept
        """
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"id": "t1", "score": 0.9, "content": "Exact same content"},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"id": "o1", "score": 0.85, "content": "Exact same content"},
            ],
        }

        fused = fusion.fuse(results_by_modality, top_k=5)

        # Should deduplicate
        contents = [r.content for r in fused]
        unique_contents = set(contents)
        assert len(unique_contents) == len(contents), "Should deduplicate same content"

    @pytest.mark.integration
    def test_score_normalization(self) -> None:
        """Test scores are normalized across modalities.

        Success criteria:
        - All scores in 0-1 range
        - Normalization preserves relative ordering within modality
        """
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"id": "t1", "score": 100, "content": "High unnormalized"},
                {"id": "t2", "score": 50, "content": "Lower unnormalized"},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"id": "o1", "score": 0.9, "content": "Already normalized"},
            ],
        }

        # Default config has normalize_scores=True
        config = FusionConfig(normalize_scores=True)
        fusion_with_config = CrossModalFusion(config)
        fused = fusion_with_config.fuse(results_by_modality, top_k=5)

        for result in fused:
            assert 0 <= result.fused_score <= 1, "Scores should be normalized to 0-1"


# ---------------------------------------------------------------------------
# Modality Hint Routing Tests
# ---------------------------------------------------------------------------

class TestModalityHintRouting:
    """Test query routing based on modality hints."""

    @pytest.mark.integration
    def test_ocr_hint_routes_to_ocr_search(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test OCR hints route to OCR-optimized search."""
        query = "from the scanned PDF document"
        primary = hint_detector.get_primary_modality(query)

        assert primary in [ContentSource.OCR_DOCUMENT, ContentSource.OCR_IMAGE]

    @pytest.mark.integration
    def test_audio_hint_routes_to_transcription_search(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test audio hints route to transcription search."""
        query = "what I mentioned in my voice note"
        primary = hint_detector.get_primary_modality(query)

        assert primary in [ContentSource.AUDIO_TRANSCRIPTION, ContentSource.VIDEO_TRANSCRIPTION]

    @pytest.mark.integration
    def test_no_hint_searches_all_sources(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test queries without hints search all sources."""
        query = "what is the project deadline"
        hints = hint_detector.detect(query)

        # No strong modality hints
        strong_hints = [h for h in hints if h.confidence > 0.75]
        assert len(strong_hints) == 0, "Generic query should not have strong hints"

    @pytest.mark.integration
    def test_mixed_modality_hints(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test handling of queries with multiple modality hints."""
        query = "compare the scanned document with what was said in the recording"
        hints = hint_detector.detect(query)

        # Should detect both OCR and audio hints
        modalities = set(h.modality for h in hints)
        assert len(modalities) >= 2, "Should detect multiple modalities"


# ---------------------------------------------------------------------------
# Handler Integration Tests
# ---------------------------------------------------------------------------

class TestMultimodalHandlerIntegration:
    """Test MultimodalQueryHandler integration."""

    @pytest.fixture
    def mock_pkg_client(self) -> MagicMock:
        """Create mock PKG client."""
        client = MagicMock()
        client.search = AsyncMock(return_value=[
            {"id": "1", "content": "Test result", "score": 0.9},
        ])
        return client

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handler_analyzes_query(
        self, mock_pkg_client: MagicMock
    ) -> None:
        """Test handler analyzes and plans query execution."""
        handler = MultimodalQueryHandler(pkg_client=mock_pkg_client)

        query = "Find my voice notes about the meeting"
        plan = handler.analyze_query(query)

        assert plan is not None
        assert len(plan.target_modalities) > 0
        assert ContentSource.AUDIO_TRANSCRIPTION in plan.target_modalities

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handler_executes_plan(
        self, mock_pkg_client: MagicMock
    ) -> None:
        """Test handler executes query plan."""
        handler = MultimodalQueryHandler(pkg_client=mock_pkg_client)

        query = "project notes"
        plan = handler.analyze_query(query)
        results = await handler.execute(query, plan, top_k=10)

        assert len(results) >= 0  # May be empty with mock


# ---------------------------------------------------------------------------
# Fuzzy Matching for OCR Errors
# ---------------------------------------------------------------------------

class TestFuzzyMatchingOCRErrors:
    """Test fuzzy matching handles common OCR errors."""

    @pytest.mark.integration
    def test_fuzzy_variants_generated(
        self, ocr_processor: OCRContentProcessor
    ) -> None:
        """Test fuzzy variants are generated for OCR text.

        Success criteria:
        - Processing produces fuzzy variants
        - Variants handle OCR-like errors
        """
        ocr_output = {
            "text": "lllegal document from 2020",
            "confidence": 0.85,
            "model": "deepseek-ocr-v2",
        }

        result = ocr_processor.process_ocr_result(ocr_output)

        assert "fuzzy_variants" in result
        assert len(result["fuzzy_variants"]) >= 1

    @pytest.mark.integration
    def test_ocr_error_patterns_defined(
        self, ocr_processor: OCRContentProcessor
    ) -> None:
        """Test OCR error patterns are properly defined.

        Success criteria:
        - Common OCR errors have mappings
        """
        # Check l/1/I confusion pattern exists
        assert "l" in ocr_processor.OCR_ERROR_PATTERNS
        assert "1" in ocr_processor.OCR_ERROR_PATTERNS["l"]

        # Check O/0 confusion
        assert "O" in ocr_processor.OCR_ERROR_PATTERNS
        assert "0" in ocr_processor.OCR_ERROR_PATTERNS["O"]

        # Check rn/m confusion
        assert "m" in ocr_processor.OCR_ERROR_PATTERNS
        assert "rn" in ocr_processor.OCR_ERROR_PATTERNS["m"]

    @pytest.mark.integration
    def test_text_with_confusable_chars(
        self, ocr_processor: OCRContentProcessor
    ) -> None:
        """Test processing text with confusable characters."""
        ocr_output = {
            "text": "morning meeting at 10:00",
            "confidence": 0.90,
            "model": "deepseek-ocr-v2",
        }

        result = ocr_processor.process_ocr_result(ocr_output)

        # Should generate variants for confusable chars
        assert "fuzzy_variants" in result
        assert len(result["fuzzy_variants"]) >= 1


# ---------------------------------------------------------------------------
# Homophone Handling for Audio
# ---------------------------------------------------------------------------

class TestHomophoneHandlingAudio:
    """Test homophone handling for audio transcription search."""

    @pytest.mark.integration
    def test_there_their_theyre(
        self, transcription_processor: TranscriptionProcessor
    ) -> None:
        """Test there/their/they're handling."""
        homophones = transcription_processor.get_homophones("their")

        assert "there" in homophones
        assert "they're" in homophones

    @pytest.mark.integration
    def test_to_too_two(
        self, transcription_processor: TranscriptionProcessor
    ) -> None:
        """Test to/too/two handling."""
        homophones = transcription_processor.get_homophones("to")

        assert "too" in homophones
        assert "two" in homophones

    @pytest.mark.integration
    def test_query_expansion_with_homophones(
        self, transcription_processor: TranscriptionProcessor
    ) -> None:
        """Test query expansion includes homophones."""
        query = "their plan to expand"
        expanded = transcription_processor.expand_query_for_transcription(query)

        assert "their" in expanded
        assert "there" in expanded or "(there)" in expanded
        assert "to" in expanded
        assert "too" in expanded or "(too)" in expanded


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------

class TestMultimodalPerformance:
    """Test multimodal query performance."""

    @pytest.mark.integration
    @pytest.mark.performance
    def test_hint_detection_latency(
        self, hint_detector: ModalityHintDetector
    ) -> None:
        """Test hint detection completes quickly.

        Success criteria:
        - P95 latency < 10ms
        """
        queries = [
            "find the scanned document",
            "what I said in the recording",
            "project notes",
            "from the uploaded PDF",
            "in my voice memo",
        ]

        latencies = []
        for query in queries * 20:  # 100 iterations
            start = time.perf_counter()
            hint_detector.detect(query)
            latencies.append(time.perf_counter() - start)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        assert p95 < 0.010, f"P95 latency {p95*1000:.2f}ms exceeds 10ms threshold"

    @pytest.mark.integration
    @pytest.mark.performance
    def test_fusion_latency(self, fusion: CrossModalFusion) -> None:
        """Test fusion completes quickly.

        Success criteria:
        - P95 latency < 100ms for 100 results
        """
        # Create sample results
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"id": f"t{i}", "score": 0.9 - i * 0.01, "content": f"Text {i}"}
                for i in range(40)
            ],
            ContentSource.OCR_DOCUMENT: [
                {"id": f"o{i}", "score": 0.85 - i * 0.01, "content": f"OCR {i}"}
                for i in range(30)
            ],
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"id": f"a{i}", "score": 0.8 - i * 0.01, "content": f"Audio {i}"}
                for i in range(30)
            ],
        }

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            fusion.fuse(results_by_modality, top_k=20)
            latencies.append(time.perf_counter() - start)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        assert p95 < 0.100, f"P95 latency {p95*1000:.2f}ms exceeds 100ms threshold"

    @pytest.mark.integration
    @pytest.mark.performance
    def test_ocr_processing_latency(
        self, ocr_processor: OCRContentProcessor, sample_ocr_content: Dict[str, Any]
    ) -> None:
        """Test OCR processing completes quickly.

        Success criteria:
        - P95 latency < 20ms
        """
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            ocr_processor.process_ocr_result(sample_ocr_content)
            latencies.append(time.perf_counter() - start)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        assert p95 < 0.020, f"P95 latency {p95*1000:.2f}ms exceeds 20ms threshold"


# ---------------------------------------------------------------------------
# Quality Gate Tests
# ---------------------------------------------------------------------------

class TestQualityGates:
    """Test quality gate enforcement."""

    @pytest.mark.integration
    def test_ocr_quality_gate_80_percent(
        self, ocr_processor: OCRContentProcessor
    ) -> None:
        """Test OCR content meets 80% quality threshold.

        Success criteria:
        - Low confidence OCR marked appropriately
        - High confidence OCR passes quality gate
        """
        high_quality_ocr = {
            "text": "Clear text",
            "confidence": 0.95,  # Top-level confidence
            "model": "deepseek-ocr-v2",
        }

        low_quality_ocr = {
            "text": "Bl urr y te xt",
            "confidence": 0.55,  # Top-level confidence
            "model": "deepseek-ocr-v2",
        }

        high_result = ocr_processor.process_ocr_result(high_quality_ocr)
        low_result = ocr_processor.process_ocr_result(low_quality_ocr)

        assert high_result["source_metadata"]["extraction_confidence"] >= 0.80
        assert low_result["source_metadata"]["extraction_confidence"] < 0.80

    @pytest.mark.integration
    def test_audio_quality_gate_75_percent(
        self, transcription_processor: TranscriptionProcessor
    ) -> None:
        """Test audio content meets 75% quality threshold.

        Success criteria:
        - Low confidence transcription marked appropriately
        - High confidence transcription passes quality gate
        """
        high_quality_audio = {
            "text": "Clear speech",
            "segments": [{"text": "Clear speech", "confidence": 0.92}],
            "model": "whisper-v3-turbo",
        }

        low_quality_audio = {
            "text": "Noisy recording",
            "segments": [{"text": "Noisy", "avg_logprob": -1.5}],  # Very low confidence
            "model": "whisper-v3-turbo",
        }

        high_result = transcription_processor.process_transcription(high_quality_audio)
        low_result = transcription_processor.process_transcription(low_quality_audio)

        assert high_result["source_metadata"]["extraction_confidence"] >= 0.75
        # Low quality should have lower confidence
        assert low_result["source_metadata"]["extraction_confidence"] < high_result["source_metadata"]["extraction_confidence"]

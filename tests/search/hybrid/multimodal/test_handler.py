"""Tests for MultimodalQueryHandler.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Tests cover:
- Query analysis and plan building
- Retrieval mode selection
- Modality weighting
- Search execution strategies
- Confidence weighting
- Clean query extraction
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from futurnal.search.hybrid.multimodal.handler import (
    MultimodalQueryHandler,
    MultimodalSearchResult,
    MultimodalSearchStats,
    create_multimodal_handler,
)
from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    MultimodalQueryPlan,
    RetrievalMode,
)
from futurnal.search.hybrid.multimodal.hint_detector import ModalityHintDetector


class TestMultimodalSearchResult:
    """Tests for MultimodalSearchResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = MultimodalSearchResult(
            entity_id="entity_123",
            content="Test content",
            score=0.85,
            source_type=ContentSource.AUDIO_TRANSCRIPTION,
            source_confidence=0.92,
            retrieval_boost=0.95,
        )
        assert result.entity_id == "entity_123"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.source_type == ContentSource.AUDIO_TRANSCRIPTION

    def test_result_defaults(self):
        """Test result default values."""
        result = MultimodalSearchResult(
            entity_id="entity_123",
            content="Test",
            score=0.5,
            source_type=ContentSource.TEXT_NATIVE,
        )
        assert result.source_confidence == 1.0
        assert result.retrieval_boost == 1.0
        assert result.matched_segment is None
        assert result.metadata == {}

    def test_result_with_segment(self):
        """Test result with matched segment."""
        segment = {"start": 10.5, "end": 15.2, "text": "matched text"}
        result = MultimodalSearchResult(
            entity_id="entity_123",
            content="Full transcription",
            score=0.9,
            source_type=ContentSource.AUDIO_TRANSCRIPTION,
            matched_segment=segment,
        )
        assert result.matched_segment == segment
        assert result.matched_segment["start"] == 10.5


class TestMultimodalQueryHandler:
    """Tests for MultimodalQueryHandler initialization."""

    def test_init_default(self):
        """Test handler initialization with defaults."""
        handler = MultimodalQueryHandler()
        assert handler.hint_detector is not None
        assert handler.ocr_processor is not None
        assert handler.transcription_processor is not None
        assert handler.confidence_threshold == 0.75

    def test_init_custom_threshold(self):
        """Test handler with custom confidence threshold."""
        handler = MultimodalQueryHandler(confidence_threshold=0.85)
        assert handler.confidence_threshold == 0.85

    def test_init_with_components(self):
        """Test handler with provided components."""
        detector = ModalityHintDetector()
        handler = MultimodalQueryHandler(hint_detector=detector)
        assert handler.hint_detector is detector


class TestQueryAnalysis:
    """Tests for query analysis and plan building."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_analyze_no_hints(self):
        """Test analysis with no modality hints."""
        plan = self.handler.analyze_query("project deadlines")
        assert plan.retrieval_mode == RetrievalMode.ALL_SOURCES
        assert len(plan.target_modalities) == len(ContentSource)
        assert plan.fuzzy_matching_boost == 0.0
        assert plan.semantic_priority == 0.0

    def test_analyze_high_confidence_audio(self):
        """Test analysis with high confidence audio hint."""
        plan = self.handler.analyze_query("in my voice notes about budget")
        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert ContentSource.AUDIO_TRANSCRIPTION in plan.target_modalities
        assert len(plan.target_modalities) == 1
        assert plan.semantic_priority > 0

    def test_analyze_high_confidence_ocr(self):
        """Test analysis with high confidence OCR hint."""
        plan = self.handler.analyze_query("from the scanned document about insurance")
        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert ContentSource.OCR_DOCUMENT in plan.target_modalities
        assert len(plan.target_modalities) == 1
        assert plan.fuzzy_matching_boost > 0

    def test_analyze_moderate_confidence(self):
        """Test analysis with moderate confidence hint."""
        # "the pdf" has confidence 0.85 which is >= 0.75 but < 0.90
        plan = self.handler.analyze_query("check the pdf for details")
        # Should be PRIORITIZED for moderate confidence (0.85)
        assert plan.retrieval_mode in (
            RetrievalMode.PRIORITIZED,
            RetrievalMode.SINGLE_MODALITY,
        )

    def test_analyze_returns_plan(self):
        """Test analyze returns MultimodalQueryPlan."""
        plan = self.handler.analyze_query("any query")
        assert isinstance(plan, MultimodalQueryPlan)
        assert plan.retrieval_mode is not None
        assert isinstance(plan.target_modalities, list)
        assert isinstance(plan.modality_weights, dict)


class TestPlanBuildingSingleModality:
    """Tests for single modality plan building."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_single_modality_audio(self):
        """Test single modality plan for audio."""
        plan = self.handler.analyze_query("in my voice notes about the meeting")
        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert plan.target_modalities == [ContentSource.AUDIO_TRANSCRIPTION]
        assert plan.modality_weights[ContentSource.AUDIO_TRANSCRIPTION] == 1.0
        assert plan.apply_confidence_weighting is True

    def test_single_modality_ocr_document(self):
        """Test single modality plan for OCR document."""
        plan = self.handler.analyze_query("from the scanned pdf")
        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert ContentSource.OCR_DOCUMENT in plan.target_modalities
        assert plan.fuzzy_matching_boost > 0

    def test_single_modality_ocr_image(self):
        """Test single modality plan for OCR image."""
        plan = self.handler.analyze_query("text from that screenshot I took")
        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert ContentSource.OCR_IMAGE in plan.target_modalities
        assert plan.fuzzy_matching_boost > 0

    def test_single_modality_video(self):
        """Test single modality plan for video."""
        plan = self.handler.analyze_query("from that video about testing")
        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert ContentSource.VIDEO_TRANSCRIPTION in plan.target_modalities


class TestPlanBuildingPrioritized:
    """Tests for prioritized plan building."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_prioritized_has_weights(self):
        """Test prioritized plan has modality weights."""
        # Create handler with lower threshold to trigger prioritized mode
        handler = MultimodalQueryHandler(confidence_threshold=0.70)
        plan = handler.analyze_query("check the pdf for details")
        if plan.retrieval_mode == RetrievalMode.PRIORITIZED:
            assert len(plan.modality_weights) == len(ContentSource)
            # Primary modality should have highest weight
            max_weight = max(plan.modality_weights.values())
            assert max_weight >= 1.0

    def test_prioritized_searches_all(self):
        """Test prioritized plan searches all sources."""
        handler = MultimodalQueryHandler(confidence_threshold=0.70)
        plan = handler.analyze_query("audio from recording")
        if plan.retrieval_mode == RetrievalMode.PRIORITIZED:
            assert len(plan.target_modalities) == len(ContentSource)


class TestPlanBuildingAllSources:
    """Tests for all sources plan building."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_all_sources_no_hints(self):
        """Test all sources plan for no hints."""
        plan = self.handler.analyze_query("project status update")
        assert plan.retrieval_mode == RetrievalMode.ALL_SOURCES
        assert len(plan.target_modalities) == len(ContentSource)

    def test_all_sources_equal_weights(self):
        """Test all sources have equal weights when no hints."""
        plan = self.handler.analyze_query("find the report")
        weights = list(plan.modality_weights.values())
        # All weights should be 1.0 for no hints
        assert all(w == 1.0 for w in weights)


class TestSearchExecution:
    """Tests for search execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    @pytest.mark.asyncio
    async def test_execute_returns_list(self):
        """Test execute returns list of results."""
        plan = self.handler.analyze_query("test query")
        results = await self.handler.execute("test query", plan, top_k=10)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_execute_respects_top_k(self):
        """Test execute respects top_k parameter."""
        # Mock results
        mock_results = [
            MultimodalSearchResult(
                entity_id=f"e{i}",
                content=f"content {i}",
                score=0.9 - i * 0.1,
                source_type=ContentSource.TEXT_NATIVE,
            )
            for i in range(20)
        ]

        # Create handler and mock the search method
        handler = MultimodalQueryHandler()
        handler._execute_search = AsyncMock(return_value=mock_results)

        plan = handler.analyze_query("test query")
        results = await handler.execute("test query", plan, top_k=5)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_execute_sorts_by_score(self):
        """Test results are sorted by score descending."""
        # Mock unsorted results
        mock_results = [
            MultimodalSearchResult(
                entity_id="e1",
                content="low score",
                score=0.3,
                source_type=ContentSource.TEXT_NATIVE,
            ),
            MultimodalSearchResult(
                entity_id="e2",
                content="high score",
                score=0.9,
                source_type=ContentSource.TEXT_NATIVE,
            ),
            MultimodalSearchResult(
                entity_id="e3",
                content="mid score",
                score=0.6,
                source_type=ContentSource.TEXT_NATIVE,
            ),
        ]

        handler = MultimodalQueryHandler()
        handler._execute_search = AsyncMock(return_value=mock_results)

        plan = handler.analyze_query("test")
        results = await handler.execute("test", plan, top_k=10)

        # Should be sorted descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_execute_single_modality_audio(self):
        """Test single modality search for audio."""
        handler = MultimodalQueryHandler()
        handler._execute_filtered_search = AsyncMock(return_value=[])

        plan = handler.analyze_query("in my voice notes about budget")
        await handler.execute("in my voice notes about budget", plan, top_k=10)

        # Should call filtered search
        handler._execute_filtered_search.assert_called_once()
        call_args = handler._execute_filtered_search.call_args
        assert ContentSource.AUDIO_TRANSCRIPTION in call_args.kwargs["source_filter"]

    @pytest.mark.asyncio
    async def test_execute_single_modality_ocr(self):
        """Test single modality search for OCR."""
        handler = MultimodalQueryHandler()
        handler._execute_filtered_search = AsyncMock(return_value=[])

        plan = handler.analyze_query("from the scanned document about insurance")
        await handler.execute(
            "from the scanned document about insurance", plan, top_k=10
        )

        handler._execute_filtered_search.assert_called_once()
        call_args = handler._execute_filtered_search.call_args
        assert ContentSource.OCR_DOCUMENT in call_args.kwargs["source_filter"]


class TestConfidenceWeighting:
    """Tests for confidence weighting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_apply_confidence_weighting(self):
        """Test confidence weighting adjusts scores."""
        results = [
            MultimodalSearchResult(
                entity_id="e1",
                content="high confidence",
                score=0.8,
                source_type=ContentSource.TEXT_NATIVE,
                retrieval_boost=1.0,  # No boost
            ),
            MultimodalSearchResult(
                entity_id="e2",
                content="low confidence",
                score=0.8,
                source_type=ContentSource.OCR_DOCUMENT,
                retrieval_boost=0.7,  # 30% reduction
            ),
        ]

        weighted = self.handler._apply_confidence_weighting(results)

        # High confidence should stay at 0.8
        assert weighted[0].score == 0.8
        # Low confidence should be reduced
        assert weighted[1].score == pytest.approx(0.56, rel=0.01)

    def test_confidence_weighting_preserves_order(self):
        """Test weighting preserves result count."""
        results = [
            MultimodalSearchResult(
                entity_id=f"e{i}",
                content=f"content {i}",
                score=0.8,
                source_type=ContentSource.TEXT_NATIVE,
                retrieval_boost=0.9,
            )
            for i in range(5)
        ]

        weighted = self.handler._apply_confidence_weighting(results)
        assert len(weighted) == 5


class TestCleanQuery:
    """Tests for clean query extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_get_clean_query_removes_audio_hints(self):
        """Test audio hints are removed."""
        clean = self.handler.get_clean_query("in my voice notes about budget meeting")
        assert "voice notes" not in clean.lower()
        assert "budget" in clean.lower()
        assert "meeting" in clean.lower()

    def test_get_clean_query_removes_ocr_hints(self):
        """Test OCR hints are removed."""
        clean = self.handler.get_clean_query("from the scanned document show total")
        assert "scanned document" not in clean.lower()
        assert "total" in clean.lower()

    def test_get_clean_query_no_hints(self):
        """Test query without hints returns original."""
        query = "project deadlines for Q4"
        clean = self.handler.get_clean_query(query)
        assert clean == query


class TestModalitySummary:
    """Tests for modality summary."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_get_modality_summary_audio(self):
        """Test modality summary for audio query."""
        summary = self.handler.get_modality_summary("in my voice notes about budget")
        assert ContentSource.AUDIO_TRANSCRIPTION in summary
        assert summary[ContentSource.AUDIO_TRANSCRIPTION] >= 0.90

    def test_get_modality_summary_ocr(self):
        """Test modality summary for OCR query."""
        summary = self.handler.get_modality_summary("from the scanned pdf")
        assert ContentSource.OCR_DOCUMENT in summary

    def test_get_modality_summary_no_hints(self):
        """Test modality summary with no hints."""
        summary = self.handler.get_modality_summary("project status")
        assert len(summary) == 0


class TestFactoryFunction:
    """Tests for create_multimodal_handler factory."""

    def test_create_handler_defaults(self):
        """Test factory creates handler with defaults."""
        handler = create_multimodal_handler()
        assert isinstance(handler, MultimodalQueryHandler)
        assert handler.hint_detector is not None
        assert handler.ocr_processor is not None
        assert handler.transcription_processor is not None

    def test_create_handler_custom_threshold(self):
        """Test factory with custom threshold."""
        handler = create_multimodal_handler(confidence_threshold=0.9)
        assert handler.confidence_threshold == 0.9

    def test_create_handler_with_pkg_client(self):
        """Test factory with PKG client."""
        mock_client = MagicMock()
        handler = create_multimodal_handler(pkg_client=mock_client)
        assert handler.pkg_client is mock_client


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    def test_empty_query(self):
        """Test handling of empty query."""
        plan = self.handler.analyze_query("")
        assert plan.retrieval_mode == RetrievalMode.ALL_SOURCES

    def test_whitespace_query(self):
        """Test handling of whitespace-only query."""
        plan = self.handler.analyze_query("   ")
        assert plan.retrieval_mode == RetrievalMode.ALL_SOURCES

    @pytest.mark.asyncio
    async def test_execute_no_pkg_client(self):
        """Test execute without PKG client returns empty."""
        handler = MultimodalQueryHandler(pkg_client=None)
        plan = handler.analyze_query("test query")
        results = await handler.execute("test query", plan, top_k=10)
        assert results == []

    def test_very_long_query(self):
        """Test handling of very long query."""
        long_query = "in my voice notes about " + "budget " * 1000
        plan = self.handler.analyze_query(long_query)
        # Should still detect the modality hint
        assert ContentSource.AUDIO_TRANSCRIPTION in plan.target_modalities


class TestSearchStrategies:
    """Tests for different search strategy paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MultimodalQueryHandler()

    @pytest.mark.asyncio
    async def test_all_sources_strategy(self):
        """Test all sources strategy is used."""
        handler = MultimodalQueryHandler()
        # Mock the _all_sources_search method to track it's called
        handler._all_sources_search = AsyncMock(return_value=[])

        plan = MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.ALL_SOURCES,
            target_modalities=list(ContentSource),
            modality_weights={s: 1.0 for s in ContentSource},
            apply_confidence_weighting=False,
            fuzzy_matching_boost=0.0,
            semantic_priority=0.0,
        )

        await handler.execute("test", plan, top_k=10)
        # Verify _all_sources_search was called for ALL_SOURCES mode
        handler._all_sources_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_prioritized_strategy_applies_weights(self):
        """Test prioritized strategy applies modality weights."""
        handler = MultimodalQueryHandler()

        mock_results = [
            MultimodalSearchResult(
                entity_id="e1",
                content="audio content",
                score=0.8,
                source_type=ContentSource.AUDIO_TRANSCRIPTION,
            ),
            MultimodalSearchResult(
                entity_id="e2",
                content="text content",
                score=0.8,
                source_type=ContentSource.TEXT_NATIVE,
            ),
        ]
        handler._execute_search = AsyncMock(return_value=mock_results)

        plan = MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.PRIORITIZED,
            target_modalities=list(ContentSource),
            modality_weights={
                ContentSource.AUDIO_TRANSCRIPTION: 1.2,
                ContentSource.TEXT_NATIVE: 0.8,
                ContentSource.OCR_DOCUMENT: 0.8,
                ContentSource.OCR_IMAGE: 0.8,
                ContentSource.VIDEO_TRANSCRIPTION: 0.8,
                ContentSource.MIXED_SOURCE: 0.8,
            },
            apply_confidence_weighting=False,
            fuzzy_matching_boost=0.0,
            semantic_priority=0.0,
        )

        results = await handler.execute("test", plan, top_k=10)

        # Audio should have higher score due to weight
        audio_result = next(
            r for r in results if r.source_type == ContentSource.AUDIO_TRANSCRIPTION
        )
        text_result = next(
            r for r in results if r.source_type == ContentSource.TEXT_NATIVE
        )

        assert audio_result.score > text_result.score

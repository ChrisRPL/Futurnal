"""Multimodal Query Handler for hybrid search.

Orchestrates multimodal search execution with source-aware strategies
for OCR documents, audio transcriptions, and mixed-source queries.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Responsibilities:
1. Analyze queries for modality hints
2. Build execution plans based on detected modalities
3. Execute source-aware search strategies
4. Apply confidence weighting from source metadata

Integration Points:
- ModalityHintDetector: Query analysis
- OCRContentProcessor: OCR-optimized search
- TranscriptionProcessor: Audio-optimized search
- SchemaAwareRetrieval: Hybrid search execution (via PKGClient)

Option B Compliance:
- Ghost model frozen (pattern-based detection, no LLM calls)
- Local-first processing
- Quality targets: OCR >80%, Audio >75% relevance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
    ModalityHint,
    MultimodalQueryPlan,
    RetrievalMode,
)
from futurnal.search.hybrid.multimodal.hint_detector import ModalityHintDetector
from futurnal.search.hybrid.multimodal.ocr_processor import OCRContentProcessor
from futurnal.search.hybrid.multimodal.transcription_processor import (
    TranscriptionProcessor,
)

if TYPE_CHECKING:
    from futurnal.pkg.client import PKGClient
    from futurnal.search.hybrid.query_router import QueryEmbeddingRouter

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSearchResult:
    """Result from multimodal search.

    Attributes:
        entity_id: ID of the matched entity
        content: Text content of the match
        score: Combined relevance score (0.0-1.0)
        source_type: Content source type
        source_confidence: Extraction confidence of the source
        retrieval_boost: Boost factor applied from source quality
        matched_segment: Specific segment that matched (for transcriptions)
        metadata: Additional metadata from the source
    """

    entity_id: str
    content: str
    score: float
    source_type: ContentSource
    source_confidence: float = 1.0
    retrieval_boost: float = 1.0
    matched_segment: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalSearchStats:
    """Statistics from multimodal search execution.

    Tracks performance and source distribution for analysis.
    """

    query: str
    plan: MultimodalQueryPlan
    total_results: int
    results_by_source: Dict[ContentSource, int]
    execution_time_ms: float
    modality_hints_detected: List[ModalityHint]
    source_filter_applied: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MultimodalQueryHandler:
    """Orchestrates multimodal search execution.

    Handles search across content from different modalities with
    source-aware strategies:

    1. ALL_SOURCES: Search all content types
    2. SINGLE_MODALITY: Filter to specific source type
    3. PRIORITIZED: Boost specific sources in ranking
    4. CROSS_MODAL: Explicit cross-modal queries

    Search Strategies:
    - OCR-optimized: Applies fuzzy matching boost for OCR error tolerance
    - Transcription-optimized: Uses homophone expansion for audio search
    - Prioritized: Standard search with source boost
    - All-sources: Standard hybrid search

    Example:
        handler = MultimodalQueryHandler(
            hint_detector=ModalityHintDetector(),
            pkg_client=pkg_client,
        )

        # Analyze and execute
        plan = handler.analyze_query("what's in my voice notes about budget?")
        results = await handler.execute(
            query="what's in my voice notes about budget?",
            plan=plan,
            top_k=10,
        )

    Attributes:
        hint_detector: Modality hint detector
        pkg_client: PKG client for search execution
        ocr_processor: OCR content processor
        transcription_processor: Transcription processor
        default_confidence_threshold: Min confidence for source filtering
        fuzzy_matching_boost: Score boost for OCR fuzzy matches
        semantic_priority_boost: Score boost for transcription semantic matches
    """

    # Default boost factors
    DEFAULT_FUZZY_BOOST: float = 0.15
    DEFAULT_SEMANTIC_BOOST: float = 0.1
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.75

    def __init__(
        self,
        hint_detector: Optional[ModalityHintDetector] = None,
        pkg_client: Optional["PKGClient"] = None,
        embedding_router: Optional["QueryEmbeddingRouter"] = None,
        ocr_processor: Optional[OCRContentProcessor] = None,
        transcription_processor: Optional[TranscriptionProcessor] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize MultimodalQueryHandler.

        Args:
            hint_detector: ModalityHintDetector for query analysis
            pkg_client: PKGClient for search execution
            embedding_router: QueryEmbeddingRouter for embeddings
            ocr_processor: OCRContentProcessor for OCR queries
            transcription_processor: TranscriptionProcessor for audio queries
            confidence_threshold: Minimum confidence for modality filtering
        """
        self.hint_detector = hint_detector or ModalityHintDetector()
        self.pkg_client = pkg_client
        self.embedding_router = embedding_router
        self.ocr_processor = ocr_processor or OCRContentProcessor()
        self.transcription_processor = (
            transcription_processor or TranscriptionProcessor()
        )
        self.confidence_threshold = confidence_threshold

        logger.info(
            "MultimodalQueryHandler initialized",
            extra={"confidence_threshold": confidence_threshold},
        )

    def analyze_query(self, query: str) -> MultimodalQueryPlan:
        """Analyze query and create execution plan.

        Detects modality hints and builds a plan with:
        - Retrieval mode (all_sources, single_modality, prioritized)
        - Target modalities to search
        - Modality weights for ranking
        - Confidence weighting settings

        Args:
            query: User search query

        Returns:
            MultimodalQueryPlan with execution settings

        Example:
            plan = handler.analyze_query("in my voice notes about budget")
            print(plan.retrieval_mode)  # RetrievalMode.SINGLE_MODALITY
            print(plan.target_modalities)  # [ContentSource.AUDIO_TRANSCRIPTION]
        """
        hints = self.hint_detector.detect(query)

        # No hints - search all sources
        if not hints:
            logger.debug(f"No modality hints in query: {query[:50]}...")
            return MultimodalQueryPlan(
                retrieval_mode=RetrievalMode.ALL_SOURCES,
                target_modalities=list(ContentSource),
                modality_weights={s: 1.0 for s in ContentSource},
                apply_confidence_weighting=True,
                fuzzy_matching_boost=0.0,
                semantic_priority=0.0,
            )

        # Get primary hint
        primary = hints[0]

        # High confidence - use single modality filtering
        if primary.confidence >= 0.90:
            logger.debug(
                f"High confidence hint ({primary.confidence:.2f}): "
                f"{primary.modality.value}"
            )
            return self._build_single_modality_plan(primary, hints)

        # Moderate confidence - prioritized search
        if primary.confidence >= self.confidence_threshold:
            logger.debug(
                f"Moderate confidence hint ({primary.confidence:.2f}): "
                f"{primary.modality.value}"
            )
            return self._build_prioritized_plan(primary, hints)

        # Low confidence - all sources with slight boost
        logger.debug(
            f"Low confidence hint ({primary.confidence:.2f}): "
            f"{primary.modality.value}"
        )
        return self._build_boosted_all_sources_plan(primary, hints)

    def _build_single_modality_plan(
        self, primary: ModalityHint, hints: List[ModalityHint]
    ) -> MultimodalQueryPlan:
        """Build plan for single modality filtering.

        Args:
            primary: Primary modality hint
            hints: All detected hints

        Returns:
            MultimodalQueryPlan with single modality mode
        """
        # Determine if OCR or audio optimizations apply
        is_ocr = primary.modality in (
            ContentSource.OCR_DOCUMENT,
            ContentSource.OCR_IMAGE,
        )
        is_audio = primary.modality in (
            ContentSource.AUDIO_TRANSCRIPTION,
            ContentSource.VIDEO_TRANSCRIPTION,
        )

        return MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.SINGLE_MODALITY,
            target_modalities=[primary.modality],
            modality_weights={primary.modality: 1.0},
            apply_confidence_weighting=True,
            fuzzy_matching_boost=self.DEFAULT_FUZZY_BOOST if is_ocr else 0.0,
            semantic_priority=self.DEFAULT_SEMANTIC_BOOST if is_audio else 0.0,
        )

    def _build_prioritized_plan(
        self, primary: ModalityHint, hints: List[ModalityHint]
    ) -> MultimodalQueryPlan:
        """Build plan for prioritized search with source boosting.

        Args:
            primary: Primary modality hint
            hints: All detected hints

        Returns:
            MultimodalQueryPlan with prioritized mode
        """
        # Get all detected modalities with confidence >= threshold
        target_modalities = self.hint_detector.get_all_modalities(
            hints[0].hint_phrase  # Use original query
        )
        if not target_modalities:
            target_modalities = [primary.modality]

        # Build weights - higher weight for detected modalities
        weights: Dict[ContentSource, float] = {}
        for source in ContentSource:
            if source == primary.modality:
                weights[source] = 1.0  # Primary gets full weight
            elif source in target_modalities:
                weights[source] = 0.8  # Secondary detected modalities
            else:
                weights[source] = 0.5  # Others get reduced weight

        # Determine optimizations
        is_ocr = primary.modality in (
            ContentSource.OCR_DOCUMENT,
            ContentSource.OCR_IMAGE,
        )
        is_audio = primary.modality in (
            ContentSource.AUDIO_TRANSCRIPTION,
            ContentSource.VIDEO_TRANSCRIPTION,
        )

        return MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.PRIORITIZED,
            target_modalities=list(ContentSource),  # Search all
            modality_weights=weights,
            apply_confidence_weighting=True,
            fuzzy_matching_boost=self.DEFAULT_FUZZY_BOOST * 0.5 if is_ocr else 0.0,
            semantic_priority=self.DEFAULT_SEMANTIC_BOOST * 0.5 if is_audio else 0.0,
        )

    def _build_boosted_all_sources_plan(
        self, primary: ModalityHint, hints: List[ModalityHint]
    ) -> MultimodalQueryPlan:
        """Build plan for all sources with slight boost for detected modality.

        Args:
            primary: Primary modality hint
            hints: All detected hints

        Returns:
            MultimodalQueryPlan with all sources and slight boost
        """
        # Slight boost for detected modality
        weights = {s: 1.0 for s in ContentSource}
        weights[primary.modality] = 1.1  # 10% boost

        return MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.ALL_SOURCES,
            target_modalities=list(ContentSource),
            modality_weights=weights,
            apply_confidence_weighting=True,
            fuzzy_matching_boost=0.0,
            semantic_priority=0.0,
        )

    async def execute(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int = 20,
    ) -> List[MultimodalSearchResult]:
        """Execute multimodal search with given plan.

        Executes the appropriate strategy based on plan:
        - SINGLE_MODALITY: Filter to specific source with optimizations
        - PRIORITIZED: Search all with source boosting
        - ALL_SOURCES: Standard hybrid search
        - CROSS_MODAL: Explicit cross-modal queries

        Args:
            query: User search query
            plan: MultimodalQueryPlan from analyze_query
            top_k: Maximum results to return

        Returns:
            List of MultimodalSearchResult sorted by score

        Example:
            results = await handler.execute(query, plan, top_k=10)
            for r in results:
                print(f"{r.source_type.value}: {r.score:.2f} - {r.content[:50]}")
        """
        import time

        start_time = time.perf_counter()

        # Route to appropriate strategy
        if plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY:
            results = await self._single_modality_search(query, plan, top_k)
        elif plan.retrieval_mode == RetrievalMode.PRIORITIZED:
            results = await self._prioritized_search(query, plan, top_k)
        elif plan.retrieval_mode == RetrievalMode.CROSS_MODAL:
            results = await self._cross_modal_search(query, plan, top_k)
        else:  # ALL_SOURCES
            results = await self._all_sources_search(query, plan, top_k)

        # Apply confidence weighting if enabled
        if plan.apply_confidence_weighting:
            results = self._apply_confidence_weighting(results)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Take top_k
        results = results[:top_k]

        # Log stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Multimodal search completed in {elapsed_ms:.1f}ms",
            extra={
                "query_length": len(query),
                "retrieval_mode": plan.retrieval_mode.value,
                "result_count": len(results),
                "elapsed_ms": elapsed_ms,
            },
        )

        return results

    async def _single_modality_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int,
    ) -> List[MultimodalSearchResult]:
        """Execute search filtered to single modality.

        Applies modality-specific optimizations:
        - OCR: Fuzzy variant matching
        - Audio: Homophone expansion

        Args:
            query: Search query
            plan: Query plan with target modality
            top_k: Max results

        Returns:
            Filtered search results
        """
        if not plan.target_modalities:
            return []

        target = plan.target_modalities[0]

        # Prepare query based on modality
        if target in (ContentSource.OCR_DOCUMENT, ContentSource.OCR_IMAGE):
            # OCR-optimized: create normalized searchable content
            search_query = self.ocr_processor.create_searchable_content(query)
        elif target in (
            ContentSource.AUDIO_TRANSCRIPTION,
            ContentSource.VIDEO_TRANSCRIPTION,
        ):
            # Audio-optimized: expand with homophones
            search_query = self.transcription_processor.expand_query_for_transcription(
                query
            )
        else:
            search_query = query

        # Execute filtered search
        return await self._execute_filtered_search(
            query=search_query,
            original_query=query,
            source_filter=[target],
            top_k=top_k,
            fuzzy_boost=plan.fuzzy_matching_boost,
        )

    async def _prioritized_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int,
    ) -> List[MultimodalSearchResult]:
        """Execute search with source prioritization.

        Searches all sources but applies weights to boost
        results from prioritized modalities.

        Args:
            query: Search query
            plan: Query plan with modality weights
            top_k: Max results

        Returns:
            Search results with weighted scores
        """
        # Search all sources
        results = await self._execute_search(query, top_k=top_k * 2)

        # Apply modality weights
        for result in results:
            weight = plan.modality_weights.get(result.source_type, 1.0)
            result.score *= weight

        return results

    async def _cross_modal_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int,
    ) -> List[MultimodalSearchResult]:
        """Execute explicit cross-modal search.

        Searches multiple specified modalities with
        balanced result distribution.

        Args:
            query: Search query
            plan: Query plan with target modalities
            top_k: Max results

        Returns:
            Cross-modal search results
        """
        if not plan.target_modalities:
            return await self._all_sources_search(query, plan, top_k)

        # Search each target modality
        all_results: List[MultimodalSearchResult] = []
        results_per_modality = max(top_k // len(plan.target_modalities), 3)

        for modality in plan.target_modalities:
            modality_results = await self._execute_filtered_search(
                query=query,
                original_query=query,
                source_filter=[modality],
                top_k=results_per_modality,
            )
            all_results.extend(modality_results)

        return all_results

    async def _all_sources_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int,
    ) -> List[MultimodalSearchResult]:
        """Execute standard search across all sources.

        Args:
            query: Search query
            plan: Query plan
            top_k: Max results

        Returns:
            Search results from all sources
        """
        return await self._execute_search(query, top_k=top_k)

    async def _execute_search(
        self,
        query: str,
        top_k: int,
    ) -> List[MultimodalSearchResult]:
        """Execute base search operation.

        Override this to integrate with actual search backend.

        Args:
            query: Search query
            top_k: Max results

        Returns:
            Search results
        """
        if self.pkg_client is None:
            logger.warning("No PKG client configured, returning empty results")
            return []

        # TODO: Integrate with PKGClient.search() when available
        # For now, return empty results as placeholder
        # In production, this would call:
        #   results = await self.pkg_client.search(query, top_k=top_k)
        #   return [self._convert_pkg_result(r) for r in results]
        return []

    async def _execute_filtered_search(
        self,
        query: str,
        original_query: str,
        source_filter: List[ContentSource],
        top_k: int,
        fuzzy_boost: float = 0.0,
    ) -> List[MultimodalSearchResult]:
        """Execute search with source type filtering.

        Override this to integrate with actual search backend.

        Args:
            query: Processed search query
            original_query: Original user query
            source_filter: Content sources to include
            top_k: Max results
            fuzzy_boost: Score boost for fuzzy matches

        Returns:
            Filtered search results
        """
        if self.pkg_client is None:
            logger.warning("No PKG client configured, returning empty results")
            return []

        # TODO: Integrate with PKGClient.search() with source_filter
        # In production, this would call:
        #   results = await self.pkg_client.search(
        #       query,
        #       top_k=top_k,
        #       source_filter={"source_type": [s.value for s in source_filter]}
        #   )
        #   return [self._convert_pkg_result(r, fuzzy_boost) for r in results]
        return []

    def _apply_confidence_weighting(
        self,
        results: List[MultimodalSearchResult],
    ) -> List[MultimodalSearchResult]:
        """Apply confidence weighting to results.

        Adjusts scores based on source extraction confidence.
        Higher extraction confidence = higher retrieval boost.

        Args:
            results: Search results to weight

        Returns:
            Results with adjusted scores
        """
        for result in results:
            # Apply retrieval boost from source confidence
            result.score *= result.retrieval_boost

        return results

    def get_clean_query(self, query: str) -> str:
        """Get query with modality hints removed.

        Useful for semantic search where modality context adds noise.

        Args:
            query: Original user query

        Returns:
            Query with hint phrases removed

        Example:
            clean = handler.get_clean_query("in my voice notes about budget")
            print(clean)  # "about budget"
        """
        return self.hint_detector.extract_query_without_hints(query)

    def get_modality_summary(self, query: str) -> Dict[ContentSource, float]:
        """Get confidence summary for detected modalities.

        Args:
            query: User query

        Returns:
            Dict mapping content sources to max confidence
        """
        return self.hint_detector.get_confidence_summary(query)


def create_multimodal_handler(
    pkg_client: Optional["PKGClient"] = None,
    embedding_router: Optional["QueryEmbeddingRouter"] = None,
    confidence_threshold: float = MultimodalQueryHandler.DEFAULT_CONFIDENCE_THRESHOLD,
) -> MultimodalQueryHandler:
    """Factory function to create MultimodalQueryHandler.

    Creates handler with default processors and hint detector.

    Args:
        pkg_client: PKG client for search execution
        embedding_router: Query embedding router
        confidence_threshold: Minimum confidence for filtering

    Returns:
        Configured MultimodalQueryHandler

    Example:
        handler = create_multimodal_handler(pkg_client=pkg_client)
        plan = handler.analyze_query("search my voice notes")
    """
    return MultimodalQueryHandler(
        hint_detector=ModalityHintDetector(),
        pkg_client=pkg_client,
        embedding_router=embedding_router,
        ocr_processor=OCRContentProcessor(),
        transcription_processor=TranscriptionProcessor(),
        confidence_threshold=confidence_threshold,
    )

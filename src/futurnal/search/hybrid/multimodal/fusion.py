"""Cross-Modal Fusion for multimodal search results.

Fuses results from multiple modalities with diversity injection
and modality balance enforcement.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Responsibilities:
1. Normalize scores across modalities (0-1 range)
2. Deduplicate by content similarity
3. Apply MMR-like diversity selection
4. Enforce modality balance (max dominance limit)
5. Ensure minimum results per modality

Integration Points:
- MultimodalQueryHandler: Uses fusion for cross-modal queries
- MultimodalSearchResult: Input/output result type

Option B Compliance:
- Local-first processing (no external calls)
- Deterministic fusion for reproducibility
- Quality gate: Cross-modal fusion MRR >0.7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from futurnal.search.hybrid.multimodal.types import ContentSource

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for cross-modal result fusion.

    Attributes:
        normalize_scores: Whether to normalize scores to 0-1 per modality
        apply_diversity: Whether to apply MMR-like diversity selection
        diversity_factor: Lambda for diversity (0=pure relevance, 1=pure diversity)
        min_results_per_modality: Minimum results to include from each modality
        max_modality_dominance: Maximum fraction from any single modality (0.0-1.0)
        similarity_threshold: Threshold for deduplication (0.0-1.0)
    """

    normalize_scores: bool = True
    apply_diversity: bool = True
    diversity_factor: float = 0.2
    min_results_per_modality: int = 2
    max_modality_dominance: float = 0.7
    similarity_threshold: float = 0.85


@dataclass
class FusedResult:
    """A fused search result from cross-modal fusion.

    Attributes:
        entity_id: ID of the matched entity
        content: Text content of the match
        original_score: Score before fusion adjustments
        fused_score: Score after fusion (normalized, diversity-adjusted)
        source_type: Content source type
        source_confidence: Extraction confidence of the source
        diversity_penalty: Penalty applied for diversity (0.0-1.0)
        metadata: Additional metadata
    """

    entity_id: str
    content: str
    original_score: float
    fused_score: float
    source_type: ContentSource
    source_confidence: float = 1.0
    diversity_penalty: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusionStats:
    """Statistics from fusion operation.

    Tracks fusion metrics for quality monitoring.
    """

    total_input_results: int
    results_per_modality_input: Dict[ContentSource, int]
    total_output_results: int
    results_per_modality_output: Dict[ContentSource, int]
    duplicates_removed: int
    diversity_reranked: int
    dominance_adjusted: int


class CrossModalFusion:
    """Fuses results from multiple modalities with diversity injection.

    Implements a multi-stage fusion pipeline:
    1. Score normalization: Normalizes scores within each modality to 0-1
    2. Deduplication: Removes near-duplicate content across modalities
    3. Diversity selection: MMR-like selection for diverse results
    4. Balance enforcement: Limits dominance of any single modality

    Example:
        fusion = CrossModalFusion(config=FusionConfig(
            diversity_factor=0.3,
            max_modality_dominance=0.6,
        ))

        results = fusion.fuse(
            results_by_modality={
                ContentSource.AUDIO_TRANSCRIPTION: audio_results,
                ContentSource.OCR_DOCUMENT: ocr_results,
            },
            top_k=10,
        )

    Attributes:
        config: FusionConfig with fusion settings
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        """Initialize CrossModalFusion.

        Args:
            config: FusionConfig with fusion settings (uses defaults if None)
        """
        self.config = config or FusionConfig()
        logger.info(
            "CrossModalFusion initialized",
            extra={
                "diversity_factor": self.config.diversity_factor,
                "max_dominance": self.config.max_modality_dominance,
            },
        )

    def fuse(
        self,
        results_by_modality: Dict[ContentSource, List[Dict]],
        top_k: int,
    ) -> List[FusedResult]:
        """Combine results from multiple modalities.

        Main entry point for cross-modal fusion. Applies the full
        fusion pipeline: normalize -> dedupe -> diversify -> balance.

        Args:
            results_by_modality: Dict mapping content sources to result lists.
                Each result dict should have: entity_id, content, score,
                and optionally: source_confidence, metadata
            top_k: Maximum number of results to return

        Returns:
            List of FusedResult sorted by fused_score descending

        Example:
            fused = fusion.fuse({
                ContentSource.AUDIO_TRANSCRIPTION: [
                    {"entity_id": "e1", "content": "...", "score": 0.9},
                ],
                ContentSource.OCR_DOCUMENT: [
                    {"entity_id": "e2", "content": "...", "score": 0.85},
                ],
            }, top_k=10)
        """
        # Track stats
        stats = FusionStats(
            total_input_results=sum(len(r) for r in results_by_modality.values()),
            results_per_modality_input={
                k: len(v) for k, v in results_by_modality.items()
            },
            total_output_results=0,
            results_per_modality_output={},
            duplicates_removed=0,
            diversity_reranked=0,
            dominance_adjusted=0,
        )

        # Convert to FusedResult objects
        all_results = self._convert_to_fused_results(results_by_modality)

        if not all_results:
            return []

        # Step 1: Normalize scores
        if self.config.normalize_scores:
            all_results = self._normalize_scores(all_results, results_by_modality)

        # Step 2: Deduplicate
        all_results, removed = self._deduplicate(all_results)
        stats.duplicates_removed = removed

        # Step 3: Apply diversity selection
        if self.config.apply_diversity:
            all_results, reranked = self._apply_diversity(all_results, top_k * 2)
            stats.diversity_reranked = reranked

        # Step 4: Enforce modality balance
        all_results, adjusted = self._enforce_modality_balance(all_results, top_k)
        stats.dominance_adjusted = adjusted

        # Take top_k and sort
        final_results = sorted(all_results, key=lambda r: r.fused_score, reverse=True)[
            :top_k
        ]

        # Update stats
        stats.total_output_results = len(final_results)
        stats.results_per_modality_output = self._count_by_modality(final_results)

        logger.debug(
            f"Fusion complete: {stats.total_input_results} -> {stats.total_output_results}",
            extra={
                "duplicates_removed": stats.duplicates_removed,
                "diversity_reranked": stats.diversity_reranked,
                "dominance_adjusted": stats.dominance_adjusted,
            },
        )

        return final_results

    def _convert_to_fused_results(
        self,
        results_by_modality: Dict[ContentSource, List[Dict]],
    ) -> List[FusedResult]:
        """Convert input dicts to FusedResult objects.

        Args:
            results_by_modality: Dict of modality -> result list

        Returns:
            List of FusedResult objects
        """
        all_results: List[FusedResult] = []

        for source_type, results in results_by_modality.items():
            for r in results:
                score = r.get("score", 0.0)
                all_results.append(
                    FusedResult(
                        entity_id=r.get("entity_id", ""),
                        content=r.get("content", ""),
                        original_score=score,
                        fused_score=score,
                        source_type=source_type,
                        source_confidence=r.get("source_confidence", 1.0),
                        metadata=r.get("metadata", {}),
                    )
                )

        return all_results

    def _normalize_scores(
        self,
        results: List[FusedResult],
        results_by_modality: Dict[ContentSource, List[Dict]],
    ) -> List[FusedResult]:
        """Normalize scores to 0-1 range per modality.

        Uses min-max normalization within each modality to ensure
        fair comparison across sources.

        Args:
            results: List of FusedResult to normalize
            results_by_modality: Original results by modality for stats

        Returns:
            Results with normalized fused_score
        """
        # Group by modality
        by_modality: Dict[ContentSource, List[FusedResult]] = {}
        for r in results:
            if r.source_type not in by_modality:
                by_modality[r.source_type] = []
            by_modality[r.source_type].append(r)

        # Normalize each modality
        for modality, modality_results in by_modality.items():
            if not modality_results:
                continue

            scores = [r.original_score for r in modality_results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for r in modality_results:
                if score_range > 0:
                    # Min-max normalization
                    r.fused_score = (r.original_score - min_score) / score_range
                else:
                    # All scores equal - normalize to 1.0
                    r.fused_score = 1.0

        return results

    def _deduplicate(
        self,
        results: List[FusedResult],
    ) -> Tuple[List[FusedResult], int]:
        """Remove near-duplicate results based on content similarity.

        Uses simple character-level similarity for efficiency.
        Keeps the result with the higher score when duplicates found.

        Args:
            results: List of results to deduplicate

        Returns:
            Tuple of (deduplicated results, count removed)
        """
        if not results:
            return [], 0

        seen_contents: Dict[str, FusedResult] = {}
        duplicates = 0

        for r in results:
            # Create normalized content key
            content_key = self._content_signature(r.content)

            if content_key in seen_contents:
                existing = seen_contents[content_key]
                # Keep higher scoring result
                if r.fused_score > existing.fused_score:
                    seen_contents[content_key] = r
                duplicates += 1
            else:
                # Check for similar content
                is_duplicate = False
                for key, existing in seen_contents.items():
                    if self._content_similarity(r.content, existing.content) >= self.config.similarity_threshold:
                        # Duplicate found - keep higher score
                        if r.fused_score > existing.fused_score:
                            seen_contents[key] = r
                        is_duplicate = True
                        duplicates += 1
                        break

                if not is_duplicate:
                    seen_contents[content_key] = r

        return list(seen_contents.values()), duplicates

    def _content_signature(self, content: str) -> str:
        """Create a signature for content deduplication.

        Args:
            content: Text content

        Returns:
            Normalized content signature
        """
        # Normalize: lowercase, remove extra whitespace
        normalized = " ".join(content.lower().split())
        # Take first 200 chars as signature
        return normalized[:200]

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple similarity between two content strings.

        Uses character-level Jaccard similarity for efficiency.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score (0.0-1.0)
        """
        if not content1 or not content2:
            return 0.0

        # Normalize
        norm1 = set(content1.lower().split())
        norm2 = set(content2.lower().split())

        if not norm1 or not norm2:
            return 0.0

        # Jaccard similarity
        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)

        return intersection / union if union > 0 else 0.0

    def _apply_diversity(
        self,
        results: List[FusedResult],
        max_results: int,
    ) -> Tuple[List[FusedResult], int]:
        """Apply MMR-like diversity selection.

        Iteratively selects results that balance relevance and diversity.
        Uses a greedy approach: at each step, select the result that
        maximizes: (1-lambda)*relevance - lambda*max_similarity_to_selected

        Args:
            results: List of candidate results
            max_results: Maximum results to select

        Returns:
            Tuple of (diverse results, count reranked)
        """
        if not results or not self.config.apply_diversity:
            return results, 0

        lambda_param = self.config.diversity_factor
        selected: List[FusedResult] = []
        remaining = list(results)
        reranked = 0

        while remaining and len(selected) < max_results:
            best_result: Optional[FusedResult] = None
            best_mmr_score = float("-inf")

            for candidate in remaining:
                # Relevance component
                relevance = candidate.fused_score

                # Diversity component: max similarity to any selected result
                if selected:
                    max_sim = max(
                        self._content_similarity(candidate.content, s.content)
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr_score = (1 - lambda_param) * relevance - lambda_param * max_sim

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_result = candidate

            if best_result:
                # Track if position changed
                original_rank = results.index(best_result)
                new_rank = len(selected)
                if original_rank != new_rank:
                    reranked += 1

                # Update diversity penalty
                if selected:
                    best_result.diversity_penalty = max(
                        self._content_similarity(best_result.content, s.content)
                        for s in selected
                    )

                selected.append(best_result)
                remaining.remove(best_result)

        return selected, reranked

    def _enforce_modality_balance(
        self,
        results: List[FusedResult],
        top_k: int,
    ) -> Tuple[List[FusedResult], int]:
        """Enforce maximum modality dominance.

        Ensures no single modality exceeds max_modality_dominance fraction
        of the final results. Redistributes slots to underrepresented modalities.

        Args:
            results: List of results to balance
            top_k: Target number of results

        Returns:
            Tuple of (balanced results, count adjusted)
        """
        if not results or self.config.max_modality_dominance >= 1.0:
            return results, 0

        max_per_modality = int(top_k * self.config.max_modality_dominance)
        min_per_modality = self.config.min_results_per_modality

        # Count by modality
        counts = self._count_by_modality(results)
        adjusted = 0

        # Identify over-represented modalities
        overflow: List[FusedResult] = []
        final_results: List[FusedResult] = []

        # Group by modality and sort each group by score
        by_modality: Dict[ContentSource, List[FusedResult]] = {}
        for r in results:
            if r.source_type not in by_modality:
                by_modality[r.source_type] = []
            by_modality[r.source_type].append(r)

        for modality, modality_results in by_modality.items():
            modality_results.sort(key=lambda x: x.fused_score, reverse=True)

            # Take up to max_per_modality
            for i, r in enumerate(modality_results):
                if i < max_per_modality:
                    final_results.append(r)
                else:
                    overflow.append(r)
                    adjusted += 1

        # Fill remaining slots from overflow, respecting min_per_modality
        remaining_slots = top_k - len(final_results)
        if remaining_slots > 0 and overflow:
            # Sort overflow by score
            overflow.sort(key=lambda x: x.fused_score, reverse=True)

            # Check which modalities need minimum results
            current_counts = self._count_by_modality(final_results)

            for r in overflow:
                if remaining_slots <= 0:
                    break

                modality_count = current_counts.get(r.source_type, 0)
                if modality_count < min_per_modality:
                    final_results.append(r)
                    current_counts[r.source_type] = modality_count + 1
                    remaining_slots -= 1

            # Fill any remaining slots with top overflow, but respect max dominance
            for r in overflow:
                if remaining_slots <= 0:
                    break
                if r not in final_results:
                    modality_count = current_counts.get(r.source_type, 0)
                    # Only add if it won't exceed max dominance
                    if modality_count < max_per_modality:
                        final_results.append(r)
                        current_counts[r.source_type] = modality_count + 1
                        remaining_slots -= 1

        return final_results, adjusted

    def _count_by_modality(
        self,
        results: List[FusedResult],
    ) -> Dict[ContentSource, int]:
        """Count results by modality.

        Args:
            results: List of results

        Returns:
            Dict mapping modality to count
        """
        counts: Dict[ContentSource, int] = {}
        for r in results:
            counts[r.source_type] = counts.get(r.source_type, 0) + 1
        return counts


def create_fusion_config(
    diversity_factor: float = 0.2,
    max_modality_dominance: float = 0.7,
    min_results_per_modality: int = 2,
) -> FusionConfig:
    """Factory function to create FusionConfig.

    Args:
        diversity_factor: Lambda for diversity (0=relevance, 1=diversity)
        max_modality_dominance: Max fraction from any modality
        min_results_per_modality: Minimum results per modality

    Returns:
        Configured FusionConfig
    """
    return FusionConfig(
        normalize_scores=True,
        apply_diversity=True,
        diversity_factor=diversity_factor,
        min_results_per_modality=min_results_per_modality,
        max_modality_dominance=max_modality_dominance,
    )

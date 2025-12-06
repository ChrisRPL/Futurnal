"""Result Fusion Algorithms.

Combines vector similarity and graph traversal results with weighted scoring.

Features:
- Weighted score combination
- Deduplication by entity_id
- Ranking by combined score
- Source tracking (vector, graph, hybrid)

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Deterministic fusion (no random sampling)
- Transparent scoring for audit
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import FusionError
from futurnal.search.hybrid.types import (
    GraphSearchResult,
    HybridSearchResult,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)


class ResultFusion:
    """Fuses vector and graph results with weighted scoring.

    Combines results from vector similarity search and graph traversal
    into a unified ranked list.

    Fusion Algorithm:
    1. Build score maps from vector and graph results
    2. Combine all unique entity IDs
    3. Calculate combined score: (vector_score * vector_weight) + (graph_score * graph_weight)
    4. Deduplicate by entity_id (keep highest combined score)
    5. Sort by combined score descending

    Source Attribution:
    - "vector": Only found in vector search
    - "graph": Only found in graph expansion
    - "hybrid": Found in both (reinforced)

    Example:
        >>> from futurnal.search.hybrid import ResultFusion, HybridSearchConfig

        >>> fusion = ResultFusion()
        >>> results = fusion.fuse_results(
        ...     vector_results=vector_results,
        ...     graph_results=graph_results,
        ...     vector_weight=0.5,
        ...     graph_weight=0.5,
        ... )
    """

    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize result fusion.

        Args:
            config: Hybrid search configuration
        """
        self._config = config or HybridSearchConfig()

    @property
    def config(self) -> HybridSearchConfig:
        """Get the configuration."""
        return self._config

    def fuse_results(
        self,
        vector_results: List[VectorSearchResult],
        graph_results: List[GraphSearchResult],
        vector_weight: float,
        graph_weight: float,
    ) -> List[HybridSearchResult]:
        """Fuse vector and graph results with weighted scoring.

        Args:
            vector_results: Results from vector similarity search
            graph_results: Results from graph traversal
            vector_weight: Weight for vector scores (0-1)
            graph_weight: Weight for graph scores (0-1)

        Returns:
            List of HybridSearchResult sorted by combined_score descending

        Raises:
            FusionError: If fusion fails
        """
        try:
            # Normalize weights
            total_weight = vector_weight + graph_weight
            if total_weight > 0:
                vector_weight = vector_weight / total_weight
                graph_weight = graph_weight / total_weight
            else:
                vector_weight = 0.5
                graph_weight = 0.5

            # Build score maps
            vector_scores: Dict[str, VectorSearchResult] = {}
            for r in vector_results:
                if r.entity_id not in vector_scores:
                    vector_scores[r.entity_id] = r
                elif r.similarity_score > vector_scores[r.entity_id].similarity_score:
                    vector_scores[r.entity_id] = r

            graph_scores: Dict[str, GraphSearchResult] = {}
            for r in graph_results:
                if r.entity_id not in graph_scores:
                    graph_scores[r.entity_id] = r
                elif r.path_score > graph_scores[r.entity_id].path_score:
                    graph_scores[r.entity_id] = r

            # Combine all entity IDs
            all_entity_ids = set(vector_scores.keys()) | set(graph_scores.keys())

            # Calculate combined scores
            fused_results = []
            for entity_id in all_entity_ids:
                vector_result = vector_scores.get(entity_id)
                graph_result = graph_scores.get(entity_id)

                v_score = vector_result.similarity_score if vector_result else 0.0
                g_score = graph_result.path_score if graph_result else 0.0

                combined_score = (v_score * vector_weight) + (g_score * graph_weight)

                # Determine source
                if vector_result and graph_result:
                    source = "hybrid"
                elif vector_result:
                    source = "vector"
                else:
                    source = "graph"

                # Get entity details (prefer vector result for content)
                if vector_result:
                    entity_type = vector_result.entity_type
                    content = vector_result.content
                    schema_version = vector_result.schema_version
                    metadata = dict(vector_result.metadata)
                else:
                    entity_type = graph_result.entity_type
                    content = ""
                    schema_version = None
                    metadata = dict(graph_result.metadata)

                # Add graph metadata if available
                if graph_result:
                    metadata["path_from_seed"] = graph_result.path_from_seed
                    metadata["relationship_types"] = graph_result.relationship_types

                fused_results.append(
                    HybridSearchResult(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        vector_score=v_score,
                        graph_score=g_score,
                        combined_score=combined_score,
                        source=source,
                        content=content,
                        schema_version=schema_version,
                        metadata=metadata,
                    )
                )

            # Sort by combined score descending
            fused_results.sort(key=lambda x: x.combined_score, reverse=True)

            logger.debug(
                f"Fused {len(vector_results)} vector + {len(graph_results)} graph "
                f"-> {len(fused_results)} results "
                f"(weights: vector={vector_weight:.2f}, graph={graph_weight:.2f})"
            )

            return fused_results

        except Exception as e:
            raise FusionError(f"Result fusion failed: {e}") from e

    def reciprocal_rank_fusion(
        self,
        vector_results: List[VectorSearchResult],
        graph_results: List[GraphSearchResult],
        k: int = 60,
    ) -> List[HybridSearchResult]:
        """Alternative fusion using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = sum(1 / (k + rank(d, q)))
        where k is a constant (typically 60) and rank is the position in each list.

        This method is less sensitive to score distributions and can work
        well when scores from different sources are not directly comparable.

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph expansion
            k: RRF constant (default 60)

        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        try:
            # Build rank maps
            vector_ranks: Dict[str, int] = {}
            for rank, r in enumerate(vector_results, start=1):
                if r.entity_id not in vector_ranks:
                    vector_ranks[r.entity_id] = rank

            graph_ranks: Dict[str, int] = {}
            for rank, r in enumerate(graph_results, start=1):
                if r.entity_id not in graph_ranks:
                    graph_ranks[r.entity_id] = rank

            # Combine all entity IDs
            all_entity_ids = set(vector_ranks.keys()) | set(graph_ranks.keys())

            # Calculate RRF scores
            rrf_scores: Dict[str, float] = {}
            for entity_id in all_entity_ids:
                score = 0.0
                if entity_id in vector_ranks:
                    score += 1.0 / (k + vector_ranks[entity_id])
                if entity_id in graph_ranks:
                    score += 1.0 / (k + graph_ranks[entity_id])
                rrf_scores[entity_id] = score

            # Build results
            vector_map = {r.entity_id: r for r in vector_results}
            graph_map = {r.entity_id: r for r in graph_results}

            fused_results = []
            for entity_id in all_entity_ids:
                vector_result = vector_map.get(entity_id)
                graph_result = graph_map.get(entity_id)

                # Determine source
                if vector_result and graph_result:
                    source = "hybrid"
                elif vector_result:
                    source = "vector"
                else:
                    source = "graph"

                # Get entity details
                if vector_result:
                    entity_type = vector_result.entity_type
                    content = vector_result.content
                    schema_version = vector_result.schema_version
                    metadata = dict(vector_result.metadata)
                else:
                    entity_type = graph_result.entity_type
                    content = ""
                    schema_version = None
                    metadata = dict(graph_result.metadata)

                fused_results.append(
                    HybridSearchResult(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        vector_score=vector_result.similarity_score if vector_result else 0.0,
                        graph_score=graph_result.path_score if graph_result else 0.0,
                        combined_score=rrf_scores[entity_id],
                        source=source,
                        content=content,
                        schema_version=schema_version,
                        metadata=metadata,
                    )
                )

            # Sort by RRF score descending
            fused_results.sort(key=lambda x: x.combined_score, reverse=True)

            return fused_results

        except Exception as e:
            raise FusionError(f"RRF fusion failed: {e}") from e

"""Tests for ResultFusion.

Tests weighted score combination and result fusion algorithms.
"""

import pytest

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import FusionError
from futurnal.search.hybrid.fusion import ResultFusion

from tests.search.hybrid.conftest import create_graph_result, create_vector_result


class TestResultFusion:
    """Test ResultFusion functionality."""

    def test_init(self, hybrid_config):
        """Fusion initializes correctly."""
        fusion = ResultFusion(config=hybrid_config)
        assert fusion.config == hybrid_config

    # -------------------------------------------------------------------------
    # Basic Fusion Tests
    # -------------------------------------------------------------------------

    def test_fuse_vector_only_results(self, hybrid_config):
        """Fusion with only vector results."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [
            create_vector_result("entity_1", similarity_score=0.9),
            create_vector_result("entity_2", similarity_score=0.8),
        ]
        graph_results = []

        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        assert len(fused) == 2
        # Scores should be halved (vector_weight * score, graph is 0)
        assert fused[0].vector_score == 0.9
        assert fused[0].graph_score == 0.0
        assert fused[0].source == "vector"

    def test_fuse_graph_only_results(self, hybrid_config):
        """Fusion with only graph results."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = []
        graph_results = [
            create_graph_result("entity_1", path_score=0.8),
            create_graph_result("entity_2", path_score=0.6),
        ]

        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        assert len(fused) == 2
        assert fused[0].vector_score == 0.0
        assert fused[0].graph_score == 0.8
        assert fused[0].source == "graph"

    def test_fuse_hybrid_results(self, sample_vector_results, sample_graph_results, hybrid_config):
        """Fusion with both vector and graph results."""
        fusion = ResultFusion(config=hybrid_config)

        fused = fusion.fuse_results(
            vector_results=sample_vector_results,
            graph_results=sample_graph_results,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        # entity_1 appears in both - should be marked as hybrid
        entity_1_result = next(r for r in fused if r.entity_id == "entity_1")
        assert entity_1_result.source == "hybrid"
        assert entity_1_result.vector_score == 0.95
        assert entity_1_result.graph_score == 0.8

        # entity_5 only in graph - should be marked as graph
        entity_5_result = next(r for r in fused if r.entity_id == "entity_5")
        assert entity_5_result.source == "graph"

    # -------------------------------------------------------------------------
    # Score Calculation Tests
    # -------------------------------------------------------------------------

    def test_combined_score_calculation(self, hybrid_config):
        """Combined score is calculated correctly."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [create_vector_result("entity_1", similarity_score=0.8)]
        graph_results = [create_graph_result("entity_1", path_score=0.6)]

        # 50/50 weight
        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        # combined = (0.8 * 0.5) + (0.6 * 0.5) = 0.4 + 0.3 = 0.7
        assert fused[0].combined_score == pytest.approx(0.7)

    def test_weight_normalization(self, hybrid_config):
        """Weights are normalized to sum to 1."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [create_vector_result("entity_1", similarity_score=1.0)]
        graph_results = [create_graph_result("entity_1", path_score=1.0)]

        # Unnormalized weights (0.3 + 0.3 = 0.6)
        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.3,
            graph_weight=0.3,
        )

        # After normalization: both 0.5
        # combined = (1.0 * 0.5) + (1.0 * 0.5) = 1.0
        assert fused[0].combined_score == pytest.approx(1.0)

    def test_asymmetric_weights(self, hybrid_config):
        """Asymmetric weights work correctly."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [create_vector_result("entity_1", similarity_score=1.0)]
        graph_results = [create_graph_result("entity_1", path_score=1.0)]

        # 80% vector, 20% graph
        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.8,
            graph_weight=0.2,
        )

        # combined = (1.0 * 0.8) + (1.0 * 0.2) = 1.0
        assert fused[0].combined_score == pytest.approx(1.0)

    # -------------------------------------------------------------------------
    # Sorting and Deduplication Tests
    # -------------------------------------------------------------------------

    def test_results_sorted_by_combined_score(self, hybrid_config):
        """Results are sorted by combined_score descending."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [
            create_vector_result("entity_low", similarity_score=0.3),
            create_vector_result("entity_high", similarity_score=0.9),
            create_vector_result("entity_mid", similarity_score=0.6),
        ]
        graph_results = []

        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=1.0,
            graph_weight=0.0,
        )

        assert fused[0].entity_id == "entity_high"
        assert fused[1].entity_id == "entity_mid"
        assert fused[2].entity_id == "entity_low"

    def test_deduplication_keeps_highest_score(self, hybrid_config):
        """Duplicate entities keep highest score from each source."""
        fusion = ResultFusion(config=hybrid_config)

        # Same entity with different scores
        vector_results = [
            create_vector_result("entity_1", similarity_score=0.9),
            create_vector_result("entity_1", similarity_score=0.5),
        ]
        graph_results = []

        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=1.0,
            graph_weight=0.0,
        )

        # Should only have one result with highest score
        assert len(fused) == 1
        assert fused[0].vector_score == 0.9

    # -------------------------------------------------------------------------
    # Metadata Preservation Tests
    # -------------------------------------------------------------------------

    def test_content_preserved_from_vector(self, hybrid_config):
        """Content is preserved from vector results."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [
            create_vector_result("entity_1", content="Vector content")
        ]
        graph_results = [
            create_graph_result("entity_1")
        ]

        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        assert fused[0].content == "Vector content"

    def test_graph_metadata_added(self, hybrid_config):
        """Graph metadata is added to fused results."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = []
        graph_results = [
            create_graph_result(
                "entity_1",
                relationship_types=["CAUSES", "ENABLES"],
            )
        ]

        fused = fusion.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        assert "relationship_types" in fused[0].metadata
        assert fused[0].metadata["relationship_types"] == ["CAUSES", "ENABLES"]


class TestReciprocalRankFusion:
    """Test RRF fusion algorithm."""

    def test_rrf_basic(self, hybrid_config):
        """Basic RRF fusion works."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [
            create_vector_result("entity_1"),
            create_vector_result("entity_2"),
        ]
        graph_results = [
            create_graph_result("entity_2"),
            create_graph_result("entity_3"),
        ]

        fused = fusion.reciprocal_rank_fusion(
            vector_results=vector_results,
            graph_results=graph_results,
            k=60,
        )

        # entity_2 appears in both lists so should rank higher
        entity_2_result = next(r for r in fused if r.entity_id == "entity_2")
        entity_1_result = next(r for r in fused if r.entity_id == "entity_1")
        entity_3_result = next(r for r in fused if r.entity_id == "entity_3")

        assert entity_2_result.combined_score > entity_1_result.combined_score
        assert entity_2_result.combined_score > entity_3_result.combined_score

    def test_rrf_source_attribution(self, hybrid_config):
        """RRF correctly attributes sources."""
        fusion = ResultFusion(config=hybrid_config)

        vector_results = [create_vector_result("entity_1")]
        graph_results = [
            create_graph_result("entity_1"),
            create_graph_result("entity_2"),
        ]

        fused = fusion.reciprocal_rank_fusion(
            vector_results=vector_results,
            graph_results=graph_results,
        )

        entity_1 = next(r for r in fused if r.entity_id == "entity_1")
        entity_2 = next(r for r in fused if r.entity_id == "entity_2")

        assert entity_1.source == "hybrid"
        assert entity_2.source == "graph"

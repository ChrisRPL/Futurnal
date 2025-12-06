"""Tests for SchemaVersionCompatibility.

Tests schema version compatibility checking and drift handling.
"""

import pytest

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import SchemaCompatibilityError
from futurnal.search.hybrid.schema_compat import SchemaVersionCompatibility
from futurnal.search.hybrid.types import VectorSearchResult

from tests.search.hybrid.conftest import create_vector_result


class TestSchemaVersionCompatibility:
    """Test SchemaVersionCompatibility functionality."""

    def test_init(self, hybrid_config):
        """Compatibility checker initializes correctly."""
        compat = SchemaVersionCompatibility(config=hybrid_config)
        assert compat.config == hybrid_config

    # -------------------------------------------------------------------------
    # Compatibility Level Tests
    # -------------------------------------------------------------------------

    def test_full_compatibility_same_version(self, hybrid_config):
        """0 version diff = full compatibility."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        result = compat.check_compatibility(
            embedding_version=5,
            current_version=5,
        )

        assert result.compatible
        assert result.score_factor == 1.0
        assert result.drift_level == "none"
        assert not result.reembedding_required
        assert result.version_diff == 0

    def test_minor_drift_one_version(self, hybrid_config):
        """1 version diff = minor drift (95% score)."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        result = compat.check_compatibility(
            embedding_version=4,
            current_version=5,
        )

        assert result.compatible
        assert result.score_factor == hybrid_config.minor_drift_penalty  # 0.95
        assert result.drift_level == "minor"
        assert not result.reembedding_required
        assert result.version_diff == 1

    def test_moderate_drift_within_threshold(self, hybrid_config):
        """2-3 version diff = moderate drift (85% score)."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        # Test 2 versions diff
        result = compat.check_compatibility(
            embedding_version=3,
            current_version=5,
        )

        assert result.compatible
        assert result.score_factor == hybrid_config.moderate_drift_penalty  # 0.85
        assert result.drift_level == "moderate"
        assert result.transform_required
        assert result.version_diff == 2

        # Test 3 versions diff (at threshold)
        result = compat.check_compatibility(
            embedding_version=2,
            current_version=5,
        )

        assert result.compatible
        assert result.drift_level == "moderate"
        assert result.version_diff == 3

    def test_severe_drift_beyond_threshold(self, hybrid_config):
        """4+ version diff = severe drift (requires re-embedding)."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        result = compat.check_compatibility(
            embedding_version=1,
            current_version=5,
        )

        assert not result.compatible
        assert result.score_factor == 0.0
        assert result.drift_level == "severe"
        assert result.reembedding_required
        assert result.version_diff == 4

    def test_future_version_handling(self, hybrid_config):
        """Embedding from future version is compatible."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        result = compat.check_compatibility(
            embedding_version=6,  # Future version
            current_version=5,
        )

        # Future versions should be compatible (shouldn't happen but handle gracefully)
        assert result.compatible
        assert result.score_factor == 1.0
        assert result.version_diff == -1

    # -------------------------------------------------------------------------
    # Result Filtering Tests
    # -------------------------------------------------------------------------

    def test_filter_compatible_results(self, hybrid_config):
        """Compatible results are kept, incompatible filtered out."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        results = [
            create_vector_result("entity_1", schema_version=5),  # Same version
            create_vector_result("entity_2", schema_version=4),  # Minor drift
            create_vector_result("entity_3", schema_version=1),  # Severe drift
        ]

        filtered = compat.filter_compatible_results(
            results=results,
            current_version=5,
        )

        # Only entity_1 and entity_2 should remain
        assert len(filtered) == 2
        assert filtered[0].entity_id == "entity_1"
        assert filtered[1].entity_id == "entity_2"

    def test_filter_applies_score_adjustment(self, hybrid_config):
        """Score adjustment is applied based on drift."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        results = [
            create_vector_result("entity_1", schema_version=5, similarity_score=1.0),
            create_vector_result("entity_2", schema_version=4, similarity_score=1.0),
            create_vector_result("entity_3", schema_version=3, similarity_score=1.0),
        ]

        filtered = compat.filter_compatible_results(
            results=results,
            current_version=5,
            apply_score_adjustment=True,
        )

        # Check score adjustments
        assert filtered[0].similarity_score == 1.0  # No drift
        assert filtered[1].similarity_score == pytest.approx(0.95)  # Minor drift
        assert filtered[2].similarity_score == pytest.approx(0.85)  # Moderate drift

    def test_filter_without_score_adjustment(self, hybrid_config):
        """Scores unchanged when adjustment disabled."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        results = [
            create_vector_result("entity_1", schema_version=4, similarity_score=1.0),
        ]

        filtered = compat.filter_compatible_results(
            results=results,
            current_version=5,
            apply_score_adjustment=False,
        )

        # Score should be unchanged
        assert filtered[0].similarity_score == 1.0

    # -------------------------------------------------------------------------
    # Utility Tests
    # -------------------------------------------------------------------------

    def test_get_minimum_compatible_version(self, hybrid_config):
        """Minimum compatible version calculation."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        # With threshold=3 and current=5, min should be 2
        min_version = compat.get_minimum_compatible_version(current_version=5)
        assert min_version == 2

        # Never goes below 1
        min_version = compat.get_minimum_compatible_version(current_version=2)
        assert min_version == 1

    def test_get_current_schema_version_fallback(self, hybrid_config):
        """Falls back to version 1 when no manager available."""
        compat = SchemaVersionCompatibility(config=hybrid_config)

        version = compat.get_current_schema_version()
        assert version == 1


class TestSchemaVersionCompatibilityCustomConfig:
    """Test with custom configuration values."""

    def test_custom_drift_threshold(self, custom_hybrid_config):
        """Custom drift threshold affects compatibility."""
        # custom_hybrid_config has threshold=2
        compat = SchemaVersionCompatibility(config=custom_hybrid_config)

        # 3 diff should now be severe (threshold is 2)
        result = compat.check_compatibility(
            embedding_version=2,
            current_version=5,
        )

        assert not result.compatible
        assert result.drift_level == "severe"
        assert result.reembedding_required

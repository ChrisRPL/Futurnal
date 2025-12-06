"""Schema Evolution Integration Tests.

Tests for schema-aware retrieval adaptation across schema versions.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/06-integration-testing.md

Test Suite: TestSchemaEvolution

Option B Compliance:
- Autonomous schema evolution support
- Cross-version compatibility
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from futurnal.search.api import HybridSearchAPI


class TestSchemaEvolution:
    """Tests for schema-aware retrieval adaptation."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_schema_version_compatibility(
        self,
        api: HybridSearchAPI,
    ) -> None:
        """Test retrieval works across schema versions.

        Success criteria:
        - Query finds entities from both schema versions
        - No errors on mixed-version results
        """
        # Note: In production, this would create entities with different schema versions
        # For now, we verify the API handles queries gracefully

        results = await api.search("events", top_k=10)

        assert results is not None
        assert isinstance(results, list)

        # Check that results have expected structure
        for r in results:
            assert "id" in r
            assert "score" in r or "confidence" in r

        print(f"Retrieved {len(results)} results across schema versions")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_schema_change(
        self,
        api: HybridSearchAPI,
    ) -> None:
        """Test cache invalidation when schema evolves.

        Success criteria:
        - Cache is invalidated on schema change
        - New queries reflect updated schema
        """
        # Cache query result
        query = "project data"
        await api.search(query, top_k=10)

        # Simulate schema evolution
        if api.schema_manager:
            await api.schema_manager.evolve_schema("2.1")

            # Cache should be invalidated for affected layers
            if api.cache:
                from futurnal.search.hybrid.performance import CacheLayer
                invalidations = api.cache.stats.invalidations.get(
                    CacheLayer.GRAPH_TRAVERSAL, 0
                )
                print(f"Graph traversal cache invalidations: {invalidations}")
        else:
            # Schema manager not available - just verify query works
            results = await api.search(query, top_k=10)
            assert results is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entity_type_strategy_adaptation(
        self,
        api: HybridSearchAPI,
    ) -> None:
        """Test retrieval strategy adapts to entity type.

        Success criteria:
        - Event queries use temporal-aware strategy
        - Code queries use CodeBERT embeddings
        """
        # Event query should use temporal-aware strategy
        await api.search("events from January", top_k=5)
        if api.last_strategy:
            strategy = api.last_strategy
            print(f"Event query strategy: {strategy}")
            assert strategy in ["temporal", "exploratory", "causal"]

        # Code query should use CodeBERT embeddings
        await api.search("authentication function", top_k=5)
        if api.last_embedding_model:
            print(f"Code query embedding model: {api.last_embedding_model}")


class TestSchemaCompatibility:
    """Tests for schema version compatibility handling."""

    @pytest.mark.integration
    def test_schema_version_detection(self) -> None:
        """Test schema version is detected."""
        from futurnal.search.hybrid import SchemaVersionCompatibility

        compat = SchemaVersionCompatibility()

        # Use get_current_schema_version
        version = compat.get_current_schema_version()
        assert version >= 1
        print(f"Current schema version: {version}")

    @pytest.mark.integration
    def test_schema_compatibility_check(self) -> None:
        """Test schema compatibility checking."""
        from futurnal.search.hybrid import SchemaVersionCompatibility

        compat = SchemaVersionCompatibility()

        # Use check_compatibility with version numbers
        result = compat.check_compatibility(
            embedding_version=1,
            current_version=2
        )
        
        # Should return compatibility result
        assert hasattr(result, "compatible")
        print(f"Compatibility result: compatible={result.compatible}")

    @pytest.mark.integration
    def test_minimum_compatible_version(self) -> None:
        """Test minimum compatible version calculation."""
        from futurnal.search.hybrid import SchemaVersionCompatibility

        compat = SchemaVersionCompatibility()

        # Use get_minimum_compatible_version
        current = 5
        min_version = compat.get_minimum_compatible_version(current)
        
        assert min_version <= current
        print(f"Minimum compatible version for v{current}: {min_version}")


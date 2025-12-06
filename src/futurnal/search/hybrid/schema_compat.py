"""Schema Version Compatibility Handler.

Handles retrieval across schema versions with drift penalties and
re-embedding triggers.

Compatibility Rules (from production plan):
- 0 diff: Full compatibility (100% score)
- 1 diff: Minor drift (95% score)
- 2-3 diff: Moderate drift (85% score)
- 4+ diff: Requires re-embedding

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Schema evolves autonomously (version tracking, not hardcoded)
- Re-embedding triggered on significant drift
- Graceful degradation for older embeddings
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import SchemaCompatibilityError
from futurnal.search.hybrid.types import SchemaCompatibilityResult, VectorSearchResult

if TYPE_CHECKING:
    from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
    from futurnal.pkg.schema.migration import SchemaVersionManager

logger = logging.getLogger(__name__)


class SchemaVersionCompatibility:
    """Handles retrieval across schema versions.

    Ensures embeddings from older schema versions are still usable
    while applying appropriate score penalties for drift.

    Compatibility Levels:
    - Full (0 diff): 100% score factor, no transform needed
    - Minor (1 diff): 95% score factor, acceptable drift
    - Moderate (2-3 diff): 85% score factor, may need transform
    - Severe (4+ diff): 0% - requires re-embedding

    The checker applies score penalties to results based on schema
    version drift and can trigger background re-embedding for
    severely outdated embeddings.

    Example:
        >>> from futurnal.embeddings import SchemaVersionedEmbeddingStore
        >>> from futurnal.search.hybrid import SchemaVersionCompatibility

        >>> store = SchemaVersionedEmbeddingStore(config, neo4j_driver)
        >>> compat = SchemaVersionCompatibility(embedding_store=store)

        >>> # Check compatibility of an embedding
        >>> result = compat.check_compatibility(
        ...     embedding_version=3,
        ...     current_version=5,
        ... )
        >>> print(f"Compatible: {result.compatible}, Score factor: {result.score_factor}")
        # Compatible: True, Score factor: 0.85 (moderate drift)

        >>> # Filter results to only compatible embeddings
        >>> filtered = compat.filter_compatible_results(results, current_version=5)
    """

    def __init__(
        self,
        schema_manager: Optional["SchemaVersionManager"] = None,
        embedding_store: Optional["SchemaVersionedEmbeddingStore"] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize schema version compatibility checker.

        Args:
            schema_manager: PKG schema version manager (for version queries)
            embedding_store: Schema-versioned embedding store (for re-embedding)
            config: Hybrid search configuration
        """
        self._schema_manager = schema_manager
        self._embedding_store = embedding_store
        self._config = config or HybridSearchConfig()

        logger.info(
            f"Initialized SchemaVersionCompatibility with "
            f"drift_threshold={self._config.schema_drift_threshold}, "
            f"minor_penalty={self._config.minor_drift_penalty}, "
            f"moderate_penalty={self._config.moderate_drift_penalty}"
        )

    @property
    def config(self) -> HybridSearchConfig:
        """Get the configuration."""
        return self._config

    def check_compatibility(
        self,
        embedding_version: int,
        current_version: int,
    ) -> SchemaCompatibilityResult:
        """Check if embedding is compatible with current schema.

        Determines compatibility based on version difference and
        returns appropriate score factor and drift level.

        Args:
            embedding_version: Schema version when embedding was created
            current_version: Current PKG schema version

        Returns:
            SchemaCompatibilityResult with compatibility info
        """
        version_diff = current_version - embedding_version

        # Handle edge cases
        if version_diff < 0:
            # Embedding from future version (shouldn't happen)
            logger.warning(
                f"Embedding version {embedding_version} is newer than "
                f"current version {current_version}"
            )
            return SchemaCompatibilityResult(
                compatible=True,
                score_factor=1.0,
                drift_level="none",
                version_diff=version_diff,
            )

        # Full compatibility (same version)
        if version_diff == 0:
            return SchemaCompatibilityResult(
                compatible=True,
                score_factor=1.0,
                transform_required=False,
                reembedding_required=False,
                drift_level="none",
                version_diff=version_diff,
            )

        # Minor drift (1 version difference)
        if version_diff == 1:
            return SchemaCompatibilityResult(
                compatible=True,
                score_factor=self._config.minor_drift_penalty,
                transform_required=False,
                reembedding_required=False,
                drift_level="minor",
                version_diff=version_diff,
            )

        # Moderate drift (2-3 version difference)
        if version_diff <= self._config.schema_drift_threshold:
            return SchemaCompatibilityResult(
                compatible=True,
                score_factor=self._config.moderate_drift_penalty,
                transform_required=True,
                reembedding_required=False,
                drift_level="moderate",
                version_diff=version_diff,
            )

        # Severe drift (beyond threshold)
        return SchemaCompatibilityResult(
            compatible=False,
            score_factor=0.0,
            transform_required=False,
            reembedding_required=True,
            drift_level="severe",
            version_diff=version_diff,
        )

    def filter_compatible_results(
        self,
        results: List[VectorSearchResult],
        current_version: int,
        apply_score_adjustment: bool = True,
    ) -> List[VectorSearchResult]:
        """Filter results to only compatible schema versions.

        Removes results that require re-embedding and optionally
        adjusts scores based on drift level.

        Args:
            results: Vector search results to filter
            current_version: Current PKG schema version
            apply_score_adjustment: Whether to apply drift penalties

        Returns:
            Filtered list of compatible results
        """
        compatible_results = []
        entities_needing_reembedding = []

        for result in results:
            compat = self.check_compatibility(
                result.schema_version,
                current_version,
            )

            if not compat.compatible:
                # Track for potential re-embedding
                entities_needing_reembedding.append(result.entity_id)
                logger.debug(
                    f"Filtered out {result.entity_id}: schema version "
                    f"{result.schema_version} incompatible with {current_version}"
                )
                continue

            # Apply score adjustment if enabled
            if apply_score_adjustment and compat.score_factor < 1.0:
                result.similarity_score *= compat.score_factor
                logger.debug(
                    f"Adjusted score for {result.entity_id}: "
                    f"drift={compat.drift_level}, factor={compat.score_factor}"
                )

            compatible_results.append(result)

        # Trigger background re-embedding for severely outdated entities
        if entities_needing_reembedding:
            logger.info(
                f"Found {len(entities_needing_reembedding)} entities "
                f"requiring re-embedding due to schema drift"
            )
            self._queue_for_reembedding(entities_needing_reembedding)

        return compatible_results

    def get_minimum_compatible_version(self, current_version: int) -> int:
        """Get minimum compatible schema version.

        Returns the oldest schema version that is still compatible
        (within drift threshold).

        Args:
            current_version: Current PKG schema version

        Returns:
            Minimum compatible schema version
        """
        return max(1, current_version - self._config.schema_drift_threshold)

    def trigger_background_reembedding(
        self,
        entity_ids: List[str],
        reason: str = "schema_evolution",
    ) -> None:
        """Trigger background re-embedding for outdated entities.

        Queues entities for re-embedding without blocking the
        current search operation.

        Args:
            entity_ids: List of entity IDs to re-embed
            reason: Reason for re-embedding (for tracking)

        Raises:
            SchemaCompatibilityError: If re-embedding queue fails
        """
        if not entity_ids:
            return

        if self._embedding_store is None:
            logger.warning(
                f"Cannot trigger re-embedding: no embedding store configured"
            )
            return

        try:
            self._queue_for_reembedding(entity_ids, reason)
            logger.info(
                f"Queued {len(entity_ids)} entities for background "
                f"re-embedding (reason: {reason})"
            )
        except Exception as e:
            raise SchemaCompatibilityError(
                f"Failed to queue entities for re-embedding: {e}"
            ) from e

    def _queue_for_reembedding(
        self,
        entity_ids: List[str],
        reason: str = "schema_evolution",
    ) -> None:
        """Internal method to queue entities for re-embedding.

        Args:
            entity_ids: List of entity IDs
            reason: Reason for re-embedding
        """
        if self._embedding_store is None:
            return

        # Use embedding store's re-embedding queue if available
        try:
            # Check if embedding store has mark_for_reembedding method
            if hasattr(self._embedding_store, "mark_for_reembedding"):
                self._embedding_store.mark_for_reembedding(
                    entity_ids=entity_ids,
                    reason=reason,
                )
            else:
                # Log for manual handling
                logger.info(
                    f"Re-embedding needed for {len(entity_ids)} entities: {reason}"
                )
        except Exception as e:
            logger.error(f"Failed to queue re-embedding: {e}")

    def get_current_schema_version(self) -> int:
        """Get current schema version from PKG.

        Returns:
            Current schema version, or 1 if unavailable

        Raises:
            SchemaCompatibilityError: If schema query fails
        """
        if self._schema_manager is not None:
            try:
                version_node = self._schema_manager.get_current_version()
                if version_node is not None:
                    return version_node.version
            except Exception as e:
                raise SchemaCompatibilityError(
                    f"Failed to get current schema version: {e}"
                ) from e

        if self._embedding_store is not None:
            try:
                # Use embedding store's cached version if available
                if hasattr(self._embedding_store, "_get_current_schema_version"):
                    return self._embedding_store._get_current_schema_version()
            except Exception:
                pass

        # Default to version 1 if no manager available
        logger.warning("No schema manager available, using default version 1")
        return 1

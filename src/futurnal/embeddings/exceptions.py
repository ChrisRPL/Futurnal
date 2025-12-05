"""Custom exceptions for the embedding service.

This module defines all exceptions specific to the temporal-aware embedding
service, enabling precise error handling and debugging.
"""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""

    pass


class ModelLoadError(EmbeddingError):
    """Raised when an embedding model fails to load.

    Common causes:
    - Model not found in cache or remote
    - Insufficient memory for model
    - Incompatible model format
    """

    pass


class ModelNotFoundError(EmbeddingError):
    """Raised when a requested model ID is not registered.

    This indicates a configuration error - the model ID should be
    registered in the ModelManager configuration.
    """

    pass


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails.

    Common causes:
    - Model inference error
    - Invalid input text
    - Resource exhaustion
    """

    pass


class FusionError(EmbeddingError):
    """Raised when embedding fusion fails.

    Common causes:
    - Dimension mismatch that cannot be resolved
    - Invalid fusion weights
    - Missing required embeddings
    """

    pass


class TemporalContextError(EmbeddingError):
    """Raised when temporal context is invalid or missing.

    Option B Compliance:
    - Temporal context is REQUIRED for event embeddings
    - This error indicates a violation of temporal-first design
    """

    pass


class StorageError(EmbeddingError):
    """Raised when embedding storage operations fail.

    Common causes:
    - ChromaDB connection error
    - Collection not found
    - Write permission denied
    """

    pass


class RoutingError(EmbeddingError):
    """Raised when model routing fails.

    Common causes:
    - Unsupported entity type
    - No models available for entity type
    - Router configuration error
    """

    pass


class BatchProcessingError(EmbeddingError):
    """Raised when batch processing fails.

    Contains partial results for successfully processed items,
    allowing recovery and continuation of processing.

    Attributes:
        message: Error description
        successful_results: List of results for successfully processed items
        failed_indices: Indices of failed items in the original batch
        errors: List of (index, exception) tuples for failed items
    """

    def __init__(
        self,
        message: str,
        successful_results: list = None,
        failed_indices: list = None,
        errors: list = None,
    ) -> None:
        """Initialize batch processing error.

        Args:
            message: Error description
            successful_results: List of results for successfully processed items
            failed_indices: Indices of failed items in the original batch
            errors: List of (index, exception) tuples for failed items
        """
        super().__init__(message)
        self.successful_results = successful_results or []
        self.failed_indices = failed_indices or []
        self.errors = errors or []

    @property
    def partial_success(self) -> bool:
        """Check if some items were successfully processed."""
        return len(self.successful_results) > 0

    @property
    def failure_count(self) -> int:
        """Get number of failed items."""
        return len(self.failed_indices)


class SchemaVersionError(EmbeddingError):
    """Raised when schema version operations fail.

    Common causes:
    - PKG not connected (Neo4j driver unavailable)
    - Schema version not found in PKG
    - Schema hash computation failed
    - Invalid schema version number

    Production Plan Reference:
    docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
    """

    pass


class ReembeddingError(EmbeddingError):
    """Raised when re-embedding operations fail.

    Common causes:
    - Entity not found in PKG
    - Embedding generation failed during re-embedding
    - Storage update failed
    - Batch processing interrupted

    Attributes:
        entity_id: ID of the entity that failed (if applicable)
        cause: Original exception that caused the failure

    Production Plan Reference:
    docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
    """

    def __init__(
        self,
        message: str,
        entity_id: str = None,
        cause: Exception = None,
    ) -> None:
        """Initialize re-embedding error.

        Args:
            message: Error description
            entity_id: ID of the entity that failed (if applicable)
            cause: Original exception that caused the failure
        """
        super().__init__(message)
        self.entity_id = entity_id
        self.cause = cause

    def __str__(self) -> str:
        """Format error message with entity context."""
        base_msg = super().__str__()
        if self.entity_id:
            base_msg = f"{base_msg} (entity_id={self.entity_id})"
        if self.cause:
            base_msg = f"{base_msg}, caused by: {self.cause}"
        return base_msg

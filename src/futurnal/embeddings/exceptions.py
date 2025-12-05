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

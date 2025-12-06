"""Hybrid Search Exceptions.

Custom exceptions for schema-aware hybrid retrieval operations.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md
"""

from __future__ import annotations


class HybridSearchError(Exception):
    """Base exception for hybrid search operations."""

    pass


class QueryRoutingError(HybridSearchError):
    """Error during query embedding routing.

    Raised when:
    - Invalid query type specified
    - Model not available for query type
    - Embedding generation fails
    """

    pass


class SchemaCompatibilityError(HybridSearchError):
    """Error related to schema version compatibility.

    Raised when:
    - Schema version mismatch exceeds threshold
    - Schema hash computation fails
    - Re-embedding required but not triggered
    """

    pass


class VectorSearchError(HybridSearchError):
    """Error during vector similarity search.

    Raised when:
    - Embedding store query fails
    - No results found for query
    - Timeout during vector search
    """

    pass


class GraphExpansionError(HybridSearchError):
    """Error during graph expansion from seed entities.

    Raised when:
    - PKG query fails
    - Temporal/causal expansion fails
    - Graph traversal timeout
    """

    pass


class FusionError(HybridSearchError):
    """Error during result fusion.

    Raised when:
    - Weight normalization fails
    - Score combination fails
    - Result deduplication fails
    """

    pass


class InvalidHybridQueryError(HybridSearchError):
    """Invalid hybrid search query parameters.

    Raised when:
    - Invalid intent specified
    - Invalid weight values (not 0-1)
    - Invalid top_k value
    - Missing required parameters
    """

    pass

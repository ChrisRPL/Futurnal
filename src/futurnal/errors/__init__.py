"""Centralized error definitions for Futurnal.

This module provides a unified error hierarchy and user-friendly error handling
for production deployment.

Usage:
    from futurnal.errors import (
        FuturnalError,
        SearchError,
        ConsentRequiredError,
        handle_error,
    )

    try:
        result = await search_api.search(query)
    except FuturnalError as e:
        user_message = handle_error(e)
        print(user_message)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from futurnal.errors.user_messages import (
    get_user_message,
    get_recovery_suggestion,
    format_error_for_user,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Base Error
# =============================================================================


class FuturnalError(Exception):
    """Base exception for all Futurnal errors.

    Attributes:
        code: Error code for categorization
        user_message: User-friendly message (optional override)
        recoverable: Whether the error is potentially recoverable
        details: Additional error details for debugging
    """

    code: str = "FUTURNAL_ERROR"
    default_message: str = "An unexpected error occurred"
    recoverable: bool = True

    def __init__(
        self,
        message: str | None = None,
        *,
        user_message: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.message = message or self.default_message
        self._user_message = user_message
        self.details = details or {}
        super().__init__(self.message)

    @property
    def user_message(self) -> str:
        """Get user-friendly message."""
        if self._user_message:
            return self._user_message
        return get_user_message(self)

    @property
    def recovery_suggestion(self) -> str:
        """Get recovery suggestion."""
        return get_recovery_suggestion(self)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "recoverable": self.recoverable,
            "details": self.details,
        }


# =============================================================================
# Search Errors
# =============================================================================


class SearchError(FuturnalError):
    """Base error for search operations."""

    code = "SEARCH_ERROR"
    default_message = "Search operation failed"


class VectorSearchError(SearchError):
    """Vector embedding search failed."""

    code = "VECTOR_SEARCH_ERROR"
    default_message = "Vector search failed"


class GraphSearchError(SearchError):
    """Graph-based search failed."""

    code = "GRAPH_SEARCH_ERROR"
    default_message = "Graph search failed"


class TemporalSearchError(SearchError):
    """Temporal query failed."""

    code = "TEMPORAL_SEARCH_ERROR"
    default_message = "Temporal search failed"


class CausalSearchError(SearchError):
    """Causal chain search failed."""

    code = "CAUSAL_SEARCH_ERROR"
    default_message = "Causal search failed"


class SearchTimeoutError(SearchError):
    """Search operation timed out."""

    code = "SEARCH_TIMEOUT"
    default_message = "Search timed out"
    recoverable = True


# =============================================================================
# Chat Errors
# =============================================================================


class ChatError(FuturnalError):
    """Base error for chat operations."""

    code = "CHAT_ERROR"
    default_message = "Chat operation failed"


class SessionNotFoundError(ChatError):
    """Chat session not found."""

    code = "SESSION_NOT_FOUND"
    default_message = "Chat session not found"


class ContextRetrievalError(ChatError):
    """Failed to retrieve context for chat."""

    code = "CONTEXT_RETRIEVAL_ERROR"
    default_message = "Failed to retrieve context"


class GenerationError(ChatError):
    """LLM generation failed."""

    code = "GENERATION_ERROR"
    default_message = "Failed to generate response"


# =============================================================================
# Connection Errors
# =============================================================================


class ConnectionError(FuturnalError):
    """Base error for connection issues."""

    code = "CONNECTION_ERROR"
    default_message = "Connection failed"
    recoverable = True


class OllamaConnectionError(ConnectionError):
    """Cannot connect to Ollama."""

    code = "OLLAMA_CONNECTION_ERROR"
    default_message = "Cannot connect to Ollama"


class PKGConnectionError(ConnectionError):
    """Cannot connect to knowledge graph database."""

    code = "PKG_CONNECTION_ERROR"
    default_message = "Cannot connect to knowledge graph"


class EmbeddingServiceError(ConnectionError):
    """Embedding service unavailable."""

    code = "EMBEDDING_SERVICE_ERROR"
    default_message = "Embedding service unavailable"


# =============================================================================
# Privacy & Consent Errors
# =============================================================================


class PrivacyError(FuturnalError):
    """Base error for privacy operations."""

    code = "PRIVACY_ERROR"
    default_message = "Privacy operation failed"


class ConsentRequiredError(PrivacyError):
    """Operation requires consent that hasn't been granted."""

    code = "CONSENT_REQUIRED"
    default_message = "Consent required for this operation"
    recoverable = True

    def __init__(
        self,
        source: str,
        scope: str,
        *,
        message: str | None = None,
    ) -> None:
        self.source = source
        self.scope = scope
        super().__init__(
            message or f"Consent required for {source} ({scope})",
            details={"source": source, "scope": scope},
        )


class ConsentRevokedError(PrivacyError):
    """Consent was revoked for this operation."""

    code = "CONSENT_REVOKED"
    default_message = "Consent has been revoked"


class AuditError(PrivacyError):
    """Audit logging failed."""

    code = "AUDIT_ERROR"
    default_message = "Audit logging failed"
    recoverable = False


# =============================================================================
# Data Source Errors
# =============================================================================


class SourceError(FuturnalError):
    """Base error for data source operations."""

    code = "SOURCE_ERROR"
    default_message = "Data source operation failed"


class SourceNotFoundError(SourceError):
    """Data source not found."""

    code = "SOURCE_NOT_FOUND"
    default_message = "Data source not found"


class SourceConnectionError(SourceError):
    """Cannot connect to data source."""

    code = "SOURCE_CONNECTION_ERROR"
    default_message = "Cannot connect to data source"
    recoverable = True


class IngestionError(SourceError):
    """Ingestion failed."""

    code = "INGESTION_ERROR"
    default_message = "Ingestion failed"


class QuarantineError(SourceError):
    """File quarantined due to processing failure."""

    code = "QUARANTINE_ERROR"
    default_message = "File moved to quarantine"


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(FuturnalError):
    """Base error for configuration issues."""

    code = "CONFIGURATION_ERROR"
    default_message = "Configuration error"
    recoverable = True


class InvalidConfigError(ConfigurationError):
    """Configuration is invalid."""

    code = "INVALID_CONFIG"
    default_message = "Invalid configuration"


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    code = "MISSING_CONFIG"
    default_message = "Missing required configuration"


# =============================================================================
# Processing Errors
# =============================================================================


class ProcessingError(FuturnalError):
    """Base error for document processing."""

    code = "PROCESSING_ERROR"
    default_message = "Document processing failed"


class NormalizationError(ProcessingError):
    """Document normalization failed."""

    code = "NORMALIZATION_ERROR"
    default_message = "Document normalization failed"


class ExtractionError(ProcessingError):
    """Entity/relationship extraction failed."""

    code = "EXTRACTION_ERROR"
    default_message = "Extraction failed"


class EmbeddingError(ProcessingError):
    """Embedding generation failed."""

    code = "EMBEDDING_ERROR"
    default_message = "Embedding generation failed"


# =============================================================================
# Error Handler
# =============================================================================


def handle_error(error: Exception) -> str:
    """Handle an error and return a user-friendly message.

    Args:
        error: The exception to handle

    Returns:
        User-friendly error message with recovery suggestion
    """
    return format_error_for_user(error)


def is_recoverable(error: Exception) -> bool:
    """Check if an error is potentially recoverable.

    Args:
        error: The exception to check

    Returns:
        True if the error is recoverable
    """
    if isinstance(error, FuturnalError):
        return error.recoverable
    return False


# =============================================================================
# Re-exports from existing modules (for backwards compatibility)
# =============================================================================

# Re-export existing errors to maintain compatibility
try:
    from futurnal.search.hybrid.exceptions import HybridSearchError
except ImportError:
    pass

try:
    from futurnal.search.temporal.exceptions import (
        TemporalSearchError as _TemporalSearchError,
    )
except ImportError:
    pass

try:
    from futurnal.search.causal.exceptions import CausalSearchError as _CausalSearchError
except ImportError:
    pass


__all__ = [
    # Base
    "FuturnalError",
    # Search
    "SearchError",
    "VectorSearchError",
    "GraphSearchError",
    "TemporalSearchError",
    "CausalSearchError",
    "SearchTimeoutError",
    # Chat
    "ChatError",
    "SessionNotFoundError",
    "ContextRetrievalError",
    "GenerationError",
    # Connection
    "ConnectionError",
    "OllamaConnectionError",
    "PKGConnectionError",
    "EmbeddingServiceError",
    # Privacy
    "PrivacyError",
    "ConsentRequiredError",
    "ConsentRevokedError",
    "AuditError",
    # Source
    "SourceError",
    "SourceNotFoundError",
    "SourceConnectionError",
    "IngestionError",
    "QuarantineError",
    # Configuration
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    # Processing
    "ProcessingError",
    "NormalizationError",
    "ExtractionError",
    "EmbeddingError",
    # Handlers
    "handle_error",
    "is_recoverable",
]

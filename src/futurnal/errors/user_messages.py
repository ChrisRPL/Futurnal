"""User-friendly error messages for Futurnal.

This module provides human-readable error messages and recovery suggestions
for all error types, ensuring users never see raw technical errors.

Privacy Note:
- Error messages NEVER include sensitive content
- File paths are generalized
- Query content is never exposed
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# =============================================================================
# Error Message Catalog
# =============================================================================

ERROR_MESSAGES: dict[str, str] = {
    # Search errors
    "SEARCH_ERROR": "We couldn't complete your search. Please try again.",
    "VECTOR_SEARCH_ERROR": "The semantic search couldn't find results. Try rephrasing your query.",
    "GRAPH_SEARCH_ERROR": "The knowledge graph search encountered an issue. Please try again.",
    "TEMPORAL_SEARCH_ERROR": "We couldn't process your time-based query. Check the date format.",
    "CAUSAL_SEARCH_ERROR": "We couldn't find causal relationships. Try a more specific query.",
    "SEARCH_TIMEOUT": "Your search took too long. Try a simpler query or fewer results.",
    # Chat errors
    "CHAT_ERROR": "The chat service encountered an issue. Please try again.",
    "SESSION_NOT_FOUND": "This chat session wasn't found. Starting a new session.",
    "CONTEXT_RETRIEVAL_ERROR": "We couldn't find relevant context for your question.",
    "GENERATION_ERROR": "We couldn't generate a response. Please try again.",
    # Connection errors
    "CONNECTION_ERROR": "A connection issue occurred. Check your services.",
    "OLLAMA_CONNECTION_ERROR": "Cannot connect to Ollama. Make sure it's running.",
    "PKG_CONNECTION_ERROR": "Cannot connect to the knowledge graph database.",
    "EMBEDDING_SERVICE_ERROR": "The embedding service is unavailable.",
    # Privacy errors
    "PRIVACY_ERROR": "A privacy-related issue occurred.",
    "CONSENT_REQUIRED": "This operation requires your consent.",
    "CONSENT_REVOKED": "Consent for this data source has been revoked.",
    "AUDIT_ERROR": "Audit logging failed. This is a critical error.",
    # Source errors
    "SOURCE_ERROR": "A data source issue occurred.",
    "SOURCE_NOT_FOUND": "The specified data source wasn't found.",
    "SOURCE_CONNECTION_ERROR": "Cannot connect to the data source.",
    "INGESTION_ERROR": "Failed to process the data. Check the file format.",
    "QUARANTINE_ERROR": "This file couldn't be processed and was quarantined.",
    # Configuration errors
    "CONFIGURATION_ERROR": "There's a configuration issue.",
    "INVALID_CONFIG": "The configuration is invalid. Check settings.",
    "MISSING_CONFIG": "Required configuration is missing.",
    # Processing errors
    "PROCESSING_ERROR": "Failed to process the document.",
    "NORMALIZATION_ERROR": "The document couldn't be normalized.",
    "EXTRACTION_ERROR": "Couldn't extract information from the document.",
    "EMBEDDING_ERROR": "Couldn't generate embeddings for the content.",
    # Generic
    "FUTURNAL_ERROR": "An unexpected error occurred. Please try again.",
    "UNKNOWN_ERROR": "Something went wrong. Please try again.",
}


# =============================================================================
# Recovery Suggestions
# =============================================================================

RECOVERY_SUGGESTIONS: dict[str, str] = {
    # Search errors
    "SEARCH_ERROR": "Try rephrasing your query or using simpler terms.",
    "VECTOR_SEARCH_ERROR": "Try different keywords or check if content has been indexed.",
    "GRAPH_SEARCH_ERROR": "Wait a moment and try again. If the issue persists, restart Futurnal.",
    "TEMPORAL_SEARCH_ERROR": "Use dates like 'yesterday', 'last week', or 'YYYY-MM-DD' format.",
    "CAUSAL_SEARCH_ERROR": "Try asking about specific events or relationships.",
    "SEARCH_TIMEOUT": "Reduce the number of results (--limit) or narrow your search.",
    # Chat errors
    "CHAT_ERROR": "Try starting a new chat session with /new.",
    "SESSION_NOT_FOUND": "Your previous session expired. Starting fresh.",
    "CONTEXT_RETRIEVAL_ERROR": "Try rephrasing your question or being more specific.",
    "GENERATION_ERROR": "Check Ollama status with: ollama ps. Restart if needed.",
    # Connection errors
    "CONNECTION_ERROR": "Check your network and service status.",
    "OLLAMA_CONNECTION_ERROR": "Run 'ollama serve' to start Ollama, then retry.",
    "PKG_CONNECTION_ERROR": "Restart Futurnal to reconnect to the database.",
    "EMBEDDING_SERVICE_ERROR": "Wait a moment and retry. The service may be loading.",
    # Privacy errors
    "PRIVACY_ERROR": "Check your privacy settings: futurnal privacy consent list",
    "CONSENT_REQUIRED": "Grant consent with: futurnal privacy consent grant --source <name>",
    "CONSENT_REVOKED": "Re-grant consent if you want to access this source.",
    "AUDIT_ERROR": "This is critical. Check disk space and file permissions.",
    # Source errors
    "SOURCE_ERROR": "Check source status: futurnal sources list",
    "SOURCE_NOT_FOUND": "Add the source first: futurnal sources <type> add <path>",
    "SOURCE_CONNECTION_ERROR": "Verify the source is accessible and credentials are correct.",
    "INGESTION_ERROR": "Check if the file format is supported. View quarantine for details.",
    "QUARANTINE_ERROR": "View details: futurnal sources quarantine info <file_id>",
    # Configuration errors
    "CONFIGURATION_ERROR": "Check config: futurnal config show",
    "INVALID_CONFIG": "Reset to defaults: futurnal config reset",
    "MISSING_CONFIG": "Add the required setting: futurnal config set <key> <value>",
    # Processing errors
    "PROCESSING_ERROR": "Check if the file is corrupted or in an unsupported format.",
    "NORMALIZATION_ERROR": "Ensure the file encoding is UTF-8 and format is supported.",
    "EXTRACTION_ERROR": "The document may be too complex. Try a simpler format.",
    "EMBEDDING_ERROR": "Check Ollama connection: curl http://localhost:11434/api/tags",
    # Generic
    "FUTURNAL_ERROR": "If this persists, please report the issue on GitHub.",
    "UNKNOWN_ERROR": "Try restarting Futurnal. Report if the issue continues.",
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_user_message(error: Any) -> str:
    """Get user-friendly message for an error.

    Args:
        error: The error (can be Exception or error code string)

    Returns:
        User-friendly error message
    """
    # Get error code
    if hasattr(error, "code"):
        code = error.code
    elif isinstance(error, str):
        code = error
    else:
        code = type(error).__name__.upper()

    # Look up message
    message = ERROR_MESSAGES.get(code, ERROR_MESSAGES["UNKNOWN_ERROR"])

    return message


def get_recovery_suggestion(error: Any) -> str:
    """Get recovery suggestion for an error.

    Args:
        error: The error (can be Exception or error code string)

    Returns:
        Recovery suggestion
    """
    # Get error code
    if hasattr(error, "code"):
        code = error.code
    elif isinstance(error, str):
        code = error
    else:
        code = type(error).__name__.upper()

    # Look up suggestion
    suggestion = RECOVERY_SUGGESTIONS.get(code, RECOVERY_SUGGESTIONS["UNKNOWN_ERROR"])

    return suggestion


def format_error_for_user(error: Any) -> str:
    """Format a complete user-friendly error message.

    Args:
        error: The error to format

    Returns:
        Complete error message with recovery suggestion
    """
    message = get_user_message(error)
    suggestion = get_recovery_suggestion(error)

    return f"{message}\n\nSuggestion: {suggestion}"


def format_error_for_cli(error: Any) -> str:
    """Format error for CLI output with color hints.

    Args:
        error: The error to format

    Returns:
        CLI-formatted error message
    """
    message = get_user_message(error)
    suggestion = get_recovery_suggestion(error)

    # Get error code for display
    if hasattr(error, "code"):
        code = error.code
    else:
        code = "ERROR"

    lines = [
        f"Error [{code}]: {message}",
        "",
        f"Suggestion: {suggestion}",
    ]

    # Add details if available
    if hasattr(error, "details") and error.details:
        lines.append("")
        lines.append("Details:")
        for key, value in error.details.items():
            # Don't expose sensitive details
            if key not in ("content", "query", "password", "token"):
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_error_for_ui(error: Any) -> dict:
    """Format error for UI/frontend display.

    Args:
        error: The error to format

    Returns:
        Dictionary with error info for UI rendering
    """
    return {
        "message": get_user_message(error),
        "suggestion": get_recovery_suggestion(error),
        "code": getattr(error, "code", "UNKNOWN_ERROR"),
        "recoverable": getattr(error, "recoverable", False),
    }


# =============================================================================
# Error Wrapping
# =============================================================================


def wrap_exception(
    exception: Exception,
    context: str | None = None,
) -> str:
    """Wrap any exception into a user-friendly message.

    This is a catch-all for exceptions that don't have our custom handling.

    Args:
        exception: Any exception
        context: Optional context about what operation failed

    Returns:
        User-friendly error message
    """
    # Check if it's already a Futurnal error
    if hasattr(exception, "user_message"):
        return exception.user_message

    # Map common exceptions
    exception_type = type(exception).__name__

    type_messages = {
        "TimeoutError": "The operation timed out. Please try again.",
        "MemoryError": "Not enough memory. Close other applications and retry.",
        "FileNotFoundError": "The requested file wasn't found.",
        "PermissionError": "Permission denied. Check file access rights.",
        "ValueError": "Invalid input provided. Check your query or settings.",
        "TypeError": "An internal error occurred. Please report this issue.",
        "KeyError": "Missing expected data. The operation couldn't complete.",
        "ConnectionRefusedError": "Connection refused. Check if services are running.",
        "ConnectionResetError": "Connection lost. Please try again.",
    }

    base_message = type_messages.get(
        exception_type,
        "An unexpected error occurred.",
    )

    if context:
        return f"{base_message} (while {context})"

    return base_message


__all__ = [
    "ERROR_MESSAGES",
    "RECOVERY_SUGGESTIONS",
    "get_user_message",
    "get_recovery_suggestion",
    "format_error_for_user",
    "format_error_for_cli",
    "format_error_for_ui",
    "wrap_exception",
]

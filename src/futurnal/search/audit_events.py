"""Search-specific audit event types and logging helpers.

This module defines search-specific audit event types and provides helper
functions for logging search operations with privacy-compliant metadata.

Event Types:
- Search events (query executed, cached hit, fallback)
- Index events (content indexed, embedding generated)
- Feedback events (quality feedback recorded)
- Error events (search failure, timeout)

Integration:
- Uses existing AuditLogger infrastructure
- NEVER logs query content (privacy requirement)
- Logs only metadata: intent, result count, latency

Privacy Guarantee:
- Query text is NEVER logged
- Only query_type/intent, result counts, and latency are recorded
- Content IDs are hashed if logged
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from ..privacy.audit import AuditLogger


class SearchAuditEvents:
    """Audit event types for search operations.

    Constants for standardized search audit event types, ensuring
    consistent event naming across the search system.
    """

    # Search events
    SEARCH_EXECUTED = "search_executed"
    SEARCH_CACHE_HIT = "search_cache_hit"
    SEARCH_FALLBACK = "search_fallback"
    SEARCH_FAILED = "search_failed"
    SEARCH_TIMEOUT = "search_timeout"

    # Hybrid search events
    HYBRID_SEARCH_EXECUTED = "hybrid_search_executed"
    VECTOR_SEARCH_EXECUTED = "vector_search_executed"
    GRAPH_EXPANSION_EXECUTED = "graph_expansion_executed"

    # Index events
    CONTENT_INDEXED = "content_indexed"
    EMBEDDING_GENERATED = "embedding_generated"

    # Multimodal events
    MULTIMODAL_SEARCH_EXECUTED = "multimodal_search_executed"
    OCR_CONTENT_INDEXED = "ocr_content_indexed"
    TRANSCRIPTION_INDEXED = "transcription_indexed"

    # Feedback events
    FEEDBACK_RECORDED = "search_feedback_recorded"


def log_search_executed(
    audit_logger: "AuditLogger",
    *,
    search_type: str,
    intent: str,
    result_count: int,
    latency_ms: float,
    cache_hit: bool = False,
    fallback_used: bool = False,
    filters_applied: Optional[Dict[str, str]] = None,
) -> None:
    """Log search execution event.

    Args:
        audit_logger: Audit logger instance
        search_type: Type of search (hybrid, temporal, causal, general)
        intent: Query intent classification
        result_count: Number of results returned
        latency_ms: Search latency in milliseconds
        cache_hit: Whether result was from cache
        fallback_used: Whether fallback search was used
        filters_applied: Optional filter types applied (not filter values)

    Privacy Note:
        Query text is NEVER logged. Only metadata about the search.
    """
    from ..privacy.audit import AuditEvent

    action = (
        SearchAuditEvents.SEARCH_CACHE_HIT
        if cache_hit
        else SearchAuditEvents.SEARCH_FALLBACK
        if fallback_used
        else SearchAuditEvents.SEARCH_EXECUTED
    )

    metadata: Dict[str, object] = {
        "search_type": search_type,
        "intent": intent,
        "result_count": result_count,
        "latency_ms": round(latency_ms, 2),
        "cache_hit": cache_hit,
        "fallback_used": fallback_used,
    }

    if filters_applied:
        # Only log filter types, not values (privacy)
        metadata["filter_types"] = list(filters_applied.keys())

    audit_logger.record(
        AuditEvent(
            job_id=f"search_{int(datetime.utcnow().timestamp() * 1000)}",
            source="search_api",
            action=action,
            status="success",
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_hybrid_search_executed(
    audit_logger: "AuditLogger",
    *,
    intent: str,
    result_count: int,
    vector_count: int,
    graph_count: int,
    latency_ms: float,
    vector_weight: float,
    graph_weight: float,
) -> None:
    """Log hybrid search execution with vector/graph breakdown.

    Args:
        audit_logger: Audit logger instance
        intent: Query intent classification
        result_count: Total results returned
        vector_count: Results from vector search
        graph_count: Results from graph expansion
        latency_ms: Total latency in milliseconds
        vector_weight: Weight applied to vector results
        graph_weight: Weight applied to graph results

    Privacy Note:
        Query text is NEVER logged.
    """
    from ..privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"hybrid_search_{int(datetime.utcnow().timestamp() * 1000)}",
            source="hybrid_retrieval",
            action=SearchAuditEvents.HYBRID_SEARCH_EXECUTED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "intent": intent,
                "result_count": result_count,
                "vector_count": vector_count,
                "graph_count": graph_count,
                "latency_ms": round(latency_ms, 2),
                "vector_weight": round(vector_weight, 2),
                "graph_weight": round(graph_weight, 2),
            },
        )
    )


def log_search_failed(
    audit_logger: "AuditLogger",
    *,
    search_type: str,
    intent: str,
    error_type: str,
    latency_ms: float,
    fallback_attempted: bool = False,
) -> None:
    """Log search failure event.

    Args:
        audit_logger: Audit logger instance
        search_type: Type of search that failed
        intent: Query intent classification
        error_type: Classification of error (timeout, vector_error, graph_error, etc.)
        latency_ms: Time until failure
        fallback_attempted: Whether fallback was attempted

    Privacy Note:
        Error messages that might contain query content are NOT logged.
    """
    from ..privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"search_fail_{int(datetime.utcnow().timestamp() * 1000)}",
            source="search_api",
            action=SearchAuditEvents.SEARCH_FAILED,
            status="failed",
            timestamp=datetime.utcnow(),
            metadata={
                "search_type": search_type,
                "intent": intent,
                "error_type": error_type,
                "latency_ms": round(latency_ms, 2),
                "fallback_attempted": fallback_attempted,
            },
        )
    )


def log_search_timeout(
    audit_logger: "AuditLogger",
    *,
    search_type: str,
    intent: str,
    timeout_ms: float,
    partial_results: int = 0,
) -> None:
    """Log search timeout event.

    Args:
        audit_logger: Audit logger instance
        search_type: Type of search that timed out
        intent: Query intent classification
        timeout_ms: Timeout threshold that was exceeded
        partial_results: Number of partial results if any
    """
    from ..privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"search_timeout_{int(datetime.utcnow().timestamp() * 1000)}",
            source="search_api",
            action=SearchAuditEvents.SEARCH_TIMEOUT,
            status="timeout",
            timestamp=datetime.utcnow(),
            metadata={
                "search_type": search_type,
                "intent": intent,
                "timeout_ms": round(timeout_ms, 2),
                "partial_results": partial_results,
            },
        )
    )


def log_content_indexed(
    audit_logger: "AuditLogger",
    *,
    content_id: str,
    content_type: str,
    source_type: str,
    size_bytes: Optional[int] = None,
) -> None:
    """Log content indexing event.

    Args:
        audit_logger: Audit logger instance
        content_id: Content identifier (may be hashed)
        content_type: Type of content (document, email, code, etc.)
        source_type: Source type (text, ocr, transcription)
        size_bytes: Optional content size

    Privacy Note:
        Content is NEVER logged, only metadata.
    """
    from ..privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "content_id_hash": _hash_id(content_id),
        "content_type": content_type,
        "source_type": source_type,
    }

    if size_bytes is not None:
        metadata["size_bytes"] = size_bytes

    audit_logger.record(
        AuditEvent(
            job_id=f"index_{_hash_id(content_id)[:16]}",
            source="search_indexer",
            action=SearchAuditEvents.CONTENT_INDEXED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_multimodal_search_executed(
    audit_logger: "AuditLogger",
    *,
    modalities: list,
    result_count: int,
    latency_ms: float,
    ocr_results: int = 0,
    transcription_results: int = 0,
) -> None:
    """Log multimodal search execution.

    Args:
        audit_logger: Audit logger instance
        modalities: List of modalities searched (text, image, audio)
        result_count: Total results returned
        latency_ms: Search latency
        ocr_results: Results from OCR content
        transcription_results: Results from transcriptions
    """
    from ..privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"multimodal_{int(datetime.utcnow().timestamp() * 1000)}",
            source="multimodal_handler",
            action=SearchAuditEvents.MULTIMODAL_SEARCH_EXECUTED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "modalities": modalities,
                "result_count": result_count,
                "latency_ms": round(latency_ms, 2),
                "ocr_results": ocr_results,
                "transcription_results": transcription_results,
            },
        )
    )


def log_ocr_content_indexed(
    audit_logger: "AuditLogger",
    *,
    content_id: str,
    confidence: float,
    source_file_hash: str,
    page_count: int = 1,
) -> None:
    """Log OCR content indexing event.

    Args:
        audit_logger: Audit logger instance
        content_id: Content identifier
        confidence: OCR confidence score
        source_file_hash: Hash of source file
        page_count: Number of pages processed
    """
    from ..privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"ocr_index_{_hash_id(content_id)[:16]}",
            source="ocr_processor",
            action=SearchAuditEvents.OCR_CONTENT_INDEXED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "content_id_hash": _hash_id(content_id),
                "confidence": round(confidence, 3),
                "source_file_hash": source_file_hash,
                "page_count": page_count,
            },
        )
    )


def log_transcription_indexed(
    audit_logger: "AuditLogger",
    *,
    content_id: str,
    duration_seconds: float,
    speaker_count: int = 0,
    language: str = "en",
) -> None:
    """Log transcription indexing event.

    Args:
        audit_logger: Audit logger instance
        content_id: Content identifier
        duration_seconds: Audio duration
        speaker_count: Number of detected speakers
        language: Detected language
    """
    from ..privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"trans_index_{_hash_id(content_id)[:16]}",
            source="transcription_processor",
            action=SearchAuditEvents.TRANSCRIPTION_INDEXED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "content_id_hash": _hash_id(content_id),
                "duration_seconds": round(duration_seconds, 2),
                "speaker_count": speaker_count,
                "language": language,
            },
        )
    )


def log_feedback_recorded(
    audit_logger: "AuditLogger",
    *,
    feedback_type: str,
    search_type: str,
    result_count: int,
    clicked_position: Optional[int] = None,
) -> None:
    """Log search quality feedback event.

    Args:
        audit_logger: Audit logger instance
        feedback_type: Type of feedback (click, no_results, refinement)
        search_type: Type of search the feedback is for
        result_count: Number of results in the original search
        clicked_position: Position of clicked result (if applicable)

    Privacy Note:
        Query and result content are NEVER logged.
    """
    from ..privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "feedback_type": feedback_type,
        "search_type": search_type,
        "result_count": result_count,
    }

    if clicked_position is not None:
        metadata["clicked_position"] = clicked_position

    audit_logger.record(
        AuditEvent(
            job_id=f"feedback_{int(datetime.utcnow().timestamp() * 1000)}",
            source="search_feedback",
            action=SearchAuditEvents.FEEDBACK_RECORDED,
            status="success",
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def _hash_id(identifier: str) -> str:
    """Create a privacy-safe hash of an identifier.

    Args:
        identifier: Content or entity identifier

    Returns:
        SHA256 hash of the identifier (first 32 chars)
    """
    return sha256(identifier.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "SearchAuditEvents",
    "log_search_executed",
    "log_hybrid_search_executed",
    "log_search_failed",
    "log_search_timeout",
    "log_content_indexed",
    "log_multimodal_search_executed",
    "log_ocr_content_indexed",
    "log_transcription_indexed",
    "log_feedback_recorded",
]

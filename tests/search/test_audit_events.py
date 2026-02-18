"""Tests for search audit event helpers."""

from datetime import datetime
from pathlib import Path
import json
import pytest

from futurnal.search.audit_events import (
    SearchAuditEvents,
    log_search_executed,
    log_hybrid_search_executed,
    log_search_failed,
    log_search_timeout,
    log_content_indexed,
    log_multimodal_search_executed,
    log_ocr_content_indexed,
    log_transcription_indexed,
    log_feedback_recorded,
    _hash_id,
)
from futurnal.privacy.audit import AuditLogger


@pytest.fixture
def audit_logger(tmp_path):
    """Create an audit logger for testing."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(exist_ok=True)
    return AuditLogger(output_dir=audit_dir)


def read_last_audit_event(audit_logger) -> dict:
    """Read the last audit event from the log."""
    log_path = audit_logger._path
    if not log_path.exists():
        return {}
    lines = log_path.read_text().strip().split("\n")
    if lines and lines[-1]:
        return json.loads(lines[-1])
    return {}


class TestSearchAuditEventConstants:
    """Test event type constants."""

    def test_search_event_constants(self):
        assert SearchAuditEvents.SEARCH_EXECUTED == "search_executed"
        assert SearchAuditEvents.SEARCH_CACHE_HIT == "search_cache_hit"
        assert SearchAuditEvents.SEARCH_FALLBACK == "search_fallback"
        assert SearchAuditEvents.SEARCH_FAILED == "search_failed"
        assert SearchAuditEvents.SEARCH_TIMEOUT == "search_timeout"

    def test_hybrid_search_event_constants(self):
        assert SearchAuditEvents.HYBRID_SEARCH_EXECUTED == "hybrid_search_executed"
        assert SearchAuditEvents.VECTOR_SEARCH_EXECUTED == "vector_search_executed"
        assert SearchAuditEvents.GRAPH_EXPANSION_EXECUTED == "graph_expansion_executed"

    def test_index_event_constants(self):
        assert SearchAuditEvents.CONTENT_INDEXED == "content_indexed"
        assert SearchAuditEvents.EMBEDDING_GENERATED == "embedding_generated"

    def test_multimodal_event_constants(self):
        assert SearchAuditEvents.MULTIMODAL_SEARCH_EXECUTED == "multimodal_search_executed"
        assert SearchAuditEvents.OCR_CONTENT_INDEXED == "ocr_content_indexed"
        assert SearchAuditEvents.TRANSCRIPTION_INDEXED == "transcription_indexed"


class TestSearchExecutedEvents:
    """Test search execution audit events."""

    def test_log_search_executed(self, audit_logger):
        log_search_executed(
            audit_logger,
            search_type="hybrid",
            intent="temporal",
            result_count=15,
            latency_ms=50.5,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.SEARCH_EXECUTED
        assert event["status"] == "success"
        assert event["source"] == "search_api"
        assert event["metadata"]["search_type"] == "hybrid"
        assert event["metadata"]["intent"] == "temporal"
        assert event["metadata"]["result_count"] == 15
        assert event["metadata"]["latency_ms"] == 50.5
        assert event["metadata"]["cache_hit"] is False
        assert event["metadata"]["fallback_used"] is False

    def test_log_search_cache_hit(self, audit_logger):
        log_search_executed(
            audit_logger,
            search_type="hybrid",
            intent="exploratory",
            result_count=10,
            latency_ms=2.5,
            cache_hit=True,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.SEARCH_CACHE_HIT
        assert event["metadata"]["cache_hit"] is True

    def test_log_search_with_fallback(self, audit_logger):
        log_search_executed(
            audit_logger,
            search_type="hybrid",
            intent="causal",
            result_count=5,
            latency_ms=100.0,
            fallback_used=True,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.SEARCH_FALLBACK
        assert event["metadata"]["fallback_used"] is True

    def test_log_search_with_filters(self, audit_logger):
        log_search_executed(
            audit_logger,
            search_type="temporal",
            intent="temporal",
            result_count=8,
            latency_ms=75.0,
            filters_applied={"date_range": "2024-01-01:2024-12-31", "source": "obsidian"},
        )

        event = read_last_audit_event(audit_logger)
        # Filter types are logged, not values (privacy)
        assert "filter_types" in event["metadata"]
        assert set(event["metadata"]["filter_types"]) == {"date_range", "source"}
        # Raw filter values should NOT be in metadata
        assert "date_range" not in event["metadata"]


class TestHybridSearchEvents:
    """Test hybrid search audit events."""

    def test_log_hybrid_search_executed(self, audit_logger):
        log_hybrid_search_executed(
            audit_logger,
            intent="causal",
            result_count=20,
            vector_count=15,
            graph_count=10,
            latency_ms=120.5,
            vector_weight=0.4,
            graph_weight=0.6,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.HYBRID_SEARCH_EXECUTED
        assert event["status"] == "success"
        assert event["source"] == "hybrid_retrieval"
        assert event["metadata"]["intent"] == "causal"
        assert event["metadata"]["result_count"] == 20
        assert event["metadata"]["vector_count"] == 15
        assert event["metadata"]["graph_count"] == 10
        assert event["metadata"]["latency_ms"] == 120.5
        assert event["metadata"]["vector_weight"] == 0.4
        assert event["metadata"]["graph_weight"] == 0.6


class TestSearchFailureEvents:
    """Test search failure audit events."""

    def test_log_search_failed(self, audit_logger):
        log_search_failed(
            audit_logger,
            search_type="hybrid",
            intent="exploratory",
            error_type="vector_error",
            latency_ms=500.0,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.SEARCH_FAILED
        assert event["status"] == "failed"
        assert event["metadata"]["search_type"] == "hybrid"
        assert event["metadata"]["error_type"] == "vector_error"
        assert event["metadata"]["fallback_attempted"] is False

    def test_log_search_failed_with_fallback(self, audit_logger):
        log_search_failed(
            audit_logger,
            search_type="hybrid",
            intent="causal",
            error_type="graph_error",
            latency_ms=250.0,
            fallback_attempted=True,
        )

        event = read_last_audit_event(audit_logger)
        assert event["metadata"]["fallback_attempted"] is True

    def test_log_search_timeout(self, audit_logger):
        log_search_timeout(
            audit_logger,
            search_type="hybrid",
            intent="temporal",
            timeout_ms=5000.0,
            partial_results=3,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.SEARCH_TIMEOUT
        assert event["status"] == "timeout"
        assert event["metadata"]["timeout_ms"] == 5000.0
        assert event["metadata"]["partial_results"] == 3


class TestIndexingEvents:
    """Test content indexing audit events."""

    def test_log_content_indexed(self, audit_logger):
        log_content_indexed(
            audit_logger,
            content_id="doc_12345",
            content_type="document",
            source_type="text",
            size_bytes=4096,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.CONTENT_INDEXED
        assert event["status"] == "success"
        assert event["source"] == "search_indexer"
        # Content ID should be hashed for privacy
        assert "content_id_hash" in event["metadata"]
        assert event["metadata"]["content_id_hash"] != "doc_12345"
        assert event["metadata"]["content_type"] == "document"
        assert event["metadata"]["source_type"] == "text"
        assert event["metadata"]["size_bytes"] == 4096

    def test_log_ocr_content_indexed(self, audit_logger):
        log_ocr_content_indexed(
            audit_logger,
            content_id="ocr_67890",
            confidence=0.92,
            source_file_hash="abc123def456",
            page_count=5,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.OCR_CONTENT_INDEXED
        assert event["status"] == "success"
        assert event["source"] == "ocr_processor"
        assert event["metadata"]["confidence"] == 0.92
        assert event["metadata"]["source_file_hash"] == "abc123def456"
        assert event["metadata"]["page_count"] == 5

    def test_log_transcription_indexed(self, audit_logger):
        log_transcription_indexed(
            audit_logger,
            content_id="audio_11111",
            duration_seconds=180.5,
            speaker_count=3,
            language="en",
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.TRANSCRIPTION_INDEXED
        assert event["status"] == "success"
        assert event["source"] == "transcription_processor"
        assert event["metadata"]["duration_seconds"] == 180.5
        assert event["metadata"]["speaker_count"] == 3
        assert event["metadata"]["language"] == "en"


class TestMultimodalSearchEvents:
    """Test multimodal search audit events."""

    def test_log_multimodal_search_executed(self, audit_logger):
        log_multimodal_search_executed(
            audit_logger,
            modalities=["text", "image", "audio"],
            result_count=12,
            latency_ms=200.0,
            ocr_results=4,
            transcription_results=2,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.MULTIMODAL_SEARCH_EXECUTED
        assert event["status"] == "success"
        assert event["source"] == "multimodal_handler"
        assert event["metadata"]["modalities"] == ["text", "image", "audio"]
        assert event["metadata"]["result_count"] == 12
        assert event["metadata"]["ocr_results"] == 4
        assert event["metadata"]["transcription_results"] == 2


class TestFeedbackEvents:
    """Test search feedback audit events."""

    def test_log_feedback_recorded_click(self, audit_logger):
        log_feedback_recorded(
            audit_logger,
            feedback_type="click",
            search_type="hybrid",
            result_count=10,
            clicked_position=2,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == SearchAuditEvents.FEEDBACK_RECORDED
        assert event["status"] == "success"
        assert event["source"] == "search_feedback"
        assert event["metadata"]["feedback_type"] == "click"
        assert event["metadata"]["search_type"] == "hybrid"
        assert event["metadata"]["result_count"] == 10
        assert event["metadata"]["clicked_position"] == 2

    def test_log_feedback_recorded_no_results(self, audit_logger):
        log_feedback_recorded(
            audit_logger,
            feedback_type="no_results",
            search_type="temporal",
            result_count=0,
        )

        event = read_last_audit_event(audit_logger)
        assert event["metadata"]["feedback_type"] == "no_results"
        assert event["metadata"]["result_count"] == 0
        assert "clicked_position" not in event["metadata"]


class TestPrivacyHelpers:
    """Test privacy helper functions."""

    def test_hash_id(self):
        id1 = "document_12345"
        id2 = "document_67890"

        hash1 = _hash_id(id1)
        hash2 = _hash_id(id2)

        # Hash should be different for different IDs
        assert hash1 != hash2
        # Hash should be consistent
        assert hash1 == _hash_id(id1)
        # Hash should be 32 characters (SHA256 truncated)
        assert len(hash1) == 32


class TestChainIntegrity:
    """Test that audit events maintain chain integrity."""

    def test_multiple_events_have_chain_fields(self, audit_logger):
        """Verify events have chain fields for tamper detection."""
        log_search_executed(
            audit_logger,
            search_type="hybrid",
            intent="exploratory",
            result_count=10,
            latency_ms=50.0,
        )

        log_content_indexed(
            audit_logger,
            content_id="doc_1",
            content_type="document",
            source_type="text",
        )

        log_hybrid_search_executed(
            audit_logger,
            intent="causal",
            result_count=15,
            vector_count=10,
            graph_count=8,
            latency_ms=100.0,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        log_feedback_recorded(
            audit_logger,
            feedback_type="click",
            search_type="hybrid",
            result_count=15,
            clicked_position=0,
        )

        # Verify all events have chain fields
        log_path = audit_logger._path
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 4

        for line in lines:
            event = json.loads(line)
            assert "chain_hash" in event
            assert "chain_prev" in event

        # Verify chain linkage
        events = [json.loads(line) for line in lines]
        assert events[0]["chain_prev"] is None  # First event has no previous
        for i in range(1, len(events)):
            assert events[i]["chain_prev"] == events[i - 1]["chain_hash"]


class TestNoContentLeakage:
    """Test privacy compliance - no content in logs."""

    def test_query_content_not_logged(self, audit_logger):
        """Verify query content is never logged."""
        # The query text itself is never passed to audit functions
        # Only metadata about the search is logged
        log_search_executed(
            audit_logger,
            search_type="hybrid",
            intent="temporal",
            result_count=5,
            latency_ms=30.0,
        )

        event = read_last_audit_event(audit_logger)
        event_str = json.dumps(event)

        # No query content should appear
        assert "what happened" not in event_str.lower()
        assert "search query" not in event_str.lower()
        # Metadata should be present
        assert "search_type" in event_str
        assert "result_count" in event_str

    def test_content_id_hashed(self, audit_logger):
        """Verify content IDs are hashed, not plain."""
        original_id = "secret_document_path_12345"

        log_content_indexed(
            audit_logger,
            content_id=original_id,
            content_type="document",
            source_type="text",
        )

        event = read_last_audit_event(audit_logger)

        # Original ID should not appear
        assert original_id not in json.dumps(event)
        # Hashed ID should be present
        assert "content_id_hash" in event["metadata"]

    def test_filter_values_not_logged(self, audit_logger):
        """Verify filter values are not logged, only types."""
        sensitive_filter = {
            "path": "/Users/john/secret/documents",
            "email": "john@secret.com",
            "date_range": "2024-01-01:2024-12-31",
        }

        log_search_executed(
            audit_logger,
            search_type="hybrid",
            intent="lookup",
            result_count=3,
            latency_ms=40.0,
            filters_applied=sensitive_filter,
        )

        event = read_last_audit_event(audit_logger)
        event_str = json.dumps(event)

        # Sensitive values should not appear
        assert "/Users/john" not in event_str
        assert "john@secret.com" not in event_str
        # Only filter types should be logged
        assert "filter_types" in event["metadata"]
        assert "path" in event["metadata"]["filter_types"]

"""Tests for EmbeddingRequest and BatchEmbeddingRequest.

Tests cover:
- Valid request creation
- Option B temporal context enforcement for Events
- Content validation
- Batch request grouping
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from futurnal.embeddings.models import TemporalEmbeddingContext
from futurnal.embeddings.request import (
    BatchEmbeddingRequest,
    EmbeddingRequest,
    TEMPORAL_ENTITY_TYPES,
)


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest model."""

    def test_valid_person_request(self):
        """Should accept Person without temporal context."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John Smith, Software Engineer at Futurnal",
        )

        assert request.entity_type == "Person"
        assert request.content == "John Smith, Software Engineer at Futurnal"
        assert request.temporal_context is None
        assert request.requires_temporal_context is False

    def test_valid_organization_request(self):
        """Should accept Organization without temporal context."""
        request = EmbeddingRequest(
            entity_type="Organization",
            content="Futurnal Inc - AI Research Company",
        )

        assert request.entity_type == "Organization"
        assert request.requires_temporal_context is False

    def test_valid_concept_request(self):
        """Should accept Concept without temporal context."""
        request = EmbeddingRequest(
            entity_type="Concept",
            content="Machine Learning - AI subset focused on pattern recognition",
        )

        assert request.entity_type == "Concept"
        assert request.requires_temporal_context is False

    def test_valid_event_request_with_temporal_context(self):
        """Should accept Event with temporal context."""
        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
        )

        request = EmbeddingRequest(
            entity_type="Event",
            content="Team Meeting: Quarterly planning discussion",
            entity_name="Team Meeting",
            temporal_context=context,
        )

        assert request.entity_type == "Event"
        assert request.temporal_context is not None
        assert request.requires_temporal_context is True
        assert request.has_temporal_context is True

    def test_event_requires_temporal_context_option_b(self):
        """Should raise error for Event without temporal_context (Option B compliance)."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingRequest(
                entity_type="Event",
                content="Team Meeting",
            )

        assert "temporal_context is REQUIRED" in str(exc_info.value)
        assert "Option B" in str(exc_info.value)

    def test_temporal_entity_types_constant(self):
        """Should have Event in TEMPORAL_ENTITY_TYPES."""
        assert "Event" in TEMPORAL_ENTITY_TYPES

    def test_content_min_length(self):
        """Should reject empty content."""
        with pytest.raises(ValueError):
            EmbeddingRequest(
                entity_type="Person",
                content="",  # Empty content
            )

    def test_content_whitespace_allowed(self):
        """Whitespace-only content is technically allowed (min_length=1).

        Note: While whitespace-only content passes validation, it would
        produce low-quality embeddings. Application layer should handle this.
        """
        # Pydantic min_length doesn't strip whitespace
        request = EmbeddingRequest(
            entity_type="Person",
            content="   ",  # Whitespace only - technically valid
        )
        assert len(request.content) == 3

    def test_optional_fields(self):
        """Should accept requests with minimal fields."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John",
        )

        assert request.entity_id is None
        assert request.entity_name is None
        assert request.metadata == {}

    def test_all_optional_fields(self):
        """Should accept requests with all optional fields."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John Smith",
            entity_id="pkg-uuid-123",
            entity_name="John Smith",
            metadata={"role": "Engineer", "department": "AI"},
        )

        assert request.entity_id == "pkg-uuid-123"
        assert request.entity_name == "John Smith"
        assert request.metadata["role"] == "Engineer"

    def test_has_causal_context_false_without_context(self):
        """Should return False for has_causal_context when no temporal context."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John",
        )

        assert request.has_causal_context is False

    def test_has_causal_context_false_empty_chain(self):
        """Should return False for has_causal_context with empty causal chain."""
        context = TemporalEmbeddingContext(
            timestamp=datetime.now(),
            causal_chain=[],
        )

        request = EmbeddingRequest(
            entity_type="Event",
            content="Meeting",
            temporal_context=context,
        )

        assert request.has_causal_context is False

    def test_has_causal_context_true_with_chain(self):
        """Should return True for has_causal_context with causal chain."""
        context = TemporalEmbeddingContext(
            timestamp=datetime.now(),
            causal_chain=["Meeting", "Discussion", "Decision"],
        )

        request = EmbeddingRequest(
            entity_type="Event",
            content="Decision",
            temporal_context=context,
        )

        assert request.has_causal_context is True

    def test_get_effective_name_with_entity_name(self):
        """Should return entity_name when set."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John Smith is a software engineer",
            entity_name="John Smith",
        )

        assert request.get_effective_name() == "John Smith"

    def test_get_effective_name_fallback_to_content(self):
        """Should return content prefix when entity_name not set."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John Smith is a software engineer at Futurnal Inc working on AI",
        )

        effective_name = request.get_effective_name()
        assert len(effective_name) <= 50
        assert effective_name.startswith("John Smith")

    def test_request_immutable(self):
        """EmbeddingRequest should be immutable (frozen)."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John",
        )

        with pytest.raises(Exception):  # ValidationError or AttributeError
            request.content = "Changed"


class TestBatchEmbeddingRequest:
    """Tests for BatchEmbeddingRequest model."""

    def test_create_batch_request(self):
        """Should create batch request with multiple requests."""
        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Organization", content="Acme"),
        ]

        batch = BatchEmbeddingRequest(requests=requests)

        assert batch.size == 2
        assert batch.fail_fast is True  # default

    def test_batch_request_fail_fast_false(self):
        """Should accept fail_fast=False."""
        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
        ]

        batch = BatchEmbeddingRequest(requests=requests, fail_fast=False)

        assert batch.fail_fast is False

    def test_batch_request_empty_list(self):
        """Should reject empty request list."""
        with pytest.raises(ValueError):
            BatchEmbeddingRequest(requests=[])

    def test_batch_entity_types(self):
        """Should return unique entity types in batch."""
        context = TemporalEmbeddingContext(timestamp=datetime.now())
        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Person", content="Jane"),
            EmbeddingRequest(entity_type="Organization", content="Acme"),
            EmbeddingRequest(
                entity_type="Event", content="Meeting", temporal_context=context
            ),
        ]

        batch = BatchEmbeddingRequest(requests=requests)

        entity_types = batch.entity_types
        assert len(entity_types) == 3
        assert "Person" in entity_types
        assert "Organization" in entity_types
        assert "Event" in entity_types

    def test_batch_group_by_entity_type(self):
        """Should group requests by entity type."""
        context = TemporalEmbeddingContext(timestamp=datetime.now())
        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Person", content="Jane"),
            EmbeddingRequest(entity_type="Organization", content="Acme"),
            EmbeddingRequest(
                entity_type="Event", content="Meeting", temporal_context=context
            ),
        ]

        batch = BatchEmbeddingRequest(requests=requests)
        grouped = batch.group_by_entity_type()

        assert len(grouped["Person"]) == 2
        assert len(grouped["Organization"]) == 1
        assert len(grouped["Event"]) == 1

    def test_batch_size_property(self):
        """Should return correct batch size."""
        requests = [
            EmbeddingRequest(entity_type="Person", content=f"Person {i}")
            for i in range(5)
        ]

        batch = BatchEmbeddingRequest(requests=requests)

        assert batch.size == 5

    def test_batch_request_immutable(self):
        """BatchEmbeddingRequest should be immutable (frozen)."""
        requests = [EmbeddingRequest(entity_type="Person", content="John")]
        batch = BatchEmbeddingRequest(requests=requests)

        with pytest.raises(Exception):  # ValidationError or AttributeError
            batch.fail_fast = False


class TestOptionBCompliance:
    """Tests specifically for Option B temporal-first compliance."""

    def test_event_without_timestamp_fails(self):
        """Option B: Events MUST have temporal context."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingRequest(
                entity_type="Event",
                content="Important Meeting",
            )

        error_message = str(exc_info.value)
        assert "REQUIRED" in error_message
        assert "Event" in error_message

    def test_static_entities_no_temporal_required(self):
        """Option B: Static entities (Person, Org, Concept) don't require temporal."""
        for entity_type in ["Person", "Organization", "Concept"]:
            request = EmbeddingRequest(
                entity_type=entity_type,
                content=f"Test {entity_type}",
            )
            assert request.temporal_context is None
            assert request.requires_temporal_context is False

    def test_event_with_minimal_temporal_context(self):
        """Option B: Event with just timestamp should be valid."""
        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15),
        )

        request = EmbeddingRequest(
            entity_type="Event",
            content="Meeting",
            temporal_context=context,
        )

        assert request.has_temporal_context is True

    def test_error_message_guides_to_option_b(self):
        """Error message should explain Option B requirement."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingRequest(
                entity_type="Event",
                content="Meeting",
            )

        error_message = str(exc_info.value)
        assert "Option B" in error_message
        assert "temporal-first" in error_message
        assert "correlation detection" in error_message

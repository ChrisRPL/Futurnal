"""PKG Schema Model Tests.

Tests for PKG node and relationship models per production plan:
docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Success Metrics:
- All node types creatable with proper constraints
- All relationship types functional
- Temporal relationships enforce ordering
- Causal relationships support Phase 3 validation

Option B Compliance:
- Event.timestamp is REQUIRED (temporal-first design)
- Causal relationships have Bradford Hill criteria structure
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from futurnal.pkg.schema.models import (
    # Node models
    PersonNode,
    OrganizationNode,
    ConceptNode,
    DocumentNode,
    EventNode,
    SchemaVersionNode,
    ChunkNode,
    # Relationship models
    StandardRelationshipProps,
    TemporalRelationshipProps,
    CausalRelationshipProps,
    ProvenanceRelationshipProps,
    # Enums
    TemporalRelationType,
    CausalRelationType,
    ProvenanceRelationType,
    StandardRelationType,
    # Validation
    TemporalOrderingError,
    validate_temporal_relationship,
)


# ---------------------------------------------------------------------------
# Node Type Creation Tests (Success Metric 1)
# ---------------------------------------------------------------------------


class TestNodeTypeCreation:
    """Validate all node types can be created with proper constraints."""

    def test_person_node_creation(self, sample_person):
        """Test PersonNode creation with all fields."""
        assert sample_person.id is not None
        assert sample_person.name == "John Doe"
        assert "Johnny" in sample_person.aliases
        assert sample_person.discovery_count == 5
        assert sample_person.confidence == 0.95
        assert sample_person.first_seen_document == "doc_123"
        assert sample_person.created_at is not None
        assert sample_person.updated_at is not None

    def test_person_node_minimal(self):
        """Test PersonNode with minimal required fields."""
        person = PersonNode(name="Jane Doe")
        assert person.id is not None
        assert person.name == "Jane Doe"
        assert person.aliases == []
        assert person.discovery_count == 0
        assert person.confidence == 1.0
        assert person.first_seen_document is None

    def test_organization_node_creation(self, sample_organization):
        """Test OrganizationNode creation."""
        assert sample_organization.id is not None
        assert sample_organization.name == "Acme Corp"
        assert sample_organization.type == "company"
        assert "ACME Corporation" in sample_organization.aliases

    def test_concept_node_creation(self, sample_concept):
        """Test ConceptNode creation."""
        assert sample_concept.id is not None
        assert sample_concept.name == "Machine Learning"
        assert sample_concept.category == "field"
        assert "ML" in sample_concept.aliases

    def test_document_node_creation(self, sample_document):
        """Test DocumentNode creation."""
        assert sample_document.id is not None
        assert sample_document.source_id == "vault/notes/test.md"
        assert sample_document.source_type == "obsidian_vault"
        assert sample_document.content_hash == "abc123def456"
        assert sample_document.format == "markdown"

    def test_event_node_creation(self, sample_event):
        """Test EventNode creation with required timestamp."""
        assert sample_event.id is not None
        assert sample_event.name == "Team Meeting"
        assert sample_event.event_type == "meeting"
        # CRITICAL: Event MUST have timestamp (Option B requirement)
        assert sample_event.timestamp is not None
        assert sample_event.timestamp == datetime(2024, 1, 15, 14, 0, 0)
        assert sample_event.duration == timedelta(hours=1)
        # End timestamp should be computed
        assert sample_event.end_timestamp == datetime(2024, 1, 15, 15, 0, 0)

    def test_event_node_requires_timestamp(self):
        """Test that EventNode requires timestamp (Option B compliance)."""
        with pytest.raises(ValidationError) as exc_info:
            EventNode(
                name="Invalid Event",
                event_type="meeting",
                source_document="doc_123",
                # Missing required timestamp
            )
        assert "timestamp" in str(exc_info.value)

    def test_event_node_requires_name(self):
        """Test that EventNode requires name."""
        with pytest.raises(ValidationError):
            EventNode(
                event_type="meeting",
                timestamp=datetime.now(),
                source_document="doc_123",
            )

    def test_event_node_requires_event_type(self):
        """Test that EventNode requires event_type."""
        with pytest.raises(ValidationError):
            EventNode(
                name="Some Event",
                timestamp=datetime.now(),
                source_document="doc_123",
            )

    def test_event_node_requires_source_document(self):
        """Test that EventNode requires source_document."""
        with pytest.raises(ValidationError):
            EventNode(
                name="Some Event",
                event_type="meeting",
                timestamp=datetime.now(),
            )

    def test_schema_version_node_creation(self):
        """Test SchemaVersionNode creation."""
        version = SchemaVersionNode(
            version=1,
            entity_types=["Person", "Organization", "Event"],
            relationship_types=["WORKS_AT", "CAUSES"],
            changes='{"type": "initial"}',
            reflection_quality=0.95,
        )
        assert version.id is not None
        assert version.version == 1
        assert "Person" in version.entity_types
        assert "CAUSES" in version.relationship_types

    def test_chunk_node_creation(self):
        """Test ChunkNode creation."""
        chunk = ChunkNode(
            document_id="doc_123",
            content_hash="hash_abc",
            position=100,
            chunk_index=2,
        )
        assert chunk.id is not None
        assert chunk.document_id == "doc_123"
        assert chunk.position == 100
        assert chunk.chunk_index == 2


class TestNodeCypherSerialization:
    """Test node serialization for Neo4j."""

    def test_person_to_cypher_properties(self, sample_person):
        """Test PersonNode conversion to Cypher properties."""
        props = sample_person.to_cypher_properties()
        assert props["name"] == "John Doe"
        assert props["discovery_count"] == 5
        # datetime should be serialized to ISO format
        assert isinstance(props["created_at"], str)
        assert "T" in props["created_at"]  # ISO format has T separator

    def test_event_to_cypher_properties(self, sample_event):
        """Test EventNode conversion to Cypher properties."""
        props = sample_event.to_cypher_properties()
        assert props["name"] == "Team Meeting"
        # Datetime fields serialized
        assert isinstance(props["timestamp"], str)
        # Duration serialized to seconds
        assert props["duration"] == 3600.0  # 1 hour in seconds


# ---------------------------------------------------------------------------
# Relationship Type Tests (Success Metric 2)
# ---------------------------------------------------------------------------


class TestRelationshipTypes:
    """Validate all relationship types functional."""

    def test_standard_relationship_works_at(self):
        """Test WORKS_AT relationship properties."""
        rel = StandardRelationshipProps(
            source_document="doc_123",
            valid_from=datetime(2020, 1, 1),
            valid_to=None,  # Ongoing
            role="Software Engineer",
            confidence=0.95,
        )
        assert rel.valid_from == datetime(2020, 1, 1)
        assert rel.valid_to is None
        assert rel.role == "Software Engineer"

    def test_standard_relationship_created(self):
        """Test CREATED relationship properties."""
        rel = StandardRelationshipProps(
            source_document="doc_456",
            confidence=1.0,
            extraction_method="metadata",
        )
        assert rel.confidence == 1.0
        assert rel.extraction_method == "metadata"

    def test_standard_relationship_related_to(self):
        """Test RELATED_TO relationship with subtype."""
        rel = StandardRelationshipProps(
            source_document="doc_789",
            relationship_subtype="subset_of",
            strength=0.8,
        )
        assert rel.relationship_subtype == "subset_of"
        assert rel.strength == 0.8

    def test_temporal_relationship_before(self):
        """Test BEFORE temporal relationship."""
        rel = TemporalRelationshipProps(
            source_document="doc_123",
            temporal_confidence=0.95,
            temporal_source="explicit_timestamp",
            temporal_gap=timedelta(hours=7),
        )
        assert rel.temporal_confidence == 0.95
        assert rel.temporal_gap == timedelta(hours=7)

    def test_temporal_relationship_during(self):
        """Test DURING temporal relationship with overlap."""
        rel = TemporalRelationshipProps(
            source_document="doc_123",
            temporal_source="inferred",
            overlap_start=datetime(2024, 1, 15, 10, 0, 0),
            overlap_end=datetime(2024, 1, 15, 11, 0, 0),
            overlap_type="partial",
        )
        assert rel.overlap_type == "partial"
        assert rel.overlap_start is not None

    def test_temporal_relationship_simultaneous(self):
        """Test SIMULTANEOUS temporal relationship."""
        rel = TemporalRelationshipProps(
            source_document="doc_123",
            simultaneity_tolerance=timedelta(minutes=5),
        )
        assert rel.simultaneity_tolerance == timedelta(minutes=5)

    def test_causal_relationship_causes(self):
        """Test CAUSES causal relationship with Bradford Hill criteria."""
        rel = CausalRelationshipProps(
            source_document="doc_123",
            causal_confidence=0.7,
            causal_evidence="Meeting resulted in decision",
            is_causal_candidate=True,
            temporal_gap=timedelta(hours=7),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
            # Phase 3 fields (nullable for Phase 1)
            strength=None,
            dose_response=None,
            consistency=None,
            plausibility=None,
        )
        assert rel.causal_confidence == 0.7
        assert rel.is_causal_candidate is True
        assert rel.temporality_satisfied is True
        # Phase 3 fields are None in Phase 1
        assert rel.strength is None

    def test_causal_relationship_enables(self):
        """Test ENABLES causal relationship."""
        rel = CausalRelationshipProps(
            source_document="doc_456",
            causal_confidence=0.6,
            causal_evidence="Training enabled skill development",
            temporal_gap=timedelta(days=30),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )
        assert rel.causal_confidence == 0.6

    def test_causal_relationship_prevents(self):
        """Test PREVENTS causal relationship."""
        rel = CausalRelationshipProps(
            source_document="doc_789",
            causal_confidence=0.5,
            causal_evidence="Lockdown prevented travel",
            temporal_gap=timedelta(days=1),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )
        assert rel.causal_confidence == 0.5

    def test_causal_relationship_triggers(self):
        """Test TRIGGERS causal relationship."""
        rel = CausalRelationshipProps(
            source_document="doc_101",
            causal_confidence=0.9,
            causal_evidence="Alarm triggered response",
            temporal_gap=timedelta(seconds=30),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )
        assert rel.temporal_gap == timedelta(seconds=30)

    def test_provenance_relationship_extracted_from(self):
        """Test EXTRACTED_FROM provenance relationship."""
        rel = ProvenanceRelationshipProps(
            source_document="doc_123",
            extraction_confidence=0.95,
            discovery_method="llm_extraction",
        )
        assert rel.extraction_confidence == 0.95
        assert rel.discovery_method == "llm_extraction"

    def test_provenance_relationship_discovered_in(self):
        """Test DISCOVERED_IN provenance relationship."""
        rel = ProvenanceRelationshipProps(
            source_document="doc_456",
            extraction_confidence=1.0,
            discovery_method="pattern_match",
        )
        assert rel.discovery_method == "pattern_match"

    def test_provenance_relationship_participated_in(self):
        """Test PARTICIPATED_IN provenance relationship."""
        rel = ProvenanceRelationshipProps(
            source_document="doc_789",
            role="organizer",
            participation_confirmed=True,
        )
        assert rel.role == "organizer"
        assert rel.participation_confirmed is True


class TestRelationshipCypherSerialization:
    """Test relationship serialization for Neo4j."""

    def test_temporal_relationship_to_cypher(self):
        """Test temporal relationship conversion to Cypher properties."""
        rel = TemporalRelationshipProps(
            source_document="doc_123",
            temporal_gap=timedelta(hours=3),
        )
        props = rel.to_cypher_properties()
        # timedelta serialized to seconds
        assert props["temporal_gap"] == 10800.0

    def test_causal_relationship_to_cypher(self):
        """Test causal relationship conversion to Cypher properties."""
        rel = CausalRelationshipProps(
            source_document="doc_123",
            temporal_gap=timedelta(days=1),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )
        props = rel.to_cypher_properties()
        assert props["temporal_ordering_valid"] is True


# ---------------------------------------------------------------------------
# Temporal Ordering Tests (Success Metric 3)
# ---------------------------------------------------------------------------


class TestTemporalOrderingConstraint:
    """Validate temporal relationships enforce ordering."""

    def test_before_relationship_valid_ordering(self, sample_events_pair):
        """Test BEFORE relationship with valid ordering."""
        event1, event2 = sample_events_pair  # event1 at 9:00, event2 at 16:00

        # Should succeed: event1 is before event2
        result = validate_temporal_relationship(
            event1, event2, TemporalRelationType.BEFORE
        )
        assert result is True

    def test_before_relationship_invalid_ordering(self, sample_events_pair):
        """Test BEFORE relationship with invalid ordering raises error."""
        event1, event2 = sample_events_pair

        # Should fail: event2 (16:00) is NOT before event1 (9:00)
        with pytest.raises(TemporalOrderingError) as exc_info:
            validate_temporal_relationship(
                event2, event1, TemporalRelationType.BEFORE
            )
        assert "BEFORE" in str(exc_info.value)

    def test_after_relationship_valid_ordering(self, sample_events_pair):
        """Test AFTER relationship with valid ordering."""
        event1, event2 = sample_events_pair

        # Should succeed: event2 is after event1
        result = validate_temporal_relationship(
            event2, event1, TemporalRelationType.AFTER
        )
        assert result is True

    def test_after_relationship_invalid_ordering(self, sample_events_pair):
        """Test AFTER relationship with invalid ordering raises error."""
        event1, event2 = sample_events_pair

        # Should fail: event1 (9:00) is NOT after event2 (16:00)
        with pytest.raises(TemporalOrderingError) as exc_info:
            validate_temporal_relationship(
                event1, event2, TemporalRelationType.AFTER
            )
        assert "AFTER" in str(exc_info.value)

    def test_during_relationship_no_ordering_required(self, sample_events_pair):
        """Test DURING relationship doesn't require ordering."""
        event1, event2 = sample_events_pair

        # Should succeed regardless of ordering
        result = validate_temporal_relationship(
            event1, event2, TemporalRelationType.DURING
        )
        assert result is True

        result = validate_temporal_relationship(
            event2, event1, TemporalRelationType.DURING
        )
        assert result is True

    def test_simultaneous_no_ordering_required(self, sample_events_pair):
        """Test SIMULTANEOUS relationship doesn't require ordering."""
        event1, event2 = sample_events_pair

        # Should succeed regardless of ordering
        result = validate_temporal_relationship(
            event1, event2, TemporalRelationType.SIMULTANEOUS
        )
        assert result is True

    def test_causal_relationship_requires_ordering(self, sample_events_pair):
        """Test causal relationships require cause before effect."""
        event1, event2 = sample_events_pair

        # Should succeed: event1 (cause at 9:00) before event2 (effect at 16:00)
        result = validate_temporal_relationship(
            event1, event2, CausalRelationType.CAUSES
        )
        assert result is True

        # Should fail: event2 (16:00) cannot cause event1 (9:00)
        with pytest.raises(TemporalOrderingError):
            validate_temporal_relationship(
                event2, event1, CausalRelationType.CAUSES
            )

    def test_enables_requires_ordering(self, sample_events_pair):
        """Test ENABLES relationship requires temporal ordering."""
        event1, event2 = sample_events_pair

        # Should succeed
        result = validate_temporal_relationship(
            event1, event2, CausalRelationType.ENABLES
        )
        assert result is True

        # Should fail
        with pytest.raises(TemporalOrderingError):
            validate_temporal_relationship(
                event2, event1, CausalRelationType.ENABLES
            )


# ---------------------------------------------------------------------------
# Phase 3 Support Tests (Success Metric 4)
# ---------------------------------------------------------------------------


class TestCausalRelationshipPhase3Support:
    """Validate causal relationships support Phase 3 validation."""

    def test_bradford_hill_criteria_structure(self):
        """Test Bradford Hill criteria fields are present."""
        rel = CausalRelationshipProps(
            source_document="doc_123",
            causal_confidence=0.7,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )

        # All Bradford Hill criteria fields should exist
        assert hasattr(rel, "temporality_satisfied")  # Criterion 1 (required)
        assert hasattr(rel, "strength")               # Criterion 2
        assert hasattr(rel, "dose_response")          # Criterion 3
        assert hasattr(rel, "consistency")            # Criterion 4
        assert hasattr(rel, "plausibility")           # Criterion 5

    def test_causal_candidate_flag(self):
        """Test is_causal_candidate flag for Phase 3."""
        rel = CausalRelationshipProps(
            source_document="doc_123",
            is_causal_candidate=True,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )
        assert rel.is_causal_candidate is True
        assert rel.is_validated is False  # Not validated in Phase 1

    def test_validation_fields(self):
        """Test Phase 3 validation fields."""
        rel = CausalRelationshipProps(
            source_document="doc_123",
            is_validated=True,
            validation_method="interactive_exploration",
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            temporality_satisfied=True,
        )
        assert rel.is_validated is True
        assert rel.validation_method == "interactive_exploration"

    def test_full_bradford_hill_validation(self):
        """Test causal relationship with full Bradford Hill criteria (Phase 3)."""
        rel = CausalRelationshipProps(
            source_document="doc_123",
            causal_confidence=0.85,
            causal_evidence="Strong causal evidence from multiple sources",
            is_causal_candidate=True,
            is_validated=True,
            validation_method="bradford_hill_analysis",
            temporal_gap=timedelta(days=7),
            temporal_ordering_valid=True,
            # Bradford Hill criteria
            temporality_satisfied=True,
            strength=0.8,
            dose_response=True,
            consistency=0.75,
            plausibility="Exercise increases endorphins which improve mood",
        )

        assert rel.temporality_satisfied is True
        assert rel.strength == 0.8
        assert rel.dose_response is True
        assert rel.consistency == 0.75
        assert rel.plausibility is not None


# ---------------------------------------------------------------------------
# Edge Cases and Validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_confidence_bounds(self):
        """Test confidence values are bounded 0-1."""
        with pytest.raises(ValidationError):
            PersonNode(name="Test", confidence=1.5)

        with pytest.raises(ValidationError):
            PersonNode(name="Test", confidence=-0.1)

    def test_discovery_count_non_negative(self):
        """Test discovery_count cannot be negative."""
        with pytest.raises(ValidationError):
            PersonNode(name="Test", discovery_count=-1)

    def test_version_positive(self):
        """Test schema version must be positive."""
        with pytest.raises(ValidationError):
            SchemaVersionNode(
                version=0,
                entity_types=["Person"],
                relationship_types=["RELATED_TO"],
            )

    def test_chunk_position_non_negative(self):
        """Test chunk position cannot be negative."""
        with pytest.raises(ValidationError):
            ChunkNode(
                document_id="doc_123",
                content_hash="hash",
                position=-1,
                chunk_index=0,
            )

    def test_temporal_ordering_error_message(self):
        """Test TemporalOrderingError provides useful message."""
        source_ts = datetime(2024, 1, 15, 16, 0, 0)
        target_ts = datetime(2024, 1, 15, 9, 0, 0)

        error = TemporalOrderingError(source_ts, target_ts, "BEFORE")
        assert "BEFORE" in str(error)
        assert str(source_ts) in str(error)
        assert str(target_ts) in str(error)

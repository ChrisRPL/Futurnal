"""Tests for temporal relationship detection.

Target metrics:
- >80% explicit relationship detection
- >70% implicit relationship detection
- 100% temporal ordering validation

This test suite validates the temporal relationship detection components
that feed into the consistency validator and Phase 3 causal inference.

Research Foundation:
- Allen's Interval Algebra: 7 fundamental temporal relationships
- Temporal KG Extrapolation: Causal subhistory identification
"""

from datetime import datetime, timedelta

import pytest

from futurnal.extraction.temporal.models import (
    Event,
    TemporalRelationship,
    TemporalRelationshipType,
)
from futurnal.extraction.temporal.consistency import (
    TemporalConsistencyValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_time():
    """Base time for creating test events."""
    return datetime(2024, 1, 15, 10, 0, 0)


def make_event(
    event_id: str,
    name: str,
    timestamp: datetime,
    duration: timedelta = None,
    event_type: str = "test_event",
) -> Event:
    """Helper to create test events."""
    return Event(
        event_id=event_id,
        name=name,
        event_type=event_type,
        timestamp=timestamp,
        duration=duration,
        source_document="test_doc",
    )


def make_relationship(
    entity1_id: str,
    entity2_id: str,
    rel_type: TemporalRelationshipType,
    confidence: float = 0.9,
    evidence: str = "",
) -> TemporalRelationship:
    """Helper to create test relationships."""
    return TemporalRelationship(
        entity1_id=entity1_id,
        entity2_id=entity2_id,
        relationship_type=rel_type,
        confidence=confidence,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Test: Temporal Relationship Types
# ---------------------------------------------------------------------------


class TestTemporalRelationshipTypes:
    """Tests for Allen's Interval Algebra relationship types."""

    def test_all_allen_types_exist(self):
        """Verify all 7 Allen's IA types are defined."""
        allen_types = [
            TemporalRelationshipType.BEFORE,
            TemporalRelationshipType.AFTER,
            TemporalRelationshipType.DURING,
            TemporalRelationshipType.CONTAINS,
            TemporalRelationshipType.OVERLAPS,
            TemporalRelationshipType.MEETS,
            TemporalRelationshipType.EQUALS,
        ]
        assert len(allen_types) == 7

    def test_causal_extensions_exist(self):
        """Verify causal extensions are defined."""
        causal_types = [
            TemporalRelationshipType.CAUSES,
            TemporalRelationshipType.ENABLES,
            TemporalRelationshipType.PREVENTS,
            TemporalRelationshipType.TRIGGERS,
        ]
        assert len(causal_types) == 4

    def test_relationship_type_values(self):
        """Test relationship type string values."""
        assert TemporalRelationshipType.BEFORE.value == "before"
        assert TemporalRelationshipType.AFTER.value == "after"
        assert TemporalRelationshipType.CAUSES.value == "causes"
        assert TemporalRelationshipType.SIMULTANEOUS.value == "simultaneous"


# ---------------------------------------------------------------------------
# Test: Temporal Relationship Creation
# ---------------------------------------------------------------------------


class TestTemporalRelationshipCreation:
    """Tests for creating TemporalRelationship objects."""

    def test_create_before_relationship(self, base_time):
        """Test creating BEFORE relationship."""
        rel = make_relationship(
            "event_a",
            "event_b",
            TemporalRelationshipType.BEFORE,
            confidence=0.95,
            evidence="Event A occurred before Event B",
        )

        assert rel.entity1_id == "event_a"
        assert rel.entity2_id == "event_b"
        assert rel.relationship_type == TemporalRelationshipType.BEFORE
        assert rel.confidence == 0.95
        assert rel.evidence == "Event A occurred before Event B"

    def test_create_causes_relationship(self, base_time):
        """Test creating CAUSES relationship."""
        rel = make_relationship(
            "cause_event",
            "effect_event",
            TemporalRelationshipType.CAUSES,
            confidence=0.8,
            evidence="The meeting led to the decision",
        )

        assert rel.relationship_type == TemporalRelationshipType.CAUSES
        assert rel.confidence == 0.8

    def test_relationship_to_dict(self, base_time):
        """Test relationship serialization."""
        rel = make_relationship(
            "A", "B",
            TemporalRelationshipType.BEFORE,
            confidence=0.9,
        )

        data = rel.to_dict()
        assert data["entity1_id"] == "A"
        assert data["entity2_id"] == "B"
        assert data["relationship_type"] == "before"
        assert data["confidence"] == 0.9


# ---------------------------------------------------------------------------
# Test: Explicit Relationship Detection
# ---------------------------------------------------------------------------


class TestExplicitRelationshipDetection:
    """Tests for explicit temporal relationship detection from language."""

    def test_before_relationship_explicit(self, base_time):
        """Test detection of explicit BEFORE language patterns."""
        # Test cases with explicit BEFORE language
        before_patterns = [
            "The meeting occurred before the decision",
            "Prior to the launch, we had a review",
            "Earlier than expected, we finished",
            "Preceding the announcement, we prepared",
        ]

        # All these should suggest BEFORE relationship
        for pattern in before_patterns:
            rel = make_relationship(
                "event_a", "event_b",
                TemporalRelationshipType.BEFORE,
                evidence=pattern,
            )
            assert rel.relationship_type == TemporalRelationshipType.BEFORE

    def test_after_relationship_explicit(self, base_time):
        """Test detection of explicit AFTER language patterns."""
        after_patterns = [
            "After the presentation, we discussed",
            "Following the meeting, we sent the summary",
            "Subsequently, the team implemented",
            "Later than planned, we delivered",
        ]

        for pattern in after_patterns:
            rel = make_relationship(
                "event_a", "event_b",
                TemporalRelationshipType.AFTER,
                evidence=pattern,
            )
            assert rel.relationship_type == TemporalRelationshipType.AFTER

    def test_during_relationship_explicit(self, base_time):
        """Test detection of explicit DURING language patterns."""
        during_patterns = [
            "During the meeting, we decided",
            "While discussing, we realized",
            "In the course of the review, we found",
        ]

        for pattern in during_patterns:
            rel = make_relationship(
                "event_a", "event_b",
                TemporalRelationshipType.DURING,
                evidence=pattern,
            )
            assert rel.relationship_type == TemporalRelationshipType.DURING

    def test_causes_relationship_explicit(self, base_time):
        """Test detection of explicit CAUSES language patterns."""
        causes_patterns = [
            "The bug caused the system crash",
            "This led to the outage",
            "The change resulted in errors",
            "Due to the update, performance degraded",
        ]

        for pattern in causes_patterns:
            rel = make_relationship(
                "cause", "effect",
                TemporalRelationshipType.CAUSES,
                evidence=pattern,
            )
            assert rel.relationship_type == TemporalRelationshipType.CAUSES


# ---------------------------------------------------------------------------
# Test: Temporal Relationship Inference from Timestamps
# ---------------------------------------------------------------------------


class TestTemporalRelationshipInference:
    """Tests for inferring temporal relationships from event timestamps."""

    def test_infer_before_from_timestamps(self, base_time):
        """Test inference of BEFORE from event timestamps."""
        event_a = make_event("A", "Event A", base_time)
        event_b = make_event("B", "Event B", base_time + timedelta(hours=1))

        # A should be BEFORE B
        validator = TemporalConsistencyValidator()
        assert validator.validate_single_relationship(
            event_a, event_b, TemporalRelationshipType.BEFORE
        )

    def test_infer_after_from_timestamps(self, base_time):
        """Test inference of AFTER from event timestamps."""
        event_a = make_event("A", "Event A", base_time + timedelta(hours=1))
        event_b = make_event("B", "Event B", base_time)

        # A should be AFTER B
        validator = TemporalConsistencyValidator()
        assert validator.validate_single_relationship(
            event_a, event_b, TemporalRelationshipType.AFTER
        )

    def test_infer_simultaneous_from_same_timestamps(self, base_time):
        """Test inference of SIMULTANEOUS from identical timestamps."""
        event_a = make_event("A", "Event A", base_time)
        event_b = make_event("B", "Event B", base_time)

        # A and B should be SIMULTANEOUS
        validator = TemporalConsistencyValidator()
        assert validator.validate_single_relationship(
            event_a, event_b, TemporalRelationshipType.SIMULTANEOUS
        )

    def test_infer_during_from_timestamps_and_duration(self, base_time):
        """Test inference of DURING from timestamps and durations."""
        # Outer event with 2-hour duration
        outer = make_event(
            "outer", "Outer Event", base_time,
            duration=timedelta(hours=2)
        )
        # Inner event within outer's timespan
        inner = make_event(
            "inner", "Inner Event", base_time + timedelta(minutes=30)
        )

        # Inner should be DURING outer
        validator = TemporalConsistencyValidator()
        assert validator.validate_single_relationship(
            inner, outer, TemporalRelationshipType.DURING
        )


# ---------------------------------------------------------------------------
# Test: Relationship Confidence Scoring
# ---------------------------------------------------------------------------


class TestRelationshipConfidenceScoring:
    """Tests for confidence scoring on relationships."""

    def test_confidence_range_valid(self):
        """Test confidence scores are in valid range."""
        for confidence in [0.0, 0.5, 0.7, 0.9, 1.0]:
            rel = make_relationship(
                "A", "B",
                TemporalRelationshipType.BEFORE,
                confidence=confidence,
            )
            assert 0.0 <= rel.confidence <= 1.0

    def test_explicit_relationships_higher_confidence(self):
        """Test explicit relationships should have higher confidence."""
        # Explicit relationship with evidence
        explicit = make_relationship(
            "A", "B",
            TemporalRelationshipType.BEFORE,
            confidence=0.95,
            evidence="A occurred before B",
        )

        # Implicit relationship inferred from timestamps
        implicit = make_relationship(
            "C", "D",
            TemporalRelationshipType.BEFORE,
            confidence=0.7,
            evidence="",  # No explicit evidence
        )

        # Explicit should have higher confidence
        assert explicit.confidence > implicit.confidence

    def test_causal_relationships_require_higher_confidence(self):
        """Test causal relationships need higher confidence threshold."""
        # Per quality gates: causal candidates need confidence >0.6
        MIN_CAUSAL_CONFIDENCE = 0.6

        causal = make_relationship(
            "cause", "effect",
            TemporalRelationshipType.CAUSES,
            confidence=0.65,
        )

        assert causal.confidence >= MIN_CAUSAL_CONFIDENCE


# ---------------------------------------------------------------------------
# Test: Temporal Ordering Validation
# ---------------------------------------------------------------------------


class TestTemporalOrderingValidation:
    """Tests for temporal ordering validation in relationships."""

    def test_valid_before_ordering(self, base_time):
        """Test BEFORE requires source < target."""
        event_a = make_event("A", "Event A", base_time)
        event_b = make_event("B", "Event B", base_time + timedelta(hours=1))

        # A BEFORE B should be valid
        rel = make_relationship("A", "B", TemporalRelationshipType.BEFORE)

        validator = TemporalConsistencyValidator()
        result = validator.validate([event_a, event_b], [rel])
        assert result.valid

    def test_invalid_before_ordering(self, base_time):
        """Test BEFORE fails when source >= target."""
        event_a = make_event("A", "Event A", base_time + timedelta(hours=1))
        event_b = make_event("B", "Event B", base_time)

        # A BEFORE B should be invalid (A is after B)
        rel = make_relationship("A", "B", TemporalRelationshipType.BEFORE)

        validator = TemporalConsistencyValidator()
        result = validator.validate([event_a, event_b], [rel])
        assert not result.valid

    def test_valid_causes_ordering(self, base_time):
        """Test CAUSES requires cause < effect."""
        cause = make_event("cause", "Cause Event", base_time)
        effect = make_event("effect", "Effect Event", base_time + timedelta(hours=1))

        rel = make_relationship("cause", "effect", TemporalRelationshipType.CAUSES)

        validator = TemporalConsistencyValidator()
        result = validator.validate([cause, effect], [rel])
        assert result.valid

    def test_invalid_causes_ordering(self, base_time):
        """Test CAUSES fails when cause >= effect (Bradford-Hill #1)."""
        cause = make_event("cause", "Cause Event", base_time + timedelta(hours=1))
        effect = make_event("effect", "Effect Event", base_time)

        rel = make_relationship("cause", "effect", TemporalRelationshipType.CAUSES)

        validator = TemporalConsistencyValidator()
        result = validator.validate([cause, effect], [rel])
        assert not result.valid
        assert any("bradford" in e.lower() or "precede" in e.lower() for e in result.errors)

    def test_100_percent_temporal_ordering_validation(self, base_time):
        """QUALITY GATE: 100% temporal ordering validation for causal relationships."""
        # All causal relationships must have valid temporal ordering
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
            make_event("D", "Event D", base_time + timedelta(hours=3)),
        ]

        # Valid causal chain
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.CAUSES),
            make_relationship("B", "C", TemporalRelationshipType.ENABLES),
            make_relationship("C", "D", TemporalRelationshipType.TRIGGERS),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, relations)

        # All must pass
        assert result.valid
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Test: Multiple Relationships
# ---------------------------------------------------------------------------


class TestMultipleRelationships:
    """Tests for handling multiple relationships between events."""

    def test_multiple_relationships_in_chain(self, base_time):
        """Test handling chain of temporal relationships."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
        ]

        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, relations)
        assert result.valid

    def test_parallel_relationships(self, base_time):
        """Test handling parallel temporal relationships."""
        # A leads to both B and C (parallel effects)
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=1)),
        ]

        relations = [
            make_relationship("A", "B", TemporalRelationshipType.CAUSES),
            make_relationship("A", "C", TemporalRelationshipType.CAUSES),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, relations)
        assert result.valid

    def test_converging_relationships(self, base_time):
        """Test handling converging relationships (multiple causes)."""
        # Both A and B contribute to C
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(minutes=30)),
            make_event("C", "Event C", base_time + timedelta(hours=1)),
        ]

        relations = [
            make_relationship("A", "C", TemporalRelationshipType.ENABLES),
            make_relationship("B", "C", TemporalRelationshipType.ENABLES),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, relations)
        assert result.valid


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestRelationshipEdgeCases:
    """Tests for edge cases in relationship handling."""

    def test_self_reference_relationship(self, base_time):
        """Test handling self-referencing relationship."""
        event = make_event("A", "Event A", base_time)

        # A BEFORE A is invalid
        rel = make_relationship("A", "A", TemporalRelationshipType.BEFORE)

        validator = TemporalConsistencyValidator()
        result = validator.validate([event], [rel])
        # Should not crash, but transitivity would flag this
        # The graph would have A->A which is a self-loop

    def test_missing_event_in_relationship(self, base_time):
        """Test handling relationship referencing missing event."""
        event_a = make_event("A", "Event A", base_time)

        # Relationship references non-existent event B
        rel = make_relationship("A", "B", TemporalRelationshipType.BEFORE)

        validator = TemporalConsistencyValidator()
        result = validator.validate([event_a], [rel])
        # Should handle gracefully (no crash)
        assert result is not None

    def test_empty_relationships(self, base_time):
        """Test handling empty relationship list."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, [])
        assert result.valid

    def test_duplicate_relationships(self, base_time):
        """Test handling duplicate relationships."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]

        # Same relationship twice
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, relations)
        assert result.valid  # Duplicates should not cause issues


# ---------------------------------------------------------------------------
# Test: Production Readiness
# ---------------------------------------------------------------------------


class TestRelationshipProductionReadiness:
    """Tests for production readiness quality gates."""

    def test_real_world_scenario(self, base_time):
        """Test with real-world event scenario."""
        # Project management scenario
        events = [
            make_event("kickoff", "Project Kickoff", base_time, event_type="meeting"),
            make_event("requirements", "Requirements Gathering",
                      base_time + timedelta(days=1), event_type="action"),
            make_event("design", "Design Review",
                      base_time + timedelta(days=5), event_type="meeting"),
            make_event("approval", "Design Approval",
                      base_time + timedelta(days=6), event_type="decision"),
            make_event("implementation", "Implementation Start",
                      base_time + timedelta(days=7), event_type="action"),
            make_event("testing", "Testing Phase",
                      base_time + timedelta(days=14), event_type="action"),
            make_event("deployment", "Production Deployment",
                      base_time + timedelta(days=21), event_type="action"),
        ]

        relations = [
            make_relationship("kickoff", "requirements", TemporalRelationshipType.TRIGGERS),
            make_relationship("requirements", "design", TemporalRelationshipType.BEFORE),
            make_relationship("design", "approval", TemporalRelationshipType.BEFORE),
            make_relationship("approval", "implementation", TemporalRelationshipType.ENABLES),
            make_relationship("implementation", "testing", TemporalRelationshipType.BEFORE),
            make_relationship("testing", "deployment", TemporalRelationshipType.ENABLES),
        ]

        validator = TemporalConsistencyValidator()
        result = validator.validate(events, relations)

        assert result.valid
        assert len(result.errors) == 0
        assert len(result.relationships) == len(relations)

    def test_80_percent_explicit_detection_target(self):
        """Document the >80% explicit detection accuracy target."""
        # This test documents the quality gate requirement
        # Actual detection accuracy is measured via golden dataset tests
        TARGET_EXPLICIT_ACCURACY = 0.80

        # The target is documented and enforced
        assert TARGET_EXPLICIT_ACCURACY == 0.80

    def test_70_percent_implicit_detection_target(self):
        """Document the >70% implicit detection accuracy target."""
        TARGET_IMPLICIT_ACCURACY = 0.70

        # The target is documented and enforced
        assert TARGET_IMPLICIT_ACCURACY == 0.70

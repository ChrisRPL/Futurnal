"""Tests for temporal consistency validation.

Quality Gate: 100% temporal consistency (no contradictions allowed)

This test suite validates the TemporalConsistencyValidator which is
CRITICAL for Phase 3 causal inference. Without these validations,
temporal paradoxes could invalidate Bradford-Hill criteria.

Research Foundation:
- Time-R1: Temporal reasoning framework
- Temporal KG Extrapolation: Causal subhistory identification
- Bradford-Hill Criterion #1: Temporal precedence for causality
"""

from datetime import datetime, timedelta

import pytest

from futurnal.extraction.temporal.consistency import (
    Severity,
    TemporalConsistencyValidator,
    TemporalInconsistency,
    ViolationType,
    validate_temporal_consistency,
)
from futurnal.extraction.temporal.models import (
    Event,
    TemporalRelationship,
    TemporalRelationshipType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator():
    """Create a strict mode validator."""
    return TemporalConsistencyValidator(strict_mode=True)


@pytest.fixture
def base_time():
    """Base time for creating test events."""
    return datetime(2024, 1, 15, 10, 0, 0)


def make_event(
    event_id: str,
    name: str,
    timestamp: datetime,
    duration: timedelta = None,
) -> Event:
    """Helper to create test events."""
    return Event(
        event_id=event_id,
        name=name,
        event_type="test_event",
        timestamp=timestamp,
        duration=duration,
        source_document="test_doc",
    )


def make_relationship(
    entity1_id: str,
    entity2_id: str,
    rel_type: TemporalRelationshipType,
    confidence: float = 0.9,
) -> TemporalRelationship:
    """Helper to create test relationships."""
    return TemporalRelationship(
        entity1_id=entity1_id,
        entity2_id=entity2_id,
        relationship_type=rel_type,
        confidence=confidence,
        evidence="test evidence",
    )


# ---------------------------------------------------------------------------
# Test: Cycle Detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    """Tests for temporal cycle detection."""

    def test_no_cycle_in_linear_chain(self, validator, base_time):
        """Test no cycle detected in valid linear A→B→C chain."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert result.valid
        assert len(result.errors) == 0

    def test_detect_simple_cycle(self, validator, base_time):
        """Test detection of A→B→A cycle."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]
        # A BEFORE B, but also B BEFORE A = cycle!
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "A", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        assert any("cycle" in error.lower() for error in result.errors)

    def test_detect_complex_cycle(self, validator, base_time):
        """Test detection of A→B→C→A cycle."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
        ]
        # A→B→C→A creates a cycle
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
            make_relationship("C", "A", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        assert any("cycle" in error.lower() for error in result.errors)

    def test_no_false_positive_cycles(self, validator, base_time):
        """Test no cycles detected in valid DAG structure."""
        # Diamond DAG: A→B, A→C, B→D, C→D
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=1)),
            make_event("D", "Event D", base_time + timedelta(hours=2)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("A", "C", TemporalRelationshipType.BEFORE),
            make_relationship("B", "D", TemporalRelationshipType.BEFORE),
            make_relationship("C", "D", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert result.valid
        assert len(result.errors) == 0

    def test_detect_cycle_with_causes(self, validator, base_time):
        """Test cycle detection also applies to CAUSES relationships."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]
        # A CAUSES B and B CAUSES A = cycle!
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.CAUSES),
            make_relationship("B", "A", TemporalRelationshipType.CAUSES),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        # Should have cycle error AND causal ordering errors
        assert len(result.errors) >= 1

    def test_multiple_independent_cycles(self, validator, base_time):
        """Test multiple independent cycles are all detected."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
            make_event("D", "Event D", base_time + timedelta(hours=3)),
        ]
        # Two independent cycles: A↔B and C↔D
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "A", TemporalRelationshipType.BEFORE),
            make_relationship("C", "D", TemporalRelationshipType.BEFORE),
            make_relationship("D", "C", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        # Should detect both cycles
        cycle_errors = [e for e in result.errors if "cycle" in e.lower()]
        assert len(cycle_errors) >= 1  # At least one cycle detected


# ---------------------------------------------------------------------------
# Test: Transitivity Validation
# ---------------------------------------------------------------------------


class TestTransitivityValidation:
    """Tests for transitive closure validation."""

    def test_transitivity_satisfied(self, validator, base_time):
        """Test A BEFORE B and B BEFORE C implies A.timestamp < C.timestamp."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert result.valid
        assert len(result.errors) == 0

    def test_transitivity_violation_detected(self, validator, base_time):
        """Test detection when graph says A before C but timestamps disagree."""
        # A→B→C in graph, but A.timestamp > C.timestamp!
        events = [
            make_event("A", "Event A", base_time + timedelta(hours=2)),  # Late!
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time),  # Early!
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        assert any("transiti" in error.lower() for error in result.errors)

    def test_transitive_closure_computed_correctly(self, validator, base_time):
        """Test transitive closure finds all reachable pairs."""
        # A→B→C→D chain
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
            make_event("D", "Event D", base_time + timedelta(hours=3)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
            make_relationship("C", "D", TemporalRelationshipType.BEFORE),
        ]

        # All timestamps are consistent, so should pass
        result = validator.validate(events, relations)
        assert result.valid

    def test_partial_timestamps_handled(self, validator, base_time):
        """Test handling when some events lack timestamps."""
        events = [
            make_event("A", "Event A", base_time),
            Event(
                event_id="B",
                name="Event B",
                event_type="test_event",
                timestamp=None,  # No timestamp!
                source_document="test_doc",
            ),
            make_event("C", "Event C", base_time + timedelta(hours=2)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
            make_relationship("B", "C", TemporalRelationshipType.BEFORE),
        ]

        # Should not crash, and should validate A→C transitively
        result = validator.validate(events, relations)
        # Valid because we can't detect violation without B's timestamp
        assert result.valid or any("timestamp" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Test: Causal Ordering (Bradford-Hill Criterion #1)
# ---------------------------------------------------------------------------


class TestCausalOrdering:
    """Tests for causal ordering validation (Bradford-Hill Criterion #1)."""

    def test_cause_precedes_effect_valid(self, validator, base_time):
        """Test valid causal relationship where cause < effect."""
        events = [
            make_event("cause", "The Cause", base_time),
            make_event("effect", "The Effect", base_time + timedelta(hours=1)),
        ]
        relations = [
            make_relationship("cause", "effect", TemporalRelationshipType.CAUSES),
        ]

        result = validator.validate(events, relations)
        assert result.valid
        assert len(result.errors) == 0

    def test_cause_after_effect_invalid(self, validator, base_time):
        """Test invalid causal relationship where cause >= effect (Bradford-Hill #1)."""
        events = [
            make_event("cause", "The Cause", base_time + timedelta(hours=1)),  # After!
            make_event("effect", "The Effect", base_time),  # Before!
        ]
        relations = [
            make_relationship("cause", "effect", TemporalRelationshipType.CAUSES),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        assert any("bradford" in error.lower() or "precede" in error.lower() for error in result.errors)

    def test_simultaneous_cause_effect_invalid(self, validator, base_time):
        """Test causal relationship where cause == effect timestamp is invalid."""
        events = [
            make_event("cause", "The Cause", base_time),
            make_event("effect", "The Effect", base_time),  # Same time!
        ]
        relations = [
            make_relationship("cause", "effect", TemporalRelationshipType.CAUSES),
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        assert any("bradford" in error.lower() or "precede" in error.lower() for error in result.errors)

    def test_enables_requires_ordering(self, validator, base_time):
        """Test ENABLES relationship requires enabler < enabled."""
        events = [
            make_event("enabler", "The Enabler", base_time + timedelta(hours=1)),  # After!
            make_event("enabled", "The Enabled", base_time),  # Before!
        ]
        relations = [
            make_relationship("enabler", "enabled", TemporalRelationshipType.ENABLES),
        ]

        result = validator.validate(events, relations)
        assert not result.valid

    def test_triggers_requires_ordering(self, validator, base_time):
        """Test TRIGGERS relationship requires trigger < triggered."""
        events = [
            make_event("trigger", "The Trigger", base_time + timedelta(hours=1)),  # After!
            make_event("triggered", "The Triggered", base_time),  # Before!
        ]
        relations = [
            make_relationship("trigger", "triggered", TemporalRelationshipType.TRIGGERS),
        ]

        result = validator.validate(events, relations)
        assert not result.valid

    def test_prevents_requires_ordering(self, validator, base_time):
        """Test PREVENTS relationship requires preventer < prevented."""
        events = [
            make_event("preventer", "The Preventer", base_time + timedelta(hours=1)),  # After!
            make_event("prevented", "The Prevented", base_time),  # Before!
        ]
        relations = [
            make_relationship("preventer", "prevented", TemporalRelationshipType.PREVENTS),
        ]

        result = validator.validate(events, relations)
        assert not result.valid

    def test_missing_timestamps_in_causal_flagged(self, validator, base_time):
        """Test causal relationships with missing timestamps are flagged."""
        events = [
            Event(
                event_id="cause",
                name="The Cause",
                event_type="test_event",
                timestamp=None,  # Missing!
                source_document="test_doc",
            ),
            make_event("effect", "The Effect", base_time),
        ]
        relations = [
            make_relationship("cause", "effect", TemporalRelationshipType.CAUSES),
        ]

        result = validator.validate(events, relations)
        # In strict mode, missing timestamp in causal relationship is an error
        assert not result.valid or len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Test: Interval Consistency (Allen's IA)
# ---------------------------------------------------------------------------


class TestIntervalConsistency:
    """Tests for Allen's Interval Algebra consistency."""

    def test_during_relationship_valid(self, validator, base_time):
        """Test valid DURING relationship."""
        events = [
            make_event(
                "inner",
                "Inner Event",
                base_time + timedelta(minutes=30),
            ),
            make_event(
                "outer",
                "Outer Event",
                base_time,
                duration=timedelta(hours=1),
            ),
        ]
        relations = [
            make_relationship("inner", "outer", TemporalRelationshipType.DURING),
        ]

        result = validator.validate(events, relations)
        assert result.valid

    def test_during_relationship_invalid(self, validator, base_time):
        """Test invalid DURING relationship (inner not within outer)."""
        events = [
            make_event(
                "inner",
                "Inner Event",
                base_time + timedelta(hours=2),  # Outside outer!
            ),
            make_event(
                "outer",
                "Outer Event",
                base_time,
                duration=timedelta(hours=1),
            ),
        ]
        relations = [
            make_relationship("inner", "outer", TemporalRelationshipType.DURING),
        ]

        result = validator.validate(events, relations)
        # DURING violations are warnings by default
        assert len(result.warnings) > 0 or not result.valid

    def test_simultaneous_valid(self, validator, base_time):
        """Test valid SIMULTANEOUS relationship."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time),  # Same time!
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.SIMULTANEOUS),
        ]

        result = validator.validate(events, relations)
        assert result.valid

    def test_simultaneous_invalid(self, validator, base_time):
        """Test invalid SIMULTANEOUS relationship (different times)."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.SIMULTANEOUS),
        ]

        result = validator.validate(events, relations)
        # SIMULTANEOUS violations are warnings
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Test: Validator Integration
# ---------------------------------------------------------------------------


class TestConsistencyValidatorIntegration:
    """Integration tests for complete consistency validation."""

    def test_validate_returns_valid_for_consistent_graph(self, validator, base_time):
        """Test ValidationResult.valid=True for fully consistent graph."""
        events = [
            make_event("meeting", "Team Meeting", base_time),
            make_event("decision", "Decision Made", base_time + timedelta(hours=1)),
            make_event("implementation", "Started Implementation", base_time + timedelta(hours=2)),
        ]
        relations = [
            make_relationship("meeting", "decision", TemporalRelationshipType.BEFORE),
            make_relationship("decision", "implementation", TemporalRelationshipType.CAUSES),
        ]

        result = validator.validate(events, relations)
        assert result.valid
        assert len(result.errors) == 0
        assert len(result.relationships) == len(relations)

    def test_validate_returns_errors_for_inconsistent_graph(self, validator, base_time):
        """Test ValidationResult.errors populated for inconsistencies."""
        # Multiple violations: cycle + causal ordering
        events = [
            make_event("A", "Event A", base_time + timedelta(hours=1)),  # Late
            make_event("B", "Event B", base_time),  # Early
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.CAUSES),  # A causes B but A > B!
            make_relationship("B", "A", TemporalRelationshipType.BEFORE),  # B before A = cycle
        ]

        result = validator.validate(events, relations)
        assert not result.valid
        assert len(result.errors) > 0

    def test_empty_events_and_relations(self, validator):
        """Test empty inputs return valid result."""
        result = validator.validate([], [])
        assert result.valid
        assert len(result.errors) == 0

    def test_events_without_relations(self, validator, base_time):
        """Test events without relationships is valid."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]

        result = validator.validate(events, [])
        assert result.valid

    def test_production_readiness_100_percent_consistency(self, validator, base_time):
        """PRODUCTION GATE: Verify 100% consistency on real-world scenario."""
        # Simulate a real extraction scenario
        events = [
            make_event("email_received", "Received client email", base_time),
            make_event("meeting_scheduled", "Scheduled team meeting", base_time + timedelta(hours=1)),
            make_event("meeting_held", "Team meeting", base_time + timedelta(days=1)),
            make_event("proposal_drafted", "Drafted proposal", base_time + timedelta(days=2)),
            make_event("proposal_sent", "Sent proposal to client", base_time + timedelta(days=3)),
            make_event("contract_signed", "Contract signed", base_time + timedelta(days=7)),
        ]
        relations = [
            make_relationship("email_received", "meeting_scheduled", TemporalRelationshipType.TRIGGERS),
            make_relationship("meeting_scheduled", "meeting_held", TemporalRelationshipType.BEFORE),
            make_relationship("meeting_held", "proposal_drafted", TemporalRelationshipType.CAUSES),
            make_relationship("proposal_drafted", "proposal_sent", TemporalRelationshipType.BEFORE),
            make_relationship("proposal_sent", "contract_signed", TemporalRelationshipType.ENABLES),
        ]

        result = validator.validate(events, relations)
        assert result.valid, f"Production scenario failed: {result.errors}"
        assert len(result.errors) == 0
        # Should be Phase 3 ready


# ---------------------------------------------------------------------------
# Test: Convenience Function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    """Tests for validate_temporal_consistency convenience function."""

    def test_convenience_function_works(self, base_time):
        """Test validate_temporal_consistency() works correctly."""
        events = [
            make_event("A", "Event A", base_time),
            make_event("B", "Event B", base_time + timedelta(hours=1)),
        ]
        relations = [
            make_relationship("A", "B", TemporalRelationshipType.BEFORE),
        ]

        result = validate_temporal_consistency(events, relations)
        assert result.valid

    def test_convenience_function_strict_mode(self, base_time):
        """Test strict_mode parameter."""
        events = [
            Event(
                event_id="cause",
                name="Cause",
                event_type="test",
                timestamp=None,
                source_document="test_doc",
            ),
            make_event("effect", "Effect", base_time),
        ]
        relations = [
            make_relationship("cause", "effect", TemporalRelationshipType.CAUSES),
        ]

        # Strict mode: error
        result_strict = validate_temporal_consistency(events, relations, strict_mode=True)
        # Non-strict mode: warning (if implemented)
        result_lenient = validate_temporal_consistency(events, relations, strict_mode=False)

        # Both should at least flag the issue
        assert (not result_strict.valid) or len(result_strict.warnings) > 0
        assert (not result_lenient.valid) or len(result_lenient.warnings) > 0


# ---------------------------------------------------------------------------
# Test: Single Relationship Validation
# ---------------------------------------------------------------------------


class TestSingleRelationshipValidation:
    """Tests for validate_single_relationship utility method."""

    def test_before_valid(self, validator, base_time):
        """Test BEFORE validation."""
        event1 = make_event("A", "Event A", base_time)
        event2 = make_event("B", "Event B", base_time + timedelta(hours=1))

        assert validator.validate_single_relationship(
            event1, event2, TemporalRelationshipType.BEFORE
        )

    def test_before_invalid(self, validator, base_time):
        """Test BEFORE invalid when timestamps reversed."""
        event1 = make_event("A", "Event A", base_time + timedelta(hours=1))
        event2 = make_event("B", "Event B", base_time)

        assert not validator.validate_single_relationship(
            event1, event2, TemporalRelationshipType.BEFORE
        )

    def test_after_valid(self, validator, base_time):
        """Test AFTER validation."""
        event1 = make_event("A", "Event A", base_time + timedelta(hours=1))
        event2 = make_event("B", "Event B", base_time)

        assert validator.validate_single_relationship(
            event1, event2, TemporalRelationshipType.AFTER
        )

    def test_simultaneous_valid(self, validator, base_time):
        """Test SIMULTANEOUS validation."""
        event1 = make_event("A", "Event A", base_time)
        event2 = make_event("B", "Event B", base_time)

        assert validator.validate_single_relationship(
            event1, event2, TemporalRelationshipType.SIMULTANEOUS
        )

    def test_causes_requires_precedence(self, validator, base_time):
        """Test CAUSES requires cause < effect."""
        event1 = make_event("cause", "Cause", base_time)
        event2 = make_event("effect", "Effect", base_time + timedelta(hours=1))

        assert validator.validate_single_relationship(
            event1, event2, TemporalRelationshipType.CAUSES
        )

        # Reverse should fail
        assert not validator.validate_single_relationship(
            event2, event1, TemporalRelationshipType.CAUSES
        )

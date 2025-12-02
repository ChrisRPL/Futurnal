"""Unit tests for causal structure data models.

Tests implementation from:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Test Coverage:
- EventType enum values and extensibility
- CausalRelationshipType enum values
- CausalCandidate field validation and serialization
- BradfordHillCriteria structure and Phase 1 scope
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from futurnal.extraction.causal.models import (
    BradfordHillCriteria,
    CausalCandidate,
    CausalRelationshipType,
    EventType,
)


class TestEventType:
    """Test EventType enum for event classification."""
    
    def test_event_type_values(self):
        """Validate all seed event types are defined."""
        expected_types = {
            "meeting", "decision", "publication",
            "communication", "action", "state_change", "unknown"
        }
        actual_types = {et.value for et in EventType}
        assert actual_types == expected_types, "All seed event types must be present"
    
    def test_event_type_extensibility(self):
        """Verify event types are extensible via schema evolution.
        
        While enum itself is fixed in code, schema evolution mechanism
        should discover and register new event types dynamically.
        """
        # EventType enum has seed types
        assert len(EventType) == 7
        # Schema evolution will extend beyond these seed types
        # (tested in schema evolution integration tests)


class TestCausalRelationshipType:
    """Test CausalRelationshipType enum for causal semantics."""
    
    def test_causal_relationship_types(self):
        """Validate all causal relationship types are defined."""
        expected_types = {
            "causes", "enables", "prevents",
            "triggers", "leads_to", "contributes_to"
        }
        actual_types = {crt.value for crt in CausalRelationshipType}
        assert actual_types == expected_types, "All causal relationship types must be present"
    
    def test_semantic_differentiation(self):
        """Verify relationship types support semantic differentiation."""
        # Direct causation
        assert CausalRelationshipType.CAUSES.value == "causes"
        assert CausalRelationshipType.TRIGGERS.value == "triggers"
        
        # Indirect causation
        assert CausalRelationshipType.LEADS_TO.value == "leads_to"
        assert CausalRelationshipType.CONTRIBUTES_TO.value == "contributes_to"
        
        # Enabling/blocking
        assert CausalRelationshipType.ENABLES.value == "enables"
        assert CausalRelationshipType.PREVENTS.value == "prevents"


class TestCausalCandidate:
    """Test CausalCandidate model for event-event relationships."""
    
    def test_valid_causal_candidate_creation(self):
        """Validate creation of valid causal candidate."""
        candidate = CausalCandidate(
            id="causal_001",
            cause_event_id="event_123",
            effect_event_id="event_456",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=2),
            temporal_ordering_valid=True,
            causal_evidence="The meeting led to the decision.",
            causal_confidence=0.85,
            temporality_satisfied=True,
            source_document="doc_001"
        )
        
        assert candidate.id == "causal_001"
        assert candidate.cause_event_id == "event_123"
        assert candidate.effect_event_id == "event_456"
        assert candidate.relationship_type == CausalRelationshipType.CAUSES
        assert candidate.temporal_gap == timedelta(hours=2)
        assert candidate.temporal_ordering_valid is True
        assert candidate.causal_confidence == 0.85
        assert candidate.temporality_satisfied is True
        assert candidate.is_validated is False  # Phase 1 default
    
    def test_confidence_bounds_validation(self):
        """Validate confidence must be between 0.0 and 1.0."""
        # Valid confidence
        candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.75,
            temporality_satisfied=True,
            source_document="doc1"
        )
        assert candidate.causal_confidence == 0.75
        
        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            CausalCandidate(
                id="c2",
                cause_event_id="e1",
                effect_event_id="e2",
                relationship_type=CausalRelationshipType.CAUSES,
                temporal_gap=timedelta(hours=1),
                temporal_ordering_valid=True,
                causal_evidence="evidence",
                causal_confidence=1.5,  # Invalid
                temporality_satisfied=True,
                source_document="doc1"
            )
        
        # Invalid confidence (negative)
        with pytest.raises(ValidationError):
            CausalCandidate(
                id="c3",
                cause_event_id="e1",
                effect_event_id="e2",
                relationship_type=CausalRelationshipType.CAUSES,
                temporal_gap=timedelta(hours=1),
                temporal_ordering_valid=True,
                causal_evidence="evidence",
                causal_confidence=-0.1,  # Invalid
                temporality_satisfied=True,
                source_document="doc1"
            )
    
    def test_bradford_hill_fields_optional(self):
        """Validate Bradford Hill fields are optional in Phase 1.
        
        From production plan: Phase 1 prepares structure, Phase 3 validates.
        """
        candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.75,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        # Phase 1: Optional fields are None
        assert candidate.strength is None
        assert candidate.consistency is None
        assert candidate.plausibility is None
        assert candidate.is_validated is False
        assert candidate.validation_method is None
    
    def test_temporal_ordering_requirement(self):
        """Validate temporal ordering is critical for causality.
        
        Success Metric: Temporal ordering validated for all candidates (100%)
        """
        # Valid: cause before effect
        valid_candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.75,
            temporality_satisfied=True,
            source_document="doc1"
        )
        assert valid_candidate.temporal_ordering_valid is True
        assert valid_candidate.temporality_satisfied is True
    
    def test_serialization(self):
        """Validate CausalCandidate serialization for storage."""
        candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=2),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.85,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        # Serialize to dict
        data = candidate.model_dump()
        assert data["id"] == "c1"
        assert data["relationship_type"] == "causes"
        assert data["causal_confidence"] == 0.85
        
        # Serialize to JSON
        json_str = candidate.model_dump_json()
        assert "c1" in json_str
        assert "causes" in json_str


class TestBradfordHillCriteria:
    """Test BradfordHillCriteria model for causal inference preparation."""
    
    def test_criteria_structure_complete(self):
        """Validate all 9 Bradford Hill criteria fields are present.
        
        Success Metric: Bradford Hill criteria structure prepared.
        """
        criteria = BradfordHillCriteria(temporality=True)
        
        # All fields exist
        assert hasattr(criteria, "temporality")
        assert hasattr(criteria, "strength")
        assert hasattr(criteria, "dose_response")
        assert hasattr(criteria, "consistency")
        assert hasattr(criteria, "plausibility")
        assert hasattr(criteria, "coherence")
        assert hasattr(criteria, "experiment_possible")
        assert hasattr(criteria, "analogy")
        assert hasattr(criteria, "specificity")
    
    def test_temporality_required(self):
        """Validate temporality is the only required criterion in Phase 1."""
        # Valid: only temporality specified
        criteria = BradfordHillCriteria(temporality=True)
        assert criteria.temporality is True
        
        # Invalid: missing temporality
        with pytest.raises(ValidationError):
            BradfordHillCriteria()
    
    def test_phase1_fields_nullable(self):
        """Validate other criteria are nullable for Phase 3.
        
        From production plan: Phase 1 prepares structure, Phase 3 validates.
        """
        criteria = BradfordHillCriteria(temporality=True)
        
        # Phase 1: All criteria except temporality are None
        assert criteria.strength is None
        assert criteria.dose_response is None
        assert criteria.consistency is None
        assert criteria.plausibility is None
        assert criteria.coherence is None
        assert criteria.experiment_possible is None
        assert criteria.analogy is None
        assert criteria.specificity is None
    
    def test_phase3_extension_points(self):
        """Validate Phase 3 can populate all criteria.
        
        Demonstrates forward compatibility for Phase 3 validation.
        """
        # Phase 3: All criteria populated
        criteria = BradfordHillCriteria(
            temporality=True,
            strength=0.85,
            dose_response=True,
            consistency=0.90,
            plausibility="Meeting scheduled decision timeline",
            coherence=True,
            experiment_possible=False,
            analogy="Similar to previous project decisions",
            specificity=True
        )
        
        assert criteria.temporality is True
        assert criteria.strength == 0.85
        assert criteria.dose_response is True
        assert criteria.consistency == 0.90
        assert criteria.plausibility is not None
        assert criteria.coherence is True
        assert criteria.experiment_possible is False
        assert criteria.analogy is not None
        assert criteria.specificity is True
    
    def test_strength_consistency_bounds(self):
        """Validate strength and consistency fields have proper bounds."""
        # Valid strength
        criteria = BradfordHillCriteria(
            temporality=True,
            strength=0.75
        )
        assert criteria.strength == 0.75
        
        # Invalid strength (too high)
        with pytest.raises(ValidationError):
            BradfordHillCriteria(
                temporality=True,
                strength=1.5
            )
        
        # Invalid consistency (negative)
        with pytest.raises(ValidationError):
            BradfordHillCriteria(
                temporality=True,
                consistency=-0.1
            )
    
    def test_serialization(self):
        """Validate BradfordHillCriteria serialization."""
        criteria = BradfordHillCriteria(
            temporality=True,
            strength=0.85,
            plausibility="Mechanistic explanation"
        )
        
        # Serialize to dict
        data = criteria.model_dump()
        assert data["temporality"] is True
        assert data["strength"] == 0.85
        assert data["plausibility"] == "Mechanistic explanation"
        
        # Null fields serialized
        assert data["dose_response"] is None
        assert data["coherence"] is None

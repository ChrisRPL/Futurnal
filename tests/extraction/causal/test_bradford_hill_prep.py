"""Unit tests for Bradford Hill preparation.

Tests implementation from:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Test Coverage:
- Bradford Hill criteria preparation
- Temporality validation
- Phase 1 scope (structure only)
- PKG context parameter support
"""

from datetime import datetime, timedelta

import pytest

from futurnal.extraction.causal.bradford_hill_prep import BradfordHillPreparation
from futurnal.extraction.causal.models import (
    CausalCandidate,
    CausalRelationshipType,
)


class TestBradfordHillPreparation:
    """Test BradfordHillPreparation class."""
    
    def test_initialization(self):
        """Validate BradfordHillPreparation can be instantiated."""
        prep = BradfordHillPreparation()
        
        assert prep is not None
    
    def test_prepare_for_validation_basic(self):
        """Validate basic preparation functionality."""
        prep = BradfordHillPreparation()
        
        # Create valid causal candidate
        candidate = CausalCandidate(
            id="test_causal_001",
            cause_event_id="event_1",
            effect_event_id="event_2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=2),
            temporal_ordering_valid=True,
            causal_evidence="The meeting caused the decision.",
            causal_confidence=0.85,
            temporality_satisfied=True,
            source_document="doc_001"
        )
        
        criteria = prep.prepare_for_validation(candidate)
        
        assert criteria is not None
        assert criteria.temporality is True
    
    def test_temporality_from_candidate(self):
        """Validate temporality correctly set from candidate.
        
        Success Metric: Bradford Hill temporality validated.
        """
        prep = BradfordHillPreparation()
        
        # Candidate with valid temporal ordering
        valid_candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.8,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        criteria = prep.prepare_for_validation(valid_candidate)
        
        assert criteria.temporality is True
    
    def test_temporality_invalid_when_ordering_invalid(self):
        """Validate temporality is False when temporal ordering invalid."""
        prep = BradfordHillPreparation()
        
        # Candidate with invalid temporal ordering
        invalid_candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=False,  # Invalid ordering
            causal_evidence="evidence",
            causal_confidence=0.8,
            temporality_satisfied=False,  # Not satisfied
            source_document="doc1"
        )
        
        criteria = prep.prepare_for_validation(invalid_candidate)
        
        assert criteria.temporality is False
    
    def test_other_criteria_null_in_phase1(self):
        """Validate other criteria are None in Phase 1.
        
        From production plan: Phase 1 prepares structure, Phase 3 validates.
        """
        prep = BradfordHillPreparation()
        
        candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.8,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        criteria = prep.prepare_for_validation(candidate)
        
        # Phase 1: Only temporality is populated
        assert criteria.temporality is True
        # All other criteria are None (Phase 3 scope)
        assert criteria.strength is None
        assert criteria.dose_response is None
        assert criteria.consistency is None
        assert criteria.plausibility is None
        assert criteria.coherence is None
        assert criteria.experiment_possible is None
        assert criteria.analogy is None
        assert criteria.specificity is None
    
    def test_with_pkg_context(self):
        """Validate PKG context parameter works.
        
        Note: PKG context not used in Phase 1, but parameter should work.
        """
        prep = BradfordHillPreparation()
        
        candidate = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.8,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        # PKG context (for future Phase 3 use)
        pkg_context = {
            "related_entities": ["entity1", "entity2"],
            "historical_patterns": []
        }
        
        criteria = prep.prepare_for_validation(candidate, pkg_context)
        
        # Should work even with context (not used in Phase 1)
        assert criteria is not None
        assert criteria.temporality is True
    
    def test_validate_temporality_helper(self):
        """Validate temporality validation helper method."""
        prep = BradfordHillPreparation()
        
        # Valid candidate
        valid = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence",
            causal_confidence=0.8,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        # Invalid candidate
        invalid = CausalCandidate(
            id="c2",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=False,
            causal_evidence="evidence",
            causal_confidence=0.8,
            temporality_satisfied=False,
            source_document="doc1"
        )
        
        assert prep.validate_temporality(valid) is True
        assert prep.validate_temporality(invalid) is False
    
    def test_multiple_preparations(self):
        """Validate multiple preparations work independently."""
        prep = BradfordHillPreparation()
        
        candidate1 = CausalCandidate(
            id="c1",
            cause_event_id="e1",
            effect_event_id="e2",
            relationship_type=CausalRelationshipType.CAUSES,
            temporal_gap=timedelta(hours=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence1",
            causal_confidence=0.8,
            temporality_satisfied=True,
            source_document="doc1"
        )
        
        candidate2 = CausalCandidate(
            id="c2",
            cause_event_id="e3",
            effect_event_id="e4",
            relationship_type=CausalRelationshipType.ENABLES,
            temporal_gap=timedelta(days=1),
            temporal_ordering_valid=True,
            causal_evidence="evidence2",
            causal_confidence=0.9,
            temporality_satisfied=True,
            source_document="doc2"
        )
        
        criteria1 = prep.prepare_for_validation(candidate1)
        criteria2 = prep.prepare_for_validation(candidate2)
        
        # Both should work independently
        assert criteria1.temporality is True
        assert criteria2.temporality is True

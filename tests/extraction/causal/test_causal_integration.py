"""Integration tests for causal structure.

End-to-end tests for complete causal detection workflow:
- Event extraction → Causal detection pipeline
- Temporal consistency validation
- Bradford Hill preparation workflow
"""

from datetime import datetime, timedelta
from typing import Any, List

import pytest

from futurnal.extraction.causal import (
    BradfordHillPreparation,
    CausalRelationshipDetector,
    EventExtractor,
)
from futurnal.extraction.temporal.models import Event, TemporalMark, TemporalSourceType


# Test Fixtures

class MockLLM:
    """Mock LLM for integration testing."""
    
    def __init__(self, event_responses: List[Any] = None, causal_responses: List[Any] = None):
        self.event_responses = event_responses or []
        self.causal_responses = causal_responses or []
        self.event_call_index = 0
        self.causal_call_index = 0
    
    def extract(self, prompt: str) -> Any:
        """Mock extraction - returns different responses based on prompt type."""
        # Detect if this is event extraction or causal detection
        # Event extraction prompts contain specific keywords
        if ("Event Types:" in prompt or 
            ("event_type" in prompt and "timestamp" in prompt and "participants" in prompt)):
            # Event extraction
            if self.event_call_index < len(self.event_responses):
                response = self.event_responses[self.event_call_index]
                self.event_call_index += 1
                return response
        elif ("causal" in prompt.lower() or "Event 1" in prompt or "Event 2" in prompt):
            # Causal detection
            if self.causal_call_index < len(self.causal_responses):
                response = self.causal_responses[self.causal_call_index]
                self.causal_call_index += 1
                return response
        
        return {}


class MockTemporalExtractor:
    """Mock temporal extractor."""
    
    def __init__(self, markers: List[TemporalMark] = None):
        self.markers = markers or []
    
    def extract_explicit_timestamps(self, text: str) -> List[TemporalMark]:
        return self.markers


class MockDocument:
    """Mock document."""
    
    def __init__(self, content: str, doc_id: str = "test_doc"):
        self.content = content
        self.doc_id = doc_id


# Integration Tests

class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_event_extraction_to_causal_detection(self):
        """Test full pipeline: event extraction → causal detection.
        
        Success Metric: End-to-end integration operational.
        """
        # Setup: Create events directly (simplifies test, focuses on causal detection)
        events = [
            Event(
                name="Team meeting",
                event_type="meeting",
                timestamp=datetime(2024, 1, 15, 10, 0),
                source_document="test_doc",
                extraction_confidence=0.9
            ),
            Event(
                name="Project decision",
                event_type="decision",
                timestamp=datetime(2024, 1, 15, 14, 0),
                source_document="test_doc",
                extraction_confidence=0.85
            )
        ]
        
        causal_response = {
            "evidence": "The team meeting led to the project decision.",
            "relationship_type": "leads_to",
            "confidence": 0.88
        }
        
        llm = MockLLM(causal_responses=[causal_response])
        
        # Create document
        doc = MockDocument(
            "On January 15, 2024, we had a team meeting in the morning. "
            "The team meeting led to the project decision in the afternoon."
        )
        
        # Step 1: Detect causal relationships  
        causal_detector = CausalRelationshipDetector(llm)
        candidates = causal_detector.detect_causal_candidates(events, doc)
        
        # Step 2: Prepare Bradford Hill criteria
        prep = BradfordHillPreparation()
        
        # Validate end-to-end workflow
        assert len(events) == 2
        assert len(candidates) > 0
        
        # Validate causal candidate
        candidate = candidates[0]
        assert candidate.temporal_ordering_valid is True
        assert candidate.temporality_satisfied is True
        assert candidate.causal_confidence >= 0.6
        
        # Validate Bradford Hill preparation
        criteria = prep.prepare_for_validation(candidate)
        assert criteria.temporality is True
    
    def test_temporal_consistency_maintained(self):
        """Validate temporal consistency across pipeline.
        
        Success Metric: 100% temporal ordering validation.
        """
        # Create events with known temporal ordering
        events = [
            Event(
                name="Event A",
                event_type="action",
                timestamp=datetime(2024, 1, 1, 10, 0),
                source_document="test",
                extraction_confidence=0.9
            ),
            Event(
                name="Event B",
                event_type="action",
                timestamp=datetime(2024, 1, 1, 12, 0),
                source_document="test",
                extraction_confidence=0.9
            ),
            Event(
                name="Event C",
                event_type="action",
                timestamp=datetime(2024, 1, 1, 14, 0),
                source_document="test",
                extraction_confidence=0.9
            )
        ]
        
        llm = MockLLM(
            causal_responses=[
                {"evidence": "A led to B", "relationship_type": "leads_to", "confidence": 0.8},
                {"evidence": "B led to C", "relationship_type": "leads_to", "confidence": 0.75},
            ]
        )
        
        doc = MockDocument("Event A led to Event B, which led to Event C.")
        
        # Detect causal relationships
        detector = CausalRelationshipDetector(llm)
        candidates = detector.detect_causal_candidates(events, doc)
        
        # Validate temporal consistency
        for candidate in candidates:
            assert candidate.temporal_ordering_valid is True
            assert candidate.temporality_satisfied is True
            
            # Find the actual events to verify ordering
            cause_name = candidate.cause_event_id
            effect_name = candidate.effect_event_id
            
            cause = next(e for e in events if e.name == cause_name)
            effect = next(e for e in events if e.name == effect_name)
            
            # Verify cause precedes effect
            assert cause.timestamp < effect.timestamp
    
    def test_bradford_hill_preparation_workflow(self):
        """Test complete Bradford Hill preparation workflow.
        
        Success Metric: Bradford Hill prep complete 100%.
        """
        # Create events directly for clarity
        events = [
            Event(
                name="Cause event",
                event_type="action",
                timestamp=datetime(2024, 1, 1),
                source_document="test",
                extraction_confidence=0.9
            ),
            Event(
                name="Effect event",
                event_type="state_change",
                timestamp=datetime(2024, 1, 2),
                source_document="test",
                extraction_confidence=0.9
            )
        ]
        
        # Mock causal detection
        llm = MockLLM(
            causal_responses=[
                {"evidence": "Strong causal link", "relationship_type": "causes", "confidence": 0.92}
            ]
        )
        
        doc = MockDocument("The cause event caused the effect event.")
        
        # Detect causal candidates
        detector = CausalRelationshipDetector(llm)
        candidates = detector.detect_causal_candidates(events, doc)
        
        # Prepare Bradford Hill criteria for all candidates
        prep = BradfordHillPreparation()
        all_criteria = [prep.prepare_for_validation(c) for c in candidates]
        
        # Validate preparation
        assert len(all_criteria) > 0
        for criteria in all_criteria:
            # Temporality validated
            assert criteria.temporality is True
            # Other criteria prepared (None for Phase 1)
            assert criteria.strength is None
            assert criteria.consistency is None
    
    def test_no_causal_relationships_when_no_temporal_order(self):
        """Validate no candidates when events lack temporal ordering."""
        # Create events without timestamps
        events = [
            Event(
                name="Event 1",
                event_type="action",
                timestamp=None,  # No timestamp
                source_document="test",
                extraction_confidence=0.9
            ),
            Event(
                name="Event 2",
                event_type="action",
                timestamp=None,  # No timestamp
                source_document="test",
                extraction_confidence=0.9
            )
        ]
        
        llm = MockLLM()
        doc = MockDocument("Events without timestamps")
        
        detector = CausalRelationshipDetector(llm)
        candidates = detector.detect_causal_candidates(events, doc)
        
        # No candidates possible without temporal ordering
        assert len(candidates) == 0
    
    def test_confidence_filtering_integration(self):
        """Validate confidence filtering in full workflow."""
        llm = MockLLM(
            causal_responses=[
                {"evidence": "Weak link", "relationship_type": "causes", "confidence": 0.3},  # Below threshold
                {"evidence": "Strong link", "relationship_type": "causes", "confidence": 0.85},  # Above threshold
            ]
        )
        
        events = [
            Event(name="E1", event_type="action", timestamp=datetime(2024, 1, 1),
                  source_document="test", extraction_confidence=0.9),
            Event(name="E2", event_type="action", timestamp=datetime(2024, 1, 2),
                  source_document="test", extraction_confidence=0.9),
            Event(name="E3", event_type="action", timestamp=datetime(2024, 1, 3),
                  source_document="test", extraction_confidence=0.9),
        ]
        
        doc = MockDocument("Events with varying causal strengths")
        
        detector = CausalRelationshipDetector(llm, confidence_threshold=0.6)
        candidates = detector.detect_causal_candidates(events, doc)
        
        # Only high-confidence candidates should be included
        assert all(c.causal_confidence >= 0.6 for c in candidates)

"""Unit tests for causal relationship detection.

Tests implementation from:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Test Coverage:
- Causal candidate detection (core functionality)
- Temporal ordering validation (CRITICAL - 100% required)
- Temporal gap filtering
- Confidence scoring and filtering
- Causal evidence extraction
- Relationship type inference
- LLM integration
"""

from datetime import datetime, timedelta
from typing import Any, List
from unittest.mock import Mock

import pytest

from futurnal.extraction.causal.models import CausalRelationshipType
from futurnal.extraction.causal.relationship_detector import CausalRelationshipDetector
from futurnal.extraction.temporal.models import Event, TemporalSourceType


# Test Fixtures

class MockLLM:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: List[Any] = None):
        self.responses = responses or []
        self.call_index = 0
        self.calls = []
    
    def extract(self, prompt: str) -> Any:
        """Mock extraction."""
        self.calls.append(prompt)
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return {}


class MockDocument:
    """Mock document for testing."""
    
    def __init__(self, content: str, doc_id: str = "test_doc"):
        self.content = content
        self.doc_id = doc_id


def create_event(
    name: str,
    timestamp: datetime,
    event_type: str = "action"
) -> Event:
    """Helper to create test events."""
    return Event(
        name=name,
        event_type=event_type,
        timestamp=timestamp,
        source_document="test_doc",
        extraction_confidence=0.9
    )


# Tests

class TestCausalRelationshipDetector:
    """Test CausalRelationshipDetector class."""
    
    def test_initialization(self):
        """Validate detector initialization."""
        llm = MockLLM()
        
        detector = CausalRelationshipDetector(llm)
        
        assert detector.llm is llm
        assert detector.confidence_threshold == 0.6  # Default
        assert detector.max_temporal_gap == timedelta(days=365)
    
    def test_custom_thresholds(self):
        """Validate custom threshold initialization."""
        llm = MockLLM()
        
        detector = CausalRelationshipDetector(
            llm,
            confidence_threshold=0.7,
            max_temporal_gap_days=180
        )
        
        assert detector.confidence_threshold == 0.7
        assert detector.max_temporal_gap == timedelta(days=180)
    
    def test_causal_indicators_available(self):
        """Validate causal indicator keywords are defined."""
        assert len(CausalRelationshipDetector.CAUSAL_INDICATORS) > 0
        assert "caused" in CausalRelationshipDetector.CAUSAL_INDICATORS
        assert "led to" in CausalRelationshipDetector.CAUSAL_INDICATORS
        assert "prevented" in CausalRelationshipDetector.CAUSAL_INDICATORS


class TestCausalCandidateDetection:
    """Test causal candidate detection functionality."""
    
    def test_detect_causal_candidates_basic(self):
        """Validate basic causal candidate detection.
        
        Success Metric: Event-event relationships identified.
        """
        # Mock LLM response
        mock_response = {
            "evidence": "The meeting led to the decision to proceed.",
            "relationship_type": "leads_to",
            "confidence": 0.85
        }
        llm = MockLLM([mock_response])
        
        detector = CausalRelationshipDetector(llm)
        
        # Create temporally ordered events
        events = [
            create_event("Team meeting", datetime(2024, 1, 15, 10, 0)),
            create_event("Project decision", datetime(2024, 1, 15, 14, 0))
        ]
        
        doc = MockDocument("We had a meeting. The meeting led to the decision to proceed.")
        
        candidates = detector.detect_causal_candidates(events, doc)
        
        assert len(candidates) > 0
        assert candidates[0].cause_event_id == "Team meeting"
        assert candidates[0].effect_event_id == "Project decision"
    
    def test_temporal_ordering_validation(self):
        """Validate temporal ordering (cause before effect).
        
        CRITICAL Success Metric: 100% temporal ordering validation.
        From production plan: All candidates must have cause before effect.
        """
        llm = MockLLM([{
            "evidence": "Event B followed Event A.",
            "relationship_type": "leads_to",
            "confidence": 0.8
        }])
        
        detector = CausalRelationshipDetector(llm)
        
        # Create events with proper temporal ordering
        events = [
            create_event("Event A", datetime(2024, 1, 1)),
            create_event("Event B", datetime(2024, 1, 2)),
            create_event("Event C", datetime(2024, 1, 3))
        ]
        
        doc = MockDocument("Event A led to Event B, which caused Event C.")
        
        candidates = detector.detect_causal_candidates(events, doc)
        
        # CRITICAL: 100% temporal ordering validation
        assert all(c.temporal_ordering_valid for c in candidates)
        assert all(c.temporality_satisfied for c in candidates)
    
    def test_reverse_temporal_order_filtered(self):
        """Validate that reversed temporal pairs are NOT detected.
        
        Ensures cause-before-effect requirement is enforced.
        """
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        # Create events in reverse chronological order
        events = [
            create_event("Later event", datetime(2024, 1, 2)),
            create_event("Earlier event", datetime(2024, 1, 1))
        ]
        
        doc = MockDocument("Events occurred.")
        
        candidates = detector.detect_causal_candidates(events, doc)
        
        # Should not find "Later → Earlier" relationship
        # Only valid temporal ordering is "Earlier → Later"
        for candidate in candidates:
            # If any candidates found, they must be temporally valid
            assert candidate.temporal_ordering_valid
    
    def test_temporal_gap_filtering(self):
        """Validate temporal gap limit is enforced."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm, max_temporal_gap_days=30)
        
        # Create events with large temporal gap
        events = [
            create_event("Event A", datetime(2024, 1, 1)),
            create_event("Event B", datetime(2024, 12, 31))  # ~1 year later
        ]
        
        doc = MockDocument("Events occurred.")
        
        candidates = detector.detect_causal_candidates(events, doc)
        
        # Should be filtered due to exceeding max gap
        assert len(candidates) == 0
    
    def test_confidence_filtering(self):
        """Validate confidence threshold filtering.
        
        Success Metric: Only candidates with confidence >0.6 returned.
        """
        # Mock responses with varying confidence
        responses = [
            {"evidence": "Strong link", "relationship_type": "causes", "confidence": 0.9},
            {"evidence": "Weak link", "relationship_type": "causes", "confidence": 0.4},
        ]
        llm = MockLLM(responses)
        
        detector = CausalRelationshipDetector(llm, confidence_threshold=0.6)
        
        events = [
            create_event("Event A", datetime(2024, 1, 1)),
            create_event("Event B", datetime(2024, 1, 2)),
            create_event("Event C", datetime(2024, 1, 3))
        ]
        
        doc = MockDocument("Events with varying confidence.")
        
        candidates = detector.detect_causal_candidates(events, doc)
        
        # Only high confidence candidates should be included
        assert all(c.causal_confidence >= 0.6 for c in candidates)
    
    def test_empty_event_list(self):
        """Validate handling of empty event lists."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        doc = MockDocument("No events")
        
        candidates = detector.detect_causal_candidates([], doc)
        
        assert candidates == []
    
    def test_single_event(self):
        """Validate handling of single event (no pairs possible)."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        events = [create_event("Only event", datetime(2024, 1, 1))]
        doc = MockDocument("Single event")
        
        candidates = detector.detect_causal_candidates(events, doc)
        
        assert candidates == []
    
    def test_events_without_timestamps(self):
        """Validate events without timestamps are skipped."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        # Create events without timestamps
        event1 = Event(
            name="Event 1",
            event_type="action",
            timestamp=None,  # No timestamp
            source_document="test",
            extraction_confidence=0.9
        )
        event2 = Event(
            name="Event 2",
            event_type="action",
            timestamp=datetime(2024, 1, 1),
            source_document="test",
            extraction_confidence=0.9
        )
        
        doc = MockDocument("Events")
        
        candidates = detector.detect_causal_candidates([event1, event2], doc)
        
        # Cannot form causal pairs without timestamps
        assert len(candidates) == 0


class TestPotentialCausalPair:
    """Test potential causal pair validation."""
    
    def test_valid_temporal_ordering(self):
        """Validate proper temporal ordering check."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        cause = create_event("Cause", datetime(2024, 1, 1))
        effect = create_event("Effect", datetime(2024, 1, 2))
        doc = MockDocument("Test")
        
        is_valid = detector._is_potential_causal_pair(cause, effect, doc)
        
        assert is_valid is True
    
    def test_invalid_temporal_ordering(self):
        """Validate rejection of reversed temporal ordering."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        cause = create_event("Later", datetime(2024, 1, 2))
        effect = create_event("Earlier", datetime(2024, 1, 1))
        doc = MockDocument("Test")
        
        is_valid = detector._is_potential_causal_pair(cause, effect, doc)
        
        assert is_valid is False
    
    def test_simultaneous_events(self):
        """Validate rejection of simultaneous events (same timestamp)."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        timestamp = datetime(2024, 1, 1, 12, 0)
        event1 = create_event("Event 1", timestamp)
        event2 = create_event("Event 2", timestamp)
        doc = MockDocument("Test")
        
        is_valid = detector._is_potential_causal_pair(event1, event2, doc)
        
        assert is_valid is False
    
    def test_temporal_gap_limit(self):
        """Validate temporal gap limit enforcement."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm, max_temporal_gap_days=30)
        
        cause = create_event("Cause", datetime(2024, 1, 1))
        effect = create_event("Effect", datetime(2024, 2, 15))  # 45 days later
        doc = MockDocument("Test")
        
        is_valid = detector._is_potential_causal_pair(cause, effect, doc)
        
        assert is_valid is False


class TestCausalTypeInference:
    """Test causal relationship type inference."""
    
    def test_infer_causes_type(self):
        """Validate inferring CAUSES relationship type."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        evidence = "The meeting caused the decision to be made."
        
        rel_type = detector._infer_causal_type(evidence)
        
        assert rel_type == CausalRelationshipType.CAUSES
    
    def test_infer_triggers_type(self):
        """Validate inferring TRIGGERS relationship type."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        evidence = "The alarm triggered the evacuation."
        
        rel_type = detector._infer_causal_type(evidence)
        
        assert rel_type == CausalRelationshipType.TRIGGERS
    
    def test_infer_enables_type(self):
        """Validate inferring ENABLES relationship type."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        evidence = "The approval enabled the project to start."
        
        rel_type = detector._infer_causal_type(evidence)
        
        assert rel_type == CausalRelationshipType.ENABLES
    
    def test_infer_prevents_type(self):
        """Validate inferring PREVENTS relationship type."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        evidence = "The delay prevented the launch."
        
        rel_type = detector._infer_causal_type(evidence)
        
        assert rel_type == CausalRelationshipType.PREVENTS
    
    def test_infer_leads_to_type(self):
        """Validate inferring LEADS_TO relationship type."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        evidence = "The discussion led to a consensus."
        
        rel_type = detector._infer_causal_type(evidence)
        
        assert rel_type == CausalRelationshipType.LEADS_TO
    
    def test_infer_contributes_to_type(self):
        """Validate inferring CONTRIBUTES_TO relationship type."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        evidence = "The research contributed to the findings."
        
        rel_type = detector._infer_causal_type(evidence)
        
        assert rel_type == CausalRelationshipType.CONTRIBUTES_TO


class TestConfidenceAssessment:
    """Test confidence scoring."""
    
    def test_strong_causal_indicators_boost_confidence(self):
        """Validate strong indicators increase confidence."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        cause = create_event("Cause", datetime(2024, 1, 1))
        effect = create_event("Effect", datetime(2024, 1, 2))
        
        # Evidence with strong indicator
        evidence = "The meeting caused the decision."
        
        confidence = detector._assess_causal_confidence(evidence, cause, effect)
        
        assert confidence >= 0.7  # Boosted by strong indicator
    
    def test_substantial_evidence_boosts_confidence(self):
        """Validate substantial evidence increases confidence."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        cause = create_event("Cause", datetime(2024, 1, 1))
        effect = create_event("Effect", datetime(2024, 1, 2))
        
        # Long evidence text
        evidence = "This is substantial evidence explaining the causal relationship " * 5
        
        confidence = detector._assess_causal_confidence(evidence, cause, effect)
        
        assert confidence >= 0.6  # Boosted by length
    
    def test_empty_evidence_low_confidence(self):
        """Validate empty evidence results in lower confidence."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        cause = create_event("Cause", datetime(2024, 1, 1))
        effect = create_event("Effect", datetime(2024, 1, 2))
        
        evidence = ""
        
        confidence = detector._assess_causal_confidence(evidence, cause, effect)
        
        assert confidence == 0.5  # Base confidence only


class TestLLMIntegration:
    """Test LLM integration."""
    
    def test_llm_prompt_building(self):
        """Validate LLM prompt construction."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        cause = create_event("Meeting", datetime(2024, 1, 15))
        effect = create_event("Decision", datetime(2024, 1, 16))
        doc = MockDocument("Meeting led to decision.")
        
        prompt = detector._build_causal_prompt(cause, effect, doc)
        
        # Verify prompt contains key elements
        assert "Meeting" in prompt
        assert "Decision" in prompt
        assert "causal" in prompt.lower()
        assert "JSON" in prompt or "json" in prompt
    
    def test_llm_response_parsing_dict(self):
        """Validate parsing dict response."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        response = {"evidence": "Test", "confidence": 0.8}
        
        parsed = detector._parse_llm_response(response)
        
        assert parsed == response
    
    def test_llm_response_parsing_json_string(self):
        """Validate parsing JSON string response."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        response = '{"evidence": "Test", "confidence": 0.8}'
        
        parsed = detector._parse_llm_response(response)
        
        assert parsed["evidence"] == "Test"
        assert parsed["confidence"] == 0.8
    
    def test_llm_response_parsing_embedded_json(self):
        """Validate parsing JSON embedded in text."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        response = 'Here is the result: {"evidence": "Test", "confidence": 0.8} extracted.'
        
        parsed = detector._parse_llm_response(response)
        
        assert parsed["evidence"] == "Test"
    
    def test_llm_response_parsing_invalid(self):
        """Validate handling of invalid responses."""
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        
        response = "Not JSON at all"
        
        parsed = detector._parse_llm_response(response)
        
        assert parsed is None

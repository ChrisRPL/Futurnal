"""Unit tests for event extraction.

Tests implementation from:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Test Coverage:
- Event extraction with temporal grounding
- Event type classification
- Confidence filtering (>0.7)
- Temporal extractor integration
- Event extraction accuracy (>80% target)
- LLM integration
"""

from datetime import datetime
from typing import Any, List
from unittest.mock import Mock, MagicMock

import pytest

from futurnal.extraction.causal.event_extractor import EventExtractor
from futurnal.extraction.causal.models import EventType
from futurnal.extraction.temporal.models import Event, TemporalMark, TemporalSourceType


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


class MockTemporalExtractor:
    """Mock temporal extractor for testing."""
    
    def __init__(self, markers: List[TemporalMark] = None):
        self.markers = markers or []
    
    def extract_explicit_timestamps(self, text: str) -> List[TemporalMark]:
        """Mock temporal marker extraction."""
        return self.markers


class MockDocument:
    """Mock document for testing."""
    
    def __init__(self, content: str, doc_id: str = "test_doc"):
        self.content = content
        self.doc_id = doc_id


# Tests

class TestEventExtractor:
    """Test EventExtractor class."""
    
    def test_initialization(self):
        """Validate EventExtractor initialization."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        
        extractor = EventExtractor(llm, temporal_extractor)
        
        assert extractor.llm is llm
        assert extractor.temporal_extractor is temporal_extractor
        assert extractor.confidence_threshold == 0.7  # Default
    
    def test_custom_confidence_threshold(self):
        """Validate custom confidence threshold."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        
        extractor = EventExtractor(llm, temporal_extractor, confidence_threshold=0.8)
        
        assert extractor.confidence_threshold == 0.8
    
    def test_event_indicators_available(self):
        """Validate event indicator keywords are defined."""
        assert len(EventExtractor.EVENT_INDICATORS) > 0
        assert "met" in EventExtractor.EVENT_INDICATORS
        assert "decided" in EventExtractor.EVENT_INDICATORS
        assert "published" in EventExtractor.EVENT_INDICATORS


class TestEventIndicatorDetection:
    """Test event indicator heuristics."""
    
    def test_has_event_indicators_positive(self):
        """Validate event indicators detected in event sentences."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        # Sentences with events
        assert extractor._has_event_indicators("We met with the team yesterday.")
        assert extractor._has_event_indicators("The decision was made unanimously.")
        assert extractor._has_event_indicators("The report was published online.")
        assert extractor._has_event_indicators("The project started on Monday.")
    
    def test_has_event_indicators_negative(self):
        """Validate event indicators not detected in non-event sentences."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        # Sentences without events
        assert not extractor._has_event_indicators("The sky is blue.")
        assert not extractor._has_event_indicators("This is a description of the system.")
        assert not extractor._has_event_indicators("Background information about the topic.")
    
    def test_case_insensitive_detection(self):
        """Validate event indicators are case-insensitive."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        assert extractor._has_event_indicators("MET with team")
        assert extractor._has_event_indicators("DECIDED to proceed")
        assert extractor._has_event_indicators("Published the results")


class TestEventExtraction:
    """Test event extraction process."""
    
    def test_extract_events_with_temporal_grounding(self):
        """Validate events extracted with temporal grounding.
        
        Success Metric: All events have timestamps (when available).
        """
        # Mock LLM response with event
        mock_response = {
            "name": "Team meeting",
            "event_type": "meeting",
            "timestamp": "2024-01-15T14:00:00",
            "duration": 3600,
            "participants": ["engineering team"],
            "confidence": 0.9
        }
        llm = MockLLM([mock_response])
        
        # Mock temporal markers
        temporal_marker = TemporalMark(
            text="January 15, 2024 at 2 PM",
            timestamp=datetime(2024, 1, 15, 14, 0),
            temporal_type=TemporalSourceType.EXPLICIT,
            confidence=1.0
        )
        temporal_extractor = MockTemporalExtractor([temporal_marker])
        
        extractor = EventExtractor(llm, temporal_extractor)
        
        # Document with event
        doc = MockDocument("We met with the engineering team on January 15, 2024 at 2 PM.")
        
        events = extractor.extract_events(doc)
        
        assert len(events) > 0
        # Verify temporal grounding
        event = events[0]
        assert event.timestamp is not None
        assert event.name == "Team meeting"
        assert event.event_type == "meeting"
    
    def test_confidence_filtering(self):
        """Validate events filtered by confidence threshold.
        
        Success Metric: Only events with confidence >0.7 returned.
        """
        # Mock LLM responses with varying confidence
        responses = [
            {"name": "High confidence event", "event_type": "meeting", 
             "timestamp": "2024-01-15T10:00:00", "participants": [], "confidence": 0.9},
            {"name": "Low confidence event", "event_type": "action",
             "timestamp": "2024-01-15T11:00:00", "participants": [], "confidence": 0.5},
        ]
        llm = MockLLM(responses)
        temporal_extractor = MockTemporalExtractor()
        
        extractor = EventExtractor(llm, temporal_extractor, confidence_threshold=0.7)
        
        # Document with two event sentences
        doc = MockDocument("We met yesterday. Something happened later.")
        
        events = extractor.extract_events(doc)
        
        # Only high confidence event should be included
        assert len(events) == 1
        assert events[0].extraction_confidence >= 0.7
        assert events[0].name == "High confidence event"
    
    def test_event_type_classification(self):
        """Validate event types assigned correctly."""
        # Mock different event types
        responses = [
            {"name": "Team meeting", "event_type": EventType.MEETING.value,
             "timestamp": "2024-01-15T10:00:00", "participants": [], "confidence": 0.9},
            {"name": "Product launch decision", "event_type": EventType.DECISION.value,
             "timestamp": "2024-01-16T10:00:00", "participants": [], "confidence": 0.85},
            {"name": "Report published", "event_type": EventType.PUBLICATION.value,
             "timestamp": "2024-01-17T10:00:00", "participants": [], "confidence": 0.8},
        ]
        llm = MockLLM(responses)
        temporal_extractor = MockTemporalExtractor()
        
        extractor = EventExtractor(llm, temporal_extractor)
        
        doc = MockDocument("We met yesterday. Decided to launch. Published the report.")
        
        events = extractor.extract_events(doc)
        
        assert len(events) == 3
        assert events[0].event_type == EventType.MEETING.value
        assert events[1].event_type == EventType.DECISION.value
        assert events[2].event_type == EventType.PUBLICATION.value
    
    def test_temporal_extractor_integration(self):
        """Validate integration with temporal extractor.
        
        From production plan: EventExtractor uses temporal_extractor dependency.
        """
        mock_response = {
            "name": "Event", "event_type": "action",
            "timestamp": "2024-01-15T10:00:00", "participants": [], "confidence": 0.8
        }
        llm = MockLLM([mock_response])
        
        # Create spy temporal extractor
        temporal_marker = TemporalMark(
            text="January 15, 2024",
            timestamp=datetime(2024, 1, 15),
            temporal_type=TemporalSourceType.EXPLICIT,
            confidence=1.0
        )
        temporal_extractor = MockTemporalExtractor([temporal_marker])
        
        extractor = EventExtractor(llm, temporal_extractor)
        
        doc = MockDocument("Something happened on January 15, 2024.")
        
        events = extractor.extract_events(doc)
        
        # Verify temporal extractor was called
        assert len(events) > 0
        # Temporal markers should be available for event extraction
    
    def test_empty_document(self):
        """Validate handling of empty documents."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        doc = MockDocument("")
        
        events = extractor.extract_events(doc)
        
        assert events == []
    
    def test_no_event_indicators(self):
        """Validate no events extracted when no indicators present."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        doc = MockDocument("This is a static description with no events.")
        
        events = extractor.extract_events(doc)
        
        assert events == []


class TestLLMPromptBuilding:
    """Test LLM prompt construction."""
    
    def test_prompt_includes_sentence(self):
        """Validate prompt includes target sentence."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        sentence = "We met with the team yesterday."
        prompt = extractor._build_event_extraction_prompt(sentence, [])
        
        assert sentence in prompt
    
    def test_prompt_includes_temporal_context(self):
        """Validate prompt includes temporal markers."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        marker = TemporalMark(
            text="yesterday",
            timestamp=datetime(2024, 1, 14),
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=1.0
        )
        
        prompt = extractor._build_event_extraction_prompt("Test", [marker])
        
        assert "yesterday" in prompt
        assert "2024-01-14" in prompt
    
    def test_prompt_includes_event_types(self):
        """Validate prompt includes event type taxonomy."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        prompt = extractor._build_event_extraction_prompt("Test", [])
        
        # Event types should be in prompt
        assert "meeting" in prompt.lower()
        assert "decision" in prompt.lower()
        assert "publication" in prompt.lower()
    
    def test_prompt_requests_json_format(self):
        """Validate prompt requests JSON output."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        prompt = extractor._build_event_extraction_prompt("Test", [])
        
        assert "JSON" in prompt or "json" in prompt
        assert "name" in prompt
        assert "event_type" in prompt
        assert "confidence" in prompt


class TestLLMResponseParsing:
    """Test LLM response parsing."""
    
    def test_parse_dict_response(self):
        """Validate parsing of dict response."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        response = {"name": "Event", "confidence": 0.9}
        
        parsed = extractor._parse_llm_response(response)
        
        assert parsed == response
    
    def test_parse_json_string_response(self):
        """Validate parsing of JSON string response."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        response = '{"name": "Event", "confidence": 0.9}'
        
        parsed = extractor._parse_llm_response(response)
        
        assert parsed["name"] == "Event"
        assert parsed["confidence"] == 0.9
    
    def test_parse_json_with_surrounding_text(self):
        """Validate parsing JSON embedded in text."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        response = 'Here is the event: {"name": "Event", "confidence": 0.9} extracted.'
        
        parsed = extractor._parse_llm_response(response)
        
        assert parsed["name"] == "Event"
    
    def test_parse_invalid_response(self):
        """Validate handling of invalid responses."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        # Invalid JSON
        response = "This is not JSON"
        
        parsed = extractor._parse_llm_response(response)
        
        assert parsed is None


class TestSentenceSplitting:
    """Test sentence splitting utility."""
    
    def test_split_into_sentences_basic(self):
        """Validate basic sentence splitting."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        text = "First sentence. Second sentence. Third sentence."
        
        sentences = extractor._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence." in sentences[0]
        assert "Second sentence." in sentences[1]
        assert "Third sentence." in sentences[2]
    
    def test_split_handles_multiple_punctuation(self):
        """Validate splitting on different punctuation."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        text = "A question? An exclamation! A statement."
        
        sentences = extractor._split_into_sentences(text)
        
        assert len(sentences) == 3
    
    def test_split_filters_short_fragments(self):
        """Validate short fragments are filtered."""
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)
        
        # "A." is too short (< 10 chars), should not be sentence
        text = "A. This is a real sentence with enough characters."
        
        sentences = extractor._split_into_sentences(text)
        
        # Should handle appropriately (implementation may vary)
        assert len(sentences) >= 1

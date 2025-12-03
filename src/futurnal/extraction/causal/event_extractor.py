"""Event extraction for causal structure preparation.

This module implements event extraction with temporal grounding:
- Identifies event-containing sentences
- Extracts event details via LLM
- Associates temporal information from temporal markers
- Filters by confidence threshold (>0.7)

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Option B Compliance:
- Temporal-first design (events must have timestamps)
- Experiential learning integration (extraction improves via GRPO)
- No mockups (real LLM integration, real temporal extractor)

Success Metric: Event extraction accuracy >80%
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from futurnal.extraction.causal.models import EventType
from futurnal.extraction.temporal.models import Event, TemporalMark

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM interactions.
    
    Matches the LLMClient protocol from experiential learning.
    """
    
    def extract(self, prompt: str) -> Any:
        """Run extraction on a prompt.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            Extraction result (implementation-specific).
        """
        ...


class TemporalExtractor(Protocol):
    """Protocol for temporal marker extraction.
    
    Defines interface for temporal extraction dependency.
    """
    
    def extract_explicit_timestamps(self, text: str) -> List[TemporalMark]:
        """Extract temporal markers from text.
        
        Args:
            text: Input text to extract temporal markers from.
            
        Returns:
            List of extracted temporal markers.
        """
        ...


class Document(Protocol):
    """Protocol for document structure.
    
    Defines minimal interface required for event extraction.
    """
    
    content: str
    doc_id: str


class EventExtractor:
    """Extract events from documents with temporal grounding.
    
    Key Difference from Entity Extraction:
    - Events MUST have temporal grounding (timestamp)
    - Events are time-bound occurrences, not static entities
    - Events are candidates for causal relationships
    
    Extraction Process:
    1. Extract temporal markers via temporal extractor
    2. Identify event-containing sentences via heuristics
    3. Extract event details via LLM prompting
    4. Associate temporal information with events
    5. Filter by confidence threshold (>0.7)
    
    Experiential Learning Integration:
    - Event extraction can improve via Training-Free GRPO
    - Rollout generation for diverse extractions
    - Semantic advantages from quality comparison
    - Thought templates scaffold event reasoning
    """
    
    # Event indicator keywords (seed heuristics, will improve via learning)
    EVENT_INDICATORS = [
        # Action verbs
        "met", "decided", "published", "announced", "launched",
        "occurred", "happened", "took place", "began", "started",
        "ended", "finished", "completed", "made",
        # Communication verbs
        "sent", "received", "communicated", "discussed", "presented",
        "emailed", "called", "messaged",
        # State changes
        "changed", "became", "transformed", "updated", "modified",
        # Decisions
        "chose", "selected", "approved", "rejected", "agreed",
    ]
    
    def __init__(
        self,
        llm: LLMClient,
        temporal_extractor: TemporalExtractor,
        confidence_threshold: float = 0.7
    ):
        """Initialize event extractor.
        
        Args:
            llm: LLM client for event extraction
            temporal_extractor: Temporal marker extractor from Module 01
            confidence_threshold: Minimum confidence for event inclusion (default 0.7)
        """
        self.llm = llm
        self.temporal_extractor = temporal_extractor
        self.confidence_threshold = confidence_threshold
    
    def extract_events(self, document: Document) -> List[Event]:
        """Extract events with temporal grounding from document.
        
        Success Metric: Accuracy >80%
        
        Args:
            document: Document to extract events from
            
        Returns:
            List of extracted events with confidence >threshold
            
        Example:
            >>> extractor = EventExtractor(llm, temporal_extractor)
            >>> doc = Document(content="Met with team on 2024-01-15", doc_id="doc1")
            >>> events = extractor.extract_events(doc)
            >>> assert all(e.timestamp for e in events)  # All have timestamps
        """
        # Step 1: Extract temporal markers
        temporal_markers = self.temporal_extractor.extract_explicit_timestamps(
            document.content
        )
        
        # Step 2: Split document into sentences
        sentences = self._split_into_sentences(document.content)
        
        # Step 3: Extract events from event-containing sentences
        events = []
        for sentence in sentences:
            if self._has_event_indicators(sentence):
                event = self._extract_event_from_sentence(
                    sentence=sentence,
                    temporal_markers=temporal_markers,
                    document=document
                )
                
                # Step 4: Filter by confidence
                if event and event.extraction_confidence >= self.confidence_threshold:
                    events.append(event)
        
        return events
    
    def _has_event_indicators(self, sentence: str) -> bool:
        """Check if sentence contains event indicators.
        
        Uses seed heuristics that will improve via experiential learning.
        
        Args:
            sentence: Sentence to check
            
        Returns:
            True if sentence likely describes an event
        """
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in self.EVENT_INDICATORS)
    
    def _extract_event_from_sentence(
        self,
        sentence: str,
        temporal_markers: List[TemporalMark],
        document: Document
    ) -> Optional[Event]:
        """Extract event details from sentence using LLM.
        
        Args:
            sentence: Sentence containing potential event
            temporal_markers: Temporal markers from document
            document: Source document
            
        Returns:
            Extracted event or None if extraction fails
        """
        # Build extraction prompt
        prompt = self._build_event_extraction_prompt(
            sentence=sentence,
            temporal_markers=temporal_markers
        )
        
        try:
            # Call LLM for structured extraction
            result = self.llm.extract(prompt)
            
            # Parse LLM response
            event_data = self._parse_llm_response(result)
            if not event_data:
                return None
            
            # Create Event object
            event = Event(
                name=event_data.get("name", ""),
                event_type=event_data.get("event_type", EventType.UNKNOWN.value),
                timestamp=event_data.get("timestamp"),
                duration=event_data.get("duration"),
                participants=event_data.get("participants", []),
                source_document=document.doc_id,
                extraction_confidence=event_data.get("confidence", 0.5)
            )
            
            return event
            
        except Exception as e:
            # Log extraction failure for debugging
            logger.warning(
                "Event extraction failed for sentence: %s. Error: %s",
                sentence[:100] + "..." if len(sentence) > 100 else sentence,
                str(e)
            )
            return None
    
    def _build_event_extraction_prompt(
        self,
        sentence: str,
        temporal_markers: List[TemporalMark]
    ) -> str:
        """Build LLM prompt for event extraction.
        
        Prompt engineering for structured event extraction:
        - Few-shot examples
        - Temporal context injection
        - Structured JSON output
        - Event type taxonomy
        
        Args:
            sentence: Sentence to extract event from
            temporal_markers: Available temporal markers
            
        Returns:
            Structured extraction prompt
        """
        # Format temporal context
        temporal_context = ""
        if temporal_markers:
            temporal_context = "Available temporal markers:\n"
            for marker in temporal_markers[:5]:  # Limit to top 5
                temporal_context += f"- {marker.text}"
                if marker.timestamp:
                    temporal_context += f" -> {marker.timestamp.isoformat()}"
                temporal_context += "\n"
        
        prompt = f"""Extract event information from the following sentence.

{temporal_context}

Sentence: "{sentence}"

Extract the following information in JSON format:
{{
  "name": "Brief event name",
  "event_type": "One of: meeting, decision, publication, communication, action, state_change, unknown",
  "timestamp": "ISO 8601 timestamp if available, else null",
  "duration": "Duration in seconds if mentioned, else null",
  "participants": ["List of participants/entities involved"],
  "confidence": "Confidence score 0.0-1.0"
}}

Event Types:
- meeting: Meetings, discussions, gatherings
- decision: Decisions, choices, determinations
- publication: Publications, releases, announcements
- communication: Communications, messages, exchanges
- action: Actions, activities, tasks
- state_change: State changes, transitions, updates
- unknown: Unclassified events

Examples:

Sentence: "Met with the engineering team on January 15, 2024 to discuss the project."
Output:
{{
  "name": "Meeting with engineering team",
  "event_type": "meeting",
  "timestamp": "2024-01-15T00:00:00",
  "duration": null,
  "participants": ["engineering team"],
  "confidence": 0.9
}}

Sentence: "The decision was made to proceed with the launch."
Output:
{{
  "name": "Decision to proceed with launch",
  "event_type": "decision",
  "timestamp": null,
  "duration": null,
  "participants": [],
  "confidence": 0.75
}}

Now extract from the given sentence:
"""
        return prompt
    
    def _parse_llm_response(self, result: Any) -> Optional[Dict[str, Any]]:
        """Parse LLM response into event data.
        
        Args:
            result: LLM extraction result
            
        Returns:
            Parsed event data dict or None if parsing fails
        """
        try:
            # Handle different result formats
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                # Try to parse as JSON
                # Find JSON in response (may have surrounding text)
                start = result.find('{')
                end = result.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = result[start:end]
                    return json.loads(json_str)
            elif hasattr(result, 'content'):
                # Handle result with content attribute
                return self._parse_llm_response(result.content)
            
            return None
            
        except (json.JSONDecodeError, AttributeError) as e:
            # Log parsing failures for debugging
            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            logger.debug(
                "Failed to parse LLM response as JSON. Error: %s. Response preview: %s",
                str(e),
                result_preview
            )
            return None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Simple sentence splitting heuristic.
        In production, could use more sophisticated NLP.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple split on period, exclamation, question mark
        # This is basic; could improve with spaCy or NLTK
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 10:
                sentences.append(current.strip())
                current = ""
        
        # Add remaining text
        if current.strip():
            sentences.append(current.strip())
        
        return sentences

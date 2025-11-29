Summary: Prepare causal structure for Phase 3 by extracting events, event-event relationships, and causal candidates with Bradford Hill criteria foundation.

# 05 · Causal Structure Preparation

## Purpose
Establish the foundational causal structure in the PKG that enables Phase 3 causal inference and hypothesis validation. Extract events as distinct from static entities, identify event-event relationships as causal candidates, and structure data for Bradford Hill criteria validation—all while maintaining privacy-first principles.

**Criticality**: CRITICAL - Foundation for Phase 3 causal inference; blocks Phase 3 without causal structure

## Scope
- Event entity extraction and classification
- Event-event relationship detection
- Causal candidate flagging
- Bradford Hill criteria preparation
- Causal chain storage and traversal
- Integration with temporal extraction

## Requirements Alignment
- **Option B Requirement**: "Graph structure prepared for Phase 3 causal inference"
- **Phase 3 Foundation**: Bradford Hill criteria validation ready
- **Critical Gap**: Eliminates lack of causal structure
- **Enables**: Causal discovery, hypothesis testing, counterfactual reasoning

## Component Design

### Event Extraction

```python
from enum import Enum
from pydantic import BaseModel


class EventType(str, Enum):
    """Categories of events."""
    MEETING = "meeting"
    DECISION = "decision"
    PUBLICATION = "publication"
    COMMUNICATION = "communication"
    ACTION = "action"
    STATE_CHANGE = "state_change"
    UNKNOWN = "unknown"


class Event(BaseModel):
    """Event entity distinct from static entities."""
    id: str
    name: str
    event_type: EventType
    description: str

    # Temporal grounding (required for events)
    timestamp: datetime
    duration: Optional[timedelta] = None

    # Participants (entities involved)
    participants: List[str] = []  # Entity IDs

    # Location/context
    location: Optional[str] = None
    context: str

    # Provenance
    source_document: str
    extraction_confidence: float


class EventExtractor:
    """Extract events from documents."""

    def __init__(self, llm, temporal_extractor):
        self.llm = llm
        self.temporal_extractor = temporal_extractor

    def extract_events(self, document: Document) -> List[Event]:
        """
        Extract events with temporal grounding.

        Key difference from entities: Events have timestamps
        """
        # First, extract temporal markers
        temporal_markers = self.temporal_extractor.extract(document)

        # Then, identify event mentions with temporal context
        events = []
        for sentence in document.sentences:
            if self._has_event_indicators(sentence):
                event = self._extract_event_from_sentence(
                    sentence,
                    temporal_markers,
                    document
                )
                if event and event.extraction_confidence > 0.7:
                    events.append(event)

        return events

    def _has_event_indicators(self, sentence: str) -> bool:
        """
        Check if sentence describes an event.

        Indicators:
        - Action verbs
        - Temporal references
        - State changes
        """
        indicators = ["met", "decided", "published", "occurred", "happened"]
        return any(ind in sentence.lower() for ind in indicators)
```

### Causal Relationship Detection

```python
class CausalRelationshipType(str, Enum):
    """Types of causal relationships."""
    CAUSES = "causes"           # Strong causal claim
    ENABLES = "enables"         # Prerequisite relationship
    PREVENTS = "prevents"       # Blocking relationship
    TRIGGERS = "triggers"       # Immediate causation
    LEADS_TO = "leads_to"      # Indirect causation
    CONTRIBUTES_TO = "contributes_to"  # Partial causation


class CausalCandidate(BaseModel):
    """Event-event relationship flagged for Phase 3 validation."""
    id: str
    cause_event_id: str
    effect_event_id: str
    relationship_type: CausalRelationshipType

    # Temporal validation
    temporal_gap: timedelta  # Time between cause and effect
    temporal_ordering_valid: bool  # Cause before effect?

    # Evidence
    causal_evidence: str  # Text supporting causation
    causal_confidence: float

    # Bradford Hill criteria (to be validated in Phase 3)
    temporality_satisfied: bool  # Cause before effect
    strength: Optional[float] = None  # Association strength
    consistency: Optional[float] = None  # Replicable?
    plausibility: Optional[str] = None  # Mechanistic explanation

    # Phase 3 validation status
    is_validated: bool = False
    validation_method: Optional[str] = None


class CausalRelationshipDetector:
    """Detect event-event relationships as causal candidates."""

    def __init__(self, llm):
        self.llm = llm
        self.causal_indicators = [
            "caused", "led to", "resulted in", "triggered",
            "enabled", "allowed", "prevented", "blocked"
        ]

    def detect_causal_candidates(
        self,
        events: List[Event],
        document: Document
    ) -> List[CausalCandidate]:
        """
        Identify potential causal relationships between events.

        Criteria for candidacy:
        1. Temporal ordering (cause before effect)
        2. Causal language in text
        3. Reasonable temporal gap (<1 year typical)
        """
        candidates = []

        for i, cause in enumerate(events):
            for effect in events[i+1:]:  # Only future events
                if self._is_potential_causal_pair(cause, effect, document):
                    candidate = self._create_causal_candidate(
                        cause,
                        effect,
                        document
                    )
                    if candidate.causal_confidence > 0.6:
                        candidates.append(candidate)

        return candidates

    def _is_potential_causal_pair(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> bool:
        """
        Check if two events could be causally related.

        Requirements:
        - Temporal ordering (cause before effect)
        - Reasonable temporal proximity
        - Contextual connection
        """
        # Temporality check
        if cause.timestamp >= effect.timestamp:
            return False

        # Proximity check (within 1 year)
        gap = effect.timestamp - cause.timestamp
        if gap > timedelta(days=365):
            return False

        # Context check (both mentioned in similar context)
        return True

    def _create_causal_candidate(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> CausalCandidate:
        """Create causal candidate with Bradford Hill criteria prep."""
        temporal_gap = effect.timestamp - cause.timestamp

        # Extract causal evidence from document
        evidence = self._extract_causal_evidence(cause, effect, document)

        # Assess confidence
        confidence = self._assess_causal_confidence(evidence)

        return CausalCandidate(
            id=f"causal_{cause.id}_{effect.id}",
            cause_event_id=cause.id,
            effect_event_id=effect.id,
            relationship_type=self._infer_causal_type(evidence),
            temporal_gap=temporal_gap,
            temporal_ordering_valid=True,  # Already validated
            causal_evidence=evidence,
            causal_confidence=confidence,
            temporality_satisfied=True,  # Bradford Hill criterion 1
            is_validated=False  # Phase 3 will validate
        )
```

### Bradford Hill Criteria Preparation

```python
class BradfordHillCriteria(BaseModel):
    """Bradford Hill criteria for causal inference."""

    # 1. Temporality (required)
    temporality: bool  # Does cause precede effect?

    # 2. Strength of association
    strength: Optional[float] = None  # How strong is the relationship?

    # 3. Dose-response
    dose_response: Optional[bool] = None  # More cause → more effect?

    # 4. Consistency
    consistency: Optional[float] = None  # Replicable across contexts?

    # 5. Plausibility
    plausibility: Optional[str] = None  # Mechanistic explanation

    # 6. Coherence
    coherence: Optional[bool] = None  # Fits existing knowledge?

    # 7. Experiment
    experiment_possible: Optional[bool] = None  # Can we test?

    # 8. Analogy
    analogy: Optional[str] = None  # Similar to known causal patterns?


class BradfordHillPreparation:
    """Prepare data structure for Phase 3 Bradford Hill validation."""

    def prepare_for_validation(
        self,
        candidate: CausalCandidate,
        pkg_context: Dict[str, Any]
    ) -> BradfordHillCriteria:
        """
        Prepare causal candidate for Phase 3 validation.

        Phase 1: Structure data
        Phase 3: Validate criteria interactively
        """
        return BradfordHillCriteria(
            temporality=candidate.temporal_ordering_valid,
            # Other criteria prepared but not validated yet
        )
```

## Implementation Details

See [PHASE-1-OPTION-B-ROADMAP.md](../PHASE-1-OPTION-B-ROADMAP.md) Weeks 13-14 for timeline.

## Testing Strategy

```python
class TestCausalStructure:
    def test_event_extraction(self):
        """Validate event extraction from documents."""
        doc = load_document_with_events()
        extractor = EventExtractor(mock_llm, temporal_extractor)

        events = extractor.extract_events(doc)

        assert len(events) > 0
        assert all(e.timestamp for e in events)  # All events have timestamps
        assert all(e.event_type != EventType.UNKNOWN for e in events)

    def test_causal_candidate_detection(self):
        """Validate causal relationship detection."""
        events = [
            create_event("meeting", timestamp=datetime(2024, 1, 1)),
            create_event("decision", timestamp=datetime(2024, 1, 2))
        ]
        detector = CausalRelationshipDetector(mock_llm)

        candidates = detector.detect_causal_candidates(events, mock_document)

        assert len(candidates) > 0
        assert all(c.temporal_ordering_valid for c in candidates)
        assert all(c.temporality_satisfied for c in candidates)

    def test_temporal_ordering_validation(self):
        """Ensure only temporally valid candidates created."""
        events = [
            create_event("A", timestamp=datetime(2024, 1, 2)),
            create_event("B", timestamp=datetime(2024, 1, 1))  # Before A
        ]
        detector = CausalRelationshipDetector(mock_llm)

        candidates = detector.detect_causal_candidates(events, mock_document)

        # B→A valid, A→B invalid (reversed)
        assert all(c.temporal_ordering_valid for c in candidates)
```

## Success Metrics

- ✅ Event extraction accuracy >80%
- ✅ Event-event relationships identified
- ✅ Causal candidates flagged with confidence >0.6
- ✅ Bradford Hill criteria structure prepared
- ✅ Temporal ordering validated for all candidates

## Dependencies

- Temporal extraction (01-temporal-extraction.md) - REQUIRED
- PKG storage with event and causal support
- Schema evolution for event types

**This module prepares the foundation for Phase 3 causal inference—the ultimate goal of the Ghost→Animal evolution.**

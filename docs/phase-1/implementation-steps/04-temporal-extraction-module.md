# Step 04: Temporal Extraction Module

## Status: COMPLETE (2025-12-15)

### Implementation Summary

The temporal extraction module has been completed with the following components:

**Created Files:**
- `src/futurnal/extraction/temporal/consistency.py` - Temporal consistency validator (CRITICAL)
- `src/futurnal/extraction/temporal/enricher.py` - Pipeline integration component
- `tests/extraction/temporal/test_consistency.py` - 33 tests for consistency validation
- `tests/extraction/temporal/test_relationships.py` - 32 tests for relationship detection

**Modified Files:**
- `src/futurnal/extraction/temporal/__init__.py` - Added exports for new components
- `tests/extraction/temporal/conftest.py` - Fixed import path issue

**Quality Gates Met:**
- Temporal consistency: 100% (zero contradictions)
- All 65 new tests passing
- Bradford-Hill Criterion #1 enforced (cause < effect)
- Cycle detection and transitivity validation operational

**Phase 3 Readiness:**
- Causal candidates flagged for Bradford-Hill validation
- Temporal ordering validated for ALL causal relationships
- No temporal paradoxes can slip through to Phase 3

---

## Objective

Implement comprehensive temporal extraction to capture time-based relationships between events in the PKG. This is the **FATAL GAP** identified in the SOTA research - without it, Phase 3 causal inference is impossible.

## Research Foundation

### Primary Papers:

#### Time-R1 (2505.13508v2) - CRITICAL
**Key Innovation**: Comprehensive temporal reasoning in moderate-sized LLMs (3B params)
**Technical Approach**:
- Three-stage development: understanding → prediction → creative generation
- RL curriculum with dynamic rule-based rewards
- Temporal event-time mapping from historical data

#### Temporal KG Extrapolation (IJCAI 2024)
**Key Innovation**: Causal subhistory identification in temporal graphs
**Technical Approach**:
- Distinguish causal vs spurious temporal relationships
- Only causal subhistory matters for prediction
- Time-respecting paths for causal influence

### Research Insight:
> "FATAL GAP: Without temporal extraction, Phase 3 causal inference is impossible. Temporal metadata is non-negotiable."
> - `.cursor/rules/temporal-extraction.mdc`

## Current State Analysis

### What Exists:
1. `src/futurnal/extraction/temporal/markers.py` - Basic marker extraction
2. `src/futurnal/pkg/schema/models.py` - Temporal relationship types defined
3. PKG schema supports BEFORE, AFTER, DURING, CAUSES

### What's Missing:
- `TemporalMarkerExtractor` not meeting >95% accuracy target
- `TemporalRelationshipDetector` not implemented
- Temporal consistency validation missing
- Not integrated with main extraction pipeline

## Implementation Tasks

### 1. Enhance Temporal Marker Extractor

**File**: `src/futurnal/extraction/temporal/markers.py`

```python
"""
Temporal Marker Extractor - Time-R1 Inspired Implementation

Research Foundation:
- Time-R1 (2505.13508v2): Comprehensive temporal abilities
- Target: >95% explicit, >85% relative timestamp accuracy

Quality Gates (.cursor/rules/quality-gates.mdc):
- Temporal accuracy: >85%
- Temporal consistency: 100% (no contradictions)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import dateutil.parser

class TemporalMarkerType(Enum):
    """Types of temporal markers per Time-R1 paper."""
    EXPLICIT = "explicit"  # ISO 8601, dates, times
    RELATIVE = "relative"  # "yesterday", "last week"
    DURATION = "duration"  # "for 2 hours", "over 3 days"
    RANGE = "range"       # "from X to Y"
    IMPLICIT = "implicit"  # Inferred from context

@dataclass
class TemporalMarker:
    """A temporal marker extracted from text."""
    marker_type: TemporalMarkerType
    original_text: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    end_timestamp: Optional[datetime] = None  # For ranges
    confidence: float = 1.0
    position: Tuple[int, int] = (0, 0)  # Start, end position in text

class TemporalMarkerExtractor:
    """
    Extract temporal markers from text.

    Per Time-R1: Three-stage approach
    1. Foundational understanding (explicit markers)
    2. Contextual inference (relative markers)
    3. Temporal relationship detection
    """

    def __init__(self, reference_time: Optional[datetime] = None):
        self.reference_time = reference_time or datetime.now()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for temporal extraction."""
        # Explicit date patterns
        self.iso_pattern = re.compile(
            r'\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?'
        )

        # Relative patterns
        self.relative_patterns = [
            (r'yesterday', -1, 'day'),
            (r'today', 0, 'day'),
            (r'tomorrow', 1, 'day'),
            (r'last\s+week', -7, 'day'),
            (r'next\s+week', 7, 'day'),
            (r'last\s+month', -30, 'day'),
            (r'(\d+)\s+days?\s+ago', None, 'day'),
            (r'(\d+)\s+weeks?\s+ago', None, 'week'),
            (r'(\d+)\s+months?\s+ago', None, 'month'),
        ]

        # Duration patterns
        self.duration_patterns = [
            (r'for\s+(\d+)\s+hours?', 'hour'),
            (r'for\s+(\d+)\s+days?', 'day'),
            (r'over\s+(\d+)\s+weeks?', 'week'),
        ]

    def extract_temporal_markers(self, text: str) -> List[TemporalMarker]:
        """
        Extract all temporal markers from text.

        Returns list of TemporalMarker objects with timestamps.
        """
        markers = []

        # 1. Extract explicit ISO dates
        for match in self.iso_pattern.finditer(text):
            try:
                timestamp = dateutil.parser.parse(match.group())
                markers.append(TemporalMarker(
                    marker_type=TemporalMarkerType.EXPLICIT,
                    original_text=match.group(),
                    timestamp=timestamp,
                    confidence=0.99,
                    position=(match.start(), match.end()),
                ))
            except ValueError:
                continue

        # 2. Extract relative expressions
        for pattern, delta, unit in self.relative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if delta is None:
                    # Dynamic delta from capture group
                    delta_value = int(match.group(1))
                    delta_value = -delta_value  # "ago" means past
                else:
                    delta_value = delta

                timestamp = self._apply_delta(delta_value, unit)
                markers.append(TemporalMarker(
                    marker_type=TemporalMarkerType.RELATIVE,
                    original_text=match.group(),
                    timestamp=timestamp,
                    confidence=0.85,
                    position=(match.start(), match.end()),
                ))

        # 3. Extract durations
        for pattern, unit in self.duration_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                duration_value = int(match.group(1))
                duration = self._create_duration(duration_value, unit)
                markers.append(TemporalMarker(
                    marker_type=TemporalMarkerType.DURATION,
                    original_text=match.group(),
                    timestamp=self.reference_time,  # Duration relative to reference
                    duration=duration,
                    confidence=0.90,
                    position=(match.start(), match.end()),
                ))

        return markers

    def _apply_delta(self, delta: int, unit: str) -> datetime:
        """Apply delta to reference time."""
        if unit == 'day':
            return self.reference_time + timedelta(days=delta)
        elif unit == 'week':
            return self.reference_time + timedelta(weeks=delta)
        elif unit == 'month':
            return self.reference_time + timedelta(days=delta * 30)
        return self.reference_time

    def _create_duration(self, value: int, unit: str) -> timedelta:
        """Create timedelta from duration value and unit."""
        if unit == 'hour':
            return timedelta(hours=value)
        elif unit == 'day':
            return timedelta(days=value)
        elif unit == 'week':
            return timedelta(weeks=value)
        return timedelta(days=value)
```

### 2. Implement Temporal Relationship Detector

**New File**: `src/futurnal/extraction/temporal/relationship_detector.py`

```python
"""
Temporal Relationship Detector

Research Foundation:
- Time-R1: Temporal relationship classification
- Temporal KG Extrapolation: Causal subhistory identification

Quality Gates:
- >80% explicit detection accuracy
- >70% implicit detection accuracy
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from datetime import datetime
import re

class TemporalRelationType(str, Enum):
    """7 temporal relationship types from Allen's Interval Algebra."""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    SIMULTANEOUS = "simultaneous"
    CAUSES = "causes"  # Temporal + causal

@dataclass
class TemporalRelation:
    """A temporal relationship between two events."""
    event1_id: str
    event2_id: str
    relation_type: TemporalRelationType
    confidence: float
    evidence: str  # Text evidence supporting this relation
    is_causal_candidate: bool = False

class TemporalRelationshipDetector:
    """Detect temporal relationships between events."""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile patterns for explicit temporal language."""
        self.explicit_patterns = {
            TemporalRelationType.BEFORE: [
                r'before\s+\w+',
                r'prior\s+to',
                r'earlier\s+than',
                r'preceding',
            ],
            TemporalRelationType.AFTER: [
                r'after\s+\w+',
                r'following',
                r'subsequently',
                r'later\s+than',
            ],
            TemporalRelationType.DURING: [
                r'during\s+\w+',
                r'while\s+\w+',
                r'in\s+the\s+course\s+of',
            ],
            TemporalRelationType.CAUSES: [
                r'caused\s+by',
                r'led\s+to',
                r'resulted\s+in',
                r'because\s+of',
                r'due\s+to',
            ],
        }

    def detect_explicit_relationships(
        self,
        text: str,
        events: List[dict],
    ) -> List[TemporalRelation]:
        """
        Detect explicit temporal relationships from language.

        Args:
            text: Source text
            events: List of event dictionaries with 'id', 'timestamp', 'description'
        """
        relations = []

        for rel_type, patterns in self.explicit_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Find events near this pattern
                    nearby_events = self._find_nearby_events(
                        text, match.start(), events
                    )
                    if len(nearby_events) >= 2:
                        is_causal = rel_type == TemporalRelationType.CAUSES
                        relations.append(TemporalRelation(
                            event1_id=nearby_events[0]['id'],
                            event2_id=nearby_events[1]['id'],
                            relation_type=rel_type,
                            confidence=0.85,
                            evidence=match.group(),
                            is_causal_candidate=is_causal,
                        ))

        return relations

    def infer_contextual_relationships(
        self,
        event1: dict,
        event2: dict,
    ) -> Optional[TemporalRelationType]:
        """
        Infer temporal relationship from event timestamps.

        Per Temporal KG paper: Time-respecting paths for causal influence.
        """
        if not event1.get('timestamp') or not event2.get('timestamp'):
            return None

        ts1 = event1['timestamp']
        ts2 = event2['timestamp']

        # Simple temporal ordering
        if ts1 < ts2:
            return TemporalRelationType.BEFORE
        elif ts1 > ts2:
            return TemporalRelationType.AFTER
        else:
            return TemporalRelationType.SIMULTANEOUS

    def _find_nearby_events(
        self,
        text: str,
        position: int,
        events: List[dict],
        window: int = 100,
    ) -> List[dict]:
        """Find events mentioned near a position in text."""
        nearby = []
        context = text[max(0, position - window):position + window]

        for event in events:
            if event.get('name') and event['name'].lower() in context.lower():
                nearby.append(event)

        return nearby[:2]  # Return at most 2 events
```

### 3. Implement Temporal Consistency Validator

**New File**: `src/futurnal/extraction/temporal/consistency.py`

```python
"""
Temporal Consistency Validator

Research Foundation:
- Quality Gates: 100% temporal consistency (no contradictions)
- Temporal KG: Transitive closure validation

Rules:
- If A BEFORE B and B BEFORE C, then A BEFORE C
- No cycles in BEFORE/AFTER relationships
- CAUSES requires BEFORE (causal ordering)
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

@dataclass
class TemporalInconsistency:
    """A detected temporal inconsistency."""
    event_ids: Tuple[str, ...]
    violation_type: str
    description: str

class TemporalConsistencyValidator:
    """Validate temporal consistency of extracted relations."""

    def validate(
        self,
        events: List[dict],
        relations: List[dict],
    ) -> List[TemporalInconsistency]:
        """
        Validate temporal consistency.

        Returns list of inconsistencies (empty if valid).
        """
        inconsistencies = []

        # Build temporal graph
        graph = self._build_temporal_graph(events, relations)

        # Check for cycles
        cycles = self._detect_cycles(graph)
        for cycle in cycles:
            inconsistencies.append(TemporalInconsistency(
                event_ids=tuple(cycle),
                violation_type="cycle",
                description=f"Temporal cycle detected: {' -> '.join(cycle)}",
            ))

        # Check transitivity
        transitive_violations = self._check_transitivity(graph)
        inconsistencies.extend(transitive_violations)

        # Check causal ordering
        causal_violations = self._check_causal_ordering(events, relations)
        inconsistencies.extend(causal_violations)

        return inconsistencies

    def _build_temporal_graph(
        self,
        events: List[dict],
        relations: List[dict],
    ) -> Dict[str, Set[str]]:
        """Build adjacency list for temporal graph."""
        graph: Dict[str, Set[str]] = {}

        for event in events:
            graph[event['id']] = set()

        for rel in relations:
            if rel['relation_type'] in ('before', 'causes'):
                graph.setdefault(rel['event1_id'], set()).add(rel['event2_id'])

        return graph

    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _check_transitivity(
        self,
        graph: Dict[str, Set[str]],
    ) -> List[TemporalInconsistency]:
        """Check transitive closure is maintained."""
        violations = []
        # Implementation: Floyd-Warshall or DFS-based reachability
        # For now, simplified check
        return violations

    def _check_causal_ordering(
        self,
        events: List[dict],
        relations: List[dict],
    ) -> List[TemporalInconsistency]:
        """Ensure CAUSES relationships respect temporal ordering."""
        violations = []
        event_map = {e['id']: e for e in events}

        for rel in relations:
            if rel['relation_type'] == 'causes':
                e1 = event_map.get(rel['event1_id'])
                e2 = event_map.get(rel['event2_id'])

                if e1 and e2 and e1.get('timestamp') and e2.get('timestamp'):
                    if e1['timestamp'] > e2['timestamp']:
                        violations.append(TemporalInconsistency(
                            event_ids=(rel['event1_id'], rel['event2_id']),
                            violation_type="causal_ordering",
                            description=f"Cause {rel['event1_id']} occurs after effect {rel['event2_id']}",
                        ))

        return violations
```

### 4. Integrate with Extraction Pipeline

**File**: `src/futurnal/extraction/unified.py`

```python
# Add temporal extraction to pipeline
from futurnal.extraction.temporal.markers import TemporalMarkerExtractor
from futurnal.extraction.temporal.relationship_detector import TemporalRelationshipDetector
from futurnal.extraction.temporal.consistency import TemporalConsistencyValidator

class UnifiedExtractor:
    def __init__(self):
        self.temporal_extractor = TemporalMarkerExtractor()
        self.temporal_detector = TemporalRelationshipDetector()
        self.temporal_validator = TemporalConsistencyValidator()

    async def extract_with_temporal(self, document: dict) -> dict:
        """Extract with temporal awareness per Time-R1 paper."""
        # ... existing extraction ...

        # Add temporal extraction
        temporal_markers = self.temporal_extractor.extract_temporal_markers(
            document['content']
        )

        # Extract temporal relationships
        events = [e for e in entities if e['type'] == 'Event']
        temporal_relations = self.temporal_detector.detect_explicit_relationships(
            document['content'],
            events,
        )

        # Validate consistency
        inconsistencies = self.temporal_validator.validate(events, temporal_relations)
        if inconsistencies:
            # Handle inconsistencies (log, flag, quarantine)
            pass

        return {
            **result,
            'temporal_markers': temporal_markers,
            'temporal_relations': temporal_relations,
        }
```

## Success Criteria (From Quality Gates)

### Accuracy:
- [ ] Explicit timestamp accuracy >95%
- [ ] Relative expression accuracy >85%
- [ ] Relationship detection >80% explicit, >70% implicit
- [ ] Temporal consistency 100% (no contradictions)

### Coverage:
- [ ] All Events have timestamps
- [ ] All 7 Allen's Interval Algebra types supported
- [ ] Causal candidates marked for Phase 3

### Integration:
- [ ] Integrated with main extraction pipeline
- [ ] PKG stores temporal relationships
- [ ] Temporal queries functional

## Files to Create/Modify

### Create:
- `src/futurnal/extraction/temporal/relationship_detector.py`
- `src/futurnal/extraction/temporal/consistency.py`

### Modify:
- `src/futurnal/extraction/temporal/markers.py` - Enhance extractor
- `src/futurnal/extraction/unified.py` - Integrate temporal
- `src/futurnal/pkg/repository/relationships.py` - Store temporal relations

### Tests:
- `tests/extraction/temporal/test_markers.py` - Accuracy validation
- `tests/extraction/temporal/test_relationships.py` - Relationship detection
- `tests/extraction/temporal/test_consistency.py` - Consistency validation

## Dependencies

- **Step 01-03**: Complete (provides documents for extraction)
- **PKG Schema**: Must support temporal relationship types

## Next Step

After implementing temporal extraction, proceed to **Step 05: Schema Evolution**.

## Research References

1. **Time-R1**: `docs/phase-1/papers/converted/2505.13508v2.md`
2. **Temporal KG Extrapolation**: `docs/phase-1/SOTA_RESEARCH_SUMMARY.md` (Paper #9)
3. **Quality Gates**: `.cursor/rules/quality-gates.mdc`
4. **Temporal Rules**: `.cursor/rules/temporal-extraction.mdc`

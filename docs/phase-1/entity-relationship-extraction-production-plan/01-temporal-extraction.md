# 01 · Temporal Extraction Module

## Purpose
Implement comprehensive temporal reasoning capabilities to extract timestamps, durations, temporal relationships, and causal temporal patterns from documents. This is the **FATAL GAP** that blocks Phase 3 causal inference—without temporal extraction, the system cannot detect temporal correlations or validate causal hypotheses.

**Criticality**: FATAL - Phase 3 impossible without this foundation

## SOTA Foundation

**Primary Reference**: Time-R1 (ArXiv 2505.13508v2, 2025)
- Comprehensive temporal reasoning framework with RL curriculum
- Dynamic rule-based reward system
- Progressive complexity: markers → ordering → causality

**Supporting**: Temporal KG Extrapolation (IJCAI 2024)
- Causal subhistory identification
- Temporal relationship inference

## Scope

### Temporal Marker Extraction

#### 1. Explicit Timestamp Detection
```python
class TemporalMarkerExtractor:
    """Extract explicit temporal markers from text."""

    def extract_explicit_timestamps(self, text: str) -> List[TemporalMark]:
        """
        Extract dates, times, and date-time combinations.

        Examples:
        - "January 15, 2024"
        - "2024-01-15T14:30:00"
        - "at 2:30 PM"
        - "on Monday"
        """
        markers = []

        # ISO 8601 formats
        markers.extend(self._extract_iso8601(text))

        # Natural language dates
        markers.extend(self._extract_natural_dates(text))

        # Time expressions
        markers.extend(self._extract_time_expressions(text))

        return markers
```

**Patterns to Support**:
- ISO 8601: `2024-01-15`, `2024-01-15T14:30:00Z`
- Natural language: "January 15, 2024", "15th of January"
- Relative: "Monday", "next Tuesday", "last Friday"
- Time: "2:30 PM", "14:30", "half past two"
- Partial: "January 2024", "2024", "Q1 2024"

#### 2. Relative Time Expression Parsing
```python
class RelativeTimeParser:
    """Parse relative temporal expressions into absolute or relative timestamps."""

    def parse_relative_expression(self, expr: str, reference_time: datetime) -> TemporalMark:
        """
        Parse expressions like "yesterday", "last month", "2 weeks ago".

        Categories:
        - Relative days: yesterday, today, tomorrow
        - Relative weeks: last week, next week
        - Relative months: last month, this month
        - Duration-based: "2 weeks ago", "in 3 days"
        """
        if expr in RELATIVE_DAY_MAPPING:
            return self._parse_relative_day(expr, reference_time)
        elif expr in RELATIVE_WEEK_MAPPING:
            return self._parse_relative_week(expr, reference_time)
        elif "ago" in expr:
            return self._parse_duration_ago(expr, reference_time)
        elif "in" in expr and self._is_future_duration(expr):
            return self._parse_future_duration(expr, reference_time)
```

**Expressions to Support**:
- Relative days: yesterday, today, tomorrow, the day before yesterday
- Relative periods: last week, this month, next year
- Duration offsets: "2 weeks ago", "in 3 months", "5 days from now"
- Contextual: "earlier that day", "later that week"

#### 3. Context-Based Temporal Inference
```python
class ContextualTemporalInferencer:
    """Infer temporal information from document context."""

    def infer_from_document_metadata(self, doc: Document) -> Optional[datetime]:
        """
        Extract temporal context from:
        - Document creation/modification timestamps
        - Frontmatter date fields (Obsidian)
        - Email headers (Date:, Received:)
        - Git commit timestamps
        """

    def infer_from_sentence_structure(self, sentence: str) -> List[TemporalInference]:
        """
        Infer temporal relationships from sentence structure.

        Examples:
        - "After finishing the report, I sent the email"
          → finish(report) BEFORE send(email)
        - "While working on X, Y happened"
          → work(X) DURING happen(Y)
        """
```

### Temporal Relationship Types

```python
class TemporalRelationshipType(Enum):
    """Allen's Interval Algebra + Causal Extensions"""

    # Core temporal relationships (Allen's Interval Algebra)
    BEFORE = "before"           # A finishes before B starts
    AFTER = "after"             # A starts after B finishes
    DURING = "during"           # A occurs within B's timespan
    CONTAINS = "contains"       # B occurs within A's timespan
    OVERLAPS = "overlaps"       # A and B overlap partially
    MEETS = "meets"             # A finishes exactly when B starts
    STARTS = "starts"           # A and B start together
    FINISHES = "finishes"       # A and B finish together
    EQUALS = "equals"           # A and B have identical timespan

    # Causal temporal relationships
    CAUSES = "causes"           # A temporally precedes and causes B
    ENABLES = "enables"         # A creates conditions for B
    PREVENTS = "prevents"       # A blocks B from occurring
    TRIGGERS = "triggers"       # A directly initiates B

    # Concurrent relationships
    SIMULTANEOUS = "simultaneous"  # A and B occur at same time
    PARALLEL = "parallel"          # A and B occur in parallel
```

### Temporal Triple Structure

```python
@dataclass
class TemporalTriple:
    """Enhanced triple with comprehensive temporal metadata."""

    # Core triple
    subject: Entity
    predicate: Relationship
    object: Entity

    # Temporal metadata
    timestamp: Optional[datetime]          # When did this occur?
    duration: Optional[timedelta]          # How long did it last?
    temporal_type: TemporalRelationshipType  # Relationship to other events

    # Temporal bounds
    valid_from: Optional[datetime]         # When did this become true?
    valid_to: Optional[datetime]           # When did this stop being true?

    # Provenance
    provenance: ChunkReference
    temporal_source: TemporalSource        # Where did temporal info come from?

    # Confidence
    confidence: float                      # Overall confidence
    temporal_confidence: float             # Confidence in temporal information

@dataclass
class TemporalSource:
    """Track origin of temporal information."""
    source_type: str  # "explicit", "inferred", "document_metadata", "relative"
    evidence: str     # Original text that led to temporal inference
    inference_method: Optional[str]  # Method used for inference
```

## Implementation Phases

### Week 1: Core Temporal Marker Extraction

**Deliverables**:
1. Explicit timestamp detector
   - ISO 8601 parser
   - Natural language date parser
   - Time expression parser

2. Relative time parser
   - Relative day/week/month mapping
   - Duration offset calculation
   - Reference time resolution

3. Unit tests
   - 100+ test cases for various temporal formats
   - Edge cases: ambiguous dates, multiple formats
   - Validation against golden dataset

**Success Criteria**:
- ✅ Parse >95% of explicit timestamps correctly
- ✅ Resolve >85% of relative expressions
- ✅ Handle ambiguous cases gracefully (e.g., "10/11/2024")

### Week 2: Temporal Relationship Detection

**Deliverables**:
1. Temporal relationship classifier
   - Rule-based detector for explicit markers (before/after/during)
   - LLM-based inference for implicit relationships
   - Allen's Interval Algebra implementation

2. Contextual temporal inferencer
   - Sentence structure analysis
   - Document metadata extraction
   - Temporal consistency checking

3. Integration tests
   - Full document temporal extraction
   - Multi-event temporal ordering
   - Causal relationship detection

**Success Criteria**:
- ✅ Identify >80% of explicit temporal relationships
- ✅ Infer >70% of implicit temporal relationships
- ✅ Maintain temporal consistency (no contradictions)

### Week 3: Integration & Validation

**Deliverables**:
1. Pipeline integration
   - Extend existing extraction pipeline with temporal module
   - Update triple structure with temporal fields
   - Integrate with provenance tracking

2. Temporal ordering validation
   - Consistency checker (detect contradictions)
   - Temporal reasoning engine (infer transitive relationships)
   - Confidence calibration

3. Comprehensive testing
   - End-to-end temporal extraction on real documents
   - Validation against labeled dataset
   - Performance benchmarking

**Success Criteria**:
- ✅ Overall temporal accuracy >85%
- ✅ No blocking temporal inconsistencies
- ✅ Ready for PKG storage integration

## Requirements Alignment

**From SOTA Research**:
- ✅ Time-R1 RL curriculum approach (progressive complexity)
- ✅ Temporal relationship types aligned with causal reasoning
- ✅ Foundation for Phase 3 causal inference

**From Option B Requirements**:
- ✅ Temporal extraction (FATAL gap) - COMPLETE
- ✅ Timestamps, durations, temporal relationships
- ✅ >85% temporal ordering accuracy

## Technical Implementation Details

### Temporal Marker Detection

```python
class TemporalDetectionPipeline:
    """Complete temporal detection pipeline."""

    def __init__(self):
        self.explicit_detector = ExplicitTimestampDetector()
        self.relative_parser = RelativeTimeParser()
        self.contextual_inferencer = ContextualTemporalInferencer()
        self.relationship_detector = TemporalRelationshipDetector()

    def extract_temporal_information(
        self,
        document: NormalizedDocument
    ) -> List[TemporalTriple]:
        """
        Full temporal extraction pipeline.

        Steps:
        1. Extract explicit timestamps from content
        2. Parse relative time expressions
        3. Infer temporal context from document metadata
        4. Detect temporal relationships between entities/events
        5. Build temporal triples with confidence scores
        """
        # Step 1: Extract all temporal markers
        explicit_markers = self.explicit_detector.extract(document.content)
        relative_markers = self.relative_parser.parse(
            document.content,
            reference_time=document.metadata.get('created_at')
        )
        contextual_time = self.contextual_inferencer.infer(document)

        # Step 2: Extract entities and events
        entities = self.extract_entities(document)
        events = self.extract_events(document)

        # Step 3: Associate temporal markers with entities/events
        temporal_entities = self.associate_temporal_markers(
            entities, events, explicit_markers + relative_markers
        )

        # Step 4: Detect temporal relationships
        relationships = self.relationship_detector.detect(temporal_entities)

        # Step 5: Build temporal triples
        triples = self.build_temporal_triples(
            temporal_entities,
            relationships,
            contextual_time
        )

        return triples
```

### Temporal Relationship Detection

```python
class TemporalRelationshipDetector:
    """Detect temporal relationships between events."""

    def __init__(self, llm):
        self.llm = llm
        self.rule_based_detector = RuleBasedTemporalDetector()

    def detect(self, temporal_entities: List[TemporalEntity]) -> List[TemporalRelationship]:
        """
        Detect relationships via hybrid approach.

        1. Rule-based: Explicit markers ("before", "after", "during")
        2. LLM-based: Implicit relationships from sentence structure
        3. Transitive inference: A before B, B before C → A before C
        """
        relationships = []

        # Rule-based detection
        relationships.extend(self.rule_based_detector.detect(temporal_entities))

        # LLM-based inference for complex cases
        for entity1, entity2 in self._get_entity_pairs(temporal_entities):
            if not self._has_explicit_relationship(entity1, entity2, relationships):
                inferred_rel = self._infer_temporal_relationship_llm(entity1, entity2)
                if inferred_rel and inferred_rel.confidence > 0.7:
                    relationships.append(inferred_rel)

        # Transitive closure
        relationships = self._apply_transitive_inference(relationships)

        return relationships

    def _infer_temporal_relationship_llm(
        self,
        entity1: TemporalEntity,
        entity2: TemporalEntity
    ) -> Optional[TemporalRelationship]:
        """Use LLM to infer temporal relationship from context."""

        prompt = f"""# Temporal Relationship Inference

Entity 1: {entity1.name}
Context 1: {entity1.context}
Timestamp 1: {entity1.timestamp or 'unknown'}

Entity 2: {entity2.name}
Context 2: {entity2.context}
Timestamp 2: {entity2.timestamp or 'unknown'}

Task: Determine the temporal relationship between these entities.

Relationships:
- BEFORE: Entity 1 finishes before Entity 2 starts
- AFTER: Entity 1 starts after Entity 2 finishes
- DURING: Entity 1 occurs within Entity 2's timespan
- SIMULTANEOUS: Entities occur at the same time
- CAUSES: Entity 1 causes Entity 2 (temporal + causal)
- ENABLES: Entity 1 enables Entity 2
- UNKNOWN: Cannot determine relationship

Output format:
{{
  "relationship": "BEFORE|AFTER|DURING|SIMULTANEOUS|CAUSES|ENABLES|UNKNOWN",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""

        result = self.llm.generate(prompt)
        return self._parse_temporal_relationship(result)
```

## Testing Strategy

### Unit Tests

```python
class TestTemporalExtraction:
    """Comprehensive temporal extraction tests."""

    def test_explicit_timestamp_detection(self):
        """Test explicit timestamp formats."""
        test_cases = [
            ("On January 15, 2024, I finished the report", "2024-01-15"),
            ("Meeting at 2:30 PM", "14:30:00"),
            ("2024-01-15T14:30:00Z", "2024-01-15T14:30:00Z"),
            ("Q1 2024", "2024-Q1"),
        ]
        for text, expected in test_cases:
            result = extractor.extract_explicit_timestamps(text)
            assert result[0].timestamp == parse_time(expected)

    def test_relative_time_parsing(self):
        """Test relative time expression parsing."""
        reference = datetime(2024, 1, 15, 14, 30)
        test_cases = [
            ("yesterday", datetime(2024, 1, 14)),
            ("last week", datetime(2024, 1, 8)),
            ("2 weeks ago", datetime(2024, 1, 1)),
            ("in 3 days", datetime(2024, 1, 18)),
        ]
        for expr, expected in test_cases:
            result = parser.parse_relative_expression(expr, reference)
            assert result.timestamp == expected

    def test_temporal_relationship_detection(self):
        """Test temporal relationship classification."""
        test_cases = [
            (
                "After finishing the report, I sent the email",
                [("finish", "report", "BEFORE", "send", "email")]
            ),
            (
                "While working on X, Y happened",
                [("work", "X", "DURING", "happen", "Y")]
            ),
            (
                "The meeting caused the decision to be made",
                [("meeting", "CAUSES", "decision")]
            ),
        ]
        for text, expected_relationships in test_cases:
            result = detector.detect_relationships(text)
            assert self._matches_expected(result, expected_relationships)
```

### Integration Tests

```python
class TestTemporalIntegration:
    """End-to-end temporal extraction tests."""

    def test_full_document_temporal_extraction(self):
        """Test complete temporal extraction pipeline on real document."""
        doc = load_fixture("docs/sample_journal_entry.md")

        triples = temporal_pipeline.extract_temporal_information(doc)

        # Validate temporal metadata present
        assert all(t.timestamp or t.temporal_type for t in triples)

        # Validate temporal ordering consistency
        assert self._is_temporally_consistent(triples)

        # Validate temporal accuracy against golden labels
        golden = load_golden_labels("docs/sample_journal_entry_temporal.json")
        accuracy = compute_temporal_accuracy(triples, golden)
        assert accuracy > 0.85
```

### Performance Benchmarks

```python
class TestTemporalPerformance:
    """Performance benchmarks for temporal extraction."""

    def test_extraction_throughput(self):
        """Benchmark temporal extraction throughput."""
        docs = load_test_corpus(num_docs=100)

        start_time = time.time()
        for doc in docs:
            temporal_pipeline.extract_temporal_information(doc)
        elapsed = time.time() - start_time

        throughput = len(docs) / elapsed
        assert throughput > 5  # >5 docs/second

    def test_temporal_accuracy_vs_speed_tradeoff(self):
        """Validate accuracy maintained at target throughput."""
        docs = load_labeled_corpus(num_docs=50)

        # Fast mode (rule-based only)
        fast_accuracy = benchmark_accuracy(docs, mode="fast")

        # Full mode (rule-based + LLM inference)
        full_accuracy = benchmark_accuracy(docs, mode="full")

        assert fast_accuracy > 0.75
        assert full_accuracy > 0.85
```

## Success Metrics

### Temporal Extraction Quality
- ✅ Explicit timestamp detection: >95% accuracy
- ✅ Relative time parsing: >85% accuracy
- ✅ Temporal relationship detection: >80% accuracy
- ✅ Overall temporal ordering: >85% accuracy

### Integration Success
- ✅ All triples include temporal metadata where applicable
- ✅ Temporal consistency maintained (no contradictions)
- ✅ Provenance tracking includes temporal sources
- ✅ PKG storage ready for temporal queries

### Performance
- ✅ Throughput: >5 documents/second
- ✅ Memory: <500MB for temporal processing
- ✅ Latency: <200ms per document (average)

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Ambiguous temporal expressions** | Medium | Confidence scoring; flag ambiguous cases |
| **Cross-document temporal ordering** | High | Document creation timestamps as reference |
| **Temporal consistency violations** | High | Validation layer; transitive inference |
| **Performance with large documents** | Medium | Streaming processing; temporal marker caching |

## Dependencies

- Existing extraction pipeline (entity/relationship detection)
- NormalizedDocument schema
- PKG storage with temporal field support
- LLM for implicit relationship inference

## Next Steps

After temporal extraction is complete:
1. Integrate with schema evolution (02-schema-evolution.md)
2. Enable temporal queries in PKG storage
3. Prepare for Phase 2 correlation detection
4. Foundation ready for Phase 3 causal inference

**This module eliminates the FATAL GAP and enables the entire Ghost→Animal evolution path.**

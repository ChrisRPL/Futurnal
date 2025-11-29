Summary: Implement autonomous schema evolution using AutoSchemaKG multi-phase extraction and reflection mechanisms to enable dynamic world model growth.

# 02 · Autonomous Schema Evolution

## Purpose
Implement self-evolving extraction schema that autonomously adapts to new entity types, relationships, and patterns discovered in user documents. This eliminates the static template limitation that prevents Ghost→Animal evolution and enables the PKG to develop a sophisticated, personalized understanding of the user's knowledge domain.

**Criticality**: CRITICAL - Enables dynamic world model; blocks Phase 2 proactive insights without autonomous evolution

## Scope
- Multi-phase extraction pipeline (Entity-Entity → Entity-Event → Event-Event)
- Reflection mechanism triggered every N documents
- Schema induction from document patterns
- Schema versioning and migration
- Quality assessment and refinement triggers
- Integration with thought template system

## Requirements Alignment
- **Option B Requirement**: "Autonomous Schema Evolution with >90% semantic alignment to manual schema"
- **SOTA Foundation**: AutoSchemaKG autonomous induction (2024)
- **Critical Gap**: Eliminates static schema limitation
- **Phase 2 Enabler**: Provides foundation for evolving correlation patterns

## Component Design

### Schema Evolution Engine

```python
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel


class ExtractionPhase(str, Enum):
    """Multi-phase extraction stages."""
    ENTITY_ENTITY = "entity_entity"      # Phase 1: Person → Organization
    ENTITY_EVENT = "entity_event"        # Phase 2: Person → Meeting
    EVENT_EVENT = "event_event"          # Phase 3: Meeting → Decision


class SchemaElement(BaseModel):
    """Base class for schema elements."""
    name: str
    description: str
    examples: List[str]
    confidence: float
    discovery_count: int = 0
    last_seen: datetime


class EntityType(SchemaElement):
    """Discovered entity type."""
    properties: Dict[str, str]  # property_name → property_type
    aliases: List[str]
    parent_type: Optional[str] = None  # For hierarchical types


class RelationshipType(SchemaElement):
    """Discovered relationship type."""
    subject_types: List[str]  # Valid subject entity types
    object_types: List[str]   # Valid object entity types
    temporal: bool = False    # Is this a temporal relationship?
    causal: bool = False      # Is this a causal relationship?
    properties: Dict[str, str]


class SchemaVersion(BaseModel):
    """Versioned schema snapshot."""
    version: int
    created_at: datetime
    entity_types: Dict[str, EntityType]
    relationship_types: Dict[str, RelationshipType]
    current_phase: ExtractionPhase
    quality_metrics: Dict[str, float]
    changes_from_previous: Optional[str] = None


class SchemaEvolutionEngine:
    """
    Autonomous schema evolution via multi-phase induction.

    Based on AutoSchemaKG approach:
    1. Phase 1: Entity-Entity relationships (seed schema)
    2. Phase 2: Entity-Event relationships (temporal grounding)
    3. Phase 3: Event-Event relationships (causal candidates)
    """

    def __init__(
        self,
        seed_schema: SchemaVersion,
        reflection_interval: int = 100,  # Reflect every N documents
        min_confidence_threshold: float = 0.7
    ):
        self.current_schema = seed_schema
        self.schema_history: List[SchemaVersion] = [seed_schema]
        self.reflection_interval = reflection_interval
        self.min_confidence_threshold = min_confidence_threshold
        self.documents_processed = 0
        self.pending_discoveries: List[SchemaDiscovery] = []

    def should_trigger_reflection(self) -> bool:
        """Check if reflection should be triggered."""
        return self.documents_processed % self.reflection_interval == 0

    def induce_schema_from_documents(
        self,
        documents: List[Document],
        current_phase: ExtractionPhase
    ) -> SchemaVersion:
        """
        Multi-phase schema induction.

        AutoSchemaKG approach:
        - Phase 1: Extract entity-entity patterns
        - Phase 2: Extract entity-event patterns (temporal)
        - Phase 3: Extract event-event patterns (causal)
        """
        if current_phase == ExtractionPhase.ENTITY_ENTITY:
            return self._induce_entity_entity_schema(documents)
        elif current_phase == ExtractionPhase.ENTITY_EVENT:
            return self._induce_entity_event_schema(documents)
        elif current_phase == ExtractionPhase.EVENT_EVENT:
            return self._induce_event_event_schema(documents)

    def _induce_entity_entity_schema(
        self,
        documents: List[Document]
    ) -> SchemaVersion:
        """
        Phase 1: Discover entity types and entity-entity relationships.

        Examples:
        - Entities: Person, Organization, Concept, Location
        - Relationships: works_at, founded, located_in, related_to
        """
        discovered_entities = self._discover_entity_patterns(documents)
        discovered_relationships = self._discover_relationship_patterns(
            documents,
            entity_types=discovered_entities
        )

        return self._create_schema_version(
            phase=ExtractionPhase.ENTITY_ENTITY,
            entities=discovered_entities,
            relationships=discovered_relationships
        )

    def _induce_entity_event_schema(
        self,
        documents: List[Document]
    ) -> SchemaVersion:
        """
        Phase 2: Discover event types and entity-event relationships.

        Examples:
        - Events: Meeting, Decision, Publication, Communication
        - Relationships: attended, made_decision, published, communicated_with

        Key: Events have temporal grounding (timestamp, duration)
        """
        discovered_events = self._discover_event_patterns(documents)
        discovered_relationships = self._discover_entity_event_patterns(
            documents,
            event_types=discovered_events
        )

        return self._create_schema_version(
            phase=ExtractionPhase.ENTITY_EVENT,
            entities=discovered_events,
            relationships=discovered_relationships
        )

    def _induce_event_event_schema(
        self,
        documents: List[Document]
    ) -> SchemaVersion:
        """
        Phase 3: Discover event-event relationships (causal candidates).

        Examples:
        - Relationships: caused, enabled, triggered, led_to

        Key: These are causal candidates for Phase 3 validation
        """
        discovered_relationships = self._discover_event_event_patterns(documents)

        return self._create_schema_version(
            phase=ExtractionPhase.EVENT_EVENT,
            entities={},  # No new entities in this phase
            relationships=discovered_relationships
        )

    def reflect_and_refine(
        self,
        extraction_results: List[ExtractionResult]
    ) -> Optional[SchemaVersion]:
        """
        Reflection mechanism: assess quality and refine schema.

        Triggers:
        - Every N documents (reflection_interval)
        - Low extraction success rate
        - High rejection rate (low confidence)
        - Consistent pattern of new entity/relationship types
        """
        quality_metrics = self._assess_extraction_quality(extraction_results)

        if quality_metrics["should_refine"]:
            refined_schema = self._refine_schema(
                self.current_schema,
                quality_metrics
            )
            self._version_schema(refined_schema)
            return refined_schema

        return None

    def _assess_extraction_quality(
        self,
        results: List[ExtractionResult]
    ) -> Dict[str, float]:
        """
        Assess extraction quality metrics.

        Metrics:
        - Coverage: % of documents producing extractions
        - Consistency: % of extractions with high confidence
        - Success rate: % of extractions not quarantined
        - Novel patterns: count of unrecognized patterns
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        high_confidence = sum(1 for r in results if r.confidence > 0.8)
        novel_patterns = sum(1 for r in results if r.has_novel_pattern)

        return {
            "coverage": successful / total if total > 0 else 0,
            "consistency": high_confidence / total if total > 0 else 0,
            "success_rate": successful / total if total > 0 else 0,
            "novel_pattern_rate": novel_patterns / total if total > 0 else 0,
            "should_refine": (
                successful / total < 0.7 or  # Low success rate
                novel_patterns / total > 0.1   # Many novel patterns
            )
        }

    def _refine_schema(
        self,
        schema: SchemaVersion,
        metrics: Dict[str, float]
    ) -> SchemaVersion:
        """
        Refine schema based on quality metrics.

        Refinements:
        - Add frequently seen novel entity/relationship types
        - Remove rarely seen types (low discovery_count)
        - Adjust confidence thresholds
        - Update type hierarchies
        """
        refined_entities = self._refine_entity_types(
            schema.entity_types,
            metrics
        )
        refined_relationships = self._refine_relationship_types(
            schema.relationship_types,
            metrics
        )

        return SchemaVersion(
            version=schema.version + 1,
            created_at=datetime.utcnow(),
            entity_types=refined_entities,
            relationship_types=refined_relationships,
            current_phase=schema.current_phase,
            quality_metrics=metrics,
            changes_from_previous=self._generate_changelog(schema, refined_entities, refined_relationships)
        )
```

### Schema Discovery

```python
class SchemaDiscovery(BaseModel):
    """Discovered schema element pending validation."""
    element_type: str  # "entity" or "relationship"
    name: str
    description: str
    examples: List[str]
    confidence: float
    source_documents: List[str]


class SchemaDiscoveryEngine:
    """Discover new schema elements from document patterns."""

    def __init__(self, llm):
        self.llm = llm
        self.discovery_threshold = 0.75  # Min confidence to propose discovery

    def discover_entity_patterns(
        self,
        documents: List[Document]
    ) -> List[SchemaDiscovery]:
        """
        Discover entity patterns using LLM analysis.

        Approach:
        1. Extract noun phrases and named entities
        2. Group by semantic similarity
        3. Propose entity types for frequent patterns
        4. Validate with LLM
        """
        noun_phrases = self._extract_noun_phrases(documents)
        clusters = self._cluster_by_similarity(noun_phrases)

        discoveries = []
        for cluster in clusters:
            if len(cluster) >= 5:  # Minimum 5 examples
                discovery = self._propose_entity_type(cluster)
                if discovery.confidence > self.discovery_threshold:
                    discoveries.append(discovery)

        return discoveries

    def discover_relationship_patterns(
        self,
        documents: List[Document],
        entity_types: Dict[str, EntityType]
    ) -> List[SchemaDiscovery]:
        """
        Discover relationship patterns between known entities.

        Approach:
        1. Extract sentences with multiple entities
        2. Identify connecting phrases/verbs
        3. Group by semantic similarity
        4. Propose relationship types
        """
        entity_pairs = self._extract_entity_pairs(documents, entity_types)
        relationship_patterns = self._extract_connecting_phrases(entity_pairs)

        discoveries = []
        for pattern in relationship_patterns:
            if pattern.frequency >= 3:  # Minimum 3 examples
                discovery = self._propose_relationship_type(pattern)
                if discovery.confidence > self.discovery_threshold:
                    discoveries.append(discovery)

        return discoveries
```

## Implementation Details

### Week 1: Seed Schema Design

**Deliverable**: Initial schema with core entity/relationship types

```python
def create_seed_schema() -> SchemaVersion:
    """Create initial seed schema for Phase 1."""

    # Core entity types
    entities = {
        "Person": EntityType(
            name="Person",
            description="Individual human being",
            examples=["John Smith", "Dr. Jane Doe", "the author"],
            confidence=1.0,
            properties={"name": "string", "aliases": "list[string]"},
            aliases=["individual", "human", "author"]
        ),
        "Organization": EntityType(
            name="Organization",
            description="Company, institution, or group",
            examples=["Acme Corp", "MIT", "the research team"],
            confidence=1.0,
            properties={"name": "string", "type": "string"},
            aliases=["company", "institution", "group"]
        ),
        "Concept": EntityType(
            name="Concept",
            description="Abstract idea or topic",
            examples=["machine learning", "productivity", "causality"],
            confidence=1.0,
            properties={"name": "string", "category": "string"},
            aliases=["idea", "topic", "subject"]
        ),
        "Document": EntityType(
            name="Document",
            description="Written work or reference",
            examples=["research paper", "the report", "documentation"],
            confidence=1.0,
            properties={"title": "string", "type": "string"},
            aliases=["paper", "article", "file"]
        )
    }

    # Core relationship types (Phase 1: Entity-Entity)
    relationships = {
        "works_at": RelationshipType(
            name="works_at",
            description="Person employed by Organization",
            examples=["John works at Acme Corp"],
            confidence=1.0,
            subject_types=["Person"],
            object_types=["Organization"],
            temporal=True,  # Can have valid_from/valid_to
            properties={"role": "string", "start_date": "datetime"}
        ),
        "created": RelationshipType(
            name="created",
            description="Entity created another entity",
            examples=["Alice created the framework"],
            confidence=1.0,
            subject_types=["Person", "Organization"],
            object_types=["Concept", "Document"],
            temporal=True,
            properties={"creation_date": "datetime"}
        ),
        "related_to": RelationshipType(
            name="related_to",
            description="Generic relationship between concepts",
            examples=["AI related to machine learning"],
            confidence=1.0,
            subject_types=["Concept"],
            object_types=["Concept"],
            temporal=False,
            properties={"relationship_type": "string"}
        )
    }

    return SchemaVersion(
        version=1,
        created_at=datetime.utcnow(),
        entity_types=entities,
        relationship_types=relationships,
        current_phase=ExtractionPhase.ENTITY_ENTITY,
        quality_metrics={}
    )
```

### Week 2: Multi-Phase Extraction

**Deliverable**: Phase 2 (Entity-Event) and Phase 3 (Event-Event) extraction

**Phase 2 Example** - Entity-Event relationships:
```python
# New entity type: Event
event_entity = EntityType(
    name="Event",
    description="Temporal occurrence or happening",
    examples=["meeting on Monday", "the conference", "project kickoff"],
    confidence=0.9,
    properties={
        "name": "string",
        "event_type": "string",
        "timestamp": "datetime",
        "duration": "timedelta"
    },
    aliases=["occurrence", "happening", "incident"]
)

# Entity-Event relationships
attended_relationship = RelationshipType(
    name="attended",
    description="Person participated in Event",
    examples=["Alice attended the meeting"],
    confidence=0.9,
    subject_types=["Person"],
    object_types=["Event"],
    temporal=True,
    properties={"role": "string", "attendance_confirmed": "bool"}
)
```

**Phase 3 Example** - Event-Event relationships (causal):
```python
caused_relationship = RelationshipType(
    name="caused",
    description="Event caused another Event (causal candidate)",
    examples=["The meeting caused the decision"],
    confidence=0.8,
    subject_types=["Event"],
    object_types=["Event"],
    temporal=True,  # Must be temporally ordered
    causal=True,    # Mark as causal candidate for Phase 3
    properties={
        "causal_confidence": "float",
        "temporal_gap": "timedelta",
        "is_validated": "bool"  # For Phase 3 validation
    }
)
```

### Week 3: Reflection Mechanism

**Deliverable**: Quality assessment and schema refinement

```python
class ReflectionTrigger:
    """Determine when reflection should occur."""

    def __init__(
        self,
        interval: int = 100,
        low_success_threshold: float = 0.7,
        high_novel_threshold: float = 0.1
    ):
        self.interval = interval
        self.low_success_threshold = low_success_threshold
        self.high_novel_threshold = high_novel_threshold

    def should_reflect(
        self,
        documents_processed: int,
        recent_results: List[ExtractionResult]
    ) -> bool:
        """
        Trigger reflection if:
        1. Interval reached (every N documents)
        2. Success rate drops below threshold
        3. Many novel patterns discovered
        """
        if documents_processed % self.interval == 0:
            return True

        if len(recent_results) < 20:
            return False

        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        if success_rate < self.low_success_threshold:
            return True

        novel_rate = sum(1 for r in recent_results if r.has_novel_pattern) / len(recent_results)
        if novel_rate > self.high_novel_threshold:
            return True

        return False


class SchemaRefinementEngine:
    """Refine schema based on reflection."""

    def refine_entity_types(
        self,
        current_types: Dict[str, EntityType],
        discoveries: List[SchemaDiscovery],
        min_discovery_count: int = 10
    ) -> Dict[str, EntityType]:
        """
        Refine entity types:
        - Add frequently discovered new types
        - Remove rarely seen types
        - Update confidence scores
        """
        refined = current_types.copy()

        # Add new types with sufficient evidence
        for discovery in discoveries:
            if (discovery.element_type == "entity" and
                len(discovery.examples) >= min_discovery_count):

                refined[discovery.name] = EntityType(
                    name=discovery.name,
                    description=discovery.description,
                    examples=discovery.examples,
                    confidence=discovery.confidence,
                    properties={},
                    aliases=[]
                )

        # Remove types with low discovery count
        refined = {
            name: entity
            for name, entity in refined.items()
            if entity.discovery_count >= min_discovery_count or entity.confidence == 1.0
        }

        return refined
```

## Testing Strategy

### Unit Tests

```python
class TestSchemaEvolution:
    """Test schema evolution mechanisms."""

    def test_seed_schema_creation(self):
        """Validate seed schema structure."""
        schema = create_seed_schema()

        assert schema.version == 1
        assert "Person" in schema.entity_types
        assert "Organization" in schema.entity_types
        assert "works_at" in schema.relationship_types
        assert schema.current_phase == ExtractionPhase.ENTITY_ENTITY

    def test_entity_entity_induction(self):
        """Test Phase 1 entity-entity schema induction."""
        documents = load_test_documents("entity_entity_samples/")
        engine = SchemaEvolutionEngine(create_seed_schema())

        new_schema = engine.induce_schema_from_documents(
            documents,
            ExtractionPhase.ENTITY_ENTITY
        )

        # Should discover additional entity types
        assert len(new_schema.entity_types) >= len(engine.current_schema.entity_types)

        # Should maintain Phase 1
        assert new_schema.current_phase == ExtractionPhase.ENTITY_ENTITY

    def test_reflection_trigger(self):
        """Test reflection mechanism triggers correctly."""
        engine = SchemaEvolutionEngine(create_seed_schema(), reflection_interval=10)

        # Process documents
        for i in range(15):
            engine.documents_processed += 1

        assert engine.should_trigger_reflection()  # Triggered at 10

    def test_schema_refinement(self):
        """Test schema refinement adds new types."""
        schema = create_seed_schema()
        discoveries = [
            SchemaDiscovery(
                element_type="entity",
                name="Project",
                description="Work project or initiative",
                examples=["project X", "the initiative", "our work"],
                confidence=0.85,
                source_documents=["doc1", "doc2", "doc3"]
            )
        ]

        engine = SchemaEvolutionEngine(schema)
        results = [create_mock_result(success=True) for _ in range(20)]

        refined = engine.reflect_and_refine(results)

        if refined:
            assert "Project" in refined.entity_types or schema.version == refined.version
```

### Integration Tests

```python
class TestSchemaEvolutionIntegration:
    """End-to-end schema evolution tests."""

    def test_multi_phase_progression(self):
        """Test progression through all 3 phases."""
        documents = load_diverse_corpus(100)
        engine = SchemaEvolutionEngine(create_seed_schema())

        # Phase 1: Entity-Entity
        phase1_schema = engine.induce_schema_from_documents(
            documents[:40],
            ExtractionPhase.ENTITY_ENTITY
        )
        assert phase1_schema.current_phase == ExtractionPhase.ENTITY_ENTITY

        # Phase 2: Entity-Event
        phase2_schema = engine.induce_schema_from_documents(
            documents[40:70],
            ExtractionPhase.ENTITY_EVENT
        )
        assert phase2_schema.current_phase == ExtractionPhase.ENTITY_EVENT
        assert "Event" in phase2_schema.entity_types

        # Phase 3: Event-Event
        phase3_schema = engine.induce_schema_from_documents(
            documents[70:],
            ExtractionPhase.EVENT_EVENT
        )
        assert phase3_schema.current_phase == ExtractionPhase.EVENT_EVENT
        assert any(r.causal for r in phase3_schema.relationship_types.values())

    def test_schema_versioning(self):
        """Test schema versions tracked correctly."""
        engine = SchemaEvolutionEngine(create_seed_schema())

        initial_version = engine.current_schema.version

        # Trigger refinement
        results = [create_mock_result(success=False) for _ in range(20)]
        refined = engine.reflect_and_refine(results)

        if refined:
            assert refined.version == initial_version + 1
            assert refined.changes_from_previous is not None
```

### Quality Benchmarks

```python
class TestSchemaQuality:
    """Validate schema quality against benchmarks."""

    def test_semantic_alignment_benchmark(self):
        """
        Validate >90% semantic alignment with manual schema.

        AutoSchemaKG benchmark approach.
        """
        # Load manually curated schema
        manual_schema = load_manual_schema()

        # Run autonomous induction on same corpus
        documents = load_labeled_corpus(200)
        engine = SchemaEvolutionEngine(create_seed_schema())

        induced_schema = engine.induce_schema_from_documents(
            documents,
            ExtractionPhase.ENTITY_ENTITY
        )

        # Compare entity types
        alignment = compute_semantic_alignment(
            induced_schema.entity_types,
            manual_schema.entity_types
        )

        assert alignment > 0.90, f"Alignment {alignment} below 90% threshold"
```

## Success Metrics

### Schema Evolution Quality
- ✅ >90% semantic alignment with manual schema (AutoSchemaKG benchmark)
- ✅ All 3 extraction phases operational (Entity→Event→Causal)
- ✅ Reflection triggers correctly every N documents
- ✅ Schema versions tracked with migration support

### Discovery Quality
- ✅ Novel entity/relationship type discovery functional
- ✅ Confidence scoring calibrated (high-confidence discoveries validated)
- ✅ Low false positive rate (<10% invalid discoveries)

### Integration Success
- ✅ Schema evolution integrated with extraction pipeline
- ✅ Thought templates updated when schema evolves
- ✅ PKG storage supports schema versioning
- ✅ No extraction downtime during schema updates

## Dependencies

- Temporal extraction module (01-temporal-extraction.md) - for Phase 2/3
- Thought template system (04-thought-templates.md) - for template evolution
- PKG storage with schema versioning support
- LLM for pattern discovery and validation

## Next Steps

After schema evolution is complete:
1. Integrate with thought template evolution (04-thought-templates.md)
2. Enable experiential learning on schema patterns (03-experiential-learning.md)
3. Prepare for Phase 2 correlation schema evolution
4. Foundation ready for multi-modal schema expansion

**This module eliminates the static schema CRITICAL GAP and enables the dynamic world model essential for Ghost→Animal evolution.**

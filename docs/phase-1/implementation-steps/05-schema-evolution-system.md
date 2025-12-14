# Step 05: Schema Evolution System

## Status: TODO

## Objective

Implement autonomous schema evolution that allows the PKG to discover and adapt its entity/relationship types as it processes more documents. This is CRITICAL for the Ghost→Animal evolution.

## Research Foundation

### Primary Papers:

#### AutoSchemaKG (2505.23628v1) - CRITICAL
**Key Innovation**: Autonomous schema induction with zero manual intervention
**Technical Approach**:
- Multi-phase extraction: Entity-Entity → Entity-Event → Event-Event
- Reflection mechanism refines schema every N documents
- 95% semantic alignment with human-crafted schemas

#### EDC Framework (2404.03868)
**Key Innovation**: Extract → Define → Canonicalize pipeline
**Application**: Phase 1 focuses on Extract, prepares Define for Phase 2

### Research Insight:
> "Static prompt templates are outdated. Autonomous evolution is proven and achievable."
> - SOTA Research Summary

## Current State

### What Exists (Stubs Only):
- `src/futurnal/extraction/schema/discovery.py` - Stub
- `src/futurnal/extraction/schema/evolution.py` - Stub
- `src/futurnal/extraction/schema/models.py` - Basic models

### What's Missing:
- Autonomous schema induction
- Multi-phase extraction
- Reflection mechanism
- Schema versioning

## Implementation Tasks

### 1. Schema Discovery Engine

**File**: `src/futurnal/extraction/schema/discovery.py`

```python
"""
Schema Discovery Engine - AutoSchemaKG Inspired

Research Foundation:
- AutoSchemaKG (2505.23628v1): Autonomous schema induction
- Target: >90% semantic alignment with manual schemas
"""

from typing import Dict, List, Set
from dataclasses import dataclass
from collections import Counter

@dataclass
class DiscoveredEntityType:
    """An entity type discovered from documents."""
    name: str
    count: int
    examples: List[str]
    confidence: float
    parent_type: str | None = None

@dataclass
class DiscoveredRelationType:
    """A relationship type discovered from documents."""
    name: str
    source_types: Set[str]
    target_types: Set[str]
    count: int
    examples: List[tuple]

class SchemaDiscoveryEngine:
    """
    Discover schema patterns from extracted triples.

    Per AutoSchemaKG: Multi-phase discovery
    Phase 1: Entity-Entity relationships
    Phase 2: Entity-Event relationships
    Phase 3: Event-Event relationships (causal candidates)
    """

    def __init__(self, reflection_interval: int = 50):
        self.reflection_interval = reflection_interval
        self.document_count = 0
        self.entity_patterns: Counter = Counter()
        self.relation_patterns: Counter = Counter()
        self.discovered_entities: Dict[str, DiscoveredEntityType] = {}
        self.discovered_relations: Dict[str, DiscoveredRelationType] = {}

    def process_extraction(self, entities: List[dict], relations: List[dict]):
        """Process extraction results to discover patterns."""
        self.document_count += 1

        # Track entity patterns
        for entity in entities:
            entity_type = entity.get('type', 'Unknown')
            self.entity_patterns[entity_type] += 1

        # Track relation patterns
        for rel in relations:
            pattern = (rel.get('subject_type'), rel.get('predicate'), rel.get('object_type'))
            self.relation_patterns[pattern] += 1

        # Trigger reflection if interval reached
        if self.document_count % self.reflection_interval == 0:
            self._reflect_and_refine()

    def _reflect_and_refine(self):
        """
        Reflection mechanism per AutoSchemaKG.

        Assess schema quality and refine based on patterns.
        """
        # Update discovered entity types
        for entity_type, count in self.entity_patterns.most_common():
            if count >= 5:  # Minimum frequency threshold
                self.discovered_entities[entity_type] = DiscoveredEntityType(
                    name=entity_type,
                    count=count,
                    examples=[],
                    confidence=min(count / 100, 1.0),
                )

        # Update discovered relation types
        for pattern, count in self.relation_patterns.most_common():
            if count >= 3 and all(pattern):
                rel_name = pattern[1]
                if rel_name not in self.discovered_relations:
                    self.discovered_relations[rel_name] = DiscoveredRelationType(
                        name=rel_name,
                        source_types=set(),
                        target_types=set(),
                        count=0,
                        examples=[],
                    )
                self.discovered_relations[rel_name].count += count
                self.discovered_relations[rel_name].source_types.add(pattern[0])
                self.discovered_relations[rel_name].target_types.add(pattern[2])

    def get_current_schema(self) -> dict:
        """Get current discovered schema."""
        return {
            'entity_types': list(self.discovered_entities.values()),
            'relation_types': list(self.discovered_relations.values()),
            'document_count': self.document_count,
            'last_reflection': self.document_count,
        }
```

### 2. Schema Evolution Manager

**File**: `src/futurnal/extraction/schema/evolution.py`

```python
"""
Schema Evolution Manager

Tracks schema versions and manages evolution over time.
"""

from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class SchemaVersion:
    """A version of the discovered schema."""
    version: int
    timestamp: datetime
    entity_types: Dict[str, dict]
    relation_types: Dict[str, dict]
    document_count: int
    quality_metrics: dict = field(default_factory=dict)

class SchemaEvolutionManager:
    """Manage schema evolution with versioning."""

    def __init__(self):
        self.versions: List[SchemaVersion] = []
        self.current_version = 0

    def create_version(self, schema: dict) -> SchemaVersion:
        """Create new schema version."""
        self.current_version += 1
        version = SchemaVersion(
            version=self.current_version,
            timestamp=datetime.now(),
            entity_types={e.name: e.__dict__ for e in schema.get('entity_types', [])},
            relation_types={r.name: r.__dict__ for r in schema.get('relation_types', [])},
            document_count=schema.get('document_count', 0),
        )
        self.versions.append(version)
        return version

    def compute_quality_metrics(self, version: SchemaVersion) -> dict:
        """Compute quality metrics for schema version."""
        return {
            'entity_type_count': len(version.entity_types),
            'relation_type_count': len(version.relation_types),
            'avg_entity_confidence': sum(
                e.get('confidence', 0) for e in version.entity_types.values()
            ) / max(len(version.entity_types), 1),
        }
```

### 3. Integrate with Extraction Pipeline

Update extraction to use discovered schema instead of hardcoded types.

### 4. Remove Hardcoded Types

Per `.cursor/rules/schema-evolution.mdc`:
> "NO hardcoded entity types - autonomous schema evolution"

Replace static entity type lists with discovered types.

## Success Criteria (From Quality Gates)

- [ ] Semantic alignment >90% vs manual schema
- [ ] Reflection triggers every N documents (N=50)
- [ ] Schema versioning operational
- [ ] Multi-phase extraction: Entity→Entity, Entity→Event, Event→Event
- [ ] No hardcoded entity types

## Files to Create/Modify

### Modify:
- `src/futurnal/extraction/schema/discovery.py` - Full implementation
- `src/futurnal/extraction/schema/evolution.py` - Full implementation

### Create:
- `src/futurnal/extraction/schema/reflection.py` - Reflection mechanism
- `src/futurnal/extraction/schema/versioning.py` - Version storage

### Tests:
- `tests/extraction/schema/test_discovery.py`
- `tests/extraction/schema/test_evolution.py`

## Dependencies

- **Step 04**: Temporal extraction (Event types must be temporal-aware)

## Next Step

After implementing schema evolution, proceed to **Step 06: Experiential Learning**.

## Research References

1. **AutoSchemaKG**: `docs/phase-1/SOTA_RESEARCH_SUMMARY.md` (Paper #1)
2. **EDC Framework**: `docs/phase-1/papers/converted/2404.03868.md`
3. **Schema Rules**: `.cursor/rules/schema-evolution.mdc`

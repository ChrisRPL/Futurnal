Summary: Define comprehensive graph schema with temporal and causal metadata support for Option B requirements.

# 01 · Graph Schema Design

## Purpose
Establish the canonical graph schema for the Personal Knowledge Graph (PKG) that supports Option B requirements including temporal metadata, causal relationships, schema evolution, and provenance tracking. This schema serves as the contract between the extraction pipeline and all downstream components (vector embeddings, hybrid search, Phase 2/3 analytics).

**Criticality**: CRITICAL - Foundation for all three phases; must support temporal queries and causal inference

## Scope
- Node types: Static entities + Events (temporal grounding)
- Relationship types: Standard + temporal/causal
- Provenance and versioning metadata
- Schema evolution support (AutoSchemaKG)
- Integration with extraction pipeline outputs

## Requirements Alignment
- **Option B Requirement**: "Schema accommodates temporal metadata, causal structure, schema versioning"
- **Phase 2 Foundation**: Temporal queries for correlation detection
- **Phase 3 Foundation**: Causal chains for hypothesis validation
- **Privacy-First**: No content in schema, only metadata

## Schema Design

### Node Types

#### Static Entities

```cypher
// Person Entity
CREATE CONSTRAINT person_id_unique IF NOT EXISTS
FOR (p:Person) REQUIRE p.id IS UNIQUE;

CREATE (p:Person {
  id: string,                    // UUID
  name: string,                  // Primary name
  aliases: [string],             // Alternative names/nicknames

  // Metadata
  created_at: datetime,
  updated_at: datetime,
  discovery_count: int,          // How many times discovered
  confidence: float,             // Average confidence across extractions

  // Provenance
  first_seen_document: string,   // Document ID where first discovered

  // Properties (extensible via map)
  properties: map                // Additional properties as key-value pairs
})

// Organization Entity
CREATE CONSTRAINT organization_id_unique IF NOT EXISTS
FOR (o:Organization) REQUIRE o.id IS UNIQUE;

CREATE (o:Organization {
  id: string,
  name: string,
  type: string,                  // "company", "institution", "group", etc.
  aliases: [string],

  // Metadata
  created_at: datetime,
  updated_at: datetime,
  confidence: float,

  // Properties
  properties: map
})

// Concept Entity
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE (c:Concept {
  id: string,
  name: string,
  description: string,
  category: string,              // "topic", "idea", "field", etc.
  aliases: [string],

  // Metadata
  created_at: datetime,
  updated_at: datetime,
  confidence: float,

  // Properties
  properties: map
})

// Document Entity (source tracking)
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE (d:Document {
  id: string,
  source_id: string,             // Connector-specific ID
  source_type: string,           // "obsidian_vault", "imap_mailbox", etc.
  content_hash: string,          // SHA-256 of content

  // Temporal
  created_at: datetime,
  modified_at: datetime,
  ingested_at: datetime,

  // Format
  format: string,                // "markdown", "email", "code", etc.

  // Properties
  properties: map
})
```

#### Event Entities (NEW - Option B)

```cypher
// Event Entity - Temporal grounding required
CREATE CONSTRAINT event_id_unique IF NOT EXISTS
FOR (e:Event) REQUIRE e.id IS UNIQUE;

CREATE (e:Event {
  id: string,
  name: string,
  event_type: string,            // "meeting", "decision", "publication", etc.
  description: string,

  // REQUIRED: Temporal grounding (distinguishes events from static entities)
  timestamp: datetime,           // When did this occur?
  duration: duration,            // How long? (optional)
  end_timestamp: datetime,       // When did it end? (computed or explicit)

  // Participants (linked via relationships)
  // participants: [Entity IDs] - stored as relationships instead

  // Location/Context
  location: string,              // Optional physical/virtual location
  context: string,               // Brief context

  // Metadata
  created_at: datetime,
  confidence: float,

  // Provenance
  source_document: string,       // Document ID
  extraction_method: string,     // "explicit", "inferred", "llm"

  // Properties
  properties: map
})
```

#### Schema Versioning (NEW - Option B)

```cypher
// Schema Version tracking for autonomous evolution
CREATE CONSTRAINT schema_version_id_unique IF NOT EXISTS
FOR (sv:SchemaVersion) REQUIRE sv.id IS UNIQUE;

CREATE (sv:SchemaVersion {
  id: string,
  version: int,                  // Incrementing version number
  created_at: datetime,

  // Schema snapshot
  entity_types: [string],        // List of entity type names
  relationship_types: [string],  // List of relationship type names

  // Evolution metadata
  changes: string,               // JSON description of changes from previous version
  reflection_quality: float,     // Quality metrics that triggered evolution
  parent_version: int,           // Previous version number

  // Stats
  documents_processed: int,      // Total documents when this version created

  // Properties
  properties: map
})
```

### Relationship Types

#### Standard Relationships

```cypher
// Generic relationship with temporal validity
CREATE (a)-[r:RELATED_TO {
  // Temporal validity
  valid_from: datetime,          // When did this relationship start?
  valid_to: datetime,            // When did it end? (null = ongoing)

  // Confidence
  confidence: float,

  // Provenance
  source_document: string,
  extraction_method: string,
  model_version: string,
  template_version: string,      // NEW - thought template tracking

  // Created
  created_at: datetime
}]->(b)

// Specific relationship types (examples)
// Person → Organization
CREATE (p:Person)-[:WORKS_AT {
  role: string,
  valid_from: datetime,
  valid_to: datetime,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(o:Organization)

// Person/Organization → Concept/Document
CREATE (a)-[:CREATED {
  creation_date: datetime,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(b)

// Concept → Concept
CREATE (c1:Concept)-[:RELATED_TO {
  relationship_type: string,     // "subset_of", "similar_to", "contrasts_with"
  strength: float,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(c2:Concept)
```

#### Temporal Relationships (NEW - Option B)

```cypher
// Event → Event: BEFORE
CREATE (e1:Event)-[:BEFORE {
  temporal_confidence: float,    // Confidence in temporal ordering
  temporal_source: string,       // "explicit_timestamp", "relative_expression", "inferred"
  temporal_gap: duration,        // Time between events

  // Standard metadata
  confidence: float,
  source_document: string,
  extraction_method: string,
  created_at: datetime
}]->(e2:Event)

// Event → Event: AFTER (inverse of BEFORE)
CREATE (e1:Event)-[:AFTER {
  temporal_confidence: float,
  temporal_source: string,
  temporal_gap: duration,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(e2:Event)

// Event → Event: DURING
CREATE (e1:Event)-[:DURING {
  overlap_start: datetime,       // When overlap started
  overlap_end: datetime,         // When overlap ended
  overlap_type: string,          // "contains", "contained_by", "partial"

  temporal_confidence: float,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(e2:Event)

// Event → Event: SIMULTANEOUS
CREATE (e1:Event)-[:SIMULTANEOUS {
  simultaneity_tolerance: duration,  // How close in time?
  temporal_confidence: float,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(e2:Event)
```

#### Causal Relationships (NEW - Option B)

```cypher
// Event → Event: CAUSES (causal candidate for Phase 3)
CREATE (cause:Event)-[:CAUSES {
  // Causal metadata
  causal_confidence: float,      // Confidence this is causal
  causal_evidence: string,       // Text snippet supporting causation
  is_causal_candidate: boolean,  // Flagged for Phase 3 validation
  is_validated: boolean,         // Has Phase 3 validated this?
  validation_method: string,     // How was it validated?

  // Temporal requirements (cause must precede effect)
  temporal_gap: duration,        // Time between cause and effect
  temporal_ordering_valid: boolean,  // Is cause before effect?

  // Bradford Hill criteria (Phase 3 validation)
  temporality_satisfied: boolean,    // Cause before effect?
  strength: float,                   // Association strength (optional)
  dose_response: boolean,            // More cause → more effect? (optional)
  consistency: float,                // Replicable? (optional)
  plausibility: string,              // Mechanistic explanation (optional)

  // Standard metadata
  confidence: float,
  source_document: string,
  extraction_method: string,
  created_at: datetime
}]->(effect:Event)

// Event → Event: ENABLES
CREATE (e1:Event)-[:ENABLES {
  causal_confidence: float,
  causal_evidence: string,
  temporal_gap: duration,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(e2:Event)

// Event → Event: PREVENTS
CREATE (e1:Event)-[:PREVENTS {
  causal_confidence: float,
  causal_evidence: string,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(e2:Event)

// Event → Event: TRIGGERS
CREATE (e1:Event)-[:TRIGGERS {
  trigger_type: string,          // "immediate", "delayed"
  causal_confidence: float,
  temporal_gap: duration,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(e2:Event)
```

#### Provenance Relationships

```cypher
// Triple → Chunk (provenance tracking)
CREATE (t:Triple)-[:EXTRACTED_FROM {
  extraction_timestamp: datetime,
  extraction_confidence: float
}]->(c:Chunk {
  id: string,
  document_id: string,
  content_hash: string,
  position: int,
  chunk_index: int,
  created_at: datetime
})

// Entity/Relationship → Document (source tracking)
CREATE (entity)-[:DISCOVERED_IN {
  discovery_timestamp: datetime,
  discovery_method: string,
  confidence: float
}]->(document:Document)

// Entity → Event (participation)
CREATE (person:Person)-[:PARTICIPATED_IN {
  role: string,                  // "organizer", "attendee", "speaker"
  participation_confirmed: boolean,
  confidence: float,
  source_document: string,
  created_at: datetime
}]->(event:Event)
```

### Indices for Performance

```cypher
// Entity indices
CREATE INDEX person_name_index IF NOT EXISTS
FOR (p:Person) ON (p.name);

CREATE INDEX organization_name_index IF NOT EXISTS
FOR (o:Organization) ON (o.name);

CREATE INDEX concept_name_index IF NOT EXISTS
FOR (c:Concept) ON (c.name);

// Event indices (critical for temporal queries)
CREATE INDEX event_timestamp_index IF NOT EXISTS
FOR (e:Event) ON (e.timestamp);

CREATE INDEX event_type_index IF NOT EXISTS
FOR (e:Event) ON (e.event_type);

// Document indices
CREATE INDEX document_source_id_index IF NOT EXISTS
FOR (d:Document) ON (d.source_id);

CREATE INDEX document_content_hash_index IF NOT EXISTS
FOR (d:Document) ON (d.content_hash);

// Composite index for temporal range queries
CREATE INDEX event_timestamp_type_index IF NOT EXISTS
FOR (e:Event) ON (e.timestamp, e.event_type);
```

## Schema Evolution Support

### Version Migration

```python
class SchemaVersionManager:
    """Manage schema versions and migrations."""

    def create_new_version(
        self,
        changes: Dict[str, Any],
        quality_metrics: Dict[str, float]
    ) -> SchemaVersion:
        """
        Create new schema version.

        Called by schema evolution engine when reflection triggers upgrade.
        """
        current = self.get_current_version()

        new_version = SchemaVersion(
            id=f"schema_v{current.version + 1}",
            version=current.version + 1,
            created_at=datetime.utcnow(),
            entity_types=changes.get("entity_types", current.entity_types),
            relationship_types=changes.get("relationship_types", current.relationship_types),
            changes=json.dumps(changes),
            reflection_quality=quality_metrics.get("should_refine", 0.0),
            parent_version=current.version
        )

        # Store in PKG
        self.store_version(new_version)

        return new_version

    def migrate_data(
        self,
        from_version: int,
        to_version: int
    ):
        """
        Migrate PKG data between schema versions.

        Typically additive (new types/properties) rather than destructive.
        """
        migration_plan = self.generate_migration_plan(from_version, to_version)

        for step in migration_plan:
            self.execute_migration_step(step)
```

## Testing Strategy

```python
class TestGraphSchema:
    """Test graph schema design and constraints."""

    def test_node_type_creation(self):
        """Validate all node types can be created."""
        # Create instances of each node type
        person = create_person_node()
        org = create_organization_node()
        concept = create_concept_node()
        event = create_event_node()
        document = create_document_node()

        assert person.id is not None
        assert event.timestamp is not None  # Events require timestamp

    def test_relationship_types(self):
        """Validate all relationship types functional."""
        person = create_person_node()
        org = create_organization_node()
        event = create_event_node()

        # Standard relationship
        works_at = create_works_at_relationship(person, org)
        assert works_at.valid_from is not None

        # Temporal relationship
        attended = create_participated_in_relationship(person, event)
        assert attended.confidence > 0

    def test_temporal_ordering_constraint(self):
        """Validate temporal relationships enforce ordering."""
        event1 = create_event_node(timestamp=datetime(2024, 1, 1))
        event2 = create_event_node(timestamp=datetime(2024, 1, 2))

        # Should succeed: event1 before event2
        before_rel = create_before_relationship(event1, event2)
        assert before_rel.temporal_ordering_valid

        # Should fail or warn: event2 before event1 (reversed)
        with pytest.raises(TemporalOrderingError):
            create_before_relationship(event2, event1)

    def test_schema_versioning(self):
        """Validate schema versioning functional."""
        v1 = create_initial_schema_version()
        assert v1.version == 1

        # Evolve schema
        v2 = evolve_schema(v1, add_entity_type="Project")
        assert v2.version == 2
        assert "Project" in v2.entity_types
        assert v2.parent_version == 1
```

## Success Metrics

- ✅ All node types creatable with proper constraints
- ✅ All relationship types functional
- ✅ Temporal relationships enforce ordering
- ✅ Causal relationships support Phase 3 validation
- ✅ Schema versioning operational
- ✅ Indices created for performance
- ✅ Provenance tracking complete

## Dependencies

- Neo4j embedded or equivalent graph database
- Entity-relationship extraction pipeline (provides triples)
- Schema evolution engine (02-schema-evolution.md from extraction)

## Next Steps

After schema design complete:
1. Implement database setup (02-database-setup.md)
2. Build data access layer (03-data-access-layer.md)
3. Enable temporal queries (04-temporal-query-support.md)

**This schema is the foundation for Ghost→Animal evolution across all three phases.**

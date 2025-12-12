# Phase 1 Completion Plan: Bridging Implementation to Vision

> **Current State**: File browser with graph visualization
> **Target State**: Experiential memory system with semantic understanding

## Executive Summary

Phase 1 (Archivist) is architecturally complete but **functionally disconnected**. The Python backend has sophisticated extraction, embedding, and search infrastructure, but:

1. **Entity extraction isn't running** during ingestion
2. **Search isn't wired** to the desktop app
3. **Graph shows files, not knowledge** - no semantic entities or relationships

This document outlines the critical path to achieve the Futurnal vision: *"Move from 'What did I know?' to 'Why do I think this?'"*

---

## Current vs Target State

### Knowledge Graph Visualization

| Aspect | Current State | Target State |
|--------|---------------|--------------|
| **Node Types** | Document, Email, Source only | + Person, Concept, Event, Organization |
| **Relationships** | "contains" only | mentions, written_by, discusses, cites, causes |
| **Labels** | Raw filenames (`2508.19855v3.md`) | Extracted titles ("Attention Is All You Need") |
| **Cross-Source Links** | None (disconnected clusters) | Emails link to related documents via shared entities |
| **Temporal Awareness** | None | Events ordered, causal chains visible |

### Search Functionality

| Aspect | Current State | Target State |
|--------|---------------|--------------|
| **Desktop Search** | Stubbed (returns empty) | Full hybrid vector + graph search |
| **Query Types** | None | Temporal, causal, semantic, exploratory |
| **Results** | Empty | Ranked results with provenance |

---

## Critical Implementation Gaps

### Gap 1: Entity Extraction Not Running (CRITICAL)

**Problem**: The `SemanticTripleExtractor` exists but isn't called during ingestion. Unstructured.io creates document elements, but no entities (Person, Concept, Event) are extracted from content.

**Impact**: Graph is just a file index, not a knowledge graph.

**Location**:
- Python: `src/futurnal/pipeline/triples.py` (exists, not used)
- Rust: `desktop/src-tauri/src/commands/graph.rs` (reads files, no extraction)

**Solution**: Wire entity extraction into the ingestion pipeline:

```
Current:  Document → Unstructured.io → JSON files → Graph nodes (files only)
Target:   Document → Unstructured.io → Entity Extraction → JSON + Entities → Graph nodes (semantic)
```

### Gap 2: Search Not Wired to Desktop (CRITICAL)

**Problem**: `desktop/src-tauri/src/commands/search.rs` returns empty results with TODO comments.

**Impact**: Users cannot search their knowledge base.

**Location**: `search.rs:15-25` - stubbed `search_query()` function

**Solution**: Wire Rust commands to Python CLI search module.

### Gap 3: No Cross-Source Connections (HIGH)

**Problem**: Email cluster and document cluster are completely disconnected.

**Impact**: Cannot discover that an email discusses a concept from your documents.

**Solution**: Entity extraction creates shared nodes (Person, Concept) that connect sources.

### Gap 4: Raw Labels Instead of Titles (MEDIUM)

**Problem**: Documents show `2508.19855v3.md` instead of extracted titles.

**Impact**: Poor UX, no semantic understanding visible.

**Solution**: Extract title from document content during ingestion.

### Gap 5: Temporal Queries Non-Functional (MEDIUM)

**Problem**: `TemporalGraphQueries` API is scaffolded but not implemented.

**Impact**: Cannot query "What was I working on last week?"

**Location**: `src/futurnal/pkg/queries/temporal.py`

---

## Implementation Roadmap

### Phase 1A: Core Pipeline Integration (Week 1-2)

#### 1.1 Wire Entity Extraction to Ingestion

**Files to modify**:
- `src/futurnal/ingestion/orchestrator.py` - Add extraction step
- `src/futurnal/pipeline/triples.py` - Ensure entity extraction runs
- `desktop/src-tauri/src/commands/graph.rs` - Load entity nodes

**Implementation**:
```python
# After document normalization, before storage:
async def process_document(doc: NormalizedDocument) -> ProcessedDocument:
    # 1. Existing: Create document elements via Unstructured.io
    elements = await partition_document(doc)

    # 2. NEW: Extract entities from content
    entities = await extract_entities(doc.content)
    # Returns: [Person("John Smith"), Concept("machine learning"), Event("meeting")]

    # 3. NEW: Extract relationships
    relationships = await extract_relationships(doc, entities)
    # Returns: [("doc", "mentions", "John Smith"), ("doc", "discusses", "machine learning")]

    # 4. Store all to PKG
    await pkg_writer.write_document(doc, entities, relationships)
```

**Entity Types to Extract**:
| Entity Type | Source | Extraction Method |
|-------------|--------|-------------------|
| Person | Email sender/recipient, document mentions | Regex + NER |
| Concept | Document content, email body | Keyword extraction + embeddings |
| Event | Calendar data, temporal markers | Temporal extraction module |
| Organization | Email domains, document mentions | NER + pattern matching |

#### 1.2 Wire Search to Desktop

**Files to modify**:
- `desktop/src-tauri/src/commands/search.rs` - Implement actual search
- `src/futurnal/cli/search.py` - Add CLI command (if missing)

**Implementation**:
```rust
// search.rs - Replace stub with actual implementation
#[tauri::command]
pub async fn search_query(query: String, options: SearchOptions) -> Result<SearchResults, String> {
    // Call Python hybrid search API
    let result = call_python_cli(&["search", "--query", &query, "--json"])?;
    Ok(serde_json::from_str(&result)?)
}
```

#### 1.3 Improve Graph Node Labels

**Files to modify**:
- `desktop/src-tauri/src/commands/graph.rs` - Extract title from content

**Implementation**:
```rust
fn extract_title(content: &str, filename: &str) -> String {
    // 1. Try frontmatter title
    if let Some(title) = extract_frontmatter_title(content) {
        return title;
    }
    // 2. Try first heading
    if let Some(heading) = extract_first_heading(content) {
        return heading;
    }
    // 3. Try email subject
    if let Some(subject) = extract_email_subject(content) {
        return subject;
    }
    // 4. Fallback to filename without extension
    filename.trim_end_matches(".md").trim_end_matches(".json").to_string()
}
```

### Phase 1B: Semantic Graph Enhancement (Week 3-4)

#### 2.1 Implement Basic NER for Person/Organization

**New file**: `src/futurnal/extraction/ner/basic_extractor.py`

```python
class BasicNERExtractor:
    """Extract Person and Organization entities using spaCy."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(Person(name=ent.text, source="ner"))
            elif ent.label_ == "ORG":
                entities.append(Organization(name=ent.text, source="ner"))
        return entities
```

#### 2.2 Implement Concept Extraction

**New file**: `src/futurnal/extraction/concepts/keyword_extractor.py`

```python
class ConceptExtractor:
    """Extract key concepts using TF-IDF + embedding clustering."""

    def extract(self, text: str, top_k: int = 5) -> List[Concept]:
        # 1. Extract keywords via TF-IDF
        keywords = self.tfidf_keywords(text, top_k=20)

        # 2. Cluster similar keywords via embeddings
        clusters = self.cluster_keywords(keywords)

        # 3. Return top concepts
        return [Concept(name=c.representative, keywords=c.members)
                for c in clusters[:top_k]]
```

#### 2.3 Create Cross-Source Relationships

**Logic**: When same entity appears in multiple sources, create linking relationships.

```python
async def link_cross_source_entities(pkg: PKGWriter):
    """Find and link entities that appear across sources."""

    # Find Person nodes mentioned in multiple documents
    query = """
    MATCH (p:Person)<-[:MENTIONS]-(d1:Document)
    MATCH (p)<-[:MENTIONS]-(d2:Document)
    WHERE d1.source <> d2.source AND id(d1) < id(d2)
    MERGE (d1)-[:SHARES_ENTITY {entity: p.name}]->(d2)
    """
    await pkg.execute(query)

    # Find Concept nodes shared between emails and documents
    query = """
    MATCH (c:Concept)<-[:DISCUSSES]-(e:Email)
    MATCH (c)<-[:DISCUSSES]-(d:Document)
    MERGE (e)-[:RELATED_TO {via: c.name}]->(d)
    """
    await pkg.execute(query)
```

### Phase 1C: Search & Temporal Features (Week 5-6)

#### 3.1 Complete Temporal Query Implementation

**Files to modify**: `src/futurnal/pkg/queries/temporal.py`

```python
class TemporalGraphQueries:
    async def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime
    ) -> List[Event]:
        query = """
        MATCH (e:Event)
        WHERE e.timestamp >= $start AND e.timestamp <= $end
        RETURN e
        ORDER BY e.timestamp
        """
        return await self.execute(query, start=start, end=end)

    async def query_documents_by_period(
        self,
        period: str  # "today", "this_week", "last_month"
    ) -> List[Document]:
        start, end = self.parse_period(period)
        query = """
        MATCH (d:Document)
        WHERE d.created_at >= $start AND d.created_at <= $end
        RETURN d
        ORDER BY d.created_at DESC
        """
        return await self.execute(query, start=start, end=end)
```

#### 3.2 Implement Hybrid Search End-to-End

**Test case**: User searches "machine learning papers I read last month"

1. **Intent Detection**: Temporal + Topic query
2. **Graph Query**: Find documents from last month
3. **Vector Query**: Find documents semantically similar to "machine learning"
4. **Fusion**: Combine results, rank by relevance + recency
5. **Return**: Ranked documents with provenance

### Phase 1D: Quality & Polish (Week 7-8)

#### 4.1 Embedding-PKG Synchronization

Ensure vector embeddings stay in sync when PKG changes:

```python
class PKGSyncHandler:
    async def on_entity_created(self, entity: Entity):
        embedding = await self.embedding_service.embed(entity)
        await self.chroma.upsert(entity.id, embedding)

    async def on_entity_updated(self, entity: Entity):
        embedding = await self.embedding_service.embed(entity)
        await self.chroma.upsert(entity.id, embedding)

    async def on_entity_deleted(self, entity_id: str):
        await self.chroma.delete(entity_id)
```

#### 4.2 Graph Visualization Enhancements

- Different colors for different entity types
- Entity type icons (Person = user icon, Concept = lightbulb, etc.)
- Relationship labels visible on hover
- Temporal layout option (x-axis = time)

---

## Success Criteria

### Minimum Viable Phase 1

- [ ] **Entity Extraction**: Person and Concept entities extracted from documents
- [ ] **Semantic Relationships**: "mentions", "discusses" relationships in graph
- [ ] **Cross-Source Links**: Emails connected to documents via shared entities
- [ ] **Desktop Search**: Basic keyword + semantic search working
- [ ] **Meaningful Labels**: Document titles instead of filenames

### Full Phase 1 Completion

- [ ] All entity types extracted (Person, Organization, Concept, Event)
- [ ] All relationship types implemented (mentions, written_by, discusses, cites, causes)
- [ ] Temporal queries functional ("What was I working on last week?")
- [ ] Hybrid search with <1s latency
- [ ] Graph shows semantic network, not file tree
- [ ] Causal chain visualization for Event→Event relationships

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| NER model too slow for on-device | High | Use lightweight spaCy model, batch processing |
| Too many entities overwhelm graph | Medium | Confidence thresholds, entity merging |
| Embedding sync causes data loss | High | Transaction safety, audit logging |
| Search latency exceeds 1s | Medium | Caching, query optimization, indexing |

---

## Resource Requirements

### Dependencies to Add
- `spacy` with `en_core_web_sm` model (~50MB)
- Consider `sentence-transformers` for concept extraction

### Compute Requirements
- Entity extraction: ~100ms per document (spaCy)
- Embedding generation: ~50ms per entity (local model)
- Graph queries: ~10ms per query (Neo4j)

---

## Appendix: Current Architecture Gap Diagram

```
CURRENT FLOW (File-Level Only):
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│ Data Source  │───▶│ Unstructured.io │───▶│ JSON Files   │───▶│ Graph Nodes │
│ (IMAP/Files) │    │ (partition)     │    │ (elements)   │    │ (Documents) │
└──────────────┘    └─────────────────┘    └──────────────┘    └─────────────┘
                                                                      │
                                                                      ▼
                                                               ┌─────────────┐
                                                               │ Star Graph  │
                                                               │ Source→Doc  │
                                                               └─────────────┘

TARGET FLOW (Semantic Knowledge Graph):
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│ Data Source  │───▶│ Unstructured.io │───▶│ JSON Files   │───▶│ Doc Nodes   │
│ (IMAP/Files) │    │ (partition)     │    │ (elements)   │    │             │
└──────────────┘    └─────────────────┘    └──────────────┘    └──────┬──────┘
                                                                      │
                           ┌──────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
                    │ Entity Extractor│───▶│ NER + Concept│───▶│ Entity Nodes│
                    │ (NEW)           │    │ Extraction   │    │ Person/Org/ │
                    └─────────────────┘    └──────────────┘    │ Concept     │
                                                               └──────┬──────┘
                                                                      │
                           ┌──────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
                    │ Relationship    │───▶│ Cross-Source │───▶│ Semantic    │
                    │ Extractor (NEW) │    │ Linking      │    │ Graph       │
                    └─────────────────┘    └──────────────┘    └─────────────┘
```

---

## Next Steps

1. **Immediate**: Start with Phase 1A.1 (Wire entity extraction)
2. **This Week**: Complete Phase 1A (Core pipeline integration)
3. **Next Week**: Phase 1B (Semantic graph enhancement)
4. **Week 3-4**: Phase 1C (Search & temporal features)

---

*This plan ensures Phase 1 delivers the Futurnal vision: a true Personal Knowledge Graph with semantic understanding, not just a file browser with a graph UI.*

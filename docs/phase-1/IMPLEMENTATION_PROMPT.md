# Phase 1 Completion Implementation Prompt

Copy the content below into a fresh Claude Code session:

---

## START OF PROMPT

You are implementing the Phase 1 (Archivist) completion for Futurnal - a privacy-first personal knowledge and causal insight engine. The architecture is built but key integrations are missing.

### Project Context

**Vision**: Move from "What did I know?" to "Why do I think this?" through proactive AI-assisted analysis of personal data.

**Tech Stack**:
- Python 3.11+ backend with Typer CLI (`src/futurnal/`)
- Rust/Tauri desktop app (`desktop/src-tauri/`)
- React/TypeScript frontend (`desktop/src/`)
- Neo4j for graph storage, ChromaDB for vectors

**Current Problem**: The knowledge graph shows a FILE BROWSER (Source → contains → Document) instead of a KNOWLEDGE GRAPH (semantic entities and relationships). The Python backend has sophisticated extraction code that ISN'T BEING USED.

### What's Implemented vs Missing

| Component | Python Backend | Rust Desktop | Status |
|-----------|---------------|--------------|--------|
| Entity Extraction | `src/futurnal/pipeline/triples.py` exists | Not called | **NOT WIRED** |
| Search API | `src/futurnal/search/api.py` (380+ LOC) | `search.rs` returns empty | **STUBBED** |
| NER | `src/futurnal/extraction/` scaffolded | N/A | **PARTIAL** |
| Temporal Queries | `src/futurnal/pkg/queries/temporal.py` | N/A | **SCAFFOLDED** |
| Embeddings | `src/futurnal/embeddings/` (80% done) | N/A | **WORKING** |

### Implementation Tasks

Execute these in order. Use the TodoWrite tool to track progress. Read existing code before modifying.

---

## TASK 1: Wire Entity Extraction to Graph (CRITICAL)

**Goal**: Extract Person, Concept entities from documents during ingestion so they appear in the knowledge graph.

**Current Flow** (in `desktop/src-tauri/src/commands/graph.rs`):
```
Read JSON files → Create one node per file → Return star graph
```

**Target Flow**:
```
Read JSON files → Extract entities from content → Create document + entity nodes → Create relationships → Return semantic graph
```

**Implementation Steps**:

1. Read `desktop/src-tauri/src/commands/graph.rs` to understand current `get_knowledge_graph()` function

2. Add entity extraction to the Rust backend:
   - Extract Person entities from email sender/recipient fields
   - Extract Concept entities from document content (keywords/topics)
   - Create entity nodes in addition to document nodes
   - Create "mentions" relationships between documents and entities

3. Modify the graph response to include:
   - Document nodes (existing)
   - Person nodes (new) - extracted from emails
   - Concept nodes (new) - extracted from content
   - "mentions" and "discusses" relationships (new)

**Key Files**:
- `desktop/src-tauri/src/commands/graph.rs` - Main graph logic
- `desktop/src/types/api.ts` - TypeScript types (add Person, Concept node types)
- `desktop/src/components/graph/KnowledgeGraph.tsx` - Graph visualization

**Entity Extraction Logic** (implement in Rust or call Python):
```rust
// For emails - extract Person from sender/recipient
fn extract_email_entities(metadata: &ImapMetadata) -> Vec<Entity> {
    let mut entities = vec![];
    if let Some(sender) = &metadata.sender {
        entities.push(Entity::Person { name: sender.clone() });
    }
    if let Some(recipients) = &metadata.recipients {
        for r in recipients {
            entities.push(Entity::Person { name: r.clone() });
        }
    }
    entities
}

// For documents - extract concepts from text
fn extract_document_concepts(text: &str) -> Vec<Entity> {
    // Simple keyword extraction - find capitalized phrases, technical terms
    // Or call Python NER: call_python_cli(&["extract-entities", "--text", text])
}
```

**Success Criteria**:
- [ ] Graph shows Person nodes connected to emails
- [ ] Graph shows Concept nodes connected to documents
- [ ] Documents/emails sharing entities are visually connected

---

## TASK 2: Wire Search to Desktop (CRITICAL)

**Goal**: Make search functional in the desktop app.

**Current State**: `desktop/src-tauri/src/commands/search.rs` has TODO comments and returns empty results.

**Implementation Steps**:

1. Read `desktop/src-tauri/src/commands/search.rs` to see current stubs

2. Read `src/futurnal/search/api.py` to understand the Python search API

3. Implement the Rust→Python bridge:
```rust
#[tauri::command]
pub async fn search_query(
    query: String,
    filters: Option<SearchFilters>
) -> Result<SearchResults, String> {
    // Call Python search CLI
    let mut args = vec!["search", "--query", &query, "--format", "json"];

    if let Some(f) = filters {
        if let Some(types) = f.entity_types {
            args.push("--types");
            args.push(&types.join(","));
        }
    }

    let output = call_python_cli(&args)
        .map_err(|e| format!("Search failed: {}", e))?;

    serde_json::from_str(&output)
        .map_err(|e| format!("Parse error: {}", e))
}
```

4. Add search CLI command to Python if missing:
   - Check `src/futurnal/cli/` for existing search command
   - If missing, create `src/futurnal/cli/search.py` that calls `HybridSearchAPI`

5. Update frontend to use real search:
   - Check `desktop/src/lib/api.ts` for search API calls
   - Ensure results are displayed properly

**Key Files**:
- `desktop/src-tauri/src/commands/search.rs` - Rust search command
- `src/futurnal/search/api.py` - Python HybridSearchAPI
- `src/futurnal/cli/` - CLI commands
- `desktop/src/lib/api.ts` - Frontend API layer

**Success Criteria**:
- [ ] User can type a query and get results
- [ ] Results show documents with relevance scores
- [ ] Results link to source documents

---

## TASK 3: Extract Titles from Documents

**Goal**: Show meaningful titles instead of raw filenames in the graph.

**Current State**: Documents show `2508.19855v3.md` instead of "Attention Is All You Need"

**Implementation** (in `graph.rs`):
```rust
fn extract_title(content: &str, filename: &str) -> String {
    // 1. Try YAML frontmatter title
    if let Some(title) = extract_frontmatter_field(content, "title") {
        return title;
    }

    // 2. Try first markdown heading
    if let Some(caps) = Regex::new(r"^#\s+(.+)$").unwrap().captures(content) {
        return caps[1].to_string();
    }

    // 3. Try email subject
    if let Some(title) = extract_frontmatter_field(content, "subject") {
        return title;
    }

    // 4. First line if short enough
    if let Some(first_line) = content.lines().next() {
        if first_line.len() < 100 && !first_line.starts_with('{') {
            return first_line.to_string();
        }
    }

    // 5. Filename without extension
    filename
        .trim_end_matches(".json")
        .trim_end_matches(".md")
        .to_string()
}
```

**Success Criteria**:
- [ ] Documents show extracted titles
- [ ] Emails show subjects
- [ ] Fallback to clean filename (no extension)

---

## TASK 4: Implement Cross-Source Connections

**Goal**: Connect emails and documents that share entities (same person mentioned, same topic discussed).

**Implementation**:

1. After extracting entities from all sources, find shared entities:
```rust
fn find_cross_source_links(nodes: &[GraphNode]) -> Vec<GraphLink> {
    let mut entity_to_docs: HashMap<String, Vec<String>> = HashMap::new();
    let mut links = vec![];

    // Group documents by shared entities
    for node in nodes {
        if let Some(entities) = &node.entities {
            for entity in entities {
                entity_to_docs
                    .entry(entity.clone())
                    .or_default()
                    .push(node.id.clone());
            }
        }
    }

    // Create links between documents sharing entities
    for (entity, doc_ids) in entity_to_docs {
        if doc_ids.len() > 1 {
            for i in 0..doc_ids.len() {
                for j in (i+1)..doc_ids.len() {
                    links.push(GraphLink {
                        source: doc_ids[i].clone(),
                        target: doc_ids[j].clone(),
                        relationship: format!("shares:{}", entity),
                    });
                }
            }
        }
    }

    links
}
```

**Success Criteria**:
- [ ] Emails from same sender are connected
- [ ] Documents discussing same concepts are connected
- [ ] Graph shows interconnected network, not isolated clusters

---

## TASK 5: Implement Basic NER (Python)

**Goal**: Add Named Entity Recognition for better Person/Organization extraction.

**Implementation**:

1. Create `src/futurnal/extraction/ner/extractor.py`:
```python
import spacy
from dataclasses import dataclass
from typing import List

@dataclass
class ExtractedEntity:
    text: str
    label: str  # PERSON, ORG, GPE, etc.
    confidence: float

class NERExtractor:
    def __init__(self):
        # Use small model for speed
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> List[ExtractedEntity]:
        doc = self.nlp(text[:10000])  # Limit for performance
        return [
            ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                confidence=0.8  # spaCy doesn't provide confidence
            )
            for ent in doc.ents
            if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT")
        ]
```

2. Add CLI command `src/futurnal/cli/extract.py`:
```python
@app.command()
def entities(
    text: str = typer.Option(..., "--text"),
    format: str = typer.Option("json", "--format")
):
    """Extract named entities from text."""
    extractor = NERExtractor()
    entities = extractor.extract(text)
    if format == "json":
        print(json.dumps([asdict(e) for e in entities]))
```

3. Add spacy to requirements.txt:
```
spacy>=3.7.0
```

4. Download model in setup:
```bash
python -m spacy download en_core_web_sm
```

**Success Criteria**:
- [ ] `futurnal extract entities --text "John Smith met with Google"` returns Person and Org
- [ ] Rust can call this CLI and parse results

---

## TASK 6: Implement Temporal Queries

**Goal**: Enable queries like "What was I working on last week?"

**Implementation**:

1. Complete `src/futurnal/pkg/queries/temporal.py`:
```python
from datetime import datetime, timedelta

class TemporalQueries:
    def __init__(self, driver):
        self.driver = driver

    def documents_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[Document]:
        query = """
        MATCH (d:Document)
        WHERE d.created_at >= $start AND d.created_at <= $end
        RETURN d
        ORDER BY d.created_at DESC
        """
        with self.driver.session() as session:
            result = session.run(query, start=start.isoformat(), end=end.isoformat())
            return [Document.from_record(r) for r in result]

    def documents_by_period(self, period: str) -> List[Document]:
        now = datetime.now()
        periods = {
            "today": (now.replace(hour=0, minute=0), now),
            "yesterday": (now - timedelta(days=1), now - timedelta(days=1)),
            "this_week": (now - timedelta(days=now.weekday()), now),
            "last_week": (now - timedelta(days=now.weekday() + 7),
                         now - timedelta(days=now.weekday())),
            "this_month": (now.replace(day=1), now),
        }
        start, end = periods.get(period, (now - timedelta(days=7), now))
        return self.documents_in_range(start, end)
```

2. Add to search API to handle temporal queries

**Success Criteria**:
- [ ] Search "last week" returns documents from last 7 days
- [ ] Graph can filter by time period

---

## TASK 7: Update Graph Visualization

**Goal**: Visual distinction for entity types, better relationship display.

**Implementation**:

1. Update `desktop/src/types/api.ts`:
```typescript
export type EntityType =
  | 'Document'
  | 'Email'
  | 'Person'
  | 'Organization'
  | 'Concept'
  | 'Event'
  | 'Source';

export type RelationshipType =
  | 'contains'
  | 'mentions'
  | 'discusses'
  | 'written_by'
  | 'sent_to'
  | 'shares_entity';
```

2. Update `KnowledgeGraph.tsx` colors:
```typescript
const ENTITY_COLORS: Record<EntityType, string> = {
  Document: '#6366f1',  // indigo
  Email: '#f59e0b',     // amber
  Person: '#10b981',    // emerald
  Organization: '#8b5cf6', // violet
  Concept: '#ec4899',   // pink
  Event: '#06b6d4',     // cyan
  Source: '#64748b',    // slate
};
```

3. Update node rendering to show icons per type

**Success Criteria**:
- [ ] Different colors for different entity types
- [ ] Relationship labels visible on hover
- [ ] Legend showing entity types

---

## Execution Order

1. **TASK 3** first (quick win - better labels)
2. **TASK 1** (critical - entity extraction)
3. **TASK 4** (critical - cross-source links)
4. **TASK 2** (critical - search)
5. **TASK 5** (NER for better extraction)
6. **TASK 6** (temporal queries)
7. **TASK 7** (visualization polish)

## Testing

After each task, run:
```bash
# Build and test desktop app
cd desktop && npm run build
cd src-tauri && cargo build

# Run the app
npm run tauri dev
```

Verify:
1. Graph shows new entity types
2. Search returns results
3. Cross-source connections visible

## Important Notes

- **Read existing code first** - don't duplicate functionality
- **Use TodoWrite tool** - track all subtasks
- **Test incrementally** - verify each change works before moving on
- **Keep privacy-first** - no PII in logs
- **Performance matters** - entity extraction should be <100ms per doc

## Reference Files

Read these files to understand the codebase:
- `CLAUDE.md` - Project overview
- `docs/phase-1/PHASE_1_COMPLETION_PLAN.md` - Detailed gap analysis
- `docs/phase-1/overview.md` - Phase 1 features list
- `desktop/src-tauri/src/commands/graph.rs` - Current graph implementation
- `src/futurnal/pipeline/triples.py` - Existing extraction code

## END OF PROMPT

---


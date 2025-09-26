Summary: Define wikilink and tag relationship extraction to build PKG note adjacency.

# 04 · Link Graph Construction

## Purpose
Translate Obsidian `[[Wiki Links]]`, `[[Page#Heading]]`, and tags into semantic relationships for the PKG with provenance.

## Parsing Rules
- Wikilinks without extension resolve to `.md` notes within vault
- Heading anchors create `references_heading` relationships
- Tags create `has_tag` relationships; namespace tags via `obsidian:tag/<tag>`
- Backlinks inferred; deduplicate cyclic references

## Graph Writes
- Node: `Note { vault_id, note_id, title, path }`
- Relationships: `links_to`, `references_heading`, `has_tag`
- Include `source_path`, `offset`, and `checksum` on edges for auditability

## Acceptance Criteria
- Cycles handled gracefully; no duplicate edges
- Renames or moves update edges without losing history

## Test Plan
- Unit: wikilink tokenizer state machine
- Integration: vault fixtures with cycles, missing references, and anchors



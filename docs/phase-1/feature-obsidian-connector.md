Summary: Details the Obsidian vault connector that grounds the Ghost in user's evolving thought patterns and knowledge network.

# Feature · Obsidian Vault Connector

## Goal
**Ground the Ghost in the user's evolving thought patterns and knowledge network** by learning from their Obsidian vault. Obsidian vaults represent the user's intellectual journey—notes are thought traces, wikilinks reveal conceptual relationships, and temporal edits show how understanding develops over time. This connector enables the Ghost to understand the user's unique mental model by learning from bidirectional links, personal categorization patterns (tags/frontmatter), and the evolution of ideas captured in markdown. The Ghost develops deep familiarity with how the user thinks, organizes knowledge, and connects concepts—essential foundation for personalized intelligence.

## Success Criteria

### Ghost Grounding Quality
- Ghost develops understanding of user's conceptual relationships revealed through wikilinks and bidirectional note connections.
- Personal categorization patterns (tags, frontmatter) inform Ghost's understanding of how user organizes knowledge.
- Temporal evolution of notes enables Ghost to learn how user's thinking develops and refines over time.
- Ghost recognizes user's unique mental model through note structure, cross-references, and idea clustering patterns.

### Technical Completeness
- Users register Obsidian vault path and Futurnal learns from markdown content, frontmatter, and embedded assets.
- Wikilinks and tags convert into semantic relationships within Ghost's experiential memory (PKG).
- Incremental sync captures thought evolution (note edits) within minutes while preserving temporal history.
- Vault processing respects Obsidian-specific ignores (templates, trash) and custom `.futurnalignore`.

## Functional Scope
- Vault registration UI/CLI with workspace metadata (vault name, icon).
- Markdown parser that normalizes frontmatter, callouts, and dataview fields.
- Link resolver translating `[[Page]]` into cross-note relationships.
- Attachment handling for embedded images and PDFs.
- Sync engine that handles rename/move events gracefully.

## Non-Functional Guarantees
- Offline-first; no remote dependencies or plugin requirements.
- Preserve Obsidian formatting fidelity for round-trip editing.
- Minimal extra disk usage; deduplicate assets via hashing.
- Logging scoped to local audit with vault-specific redaction of sensitive note titles.

## Dependencies
- Local connector primitives from [feature-local-files-connector](feature-local-files-connector.md).
- Parsing patterns aligned with [system-architecture.md](../architecture/system-architecture.md) ingestion guidance.
- PKG schema for note, tag, and backlink relationships.

## Implementation Guide

### Thought Pattern Learning Pipeline
1. **Vault Descriptor:** Extend source registry with Obsidian-specific metadata to identify user's thought repository.
2. **Markdown Normalizer:** Leverage `markdown-it` with custom plugins to extract conceptual structure from user's notes; map frontmatter keys to Ghost's understanding of user's categorization patterns.
3. **Conceptual Graph Construction:** Build note adjacency lists representing user's mental model—wikilinks reveal how user connects concepts. Use automata-based parsing to capture bidirectional relationship patterns.
4. **Asset Pipeline:** Process embedded attachments (images, PDFs) to understand visual/document context within user's thought traces.
5. **Temporal Sync Strategy:** Track note evolution over time to enable Ghost to learn how user's understanding develops. Use rename detection to maintain thought continuity across refactoring.
6. **Learning Quality Gate:** Generate reports showing Ghost's understanding of vault structure: note clusters, concept connectivity, temporal patterns, and missing conceptual links.

## Testing Strategy
- **Unit Tests:** Wikilink parser, tag extraction, frontmatter mapping.
- **Integration Tests:** Fixture vault ingestion with intentional link cycles, embeds, and renames.
- **Regression Tests:** Ensure vault ingestion does not corrupt generic local connector behaviors.
- **User Acceptance:** Pilot with live vault, validating graph relationships via visualization module.

## Code Review Checklist
- Obsidian-specific rules isolated from generic filesystem connector logic.
- Link and tag relationships captured and deduplicated in PKG inserts.
- Attachments handled securely without leaking outside vault sandbox.
- Tests cover complex markdown constructs (dataview, callouts).
- Logging avoids writing note content while still aiding debugging.

## Documentation & Follow-up
- Update user onboarding docs with Obsidian setup steps.
- Capture mapping of Obsidian features to PKG schema in shared reference.
- Feed lessons into future note-centric connectors (Logseq, Notion export).



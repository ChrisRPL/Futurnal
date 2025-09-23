Summary: Details the Obsidian vault connector feature with scope, testing, and review guidance.

# Feature · Obsidian Vault Connector

## Goal
Support one-click ingestion of Obsidian vaults while preserving markdown links, tags, and graph relationships so vault context translates cleanly into the PKG.

## Success Criteria
- Users register an Obsidian vault path and Futurnal mirrors markdown content, frontmatter, and embedded assets.
- Wikilinks and tags convert into semantic relationships stored alongside provenance metadata.
- Incremental updates capture note edits within minutes without duplicating nodes.
- Vault scraping respects Obsidian-specific ignores (templates, trash) and custom `.futurnalignore`.

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
1. **Vault Descriptor:** Extend source registry with Obsidian-specific metadata (vault UID, base path).
2. **Markdown Normalizer:** Leverage `markdown-it` with custom plugins to capture callouts and tables; map frontmatter keys to graph properties.
3. **Link Graph Construction:** Build note adjacency lists using automata-style state machines for wikilink parsing per @Web Automata approach.
4. **Asset Pipeline:** Reuse modOpt-inspired modular pipelines for attachments (image, PDF) to route through Unstructured.io as needed.
5. **Sync Strategy:** Use file rename detection to update PKG node IDs while maintaining history.
6. **Quality Gate:** Generate Obsidian-specific ingestion reports summarizing ingested notes, missing references, parse warnings.

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



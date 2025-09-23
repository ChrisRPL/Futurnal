Summary: Lists functional and non-functional requirements for Futurnal across data ingestion, privacy, insights, and user experience.

# Futurnal System Requirements

## Functional Requirements

### Data Ingestion & Processing
- Support connectors for local directories, Obsidian vaults, IMAP email, and GitHub repositories in Phase 1.
- Normalize 60+ document formats using Unstructured.io and store provenance metadata.
- Execute automated entity and relationship extraction from ingested text to produce PKG triples.
- Maintain synchronization so that updates, deletions, and additions propagate to the PKG within minutes.

### Personal Knowledge Graph (PKG)
- Store triples in an embedded graph database with version tracking and temporal metadata.
- Generate vector embeddings for nodes and associated documents and maintain a hybrid index (graph + vector).
- Provide search APIs that accept natural language queries and return combined semantic and graph results.
- Offer an interactive visualization that renders selected subgraphs with filtering and tagging capabilities.

### Insight Engine
- Periodically analyze the PKG to detect statistically significant correlations and thematic clusters.
- Produce human-readable "Emergent Insights" statements with supporting evidence.
- Allow users to bookmark, dismiss, or request deeper analysis for each insight.
- Enable conversational follow-up that queries the PKG and checks for confounding factors before presenting causal guidance.

### Aspirational Self Feature
- Let users define goals, habits, and values as structured entries.
- Link Aspirational Self nodes to associated data (e.g., documents, events, insights) for context-aware recommendations.
- Highlight misalignments between user-stated aspirations and observed behavior patterns.

### User Interaction & Interface
- Provide search, insight dashboard, and conversational modes accessible via desktop application.
- Support dark-mode-first UI with keyboard-centric navigation and customizable panes.
- Enable notifications for new insights and causal exploration prompts.

## Non-Functional Requirements

### Privacy & Security
- Process raw content locally; restrict cloud escalation to structured queries with per-request consent.
- Log all model interactions with timestamps and data source references for user auditing.
- Encrypt local storage for the PKG and embeddings using OS-native keychains.

### Performance
- Deliver sub-second responses for standard search queries on reference hardware (Apple M-series, RTX 3060-class GPU).
- Complete initial ingestion of 10 GB mixed corpus within 24 hours using parallel processing.
- Run insight detection jobs without blocking user interaction; schedule during idle periods.

### Extensibility
- Expose modular connectors for additional data sources (calendars, RSS, cloud drives) via plugin API.
- Design PKG schema to accommodate new node and edge types without migrations.
- Support feature toggles so phases (Archivist, Analyst, Guide) can be activated progressively.

### Reliability
- Include automated backup routines for PKG, embeddings, and configuration.
- Gracefully handle missing or malformed documents with retry and quarantine workflows.
- Provide telemetry (local by default) for system health metrics and optional anonymized reporting.

## Roadmap Alignment
- Map each requirement to development phases: Phase 1 focuses on ingestion, PKG, and search; Phase 2 introduces insight detection and dashboards; Phase 3 activates causal dialogue and Aspirational Self integration.


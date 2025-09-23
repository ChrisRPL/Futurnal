Summary: Outlines the Phase 1 Archivist plan with feature-level specs, testing, and review checkpoints.

# Phase 1 · Archivist Planning Overview

## Phase Objective
Phase 1 focuses on proving Futurnal's Archivist value proposition: private ingestion, dynamic PKG creation, and hybrid search that surpasses traditional note and PKG tools. This plan translates the [system requirements](../requirements/system-requirements.md) and [system architecture](../architecture/system-architecture.md) into testable feature workstreams, sequenced for incremental validation.

## Planning Principles
- Deliver user-facing value at the end of each feature iteration while keeping ingestion → PKG → search wiring intact.
- Maintain privacy-first constraints, escalating to cloud models only through explicit consent flows.
- Keep graph and vector indices synchronized; features that mutate one must update the other.
- Anchor implementation choices in state-of-the-art, on-device friendly techniques validated via @Web research.

## Feature Breakdown
Each feature below has a dedicated playbook in `docs/phase-1/feature-*.md` describing scope, testing, and review. Sequencing assumes parallel tracks where dependencies allow.

| Order | Feature | Outcome | Primary Dependencies |
| --- | --- | --- | --- |
| 1 | [Local Files Connector](feature-local-files-connector.md) | Baseline ingestion loop for filesystem sources | Unstructured.io parsing, ingestion orchestrator |
| 2 | [Obsidian Vault Connector](feature-obsidian-connector.md) | Vault-aware ingestion with link preservation | Local connector primitives |
| 3 | [IMAP Email Connector](feature-imap-connector.md) | Email ingestion with selective sync | Scheduler, metadata normalizer |
| 4 | [GitHub Repository Connector](feature-github-connector.md) | Repository mirror with code/document segregation | Scheduler, normalizer |
| 5 | [Ingestion Orchestrator](feature-ingestion-orchestrator.md) | Deterministic job routing & retries | Connectors |
| 6 | [Document Normalization Pipeline](feature-document-normalization.md) | Unified chunk + provenance store | Unstructured.io, orchestrator |
| 7 | [Entity & Relationship Extraction](feature-entity-relationship-extraction.md) | Triple generation powering PKG | Normalized documents, local LLM |
| 8 | [PKG Graph Storage Layer](feature-pkg-graph-storage.md) | Embedded Neo4j with versioned triples | Extraction pipeline |
| 9 | [Vector Embedding Service](feature-vector-embedding-service.md) | Embedding pipeline + store sync | Normalized documents |
| 10 | [Hybrid Search API](feature-hybrid-search-api.md) | Graph + vector retrieval contract | Graph layer, embedding service |
| 11 | [Search Desktop Shell](feature-search-desktop-shell.md) | Electron/Tauri interface for queries | Hybrid search API |
| 12 | [Graph Visualization Module](feature-graph-visualization.md) | Interactive PKG explorer | Graph layer |
| 13 | [Privacy & Audit Logging](feature-privacy-audit-logging.md) | Consent and audit trail foundations | Orchestrator, connectors |
| 14 | [Cloud Escalation Consent Flow](feature-cloud-escalation-consent.md) | Structured off-device reasoning requests | Privacy & audit logging |
| 15 | [Performance Telemetry Baseline](feature-performance-telemetry.md) | Profiling + health metrics | Orchestrator, search API |

## Testing & Review Cadence
- **Feature Completion Definition:** Implementation merged, automated tests passing, exploratory QA notes captured, code review checklist signed off, and documentation updated.
- **Code Review Rhythm:** Minimum two-person review per feature; reviewers use feature-specific checklist plus shared [planning prompts](../prompts/phase-1-archivist.md).
- **Regression Sweeps:** Nightly ingestion and search smoke tests across connectors to guard against schema drift.

## Documentation Responsibilities
- Update this overview when feature scope shifts or new dependencies emerge.
- Cross-link future Analyst/Guide docs once Phase 1 dependencies stabilize.
- Reflect learnings back into [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) and roadmap milestones.



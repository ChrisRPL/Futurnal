Summary: Describes the search desktop shell feature with scope, testing, and review criteria.

# Feature · Search Desktop Shell

## Goal
Deliver a cross-platform desktop interface (Electron or Tauri) enabling Archivist users to issue hybrid search queries, view results with provenance, and manage connectors while adhering to privacy-first principles.

## Success Criteria
- Search UI executes queries via hybrid search API with sub-second feedback.
- Results view displays snippets, provenance metadata, and quick actions (open source, save, share).
- Connector management panel allows enabling/disabling sources and viewing ingestion status.
- App respects dark-mode-first aesthetic with keyboard-centric navigation.
- Telemetry and logging integrated per privacy guidelines.

## Functional Scope
- Application shell scaffolding with window management, preferences storage, and update channel.
- Search input with command palette behavior, supporting natural language and filters.
- Result presentation components with expandable context and tags.
- Connector dashboard showing status, last sync, error alerts.
- Settings for privacy controls, cloud consent toggles, telemetry opt-in.

## Non-Functional Guarantees
- Offline usage with graceful handling of unavailable services.
- Secure IPC between frontend and backend, avoiding Node integration vulnerabilities.
- Responsive layout optimized for 13"–16" displays; accessible keyboard shortcuts.
- Update mechanism optional; manual updates supported to respect air-gapped setups.

## Dependencies
- Hybrid search API ([feature-hybrid-search-api](feature-hybrid-search-api.md)).
- Ingestion orchestrator status endpoints.
- Privacy/audit logging for user actions.

## Implementation Guide
1. **Framework Selection:** Evaluate Electron vs. Tauri referencing @Web research on performance and security; document decision.
2. **Design System:** Establish reusable components consistent with brand pillars; align with `docs/product-vision.md` aesthetic guidance.
3. **State Management:** Use reactive store (Redux/ Zustand) with automata-inspired state machines for query lifecycle to ensure predictable transitions.
4. **IPC Layer:** Define secure channels for search requests, connector commands, and telemetry; disable unnecessary Node APIs.
5. **Keyboard UX:** Implement command palette with fuzzy matching; provide shortcuts for connector toggles and settings.
6. **Privacy Controls:** Surface consent status, data export options, and audit log viewer within settings.

## Testing Strategy
- **Unit Tests:** Component rendering, state reducers, IPC handlers.
- **Integration Tests:** End-to-end search flows using headless renderer (Playwright/Spectron).
- **Accessibility Tests:** Keyboard navigation, contrast checks, screen reader compatibility.
- **Performance Tests:** Measure initial load time, query response rendering, memory footprint.

## Code Review Checklist
- IPC channels hardened; no direct DOM access from backend.
- UI aligns with dark-mode aesthetic and accessibility guidelines.
- Connector status accurately reflects orchestrator state with error handling.
- Tests cover primary user journeys and regression scenarios.
- Logging adheres to privacy rules; sensitive data not persisted.

## Documentation & Follow-up
- Update user guide with onboarding walkthrough and shortcut reference.
- Capture usability feedback during beta to inform Analyst phase dashboards.
- Coordinate with design team for future visualization integration.



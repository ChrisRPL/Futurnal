Summary: Plan for quarantine management tooling and operator workflows.

# Task · Quarantine Operations

## Objectives
- Provide clear operator visibility into quarantined files, reasons, and remediation steps.
- Allow retry, dismissal, or export of quarantined entries via CLI/desktop UI.
- Ensure telemetry and audit records reference quarantine metrics for monitoring.

## Deliverables
- CLI commands (and hooks for future desktop UI) to list, inspect, retry, and remove quarantine entries.
- Documentation explaining quarantine reasons, remediation workflow, and privacy considerations.
- Telemetry counters exposing active quarantine items; audit logs capturing retry actions.
- Automated tests covering quarantine commands and ensuring data integrity on retries/deletions.

## Work Breakdown
1. **Quarantine Schema Enhancements**
   - Standardize JSON structure (path, source, reason, detail, timestamp, last_retry).
   - Add optional field to track retry attempts and operator notes.
2. **CLI Tooling**
   - `futurnal local-sources quarantine list`: summarize quarantined items with filters (reason, source).
   - `quarantine inspect <id>`: show detailed payload, including stack traces.
   - `quarantine retry <id>`: requeue ingestion for the file; update audit trail.
   - `quarantine dismiss <id>`: archive entry with rationale.
3. **Ingestion Hooks**
   - On retry, ensure state-store, PKG/vector, and parsed cache are consistent before reprocessing.
   - Prevent infinite retry loops; cap attempts and surface guidance.
4. **Telemetry & Audit**
   - Emit metrics: total quarantined, retries attempted/succeeded, oldest outstanding item age.
   - Log operator actions (retry/dismiss) with user identifier (placeholder until auth is available).
5. **Documentation & UX**
   - Update feature docs with step-by-step remediation playbook.
   - Provide troubleshooting FAQ (common reasons, recommended fixes).

## Open Questions
- Should retries trigger immediate ingestion or place a job in the scheduler queue?
- How do we handle quarantined items referencing deleted files (e.g., operator cleaned up manually)?
- Where should dismissed entries live (separate archive folder vs. same directory with status flag)?



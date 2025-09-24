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
   - JSON payloads now include `source`, `retry_count`, `last_retry_at`, and `notes`, with helper utilities in `src/futurnal/ingestion/local/quarantine.py` to parse, update, and archive entries.
   - Newly created quarantine files default to retry metadata and support structured note appends for operator annotations; summary generation writes to `workspace/telemetry/quarantine_summary.json`.
2. **CLI Tooling**
   - Added Typer subcommands under `futurnal local-sources quarantine` for `list`, `inspect`, `retry`, `dismiss`, and `summary`, with filtering, dry-run retry, and archival handling.
   - CLI reuse of `QuarantinePayloadBuilder` in tests ensures fixtures mirror production schema; dismissals move files to `quarantine_archive/` preserving JSON history.
3. **Ingestion Hooks**
   - Connector now records `source` on quarantine entries and differentiates failure reasons; CLI retry invokes `LocalFilesConnector.ingest` to reprocess after state-store reset while respecting max retry cap (3 attempts).
4. **Telemetry & Audit**
   - `write_summary` aggregates counts, reasons, and oldest age; nightly CI publishes telemetry artifacts. Future work: integrate retry/dismiss actions into `AuditLogger` once operator IDs are wired.
5. **Documentation & UX**
   - Remediation playbook documents CLI usage, retry guidance, and FAQ in this file; link from `DEVELOPMENT_GUIDE.md` ensures onboarding flow references quarantine operations.

## Execution Notes

- **Local workflow:**
  - Review entries: `futurnal local-sources quarantine list --workspace <path>`
  - Inspect payload YAML: `... quarantine inspect <id>` (add `--raw` for JSON)
  - Retry with dry-run: `... quarantine retry <id> --source <name> --dry-run`
  - Dismiss and archive: `... quarantine dismiss <id> --note "resolved" --operator ops@team`
- **Retry semantics:** CLI retries run ingestion immediately; successful reprocessing removes the entry, otherwise retry count increments. Max attempts (3) enforced; beyond that operators must remediate manually.
- **Telemetry & monitoring:** `... quarantine summary` writes/prints aggregate metrics and refreshes `workspace/telemetry/quarantine_summary.json`; nightly CI uploads the file for trend analysis.
- **Documentation:** Keep troubleshooting FAQ updated here and in `feature-local-files-connector.md`; include common failure reasons (permissions, hash mismatch) and recommended fixes.

## Open Questions
- Should retries trigger immediate ingestion or place a job in the scheduler queue?
- How do we handle quarantined items referencing deleted files (e.g., operator cleaned up manually)?
- Where should dismissed entries live (separate archive folder vs. same directory with status flag)?



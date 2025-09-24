Summary: Plan for privacy safeguards and audit logging enhancements in the Local Files Connector.

# Task · Privacy & Audit Hardening

## Objectives
- Strengthen privacy guarantees and audit trails to align with Futurnal’s privacy-by-design principles.
- Ensure all external processing pathways include consent prompts and redaction policies.
- Provide clear, immutable records of ingestion activity without leaking sensitive file contents.

## Deliverables
- Enhanced audit log format capturing source, action, status, hash, redacted path, operator actions, and optional consent tokens.
- Consent flow scaffolding for any optional cloud escalations or external parsers (even if not yet enabled).
- Redaction policy ensuring telemetry/audit never stores raw filenames for sensitive directories while maintaining traceability.
- Documentation describing privacy controls, retention, and operator responsibilities.

## Work Breakdown
1. **Audit Schema Upgrade**
   - Introduce structured fields: `job_id`, `source`, `status`, `sha256`, `redacted_path`, `attempt`, `operator_action`.
   - Store audit entries in append-only log with rotation and integrity checks (e.g., hash chain).
2. **Redaction & Masking**
   - Define rules for masking paths (e.g., keep filename extension, hash directory names).
   - Apply consistent redaction across telemetry, audit, and CLI outputs.
3. **Consent Handling**
   - Add configuration toggles for external parser escalation; prompt before first use per source.
   - Log consent decisions with timestamps and context.
4. **Privacy Review Hooks**
   - Integrate with future privacy modules (link to `feature-privacy-audit-logging.md`) to ensure audit events feed central review.
   - Prepare for encryption at rest using OS keychain (shared with configuration/security task).
5. **Documentation & Training**
   - Update feature docs and onboarding materials covering privacy controls, logging expectations, and review processes.
   - Provide operator checklist before enabling external integrations.

## Implementation Summary
- Audit events now include structured fields (`job_id`, `source`, `action`, `status`, `sha256`, `redacted_path`, `path_hash`, `operator_action`, `metadata`) with a chained hash for tamper evidence and automatic file rotation/retention (30 days by default).
- Redaction policy hashes every directory segment while preserving filename extensions; CLI telemetry, audit views, and quarantine tooling print masked paths with stable hashes.
- Consent registry persists per-source scope decisions, exposes CLI grant/revoke/status commands, and blocks ingestion escalations without active approval; all decisions feed the audit log with optional token hashes.
- Privacy review hooks mirror audit entries into a dedicated workspace channel for downstream inspection, priming future encryption at rest integration via manifest metadata.
- Operator documentation references CLI commands (`local-sources audit --verify`, `local-sources consent`, `local-sources quarantine`) to inspect logs, manage consent, and validate privacy posture.

## Open Questions
- What retention policy should we enforce for audit logs? Daily rotation? Configurable TTL?
- Do we need to support per-source redaction rules (e.g., allow plaintext for non-sensitive directories)?
- How will user identities be captured for audit entries prior to authentication implementation?



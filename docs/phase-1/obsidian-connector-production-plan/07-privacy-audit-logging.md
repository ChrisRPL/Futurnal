Summary: Define privacy controls and local audit logging for Obsidian vault ingestion.

# 07 · Privacy & Audit Logging

## Purpose
Ensure vault ingestion adheres to privacy-by-design: local-only processing by default, explicit consent for any cloud escalation, and redaction-aware audit trails.

## Controls
- Redact note titles/paths matching configured patterns
- Disable content logging; only log checksums and record counts
- Per-vault consent flags for any cloud model usage

## Audit Events
- Vault registered/updated/removed
- Note scanned/updated/renamed; edges added/removed
- Asset extracted; text extraction invoked

## Acceptance Criteria
- No raw content written to logs
- Redactions applied consistently across all audit entries

## Test Plan
- Unit: redaction filters and formatting
- Integration: end-to-end ingestion with audit inspection



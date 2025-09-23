Summary: Defines the IMAP email ingestion feature with scope, testing, and review guardrails.

# Feature · IMAP Email Connector

## Goal
Ingest user-selected IMAP mailboxes into Futurnal, transforming emails into structured documents with secure credential handling and incremental sync that respects privacy and consent boundaries.

## Success Criteria
- OAuth or app-password based authentication stored via OS keychain with explicit consent prompts.
- Selective folder sync (Inbox, Sent, labels) with configurable look-back window.
- Emails normalized into document chunks with preserved headers, threading, and attachment links.
- Delta sync processes new, updated, and deleted messages within 5 minutes of detection.
- <0.5% parsing failure rate across test mailboxes.

## Functional Scope
- Account onboarding flow with consent logging.
- IDLE/NOOP polling scheduler leveraging ingestion orchestrator.
- MIME parsing with attachment extraction and classification.
- Thread reconstruction using `Message-ID` and `References` headers for PKG edge creation.
- Quarantine pipeline for oversized or encrypted attachments.

## Non-Functional Guarantees
- Encrypted credential storage with automatic token refresh.
- Network operations constrained to secure TLS connections; retry using exponential backoff (modOpt-inspired modular retries).
- Offline resilience: queue sync tasks for later execution when offline.
- Audit logging for every mailbox action without storing email bodies in logs.

## Dependencies
- [feature-ingestion-orchestrator](feature-ingestion-orchestrator.md) for job scheduling.
- MIME parsing libs aligned with on-device footprint targets.
- PKG schema support for email threads and participants per [system-architecture.md](../architecture/system-architecture.md).

## Implementation Guide
1. **Credential Manager:** Integrate native keychain APIs and build consent modal referencing privacy policy.
2. **Folder Selector:** Allow users to choose folders/labels; map to internal source configuration.
3. **Sync Engine:** Implement incremental fetch using IMAP UIDs and `MODSEQ`; maintain checkpoint store locally.
4. **Parser Pipeline:** Employ streaming MIME parser; convert HTML to markdown using well-maintained libraries; generate semantic triples linking sender, recipients, topics.
5. **Attachment Handling:** Route supported attachments (PDF, DOCX) through normalization pipeline; flag unsupported types for user review.
6. **State Machine:** Use automata-based programming principles for connection lifecycle (connect → sync → idle) to improve reliability under varying network conditions.

## Testing Strategy
- **Unit Tests:** Credential encryption/decryption, folder filtering, MIME edge cases.
- **Integration Tests:** Mock IMAP server scenarios (new mail, updates, deletions, connection drops).
- **Load Tests:** High-volume mailbox ingestion to validate throughput and memory footprint.
- **Security Tests:** Validate TLS enforcement, credential storage, and consent flow logging.

## Code Review Checklist
- Credentials never logged; memory cleared after use.
- Delta sync handles edge cases (moved emails, duplicates).
- Attachments processed or quarantined deterministically.
- Tests simulate network instability and recover gracefully.
- Documentation updated with mailbox configuration instructions.

## Documentation & Follow-up
- Add troubleshooting guide for common IMAP providers.
- Capture metrics on sync latency and error rates in telemetry baseline.
- Share insights with future connectors (Gmail API, Outlook Graph) for Phase 2+ planning.



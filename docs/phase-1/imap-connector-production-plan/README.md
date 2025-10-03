Summary: Roadmap and task breakdown to bring the IMAP Email Connector to production readiness for Ghost grounding.

# IMAP Email Connector · Production Plan

This folder tracks the work required to ship Feature 3 (IMAP Email Connector) with production-quality stability, observability, and privacy compliance—enabling the Ghost to learn from user's email communications, relationship dynamics, and conversational patterns. Each task ensures the connector captures not just email content, but the conversational threads, participant relationships, and temporal communication evolution embedded in IMAP mailboxes. Task documents define scope, acceptance criteria, test plans, and operational guidance aligned to the experiential learning architecture in [system-architecture.md](../../architecture/system-architecture.md).

## Task Index
- [01-mailbox-descriptor.md](01-mailbox-descriptor.md)
- [02-credential-manager.md](02-credential-manager.md)
- [03-connection-manager.md](03-connection-manager.md)
- [04-email-parser-normalizer.md](04-email-parser-normalizer.md)
- [05-thread-reconstruction.md](05-thread-reconstruction.md)
- [06-incremental-sync-strategy.md](06-incremental-sync-strategy.md)
- [07-attachment-pipeline.md](07-attachment-pipeline.md)
- [08-privacy-audit-integration.md](08-privacy-audit-integration.md)
- [09-connector-orchestrator-integration.md](09-connector-orchestrator-integration.md)
- [10-quality-gates-testing.md](10-quality-gates-testing.md)

## Technical Foundation

### IMAP Library Selection
**IMAPClient** ([/mjs/imapclient](https://github.com/mjs/imapclient)) - Trust Score 9.3
- Pythonic, complete IMAP4rev1 implementation
- Native OAuth2 support (XOAUTH2)
- IDLE support with connection lifecycle management
- Context manager for automatic cleanup
- Extensive code examples and production usage

### Credential Storage
**Python keyring** - OS-native secure credential storage
- macOS: Keychain
- Windows: Credential Manager
- Linux: Secret Service API
- Automatic token refresh for OAuth2

### Email Processing
- **Python email.parser**: RFC822/MIME parsing
- **Unstructured.io**: Email body text extraction and normalization
- **Message threading**: RFC 2822 Message-ID/References/In-Reply-To

### Connection Management
- TLS-only enforcement (no plaintext IMAP)
- Exponential backoff retry (modOpt-inspired)
- Connection pooling for multi-folder sync
- IDLE with 10-minute renewal (per IMAPClient best practices)

## Architectural Patterns

Following established patterns from Obsidian/Local Files connectors:

1. **Descriptor + Registry Pattern**
   - `ImapMailboxDescriptor`: Persistent mailbox configuration
   - `MailboxRegistry`: File-based registry under `~/.futurnal/sources/imap/`

2. **Privacy-First Design**
   - Explicit consent via `ConsentRegistry`
   - No email bodies in logs or audit trails
   - Participant email redaction
   - Header-only audit events

3. **Incremental Learning**
   - IMAP UID-based state tracking
   - MODSEQ for efficient delta sync
   - 5-minute detection window for new/updated/deleted messages

4. **Quarantine & Resilience**
   - Failed email processing → quarantine with retry policy
   - Connection failures → exponential backoff
   - Offline mode → queue sync tasks for later execution

5. **Orchestrator Integration**
   - Register with `IngestionOrchestrator`
   - IDLE/NOOP polling via APScheduler
   - ElementSink for processed emails
   - StateStore for sync checkpoints

## AI Learning Focus

Transform email communications into experiential memory:

- **Relationship Dynamics**: Extract participant patterns, communication frequency, response times
- **Conversational Context**: Reconstruct threads via Message-ID graphs for conversation understanding
- **Communication Patterns**: Analyze temporal patterns, subject evolution, sentiment shifts
- **Content Understanding**: Feed email bodies through Unstructured.io → semantic triples → PKG

## Usage

- Update these plans as tasks progress; each file captures scope, deliverables, and open questions.
- Cross-link implementation PRs and test evidence directly inside the relevant markdown files.
- When a task reaches completion, summarize learnings and move any follow-up work to the appropriate phase-2 documents.



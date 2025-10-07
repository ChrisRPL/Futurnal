Summary: Roadmap and task breakdown to bring the GitHub Repository Connector to production readiness for Ghost grounding.

# GitHub Repository Connector · Production Plan

This folder tracks the work required to ship Feature 4 (GitHub Repository Connector) with production-quality stability, observability, and privacy compliance—enabling the Ghost to learn from user's code repositories, documentation, and development patterns. Each task ensures the connector captures not just source code, but the architectural decisions, documentation evolution, and collaboration dynamics embedded in GitHub repositories. Task documents define scope, acceptance criteria, test plans, and operational guidance aligned to the experiential learning architecture in [system-architecture.md](../../architecture/system-architecture.md).

## Task Index
- [01-repository-descriptor.md](01-repository-descriptor.md)
- [02-oauth-authentication-manager.md](02-oauth-authentication-manager.md)
- [03-api-client-manager.md](03-api-client-manager.md)
- [04-repository-sync-strategy.md](04-repository-sync-strategy.md)
- [05-content-classifier.md](05-content-classifier.md)
- [06-issue-pr-normalizer.md](06-issue-pr-normalizer.md)
- [07-incremental-sync-engine.md](07-incremental-sync-engine.md)
- [08-webhook-integration.md](08-webhook-integration.md)
- [09-privacy-consent-integration.md](09-privacy-consent-integration.md)
- [10-orchestrator-integration.md](10-orchestrator-integration.md)
- [11-quality-gates-testing.md](11-quality-gates-testing.md)

## Technical Foundation

### GitHub API Client
**GitHubKit** ([/yanyongyu/githubkit](https://github.com/yanyongyu/githubkit)) - Trust Score 9.0
- Modern, all-batteries-included, fully typed GitHub API SDK
- Native sync and async support
- OAuth Device Flow support (perfect for CLI)
- Built-in token refresh and authentication strategies
- Comprehensive REST and GraphQL API coverage

### Git Operations
**GitPython** - Python Git library for clone mode operations
- Full-fidelity repository cloning
- Shallow clone and sparse checkout support
- Commit history traversal
- Branch and tag management

### Credential Storage
**Python keyring** - OS-native secure credential storage
- macOS: Keychain
- Windows: Credential Manager
- Linux: Secret Service API
- Automatic token refresh for OAuth2

### API Strategy
- **GraphQL-first**: Efficient batched queries, lower rate limit consumption
- **REST fallback**: Operations not available in GraphQL
- **Conditional requests**: ETag support for cached responses
- **Webhooks preferred**: Real-time updates over polling

### Rate Limiting (2025)
- **REST API**: 5000 requests/hour (authenticated), 60/hour (unauthenticated)
- **GraphQL API**: 5000 points/hour (query cost varies by complexity)
- **Concurrent limit**: Maximum 100 concurrent requests (shared across REST + GraphQL)
- **Strategy**: Adaptive backoff, request queuing, webhook preference over polling

## Architectural Patterns

Following established patterns from Obsidian/Local Files/IMAP connectors:

1. **Descriptor + Registry Pattern**
   - `GitHubRepositoryDescriptor`: Persistent repository configuration
   - `RepositoryRegistry`: File-based registry under `~/.futurnal/sources/github/`

2. **Privacy-First Design**
   - Explicit consent via `ConsentRegistry`
   - No source code in logs or audit trails
   - File path redaction for sensitive repositories
   - Pattern-based file exclusions (secrets, credentials)

3. **Incremental Learning**
   - Commit SHA-based state tracking
   - Delta sync with force-push detection
   - 5-minute detection window for new commits/issues/PRs

4. **Quarantine & Resilience**
   - Failed file processing → quarantine with retry policy
   - API failures → exponential backoff
   - Offline mode → queue sync tasks for later execution

5. **Orchestrator Integration**
   - Register with `IngestionOrchestrator`
   - Webhook/polling via APScheduler
   - ElementSink for processed content
   - StateStore for sync checkpoints

## Sync Modes

### GraphQL API Mode (Lightweight)
**Best for:**
- Selective file access
- Metadata-only ingestion
- Issue/PR tracking
- Limited disk space
- Multiple repository monitoring

**Characteristics:**
- Lower disk footprint
- Rate-limit efficient for targeted queries
- No full repository history
- Online-only operation

### Git Clone Mode (Full Fidelity)
**Best for:**
- Complete repository analysis
- Offline code exploration
- Full commit history needed
- Advanced git operations

**Characteristics:**
- Complete repository on disk
- Offline operation after initial clone
- Higher disk usage
- Full git history and branches

## AI Learning Focus

Transform GitHub repositories into experiential memory:

- **Code Understanding**: Programming patterns, architectural decisions, API usage, library choices
- **Documentation Evolution**: README quality, wiki updates, inline comments, documentation debt
- **Development Patterns**: Commit frequency, code churn hotspots, refactoring trends, technical debt
- **Collaboration Dynamics**: Issue discussions, PR review patterns, contributor interactions, decision-making processes
- **Project Trajectory**: Feature timelines, dependency evolution, version history, release cadence
- **Knowledge Gaps**: Undocumented features, missing tests, TODO markers, technical debt annotations

## Content Classification

Files are classified into categories for appropriate processing:

- **Source Code**: `.py`, `.js`, `.java`, `.go`, etc. → CodeBERT embeddings → PKG with code triples
- **Documentation**: `README.md`, `.md`, wiki pages → Standard embeddings → Documentation nodes
- **Configuration**: `pyproject.toml`, `package.json`, `.yml` → Configuration analysis
- **Issues/PRs**: Metadata extraction → Conversation threads → Participant graphs
- **Commits**: Author, message, timestamp → Temporal activity patterns

## Security Considerations

- **Token Security**: OAuth tokens in OS keychain, never in files or logs
- **Webhook Security**: HMAC-SHA256 signature verification
- **Secret Detection**: Exclude `.env`, `credentials.json`, private key files
- **Rate Limit Compliance**: Adaptive backoff, no aggressive scraping
- **Audit Transparency**: All repository access logged (without code content)

## Usage

- Update these plans as tasks progress; each file captures scope, deliverables, and open questions.
- Cross-link implementation PRs and test evidence directly inside the relevant markdown files.
- When a task reaches completion, summarize learnings and move any follow-up work to the appropriate phase-2 documents.



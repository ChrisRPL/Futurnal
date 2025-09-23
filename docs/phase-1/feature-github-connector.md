Summary: Captures the GitHub repository connector plan with scope, testing, and review standards.

# Feature · GitHub Repository Connector

## Goal
Mirror selected GitHub repositories into Futurnal, transforming source files, docs, and issues into structured knowledge while honoring privacy and rate limits.

## Success Criteria
- Users authenticate via GitHub OAuth with least-privilege scopes.
- Repository sync pulls branches, README/wiki docs, and issues/pull requests metadata.
- Code vs. documentation content routed appropriately for embeddings vs. graph enrichment.
- Incremental sync (webhooks or polling) catches new commits/issues within 5 minutes.
- Rate limiting respected; retries scheduled without data loss.

## Functional Scope
- Repository selector with private repo support and consent log.
- Git clone or GraphQL API ingestion (configurable for on-device resource usage).
- Content classifier to separate code, docs, discussions.
- Issue/PR metadata normalization into PKG entities (authors, labels, milestones).
- Diff-aware sync to avoid reprocessing unchanged files.

## Non-Functional Guarantees
- Offline resilience by queuing sync jobs.
- Secure storage of tokens using OS keychain.
- Minimal disk duplication by using sparse checkout or shallow clones where possible.
- Logging sanitized to avoid sensitive repo names when user opts out.

## Dependencies
- [feature-ingestion-orchestrator](feature-ingestion-orchestrator.md) for scheduling.
- Graph schema accommodating code/document nodes per [system-architecture.md](../architecture/system-architecture.md).
- Vector embedding service tuned for code snippets (state-of-the-art models from @Web research).

## Implementation Guide
1. **Auth Flow:** Integrate GitHub OAuth device code flow; capture consent artifacts.
2. **Repo Sync Strategy:** Offer dual mode—Git clone for full fidelity, GraphQL API for lightweight setups; use modOpt-inspired modular pipeline to switch.
3. **Content Classification:** Apply file path heuristics plus model-based classification; adopt @Web findings on repository embedding via state-of-the-art code embedding techniques (e.g., CodeBERT variants).
4. **Entity Extraction:** Generate triples linking commits to authors, files, and issues; align with PKG schema.
5. **Delta Sync:** Track latest commit SHA per branch; process diffs to update affected documents only.
6. **Rate Limiting:** Implement adaptive backoff and caching to avoid hitting GitHub API limits.

## Testing Strategy
- **Unit Tests:** Token storage, content classification, diff detection.
- **Integration Tests:** Repository sync against fixture repos with docs, code, issues; simulate force-push and PR merges.
- **Load Tests:** High-commit repositories to validate throughput and caching.
- **Security Tests:** Validate token scopes, secure storage, and audit logging.

## Code Review Checklist
- Auth flow handles revocation and token refresh gracefully.
- Delta sync avoids redundant processing and handles rebases.
- Classification pipeline correctly routes code vs. docs vs. discussions.
- Tests cover private repo access and rate limit scenarios.
- Documentation updated with GitHub setup steps and scopes required.

## Documentation & Follow-up
- Publish how-to guide for connecting repos and managing sync.
- Track metrics on sync latency, processed files, and API usage.
- Apply lessons to future git-based connectors (GitLab, Bitbucket).



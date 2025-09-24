Summary: Plan for configuration management, security, and resilience of the Local Files Connector.

# Task · Configuration & Security

## Objectives
- Provide secure, manageable configuration for storage backends (Neo4j, Chroma) and connector settings.
- Implement backup, health checks, and encryption to meet reliability and privacy requirements.
- Ensure workspace initialization handles migrations and recovery gracefully.

## Deliverables
- Configuration layer (e.g., `.futurnal/config.toml` or environment-based) covering:
  - Neo4j URI, credentials, encryption flags
  - Chroma persistence path, authentication (if applicable)
  - Storage encryption keys managed via OS keychain
- Backup/restore tooling for state store, PKG, and vector data.
- Health check commands verifying connectivity, schema readiness, and disk space.
- Documentation for secure setup, including secrets handling and recovery procedures.

## Work Breakdown
1. **Config Refactor**
   - Introduce structured settings module with validation; integrate with CLI for initialization.
   - Support environment overrides for CI and production deployments.
2. **Secrets Management**
   - Store credentials in OS keychain (macOS Keychain) or encrypted file; provide rotation workflow.
   - Ensure audit logs never expose secrets; mask values in CLI output.
3. **Backup & Restore**
   - Implement scripts/commands to snapshot state store, PKG graph, and vector index.
   - Verify restores rebuild state correctly and maintain provenance accuracy.
4. **Health Checks**
   - Add CLI command to run diagnostics (database connectivity, WAL status, disk usage, telemetry folder health).
   - Integrate with telemetry to record health check results.
5. **Workspace Migrations**
   - Provide versioning for local datastore schema; run migrations on startup.
   - Document upgrade/downgrade steps and failure recovery.

## Open Questions
- Which encryption mechanism best balances ease of use and security for local deployments?
- How frequently should backups run, and where should they be stored?
- Do we need remote backup/export support in Phase 1 or is local-only acceptable?



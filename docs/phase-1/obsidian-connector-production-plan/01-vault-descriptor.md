Summary: Specify vault metadata model, storage, and registration flow for Obsidian vaults.

# 01 · Vault Descriptor

## Purpose
Define the Obsidian-specific source descriptor used by the ingestion orchestrator and local connector: vault UID, base path, ignores, provenance, and privacy flags. Ensures the connector can register, enumerate, and monitor a vault safely.

## Scope
- Descriptor schema with required/optional fields
- Persistent storage location and format
- Validation rules and conflict handling
- Integration points with `src/futurnal/ingestion/local/*` and orchestrator scheduler

## Requirements Alignment
- Privacy by default; all processing is local unless explicitly escalated
- Feed Unstructured.io parsing and emit semantic triples into PKG and vectors
- Keep graph and vector indices synchronized during updates

## Data Model
- id: deterministic ULID/UUIDv7 derived from path + creation timestamp
- name: human label for the vault
- base_path: absolute path to vault root
- icon: optional emoji or path to an icon file
- ignore_rules: merge of Obsidian defaults (templates, .trash) and `.futurnalignore`
- redact_title_patterns: optional patterns to mask sensitive note titles in logs
- created_at, updated_at
- provenance: OS user, machine id hash, tool version

## Storage
- Stored in workspace registry under `~/.futurnal/sources/obsidian/<vault_id>.json`
- Transaction-safe updates guarded by file locks
- Backed up by `workspace/backup.py` routines

## CLI Registration Flow
1. `futurnal sources add obsidian --path <vault>`
2. Validate path exists and contains `.obsidian/`
3. Compute defaults, load optional `.futurnalignore`
4. Persist descriptor JSON and print vault_id

## Validation Rules
- Base path must be absolute and readable
- Reject duplicates: same normalized path → same vault id
- Warn on network mounts or high-latency volumes

## Acceptance Criteria
- Creating, reading, and listing descriptors works via CLI and programmatic API
- Ignores honored by scanner; audit shows masked titles when configured
- Backups include descriptor files; restore round-trips cleanly

## Test Plan
- Unit: schema validation, id determinism, ignore merging
- Integration: CLI add/list/remove; scanner respects ignores
- Regression: duplicate registration idempotent; restore from backup

## Open Questions
- Should we auto-detect vault rename and remap vault_id while preserving history?



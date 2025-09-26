Summary: Define CLI/UX flows for registering, listing, and removing Obsidian vaults.

# 02 · CLI/UX Vault Registration

## Purpose
Provide clear, scriptable commands and UX guidance to manage Obsidian vaults as Futurnal sources.

## Commands
- `futurnal sources add obsidian --path <vault_path> [--name <name>] [--icon <emoji|path>] [--redact-title <pattern>...]`
- `futurnal sources list --type obsidian`
- `futurnal sources remove <vault_id>`
- `futurnal sources inspect <vault_id>`

## Behaviour
- Adds an Obsidian descriptor per [01-vault-descriptor.md](01-vault-descriptor.md)
- Validates `.obsidian/` presence; warns for empty vaults
- Supports multiple `--redact-title` patterns for privacy
- Prints machine-readable JSON when `--json` is provided

## UX Notes
- Show next steps: how to start an ingestion run and view reports
- On remove, ask for `--yes` or require confirmation token in non-interactive contexts

## Acceptance Criteria
- All commands function with non-interactive flags
- Helpful error messages; zero leaking of file paths when `--redact` configured

## Test Plan
- Unit: argument parsing, formatting
- Integration: end-to-end add → list → inspect → remove



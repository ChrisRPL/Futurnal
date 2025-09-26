Summary: Define incremental sync, rename/move handling, and event-driven updates for Obsidian vaults.

# 06 · Sync Strategy

## Purpose
Provide a robust, low-latency sync engine for Obsidian vaults that captures edits, renames, and moves without duplicating nodes.

## Event Sources
- Initial full scan via local connector scanner
- File system events using `watchdog` or `watchfiles`

## Change Detection
- Content checksum and mtime comparisons
- Path rename detection to update `note_id` mapping while preserving history

## Concurrency & Backpressure
- Batch updates; apply queue priorities to notes vs assets
- Retry with exponential backoff for transient errors

## Acceptance Criteria
- Edits reflected within minutes under normal load
- Renames retain edges/history; no duplicate nodes

## Test Plan
- Integration: rename/move cascades; burst edits; large vaults



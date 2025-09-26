Summary: Define Obsidian-specific ingestion quality gates and reporting artifacts.

# 08 · Quality Gate & Ingestion Report

## Purpose
Provide a deterministic quality gate and human-readable report for Obsidian ingestion, surfacing missing references, parse warnings, and asset coverage.

## Report Contents
- Totals: notes scanned/ingested/updated, assets deduped, edges created
- Warnings: missing references, unresolved embeds, parse failures
- Privacy: redactions applied, consent statuses
- Performance: throughput, queue latencies, watchdog events

## Output
- JSON artifact for CI and machine consumption
- Markdown summary for operators

## Acceptance Criteria
- Report generated per run and accessible via CLI
- Exit codes reflect gate pass/fail; blocks release when failing in CI

## Test Plan
- Integration: run on fixture vaults with intentional errors and verify gate behavior



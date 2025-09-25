Summary: How to operate the Local Files scheduler via CLI, including schedule management, priorities, manual runs, and queue inspection.

# Scheduler Operations Playbook

## Overview
Futurnal ships scheduler controls inside the `futurnal local-sources` CLI. Operators can:
- review source schedules and compute next runs
- switch between cron and interval cadence, or disable automation for manual-only feeds
- adjust ingestion priorities for bursty sources
- pause and resume ingestion while respecting audit requirements
- trigger manual ingestions (with safeguards for paused feeds)
- inspect the job queue, including retry attempts and priority metadata

All commands accept `--config-path` and `--workspace-path` overrides; defaults point to `~/.futurnal/sources.json` and `~/.futurnal/workspace` respectively.

## Schedule Management
- List schedules with optional JSON output:
  - `futurnal local-sources schedule list`
  - `futurnal local-sources schedule list --json`
- Update a source to cron cadence: `futurnal local-sources schedule update docs --cron "*/15 * * * *"`
- Switch to interval cadence (seconds): `futurnal local-sources schedule update docs --interval 900`
- Force manual-only mode: `futurnal local-sources schedule update docs --manual`
- Remove an existing schedule (alias for manual-only): `futurnal local-sources schedule remove docs`

Validators block invalid cron strings and missing intervals; manual updates recalculate next-run previews.

## Priority and Manual Overrides
- Set priority (`low|normal|high`): `futurnal local-sources priority docs --level high`
- Trigger manual ingestion: `futurnal local-sources run docs`
  - Attempting to run a paused source exits with code `1`; add `--force` to override intentionally.

## Pause / Resume and Audit
- Pause automation: `futurnal local-sources pause docs --operator ops`
- Resume automation: `futurnal local-sources resume docs --operator ops`

Both commands emit audit events (`action=scheduler`, `status=paused|resumed`) so the privacy log maintains tamper-evident operator records.

## Queue Inspection
- Snapshot queue status: `futurnal local-sources queue status`
- JSON snapshot for monitoring pipelines: `futurnal local-sources queue status --json`
- Filter by status (pending, running, succeeded, failed) or limit entries: `futurnal local-sources queue status --status pending --limit 10`

Each entry reports job id, source, priority, attempts, timestamps, and trigger (schedule/interval/manual). Retry-triggered jobs retain their attempt counter.

## Troubleshooting
- Paused sources still register filesystem watchers but ignore events until resumed; use `--force` when a one-off run is required.
- When changing intervals, confirm `interval_seconds` <= 86400 (per config validation).
- Queue JSON output can feed dashboards; combine with `jq` or monitoring agents to track backlog growth and retry storms.

## Related References
- [`05-scheduler-ux-enhancements.md`](05-scheduler-ux-enhancements.md)
- [`../README.md`](README.md)
- [`../../../../DEVELOPMENT_GUIDE.md`](../../DEVELOPMENT_GUIDE.md)


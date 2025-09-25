Summary: Plan for improving scheduling UX and operator controls in the Local Files Connector.

# Task · Scheduler UX Enhancements

## Objectives
- Provide intuitive controls for managing ingestion schedules, priorities, and manual runs.
- Surface job queue status, retry counts, and pause/resume functionality for operators.
- Ensure scheduling remains deterministic and transparent across CLI and future desktop UI.

## Deliverables
- CLI commands for listing schedules, updating cron/interval settings, changing priorities, and triggering manual jobs.
- Pause/resume support per source with persisted state to avoid unintended ingestion.
- Queue visibility tools (e.g., pending jobs, retry counts, next run times) integrated into CLI or dashboard.
- Documentation describing scheduling workflows and best practices. Refer to
  [`scheduler-operations.md`](scheduler-operations.md) for operator guidance.

## Work Breakdown
1. **Schedule Management CLI**
   - `schedule list`, `schedule update <source> --cron ...`, `schedule remove`, etc.
   - Validation helpers to prevent invalid cron expressions; preview next run.
2. **Priority & Manual Controls**
   - Allow changing job priority per source (high/normal/low).
   - Provide `run --source <name>` to enqueue manual jobs with optional immediate execution.
3. **Pause/Resume & Disable**
   - Add state flags persisted in config or orchestrator store; paused sources should skip both cron and watcher events.
   - Emit audit entries when operators pause/resume ingestion.
4. **Queue Inspection**
   - CLI command to show pending/running jobs, attempts, and timestamps.
   - Optional JSON output for integration with monitoring.
5. **UX Documentation**
   - Update docs/DEVELOPMENT_GUIDE with scheduling tutorials and troubleshooting tips.
   - Provide examples for common scenarios (daily sync, manual-only, burst sync windows).

## Open Questions
- Should scheduler configuration live in source config files or a separate orchestration config?
- Do we need to enforce global limits on manual runs (e.g., max N per hour) to avoid resource contention?
- How will desktop UI integrate with these controls (e.g., using the same CLI commands behind the scenes)?



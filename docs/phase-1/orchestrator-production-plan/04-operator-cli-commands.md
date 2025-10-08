Summary: Implement comprehensive CLI commands for orchestrator management, job inspection, and operational control.

# 04 · Operator CLI Commands

## Purpose
Provide operators with comprehensive CLI commands for managing the ingestion orchestrator, inspecting job queues, controlling sources, and troubleshooting issues. Enables operators to maintain visibility into the Ghost's experiential learning pipeline and intervene when necessary.

## Scope
- Orchestrator status dashboard (queue depth, active workers, system resources)
- Job management commands (list, show, cancel, retry)
- Source control commands (list, pause, resume, trigger)
- Quarantine management (list, show, retry, purge) - extends Task 01
- Health checks and diagnostics
- Telemetry inspection
- Configuration display

## Requirements Alignment
- **Operator Console**: "Provide CLI and minimal UI for monitoring, pause/resume, and manual retries" (implementation guide)
- **Observability**: Expose orchestrator state for operational decisions
- **Control**: Enable operators to manage orchestrator behavior
- **Troubleshooting**: Surface diagnostic information for problem resolution

## CLI Command Structure

### Primary Command Group
```bash
futurnal orchestrator [SUBCOMMAND]

Subcommands:
  status        Display orchestrator status dashboard
  jobs          Manage ingestion jobs
  sources       Manage ingestion sources
  quarantine    Manage quarantined jobs
  health        Run health checks
  telemetry     View telemetry metrics
  config        Display configuration
```

## Command Specifications

### 01 - Orchestrator Status
```bash
futurnal orchestrator status [OPTIONS]

Display real-time orchestrator status dashboard.

Options:
  --refresh INTEGER    Auto-refresh interval in seconds (0 for no refresh)
  --format [table|json|yaml]  Output format (default: table)

Output:
  - Queue depth (pending/running/completed/failed)
  - Active workers (current/max)
  - System resources (CPU/memory/disk)
  - Recent job throughput
  - Source status (active/paused)
  - Quarantine count

Example:
  futurnal orchestrator status
  futurnal orchestrator status --refresh 5  # Auto-refresh every 5s
```

Example Output:
```
╭─────────────── Orchestrator Status ───────────────╮
│ Queue                                              │
│   Pending:     12 jobs                             │
│   Running:      3 jobs                             │
│   Completed:  487 jobs (last 24h)                  │
│   Failed:      15 jobs (last 24h)                  │
│   Quarantined:  4 jobs                             │
│                                                    │
│ Workers                                            │
│   Active:   3 / 8 workers                          │
│   Utilization: 37.5%                               │
│                                                    │
│ System Resources                                   │
│   CPU:     42% (8 cores)                           │
│   Memory:  6.2 GB / 16 GB (38%)                    │
│   Disk:    120 GB free                             │
│                                                    │
│ Throughput (last hour)                             │
│   Files:    324 files                              │
│   Data:     1.2 GB                                 │
│   Rate:     5.8 MB/s                               │
│                                                    │
│ Sources                                            │
│   ✓ notes (active, last run: 5m ago)              │
│   ✓ vault (active, last run: 15m ago)             │
│   ⏸ email (paused)                                 │
│   ✓ github (active, last run: 1h ago)             │
╰────────────────────────────────────────────────────╯
```

### 02 - Jobs List
```bash
futurnal orchestrator jobs list [OPTIONS]

List ingestion jobs with filtering.

Options:
  --status [pending|running|succeeded|failed]  Filter by status
  --job-type [local_files|obsidian_vault|imap_mailbox|github_repository]  Filter by type
  --source TEXT         Filter by source name
  --limit INTEGER       Limit results (default: 20)
  --since TEXT          Show jobs since date (e.g., "2024-01-01", "24h")
  --format [table|json|yaml]  Output format (default: table)

Example:
  futurnal orchestrator jobs list
  futurnal orchestrator jobs list --status running
  futurnal orchestrator jobs list --job-type imap_mailbox --limit 50
  futurnal orchestrator jobs list --since 24h --status failed
```

Example Output:
```
┌────────────┬──────────────┬─────────────────┬────────┬───────────────┬──────────┐
│ Job ID     │ Type         │ Source          │ Status │ Started       │ Duration │
├────────────┼──────────────┼─────────────────┼────────┼───────────────┼──────────┤
│ abc123...  │ local_files  │ notes           │ run    │ 2m ago        │ 00:02:15 │
│ def456...  │ obsidian     │ vault           │ run    │ 5m ago        │ 00:05:42 │
│ ghi789...  │ imap         │ email           │ pend   │ scheduled 1h  │ -        │
│ jkl012...  │ github       │ myrepo          │ succ   │ 1h ago        │ 00:08:23 │
│ mno345...  │ local_files  │ docs            │ fail   │ 2h ago        │ 00:00:45 │
└────────────┴──────────────┴─────────────────┴────────┴───────────────┴──────────┘
```

### 03 - Jobs Show
```bash
futurnal orchestrator jobs show JOB_ID [OPTIONS]

Display detailed information about a specific job.

Options:
  --format [table|json|yaml]  Output format (default: table)

Example:
  futurnal orchestrator jobs show abc123-def456-789
```

Example Output:
```
╭─────────────── Job Details ───────────────╮
│ Job ID:       abc123-def456-789            │
│ Type:         local_files                  │
│ Source:       notes                        │
│ Status:       running                      │
│ Priority:     normal                       │
│                                            │
│ Timing                                     │
│   Created:    2024-01-15 10:30:00 UTC      │
│   Started:    2024-01-15 10:32:15 UTC      │
│   Duration:   00:02:45                     │
│                                            │
│ Progress                                   │
│   Files:      142 / ~200                   │
│   Bytes:      45 MB / ~65 MB               │
│   Throughput: 6.2 MB/s                     │
│                                            │
│ Payload                                    │
│   root_path: /Users/user/notes             │
│   trigger:   schedule                      │
│   attempts:  1                             │
╰────────────────────────────────────────────╯
```

### 04 - Jobs Cancel
```bash
futurnal orchestrator jobs cancel JOB_ID [OPTIONS]

Cancel a pending or running job.

Options:
  --reason TEXT    Cancellation reason (for audit log)
  --force          Force cancel running job (may leave inconsistent state)

Example:
  futurnal orchestrator jobs cancel abc123 --reason "Duplicate job"
```

### 05 - Sources List
```bash
futurnal orchestrator sources list [OPTIONS]

List registered ingestion sources.

Options:
  --status [active|paused]  Filter by status
  --format [table|json|yaml]  Output format (default: table)

Example:
  futurnal orchestrator sources list
```

Example Output:
```
┌───────────┬──────────────┬────────────────┬────────┬──────────────┬─────────────┐
│ Source    │ Type         │ Schedule       │ Status │ Last Run     │ Next Run    │
├───────────┼──────────────┼────────────────┼────────┼──────────────┼─────────────┤
│ notes     │ local_files  │ 0 * * * *      │ active │ 15m ago      │ in 45m      │
│ vault     │ obsidian     │ */30 * * * *   │ active │ 5m ago       │ in 25m      │
│ email     │ imap         │ @manual        │ paused │ never        │ -           │
│ myrepo    │ github       │ 0 */4 * * *    │ active │ 2h ago       │ in 2h       │
└───────────┴──────────────┴────────────────┴────────┴──────────────┴─────────────┘
```

### 06 - Sources Pause/Resume
```bash
futurnal orchestrator sources pause SOURCE_NAME [OPTIONS]
futurnal orchestrator sources resume SOURCE_NAME [OPTIONS]

Pause or resume a source's scheduled jobs.

Options:
  --reason TEXT    Reason for pause/resume (for audit log)

Example:
  futurnal orchestrator sources pause email --reason "Fixing credentials"
  futurnal orchestrator sources resume email
```

### 07 - Sources Trigger
```bash
futurnal orchestrator sources trigger SOURCE_NAME [OPTIONS]

Manually trigger a source ingestion job.

Options:
  --force          Trigger even if source is paused
  --priority [low|normal|high]  Job priority (default: high for manual)
  --wait           Wait for job to complete

Example:
  futurnal orchestrator sources trigger notes
  futurnal orchestrator sources trigger vault --wait
```

### 08 - Health Check
```bash
futurnal orchestrator health [OPTIONS]

Run comprehensive health checks on orchestrator subsystems.

Options:
  --format [table|json|yaml]  Output format (default: table)
  --verbose        Show detailed health information

Example:
  futurnal orchestrator health
```

Example Output:
```
╭─────────────── Health Checks ───────────────╮
│ Component      │ Status  │ Detail           │
├────────────────┼─────────┼──────────────────┤
│ Queue          │ ✓ ok    │ SQLite healthy   │
│ Scheduler      │ ✓ ok    │ 4 jobs scheduled │
│ Workers        │ ✓ ok    │ 3 / 8 active     │
│ Disk Space     │ ✓ ok    │ 120 GB free      │
│ State Store    │ ✓ ok    │ Reachable        │
│ Neo4j          │ ✓ ok    │ Connected        │
│ ChromaDB       │ ✓ ok    │ Collection ready │
│ IMAP Sync      │ ⚠ warn  │ 1 mailbox errors │
│ GitHub Sync    │ ✓ ok    │ 2 repos synced   │
╰─────────────────────────────────────────────╯
```

### 09 - Telemetry View
```bash
futurnal orchestrator telemetry [METRIC] [OPTIONS]

View telemetry metrics.

Metrics:
  summary       Overall telemetry summary
  throughput    Throughput metrics over time
  failures      Failure statistics
  by-connector  Per-connector metrics

Options:
  --since TEXT         Time range (e.g., "24h", "7d", "2024-01-01")
  --format [table|json|yaml]  Output format (default: table)

Example:
  futurnal orchestrator telemetry summary
  futurnal orchestrator telemetry throughput --since 24h
  futurnal orchestrator telemetry by-connector
```

### 10 - Config Display
```bash
futurnal orchestrator config [OPTIONS]

Display orchestrator configuration.

Options:
  --section TEXT       Show specific section (workers, retry, resources)
  --format [table|json|yaml]  Output format (default: yaml)

Example:
  futurnal orchestrator config
  futurnal orchestrator config --section retry
```

## Implementation Structure

### CLI Command Handler
```python
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(name="orchestrator")
console = Console()

@app.command(name="status")
def status_command(
    refresh: int = typer.Option(0, help="Auto-refresh interval in seconds"),
    format: str = typer.Option("table", help="Output format"),
) -> None:
    """Display orchestrator status dashboard."""
    if refresh > 0:
        import time
        while True:
            console.clear()
            _display_status(format)
            time.sleep(refresh)
    else:
        _display_status(format)

def _display_status(format: str) -> None:
    """Render status dashboard."""
    from futurnal.orchestrator.status import collect_status_report

    status = collect_status_report()

    if format == "json":
        console.print_json(data=status)
    elif format == "yaml":
        import yaml
        console.print(yaml.dump(status))
    else:
        _render_status_table(status)

def _render_status_table(status: Dict[str, Any]) -> None:
    """Render status as formatted table."""
    # Queue section
    queue_table = Table.grid(padding=(0, 2))
    queue_table.add_row("Pending:", f"{status['queue']['pending']} jobs")
    queue_table.add_row("Running:", f"{status['queue']['running']} jobs")
    queue_table.add_row("Completed:", f"{status['queue']['completed']} jobs (last 24h)")
    queue_table.add_row("Failed:", f"{status['queue']['failed']} jobs (last 24h)")
    queue_table.add_row("Quarantined:", f"{status['queue']['quarantined']} jobs")

    # Workers section
    workers_table = Table.grid(padding=(0, 2))
    workers_table.add_row(
        "Active:",
        f"{status['workers']['active']} / {status['workers']['max']} workers"
    )
    workers_table.add_row("Utilization:", f"{status['workers']['utilization']:.1f}%")

    # System resources section
    system_table = Table.grid(padding=(0, 2))
    system_table.add_row(
        "CPU:",
        f"{status['system']['cpu_percent']:.0f}% ({status['system']['cpu_count']} cores)"
    )
    system_table.add_row(
        "Memory:",
        f"{status['system']['memory_used_gb']:.1f} GB / {status['system']['memory_total_gb']:.1f} GB "
        f"({status['system']['memory_percent']:.0f}%)"
    )
    system_table.add_row("Disk:", f"{status['system']['disk_free_gb']:.0f} GB free")

    # Assemble into panel
    panel = Panel(
        Table.grid(
            queue_table,
            "",
            workers_table,
            "",
            system_table,
        ),
        title="Orchestrator Status",
        border_style="blue",
    )
    console.print(panel)
```

### Status Report Collection
```python
# src/futurnal/orchestrator/status.py

def collect_status_report(
    *,
    orchestrator: Optional[IngestionOrchestrator] = None,
    job_queue: Optional[JobQueue] = None,
) -> Dict[str, Any]:
    """Collect comprehensive orchestrator status."""
    import psutil

    # Queue metrics
    queue_metrics = {
        "pending": job_queue.pending_count() if job_queue else 0,
        "running": job_queue.running_count() if job_queue else 0,
        "completed_24h": job_queue.completed_count(since=datetime.utcnow() - timedelta(days=1)) if job_queue else 0,
        "failed_24h": job_queue.failed_count(since=datetime.utcnow() - timedelta(days=1)) if job_queue else 0,
        "quarantined": orchestrator._quarantine.count() if orchestrator else 0,
    }

    # Worker metrics
    worker_metrics = {
        "active": orchestrator._active_jobs if orchestrator else 0,
        "max": orchestrator._configured_workers if orchestrator else 0,
        "utilization": (orchestrator._active_jobs / orchestrator._configured_workers * 100)
            if orchestrator and orchestrator._configured_workers > 0 else 0.0,
    }

    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    system_metrics = {
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "memory_used_gb": memory.used / (1024 ** 3),
        "memory_total_gb": memory.total / (1024 ** 3),
        "memory_percent": memory.percent,
        "disk_free_gb": disk.free / (1024 ** 3),
    }

    # Source metrics
    sources = []
    if orchestrator:
        for name, registration in orchestrator._sources.items():
            sources.append({
                "name": name,
                "type": registration.source.__class__.__name__,
                "schedule": registration.schedule,
                "status": "paused" if registration.paused else "active",
            })

    return {
        "queue": queue_metrics,
        "workers": worker_metrics,
        "system": system_metrics,
        "sources": sources,
    }
```

## Acceptance Criteria

- ✅ `orchestrator status` displays comprehensive dashboard
- ✅ `orchestrator jobs list` shows filterable job list
- ✅ `orchestrator jobs show` displays detailed job information
- ✅ `orchestrator jobs cancel` cancels pending/running jobs
- ✅ `orchestrator sources list` shows all registered sources
- ✅ `orchestrator sources pause/resume` controls source scheduling
- ✅ `orchestrator sources trigger` manually triggers source ingestion
- ✅ `orchestrator health` runs comprehensive health checks
- ✅ `orchestrator telemetry` views metrics and statistics
- ✅ `orchestrator config` displays configuration
- ✅ All commands support JSON/YAML output formats
- ✅ Audit events logged for all operator actions
- ✅ Status dashboard supports auto-refresh mode
- ✅ Rich formatting with tables, panels, and colors
- ✅ Error messages provide actionable guidance

## Test Plan

### Unit Tests
- `test_status_report_collection.py`: Status data aggregation
- `test_job_list_filtering.py`: Job filtering logic
- `test_output_formatting.py`: JSON/YAML/table rendering

### Integration Tests
- `test_cli_orchestrator_status.py`: End-to-end status command
- `test_cli_jobs_management.py`: Job list/show/cancel commands
- `test_cli_sources_control.py`: Source pause/resume/trigger
- `test_cli_audit_logging.py`: Audit events for CLI actions

### User Experience Tests
- `test_table_formatting.py`: Table rendering with various data sizes
- `test_auto_refresh.py`: Status auto-refresh functionality
- `test_error_messages.py`: User-friendly error handling

## Implementation Notes

### Rich Library Integration
Use Rich for enhanced CLI formatting:
- Tables with borders and colors
- Panels for grouped information
- Progress bars for long operations
- Syntax highlighting for JSON/YAML

### Auto-Refresh Implementation
```python
def auto_refresh_status(interval_seconds: int) -> None:
    """Auto-refresh status dashboard."""
    import time
    from rich.live import Live

    with Live(auto_refresh=True, refresh_per_second=1) as live:
        while True:
            status_panel = _render_status_panel()
            live.update(status_panel)
            time.sleep(interval_seconds)
```

### Audit Logging for CLI Actions
```python
def _log_cli_action(action: str, details: Dict[str, Any]) -> None:
    """Log operator CLI action to audit trail."""
    audit_logger.record(
        AuditEvent(
            job_id=f"cli_{action}_{uuid.uuid4().hex[:8]}",
            source="operator_cli",
            action=action,
            status="executed",
            timestamp=datetime.utcnow(),
            metadata=details,
            operator_action="cli",
        )
    )
```

## Open Questions

- Should we support shell completion (bash/zsh) for commands?
- How to handle very large job lists (pagination vs. streaming)?
- Should status dashboard be a TUI (text-based UI) instead of CLI?
- What level of detail in job show command (logs, stack traces)?
- Should we support batch operations (cancel multiple jobs)?
- How to handle concurrent CLI operations on orchestrator?
- Should we provide a web-based dashboard in addition to CLI?

## Dependencies

- Typer for CLI framework
- Rich for enhanced terminal formatting
- Existing IngestionOrchestrator for state access
- JobQueue for job management
- AuditLogger for CLI action logging
- QuarantineStore (Task 01) for quarantine commands



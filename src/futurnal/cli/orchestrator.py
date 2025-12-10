"""CLI commands for orchestrator management and monitoring."""

from __future__ import annotations

import asyncio
import json
import signal
import time
import uuid
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

from futurnal.configuration.settings import Settings, load_settings
from futurnal.orchestrator.scheduler import IngestionOrchestrator, SourceRegistration
from futurnal.orchestrator.queue import JobQueue, JobStatus
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.quarantine_cli import quarantine_app
from futurnal.orchestrator.config_cli import orchestrator_config_app
from futurnal.orchestrator.status import collect_status_report
from futurnal.orchestrator.source_control import PausedSourcesRegistry
from futurnal.orchestrator.health import collect_health_report
from futurnal.orchestrator.daemon import OrchestratorDaemon, AlreadyRunningError, NotRunningError, DaemonError
from futurnal.orchestrator.telemetry_analysis import TelemetryAnalyzer
from futurnal.orchestrator.db_utils import DatabaseManager, BackupError, RestoreError, IntegrityError, DatabaseError
from futurnal.privacy.audit import AuditEvent, AuditLogger

console = Console()
orchestrator_app = typer.Typer(help="Orchestrator management commands")

# Add quarantine management as a sub-command
orchestrator_app.add_typer(quarantine_app, name="quarantine")

# Add orchestrator configuration management
orchestrator_app.add_typer(orchestrator_config_app, name="config")


def _get_workspace_path(workspace: Optional[Path] = None) -> Path:
    """Get workspace path from option or settings."""
    if workspace:
        return Path(workspace).expanduser()
    try:
        settings = load_settings()
        return settings.workspace.workspace_path
    except Exception:
        return Path.home() / ".futurnal"


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _format_bytes(bytes_value: float) -> str:
    """Format bytes in human-readable format."""
    if bytes_value < 1024:
        return f"{bytes_value:.0f} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_value / (1024 ** 3):.2f} GB"


def _format_timestamp(timestamp_str: Optional[str]) -> str:
    """Format ISO timestamp for display."""
    if not timestamp_str:
        return "never"
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        now = datetime.utcnow()
        delta = now - dt

        if delta.total_seconds() < 60:
            return f"{int(delta.total_seconds())}s ago"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        elif delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)}h ago"
        else:
            return f"{int(delta.total_seconds() / 86400)}d ago"
    except (ValueError, AttributeError):
        return timestamp_str


def _log_cli_action(workspace: Path, action: str, details: dict) -> None:
    """Log operator CLI action to audit trail."""
    audit_logger = AuditLogger(workspace / "audit")
    audit_logger.record(
        AuditEvent(
            job_id=f"cli_{action}_{uuid.uuid4().hex[:8]}",
            source="operator_cli",
            action=action,
            status="executed",
            timestamp=datetime.utcnow(),
            metadata=details,
        )
    )


def _load_job_queue(workspace: Path) -> JobQueue:
    """Initialize job queue database."""
    queue_dir = workspace / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    return JobQueue(queue_dir / "jobs.db")


def _load_state_store(workspace: Path) -> StateStore:
    """Initialize state store database."""
    state_dir = workspace / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return StateStore(state_dir / "state.db")


@orchestrator_app.command("status")
def status_command(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    refresh: int = typer.Option(0, "--refresh", help="Auto-refresh interval in seconds (0 for no refresh)"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """Display orchestrator status dashboard with queue, workers, and system metrics."""
    workspace_path = _get_workspace_path(workspace)

    if refresh > 0:
        # Auto-refresh mode
        with Live(auto_refresh=False, console=console) as live:
            try:
                while True:
                    status_display = _render_status(workspace_path, format_output)
                    live.update(status_display)
                    time.sleep(refresh)
            except KeyboardInterrupt:
                console.print("\n[yellow]Auto-refresh stopped[/yellow]")
    else:
        # Single display
        status_display = _render_status(workspace_path, format_output)
        if isinstance(status_display, str):
            console.print(status_display)
        else:
            console.print(status_display)


def _render_status(workspace_path: Path, format_output: str) -> str | Panel:
    """Render status report in requested format."""
    status = collect_status_report(workspace_path=workspace_path)

    if format_output == "json":
        return json.dumps(status, indent=2)
    elif format_output == "yaml":
        return yaml.dump(status, default_flow_style=False)
    else:
        # Rich table format
        return _render_status_panel(status)


def _render_status_panel(status: dict) -> Panel:
    """Render status as formatted Rich panel."""
    queue = status["queue"]
    workers = status["workers"]
    system = status["system"]
    throughput = status["throughput"]
    sources = status["sources"]

    # Build content sections
    sections = []

    # Queue section
    sections.append("[bold cyan]Queue[/bold cyan]")
    sections.append(f"  Pending:      {queue['pending']:>5} jobs")
    sections.append(f"  Running:      {queue['running']:>5} jobs")
    sections.append(f"  Completed:    {queue['completed_24h']:>5} jobs (last 24h)")
    sections.append(f"  Failed:       {queue['failed_24h']:>5} jobs (last 24h)")
    sections.append(f"  Quarantined:  {queue['quarantined']:>5} jobs")
    sections.append("")

    # Workers section
    sections.append("[bold cyan]Workers[/bold cyan]")
    sections.append(f"  Active:       {workers['active']:>3} / {workers['max']} workers")
    sections.append(f"  Utilization:  {workers['utilization']:>5.1f}%")
    sections.append("")

    # System resources section
    sections.append("[bold cyan]System Resources[/bold cyan]")
    sections.append(f"  CPU:          {system['cpu_percent']:>5.1f}% ({system['cpu_count']} cores)")
    sections.append(f"  Memory:       {system['memory_used_gb']:>5.1f} GB / {system['memory_total_gb']:.1f} GB ({system['memory_percent']:.0f}%)")
    sections.append(f"  Disk:         {system['disk_free_gb']:>5.0f} GB free ({100 - system['disk_percent']:.0f}% available)")
    sections.append("")

    # Throughput section
    sections.append("[bold cyan]Throughput (last hour)[/bold cyan]")
    sections.append(f"  Files:        {throughput['files_last_hour']:>5}")
    sections.append(f"  Data:         {_format_bytes(throughput['bytes_last_hour'])}")
    sections.append(f"  Rate:         {_format_bytes(throughput['rate_bytes_per_second'])}/s")
    sections.append("")

    # Sources section
    sections.append("[bold cyan]Sources[/bold cyan]")
    if sources:
        for source in sources[:10]:  # Limit to first 10
            status_icon = "✓" if source["status"] == "active" else "⏸"
            sections.append(f"  {status_icon} {source['name'][:40]:<40} ({source['type']})")
        if len(sources) > 10:
            sections.append(f"  ... and {len(sources) - 10} more")
    else:
        sections.append("  No sources configured")

    content = "\n".join(sections)
    return Panel(content, title="[bold]Orchestrator Status[/bold]", border_style="blue")


@orchestrator_app.command("jobs")
def jobs_command() -> None:
    """Job management commands (use subcommands: list, show, cancel)."""
    console.print("[yellow]Use: futurnal orchestrator jobs [list|show|cancel][/yellow]")


jobs_app = typer.Typer(help="Job management commands")
orchestrator_app.add_typer(jobs_app, name="jobs")


@jobs_app.command("list")
def jobs_list(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status: pending, running, succeeded, failed"),
    job_type: Optional[str] = typer.Option(None, "--job-type", help="Filter by type: local_files, obsidian_vault, imap_mailbox, github_repository"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source name"),
    limit: int = typer.Option(20, "--limit", help="Limit results (default: 20)"),
    since: Optional[str] = typer.Option(None, "--since", help="Show jobs since date (e.g., '2024-01-01' or '24h')"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """List ingestion jobs with optional filtering."""
    workspace_path = _get_workspace_path(workspace)
    queue = JobQueue(workspace_path / "queue" / "jobs.db")

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            console.print(f"Valid: {', '.join([s.value for s in JobStatus])}")
            raise typer.Exit(1)

    # Get jobs from queue
    jobs = queue.snapshot(status=status_filter, limit=limit)

    # Apply additional filters
    if job_type:
        jobs = [j for j in jobs if j["job_type"] == job_type]
    if source:
        jobs = [j for j in jobs if j["payload"].get("source_name") == source]

    # Parse and apply since filter
    if since:
        cutoff = _parse_since(since)
        if cutoff:
            jobs = [j for j in jobs if datetime.fromisoformat(j["created_at"]) >= cutoff]

    if format_output == "json":
        console.print(json.dumps(jobs, indent=2))
    elif format_output == "yaml":
        console.print(yaml.dump(jobs, default_flow_style=False))
    else:
        # Table format
        if not jobs:
            console.print("[yellow]No jobs found matching criteria[/yellow]")
            return

        table = Table(title=f"Ingestion Jobs ({len(jobs)} total)")
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Source", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="magenta")
        table.add_column("Attempts", justify="right")

        for job in jobs:
            job_id_short = job["job_id"][:8]
            source_name = job["payload"].get("source_name", job["payload"].get("mailbox_id", "N/A"))
            created_time = _format_timestamp(job["created_at"])

            table.add_row(
                job_id_short,
                job["job_type"],
                source_name[:20],
                job["status"],
                created_time,
                str(job["attempts"]),
            )

        console.print(table)


@jobs_app.command("show")
def jobs_show(
    job_id: str = typer.Argument(..., help="Job ID to display"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """Display detailed information about a specific job."""
    workspace_path = _get_workspace_path(workspace)
    queue = JobQueue(workspace_path / "queue" / "jobs.db")

    job = queue.get_job(job_id)
    if not job:
        console.print(f"[red]Job {job_id} not found[/red]")
        raise typer.Exit(1)

    if format_output == "json":
        console.print(json.dumps(job, indent=2))
    elif format_output == "yaml":
        console.print(yaml.dump(job, default_flow_style=False))
    else:
        # Rich panel format
        sections = []
        sections.append(f"[bold]Job ID:[/bold]       {job['job_id']}")
        sections.append(f"[bold]Type:[/bold]         {job['job_type']}")
        sections.append(f"[bold]Status:[/bold]       {job['status']}")
        sections.append(f"[bold]Priority:[/bold]     {job['priority']}")
        sections.append(f"[bold]Attempts:[/bold]     {job['attempts']}")
        sections.append("")
        sections.append(f"[bold]Created:[/bold]      {job['created_at']}")
        sections.append(f"[bold]Updated:[/bold]      {job['updated_at']}")
        if job.get("scheduled_for"):
            sections.append(f"[bold]Scheduled:[/bold]    {job['scheduled_for']}")
        sections.append("")
        sections.append("[bold]Payload:[/bold]")
        for key, value in job["payload"].items():
            sections.append(f"  {key}: {value}")

        content = "\n".join(sections)
        console.print(Panel(content, title=f"Job Details: {job_id[:8]}", border_style="green"))


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: str = typer.Argument(..., help="Job ID to cancel"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    reason: Optional[str] = typer.Option(None, "--reason", help="Cancellation reason (for audit log)"),
    force: bool = typer.Option(False, "--force", help="Force cancel running job"),
) -> None:
    """Cancel a pending or running job."""
    workspace_path = _get_workspace_path(workspace)
    queue = JobQueue(workspace_path / "queue" / "jobs.db")

    # Check job exists
    job = queue.get_job(job_id)
    if not job:
        console.print(f"[red]Job {job_id} not found[/red]")
        raise typer.Exit(1)

    current_status = job["status"]

    # Validate status
    if current_status == "running" and not force:
        console.print(f"[yellow]Job {job_id} is currently running.[/yellow]")
        console.print("Use --force to cancel running job (may leave inconsistent state)")
        raise typer.Exit(1)

    if current_status not in ("pending", "running"):
        console.print(f"[red]Cannot cancel job with status '{current_status}'[/red]")
        console.print("Only pending or running jobs can be cancelled")
        raise typer.Exit(1)

    # Cancel the job
    try:
        queue.cancel_job(job_id)
        console.print(f"[green]✓ Job {job_id} cancelled[/green]")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "job_cancel",
            {
                "job_id": job_id,
                "previous_status": current_status,
                "reason": reason or "operator_cancel",
                "forced": force,
            },
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


sources_app = typer.Typer(help="Source management commands")
orchestrator_app.add_typer(sources_app, name="sources")


@sources_app.command("list")
def sources_list(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    status_filter: Optional[str] = typer.Option(None, "--status", help="Filter by status: active or paused"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """List registered ingestion sources."""
    workspace_path = _get_workspace_path(workspace)
    status_report = collect_status_report(workspace_path=workspace_path)
    sources = status_report["sources"]

    # Apply status filter
    if status_filter:
        sources = [s for s in sources if s["status"] == status_filter.lower()]

    if format_output == "json":
        console.print(json.dumps(sources, indent=2))
    elif format_output == "yaml":
        console.print(yaml.dump(sources, default_flow_style=False))
    else:
        # Table format
        if not sources:
            console.print("[yellow]No sources found matching criteria[/yellow]")
            return

        table = Table(title=f"Ingestion Sources ({len(sources)} total)")
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Last Run", style="magenta")

        for source in sources:
            status_icon = "✓" if source["status"] == "active" else "⏸"
            last_run = source.get("last_run") or "never"

            table.add_row(
                source["name"][:50],
                source["type"],
                f"{status_icon} {source['status']}",
                last_run,
            )

        console.print(table)


@sources_app.command("pause")
def sources_pause(
    source_name: str = typer.Argument(..., help="Source name to pause"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    reason: Optional[str] = typer.Option(None, "--reason", help="Reason for pause (for audit log)"),
) -> None:
    """Pause a source's scheduled jobs."""
    workspace_path = _get_workspace_path(workspace)
    paused_registry = PausedSourcesRegistry(workspace_path / "orchestrator" / "paused_sources.json")

    # Check if already paused
    if paused_registry.is_paused(source_name):
        console.print(f"[yellow]Source {source_name} is already paused[/yellow]")
        return

    paused_registry.pause(source_name)
    console.print(f"[green]✓ Source {source_name} paused[/green]")
    console.print("  Scheduled jobs will no longer be enqueued")

    # Log audit event
    _log_cli_action(
        workspace_path,
        "source_pause",
        {
            "source_name": source_name,
            "reason": reason or "operator_pause",
        },
    )


@sources_app.command("resume")
def sources_resume(
    source_name: str = typer.Argument(..., help="Source name to resume"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    reason: Optional[str] = typer.Option(None, "--reason", help="Reason for resume (for audit log)"),
) -> None:
    """Resume a paused source's scheduled jobs."""
    workspace_path = _get_workspace_path(workspace)
    paused_registry = PausedSourcesRegistry(workspace_path / "orchestrator" / "paused_sources.json")

    # Check if actually paused
    if not paused_registry.is_paused(source_name):
        console.print(f"[yellow]Source {source_name} is not paused[/yellow]")
        return

    try:
        paused_registry.resume(source_name)
        console.print(f"[green]✓ Source {source_name} resumed[/green]")
        console.print("  Scheduled jobs will now be enqueued normally")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "source_resume",
            {
                "source_name": source_name,
                "reason": reason or "operator_resume",
            },
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@sources_app.command("trigger")
def sources_trigger(
    source_name: str = typer.Argument(..., help="Source name to trigger"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    force: bool = typer.Option(False, "--force", help="Trigger even if source is paused"),
    priority: str = typer.Option("high", "--priority", help="Job priority: low, normal, high"),
    wait: bool = typer.Option(False, "--wait", help="Wait for job to complete"),
) -> None:
    """Manually trigger a source ingestion job."""
    workspace_path = _get_workspace_path(workspace)
    queue = JobQueue(workspace_path / "queue" / "jobs.db")
    paused_registry = PausedSourcesRegistry(workspace_path / "orchestrator" / "paused_sources.json")

    # Check if paused
    if paused_registry.is_paused(source_name) and not force:
        console.print(f"[yellow]Source {source_name} is paused[/yellow]")
        console.print("Use --force to trigger anyway")
        raise typer.Exit(1)

    # Parse priority
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
    }
    job_priority = priority_map.get(priority.lower(), JobPriority.HIGH)

    # Create and enqueue job
    job_id = str(uuid.uuid4())
    job = IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,  # Default, could be improved with source type detection
        payload={
            "source_name": source_name,
            "trigger": "manual",
        },
        priority=job_priority,
        scheduled_for=datetime.utcnow(),
    )

    queue.enqueue(job)
    console.print(f"[green]✓ Job {job_id[:8]} enqueued for source {source_name}[/green]")
    console.print(f"  Priority: {priority}")

    # Log audit event
    _log_cli_action(
        workspace_path,
        "source_trigger",
        {
            "source_name": source_name,
            "job_id": job_id,
            "priority": priority,
            "forced": force,
        },
    )

    if wait:
        console.print("[yellow]Waiting for job to complete...[/yellow]")
        # Simple polling (could be improved with proper job monitoring)
        while True:
            time.sleep(2)
            job_status = queue.get_job(job_id)
            if job_status and job_status["status"] in ("succeeded", "failed"):
                status = job_status["status"]
                if status == "succeeded":
                    console.print(f"[green]✓ Job completed successfully[/green]")
                else:
                    console.print(f"[red]✗ Job failed[/red]")
                break


@orchestrator_app.command("health")
def health_command(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed health information"),
) -> None:
    """Run comprehensive health checks on orchestrator subsystems."""
    workspace_path = _get_workspace_path(workspace)

    try:
        settings = load_settings()
        health_report = collect_health_report(settings=settings, workspace_path=workspace_path)
    except Exception as e:
        console.print(f"[red]Failed to collect health report: {e}[/red]")
        raise typer.Exit(1)

    if format_output == "json":
        console.print(json.dumps(health_report, indent=2))
    elif format_output == "yaml":
        console.print(yaml.dump(health_report, default_flow_style=False))
    else:
        # Table format
        table = Table(title="Health Checks")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Detail", style="white")

        for check in health_report["checks"]:
            status_icon = "✓" if check["status"] == "ok" else "⚠"
            status_style = "green" if check["status"] == "ok" else "yellow"

            table.add_row(
                check["name"],
                f"[{status_style}]{status_icon} {check['status']}[/{status_style}]",
                check["detail"][:60],
            )

        overall_status = health_report["status"]
        overall_style = "green" if overall_status == "ok" else "yellow"
        console.print(f"\n[{overall_style}]Overall Status: {overall_status.upper()}[/{overall_style}]\n")
        console.print(table)


telemetry_app = typer.Typer(help="Telemetry viewing commands")
orchestrator_app.add_typer(telemetry_app, name="telemetry")


@telemetry_app.command("summary")
def telemetry_summary(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    since: Optional[str] = typer.Option(None, "--since", help="Show metrics since date (e.g., '24h', '7d')"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """View overall telemetry summary."""
    workspace_path = _get_workspace_path(workspace)
    summary_file = workspace_path / "telemetry" / "telemetry_summary.json"

    if not summary_file.exists():
        console.print("[yellow]No telemetry data available[/yellow]")
        return

    summary = json.loads(summary_file.read_text())

    if format_output == "json":
        console.print(json.dumps(summary, indent=2))
    elif format_output == "yaml":
        console.print(yaml.dump(summary, default_flow_style=False))
    else:
        # Table format
        overall = summary.get("overall", {})
        console.print("\n[bold cyan]Overall Statistics[/bold cyan]")
        console.print(f"  Total Jobs:   {overall.get('jobs', 0)}")
        console.print(f"  Files:        {overall.get('files', 0)}")
        console.print(f"  Data:         {_format_bytes(overall.get('bytes', 0))}")
        console.print(f"  Avg Duration: {_format_duration(overall.get('avg_duration', 0))}")
        console.print(f"  Throughput:   {_format_bytes(overall.get('throughput_bytes_per_second', 0))}/s")

        console.print("\n[bold cyan]By Status[/bold cyan]")
        table = Table()
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Files", justify="right")
        table.add_column("Data", justify="right")
        table.add_column("Throughput", justify="right")

        for status, stats in summary.get("statuses", {}).items():
            table.add_row(
                status,
                str(stats.get("count", 0)),
                str(stats.get("files", 0)),
                _format_bytes(stats.get("bytes", 0)),
                f"{_format_bytes(stats.get('throughput_bytes_per_second', 0))}/s",
            )

        console.print(table)


@telemetry_app.command("failures")
def telemetry_failures(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    since: Optional[str] = typer.Option(None, "--since", help="Show failures since date (e.g., '24h', '7d')"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """View failure statistics with breakdown by reason and connector."""
    workspace_path = _get_workspace_path(workspace)
    telemetry_dir = workspace_path / "telemetry"

    if not telemetry_dir.exists():
        console.print("[yellow]No telemetry data available[/yellow]")
        return

    analyzer = TelemetryAnalyzer(telemetry_dir)

    # Parse since filter
    since_dt = _parse_since(since) if since else None

    # Analyze failures
    stats = analyzer.analyze_failures(since=since_dt)

    if format_output == "json":
        output = {
            "total_failures": stats.total_failures,
            "recent_failures_24h": stats.recent_failures_24h,
            "failure_rate": stats.failure_rate,
            "by_reason": stats.failures_by_reason,
            "by_connector": stats.failures_by_connector,
        }
        console.print(json.dumps(output, indent=2))
    elif format_output == "yaml":
        output = {
            "total_failures": stats.total_failures,
            "recent_failures_24h": stats.recent_failures_24h,
            "failure_rate": stats.failure_rate,
            "by_reason": stats.failures_by_reason,
            "by_connector": stats.failures_by_connector,
        }
        console.print(yaml.dump(output, default_flow_style=False))
    else:
        # Table format
        console.print("\n[bold cyan]Failure Statistics[/bold cyan]")
        console.print(f"  Total Failures:    {stats.total_failures}")
        console.print(f"  Recent (24h):      {stats.recent_failures_24h}")
        console.print(f"  Failure Rate:      {stats.failure_rate:.1f}%")

        if stats.failures_by_reason:
            console.print("\n[bold cyan]Failures by Reason:[/bold cyan]")
            table = Table()
            table.add_column("Reason", style="cyan")
            table.add_column("Count", justify="right", style="yellow")
            table.add_column("Percentage", justify="right", style="green")

            for reason, count in sorted(stats.failures_by_reason.items(), key=lambda x: -x[1]):
                percentage = (count / stats.total_failures * 100.0) if stats.total_failures > 0 else 0.0
                table.add_row(reason, str(count), f"{percentage:.1f}%")

            console.print(table)

        if stats.failures_by_connector:
            console.print("\n[bold cyan]Failures by Connector:[/bold cyan]")
            table = Table()
            table.add_column("Connector", style="cyan")
            table.add_column("Count", justify="right", style="yellow")
            table.add_column("Percentage", justify="right", style="green")

            for connector, count in sorted(stats.failures_by_connector.items(), key=lambda x: -x[1]):
                percentage = (count / stats.total_failures * 100.0) if stats.total_failures > 0 else 0.0
                table.add_row(connector, str(count), f"{percentage:.1f}%")

            console.print(table)


@telemetry_app.command("throughput")
def telemetry_throughput(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    since: Optional[str] = typer.Option(None, "--since", help="Calculate throughput since date (e.g., '1h', '24h')"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """View throughput metrics over time window."""
    workspace_path = _get_workspace_path(workspace)
    telemetry_dir = workspace_path / "telemetry"

    if not telemetry_dir.exists():
        console.print("[yellow]No telemetry data available[/yellow]")
        return

    analyzer = TelemetryAnalyzer(telemetry_dir)

    # Parse since filter
    since_dt = _parse_since(since) if since else None

    # Calculate throughput
    metrics = analyzer.calculate_throughput(since=since_dt)

    if format_output == "json":
        output = {
            "files_processed": metrics.files_processed,
            "bytes_processed": metrics.bytes_processed,
            "duration_seconds": metrics.duration_seconds,
            "throughput_bytes_per_second": metrics.throughput_bytes_per_second,
            "throughput_files_per_second": metrics.throughput_files_per_second,
            "throughput_mbps": metrics.throughput_mbps,
        }
        console.print(json.dumps(output, indent=2))
    elif format_output == "yaml":
        output = {
            "files_processed": metrics.files_processed,
            "bytes_processed": metrics.bytes_processed,
            "duration_seconds": metrics.duration_seconds,
            "throughput_bytes_per_second": metrics.throughput_bytes_per_second,
            "throughput_files_per_second": metrics.throughput_files_per_second,
            "throughput_mbps": metrics.throughput_mbps,
        }
        console.print(yaml.dump(output, default_flow_style=False))
    else:
        # Table format
        time_window = f"since {since}" if since else "all time"
        console.print(f"\n[bold cyan]Throughput Metrics ({time_window})[/bold cyan]")
        console.print(f"  Files Processed:   {metrics.files_processed:,}")
        console.print(f"  Data Processed:    {_format_bytes(metrics.bytes_processed)}")
        console.print(f"  Total Duration:    {_format_duration(metrics.duration_seconds)}")
        console.print(f"  Throughput:        {metrics.throughput_mbps:.2f} MB/s")
        console.print(f"  Files/Second:      {metrics.throughput_files_per_second:.2f}")


@telemetry_app.command("by-connector")
def telemetry_by_connector(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    since: Optional[str] = typer.Option(None, "--since", help="Show metrics since date (e.g., '24h', '7d')"),
    format_output: str = typer.Option("table", "--format", help="Output format: table, json, or yaml"),
) -> None:
    """View metrics broken down by connector type."""
    workspace_path = _get_workspace_path(workspace)
    telemetry_dir = workspace_path / "telemetry"

    if not telemetry_dir.exists():
        console.print("[yellow]No telemetry data available[/yellow]")
        return

    analyzer = TelemetryAnalyzer(telemetry_dir)

    # Parse since filter
    since_dt = _parse_since(since) if since else None

    # Get connector metrics
    metrics = analyzer.metrics_by_connector(since=since_dt)

    if format_output == "json":
        output = []
        for m in metrics:
            output.append({
                "connector_type": m.connector_type,
                "total_jobs": m.total_jobs,
                "succeeded": m.succeeded_jobs,
                "failed": m.failed_jobs,
                "files": m.total_files,
                "bytes": m.total_bytes,
                "duration": m.total_duration,
                "throughput_bps": m.avg_throughput_bps,
                "success_rate": m.success_rate,
            })
        console.print(json.dumps(output, indent=2))
    elif format_output == "yaml":
        output = []
        for m in metrics:
            output.append({
                "connector_type": m.connector_type,
                "total_jobs": m.total_jobs,
                "succeeded": m.succeeded_jobs,
                "failed": m.failed_jobs,
                "files": m.total_files,
                "bytes": m.total_bytes,
                "duration": m.total_duration,
                "throughput_bps": m.avg_throughput_bps,
                "success_rate": m.success_rate,
            })
        console.print(yaml.dump(output, default_flow_style=False))
    else:
        # Table format
        if not metrics:
            console.print("[yellow]No connector metrics available[/yellow]")
            return

        time_window = f"since {since}" if since else "all time"
        console.print(f"\n[bold cyan]Metrics by Connector ({time_window})[/bold cyan]\n")

        table = Table()
        table.add_column("Connector", style="cyan")
        table.add_column("Jobs", justify="right", style="yellow")
        table.add_column("Success", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Files", justify="right", style="blue")
        table.add_column("Data", justify="right", style="magenta")
        table.add_column("Throughput", justify="right", style="green")
        table.add_column("Rate", justify="right", style="yellow")

        for m in metrics:
            table.add_row(
                m.connector_type,
                str(m.total_jobs),
                str(m.succeeded_jobs),
                str(m.failed_jobs),
                str(m.total_files),
                _format_bytes(m.total_bytes),
                f"{m.avg_throughput_bps / (1024 * 1024):.2f} MB/s",
                f"{m.success_rate:.1f}%",
            )

        console.print(table)


@telemetry_app.command("clean")
def telemetry_clean(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    older_than_days: int = typer.Option(..., "--older-than-days", help="Remove entries older than N days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed without deleting"),
) -> None:
    """Clean up old telemetry data to save disk space."""
    workspace_path = _get_workspace_path(workspace)
    telemetry_dir = workspace_path / "telemetry"

    if not telemetry_dir.exists():
        console.print("[yellow]No telemetry data to clean[/yellow]")
        return

    analyzer = TelemetryAnalyzer(telemetry_dir)

    console.print(f"[yellow]Cleaning telemetry entries older than {older_than_days} days...[/yellow]")

    if dry_run:
        console.print("[cyan](Dry run - no data will be deleted)[/cyan]")

    # Clean telemetry
    lines_removed = analyzer.clean_old_telemetry(
        older_than_days=older_than_days,
        dry_run=dry_run,
    )

    if lines_removed == 0:
        console.print("[green]No old telemetry entries found[/green]")
    elif dry_run:
        console.print(f"[yellow]Would remove {lines_removed} telemetry entries[/yellow]")
    else:
        console.print(f"[green]✓ Removed {lines_removed} telemetry entries[/green]")
        console.print(f"[cyan]Backup created at: {telemetry_dir / 'telemetry.log.backup'}[/cyan]")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "telemetry_clean",
            {
                "older_than_days": older_than_days,
                "lines_removed": lines_removed,
            },
        )


# Database management commands
db_app = typer.Typer(help="Database management commands")
orchestrator_app.add_typer(db_app, name="db")


@db_app.command("backup")
def db_backup(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    comment: Optional[str] = typer.Option(None, "--comment", help="Optional comment for backup filename"),
) -> None:
    """Create a backup of the orchestrator queue database."""
    workspace_path = _get_workspace_path(workspace)
    db_path = workspace_path / "queue" / "jobs.db"

    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    db_manager = DatabaseManager(db_path)

    try:
        console.print("[yellow]Creating database backup...[/yellow]")
        backup_path = db_manager.backup(comment=comment)
        console.print(f"[green]✓ Backup created: {backup_path}[/green]")

        # Show backup size
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        console.print(f"  Size: {size_mb:.2f} MB")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "db_backup",
            {
                "backup_path": str(backup_path),
                "comment": comment,
                "size_bytes": backup_path.stat().st_size,
            },
        )

    except BackupError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@db_app.command("restore")
def db_restore(
    backup_path: Path = typer.Argument(..., help="Path to backup file"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing database without confirmation"),
) -> None:
    """Restore orchestrator queue database from a backup."""
    workspace_path = _get_workspace_path(workspace)
    db_path = workspace_path / "queue" / "jobs.db"

    db_manager = DatabaseManager(db_path)

    # Confirm restoration
    if db_path.exists() and not force:
        console.print("[yellow]Warning: This will replace the current database![/yellow]")
        console.print(f"Current database: {db_path}")
        console.print(f"Backup to restore: {backup_path}")

        if not typer.confirm("\nProceed with restore?"):
            console.print("Restore cancelled")
            return

    try:
        console.print("[yellow]Restoring database from backup...[/yellow]")
        db_manager.restore(backup_path, force=True)
        console.print(f"[green]✓ Database restored from: {backup_path}[/green]")
        console.print(f"[cyan]Previous database backed up to: {db_path}.before-restore[/cyan]")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "db_restore",
            {
                "backup_path": str(backup_path),
                "database_path": str(db_path),
            },
        )

    except RestoreError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@db_app.command("check")
def db_check(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
) -> None:
    """Check orchestrator queue database integrity."""
    workspace_path = _get_workspace_path(workspace)
    db_path = workspace_path / "queue" / "jobs.db"

    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    db_manager = DatabaseManager(db_path)

    try:
        console.print("[yellow]Checking database integrity...[/yellow]")
        is_valid, detail = db_manager.check_integrity()

        if is_valid:
            console.print(f"[green]✓ {detail}[/green]")
        else:
            console.print(f"[red]✗ {detail}[/red]")
            console.print("\n[yellow]Recommendation:[/yellow]")
            console.print("  1. Create a backup: futurnal orchestrator db backup")
            console.print("  2. Restore from a known good backup")
            raise typer.Exit(1)

        # Show database stats
        stats = db_manager.get_stats()
        console.print(f"\n[bold cyan]Database Statistics:[/bold cyan]")
        console.print(f"  Size:      {stats['size_mb']:.2f} MB")
        console.print(f"  Jobs:      {stats['job_count']:,}")
        console.print(f"  Pages:     {stats['page_count']:,}")

    except IntegrityError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@db_app.command("list-backups")
def db_list_backups(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    format_output: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """List available database backups."""
    workspace_path = _get_workspace_path(workspace)
    db_path = workspace_path / "queue" / "jobs.db"

    db_manager = DatabaseManager(db_path)
    backups = db_manager.list_backups()

    if not backups:
        console.print("[yellow]No backups found[/yellow]")
        return

    if format_output == "json":
        output = []
        for backup_path, created, size in backups:
            output.append({
                "path": str(backup_path),
                "created": created.isoformat(),
                "size_bytes": size,
                "size_mb": size / (1024 * 1024),
            })
        console.print(json.dumps(output, indent=2))
    else:
        # Table format
        table = Table(title=f"Database Backups ({len(backups)} total)")
        table.add_column("Filename", style="cyan")
        table.add_column("Created", style="yellow")
        table.add_column("Age", style="magenta")
        table.add_column("Size", style="green", justify="right")

        now = datetime.utcnow()
        for backup_path, created, size in backups:
            age_delta = now - created
            age_str = _format_timestamp(created.isoformat())
            size_mb = size / (1024 * 1024)

            table.add_row(
                backup_path.name,
                created.strftime("%Y-%m-%d %H:%M:%S"),
                age_str,
                f"{size_mb:.2f} MB",
            )

        console.print(table)


@db_app.command("purge-backups")
def db_purge_backups(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    keep_count: int = typer.Option(10, "--keep-count", help="Keep this many recent backups"),
    older_than_days: Optional[int] = typer.Option(None, "--older-than-days", help="Remove backups older than N days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed without deleting"),
) -> None:
    """Remove old database backups."""
    workspace_path = _get_workspace_path(workspace)
    db_path = workspace_path / "queue" / "jobs.db"

    db_manager = DatabaseManager(db_path)

    console.print(f"[yellow]Purging old backups (keep: {keep_count} recent)...[/yellow]")
    if older_than_days:
        console.print(f"[yellow]Also removing backups older than {older_than_days} days...[/yellow]")

    if dry_run:
        console.print("[cyan](Dry run - no backups will be deleted)[/cyan]")

    removed_count = db_manager.purge_old_backups(
        keep_count=keep_count,
        older_than_days=older_than_days,
        dry_run=dry_run,
    )

    if removed_count == 0:
        console.print("[green]No backups need to be removed[/green]")
    elif dry_run:
        console.print(f"[yellow]Would remove {removed_count} backup(s)[/yellow]")
    else:
        console.print(f"[green]✓ Removed {removed_count} backup(s)[/green]")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "db_purge_backups",
            {
                "keep_count": keep_count,
                "older_than_days": older_than_days,
                "removed_count": removed_count,
            },
        )


@db_app.command("vacuum")
def db_vacuum(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
) -> None:
    """Vacuum the database to reclaim space and optimize performance."""
    workspace_path = _get_workspace_path(workspace)
    db_path = workspace_path / "queue" / "jobs.db"

    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    db_manager = DatabaseManager(db_path)

    # Get size before
    stats_before = db_manager.get_stats()
    size_before_mb = stats_before['size_mb']

    try:
        console.print("[yellow]Vacuuming database...[/yellow]")
        db_manager.vacuum()

        # Get size after
        stats_after = db_manager.get_stats()
        size_after_mb = stats_after['size_mb']
        reclaimed_mb = size_before_mb - size_after_mb

        console.print(f"[green]✓ Database vacuumed[/green]")
        console.print(f"  Before: {size_before_mb:.2f} MB")
        console.print(f"  After:  {size_after_mb:.2f} MB")
        console.print(f"  Reclaimed: {reclaimed_mb:.2f} MB")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "db_vacuum",
            {
                "size_before_mb": size_before_mb,
                "size_after_mb": size_after_mb,
                "reclaimed_mb": reclaimed_mb,
            },
        )

    except DatabaseError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# NOTE: Orchestrator config commands have been moved to orchestrator_config_app
# See orchestrator_app.add_typer(orchestrator_config_app, name="config") above
# This provides: validate, show, migrate, init, and set commands


def _parse_since(since_str: str) -> Optional[datetime]:
    """Parse --since option into datetime.

    Args:
        since_str: Time string like '24h', '7d', or '2024-01-01'

    Returns:
        Datetime or None if parse fails
    """
    try:
        # Try parsing as ISO date first
        return datetime.fromisoformat(since_str)
    except ValueError:
        pass

    # Try parsing as relative time (e.g., "24h", "7d")
    if since_str.endswith("h"):
        hours = int(since_str[:-1])
        return datetime.utcnow() - timedelta(hours=hours)
    elif since_str.endswith("d"):
        days = int(since_str[:-1])
        return datetime.utcnow() - timedelta(days=days)
    elif since_str.endswith("m"):
        minutes = int(since_str[:-1])
        return datetime.utcnow() - timedelta(minutes=minutes)

    return None


@orchestrator_app.command("stop")
def stop_orchestrator(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    force: bool = typer.Option(False, "--force", help="Force stop (send SIGKILL immediately)"),
    timeout: float = typer.Option(30.0, "--timeout", help="Graceful shutdown timeout in seconds"),
) -> None:
    """Stop the running orchestrator daemon.

    By default, sends SIGTERM for graceful shutdown and waits up to 30 seconds.
    Use --force to send SIGKILL immediately for unresponsive daemons.
    """
    workspace_path = _get_workspace_path(workspace)
    daemon = OrchestratorDaemon(workspace_path)

    # Check status first
    status = daemon.status()

    if not status.running:
        if status.stale_pid_file:
            console.print("[yellow]Orchestrator is not running (cleaned up stale PID file)[/yellow]")
            daemon.register_stop()  # Clean up
        else:
            console.print("[yellow]Orchestrator is not running[/yellow]")
        return

    # Stop the daemon
    console.print(f"[yellow]Stopping orchestrator (PID: {status.pid})...[/yellow]")

    if force:
        console.print("[red]Force stopping (SIGKILL)...[/red]")
    else:
        console.print(f"[yellow]Graceful shutdown (timeout: {timeout}s)...[/yellow]")

    try:
        daemon.stop(force=force, timeout=timeout)
        console.print("[green]✓ Orchestrator stopped[/green]")

        # Log audit event
        _log_cli_action(
            workspace_path,
            "orchestrator_stop",
            {
                "pid": status.pid,
                "forced": force,
                "timeout": timeout,
            },
        )

    except NotRunningError as e:
        console.print(f"[yellow]{e}[/yellow]")
    except DaemonError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Process was forcefully terminated[/yellow]")
        raise typer.Exit(1)


@orchestrator_app.command("daemon-status")
def daemon_status(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for IPC"),
) -> None:
    """Show orchestrator daemon running status.

    This command checks if the orchestrator daemon process is currently running.
    Use --json for machine-readable output (used by desktop app IPC).
    """
    workspace_path = _get_workspace_path(workspace)
    daemon = OrchestratorDaemon(workspace_path)
    status = daemon.status()

    status_data = {
        "running": status.running,
        "pid": status.pid,
        "workspace": str(workspace_path),
        "stale_pid_file": status.stale_pid_file,
    }

    if json_output:
        console.print(json.dumps(status_data))
    else:
        if status.running:
            console.print(f"[green]✓ Orchestrator is running (PID: {status.pid})[/green]")
        elif status.stale_pid_file:
            console.print("[yellow]Orchestrator is not running (stale PID file detected)[/yellow]")
        else:
            console.print("[yellow]Orchestrator is not running[/yellow]")
        console.print(f"Workspace: {workspace_path}")


# Keep original start command for orchestrator daemon
@orchestrator_app.command("start")
def start_orchestrator(
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
    imap_interval: int = typer.Option(
        300,
        "--imap-interval",
        help="IMAP sync interval in seconds (default: 300 = 5 minutes)"
    ),
    imap_priority: str = typer.Option(
        "normal",
        "--imap-priority",
        help="IMAP job priority: low, normal, high"
    ),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Run in foreground (for debugging)"
    ),
) -> None:
    """Start the ingestion orchestrator daemon with all configured sources."""
    console.print("[bold blue]Starting Futurnal Ingestion Orchestrator[/bold blue]")

    workspace_path = Path(workspace).expanduser()
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Check if already running and register start
    daemon = OrchestratorDaemon(workspace_path)
    try:
        daemon.register_start()
    except AlreadyRunningError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Use 'futurnal orchestrator stop' to stop the running instance")
        raise typer.Exit(1)

    if foreground:
        console.print("[yellow]Running in foreground mode (debugging)[/yellow]")

    # Parse priority
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
    }
    job_priority = priority_map.get(imap_priority.lower(), JobPriority.NORMAL)

    # Initialize orchestrator
    console.print(f"Workspace: {workspace_path}")
    console.print(f"IMAP sync interval: {imap_interval}s")
    console.print(f"IMAP job priority: {imap_priority}")

    try:
        from futurnal.ingestion.imap.descriptor import MailboxRegistry
        from futurnal.ingestion.imap.orchestrator_integration import ImapSourceRegistration

        # Initialize job queue and state store
        job_queue = _load_job_queue(workspace_path)
        state_store = _load_state_store(workspace_path)

        orchestrator = IngestionOrchestrator(
            workspace_dir=str(workspace_path),
            job_queue=job_queue,
            state_store=state_store,
        )

        # Load and register IMAP mailboxes
        registry_root = workspace_path / "sources" / "imap"
        if registry_root.exists():
            mailbox_registry = MailboxRegistry(registry_root=registry_root)
            mailboxes = mailbox_registry.list()

            if mailboxes:
                console.print(f"\n[bold yellow]Registering {len(mailboxes)} IMAP mailbox(es):[/bold yellow]")

                table = Table()
                table.add_column("Email", style="cyan")
                table.add_column("Folders", style="green")
                table.add_column("Schedule", style="blue")

                for mailbox in mailboxes:
                    # Register with orchestrator
                    ImapSourceRegistration.register_mailbox(
                        orchestrator=orchestrator,
                        mailbox_descriptor=mailbox,
                        schedule="@interval",
                        interval_seconds=imap_interval,
                        priority=job_priority,
                    )

                    table.add_row(
                        mailbox.email_address,
                        ", ".join(mailbox.folders[:3]) + ("..." if len(mailbox.folders) > 3 else ""),
                        f"Every {imap_interval}s",
                    )

                console.print(table)
            else:
                console.print("[yellow]No IMAP mailboxes configured[/yellow]")
        else:
            console.print("[yellow]No IMAP mailboxes configured[/yellow]")

        # Load and register local folder sources from sources.json
        local_sources_config = Path.home() / ".futurnal" / "sources.json"
        if local_sources_config.exists():
            try:
                from futurnal.ingestion.local.config import load_config_from_dict

                config_data = json.loads(local_sources_config.read_text())
                sources_list = load_config_from_dict(config_data)

                if sources_list.root:
                    console.print(f"\n[bold yellow]Registering {len(sources_list.root)} local source(s):[/bold yellow]")

                    table = Table()
                    table.add_column("Name", style="cyan")
                    table.add_column("Path", style="green")
                    table.add_column("Schedule", style="blue")

                    priority_map = {
                        "low": JobPriority.LOW,
                        "normal": JobPriority.NORMAL,
                        "high": JobPriority.HIGH,
                    }

                    for source in sources_list.root:
                        # Skip paused sources
                        if source.paused:
                            continue

                        # Determine schedule and interval (preserve original or default to @interval)
                        schedule = source.schedule if source.schedule else "@interval"
                        interval = source.interval_seconds if source.interval_seconds else 300

                        # Create and register the source
                        registration = SourceRegistration(
                            source=source,
                            schedule=schedule,
                            interval_seconds=int(interval) if schedule == "@interval" else None,
                            priority=priority_map.get(source.priority, JobPriority.NORMAL),
                            paused=source.paused,
                        )
                        orchestrator.register_source(registration)

                        # Display in table
                        if schedule == "@manual":
                            schedule_display = "Manual (file watcher)"
                        elif schedule == "@interval":
                            schedule_display = f"Every {interval}s"
                        else:
                            schedule_display = schedule
                        table.add_row(
                            source.name,
                            str(source.root_path)[:50] + "..." if len(str(source.root_path)) > 50 else str(source.root_path),
                            schedule_display,
                        )

                    console.print(table)
                else:
                    console.print("[yellow]No local sources configured[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load local sources: {e}[/yellow]")
        else:
            console.print("[yellow]No local sources configured[/yellow]")

        # Start orchestrator
        console.print("\n[bold green]Orchestrator started![/bold green]")
        console.print("Press Ctrl+C to stop\n")

        # Setup signal handler for graceful shutdown
        shutdown_event = asyncio.Event()

        def signal_handler(sig, frame):
            console.print("\n[bold yellow]Shutting down orchestrator...[/bold yellow]")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start orchestrator
        orchestrator.start()

        # Wait for shutdown signal
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(shutdown_event.wait())
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(orchestrator.shutdown())
            daemon.register_stop()
            console.print("[bold green]Orchestrator stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Failed to start orchestrator: {e}[/bold red]")
        # Clean up PID file on failure
        daemon.register_stop()
        raise typer.Exit(1)


__all__ = ["orchestrator_app"]

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
from futurnal.orchestrator.scheduler import IngestionOrchestrator
from futurnal.orchestrator.queue import JobQueue, JobStatus
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.quarantine_cli import quarantine_app
from futurnal.orchestrator.config_cli import orchestrator_config_app
from futurnal.orchestrator.status import collect_status_report
from futurnal.orchestrator.source_control import PausedSourcesRegistry
from futurnal.orchestrator.health import collect_health_report
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
) -> None:
    """Start the ingestion orchestrator daemon with all configured sources."""
    console.print("[bold blue]Starting Futurnal Ingestion Orchestrator[/bold blue]")

    workspace_path = Path(workspace).expanduser()
    workspace_path.mkdir(parents=True, exist_ok=True)

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

        orchestrator = IngestionOrchestrator(
            workspace_dir=str(workspace_path),
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
            console.print("[bold green]Orchestrator stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Failed to start orchestrator: {e}[/bold red]")
        raise typer.Exit(1)


__all__ = ["orchestrator_app"]

"""CLI commands for orchestrator management."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from futurnal.orchestrator.scheduler import IngestionOrchestrator
from futurnal.ingestion.imap.descriptor import MailboxRegistry
from futurnal.ingestion.imap.orchestrator_integration import ImapSourceRegistration
from futurnal.orchestrator.models import JobPriority
from futurnal.orchestrator.quarantine_cli import quarantine_app

console = Console()
orchestrator_app = typer.Typer(help="Orchestrator management commands")

# Add quarantine management as a sub-command
orchestrator_app.add_typer(quarantine_app, name="quarantine")


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
    """Start the ingestion orchestrator with all configured sources.

    This command:
    - Loads all registered IMAP mailboxes
    - Registers them with the orchestrator for scheduled syncing
    - Starts the APScheduler event loop
    - Runs in foreground until Ctrl+C

    Examples:
        # Start with default settings (5 min intervals)
        futurnal orchestrator start

        # Custom interval (10 minutes)
        futurnal orchestrator start --imap-interval 600

        # High priority for IMAP jobs
        futurnal orchestrator start --imap-priority high
    """
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


@orchestrator_app.command("status")
def orchestrator_status(
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Show orchestrator status and job queue statistics."""
    workspace_path = Path(workspace).expanduser()

    console.print("[bold blue]Orchestrator Status[/bold blue]\n")

    # Check if orchestrator is running
    # Note: In production, you'd check a PID file or use a service manager
    console.print("[yellow]Status check not yet implemented[/yellow]")
    console.print("Use orchestrator logs and telemetry for status monitoring")

    # Show telemetry summary
    telemetry_summary = workspace_path / "telemetry" / "telemetry_summary.json"
    if telemetry_summary.exists():
        import json
        summary = json.loads(telemetry_summary.read_text())

        console.print("\n[bold yellow]Job Statistics:[/bold yellow]")
        overall = summary.get("overall", {})
        console.print(f"Total jobs: {overall.get('jobs', 0)}")
        console.print(f"Files processed: {overall.get('files', 0)}")
        console.print(f"Bytes processed: {overall.get('bytes', 0):.2f}")
        console.print(f"Avg duration: {overall.get('avg_duration', 0):.2f}s")
        console.print(f"Throughput: {overall.get('throughput_bytes_per_second', 0):.2f} bytes/s")

        console.print("\n[bold yellow]By Status:[/bold yellow]")
        for status, stats in summary.get("statuses", {}).items():
            console.print(f"  {status}: {stats.get('count', 0)} jobs")
    else:
        console.print("\n[yellow]No telemetry data available[/yellow]")


__all__ = ["orchestrator_app"]

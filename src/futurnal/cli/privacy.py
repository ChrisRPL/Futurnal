"""CLI commands for privacy management and data purge operations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from futurnal.configuration.settings import load_settings
from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry
from futurnal.privacy.purge import (
    DataPurgeService,
    PurgeConfirmationRequired,
    create_purge_service,
)

console = Console()
privacy_app = typer.Typer(help="Privacy management commands")
purge_app = typer.Typer(help="Data purge commands")
privacy_app.add_typer(purge_app, name="purge")


def _get_workspace_path(workspace: Optional[Path] = None) -> Path:
    """Get workspace path from option or settings."""
    if workspace:
        return Path(workspace).expanduser()
    try:
        settings = load_settings()
        return settings.workspace.workspace_path
    except Exception:
        return Path.home() / ".futurnal"


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


def _get_audit_logger(workspace: Path) -> Optional[AuditLogger]:
    """Get audit logger for workspace if exists."""
    audit_dir = workspace / "audit"
    if audit_dir.exists():
        return AuditLogger(output_dir=audit_dir)
    return None


def _get_consent_registry(workspace: Path) -> Optional[ConsentRegistry]:
    """Get consent registry for workspace if exists."""
    consent_dir = workspace / "consent"
    if consent_dir.exists():
        return ConsentRegistry(directory=consent_dir)
    return None


@purge_app.command("all")
def purge_all(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm purge operation (required)"),
    include_workspace_data: bool = typer.Option(False, "--include-workspace-data", help="Also purge workspace data directory"),
) -> None:
    """Purge ALL privacy-sensitive data (audit logs, consent records).

    WARNING: This operation is irreversible! Use with extreme caution.

    This will delete:
    - All audit log files
    - All consent records
    - All rotated/archived logs

    Use --include-workspace-data to also remove workspace data.
    """
    workspace_path = _get_workspace_path(workspace)

    if not confirm:
        console.print("[bold red]WARNING: This will permanently delete all privacy data![/bold red]")
        console.print("\nThe following will be deleted:")
        console.print("  - All audit logs")
        console.print("  - All consent records")
        if include_workspace_data:
            console.print("  - Workspace data directory")
        console.print("\n[yellow]To proceed, add --confirm flag[/yellow]")
        console.print("\nExample: futurnal privacy purge all --confirm")
        raise typer.Exit(1)

    # Double confirmation for dangerous operation
    if not typer.confirm("\nAre you SURE you want to purge all privacy data? This cannot be undone."):
        console.print("[yellow]Purge cancelled[/yellow]")
        raise typer.Exit(0)

    # Get components
    audit_logger = _get_audit_logger(workspace_path)
    consent_registry = _get_consent_registry(workspace_path)

    additional_paths = []
    if include_workspace_data:
        data_dir = workspace_path / "data"
        if data_dir.exists():
            additional_paths.append(data_dir)

    # Create purge service
    purge_service = DataPurgeService(
        audit_logger=audit_logger,
        consent_registry=consent_registry,
        additional_paths=additional_paths,
    )

    console.print("\n[bold yellow]Starting complete data purge...[/bold yellow]")

    try:
        result = purge_service.purge_all(confirm=True)

        if result.success:
            console.print("\n[bold green]Purge completed successfully![/bold green]")
        else:
            console.print("\n[bold yellow]Purge completed with errors[/bold yellow]")

        # Display results
        console.print(f"\n  Files deleted:    {result.files_deleted}")
        console.print(f"  Bytes freed:      {_format_bytes(result.bytes_freed)}")
        console.print(f"  Sources purged:   {', '.join(result.sources_purged)}")

        if result.errors:
            console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
            for error in result.errors:
                console.print(f"  - {error}")

        # Verify purge
        if purge_service.verify_purge():
            console.print("\n[green]Verification: All data successfully purged[/green]")
        else:
            console.print("\n[yellow]Verification: Some data may remain[/yellow]")

    except PurgeConfirmationRequired as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@purge_app.command("audit")
def purge_audit(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm purge operation (required)"),
) -> None:
    """Purge only audit logs.

    This will delete all audit log files including:
    - Main audit log
    - Rotated/archived logs
    - Daily review logs
    - Audit manifest
    """
    workspace_path = _get_workspace_path(workspace)

    if not confirm:
        console.print("[bold red]WARNING: This will permanently delete all audit logs![/bold red]")
        console.print("\n[yellow]To proceed, add --confirm flag[/yellow]")
        console.print("\nExample: futurnal privacy purge audit --confirm")
        raise typer.Exit(1)

    audit_logger = _get_audit_logger(workspace_path)

    if not audit_logger:
        console.print("[yellow]No audit logs found[/yellow]")
        return

    # Create purge service
    purge_service = DataPurgeService(
        audit_logger=audit_logger,
    )

    console.print("\n[bold yellow]Purging audit logs...[/bold yellow]")

    try:
        result = purge_service.purge_audit_logs(confirm=True)

        if result.success:
            console.print("\n[bold green]Audit logs purged successfully![/bold green]")
        else:
            console.print("\n[bold yellow]Purge completed with errors[/bold yellow]")

        console.print(f"\n  Files deleted:    {result.files_deleted}")
        console.print(f"  Bytes freed:      {_format_bytes(result.bytes_freed)}")

        if result.errors:
            console.print(f"\n[red]Errors:[/red]")
            for error in result.errors:
                console.print(f"  - {error}")

    except PurgeConfirmationRequired as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@purge_app.command("consent")
def purge_consent(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm purge operation (required)"),
) -> None:
    """Purge only consent records.

    This will delete all consent records. You will need to
    re-grant consent for any data sources to be processed.
    """
    workspace_path = _get_workspace_path(workspace)

    if not confirm:
        console.print("[bold red]WARNING: This will permanently delete all consent records![/bold red]")
        console.print("[bold yellow]You will need to re-grant consent for data processing.[/bold yellow]")
        console.print("\n[yellow]To proceed, add --confirm flag[/yellow]")
        console.print("\nExample: futurnal privacy purge consent --confirm")
        raise typer.Exit(1)

    consent_registry = _get_consent_registry(workspace_path)

    if not consent_registry:
        console.print("[yellow]No consent records found[/yellow]")
        return

    # Create purge service
    purge_service = DataPurgeService(
        consent_registry=consent_registry,
    )

    console.print("\n[bold yellow]Purging consent records...[/bold yellow]")

    try:
        result = purge_service.purge_consent(confirm=True)

        if result.success:
            console.print("\n[bold green]Consent records purged successfully![/bold green]")
        else:
            console.print("\n[bold yellow]Purge completed with errors[/bold yellow]")

        console.print(f"\n  Files deleted:    {result.files_deleted}")
        console.print(f"  Bytes freed:      {_format_bytes(result.bytes_freed)}")

        if result.errors:
            console.print(f"\n[red]Errors:[/red]")
            for error in result.errors:
                console.print(f"  - {error}")

    except PurgeConfirmationRequired as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@purge_app.command("source")
def purge_source(
    source_name: str = typer.Argument(..., help="Source name to purge"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm purge operation (required)"),
) -> None:
    """Purge data for a specific source.

    This will:
    - Revoke all consent for the source
    - Remove source from active consent list

    Use this when disconnecting a data source.
    """
    workspace_path = _get_workspace_path(workspace)

    if not confirm:
        console.print(f"[bold red]WARNING: This will purge all data for source '{source_name}'![/bold red]")
        console.print("\n[yellow]To proceed, add --confirm flag[/yellow]")
        console.print(f"\nExample: futurnal privacy purge source {source_name} --confirm")
        raise typer.Exit(1)

    consent_registry = _get_consent_registry(workspace_path)

    if not consent_registry:
        console.print("[yellow]No consent registry found[/yellow]")
        return

    # Create purge service
    purge_service = DataPurgeService(
        consent_registry=consent_registry,
    )

    console.print(f"\n[bold yellow]Purging data for source '{source_name}'...[/bold yellow]")

    try:
        result = purge_service.purge_by_source(source_name, confirm=True)

        if result.success:
            if source_name in result.sources_purged:
                console.print(f"\n[bold green]Source '{source_name}' purged successfully![/bold green]")
                console.print(f"\n  Consents revoked: {result.files_deleted}")
            else:
                console.print(f"\n[yellow]No data found for source '{source_name}'[/yellow]")
        else:
            console.print("\n[bold yellow]Purge completed with errors[/bold yellow]")
            for error in result.errors:
                console.print(f"  - {error}")

    except PurgeConfirmationRequired as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@purge_app.command("verify")
def verify_purge(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
) -> None:
    """Verify that data has been successfully purged.

    Checks that audit logs and consent records are empty or deleted.
    """
    workspace_path = _get_workspace_path(workspace)

    audit_logger = _get_audit_logger(workspace_path)
    consent_registry = _get_consent_registry(workspace_path)

    purge_service = DataPurgeService(
        audit_logger=audit_logger,
        consent_registry=consent_registry,
    )

    console.print("[bold cyan]Verifying purge status...[/bold cyan]")

    if purge_service.verify_purge():
        console.print("\n[bold green]Verification PASSED[/bold green]")
        console.print("  All privacy data has been purged")
    else:
        console.print("\n[bold yellow]Verification FAILED[/bold yellow]")
        console.print("  Some privacy data may still exist")

        # Show what remains
        if audit_logger:
            audit_dir = audit_logger.output_dir
            log_files = list(audit_dir.glob("audit*.log"))
            if log_files:
                console.print(f"\n  Remaining audit logs: {len(log_files)}")
                for f in log_files[:5]:
                    console.print(f"    - {f.name}")
                if len(log_files) > 5:
                    console.print(f"    ... and {len(log_files) - 5} more")

        if consent_registry:
            consent_path = consent_registry._path
            if consent_path.exists():
                console.print(f"\n  Consent file exists: {consent_path}")


@privacy_app.command("status")
def privacy_status(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    format_output: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """Show privacy data status and storage usage.

    Displays information about:
    - Audit log storage
    - Consent records
    - Data retention settings
    """
    workspace_path = _get_workspace_path(workspace)

    audit_logger = _get_audit_logger(workspace_path)
    consent_registry = _get_consent_registry(workspace_path)

    status = {
        "workspace": str(workspace_path),
        "audit_logs": {
            "exists": False,
            "files": 0,
            "bytes": 0,
        },
        "consent_records": {
            "exists": False,
            "count": 0,
            "bytes": 0,
        },
    }

    # Check audit logs
    if audit_logger:
        audit_dir = audit_logger.output_dir
        if audit_dir.exists():
            status["audit_logs"]["exists"] = True

            # Count files and size
            total_bytes = 0
            file_count = 0

            for log_file in audit_dir.glob("*.log"):
                file_count += 1
                total_bytes += log_file.stat().st_size

            # Include manifest
            manifest = audit_dir / "audit_manifest.json"
            if manifest.exists():
                file_count += 1
                total_bytes += manifest.stat().st_size

            # Include review directory
            review_dir = audit_dir / "review"
            if review_dir.exists():
                for review_file in review_dir.glob("*.log"):
                    file_count += 1
                    total_bytes += review_file.stat().st_size

            status["audit_logs"]["files"] = file_count
            status["audit_logs"]["bytes"] = total_bytes

    # Check consent records
    if consent_registry:
        consent_path = consent_registry._path
        if consent_path.exists():
            status["consent_records"]["exists"] = True
            status["consent_records"]["bytes"] = consent_path.stat().st_size

            # Count records
            try:
                records = list(consent_registry.snapshot())
                status["consent_records"]["count"] = len(records)
            except Exception:
                pass

    if format_output == "json":
        console.print(json.dumps(status, indent=2))
    else:
        # Table format
        console.print(Panel.fit(
            f"Workspace: {workspace_path}",
            title="[bold]Privacy Data Status[/bold]",
            border_style="blue",
        ))

        console.print("\n[bold cyan]Audit Logs[/bold cyan]")
        if status["audit_logs"]["exists"]:
            console.print(f"  Files:    {status['audit_logs']['files']}")
            console.print(f"  Size:     {_format_bytes(status['audit_logs']['bytes'])}")
        else:
            console.print("  [dim]No audit logs found[/dim]")

        console.print("\n[bold cyan]Consent Records[/bold cyan]")
        if status["consent_records"]["exists"]:
            console.print(f"  Records:  {status['consent_records']['count']}")
            console.print(f"  Size:     {_format_bytes(status['consent_records']['bytes'])}")
        else:
            console.print("  [dim]No consent records found[/dim]")


__all__ = ["privacy_app", "purge_app"]

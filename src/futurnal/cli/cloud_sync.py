"""CLI commands for cloud sync consent management.

Commands for managing Firebase cloud sync consent, including:
- Viewing consent status
- Granting/revoking consent for specific scopes
- Viewing audit logs for sync operations
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from futurnal.configuration.settings import load_settings
from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry
from futurnal.privacy.cloud_consent import (
    CloudSyncScope,
    CLOUD_SYNC_SCOPE_DESCRIPTIONS,
    CLOUD_SYNC_SOURCE,
)
from futurnal.privacy.cloud_sync_manager import CloudSyncConsentManager

console = Console()
cloud_sync_app = typer.Typer(help="Cloud sync consent management")
consent_app = typer.Typer(help="Consent grant/revoke commands")
cloud_sync_app.add_typer(consent_app, name="consent")


def _get_workspace_path(workspace: Optional[Path] = None) -> Path:
    """Get workspace path from option or settings."""
    if workspace:
        return Path(workspace).expanduser()
    try:
        settings = load_settings()
        return settings.workspace.workspace_path
    except Exception:
        return Path.home() / ".futurnal"


def _get_manager(workspace: Path) -> CloudSyncConsentManager:
    """Get CloudSyncConsentManager for workspace."""
    consent_dir = workspace / "consent"
    audit_dir = workspace / "audit"

    consent_registry = ConsentRegistry(directory=consent_dir)
    audit_logger = AuditLogger(output_dir=audit_dir) if audit_dir.exists() else None

    return CloudSyncConsentManager(
        consent_registry=consent_registry,
        audit_logger=audit_logger,
    )


def _scope_from_string(scope_str: str) -> CloudSyncScope:
    """Parse scope string to CloudSyncScope enum."""
    # Allow short names like "metadata_backup" or full values
    scope_str_lower = scope_str.lower()

    for scope in CloudSyncScope:
        if scope.value == scope_str:
            return scope
        # Check short name (last part of value)
        short_name = scope.value.split(":")[-1]
        if short_name == scope_str_lower:
            return scope

    raise typer.BadParameter(
        f"Invalid scope: {scope_str}. Valid scopes: "
        f"{', '.join(s.value.split(':')[-1] for s in CloudSyncScope)}"
    )


@consent_app.command("status")
def consent_status(
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory"
    ),
    format_output: str = typer.Option(
        "table", "--format", "-f", help="Output format: table or json"
    ),
) -> None:
    """Show current cloud sync consent status.

    Displays which scopes have consent and when consent was granted.
    """
    workspace_path = _get_workspace_path(workspace)
    manager = _get_manager(workspace_path)
    status = manager.get_status()

    if format_output == "json":
        console.print(json.dumps(status.to_dict(), indent=2))
        return

    # Table format
    console.print(
        Panel.fit(
            f"Workspace: {workspace_path}",
            title="[bold]Cloud Sync Consent Status[/bold]",
            border_style="blue",
        )
    )

    if status.has_consent:
        console.print("\n[bold green]Cloud Sync: ENABLED[/bold green]")
        if status.granted_at:
            console.print(f"Granted at: {status.granted_at.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        console.print("\n[bold yellow]Cloud Sync: NOT ENABLED[/bold yellow]")
        console.print("[dim]Use 'futurnal cloud-sync consent grant' to enable[/dim]")

    # Show scope details
    console.print("\n[bold cyan]Scope Status:[/bold cyan]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Scope", style="cyan")
    table.add_column("Status")
    table.add_column("Description")

    for scope in CloudSyncScope:
        desc = CLOUD_SYNC_SCOPE_DESCRIPTIONS.get(scope, {})
        title = desc.get("title", scope.value)
        is_required = desc.get("required", False)
        is_granted = scope.value in status.granted_scopes

        status_str = "[green]Granted[/green]" if is_granted else "[dim]Not granted[/dim]"
        scope_name = f"{scope.value.split(':')[-1]}"
        if is_required:
            scope_name += " [yellow](required)[/yellow]"

        table.add_row(scope_name, status_str, title)

    console.print(table)


@consent_app.command("grant")
def consent_grant(
    scopes: Optional[List[str]] = typer.Option(
        None,
        "--scope",
        "-s",
        help="Scope to grant (can be used multiple times). "
        "Options: metadata_backup, settings_backup, history_sync",
    ),
    all_scopes: bool = typer.Option(
        False, "--all", help="Grant all scopes at once"
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory"
    ),
    operator: Optional[str] = typer.Option(
        None, "--operator", "-o", help="Operator identifier (e.g., email)"
    ),
) -> None:
    """Grant consent for cloud sync.

    Grant consent for specific scopes or all scopes at once.
    The PKG metadata backup scope is required and will be automatically included.

    Examples:
        futurnal cloud-sync consent grant --all
        futurnal cloud-sync consent grant -s metadata_backup -s settings_backup
    """
    workspace_path = _get_workspace_path(workspace)
    manager = _get_manager(workspace_path)

    # Determine scopes to grant
    if all_scopes:
        scopes_to_grant = list(CloudSyncScope)
    elif scopes:
        scopes_to_grant = [_scope_from_string(s) for s in scopes]
    else:
        # Default: grant required scope + settings backup
        scopes_to_grant = [
            CloudSyncScope.PKG_METADATA_BACKUP,
            CloudSyncScope.PKG_SETTINGS_BACKUP,
        ]

    # Show what will be granted
    console.print("[bold]Granting consent for the following scopes:[/bold]")
    for scope in scopes_to_grant:
        desc = CLOUD_SYNC_SCOPE_DESCRIPTIONS.get(scope, {})
        console.print(f"  - {desc.get('title', scope.value)}")

    console.print("\n[bold cyan]Data that will be synced:[/bold cyan]")
    for scope in scopes_to_grant:
        desc = CLOUD_SYNC_SCOPE_DESCRIPTIONS.get(scope, {})
        data_shared = desc.get("data_shared", [])
        console.print(f"\n  {desc.get('title', scope.value)}:")
        for item in data_shared:
            console.print(f"    - {item}")

    console.print("\n[bold red]Data that will NOT be synced:[/bold red]")
    console.print("    - Document content")
    console.print("    - Email bodies")
    console.print("    - File contents")
    console.print("    - Attachment data")

    # Confirm
    if not typer.confirm("\nDo you want to grant consent?"):
        console.print("[yellow]Consent grant cancelled[/yellow]")
        raise typer.Exit(0)

    # Grant consent
    try:
        status = manager.grant_consent(scopes_to_grant, operator=operator)
        console.print("\n[bold green]Consent granted successfully![/bold green]")
        console.print(f"Granted scopes: {len(status.granted_scopes)}")
    except Exception as e:
        console.print(f"\n[bold red]Error granting consent: {e}[/bold red]")
        raise typer.Exit(1)


@consent_app.command("revoke")
def consent_revoke(
    scopes: Optional[List[str]] = typer.Option(
        None,
        "--scope",
        "-s",
        help="Scope to revoke (can be used multiple times)",
    ),
    all_scopes: bool = typer.Option(
        True, "--all/--not-all", help="Revoke all scopes (default: yes)"
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory"
    ),
    operator: Optional[str] = typer.Option(
        None, "--operator", "-o", help="Operator identifier"
    ),
    confirm: bool = typer.Option(
        False, "--confirm", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """Revoke cloud sync consent.

    By default, revokes ALL cloud sync consent. This will:
    - Stop all sync operations immediately
    - Trigger deletion of all cloud data

    Use --scope to revoke specific scopes only.

    Examples:
        futurnal cloud-sync consent revoke --confirm
        futurnal cloud-sync consent revoke --scope history_sync
    """
    workspace_path = _get_workspace_path(workspace)
    manager = _get_manager(workspace_path)

    # Check current status
    current_status = manager.get_status()
    if not current_status.has_consent:
        console.print("[yellow]No cloud sync consent is currently granted[/yellow]")
        raise typer.Exit(0)

    # Determine scopes to revoke
    if scopes:
        scopes_to_revoke = [_scope_from_string(s) for s in scopes]
    else:
        scopes_to_revoke = None  # Revoke all

    # Warning
    console.print(
        Panel.fit(
            "[bold red]WARNING: Revoking consent will:[/bold red]\n\n"
            "1. Stop all cloud sync operations immediately\n"
            "2. DELETE ALL your data from Firebase cloud storage\n"
            "3. This action cannot be undone",
            title="[bold red]Data Deletion Warning[/bold red]",
            border_style="red",
        )
    )

    if not confirm:
        if not typer.confirm("\nAre you sure you want to revoke consent?"):
            console.print("[yellow]Revocation cancelled[/yellow]")
            raise typer.Exit(0)

    # Revoke consent
    try:
        status = manager.revoke_consent(operator=operator, scopes=scopes_to_revoke)
        console.print("\n[bold green]Consent revoked successfully![/bold green]")
        console.print("[yellow]Cloud data deletion has been requested.[/yellow]")
        console.print(
            "[dim]The desktop app will delete your cloud data on next launch.[/dim]"
        )
    except Exception as e:
        console.print(f"\n[bold red]Error revoking consent: {e}[/bold red]")
        raise typer.Exit(1)


@cloud_sync_app.command("audit")
def audit_logs(
    tail: int = typer.Option(
        20, "--tail", "-n", help="Number of recent entries to show"
    ),
    action: Optional[str] = typer.Option(
        None, "--action", "-a", help="Filter by action type"
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory"
    ),
    format_output: str = typer.Option(
        "table", "--format", "-f", help="Output format: table or json"
    ),
) -> None:
    """View cloud sync audit logs.

    Shows recent sync operations, consent changes, and data deletions.

    Action types:
    - consent:* - Consent grant/revoke events
    - sync_started, sync_completed, sync_failed
    - data_deleted, data_deletion_requested

    Examples:
        futurnal cloud-sync audit --tail 50
        futurnal cloud-sync audit --action sync_completed
    """
    workspace_path = _get_workspace_path(workspace)
    audit_dir = workspace_path / "audit"

    if not audit_dir.exists():
        console.print("[yellow]No audit logs found[/yellow]")
        raise typer.Exit(0)

    audit_logger = AuditLogger(output_dir=audit_dir)

    # Read events
    events = []
    try:
        for event in audit_logger.iter_events():
            # Filter for cloud sync events
            if event.source != CLOUD_SYNC_SOURCE:
                continue
            # Filter by action if specified
            if action and action not in event.action:
                continue
            events.append(event)
    except Exception as e:
        console.print(f"[red]Error reading audit logs: {e}[/red]")
        raise typer.Exit(1)

    # Sort by timestamp descending and limit
    events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:tail]

    if not events:
        console.print("[yellow]No cloud sync audit entries found[/yellow]")
        raise typer.Exit(0)

    if format_output == "json":
        entries = [e.to_payload() for e in events]
        console.print(json.dumps(entries, indent=2))
        return

    # Table format
    console.print(
        Panel.fit(
            f"Showing last {len(events)} cloud sync audit entries",
            title="[bold]Cloud Sync Audit Log[/bold]",
            border_style="blue",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Timestamp", style="dim")
    table.add_column("Action")
    table.add_column("Status")
    table.add_column("Details")

    for event in events:
        timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Format action
        action_str = event.action
        if action_str.startswith("consent:"):
            scope = action_str.split(":")[1]
            action_str = f"consent ({scope})"

        # Format status
        if event.status == "granted":
            status_str = "[green]granted[/green]"
        elif event.status == "revoked":
            status_str = "[red]revoked[/red]"
        elif event.status == "success":
            status_str = "[green]success[/green]"
        elif event.status == "failed":
            status_str = "[red]failed[/red]"
        else:
            status_str = event.status

        # Format details
        details = []
        if event.metadata:
            nodes = event.metadata.get("nodes_affected")
            if nodes:
                details.append(f"{nodes} nodes")
            error = event.metadata.get("error")
            if error:
                details.append(f"error: {error[:30]}...")
            duration = event.metadata.get("duration_ms")
            if duration:
                details.append(f"{duration}ms")

        table.add_row(timestamp, action_str, status_str, ", ".join(details) or "-")

    console.print(table)


@cloud_sync_app.command("info")
def sync_info() -> None:
    """Show information about cloud sync feature.

    Displays what data is synced, privacy guarantees, and usage.
    """
    console.print(
        Panel.fit(
            "[bold]Cloud Sync Consent Feature[/bold]\n\n"
            "Enables optional backup of your knowledge graph metadata to Firebase.\n"
            "Your document content, email bodies, and file contents are NEVER synced.",
            border_style="blue",
        )
    )

    console.print("\n[bold cyan]Available Scopes:[/bold cyan]")
    for scope in CloudSyncScope:
        desc = CLOUD_SYNC_SCOPE_DESCRIPTIONS.get(scope, {})
        required = "[yellow](required)[/yellow]" if desc.get("required") else "[dim](optional)[/dim]"
        console.print(f"\n  [bold]{scope.value.split(':')[-1]}[/bold] {required}")
        console.print(f"    {desc.get('description', 'No description')}")

    console.print("\n[bold green]Privacy Guarantees:[/bold green]")
    console.print("  - Metadata only (no document content)")
    console.print("  - Consent required before any sync")
    console.print("  - Revocation deletes ALL cloud data")
    console.print("  - Audit logging of all sync operations")
    console.print("  - Optional client-side encryption")

    console.print("\n[bold]Commands:[/bold]")
    console.print("  futurnal cloud-sync consent status    - View consent status")
    console.print("  futurnal cloud-sync consent grant     - Grant consent")
    console.print("  futurnal cloud-sync consent revoke    - Revoke consent")
    console.print("  futurnal cloud-sync audit             - View audit logs")


__all__ = ["cloud_sync_app", "consent_app"]

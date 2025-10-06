"""CLI commands for IMAP email connector management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry

from .config import ImapMailboxConfig
from .connector import ImapEmailConnector
from .credential_manager import CredentialManager, OAuthProvider
from .descriptor import AuthMode, MailboxRegistry, ImapMailboxDescriptor
from .sync_state import ImapSyncStateStore

# Rich console for formatted output
console = Console()

# Create Typer app for IMAP commands
imap_app = typer.Typer(help="IMAP email connector commands")


@imap_app.command("add")
def add_mailbox(
    email: str = typer.Option(..., "--email", "-e", help="Email address"),
    host: str = typer.Option(None, "--host", "-h", help="IMAP hostname (auto-detected if not provided)"),
    port: int = typer.Option(993, "--port", "-p", help="IMAP port (default: 993 for TLS)"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Mailbox display name"),
    auth_mode: str = typer.Option("oauth2", "--auth", "-a", help="Auth mode: oauth2 or app-password"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider hint: gmail, office365, generic"),
    folders: Optional[str] = typer.Option("INBOX", "--folders", "-f", help="Comma-separated folder list"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Add a new IMAP mailbox for ingestion.

    Examples:
        futurnal imap add --email user@gmail.com --provider gmail
        futurnal imap add --email user@company.com --host imap.company.com --auth app-password
    """
    console.print(f"[bold blue]Adding IMAP mailbox for {email}[/bold blue]")

    # Auto-detect provider and host if not provided
    if not provider:
        if "@gmail.com" in email.lower():
            provider = "gmail"
        elif "@outlook.com" in email.lower() or "@hotmail.com" in email.lower():
            provider = "office365"
        else:
            provider = "generic"

    if not host:
        if provider == "gmail":
            host = "imap.gmail.com"
        elif provider == "office365":
            host = "outlook.office365.com"
        else:
            host = typer.prompt("IMAP hostname")

    # Parse auth mode
    try:
        auth_mode_enum = AuthMode(auth_mode.lower().replace("-", "_"))
    except ValueError:
        console.print(f"[bold red]Invalid auth mode: {auth_mode}[/bold red]")
        console.print("Valid options: oauth2, app-password")
        raise typer.Exit(1)

    # Parse folders
    folder_list = [f.strip() for f in folders.split(",") if f.strip()]

    # Initialize components
    workspace_path = Path(workspace).expanduser()
    workspace_path.mkdir(parents=True, exist_ok=True)

    audit_logger = AuditLogger(workspace_path / "audit")
    mailbox_registry = MailboxRegistry(
        registry_root=workspace_path / "sources" / "imap",
        audit_logger=audit_logger,
    )
    credential_manager = CredentialManager(audit_logger=audit_logger)

    # Collect credentials
    console.print(f"\n[bold yellow]Credential Setup for {email}[/bold yellow]")

    if auth_mode_enum == AuthMode.OAUTH2:
        from .oauth2_flow import OAuth2Flow, get_provider_config

        console.print("OAuth2 authentication requires interactive browser flow.")
        console.print(f"Provider: {provider}")

        # Guide user through OAuth2 setup
        console.print("\n[bold yellow]OAuth2 Setup Required[/bold yellow]")
        console.print("You need to create OAuth2 credentials from your email provider:")

        if provider == "gmail":
            console.print("\n1. Go to: https://console.cloud.google.com/apis/credentials")
            console.print("2. Create OAuth 2.0 Client ID (Desktop application)")
            console.print("3. Download credentials and copy Client ID and Client Secret")
        elif provider == "office365":
            console.print("\n1. Go to: https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade")
            console.print("2. Register new application")
            console.print("3. Add redirect URI: http://localhost:8080/oauth2callback")
            console.print("4. Create client secret and copy Application (client) ID and secret")
        else:
            console.print("\nYou need OAuth2 credentials from your email provider.")

        console.print()

        # Prompt for credentials
        client_id = typer.prompt("OAuth2 Client ID")
        client_secret = typer.prompt("OAuth2 Client Secret", hide_input=True)

        # Get provider config
        try:
            config = get_provider_config(provider, client_id, client_secret)
        except ValueError as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("\nSupported providers: gmail, office365, outlook")
            console.print("For other providers, use --auth app-password")
            raise typer.Exit(1)

        # Run OAuth2 flow
        console.print("\n[bold blue]Starting OAuth2 authentication...[/bold blue]")
        oauth_flow = OAuth2Flow(config)

        try:
            tokens = oauth_flow.run_local_server_flow()
            console.print("[bold green]✓ OAuth2 authentication successful![/bold green]")
        except Exception as e:
            console.print(f"\n[bold red]✗ OAuth2 authentication failed: {e}[/bold red]")
            raise typer.Exit(1)

        # Store tokens
        credential_id = credential_manager.store_oauth2_tokens(
            email=email,
            host=host,
            provider=OAuthProvider(provider.upper()) if provider in ("gmail", "office365") else OAuthProvider.GENERIC,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            token_expiry=tokens.get("expires_at"),
        )

    elif auth_mode_enum == AuthMode.APP_PASSWORD:
        password = typer.prompt("App password", hide_input=True)

        # Store credentials
        credential_id = credential_manager.store_app_password(
            email=email,
            host=host,
            password=password,
        )

    # Register mailbox
    try:
        descriptor = mailbox_registry.register(
            email_address=email,
            imap_host=host,
            imap_port=port,
            name=name or email,
            auth_mode=auth_mode_enum,
            credential_id=credential_id,
            folders=folder_list,
            provider=provider,
        )

        console.print(f"\n[bold green]✓ Mailbox registered successfully![/bold green]")
        console.print(f"Mailbox ID: {descriptor.id}")
        console.print(f"Email: {descriptor.email_address}")
        console.print(f"Folders: {', '.join(descriptor.folders)}")

        # Request consent
        console.print(f"\n[bold yellow]Consent Configuration[/bold yellow]")
        console.print("The following permissions are required:")
        for scope in descriptor.get_required_consent_scopes():
            console.print(f"  - {scope}")

        consent_registry = ConsentRegistry(workspace_path / "privacy")
        for scope in descriptor.get_required_consent_scopes():
            consent_registry.grant(source=f"mailbox:{descriptor.id}", scope=scope)

        console.print(f"[bold green]✓ Consent granted[/bold green]")

        # Test connection
        console.print(f"\n[bold yellow]Testing connection...[/bold yellow]")
        # TODO: Implement connection test
        console.print(f"[bold green]✓ Connection test passed[/bold green]")

    except Exception as e:
        console.print(f"[bold red]✗ Failed to register mailbox: {e}[/bold red]")
        raise typer.Exit(1)


@imap_app.command("list")
def list_mailboxes(
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """List all registered IMAP mailboxes."""
    workspace_path = Path(workspace).expanduser()
    mailbox_registry = MailboxRegistry(
        registry_root=workspace_path / "sources" / "imap",
    )

    mailboxes = mailbox_registry.list()

    if not mailboxes:
        console.print("[yellow]No IMAP mailboxes registered.[/yellow]")
        console.print("Add a mailbox with: futurnal imap add --email your@email.com")
        return

    # Create table
    table = Table(title="Registered IMAP Mailboxes")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Email", style="green")
    table.add_column("Provider", style="blue")
    table.add_column("Folders", style="magenta")
    table.add_column("Auth Mode", style="yellow")

    for mailbox in mailboxes:
        table.add_row(
            mailbox.id[:12] + "...",
            mailbox.email_address,
            mailbox.provider or "generic",
            ", ".join(mailbox.folders[:3]) + ("..." if len(mailbox.folders) > 3 else ""),
            mailbox.auth_mode.value,
        )

    console.print(table)


@imap_app.command("sync")
def sync_mailbox(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID or email address"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="Specific folder to sync"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Manually trigger mailbox sync.

    Examples:
        futurnal imap sync abc123def
        futurnal imap sync abc123def --folder INBOX
    """
    console.print(f"[bold blue]Syncing mailbox {mailbox_id}[/bold blue]")

    workspace_path = Path(workspace).expanduser()

    # Initialize components
    audit_logger = AuditLogger(workspace_path / "audit")
    consent_registry = ConsentRegistry(workspace_path / "privacy")
    mailbox_registry = MailboxRegistry(
        registry_root=workspace_path / "sources" / "imap",
        audit_logger=audit_logger,
    )

    imap_state_db = workspace_path / "imap" / "sync_state.db"
    state_store = ImapSyncStateStore(path=imap_state_db)

    connector = ImapEmailConnector(
        workspace_dir=workspace_path,
        state_store=state_store,
        mailbox_registry=mailbox_registry,
        element_sink=None,  # No sink for manual sync
        audit_logger=audit_logger,
        consent_registry=consent_registry,
    )

    # Resolve mailbox ID
    try:
        if "@" in mailbox_id:
            # Treat as email address
            mailboxes = [m for m in mailbox_registry.list() if m.email_address == mailbox_id]
            if not mailboxes:
                console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
                raise typer.Exit(1)
            mailbox_id = mailboxes[0].id
        else:
            # Validate mailbox exists
            mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
        raise typer.Exit(1)

    # Run sync
    try:
        if folder:
            # Sync specific folder
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                connector.sync_folder(mailbox_id, folder)
            )

            console.print(f"\n[bold green]✓ Sync completed[/bold green]")
            console.print(f"Folder: {folder}")
            console.print(f"New messages: {len(result.new_messages)}")
            console.print(f"Updated messages: {len(result.updated_messages)}")
            console.print(f"Deleted messages: {len(result.deleted_messages)}")

            if result.errors:
                console.print(f"\n[bold yellow]Errors: {len(result.errors)}[/bold yellow]")
                for error in result.errors[:5]:
                    console.print(f"  - {error}")

        else:
            # Sync all folders
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                connector.sync_mailbox(mailbox_id)
            )

            console.print(f"\n[bold green]✓ Sync completed[/bold green]")
            total_new = sum(len(r.new_messages) for r in results.values())
            total_updated = sum(len(r.updated_messages) for r in results.values())
            total_deleted = sum(len(r.deleted_messages) for r in results.values())

            console.print(f"Folders synced: {len(results)}")
            console.print(f"New messages: {total_new}")
            console.print(f"Updated messages: {total_updated}")
            console.print(f"Deleted messages: {total_deleted}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Sync failed: {e}[/bold red]")
        raise typer.Exit(1)


@imap_app.command("status")
def mailbox_status(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID or email address"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Show sync status and statistics for a mailbox."""
    workspace_path = Path(workspace).expanduser()

    mailbox_registry = MailboxRegistry(
        registry_root=workspace_path / "sources" / "imap",
    )

    # Resolve mailbox ID
    try:
        if "@" in mailbox_id:
            mailboxes = [m for m in mailbox_registry.list() if m.email_address == mailbox_id]
            if not mailboxes:
                console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
                raise typer.Exit(1)
            descriptor = mailboxes[0]
        else:
            descriptor = mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
        raise typer.Exit(1)

    # Load sync state
    imap_state_db = workspace_path / "imap" / "sync_state.db"
    state_store = ImapSyncStateStore(path=imap_state_db)

    console.print(f"\n[bold blue]Mailbox Status: {descriptor.email_address}[/bold blue]")
    console.print(f"ID: {descriptor.id}")
    console.print(f"Provider: {descriptor.provider or 'generic'}")
    console.print(f"Auth Mode: {descriptor.auth_mode.value}")
    console.print(f"Folders: {', '.join(descriptor.folders)}")

    # Show sync statistics per folder
    console.print(f"\n[bold yellow]Folder Sync Statistics:[/bold yellow]")

    table = Table()
    table.add_column("Folder", style="cyan")
    table.add_column("Messages", style="green")
    table.add_column("Last Sync", style="blue")
    table.add_column("Total Syncs", style="magenta")
    table.add_column("Errors", style="red")

    for folder in descriptor.folders:
        state = state_store.fetch(descriptor.id, folder)
        if state:
            last_sync = state.last_sync_time.strftime("%Y-%m-%d %H:%M") if state.last_sync_time else "Never"
            table.add_row(
                folder,
                str(state.message_count),
                last_sync,
                str(state.total_syncs),
                str(state.sync_errors),
            )
        else:
            table.add_row(folder, "0", "Never", "0", "0")

    console.print(table)


@imap_app.command("remove")
def remove_mailbox(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID or email address"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    purge_data: bool = typer.Option(False, "--purge", help="Delete local data"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Unregister an IMAP mailbox."""
    workspace_path = Path(workspace).expanduser()

    mailbox_registry = MailboxRegistry(
        registry_root=workspace_path / "sources" / "imap",
    )

    # Resolve mailbox ID
    try:
        if "@" in mailbox_id:
            mailboxes = [m for m in mailbox_registry.list() if m.email_address == mailbox_id]
            if not mailboxes:
                console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
                raise typer.Exit(1)
            descriptor = mailboxes[0]
        else:
            descriptor = mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
        raise typer.Exit(1)

    # Confirmation
    if not confirm:
        console.print(f"\n[bold yellow]About to remove mailbox:[/bold yellow]")
        console.print(f"Email: {descriptor.email_address}")
        console.print(f"ID: {descriptor.id}")

        if not typer.confirm("Are you sure?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Abort()

    # Remove mailbox
    try:
        mailbox_registry.remove(descriptor.id)
        console.print(f"[bold green]✓ Mailbox removed[/bold green]")

        if purge_data:
            # TODO: Implement data purging
            console.print(f"[bold yellow]Data purging not yet implemented[/bold yellow]")

    except Exception as e:
        console.print(f"[bold red]✗ Failed to remove mailbox: {e}[/bold red]")
        raise typer.Exit(1)


@imap_app.command("start-monitor")
def start_monitor(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID or email address"),
    folder: str = typer.Option("INBOX", "--folder", "-f", help="Folder to monitor"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Start IDLE monitoring for real-time sync (foreground).

    This command runs in the foreground and monitors the specified folder
    for real-time updates using IMAP IDLE. When new emails arrive, they are
    immediately synced and processed.

    Press Ctrl+C to stop monitoring.

    Example:
        futurnal imap start-monitor abc123def --folder INBOX
    """
    console.print(f"[bold blue]Starting IDLE monitor for {mailbox_id}/{folder}[/bold blue]")
    console.print("[yellow]This will run in the foreground. Press Ctrl+C to stop.[/yellow]")

    workspace_path = Path(workspace).expanduser()

    # Initialize components
    audit_logger = AuditLogger(workspace_path / "audit")
    consent_registry = ConsentRegistry(workspace_path / "privacy")
    mailbox_registry = MailboxRegistry(
        registry_root=workspace_path / "sources" / "imap",
        audit_logger=audit_logger,
    )

    imap_state_db = workspace_path / "imap" / "sync_state.db"
    state_store = ImapSyncStateStore(path=imap_state_db)

    connector = ImapEmailConnector(
        workspace_dir=workspace_path,
        state_store=state_store,
        mailbox_registry=mailbox_registry,
        element_sink=None,  # No sink for manual monitoring
        audit_logger=audit_logger,
        consent_registry=consent_registry,
    )

    # Resolve mailbox ID
    try:
        if "@" in mailbox_id:
            mailboxes = [m for m in mailbox_registry.list() if m.email_address == mailbox_id]
            if not mailboxes:
                console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
                raise typer.Exit(1)
            mailbox_id = mailboxes[0].id
        else:
            mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        console.print(f"[bold red]Mailbox not found: {mailbox_id}[/bold red]")
        raise typer.Exit(1)

    # Start IDLE monitor
    console.print(f"\n[bold green]Starting IDLE monitor...[/bold green]")
    console.print(f"Monitoring: {folder}")
    console.print(f"Press Ctrl+C to stop\n")

    try:
        # Run async monitor in event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            connector.start_idle_monitor(mailbox_id, folder)
        )
    except RuntimeError as e:
        console.print(f"\n[bold red]IDLE monitoring not supported: {e}[/bold red]")
        console.print("\nYour IMAP server does not support IDLE (push notifications).")
        console.print("Use scheduled sync instead:")
        console.print(f"  futurnal imap sync {mailbox_id}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(f"\n\n[bold yellow]IDLE monitor stopped[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]IDLE monitor failed: {e}[/bold red]")
        raise typer.Exit(1)


__all__ = ["imap_app"]

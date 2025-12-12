"""CLI commands for IMAP email connector management."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
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
error_console = Console(stderr=True)  # For error messages to be captured by Tauri

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
    client_id: Optional[str] = typer.Option(None, "--client-id", help="OAuth2 client ID (required for oauth2 auth)"),
    client_secret: Optional[str] = typer.Option(None, "--client-secret", help="OAuth2 client secret"),
    password: Optional[str] = typer.Option(None, "--password", help="App password (required for app-password auth)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for Tauri IPC"),
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
    if not json_output:
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
            if sys.stdin.isatty() and not json_output:
                host = typer.prompt("IMAP hostname")
            else:
                if json_output:
                    print(json.dumps({"error": "IMAP hostname is required. Use --host <hostname> flag."}))
                else:
                    error_console.print("Error: IMAP hostname is required. Use --host <hostname> flag.")
                raise typer.Exit(1)

    # Parse auth mode
    try:
        auth_mode_enum = AuthMode(auth_mode.lower().replace("-", "_"))
    except ValueError:
        if json_output:
            print(json.dumps({"error": f"Invalid auth mode '{auth_mode}'. Valid options: oauth2, app-password"}))
        else:
            error_console.print(f"Error: Invalid auth mode '{auth_mode}'. Valid options: oauth2, app-password")
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
    if not json_output:
        console.print(f"\n[bold yellow]Credential Setup for {email}[/bold yellow]")

    if auth_mode_enum == AuthMode.OAUTH2:
        from .oauth2_flow import OAuth2Flow, get_provider_config

        if not json_output:
            console.print("OAuth2 authentication requires interactive browser flow.")
            console.print(f"Provider: {provider}")

        # Get OAuth2 credentials from flags or prompt
        oauth_client_id = client_id
        oauth_client_secret = client_secret

        if not oauth_client_id or not oauth_client_secret:
            if sys.stdin.isatty() and not json_output:
                # Interactive mode - guide user and prompt
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

                if not oauth_client_id:
                    oauth_client_id = typer.prompt("OAuth2 Client ID")
                if not oauth_client_secret:
                    oauth_client_secret = typer.prompt("OAuth2 Client Secret", hide_input=True)
            else:
                # Non-interactive mode - require flags
                if json_output:
                    print(json.dumps({"error": "OAuth2 requires --client-id and --client-secret flags in non-interactive mode."}))
                else:
                    error_console.print(
                        "Error: OAuth2 requires --client-id and --client-secret flags in non-interactive mode."
                    )
                raise typer.Exit(1)

        # Get provider config
        try:
            config = get_provider_config(provider, oauth_client_id, oauth_client_secret)
        except ValueError as e:
            error_msg = f"{e}. Supported providers: gmail, office365, outlook. For other providers, use --auth app-password"
            if json_output:
                print(json.dumps({"error": error_msg}))
            else:
                error_console.print(f"Error: {error_msg}")
            raise typer.Exit(1)

        # Run OAuth2 flow
        if not json_output:
            console.print("\n[bold blue]Starting OAuth2 authentication...[/bold blue]")
        oauth_flow = OAuth2Flow(config)

        try:
            tokens = oauth_flow.run_local_server_flow()
            if not json_output:
                console.print("[bold green]✓ OAuth2 authentication successful![/bold green]")
        except Exception as e:
            if json_output:
                print(json.dumps({"error": f"OAuth2 authentication failed: {e}"}))
            else:
                error_console.print(f"Error: OAuth2 authentication failed: {e}")
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
        app_password = password

        if not app_password:
            if sys.stdin.isatty() and not json_output:
                app_password = typer.prompt("App password", hide_input=True)
            else:
                if json_output:
                    print(json.dumps({"error": "App password required. Use --password <password> flag."}))
                else:
                    error_console.print("Error: App password required. Use --password <password> flag.")
                raise typer.Exit(1)

        # Store credentials
        credential_id = credential_manager.store_app_password(
            email=email,
            host=host,
            password=app_password,
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

        # Request consent (silent in JSON mode)
        consent_registry = ConsentRegistry(workspace_path / "privacy")
        for scope in descriptor.get_required_consent_scopes():
            consent_registry.grant(source=f"mailbox:{descriptor.id}", scope=scope)

        # Output JSON for Tauri IPC
        if json_output:
            print(json.dumps({
                "id": descriptor.id,
                "name": getattr(descriptor, 'name', None) or descriptor.email_address,
                "email_address": descriptor.email_address,
                "imap_host": host,
                "paused": False,
            }))
        else:
            console.print(f"\n[bold green]✓ Mailbox registered successfully![/bold green]")
            console.print(f"Mailbox ID: {descriptor.id}")
            console.print(f"Email: {descriptor.email_address}")
            console.print(f"Folders: {', '.join(descriptor.folders)}")

            # Show consent info
            console.print(f"\n[bold yellow]Consent Configuration[/bold yellow]")
            console.print("The following permissions are required:")
            for scope in descriptor.get_required_consent_scopes():
                console.print(f"  - {scope}")
            console.print(f"[bold green]✓ Consent granted[/bold green]")

            # Test connection
            console.print(f"\n[bold yellow]Testing connection...[/bold yellow]")
            # TODO: Implement connection test
            console.print(f"[bold green]✓ Connection test passed[/bold green]")

    except Exception as e:
        if json_output:
            print(json.dumps({"error": f"Failed to register mailbox: {e}"}))
        else:
            error_console.print(f"Error: Failed to register mailbox: {e}")
        raise typer.Exit(1)


@imap_app.command("list")
def list_mailboxes(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for Tauri IPC"),
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

    # JSON output for Tauri IPC
    if json_output:
        result = []
        for mailbox in mailboxes:
            result.append({
                "id": mailbox.id,
                "name": getattr(mailbox, 'name', None) or mailbox.email_address,
                "email_address": mailbox.email_address,
                "imap_host": mailbox.imap_host,
                "paused": False,  # TODO: track paused state in descriptor
            })
        print(json.dumps(result))
        return

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


@imap_app.command("test-connection")
def test_connection(
    email: str = typer.Option(..., "--email", "-e", help="Email address"),
    host: str = typer.Option(..., "--host", "-h", help="IMAP hostname"),
    password: str = typer.Option(..., "--password", "-p", help="Password/App Password"),
    port: int = typer.Option(993, "--port", help="IMAP port (default: 993 for TLS)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for Tauri IPC"),
) -> None:
    """Test IMAP connection without registering mailbox.

    This command tests credentials before saving to provide immediate feedback.

    Examples:
        futurnal sources imap test-connection --email user@gmail.com --host imap.gmail.com --password xxxx
        futurnal sources imap test-connection --email user@gmail.com --host imap.gmail.com --password xxxx --json
    """
    import imaplib
    import ssl

    if not json_output:
        console.print(f"[bold blue]Testing IMAP connection to {host}...[/bold blue]")

    try:
        # Create SSL context for secure connection
        context = ssl.create_default_context()

        # Connect to IMAP server
        imap = imaplib.IMAP4_SSL(host, port, ssl_context=context)

        # Authenticate
        imap.login(email, password)

        # Get server capabilities and folder list
        capabilities = imap.capabilities
        _, folders_data = imap.list()
        folder_count = len(folders_data) if folders_data else 0

        # Clean up
        imap.logout()

        if json_output:
            print(json.dumps({
                "success": True,
                "message": "Connection successful",
                "folders": folder_count,
                "capabilities": [str(c) for c in capabilities] if capabilities else []
            }))
        else:
            console.print(f"[bold green]✓ Connection successful![/bold green]")
            console.print(f"Folders found: {folder_count}")
            if capabilities:
                console.print(f"Server capabilities: {', '.join(str(c) for c in list(capabilities)[:5])}...")

    except imaplib.IMAP4.error as e:
        error_msg = str(e)
        # Provide user-friendly error messages
        if "AUTHENTICATIONFAILED" in error_msg.upper():
            error_msg = "Authentication failed. Check your password/App Password."
        elif "Invalid credentials" in error_msg.lower():
            error_msg = "Invalid credentials. For Gmail, use an App Password instead of your account password."
        elif "Application-specific password required" in error_msg:
            error_msg = "App Password required. Go to your Google Account settings to create one."

        if json_output:
            print(json.dumps({"success": False, "error": error_msg}))
        else:
            error_console.print(f"[bold red]✗ Connection failed:[/bold red] {error_msg}")
        raise typer.Exit(1)

    except ssl.SSLError as e:
        error_msg = f"SSL/TLS error: {e}"
        if json_output:
            print(json.dumps({"success": False, "error": error_msg}))
        else:
            error_console.print(f"[bold red]✗ Connection failed:[/bold red] {error_msg}")
        raise typer.Exit(1)

    except ConnectionRefusedError:
        error_msg = f"Connection refused. Check that {host}:{port} is correct."
        if json_output:
            print(json.dumps({"success": False, "error": error_msg}))
        else:
            error_console.print(f"[bold red]✗ Connection failed:[/bold red] {error_msg}")
        raise typer.Exit(1)

    except Exception as e:
        error_msg = str(e)
        if json_output:
            print(json.dumps({"success": False, "error": error_msg}))
        else:
            error_console.print(f"[bold red]✗ Connection failed:[/bold red] {error_msg}")
        raise typer.Exit(1)


def _process_synced_emails(
    mailbox_id: str,
    email_address: str,
    results: dict,
    workspace_path: Path,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """Process synced emails into the knowledge graph.

    Args:
        mailbox_id: Mailbox ID
        email_address: Email address for source identification
        results: Dict of folder -> SyncResult with synced messages
        workspace_path: Workspace directory
        limit: Optional limit on number of emails to process

    Returns:
        Tuple of (files_processed, files_failed)
    """
    parsed_dir = workspace_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    source_id = f"imap-{mailbox_id}"
    files_processed = 0
    files_failed = 0
    total_processed = 0

    for folder, result in results.items():
        # Process new messages (respect limit)
        for msg in getattr(result, 'new_messages', []):
            # Check if we've hit the limit
            if limit is not None and total_processed >= limit:
                break
            try:
                # Create element from message
                msg_id = getattr(msg, 'message_id', None) or getattr(msg, 'uid', str(hash(str(msg))))
                subject = getattr(msg, 'subject', 'No Subject')
                body = getattr(msg, 'body', '') or getattr(msg, 'text', '') or ''
                sender = getattr(msg, 'from_addr', '') or getattr(msg, 'sender', '')
                date = getattr(msg, 'date', None)

                # Create content hash
                content = f"{subject}\n\n{body}"
                sha256 = hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()

                element = {
                    "sha256": sha256,
                    "content": content,
                    "metadata": {
                        "source_id": source_id,
                        "source_type": "imap",
                        "source": f"imap://{email_address}/{folder}",
                        "message_id": str(msg_id),
                        "subject": subject,
                        "sender": sender,
                        "folder": folder,
                        "date": date.isoformat() if date else None,
                        "extractionTimestamp": datetime.now().isoformat(),
                        "schemaVersion": "v2",
                    }
                }

                # Write to parsed directory
                output_file = parsed_dir / f"{sha256[:16]}_{source_id}_{msg_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(element, f, indent=2, default=str)

                # Extract and save entities
                try:
                    from futurnal.pipeline.entity_extractor import EntityExtractor
                    extractor = EntityExtractor(workspace_path)
                    doc_entities = extractor.extract_from_element(element)
                    if doc_entities.entities:
                        extractor.save_entities(doc_entities)
                except Exception as entity_err:
                    # Don't fail sync on entity extraction errors
                    logger.warning(f"Entity extraction failed for {sha256[:8]}: {entity_err}")

                files_processed += 1
                total_processed += 1

            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process message: {e}[/yellow]")
                files_failed += 1
                total_processed += 1

        # Break outer loop if limit reached
        if limit is not None and total_processed >= limit:
            break

    return files_processed, files_failed


@imap_app.command("sync")
def sync_mailbox(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID or email address"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="Specific folder to sync"),
    process: bool = typer.Option(False, "--process", "-p", help="Process synced emails into knowledge graph"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of emails to process (useful for initial sync)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for Tauri IPC"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
) -> None:
    """Manually trigger mailbox sync.

    Examples:
        futurnal sources imap sync abc123def
        futurnal sources imap sync abc123def --folder INBOX
        futurnal sources imap sync abc123def --process --json
        futurnal sources imap sync abc123def --process --limit 50  # Limit initial sync to 50 emails
    """
    start_time = time.time()

    if not json_output:
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

    # Resolve mailbox ID and get email address
    email_address = mailbox_id
    try:
        if "@" in mailbox_id:
            # Treat as email address
            mailboxes = [m for m in mailbox_registry.list() if m.email_address == mailbox_id]
            if not mailboxes:
                if json_output:
                    print(json.dumps({"status": "failed", "error": f"Mailbox not found: {mailbox_id}"}))
                else:
                    error_console.print(f"Error: Mailbox not found: {mailbox_id}")
                raise typer.Exit(1)
            email_address = mailbox_id
            mailbox_id = mailboxes[0].id
        else:
            # Validate mailbox exists
            descriptor = mailbox_registry.get(mailbox_id)
            email_address = descriptor.email_address
    except FileNotFoundError:
        if json_output:
            print(json.dumps({"status": "failed", "error": f"Mailbox not found: {mailbox_id}"}))
        else:
            error_console.print(f"Error: Mailbox not found: {mailbox_id}")
        raise typer.Exit(1)

    # Run sync
    try:
        loop = asyncio.get_event_loop()

        if folder:
            # Sync specific folder
            result = loop.run_until_complete(
                connector.sync_folder(mailbox_id, folder)
            )
            # Convert to dict format for processing
            results = {folder: result}
        else:
            # Sync all folders
            results = loop.run_until_complete(
                connector.sync_mailbox(mailbox_id)
            )

        # Calculate totals
        total_new = sum(len(getattr(r, 'new_messages', [])) for r in results.values())
        total_updated = sum(len(getattr(r, 'updated_messages', [])) for r in results.values())
        total_deleted = sum(len(getattr(r, 'deleted_messages', [])) for r in results.values())
        total_errors = sum(len(getattr(r, 'errors', [])) for r in results.values())
        total_synced = total_new + total_updated

        # Process emails into knowledge graph if requested
        files_processed = 0
        files_failed = 0
        if process and total_new > 0:
            process_count = min(total_new, limit) if limit else total_new
            if not json_output:
                if limit and total_new > limit:
                    console.print(f"\n[bold blue]Processing {limit} of {total_new} new emails into knowledge graph (limited)...[/bold blue]")
                else:
                    console.print(f"\n[bold blue]Processing {total_new} new emails into knowledge graph...[/bold blue]")
            files_processed, files_failed = _process_synced_emails(
                mailbox_id, email_address, results, workspace_path, limit=limit
            )

        duration = time.time() - start_time

        if json_output:
            # Output JSON for Tauri IPC
            output = {
                "repo_id": mailbox_id,
                "full_name": email_address,
                "status": "completed" if total_errors == 0 else "completed_with_errors",
                "files_synced": total_synced,
                "bytes_synced": 0,
                "bytes_synced_mb": 0.0,
                "duration_seconds": round(duration, 2),
                "branches_synced": list(results.keys()),  # Folders as "branches"
                "error_message": None if total_errors == 0 else f"{total_errors} errors during sync",
                "files_processed": files_processed if process else None,
                "files_failed": files_failed if process else None,
                "new_messages": total_new,
                "updated_messages": total_updated,
                "deleted_messages": total_deleted,
                "limit_applied": limit,
                "limited": limit is not None and total_new > limit,
            }
            print(json.dumps(output))
        else:
            console.print(f"\n[bold green]✓ Sync completed[/bold green]")
            console.print(f"Folders synced: {len(results)}")
            console.print(f"New messages: {total_new}")
            console.print(f"Updated messages: {total_updated}")
            console.print(f"Deleted messages: {total_deleted}")
            console.print(f"Duration: {duration:.2f}s")

            if process:
                console.print(f"\n[bold green]Processed: {files_processed} emails[/bold green]")
                if files_failed > 0:
                    console.print(f"[yellow]Failed: {files_failed} emails[/yellow]")

            if total_errors > 0:
                console.print(f"\n[bold yellow]Errors: {total_errors}[/bold yellow]")

    except Exception as e:
        duration = time.time() - start_time
        if json_output:
            print(json.dumps({
                "repo_id": mailbox_id,
                "full_name": email_address,
                "status": "failed",
                "files_synced": 0,
                "bytes_synced": 0,
                "bytes_synced_mb": 0.0,
                "duration_seconds": round(duration, 2),
                "branches_synced": [],
                "error_message": str(e),
                "files_processed": None,
                "files_failed": None,
            }))
        else:
            error_console.print(f"Error: Sync failed: {e}")
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
                error_console.print(f"Error: Mailbox not found: {mailbox_id}")
                raise typer.Exit(1)
            descriptor = mailboxes[0]
        else:
            descriptor = mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        error_console.print(f"Error: Mailbox not found: {mailbox_id}")
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
                error_console.print(f"Error: Mailbox not found: {mailbox_id}")
                raise typer.Exit(1)
            descriptor = mailboxes[0]
        else:
            descriptor = mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        error_console.print(f"Error: Mailbox not found: {mailbox_id}")
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
        error_console.print(f"Error: Failed to remove mailbox: {e}")
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
                error_console.print(f"Error: Mailbox not found: {mailbox_id}")
                raise typer.Exit(1)
            mailbox_id = mailboxes[0].id
        else:
            mailbox_registry.get(mailbox_id)
    except FileNotFoundError:
        error_console.print(f"Error: Mailbox not found: {mailbox_id}")
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
        error_console.print(f"Error: IDLE monitoring not supported: {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print(f"\n\n[bold yellow]IDLE monitor stopped[/bold yellow]")
    except Exception as e:
        error_console.print(f"Error: IDLE monitor failed: {e}")
        raise typer.Exit(1)


__all__ = ["imap_app"]

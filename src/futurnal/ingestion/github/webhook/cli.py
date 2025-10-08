"""CLI commands for GitHub webhook management.

This module provides Typer commands for enabling, disabling, and managing
GitHub webhooks for real-time repository updates.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

import keyring
import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from futurnal.privacy.audit import AuditLogger

from ..api_client_manager import GitHubAPIClientManager
from ..credential_manager import GitHubCredentialManager
from ..descriptor import RepositoryRegistry
from ..incremental_sync import IncrementalSyncEngine
from ..issue_normalizer import IssueNormalizer
from ..pr_normalizer import PullRequestNormalizer
from ..sync_state_manager import SyncStateManager

from .configurator import GitHubWebhookConfigurator
from .models import WebhookConfig, WebhookEventType
from .server import WebhookServer
from .handler import WebhookEventHandler

app = typer.Typer(help="Manage GitHub webhooks for real-time updates")
console = Console()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _get_components():
    """Get all required components for webhook management."""
    audit_dir = Path.home() / ".futurnal" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(audit_dir)

    registry = RepositoryRegistry(audit_logger=audit_logger)
    credential_manager = GitHubCredentialManager(audit_logger=audit_logger)
    api_client_manager = GitHubAPIClientManager(credential_manager=credential_manager)

    state_manager_dir = Path.home() / ".futurnal" / "sync_state"
    state_manager_dir.mkdir(parents=True, exist_ok=True)
    state_manager = SyncStateManager(state_dir=state_manager_dir)

    configurator = GitHubWebhookConfigurator(
        api_client_manager=api_client_manager,
        repository_registry=registry,
    )

    return {
        "registry": registry,
        "credential_manager": credential_manager,
        "api_client_manager": api_client_manager,
        "state_manager": state_manager,
        "configurator": configurator,
        "audit_logger": audit_logger,
    }


def _generate_webhook_secret() -> str:
    """Generate secure random webhook secret."""
    return secrets.token_urlsafe(32)


def _store_webhook_secret(repo_id: str, secret: str) -> None:
    """Store webhook secret in keyring.

    Args:
        repo_id: Repository ID
        secret: Webhook secret
    """
    service_name = "futurnal"
    username = f"webhook:{repo_id}"
    keyring.set_password(service_name, username, secret)
    console.print(f"[green]✓[/green] Webhook secret stored securely in keyring")


def _get_webhook_secret(repo_id: str) -> Optional[str]:
    """Retrieve webhook secret from keyring.

    Args:
        repo_id: Repository ID

    Returns:
        Webhook secret or None if not found
    """
    service_name = "futurnal"
    username = f"webhook:{repo_id}"
    return keyring.get_password(service_name, username)


def _delete_webhook_secret(repo_id: str) -> None:
    """Delete webhook secret from keyring.

    Args:
        repo_id: Repository ID
    """
    service_name = "futurnal"
    username = f"webhook:{repo_id}"
    try:
        keyring.delete_password(service_name, username)
        console.print(f"[green]✓[/green] Webhook secret removed from keyring")
    except keyring.errors.PasswordDeleteError:
        pass  # Secret didn't exist


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command("enable")
def enable_webhook(
    repo_id: str = typer.Argument(..., help="Repository ID"),
    public_url: str = typer.Option(
        ...,
        "--public-url",
        "-u",
        help="Public URL for webhook (e.g., https://example.com/webhook/github)",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Local port for webhook server",
    ),
    secret: Optional[str] = typer.Option(
        None,
        "--secret",
        "-s",
        help="Webhook secret (auto-generated if not provided)",
    ),
    events: Optional[str] = typer.Option(
        None,
        "--events",
        "-e",
        help="Comma-separated event types (default: push,pull_request,issues)",
    ),
) -> None:
    """Enable webhook for GitHub repository.

    This command:
    1. Generates a secure webhook secret (or uses provided one)
    2. Stores secret in keyring
    3. Creates webhook via GitHub API
    4. Displays webhook configuration

    Example:
        $ futurnal sources github webhook enable my-repo \\
            --public-url https://abc123.ngrok.io/webhook/github \\
            --port 8765
    """
    console.print(f"\n[bold]Enabling webhook for repository:[/bold] {repo_id}")

    # Get components
    components = _get_components()
    registry = components["registry"]
    configurator = components["configurator"]

    # Get repository descriptor
    descriptor = None
    for repo in registry.list_repositories():
        if repo.id == repo_id:
            descriptor = repo
            break

    if not descriptor:
        console.print(
            f"[red]Error:[/red] Repository '{repo_id}' not found",
            style="bold",
        )
        console.print(
            "\nUse 'futurnal sources github list' to see registered repositories",
            style="dim",
        )
        raise typer.Exit(1)

    # Generate or use provided secret
    if secret is None:
        secret = _generate_webhook_secret()
        console.print(f"[green]✓[/green] Generated secure webhook secret")
    else:
        if len(secret) < 16:
            console.print(
                "[red]Error:[/red] Webhook secret must be at least 16 characters",
                style="bold",
            )
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] Using provided webhook secret")

    # Store secret in keyring
    _store_webhook_secret(repo_id, secret)

    # Parse events
    event_list = None
    if events:
        event_names = [e.strip() for e in events.split(",")]
        try:
            event_list = [WebhookEventType(name) for name in event_names]
        except ValueError as e:
            console.print(
                f"[red]Error:[/red] Invalid event type: {e}",
                style="bold",
            )
            raise typer.Exit(1)

    # Configure webhook via GitHub API
    try:
        import asyncio

        webhook_response = asyncio.run(
            configurator.configure_webhook(
                descriptor=descriptor,
                webhook_url=public_url,
                secret=secret,
                events=event_list,
                active=True,
            )
        )

        console.print(f"\n[green]✓[/green] Webhook configured successfully!")

        # Display webhook information
        table = Table(title="Webhook Configuration")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Webhook ID", str(webhook_response.get("id")))
        table.add_row("Repository", descriptor.full_name)
        table.add_row("Webhook URL", public_url)
        table.add_row("Secret", secret[:8] + "..." + secret[-8:])  # Masked
        table.add_row("Local Port", str(port))
        table.add_row(
            "Events", ", ".join(webhook_response.get("events", []))
        )
        table.add_row("Active", str(webhook_response.get("active")))

        console.print(table)

        console.print(
            f"\n[yellow]Note:[/yellow] Make sure webhook server is running on port {port}",
            style="dim",
        )
        console.print(
            f"[yellow]Note:[/yellow] Polling will continue as fallback",
            style="dim",
        )

    except Exception as e:
        console.print(
            f"[red]Error:[/red] Failed to configure webhook: {e}",
            style="bold",
        )
        _delete_webhook_secret(repo_id)  # Clean up on failure
        raise typer.Exit(1)


@app.command("disable")
def disable_webhook(
    repo_id: str = typer.Argument(..., help="Repository ID"),
    webhook_id: Optional[int] = typer.Option(
        None,
        "--webhook-id",
        "-w",
        help="Webhook ID to delete (auto-detected if not provided)",
    ),
) -> None:
    """Disable webhook for GitHub repository.

    This command:
    1. Deletes webhook via GitHub API
    2. Removes secret from keyring

    Example:
        $ futurnal sources github webhook disable my-repo
    """
    console.print(f"\n[bold]Disabling webhook for repository:[/bold] {repo_id}")

    # Get components
    components = _get_components()
    registry = components["registry"]
    configurator = components["configurator"]

    # Get repository descriptor
    descriptor = None
    for repo in registry.list_repositories():
        if repo.id == repo_id:
            descriptor = repo
            break

    if not descriptor:
        console.print(
            f"[red]Error:[/red] Repository '{repo_id}' not found",
            style="bold",
        )
        raise typer.Exit(1)

    # List webhooks if ID not provided
    try:
        import asyncio

        if webhook_id is None:
            webhooks = asyncio.run(configurator.list_webhooks(descriptor))
            if not webhooks:
                console.print(
                    f"[yellow]Warning:[/yellow] No webhooks found for {descriptor.full_name}",
                )
                _delete_webhook_secret(repo_id)
                raise typer.Exit(0)

            # Use first webhook if only one exists
            if len(webhooks) == 1:
                webhook_id = webhooks[0]["id"]
            else:
                console.print(f"\nFound {len(webhooks)} webhooks:")
                for i, hook in enumerate(webhooks, 1):
                    console.print(
                        f"  {i}. ID: {hook['id']} - URL: {hook['config'].get('url')}"
                    )
                console.print(
                    "\n[yellow]Note:[/yellow] Please specify --webhook-id to delete a specific webhook"
                )
                raise typer.Exit(1)

        # Confirm deletion
        if not Confirm.ask(
            f"\nDelete webhook {webhook_id} for {descriptor.full_name}?"
        ):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

        # Delete webhook
        asyncio.run(configurator.delete_webhook(descriptor, webhook_id))

        # Remove secret from keyring
        _delete_webhook_secret(repo_id)

        console.print(f"\n[green]✓[/green] Webhook disabled successfully!")

    except Exception as e:
        console.print(
            f"[red]Error:[/red] Failed to disable webhook: {e}",
            style="bold",
        )
        raise typer.Exit(1)


@app.command("status")
def webhook_status(
    repo_id: Optional[str] = typer.Argument(None, help="Repository ID (optional)"),
) -> None:
    """Show webhook status for repository or all repositories.

    Example:
        $ futurnal sources github webhook status
        $ futurnal sources github webhook status my-repo
    """
    # Get components
    components = _get_components()
    registry = components["registry"]
    configurator = components["configurator"]

    # Get repositories to check
    repos_to_check = []
    if repo_id:
        for repo in registry.list_repositories():
            if repo.id == repo_id:
                repos_to_check.append(repo)
                break
        if not repos_to_check:
            console.print(
                f"[red]Error:[/red] Repository '{repo_id}' not found",
                style="bold",
            )
            raise typer.Exit(1)
    else:
        repos_to_check = list(registry.list_repositories())

    if not repos_to_check:
        console.print("[yellow]No repositories registered[/yellow]")
        raise typer.Exit(0)

    # Display webhook status
    import asyncio

    table = Table(title="Webhook Status")
    table.add_column("Repository", style="cyan")
    table.add_column("Webhook ID", style="green")
    table.add_column("URL", style="yellow")
    table.add_column("Active", style="magenta")
    table.add_column("Events", style="blue")

    for descriptor in repos_to_check:
        try:
            webhooks = asyncio.run(configurator.list_webhooks(descriptor))
            if webhooks:
                for hook in webhooks:
                    table.add_row(
                        descriptor.full_name,
                        str(hook.get("id")),
                        hook.get("config", {}).get("url", "N/A")[:50],
                        "✓" if hook.get("active") else "✗",
                        ", ".join(hook.get("events", [])[:3]),
                    )
            else:
                table.add_row(
                    descriptor.full_name,
                    "-",
                    "-",
                    "-",
                    "No webhooks",
                )
        except Exception as e:
            table.add_row(
                descriptor.full_name,
                "-",
                "-",
                "-",
                f"Error: {str(e)[:20]}",
            )

    console.print(table)


@app.command("test")
def test_webhook(
    repo_id: str = typer.Argument(..., help="Repository ID"),
    webhook_id: Optional[int] = typer.Option(
        None,
        "--webhook-id",
        "-w",
        help="Webhook ID to test (auto-detected if not provided)",
    ),
) -> None:
    """Send test ping to webhook.

    This triggers GitHub to send a test ping event to the webhook URL.

    Example:
        $ futurnal sources github webhook test my-repo
    """
    console.print(f"\n[bold]Testing webhook for repository:[/bold] {repo_id}")

    # Get components
    components = _get_components()
    registry = components["registry"]
    configurator = components["configurator"]

    # Get repository descriptor
    descriptor = None
    for repo in registry.list_repositories():
        if repo.id == repo_id:
            descriptor = repo
            break

    if not descriptor:
        console.print(
            f"[red]Error:[/red] Repository '{repo_id}' not found",
            style="bold",
        )
        raise typer.Exit(1)

    try:
        import asyncio

        # List webhooks if ID not provided
        if webhook_id is None:
            webhooks = asyncio.run(configurator.list_webhooks(descriptor))
            if not webhooks:
                console.print(
                    f"[yellow]Warning:[/yellow] No webhooks found for {descriptor.full_name}",
                )
                raise typer.Exit(1)
            webhook_id = webhooks[0]["id"]

        # Send test
        asyncio.run(configurator.test_webhook(descriptor, webhook_id))

        console.print(
            f"\n[green]✓[/green] Test ping sent to webhook {webhook_id}!"
        )
        console.print(
            "[yellow]Note:[/yellow] Check your webhook server logs for the test event",
            style="dim",
        )

    except Exception as e:
        console.print(
            f"[red]Error:[/red] Failed to test webhook: {e}",
            style="bold",
        )
        raise typer.Exit(1)


__all__ = ["app"]

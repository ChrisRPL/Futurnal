"""CLI commands for GitHub repository management.

This module provides Typer commands for registering, managing, and monitoring
GitHub repositories as data sources for Futurnal.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ...privacy.audit import AuditLogger
from .api_client import GitHubAPIClient
from .credential_manager import (
    CredentialType,
    GitHubCredentialManager,
    detect_token_type,
    validate_token_format,
)
from .descriptor import (
    ConsentScope,
    PrivacyLevel,
    RepositoryPrivacySettings,
    RepositoryRegistry,
    SyncMode,
    VisibilityType,
    create_credential_id,
)
from .oauth_flow import GitHubOAuthDeviceFlow


app = typer.Typer(help="Manage GitHub repository sources")
console = Console()
error_console = Console(stderr=True)  # For error messages to be captured by Tauri


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_registry() -> RepositoryRegistry:
    """Get repository registry instance."""
    audit_dir = Path.home() / ".futurnal" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(audit_dir)
    return RepositoryRegistry(audit_logger=audit_logger)


def _get_credential_manager() -> GitHubCredentialManager:
    """Get credential manager instance."""
    audit_dir = Path.home() / ".futurnal" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(audit_dir)
    return GitHubCredentialManager(audit_logger=audit_logger)


def _parse_owner_repo(repo_spec: str) -> tuple[str, str]:
    """Parse owner/repo specification."""
    if "/" not in repo_spec:
        error_console.print(
            "Error: Repository must be in format 'owner/repo'"
        )
        raise typer.Exit(1)

    parts = repo_spec.split("/")
    if len(parts) != 2:
        error_console.print(
            "Error: Repository must be in format 'owner/repo'"
        )
        raise typer.Exit(1)

    return parts[0].strip(), parts[1].strip()


def _get_github_client_id() -> str:
    """Get GitHub OAuth client ID from environment."""
    client_id = os.getenv("FUTURNAL_GITHUB_CLIENT_ID")
    if not client_id:
        error_console.print(
            "Error: FUTURNAL_GITHUB_CLIENT_ID environment variable not set. "
            "To use OAuth, create a GitHub OAuth App at https://github.com/settings/developers"
        )
        raise typer.Exit(1)
    return client_id


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command("add")
def add_repository(
    repo: str = typer.Argument(..., help="Repository in format 'owner/repo'"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Human-readable name"),
    auth: str = typer.Option(
        "token", "--auth", "-a", help="Authentication method: oauth or token"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", "-t", help="Personal access token (for token auth)"
    ),
    host: str = typer.Option(
        "github.com", "--host", help="GitHub hostname (for Enterprise)"
    ),
    api_base_url: Optional[str] = typer.Option(
        None, "--api-base", help="Custom API base URL (for Enterprise)"
    ),
    branches: Optional[str] = typer.Option(
        None, "--branches", "-b", help="Comma-separated list of branches to sync"
    ),
    privacy_level: str = typer.Option(
        "standard",
        "--privacy",
        "-p",
        help="Privacy level: strict, standard, or permissive",
    ),
) -> None:
    """Add a GitHub repository as a data source."""
    owner, repo_name = _parse_owner_repo(repo)

    console.print(f"\n[bold]Adding GitHub repository:[/bold] {owner}/{repo_name}")

    # Parse privacy level
    try:
        privacy = PrivacyLevel(privacy_level.lower())
    except ValueError:
        error_console.print(
            f"Error: Invalid privacy level '{privacy_level}'. "
            "Must be: strict, standard, or permissive"
        )
        raise typer.Exit(1)

    # Handle authentication
    access_token = None
    cred_type = None

    if auth.lower() == "oauth":
        # OAuth Device Flow
        console.print("\n[bold cyan]Starting OAuth Device Flow...[/bold cyan]")
        client_id = _get_github_client_id()

        try:
            flow = GitHubOAuthDeviceFlow(
                client_id=client_id,
                github_host=host,
                scopes=["repo"],  # Request repo scope for full access
            )

            # Initiate device flow
            device_response = flow.initiate_device_flow()

            console.print(
                f"\n[bold yellow]Please authorize this application:[/bold yellow]"
            )
            console.print(f"1. Go to: [cyan]{device_response.verification_uri}[/cyan]")
            console.print(f"2. Enter code: [green bold]{device_response.user_code}[/green bold]")
            console.print("\nWaiting for authorization...", style="dim")

            # Poll for token
            result = flow.poll_for_token(
                device_response.device_code,
                interval=device_response.interval,
                on_pending=lambda attempt, _: console.print(
                    f"  Still waiting... (attempt {attempt})", style="dim"
                )
                if attempt % 5 == 0
                else None,
            )

            access_token = result.access_token
            cred_type = CredentialType.OAUTH_TOKEN
            console.print("\n[green]✓[/green] Authorization successful!")

        except RuntimeError as e:
            error_console.print(f"Error: OAuth flow failed: {e}")
            raise typer.Exit(1)

    elif auth.lower() == "token":
        # Personal Access Token
        if not token:
            # Only prompt interactively if stdin is a TTY
            if sys.stdin.isatty():
                token = Prompt.ask(
                    "\nEnter your GitHub Personal Access Token", password=True
                )
            else:
                # Non-interactive mode - require --token flag
                error_console.print(
                    "Error: Token is required. Use --token <your-token> flag."
                )
                raise typer.Exit(1)

        if not token:
            error_console.print("Error: Token is required for token authentication")
            raise typer.Exit(1)

        access_token = token.strip()
        cred_type = detect_token_type(access_token)

        # Validate token format
        if not validate_token_format(access_token, cred_type):
            error_console.print(
                "Error: Invalid token format. "
                "Expected classic PAT (ghp_*) or fine-grained PAT (github_pat_*)"
            )
            raise typer.Exit(1)

    else:
        error_console.print(
            f"Error: Invalid auth method '{auth}'. Must be: oauth or token"
        )
        raise typer.Exit(1)

    # Validate token and fetch repository metadata
    console.print("\n[bold]Validating access and fetching metadata...[/bold]")

    try:
        api_client = GitHubAPIClient(
            token=access_token, github_host=host, api_base_url=api_base_url
        )

        # Validate token
        token_info = api_client.validate_token()
        console.print(f"  Token scopes: {', '.join(token_info.scopes) or 'none'}")

        # Fetch repository metadata
        repo_info = api_client.get_repository(owner, repo_name)
        console.print(f"  Repository: {repo_info.full_name}")
        console.print(f"  Visibility: {repo_info.visibility.value}")
        console.print(f"  Default branch: {repo_info.default_branch}")
        console.print(f"  Description: {repo_info.description or '(none)'}")

        # Check if archived
        if repo_info.is_archived:
            console.print(
                "\n[yellow]Warning:[/yellow] This repository is archived.",
                style="bold",
            )
            if not Confirm.ask("Continue anyway?"):
                raise typer.Exit(0)

        # Verify required scopes
        if not api_client.verify_required_scopes([], repo_info.visibility):
            if repo_info.visibility == VisibilityType.PRIVATE:
                error_console.print(
                    "Error: Token does not have required scopes. Private repositories require the 'repo' scope."
                )
            else:
                error_console.print(
                    "Error: Token does not have required scopes for this repository."
                )
            raise typer.Exit(1)

    except RuntimeError as e:
        error_console.print(f"Error: {e}")
        raise typer.Exit(1)

    # Parse branches
    branch_list = None
    if branches:
        branch_list = [b.strip() for b in branches.split(",") if b.strip()]
    else:
        # Use default branch
        branch_list = [repo_info.default_branch]

    # Store credentials
    console.print("\n[bold]Storing credentials securely...[/bold]")
    cred_manager = _get_credential_manager()
    repo_id = f"{owner}_{repo_name}_{host}".replace("/", "_").replace(".", "_")
    credential_id = create_credential_id(repo_id)

    try:
        if cred_type == CredentialType.OAUTH_TOKEN:
            cred_manager.store_oauth_token(
                credential_id=credential_id,
                token=access_token,
                scopes=token_info.scopes,
                github_host=host,
                note=f"OAuth token for {owner}/{repo_name}",
            )
        else:
            cred_manager.store_personal_access_token(
                credential_id=credential_id,
                token=access_token,
                scopes=token_info.scopes,
                github_host=host,
                note=f"PAT for {owner}/{repo_name}",
            )
        console.print("  [green]✓[/green] Credentials stored in OS keychain")
    except Exception as e:
        error_console.print(f"Error: Failed to store credentials: {e}")
        raise typer.Exit(1)

    # Register repository
    console.print("\n[bold]Registering repository...[/bold]")
    registry = _get_registry()

    privacy_settings = RepositoryPrivacySettings(
        privacy_level=privacy,
        required_consent_scopes=[
            ConsentScope.GITHUB_REPO_ACCESS,
            ConsentScope.GITHUB_CODE_ANALYSIS,
        ],
    )

    try:
        descriptor = registry.register(
            owner=owner,
            repo=repo_name,
            github_host=host,
            api_base_url=api_base_url or repo_info.visibility.value,
            name=name,
            credential_id=credential_id,
            visibility=repo_info.visibility,
            sync_mode=SyncMode.GRAPHQL_API,
            branches=branch_list,
            privacy_settings=privacy_settings,
        )

        console.print(f"  [green]✓[/green] Repository registered: {descriptor.id}")
        console.print(
            f"\n[green bold]Successfully added {owner}/{repo_name}![/green bold]"
        )

    except Exception as e:
        error_console.print(f"Error: Registration failed: {e}")
        # Cleanup credentials on failure
        try:
            cred_manager.delete_credentials(credential_id)
        except:
            pass
        raise typer.Exit(1)


@app.command("list")
def list_repositories(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
) -> None:
    """List all registered GitHub repositories."""
    import json as json_module

    registry = _get_registry()
    repos = registry.list()

    if json_output:
        # Output as JSON for desktop app
        result = []
        for repo in repos:
            result.append({
                "id": repo.id,
                "name": repo.name,
                "full_name": repo.full_name,
                "github_host": repo.github_host,
                "visibility": repo.visibility.value,
                "branches": repo.branches,
                "updated_at": repo.updated_at.isoformat(),
            })
        print(json_module.dumps(result))
        return

    if not repos:
        console.print("\n[yellow]No GitHub repositories registered.[/yellow]")
        console.print("\nUse [cyan]futurnal sources github add[/cyan] to add a repository.")
        return

    table = Table(title="\nRegistered GitHub Repositories")
    table.add_column("ID", style="cyan")
    table.add_column("Repository", style="green")
    table.add_column("Host", style="blue")
    table.add_column("Visibility", style="magenta")
    table.add_column("Branches", style="yellow")
    table.add_column("Updated", style="dim")

    for repo in repos:
        table.add_row(
            repo.id[:12],
            repo.full_name,
            repo.github_host,
            repo.visibility.value,
            ", ".join(repo.branches[:3]) + ("..." if len(repo.branches) > 3 else ""),
            repo.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@app.command("inspect")
def inspect_repository(
    repo_id: str = typer.Argument(..., help="Repository ID to inspect")
) -> None:
    """Inspect detailed information about a registered repository."""
    registry = _get_registry()

    try:
        descriptor = registry.get(repo_id)
    except FileNotFoundError:
        error_console.print(f"Error: Repository '{repo_id}' not found")
        raise typer.Exit(1)

    console.print(f"\n[bold]Repository Details:[/bold]")
    console.print(f"  ID: {descriptor.id}")
    console.print(f"  Name: {descriptor.name or '(none)'}")
    console.print(f"  Owner/Repo: {descriptor.full_name}")
    console.print(f"  Host: {descriptor.github_host}")
    console.print(f"  Visibility: {descriptor.visibility.value}")
    console.print(f"  Sync Mode: {descriptor.sync_mode.value}")
    console.print(f"\n[bold]Branches:[/bold]")
    console.print(f"  Include: {', '.join(descriptor.branches)}")
    console.print(f"  Exclude: {', '.join(descriptor.exclude_branches)}")
    console.print(f"\n[bold]Content Scope:[/bold]")
    console.print(f"  Issues: {'✓' if descriptor.sync_issues else '✗'}")
    console.print(f"  Pull Requests: {'✓' if descriptor.sync_pull_requests else '✗'}")
    console.print(f"  Wiki: {'✓' if descriptor.sync_wiki else '✗'}")
    console.print(f"  Releases: {'✓' if descriptor.sync_releases else '✗'}")
    console.print(f"\n[bold]Privacy:[/bold]")
    console.print(f"  Level: {descriptor.privacy_settings.privacy_level.value}")
    console.print(
        f"  Path Anonymization: {'✓' if descriptor.privacy_settings.enable_path_anonymization else '✗'}"
    )
    console.print(
        f"  Secret Detection: {'✓' if descriptor.privacy_settings.detect_secrets else '✗'}"
    )
    console.print(f"\n[bold]Timestamps:[/bold]")
    console.print(f"  Created: {descriptor.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"  Updated: {descriptor.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")


@app.command("remove")
def remove_repository(
    repo_id: str = typer.Argument(..., help="Repository ID to remove"),
    delete_credentials: bool = typer.Option(
        False, "--delete-credentials", help="Also delete stored credentials"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Remove a registered GitHub repository."""
    registry = _get_registry()

    try:
        descriptor = registry.get(repo_id)
    except FileNotFoundError:
        error_console.print(f"Error: Repository '{repo_id}' not found")
        raise typer.Exit(1)

    console.print(f"\n[bold]Repository to remove:[/bold]")
    console.print(f"  {descriptor.full_name} ({descriptor.github_host})")

    if not yes:
        if not Confirm.ask("\nAre you sure you want to remove this repository?"):
            console.print("Cancelled.")
            raise typer.Exit(0)

    # Remove repository
    try:
        registry.remove(repo_id)
        console.print(f"[green]✓[/green] Repository removed from registry")
    except Exception as e:
        error_console.print(f"Error: Failed to remove repository: {e}")
        raise typer.Exit(1)

    # Delete credentials if requested
    if delete_credentials:
        cred_manager = _get_credential_manager()
        try:
            cred_manager.delete_credentials(descriptor.credential_id)
            console.print(f"[green]✓[/green] Credentials deleted from keychain")
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] Failed to delete credentials: {e}",
                style="dim",
            )

    console.print(f"\n[green bold]Successfully removed {descriptor.full_name}[/green bold]")


@app.command("test-connection")
def test_connection(
    repo_id: str = typer.Argument(..., help="Repository ID to test")
) -> None:
    """Test connection to a registered repository."""
    registry = _get_registry()
    cred_manager = _get_credential_manager()

    try:
        descriptor = registry.get(repo_id)
    except FileNotFoundError:
        error_console.print(f"Error: Repository '{repo_id}' not found")
        raise typer.Exit(1)

    console.print(f"\n[bold]Testing connection to:[/bold] {descriptor.full_name}")

    # Retrieve credentials
    try:
        creds = cred_manager.retrieve_credentials(descriptor.credential_id)
        console.print("  [green]✓[/green] Credentials retrieved")
    except Exception as e:
        error_console.print(f"Error: Failed to retrieve credentials: {e}")
        raise typer.Exit(1)

    # Test API access
    try:
        api_client = GitHubAPIClient(
            token=creds.token,
            github_host=descriptor.github_host,
            api_base_url=descriptor.api_base_url,
        )

        token_info = api_client.validate_token()
        console.print("  [green]✓[/green] Token is valid")
        console.print(f"    Scopes: {', '.join(token_info.scopes) or 'none'}")
        console.print(
            f"    Rate limit: {token_info.rate_remaining}/{token_info.rate_limit}"
        )

        repo_info = api_client.get_repository(descriptor.owner, descriptor.repo)
        console.print("  [green]✓[/green] Repository is accessible")
        console.print(f"    Default branch: {repo_info.default_branch}")
        console.print(f"    Last pushed: {repo_info.pushed_at}")

        console.print(f"\n[green bold]Connection test successful![/green bold]")

    except RuntimeError as e:
        error_console.print(f"Error: Connection failed: {e}")
        raise typer.Exit(1)


@app.command("sync")
def sync_repository(
    repo_id: str = typer.Argument(..., help="Repository ID to sync (or 'all' for all repos)"),
    shallow: bool = typer.Option(True, "--shallow/--full", help="Use shallow clone (default: shallow)"),
    depth: int = typer.Option(1, "--depth", "-d", help="Clone depth for shallow clones"),
    process: bool = typer.Option(False, "--process", "-p", help="Process synced files into knowledge graph"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Sync (clone/update) a GitHub repository to local storage."""
    import asyncio
    import json as json_module

    from .git_clone_sync import GitCloneRepositorySync
    from .sync_models import SyncStrategy, SyncStatus

    registry = _get_registry()
    cred_manager = _get_credential_manager()

    # Determine which repos to sync
    if repo_id.lower() == "all":
        repos_to_sync = registry.list()
        if not repos_to_sync:
            if json_output:
                print(json_module.dumps({"error": "No repositories registered"}))
            else:
                error_console.print("Error: No repositories registered")
            raise typer.Exit(1)
    else:
        try:
            descriptor = registry.get(repo_id)
            repos_to_sync = [descriptor]
        except FileNotFoundError:
            if json_output:
                print(json_module.dumps({"error": f"Repository '{repo_id}' not found"}))
            else:
                error_console.print(f"Error: Repository '{repo_id}' not found")
            raise typer.Exit(1)

    # Initialize sync infrastructure
    try:
        syncer = GitCloneRepositorySync(credential_manager=cred_manager)
    except RuntimeError as e:
        if json_output:
            print(json_module.dumps({"error": str(e)}))
        else:
            error_console.print(f"Error: {e}")
        raise typer.Exit(1)

    # Build sync strategy
    strategy = SyncStrategy(
        clone_depth=depth if shallow else None,
        single_branch=True,
        include_tags=not shallow,
    )

    results = []

    async def run_sync():
        nonlocal results
        for descriptor in repos_to_sync:
            if not json_output:
                console.print(f"\n[bold]Syncing:[/bold] {descriptor.full_name}")

            # Update strategy with repo's branches
            repo_strategy = strategy.model_copy()
            repo_strategy.branches = descriptor.branches or ["main"]

            result = await syncer.sync_repository(
                descriptor=descriptor,
                strategy=repo_strategy,
            )

            result_dict = {
                "repo_id": result.repo_id,
                "full_name": descriptor.full_name,
                "status": result.status.value,
                "files_synced": result.files_synced,
                "bytes_synced": result.bytes_synced,
                "bytes_synced_mb": round(result.bytes_synced_mb, 2),
                "duration_seconds": round(result.duration_seconds or 0, 2),
                "branches_synced": result.branches_synced,
                "error_message": result.error_message,
            }
            results.append(result_dict)

            if not json_output:
                if result.status == SyncStatus.COMPLETED:
                    console.print(f"  [green]✓[/green] Sync completed")
                    console.print(f"    Files: {result.files_synced}")
                    console.print(f"    Size: {result.bytes_synced_mb:.2f} MB")
                    console.print(f"    Duration: {result.duration_seconds:.1f}s")
                    console.print(f"    Clone path: {syncer.clone_base_dir / descriptor.id}")
                else:
                    console.print(f"  [red]✗[/red] Sync failed: {result.error_message}")

    # Run async sync
    asyncio.run(run_sync())

    # Process files if requested
    if process:
        for result_dict in results:
            if result_dict["status"] == "completed":
                repo_id_done = result_dict["repo_id"]
                full_name = result_dict["full_name"]
                clone_path = syncer.clone_base_dir / repo_id_done

                if clone_path.exists():
                    files_processed, files_failed = _process_cloned_files(
                        clone_path=clone_path,
                        repo_id=repo_id_done,
                        full_name=full_name,
                        json_output=json_output,
                    )
                    result_dict["files_processed"] = files_processed
                    result_dict["files_failed"] = files_failed

    if json_output:
        print(json_module.dumps(results if len(results) > 1 else results[0] if results else {}))
    else:
        # Summary for multiple repos
        if len(repos_to_sync) > 1:
            succeeded = sum(1 for r in results if r["status"] == "completed")
            failed = len(results) - succeeded
            console.print(f"\n[bold]Summary:[/bold] {succeeded} succeeded, {failed} failed")


def _process_cloned_files(
    clone_path: Path,
    repo_id: str,
    full_name: str,
    json_output: bool = False,
) -> tuple[int, int]:
    """Process cloned files through the normalization pipeline.

    Args:
        clone_path: Path to cloned repository
        repo_id: Repository identifier
        full_name: Full repository name (owner/repo)
        json_output: Whether to suppress console output

    Returns:
        Tuple of (files_processed, files_failed)
    """
    import asyncio
    import hashlib
    import json as json_module
    from datetime import datetime, timezone

    workspace_path = Path.home() / ".futurnal" / "workspace"
    parsed_dir = workspace_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    if not json_output:
        console.print(f"\n[bold]Processing files from:[/bold] {full_name}")

    # Supported file extensions for processing
    SUPPORTED_EXTENSIONS = {
        '.md', '.markdown', '.txt', '.rst',  # Documentation
        '.py', '.js', '.ts', '.tsx', '.jsx',  # Code
        '.java', '.kt', '.scala', '.go', '.rs', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.rb', '.php', '.swift', '.m', '.mm',
        '.json', '.yaml', '.yml', '.toml', '.xml',  # Config
        '.html', '.css', '.scss', '.less',  # Web
        '.sh', '.bash', '.zsh', '.fish',  # Shell
        '.sql', '.graphql',  # Query languages
        '.dockerfile', '.Dockerfile',  # Containers
    }

    # Files/directories to skip
    SKIP_PATTERNS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        '.idea', '.vscode', '.DS_Store', 'dist', 'build',
        '.egg-info', '.pytest_cache', '.mypy_cache', '.coverage',
    }

    files_processed = 0
    files_failed = 0

    def should_skip(path: Path) -> bool:
        for part in path.parts:
            if part in SKIP_PATTERNS:
                return True
        return False

    def get_file_hash(file_path: Path) -> str:
        """Generate SHA256 hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    # Walk through the clone directory
    for file_path in clone_path.rglob('*'):
        if not file_path.is_file():
            continue

        if should_skip(file_path):
            continue

        # Check extension
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if not content.strip():
                continue

            # Generate file hash for document ID
            file_hash = get_file_hash(file_path)
            relative_path = file_path.relative_to(clone_path)

            # Build source_id that matches "github-{owner}-{repo}" pattern
            # This ensures files are attributed to the GitHub source
            owner_repo = full_name.replace('/', '-')
            source_id = f"github-{owner_repo}"

            # Create parsed document in standard format
            doc = {
                "type": "NarrativeText",
                "element_id": file_hash[:32],
                "text": content[:50000],  # Truncate very large files
                "metadata": {
                    "languages": ["eng"],  # Default to English
                    "file_directory": str(file_path.parent),
                    "filename": file_path.name,
                    "filetype": _get_mimetype(file_path),
                    "last_modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                    "source": f"GitHub:{full_name}",
                    "source_type": "github",
                    "source_id": source_id,
                    "path": str(relative_path),
                    "repo_id": repo_id,
                    "repo_full_name": full_name,
                    "sha256": file_hash,
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            # Write to parsed directory
            output_path = parsed_dir / f"{file_hash}.json"
            output_path.write_text(json_module.dumps(doc, indent=2))

            # Extract and save entities
            try:
                from futurnal.pipeline.entity_extractor import EntityExtractor
                extractor = EntityExtractor(workspace_path)
                doc_entities = extractor.extract_from_element(doc)
                if doc_entities.entities:
                    extractor.save_entities(doc_entities)
            except Exception as entity_err:
                # Don't fail on entity extraction errors
                pass

            files_processed += 1

            if not json_output and files_processed % 50 == 0:
                console.print(f"  Processed {files_processed} files...", style="dim")

        except Exception as e:
            files_failed += 1
            if not json_output:
                console.print(f"  [yellow]Warning:[/yellow] Failed to process {file_path.name}: {e}", style="dim")

    if not json_output:
        console.print(f"  [green]✓[/green] Processed {files_processed} files ({files_failed} failed)")

    return files_processed, files_failed


def _get_mimetype(file_path: Path) -> str:
    """Get MIME type for a file based on extension."""
    ext_to_mime = {
        '.md': 'text/markdown',
        '.markdown': 'text/markdown',
        '.txt': 'text/plain',
        '.rst': 'text/x-rst',
        '.py': 'text/x-python',
        '.js': 'application/javascript',
        '.ts': 'application/typescript',
        '.tsx': 'application/typescript',
        '.jsx': 'application/javascript',
        '.java': 'text/x-java',
        '.kt': 'text/x-kotlin',
        '.scala': 'text/x-scala',
        '.go': 'text/x-go',
        '.rs': 'text/x-rust',
        '.c': 'text/x-c',
        '.cpp': 'text/x-c++',
        '.h': 'text/x-c',
        '.hpp': 'text/x-c++',
        '.cs': 'text/x-csharp',
        '.rb': 'text/x-ruby',
        '.php': 'text/x-php',
        '.swift': 'text/x-swift',
        '.json': 'application/json',
        '.yaml': 'text/x-yaml',
        '.yml': 'text/x-yaml',
        '.toml': 'text/x-toml',
        '.xml': 'application/xml',
        '.html': 'text/html',
        '.css': 'text/css',
        '.scss': 'text/x-scss',
        '.less': 'text/x-less',
        '.sh': 'application/x-sh',
        '.bash': 'application/x-sh',
        '.sql': 'application/sql',
        '.graphql': 'application/graphql',
    }
    return ext_to_mime.get(file_path.suffix.lower(), 'text/plain')


if __name__ == "__main__":
    app()

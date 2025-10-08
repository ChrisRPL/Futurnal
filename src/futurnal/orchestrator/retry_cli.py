"""CLI commands for managing retry policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .models import JobType
from .retry_policy import RetryPolicyRegistry, RetryStrategy


retry_app = typer.Typer(help="Manage retry policies")


def _get_retry_registry(
    workspace_path: Optional[Path] = None, config_path: Optional[Path] = None
) -> RetryPolicyRegistry:
    """Get retry policy registry instance.

    Args:
        workspace_path: Optional workspace directory
        config_path: Optional path to retry policy configuration

    Returns:
        RetryPolicyRegistry instance
    """
    if config_path and config_path.exists():
        return RetryPolicyRegistry(config_path=config_path)

    # Try default config locations
    workspace = workspace_path or Path.home() / ".futurnal"
    default_config = workspace / "config" / "retry_policies.yaml"

    if default_config.exists():
        return RetryPolicyRegistry(config_path=default_config)

    # Return default registry
    return RetryPolicyRegistry()


@retry_app.command("show")
def show_policies(
    workspace: Optional[Path] = typer.Option(None, help="Workspace directory"),
    config: Optional[Path] = typer.Option(None, help="Custom retry policy config file"),
    format_output: str = typer.Option(
        "table", "--format", help="Output format: table or json"
    ),
) -> None:
    """Display current retry policies for all connector types."""
    registry = _get_retry_registry(workspace, config)
    policies = registry.list_policies()

    if format_output == "json":
        output = {}
        for job_type, policy in policies.items():
            output[job_type.value] = {
                "strategy": policy.strategy.value,
                "max_attempts": policy.max_attempts,
                "base_delay_seconds": policy.base_delay_seconds,
                "max_delay_seconds": policy.max_delay_seconds,
                "jitter_factor": policy.jitter_factor,
                "backoff_multiplier": policy.backoff_multiplier,
                "transient_max_attempts": policy.transient_max_attempts,
                "rate_limit_delay_seconds": policy.rate_limit_delay_seconds,
                "permanent_failures_no_retry": policy.permanent_failures_no_retry,
            }
        typer.echo(json.dumps(output, indent=2))
    else:
        # Table format
        typer.echo("\n" + "=" * 120)
        typer.echo("RETRY POLICIES BY CONNECTOR TYPE")
        typer.echo("=" * 120 + "\n")

        # Header
        header = (
            f"{'CONNECTOR':<20} | {'STRATEGY':<20} | {'ATTEMPTS':<8} | "
            f"{'BASE DELAY':<11} | {'MAX DELAY':<10} | {'RATE LIMIT':<11}"
        )
        typer.echo(header)
        typer.echo("-" * 120)

        # Rows
        for job_type, policy in sorted(policies.items(), key=lambda x: x[0].value):
            connector = job_type.value
            strategy = policy.strategy.value
            attempts = f"{policy.max_attempts}"
            if policy.transient_max_attempts:
                attempts += f" ({policy.transient_max_attempts}T)"

            base_delay = f"{policy.base_delay_seconds}s"
            max_delay = f"{policy.max_delay_seconds}s"
            rate_limit = (
                f"{policy.rate_limit_delay_seconds}s"
                if policy.rate_limit_delay_seconds
                else "N/A"
            )

            row = (
                f"{connector:<20} | {strategy:<20} | {attempts:<8} | "
                f"{base_delay:<11} | {max_delay:<10} | {rate_limit:<11}"
            )
            typer.echo(row)

        typer.echo("\n" + "=" * 120)
        typer.echo(
            "Note: (T) indicates transient failure override, e.g., '5 (7T)' means max 5 attempts, 7 for transient"
        )
        typer.echo("=" * 120 + "\n")


@retry_app.command("validate")
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to retry policy YAML config"),
) -> None:
    """Validate a retry policy configuration file.

    Args:
        config_path: Path to YAML configuration file to validate
    """
    if not config_path.exists():
        typer.echo(f"Error: Configuration file not found: {config_path}")
        raise typer.Exit(1)

    typer.echo(f"Validating retry policy configuration: {config_path}\n")

    try:
        # Attempt to load the configuration
        registry = RetryPolicyRegistry(config_path=config_path)
        policies = registry.list_policies()

        typer.echo("✓ Configuration is valid!\n")
        typer.echo(f"Loaded {len(policies)} retry policies:\n")

        for job_type, policy in sorted(policies.items(), key=lambda x: x[0].value):
            typer.echo(f"  • {job_type.value}")
            typer.echo(f"      Strategy: {policy.strategy.value}")
            typer.echo(f"      Max Attempts: {policy.max_attempts}")
            typer.echo(f"      Base Delay: {policy.base_delay_seconds}s")
            typer.echo(f"      Max Delay: {policy.max_delay_seconds}s")
            if policy.transient_max_attempts:
                typer.echo(
                    f"      Transient Override: {policy.transient_max_attempts} attempts"
                )
            if policy.rate_limit_delay_seconds:
                typer.echo(
                    f"      Rate Limit Delay: {policy.rate_limit_delay_seconds}s"
                )
            typer.echo()

    except ValueError as e:
        typer.echo(f"✗ Configuration validation failed:\n")
        typer.echo(f"  {e}\n")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"✗ Unexpected error during validation:\n")
        typer.echo(f"  {type(e).__name__}: {e}\n")
        raise typer.Exit(1)


@retry_app.command("example")
def show_example(
    connector: Optional[str] = typer.Option(
        None, help="Show example for specific connector type"
    ),
) -> None:
    """Display example retry policy configuration.

    Args:
        connector: Optional connector type to show specific example
    """
    if connector:
        # Validate connector type
        try:
            job_type = JobType(connector)
        except ValueError:
            typer.echo(f"Invalid connector type: {connector}")
            typer.echo(
                f"Valid types: {', '.join(t.value for t in JobType)}"
            )
            raise typer.Exit(1)

        # Show single connector example
        registry = RetryPolicyRegistry()
        policy = registry.get_policy(job_type)

        example = {
            "retry_policies": {
                connector: {
                    "strategy": policy.strategy.value,
                    "max_attempts": policy.max_attempts,
                    "base_delay_seconds": policy.base_delay_seconds,
                    "max_delay_seconds": policy.max_delay_seconds,
                    "jitter_factor": policy.jitter_factor,
                    "backoff_multiplier": policy.backoff_multiplier,
                }
            }
        }

        if policy.transient_max_attempts:
            example["retry_policies"][connector][
                "transient_max_attempts"
            ] = policy.transient_max_attempts
        if policy.rate_limit_delay_seconds:
            example["retry_policies"][connector][
                "rate_limit_delay_seconds"
            ] = policy.rate_limit_delay_seconds

        typer.echo(f"\nExample configuration for {connector}:\n")
        typer.echo("```yaml")
        import yaml

        typer.echo(yaml.dump(example, default_flow_style=False, sort_keys=False))
        typer.echo("```\n")
    else:
        # Show full example
        typer.echo("\nExample retry_policies.yaml configuration:\n")
        typer.echo("```yaml")
        typer.echo(
            """# Retry policy configuration for Futurnal ingestion connectors
# Place this file at: ~/.futurnal/config/retry_policies.yaml

retry_policies:
  local_files:
    strategy: exponential_backoff
    max_attempts: 3
    base_delay_seconds: 30
    max_delay_seconds: 300
    jitter_factor: 0.2
    backoff_multiplier: 2.0
    permanent_failures_no_retry: true

  obsidian_vault:
    strategy: exponential_backoff
    max_attempts: 3
    base_delay_seconds: 30
    max_delay_seconds: 300
    jitter_factor: 0.2
    backoff_multiplier: 2.0

  imap_mailbox:
    strategy: exponential_backoff
    max_attempts: 5
    base_delay_seconds: 120
    max_delay_seconds: 1800
    jitter_factor: 0.3
    backoff_multiplier: 2.0
    transient_max_attempts: 7  # Extra attempts for transient failures
    rate_limit_delay_seconds: 600  # 10 min for rate limits

  github_repository:
    strategy: exponential_backoff
    max_attempts: 5
    base_delay_seconds: 300  # 5 min base for API limits
    max_delay_seconds: 3600  # 1 hour max
    jitter_factor: 0.25
    backoff_multiplier: 2.0
    rate_limit_delay_seconds: 900  # 15 min for rate limits
    permanent_failures_no_retry: true

# Available strategies:
#   - exponential_backoff: delay = base * (multiplier ** attempt)
#   - linear_backoff: delay = base * (attempt + 1)
#   - fixed_delay: constant delay
#   - immediate: no delay (testing only)
#   - no_retry: fail immediately

# Failure types (auto-detected):
#   - transient: Temporary issues (network, timeouts)
#   - rate_limited: API rate limits
#   - permanent: Permission errors, invalid credentials
#   - unknown: Unclassified errors
"""
        )
        typer.echo("```\n")

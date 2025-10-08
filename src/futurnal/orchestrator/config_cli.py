"""CLI commands for orchestrator configuration management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import (
    ConfigurationError,
    ConfigurationManager,
    ConfigMigrationManager,
    OrchestratorConfig,
    SecretsManager,
)
from ..privacy.audit import AuditLogger


orchestrator_config_app = typer.Typer(
    help="Manage orchestrator configuration",
    name="config"
)


@orchestrator_config_app.command("validate")
def validate_config(
    config_path: Path = typer.Option(
        Path.home() / ".futurnal" / "config" / "orchestrator.yaml",
        "--config",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed validation output"
    ),
) -> None:
    """Validate orchestrator configuration.

    Checks configuration file for errors without loading it.
    Displays validation errors if any are found.
    """
    manager = ConfigurationManager(config_path=config_path)
    errors = manager.validate()

    if not errors:
        typer.echo(f"✅ Configuration is valid: {config_path}")
        if verbose:
            # Load and display config details
            config = manager.load()
            typer.echo("\nConfiguration details:")
            typer.echo(f"  Version: {config.version}")
            typer.echo(f"  Workers: {config.workers.max_workers} (hardware_cap: {config.workers.hardware_cap_enabled})")
            typer.echo(f"  Queue: {config.queue.database_path} (WAL: {config.queue.wal_mode})")
            typer.echo(f"  Retry: {config.retry.max_attempts} attempts, {config.retry.base_delay_seconds}s base delay")
            typer.echo(f"  Quarantine: {'enabled' if config.quarantine.enabled else 'disabled'}")
            typer.echo(f"  Telemetry: {'enabled' if config.telemetry.enabled else 'disabled'}")
            typer.echo(f"  Security: audit={config.security.audit_logging}, redaction={config.security.path_redaction}")
    else:
        typer.echo(f"❌ Configuration validation failed: {config_path}")
        typer.echo("\nErrors:")
        for error in errors:
            typer.echo(f"  - {error}")
        raise typer.Exit(code=1)


@orchestrator_config_app.command("show")
def show_config(
    config_path: Path = typer.Option(
        Path.home() / ".futurnal" / "config" / "orchestrator.yaml",
        "--config",
        help="Path to configuration file"
    ),
    section: Optional[str] = typer.Option(
        None,
        "--section",
        help="Show specific section (workers, queue, retry, quarantine, telemetry, security)"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        help="Output format (yaml or json)"
    ),
) -> None:
    """Display current orchestrator configuration.

    Shows the effective configuration with all values.
    Can filter to specific sections and output in YAML or JSON.
    """
    try:
        manager = ConfigurationManager(config_path=config_path)
        config = manager.load()
    except ConfigurationError as exc:
        typer.echo(f"❌ Failed to load configuration: {exc}")
        raise typer.Exit(code=1)

    # Get config data
    data = config.model_dump(mode="python")

    # Filter to section if requested
    if section:
        if section not in data:
            typer.echo(f"❌ Unknown section: {section}")
            typer.echo(f"Available sections: {', '.join(data.keys())}")
            raise typer.Exit(code=1)
        data = {section: data[section]}

    # Output in requested format
    if format == "json":
        output = json.dumps(data, indent=2, default=str)
    else:  # yaml
        output = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)

    typer.echo(output)


@orchestrator_config_app.command("migrate")
def migrate_config(
    config_path: Path = typer.Option(
        Path.home() / ".futurnal" / "config" / "orchestrator.yaml",
        "--config",
        help="Path to configuration file"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show migration without applying changes"
    ),
) -> None:
    """Migrate configuration to current schema version.

    Automatically upgrades configuration to the latest schema version,
    creating a backup of the original file.
    """
    if not config_path.exists():
        typer.echo(f"❌ Configuration file not found: {config_path}")
        raise typer.Exit(code=1)

    # Load current config
    with open(config_path, encoding="utf-8") as f:
        current_data = yaml.safe_load(f)

    current_version = current_data.get("version", 1)
    target_version = ConfigMigrationManager.CURRENT_VERSION

    if current_version == target_version:
        typer.echo(f"✅ Configuration is already at version {target_version}")
        return

    typer.echo(f"Migrating configuration from v{current_version} to v{target_version}")

    if dry_run:
        typer.echo("\n🔍 Dry run mode - no changes will be made")
        typer.echo("\nCurrent configuration:")
        typer.echo(yaml.safe_dump(current_data, default_flow_style=False))

        # Perform migration in memory
        migrator = ConfigMigrationManager()
        if current_version == 1:
            migrated_data = migrator._migrate_v1_to_v2(current_data)
            migrated_data["version"] = target_version

        typer.echo("\nMigrated configuration:")
        typer.echo(yaml.safe_dump(migrated_data, default_flow_style=False))
    else:
        # Perform actual migration
        try:
            migrator = ConfigMigrationManager()
            migrated_config = migrator.migrate(config_path)

            backup_path = config_path.with_suffix(".yaml.backup")
            typer.echo(f"✅ Migration successful!")
            typer.echo(f"   Backup saved to: {backup_path}")
            typer.echo(f"   Configuration updated to v{target_version}")
        except Exception as exc:
            typer.echo(f"❌ Migration failed: {exc}")
            raise typer.Exit(code=1)


@orchestrator_config_app.command("init")
def init_config(
    config_path: Path = typer.Option(
        Path.home() / ".futurnal" / "config" / "orchestrator.yaml",
        "--config",
        help="Path to configuration file"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing configuration"
    ),
) -> None:
    """Initialize orchestrator configuration with secure defaults.

    Creates a new configuration file with all recommended settings.
    """
    if config_path.exists() and not force:
        typer.echo(f"❌ Configuration already exists: {config_path}")
        typer.echo("   Use --force to overwrite")
        raise typer.Exit(code=1)

    # Create default configuration
    config = OrchestratorConfig()
    manager = ConfigurationManager(config_path=config_path)
    manager.save(config)

    typer.echo(f"✅ Configuration initialized: {config_path}")
    typer.echo("\nDefault settings:")
    typer.echo(f"  Workers: {config.workers.max_workers} concurrent")
    typer.echo(f"  Queue: {config.queue.database_path}")
    typer.echo(f"  Retry: {config.retry.max_attempts} attempts")
    typer.echo(f"  Quarantine: {'enabled' if config.quarantine.enabled else 'disabled'}")
    typer.echo(f"  Telemetry: {'enabled' if config.telemetry.enabled else 'disabled'}")
    typer.echo(f"  Security: All features enabled")


@orchestrator_config_app.command("set")
def set_config_value(
    key: str = typer.Argument(
        ...,
        help="Configuration key (e.g., workers.max_workers, retry.max_attempts)"
    ),
    value: str = typer.Argument(
        ...,
        help="New value"
    ),
    config_path: Path = typer.Option(
        Path.home() / ".futurnal" / "config" / "orchestrator.yaml",
        "--config",
        help="Path to configuration file"
    ),
) -> None:
    """Update a configuration value.

    Modifies a specific configuration setting and validates the result.
    """
    try:
        manager = ConfigurationManager(config_path=config_path)
        config = manager.load()
    except ConfigurationError as exc:
        typer.echo(f"❌ Failed to load configuration: {exc}")
        raise typer.Exit(code=1)

    # Parse key path
    keys = key.split(".")
    if len(keys) != 2:
        typer.echo(f"❌ Invalid key format. Use: section.field (e.g., workers.max_workers)")
        raise typer.Exit(code=1)

    section_name, field_name = keys

    # Get the section
    try:
        section = getattr(config, section_name)
    except AttributeError:
        typer.echo(f"❌ Unknown section: {section_name}")
        typer.echo(f"Available sections: workers, queue, retry, quarantine, telemetry, security")
        raise typer.Exit(code=1)

    # Convert value to appropriate type
    try:
        current_value = getattr(section, field_name)
        if isinstance(current_value, bool):
            typed_value = value.lower() in {"true", "1", "yes"}
        elif isinstance(current_value, int):
            typed_value = int(value)
        elif isinstance(current_value, float):
            typed_value = float(value)
        elif isinstance(current_value, Path):
            typed_value = Path(value)
        else:
            typed_value = value
    except AttributeError:
        typer.echo(f"❌ Unknown field: {section_name}.{field_name}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        typer.echo(f"❌ Invalid value type: {exc}")
        raise typer.Exit(code=1)

    # Update the value
    try:
        setattr(section, field_name, typed_value)
        manager.save(config)
        typer.echo(f"✅ Updated {key} = {typed_value}")
    except Exception as exc:
        typer.echo(f"❌ Failed to update configuration: {exc}")
        raise typer.Exit(code=1)


__all__ = ["orchestrator_config_app"]

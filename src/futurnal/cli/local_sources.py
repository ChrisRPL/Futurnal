"""CLI utilities for managing local ingestion sources."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import typer
from croniter import croniter

from ..ingestion.local.config import LocalIngestionSource, load_config_from_dict
from ..ingestion.local.connector import LocalFilesConnector
from ..ingestion.local.quarantine import (
    QUARANTINE_SUMMARY_FILENAME,
    MAX_RETRY_ATTEMPTS,
    QuarantineEntry,
    archive_entry,
    append_note,
    iter_entries,
    update_entry,
    write_summary,
)
from ..ingestion.local.state import StateStore
from ..orchestrator.models import IngestionJob, JobPriority, JobType
from ..orchestrator.queue import JobQueue, JobStatus
from ..privacy import AuditLogger, ConsentRegistry, redact_path
from ..privacy.audit import AuditEvent
from ..ingestion.obsidian.descriptor import VaultRegistry, ObsidianVaultDescriptor
from ..ingestion.imap.descriptor import (
    AuthMode,
    ConsentScope,
    ImapMailboxDescriptor,
    MailboxPrivacySettings,
    MailboxRegistry,
    PrivacyLevel,
    create_credential_id,
    generate_mailbox_id,
)
from ..ingestion.obsidian.quality_gate import (
    QualityGateConfig,
    QualityGateEvaluator,
    create_quality_gate_evaluator
)
from ..ingestion.obsidian.report_generator import (
    ReportGenerator,
    JSONReportFormatter,
    MarkdownReportFormatter,
    create_report_generator,
    create_json_formatter,
    create_markdown_formatter
)
from ..ingestion.obsidian.sync_metrics import create_metrics_collector
from ..ingestion.obsidian.quality_gate import (
    QualityGateConfig,
    QualityGateEvaluator,
    create_quality_gate_evaluator
)
from ..ingestion.obsidian.report_generator import (
    ReportGenerator,
    JSONReportFormatter,
    MarkdownReportFormatter,
    create_report_generator,
    create_json_formatter,
    create_markdown_formatter
)
from ..ingestion.obsidian.sync_metrics import create_metrics_collector

app = typer.Typer(help="Manage Futurnal local data sources")
obsidian_app = typer.Typer(help="Manage Obsidian vault sources")
imap_app = typer.Typer(help="Manage IMAP mailbox sources")
imap_app = typer.Typer(help="Manage IMAP mailbox sources")
imap_app = typer.Typer(help="Manage IMAP mailbox sources")

DEFAULT_CONFIG_PATH = Path.home() / ".futurnal" / "sources.json"
DEFAULT_WORKSPACE_PATH = Path.home() / ".futurnal" / "workspace"
DEFAULT_IMAP_REGISTRY = Path.home() / ".futurnal" / "sources" / "imap"
DEFAULT_IMAP_WORKSPACE = DEFAULT_WORKSPACE_PATH / "imap"


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_datetime_option(option_name: str, value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid datetime for {option_name}; use ISO 8601 (e.g. 2024-01-01T12:30:00)"
        ) from exc


def _resolve_consent_scopes(values: Iterable[str]) -> List[ConsentScope]:
    scopes = {
        ConsentScope.MAILBOX_ACCESS,
        ConsentScope.EMAIL_CONTENT_ANALYSIS,
    }
    for raw in values:
        candidate = raw.strip()
        if not candidate:
            continue
        normalized = candidate.lower().replace("-", "_")
        resolved = None
        for scope in ConsentScope:
            if normalized == scope.name.lower() or candidate.lower() == scope.value.lower():
                resolved = scope
                break
        if resolved is None:
            raise typer.BadParameter(
                f"Unknown consent scope '{candidate}'. Valid options: "
                + ", ".join(scope.name.lower() for scope in ConsentScope)
            )
        scopes.add(resolved)
    return sorted(scopes, key=lambda item: item.value)


def _hash_email(email: str) -> str:
    return sha256(email.encode()).hexdigest()[:16]


def _build_privacy_settings(
    *,
    privacy_level: PrivacyLevel,
    consent_scopes: Iterable[ConsentScope],
    sender_anonymization: bool,
    recipient_anonymization: bool,
    subject_redaction: bool,
    redact_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    privacy_keywords: Optional[List[str]] = None,
    audit_sync_events: bool = True,
    audit_content_changes: bool = False,
    retain_audit_days: Optional[int] = None,
) -> MailboxPrivacySettings:
    kwargs: Dict[str, object] = {
        "privacy_level": privacy_level,
        "required_consent_scopes": list(consent_scopes),
        "enable_sender_anonymization": sender_anonymization,
        "enable_recipient_anonymization": recipient_anonymization,
        "enable_subject_redaction": subject_redaction,
        "audit_sync_events": audit_sync_events,
        "audit_content_changes": audit_content_changes,
    }
    if redact_patterns is not None:
        kwargs["redact_email_patterns"] = redact_patterns
    if exclude_patterns is not None:
        kwargs["exclude_email_patterns"] = exclude_patterns
    if privacy_keywords is not None and privacy_keywords:
        kwargs["privacy_subject_keywords"] = privacy_keywords
    if retain_audit_days is not None:
        kwargs["retain_audit_days"] = retain_audit_days
    return MailboxPrivacySettings(**kwargs)


def _load_imap_registry(registry_path: Optional[Path]) -> MailboxRegistry:
    return MailboxRegistry(registry_root=registry_path) if registry_path else MailboxRegistry()


def _privacy_level_from_string(value: str) -> PrivacyLevel:
    try:
        return PrivacyLevel(value.lower())
    except ValueError as exc:
        valid = ", ".join(level.value for level in PrivacyLevel)
        raise typer.BadParameter(f"Invalid privacy level '{value}'. Valid options: {valid}") from exc


def _auth_mode_from_string(value: str) -> AuthMode:
    try:
        return AuthMode(value.lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in AuthMode)
        raise typer.BadParameter(f"Invalid auth mode '{value}'. Valid options: {valid}") from exc

TELEMETRY_DIR_NAME = "telemetry"
TELEMETRY_LOG_FILE = "telemetry.log"
TELEMETRY_SUMMARY_FILE = "telemetry_summary.json"
AUDIT_DIR_NAME = "audit"
AUDIT_LOG_FILE = "audit.log"
QUARANTINE_DIR_NAME = "quarantine"
QUARANTINE_ARCHIVE_DIR_NAME = "quarantine_archive"
QUEUE_DIR_NAME = "queue"
QUEUE_DB_FILE = "queue.db"
MAX_RETRIES = MAX_RETRY_ATTEMPTS
DISMISS_NOTE_KEY = "dismissed_by"
CONSENT_DIR_NAME = "privacy"
CONSENT_SCOPE_LOCAL_EXTERNAL = "local.external_processing"


def _load_state_store(workspace: Path) -> StateStore:
    state_dir = workspace / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return StateStore(state_dir / "state.db")


def _load_connector(workspace: Path, state_store: StateStore) -> LocalFilesConnector:
    return LocalFilesConnector(workspace_dir=workspace, state_store=state_store)


def _resolve_workspace(path: Optional[Path]) -> Path:
    return path or DEFAULT_WORKSPACE_PATH


def _load_quarantine_dir(workspace: Path) -> Path:
    return workspace / QUARANTINE_DIR_NAME


def _load_archive_dir(workspace: Path) -> Path:
    return workspace / QUARANTINE_ARCHIVE_DIR_NAME


def _resolve_source(config: dict, name: str) -> Dict:
    for source in config.get("sources", []):
        if source.get("name") == name:
            return source
    raise KeyError(f"Source '{name}' not found in configuration")


def _load_source(config_path: Path, source_name: str) -> LocalIngestionSource:
    config = _load_config(config_path)
    load_config_from_dict(config)
    source_dict = _resolve_source(config, source_name)
    return LocalIngestionSource(**source_dict)


def _quarantine_summary_path(workspace: Path) -> Path:
    return workspace / TELEMETRY_DIR_NAME / QUARANTINE_SUMMARY_FILENAME


def _audit_logger(workspace: Path) -> AuditLogger:
    return AuditLogger(workspace / AUDIT_DIR_NAME)


def _load_consent_registry(workspace: Path) -> ConsentRegistry:
    return ConsentRegistry(workspace / CONSENT_DIR_NAME)


def _load_job_queue(workspace: Path) -> JobQueue:
    queue_dir = workspace / QUEUE_DIR_NAME
    queue_dir.mkdir(parents=True, exist_ok=True)
    return JobQueue(queue_dir / QUEUE_DB_FILE)


def _load_sources(config_path: Path) -> Dict[str, LocalIngestionSource]:
    config = _load_config(config_path)
    valid = load_config_from_dict(config)
    return {source.name: source for source in valid.root}


def _save_sources(config_path: Path, sources: Iterable[LocalIngestionSource]) -> None:
    serialized = []
    for source in sorted(sources, key=lambda item: item.name):
        payload = source.model_dump(mode="json")
        payload["root_path"] = str(source.root_path)
        if payload.get("ignore_file") is None:
            payload["ignore_file"] = None
        serialized.append(payload)
    _save_config(config_path, {"sources": serialized})


def _update_source(
    config_path: Path,
    name: str,
    *,
    mutate: Callable[[LocalIngestionSource], LocalIngestionSource],
) -> LocalIngestionSource:
    sources = _load_sources(config_path)
    try:
        existing = sources[name]
    except KeyError as exc:
        raise typer.BadParameter(f"Source '{name}' not found") from exc
    updated = mutate(existing)
    sources[name] = updated
    _save_sources(config_path, sources.values())
    return updated


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {"sources": []}
    return json.loads(path.read_text())


def _save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


@app.command("register")
def register_local_source(
    name: str = typer.Option(..., help="Unique name for the source"),
    root: Path = typer.Option(..., exists=True, file_okay=False, help="Root directory to ingest"),
    include: Optional[str] = typer.Option(None, help="Comma-separated glob include patterns"),
    exclude: Optional[str] = typer.Option(None, help="Comma-separated glob exclude patterns"),
    ignore_file: Optional[Path] = typer.Option(None, help="Custom ignore file path"),
    follow_symlinks: bool = typer.Option(False, help="Follow symlinks when scanning"),
    max_workers: Optional[int] = typer.Option(
        None,
        min=1,
        max=32,
        help="Upper bound on concurrent ingestion workers for this source",
    ),
    max_files_per_batch: Optional[int] = typer.Option(
        None,
        min=1,
        max=1000,
        help="Maximum number of files an ingestion worker processes per batch",
    ),
    scan_interval_seconds: Optional[float] = typer.Option(
        None,
        min=0.1,
        max=3600.0,
        help="Fallback scan interval in seconds when watcher is unavailable",
    ),
    watcher_debounce_seconds: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=120.0,
        help="Debounce interval in seconds between watcher-triggered job enqueues",
    ),
    schedule: str = typer.Option(
        "@manual",
        help="Schedule: '@manual', '@interval', or cron expression",
    ),
    interval_seconds: Optional[float] = typer.Option(
        None,
        min=60.0,
        max=86400.0,
        help="Interval in seconds for '@interval' schedule (default: 300)",
    ),
    priority: str = typer.Option(
        "normal",
        help="Job priority: low, normal, high",
    ),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Register or update a local directory source."""

    include_list = [pattern.strip() for pattern in include.split(",") if pattern.strip()] if include else []
    exclude_list = [pattern.strip() for pattern in exclude.split(",") if pattern.strip()] if exclude else []

    source_dict = {
        "name": name,
        "root_path": str(root),
        "include": include_list,
        "exclude": exclude_list,
        "follow_symlinks": follow_symlinks,
        "ignore_file": str(ignore_file) if ignore_file else None,
        "max_workers": max_workers,
        "max_files_per_batch": max_files_per_batch,
        "scan_interval_seconds": scan_interval_seconds,
        "watcher_debounce_seconds": watcher_debounce_seconds,
        "schedule": schedule,
        "interval_seconds": interval_seconds,
        "priority": priority,
    }

    new_source = LocalIngestionSource(**source_dict)
    sources = _load_sources(config_path)
    sources[name] = new_source
    _save_sources(config_path, sources.values())
    typer.echo(f"Registered source '{name}' at {root}")


@app.command("list")
def list_sources(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    source_type: Optional[str] = typer.Option(None, help="Filter by source type (e.g., 'obsidian')"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List configured local sources."""
    import json as json_module

    sources = _load_sources(config_path)

    # If filtering by obsidian type, also include Obsidian vault descriptors
    lower_type = source_type.lower() if source_type else None

    if lower_type == "obsidian":
        try:
            registry = VaultRegistry()
            obsidian_descriptors = registry.list()

            # Convert descriptors to a dict format similar to sources
            obsidian_sources = {}
            for descriptor in obsidian_descriptors:
                source_name = descriptor.name or f"obsidian-{descriptor.id[:8]}"
                obsidian_sources[source_name] = {
                    'name': source_name,
                    'root_path': str(descriptor.base_path),
                    'type': 'obsidian',
                    'descriptor': descriptor
                }

            # Merge with regular sources
            all_sources = {}
            all_sources.update(sources)
            all_sources.update(obsidian_sources)

            # Filter to only obsidian sources
            sources = {name: src for name, src in all_sources.items() if src.get('type') == 'obsidian'}

        except Exception:
            # If we can't load obsidian registry, just use regular sources
            sources = {}

        if not sources:
            typer.echo("No Obsidian vaults found")
            return
    elif lower_type == "imap":
        try:
            registry = MailboxRegistry()
            mailboxes = registry.list()
            imap_sources: Dict[str, Dict[str, object]] = {}
            for descriptor in mailboxes:
                source_name = descriptor.name or f"imap-{descriptor.id[:8]}"
                hashed = _hash_email(descriptor.email_address)
                imap_sources[source_name] = {
                    "name": source_name,
                    "root_path": f"imap://{descriptor.imap_host}/{hashed}",
                    "type": "imap",
                    "descriptor": descriptor,
                }
            all_sources = {}
            all_sources.update(sources)
            all_sources.update(imap_sources)
            sources = {
                name: src for name, src in all_sources.items() if src.get("type") == "imap"
            }
        except Exception:
            sources = {}

        if not sources:
            typer.echo("No IMAP mailboxes found")
            return

    elif not sources:
        if json_output:
            print(json_module.dumps([]))
            return
        typer.echo("No sources configured")
        return

    # JSON output mode
    if json_output:
        result = []
        for source_name, source in sources.items():
            if hasattr(source, "root_path"):
                root_path = str(source.root_path)
                source_type_val = getattr(source, "source_type", "local_folder")
            else:
                root_path = str(source["root_path"])
                source_type_val = source.get("type", "local_folder")
            result.append({
                "id": source_name,
                "name": source_name,
                "connector_type": source_type_val,
                "path": root_path,
            })
        print(json_module.dumps(result))
        return

    for source_name, source in sources.items():
        # Handle both LocalIngestionSource objects and dict representations
        if hasattr(source, "root_path"):
            root_path = str(source.root_path)
            source_dict: Dict[str, Any] | None = None
        else:
            source_dict = source  # type: ignore[assignment]
            root_path = str(source_dict["root_path"])

        redacted_path: str
        if source_dict and source_dict.get("type") == "obsidian" and "descriptor" in source_dict:
            descriptor = source_dict["descriptor"]
            redaction_policy = descriptor.build_redaction_policy()
            redacted_path = redaction_policy.apply(root_path).redacted
        elif source_dict and source_dict.get("type") == "imap" and "descriptor" in source_dict:
            descriptor = source_dict["descriptor"]
            redacted_path = redact_path(descriptor.email_address).redacted
        else:
            redacted_path = redact_path(root_path).redacted

        typer.echo(f"- {source_name}: {redacted_path}")


# -----------------
# Obsidian Commands
# -----------------


@obsidian_app.command("add")
def obsidian_add(
    path: Path = typer.Option(..., exists=True, file_okay=False, help="Path to Obsidian vault"),
    name: Optional[str] = typer.Option(None, help="Human-readable vault name"),
    icon: Optional[str] = typer.Option(None, help="Emoji or path to icon"),
    extra_ignore: Optional[str] = typer.Option(None, help="Comma-separated extra ignore rules"),
    redact_title: Optional[str] = typer.Option(None, help="Comma-separated patterns to redact sensitive note titles in logs"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    registry_path: Optional[Path] = typer.Option(
        None,
        help="Override path to the Obsidian registry directory (for tests)",
    ),
) -> None:
    """Register or update an Obsidian vault descriptor."""

    extras = [r.strip() for r in (extra_ignore or "").split(",") if r.strip()]
    redact_patterns = [p.strip() for p in (redact_title or "").split(",") if p.strip()] if redact_title else None
    registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
    descriptor = registry.register_path(path, name=name, icon=icon, extra_ignores=extras, redact_title_patterns=redact_patterns)
    
    # Show network mount warning if applicable
    warning = descriptor.get_network_warning()
    if warning:
        typer.echo(f"⚠️  WARNING: {warning}", err=True)
    
    # Show empty vault warning if applicable
    empty_warning = descriptor.get_empty_vault_warning()
    if empty_warning:
        typer.echo(f"⚠️  WARNING: {empty_warning}", err=True)
    
    if json_out:
        typer.echo(json.dumps(descriptor.model_dump(mode="json"), indent=2))
    else:
        vault_name = descriptor.name or f"vault-{descriptor.id[:8]}"
        # Use redaction policy to safely display path
        redaction_policy = descriptor.build_redaction_policy()
        redacted_path = redaction_policy.apply(descriptor.base_path).redacted
        typer.echo(f"✅ Registered Obsidian vault '{vault_name}' at {redacted_path}")
        typer.echo(f"   Vault ID: {descriptor.id}")
        
        # Show next steps
        typer.echo("\n📋 Next Steps:")
        typer.echo("   1. Convert to ingestion source:")
        typer.echo(f"      futurnal sources obsidian to-local-source {descriptor.id}")
        typer.echo("   2. Start ingestion:")
        typer.echo(f"      futurnal sources run <source_name>")
        typer.echo("   3. View ingestion reports:")
        typer.echo("      futurnal sources telemetry")
        typer.echo("      futurnal sources audit")


# -----------------
# IMAP Commands
# -----------------


def _construct_mailbox_descriptor(
    *,
    email_address: str,
    imap_host: str,
    imap_port: int,
    name: Optional[str],
    icon: Optional[str],
    auth_mode: AuthMode,
    credential_id: Optional[str],
    folders: Iterable[str],
    folder_patterns: Iterable[str],
    exclude_folders: Iterable[str],
    sync_from_date: Optional[datetime],
    max_message_age_days: Optional[int],
    provider: Optional[str],
    privacy_level: PrivacyLevel,
    consent_scopes: Iterable[ConsentScope],
    sender_anonymization: bool,
    recipient_anonymization: bool,
    subject_redaction: bool,
    redact_patterns: Iterable[str],
    exclude_patterns: Iterable[str],
    privacy_keywords: Iterable[str],
    audit_sync_events: bool,
    audit_content_changes: bool,
    retain_audit_days: Optional[int],
) -> ImapMailboxDescriptor:
    descriptor_privacy = _build_privacy_settings(
        privacy_level=privacy_level,
        consent_scopes=consent_scopes,
        sender_anonymization=sender_anonymization,
        recipient_anonymization=recipient_anonymization,
        subject_redaction=subject_redaction,
        redact_patterns=list(redact_patterns),
        exclude_patterns=list(exclude_patterns),
        privacy_keywords=list(privacy_keywords),
        audit_sync_events=audit_sync_events,
        audit_content_changes=audit_content_changes,
        retain_audit_days=retain_audit_days,
    )

    mailbox_id = generate_mailbox_id(email_address, imap_host)
    derived_credential_id = credential_id or create_credential_id(mailbox_id)

    descriptor = ImapMailboxDescriptor.from_registration(
        email_address=email_address,
        imap_host=imap_host,
        imap_port=imap_port,
        name=name,
        icon=icon,
        auth_mode=auth_mode,
        credential_id=derived_credential_id,
        folders=folders,
        folder_patterns=folder_patterns,
        exclude_folders=exclude_folders,
        sync_from_date=sync_from_date,
        max_message_age_days=max_message_age_days,
        provider=provider,
        privacy_settings=descriptor_privacy,
    )

    return descriptor.update(id=mailbox_id, credential_id=derived_credential_id)


def _display_mailbox_descriptor(descriptor: ImapMailboxDescriptor) -> None:
    redacted_email = redact_path(descriptor.email_address).redacted
    typer.echo(f"   Mailbox ID: {descriptor.id}")
    typer.echo(f"   Email: {redacted_email}")
    typer.echo(f"   Host: {descriptor.imap_host}:{descriptor.imap_port}")
    typer.echo(f"   Provider: {descriptor.provider or 'generic'}")
    typer.echo(f"   Auth mode: {descriptor.auth_mode.value}")
    typer.echo(f"   Folders: {', '.join(descriptor.folders) or 'INBOX'}")


@imap_app.command("add")
def imap_add(
    email: str = typer.Option(..., help="Mailbox email address"),
    host: str = typer.Option(..., help="IMAP hostname"),
    port: int = typer.Option(993, help="IMAP port", min=1, max=65535),
    name: Optional[str] = typer.Option(None, help="Human-readable mailbox name"),
    icon: Optional[str] = typer.Option(None, help="Emoji or icon"),
    auth: str = typer.Option("oauth2", help="Authentication mode: oauth2|app_password"),
    password: Optional[str] = typer.Option(None, help="App password for app_password auth mode"),
    credential_id: Optional[str] = typer.Option(None, help="Credential identifier in keychain"),
    folders: Optional[str] = typer.Option(None, help="Comma-separated folder whitelist"),
    folder_patterns: Optional[str] = typer.Option(None, help="Comma-separated folder glob patterns"),
    exclude_folders: Optional[str] = typer.Option(None, help="Comma-separated folders to skip"),
    sync_from: Optional[str] = typer.Option(None, help="Only ingest messages after this ISO timestamp"),
    max_age_days: Optional[int] = typer.Option(None, help="Limit ingestion to this many recent days"),
    provider: Optional[str] = typer.Option(None, help="Provider hint (gmail, office365, generic)"),
    privacy_level: str = typer.Option("standard", help="Privacy level: strict|standard|permissive"),
    consent: Optional[str] = typer.Option(None, help="Extra consent scopes (comma-separated)"),
    disable_sender_anonymization: bool = typer.Option(False, help="Allow sender addresses in telemetry"),
    disable_recipient_anonymization: bool = typer.Option(False, help="Allow recipient addresses in telemetry"),
    enable_subject_redaction: bool = typer.Option(False, help="Redact subject lines in telemetry"),
    redact_pattern: Optional[str] = typer.Option(None, help="Comma-separated regex patterns to redact"),
    exclude_pattern: Optional[str] = typer.Option(None, help="Comma-separated patterns to skip ingress"),
    privacy_keywords: Optional[str] = typer.Option(None, help="Comma-separated subject keywords triggering redaction"),
    disable_audit_sync: bool = typer.Option(False, help="Disable audit logging for sync events"),
    enable_audit_content: bool = typer.Option(False, help="Enable checksum-based audit content logging"),
    retain_audit_days: Optional[int] = typer.Option(None, help="Override audit log retention days"),
    registry_path: Optional[Path] = typer.Option(None, help="Override IMAP registry path"),
    workspace_root: Optional[Path] = typer.Option(None, help="Override IMAP workspace root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    auth_mode = _auth_mode_from_string(auth)
    privacy = _privacy_level_from_string(privacy_level)

    parsed_folders = _parse_csv(folders) or ["INBOX"]
    parsed_folder_patterns = _parse_csv(folder_patterns)
    parsed_exclude_folders = _parse_csv(exclude_folders)
    parsed_redact_patterns = _parse_csv(redact_pattern)
    parsed_exclude_patterns = _parse_csv(exclude_pattern)
    parsed_keywords = _parse_csv(privacy_keywords)
    parsed_consent_scopes = _resolve_consent_scopes(_parse_csv(consent))
    sync_from_date = _parse_datetime_option("sync-from", sync_from)

    # Handle credential storage for app_password auth mode
    actual_credential_id = credential_id
    if auth_mode == AuthMode.APP_PASSWORD and password:
        from ..ingestion.imap.credential_manager import CredentialManager
        from ..privacy.audit import AuditLogger

        workspace_path = workspace_root or DEFAULT_WORKSPACE_PATH
        audit_logger = AuditLogger(workspace_path / "audit")
        cred_manager = CredentialManager(audit_logger=audit_logger)

        # Generate credential ID and store credentials in keychain
        mailbox_id = generate_mailbox_id(email, host)
        actual_credential_id = create_credential_id(mailbox_id)
        cred_manager.store_app_password(
            credential_id=actual_credential_id,
            email_address=email,
            password=password,
        )

    descriptor = _construct_mailbox_descriptor(
        email_address=email,
        imap_host=host,
        imap_port=port,
        name=name,
        icon=icon,
        auth_mode=auth_mode,
        credential_id=actual_credential_id,
        folders=parsed_folders,
        folder_patterns=parsed_folder_patterns,
        exclude_folders=parsed_exclude_folders,
        sync_from_date=sync_from_date,
        max_message_age_days=max_age_days,
        provider=provider,
        privacy_level=privacy,
        consent_scopes=parsed_consent_scopes,
        sender_anonymization=not disable_sender_anonymization,
        recipient_anonymization=not disable_recipient_anonymization,
        subject_redaction=enable_subject_redaction,
        redact_patterns=parsed_redact_patterns,
        exclude_patterns=parsed_exclude_patterns,
        privacy_keywords=parsed_keywords,
        audit_sync_events=not disable_audit_sync,
        audit_content_changes=enable_audit_content,
        retain_audit_days=retain_audit_days,
    )

    registry = _load_imap_registry(registry_path)
    saved = registry.add_or_update(descriptor)

    if json_out:
        # Output compact JSON format for Tauri IPC
        typer.echo(json.dumps({
            "id": saved.id,
            "name": saved.name or saved.email_address,
            "email_address": saved.email_address,
            "imap_host": saved.imap_host,
            "paused": False,
        }))
    else:
        typer.echo(f"✅ Registered IMAP mailbox '{saved.name or saved.id[:8]}'")
        _display_mailbox_descriptor(saved)
        typer.echo("\n📋 Next Steps:")
        typer.echo("   1. Configure credentials via futurnal sources imap credentials add")
        typer.echo("   2. Convert to ingestion source:")
        typer.echo(f"      futurnal sources imap to-local-source {saved.id}")
        typer.echo("   3. Start ingestion: futurnal sources run <source_name>")


@imap_app.command("list")
def imap_list(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    registry = _load_imap_registry(registry_path)
    items = registry.list()
    if json_out:
        # Output compact JSON format for Tauri IPC
        result = []
        for i in items:
            result.append({
                "id": i.id,
                "name": i.name or i.email_address,
                "email_address": i.email_address,
                "imap_host": i.imap_host,
                "paused": False,
            })
        typer.echo(json.dumps(result))
        return
    if not items:
        typer.echo("No IMAP mailboxes registered")
        return
    for descriptor in items:
        typer.echo(f"- {descriptor.id}: {redact_path(descriptor.email_address).redacted}")


@imap_app.command("show")
def imap_show(
    mailbox_id: str,
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    registry = _load_imap_registry(registry_path)
    descriptor = registry.get(mailbox_id)
    typer.echo(json.dumps(descriptor.model_dump(mode="json"), indent=2))


@imap_app.command("test-connection")
def imap_test_connection(
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
        typer.echo(f"Testing IMAP connection to {host}...")

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
            typer.echo(json.dumps({
                "success": True,
                "message": "Connection successful",
                "folders": folder_count,
                "capabilities": [str(c) for c in capabilities] if capabilities else []
            }))
        else:
            typer.echo(f"✅ Connection successful!")
            typer.echo(f"   Folders found: {folder_count}")
            if capabilities:
                typer.echo(f"   Server capabilities: {', '.join(str(c) for c in list(capabilities)[:5])}...")

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
            typer.echo(json.dumps({"success": False, "error": error_msg}))
        else:
            typer.echo(f"❌ Connection failed: {error_msg}")
        raise typer.Exit(1)

    except ssl.SSLError as e:
        error_msg = f"SSL/TLS error: {e}"
        if json_output:
            typer.echo(json.dumps({"success": False, "error": error_msg}))
        else:
            typer.echo(f"❌ Connection failed: {error_msg}")
        raise typer.Exit(1)

    except ConnectionRefusedError:
        error_msg = f"Connection refused. Check that {host}:{port} is correct."
        if json_output:
            typer.echo(json.dumps({"success": False, "error": error_msg}))
        else:
            typer.echo(f"❌ Connection failed: {error_msg}")
        raise typer.Exit(1)

    except Exception as e:
        error_msg = str(e)
        if json_output:
            typer.echo(json.dumps({"success": False, "error": error_msg}))
        else:
            typer.echo(f"❌ Connection failed: {error_msg}")
        raise typer.Exit(1)


@imap_app.command("remove")
def imap_remove(
    mailbox_id: str,
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    registry = _load_imap_registry(registry_path)
    try:
        descriptor = registry.get(mailbox_id)
    except FileNotFoundError:
        typer.echo(f"❌ Mailbox '{mailbox_id}' not found")
        raise typer.Exit(code=1)

    if not yes:
        typer.echo("⚠️  About to remove IMAP mailbox:")
        typer.echo(f"   Name: {descriptor.name or descriptor.id[:8]}")
        typer.echo(f"   Email: {redact_path(descriptor.email_address).redacted}")
        typer.echo(f"   ID: {mailbox_id}")
        confirm = typer.confirm("Are you sure you want to remove this mailbox?")
        if not confirm:
            typer.echo("❌ Operation cancelled")
            raise typer.Exit(code=0)

    registry.remove(mailbox_id)
    typer.echo(f"✅ Removed IMAP mailbox '{descriptor.name or descriptor.id[:8]}' ({mailbox_id})")


@imap_app.command("to-local-source")
def imap_to_local_source(
    mailbox_id: str,
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
    workspace_root: Optional[Path] = typer.Option(None, help="Override IMAP workspace root"),
    max_workers: Optional[int] = typer.Option(None, help="Max concurrent workers"),
    max_files_per_batch: Optional[int] = typer.Option(None, help="Max files per batch"),
    schedule: str = typer.Option("@manual", help="Cron schedule or @manual/@interval"),
    priority: str = typer.Option("normal", help="Job priority (low/normal/high)"),
) -> None:
    registry = _load_imap_registry(registry_path)
    descriptor = registry.get(mailbox_id)
    local_source = descriptor.to_local_source(
        workspace_root=workspace_root,
        max_workers=max_workers,
        max_files_per_batch=max_files_per_batch,
        schedule=schedule,
        priority=priority,
    )
    typer.echo(json.dumps(local_source.model_dump(mode="json"), indent=2))


@imap_app.command("sync")
def imap_sync(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID or email address"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="Specific folder to sync"),
    process: bool = typer.Option(False, "--process", "-p", help="Process synced emails into knowledge graph"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of emails to process (useful for initial sync)"),
    fast: bool = typer.Option(True, "--fast/--no-fast", help="Use fast sync mode (batch fetch, skip heavy processing)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for Tauri IPC"),
    workspace: Path = typer.Option(
        Path.home() / ".futurnal" / "workspace",
        "--workspace", "-w",
        help="Workspace directory"
    ),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    """Manually trigger mailbox sync.

    Fast mode (default) uses batch IMAP fetch and simple text extraction for speed.
    Use --no-fast for full Unstructured.io processing (slower but more thorough).

    Examples:
        futurnal sources imap sync abc123def                        # Fast sync (default)
        futurnal sources imap sync abc123def --folder INBOX
        futurnal sources imap sync abc123def --process --json
        futurnal sources imap sync abc123def --process --limit 50   # Limit initial sync
        futurnal sources imap sync abc123def --no-fast              # Full processing
    """
    from ..ingestion.imap.connector import ImapEmailConnector
    from ..ingestion.imap.sync_state import ImapSyncStateStore

    start_time = time.time()

    if not json_output:
        typer.echo(f"[bold blue]Syncing mailbox {mailbox_id}[/bold blue]")

    workspace_path = Path(workspace).expanduser()

    # Initialize components
    audit_logger = AuditLogger(workspace_path / "audit")
    consent_registry = ConsentRegistry(workspace_path / "privacy")
    mailbox_registry = _load_imap_registry(registry_path)

    # Use IMAP-specific state store (expects a file path, not directory)
    imap_state_file = workspace_path / "imap" / "state" / "sync_state.db"
    state_store = ImapSyncStateStore(imap_state_file)

    # Initialize connector
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
                    typer.echo(f"Error: Mailbox not found: {mailbox_id}")
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
            typer.echo(f"Error: Mailbox not found: {mailbox_id}")
        raise typer.Exit(1)

    # Run sync
    try:
        # Get descriptor for fast mode
        descriptor = mailbox_registry.get(mailbox_id)

        if fast:
            # Fast sync mode - batch fetch with simple text extraction
            if not json_output:
                typer.echo(f"[bold green]Using fast sync mode...[/bold green]")

            fast_result = _fast_sync_imap(
                mailbox_id=mailbox_id,
                email_address=email_address,
                descriptor=descriptor,
                folder=folder,
                limit=limit,
                workspace_path=workspace_path,
                json_output=json_output,
            )

            duration = time.time() - start_time
            files_processed = fast_result.get("files_processed", 0)
            files_failed = fast_result.get("files_failed", 0)
            total_new = fast_result.get("total_new", 0)
            total_bytes = fast_result.get("total_bytes", 0)
            folders_synced = fast_result.get("folders_synced", [])
            errors = fast_result.get("errors", [])
            status = fast_result.get("status", "completed")

            # Index processed emails to knowledge graph (ChromaDB + Neo4j) for GraphRAG
            kg_indexed = 0
            if files_processed > 0:
                if not json_output:
                    typer.echo("  Indexing to knowledge graph (ChromaDB + Neo4j)...")
                try:
                    from futurnal.pipeline.kg_indexer import KnowledgeGraphIndexer
                    indexer = KnowledgeGraphIndexer(workspace_dir=workspace_path)
                    index_stats = indexer.index_all_parsed()
                    kg_indexed = index_stats.get("chroma_indexed", 0) + index_stats.get("neo4j_indexed", 0)
                    if not json_output:
                        typer.echo(
                            f"    Indexed {index_stats.get('chroma_indexed', 0)} to ChromaDB, "
                            f"{index_stats.get('neo4j_indexed', 0)} to Neo4j"
                        )
                    indexer.close()
                except Exception as kg_err:
                    if not json_output:
                        typer.echo(f"  Warning: Knowledge graph indexing failed: {kg_err}", err=True)

            if json_output:
                output = {
                    "repo_id": mailbox_id,
                    "full_name": email_address,
                    "status": status,
                    "files_synced": files_processed,
                    "bytes_synced": total_bytes,
                    "bytes_synced_mb": round(total_bytes / (1024 * 1024), 2) if total_bytes else 0.0,
                    "duration_seconds": round(duration, 2),
                    "branches_synced": folders_synced,
                    "error_message": "; ".join(errors) if errors else None,
                    "files_processed": files_processed,
                    "files_failed": files_failed,
                    "new_messages": total_new,
                    "updated_messages": 0,
                    "deleted_messages": 0,
                    "limit_applied": limit,
                    "limited": limit is not None and total_new > limit if total_new else False,
                    "fast_mode": True,
                }
                print(json.dumps(output))
            else:
                typer.echo(f"\n✓ Fast sync completed")
                typer.echo(f"Folders synced: {len(folders_synced)}")
                typer.echo(f"Messages processed: {files_processed}")
                typer.echo(f"Data synced: {total_bytes / 1024:.1f} KB")
                typer.echo(f"Duration: {duration:.2f}s")
                if files_failed > 0:
                    typer.echo(f"Failed: {files_failed} emails")
                if errors:
                    typer.echo(f"\nErrors: {len(errors)}")

        else:
            # Slow sync mode - full Unstructured.io processing
            if not json_output:
                typer.echo(f"[bold yellow]Using full processing mode (slower)...[/bold yellow]")

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
                        typer.echo(f"\n[bold blue]Processing {limit} of {total_new} new emails into knowledge graph (limited)...[/bold blue]")
                    else:
                        typer.echo(f"\n[bold blue]Processing {total_new} new emails into knowledge graph...[/bold blue]")
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
                    "fast_mode": False,
                }
                print(json.dumps(output))
            else:
                typer.echo(f"\n✓ Sync completed")
                typer.echo(f"Folders synced: {len(results)}")
                typer.echo(f"New messages: {total_new}")
                typer.echo(f"Updated messages: {total_updated}")
                typer.echo(f"Deleted messages: {total_deleted}")
                typer.echo(f"Duration: {duration:.2f}s")

                if process:
                    typer.echo(f"\nProcessed: {files_processed} emails")
                    if files_failed > 0:
                        typer.echo(f"Failed: {files_failed} emails")

                if total_errors > 0:
                    typer.echo(f"\nErrors: {total_errors}")

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
            typer.echo(f"Error: Sync failed: {e}")
        raise typer.Exit(1)


def _fast_sync_imap(
    mailbox_id: str,
    email_address: str,
    descriptor: ImapMailboxDescriptor,
    folder: Optional[str],
    limit: Optional[int],
    workspace_path: Path,
    json_output: bool,
) -> dict:
    """Fast IMAP sync using batch fetch and simple text extraction.

    This bypasses the heavy Unstructured.io processing and instead:
    1. Connects directly to IMAP
    2. Batch-fetches messages in a single command
    3. Parses with Python's email module (fast)
    4. Stores as JSON directly

    Returns:
        Dict with sync results including files_processed, total_new, etc.
    """
    import email
    from email.header import decode_header
    from email.utils import parsedate_to_datetime
    import ssl

    from imapclient import IMAPClient
    from ..ingestion.imap.credential_manager import CredentialManager

    # Write to the main parsed directory so the knowledge graph can find the files
    parsed_dir = workspace_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    source_id = f"imap-{mailbox_id}"
    files_processed = 0
    files_failed = 0
    total_new = 0
    total_bytes = 0
    folders_synced = []
    errors = []

    # Get credentials
    from ..ingestion.imap.credential_manager import AppPassword, OAuth2Tokens

    credential_manager = CredentialManager()
    try:
        credentials = credential_manager.retrieve_credentials(descriptor.credential_id)
        if not credentials:
            raise ValueError(f"No credentials found for mailbox {mailbox_id}")
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Credential error: {e}",
            "files_processed": 0,
            "total_new": 0,
        }

    # Determine folders to sync
    folders_to_sync = [folder] if folder else descriptor.folders

    try:
        # Connect to IMAP server
        ssl_context = ssl.create_default_context()
        client = IMAPClient(
            host=descriptor.imap_host,
            port=descriptor.imap_port,
            ssl=True,
            ssl_context=ssl_context,
        )

        # Login based on credential type
        if isinstance(credentials, OAuth2Tokens):
            # OAuth2 authentication
            client.oauth2_login(email_address, credentials.access_token)
        elif isinstance(credentials, AppPassword):
            # App password authentication
            client.login(email_address, credentials.password)
        else:
            raise ValueError(f"Unknown credential type: {type(credentials)}")

        # Process each folder
        for folder_name in folders_to_sync:
            if not json_output:
                typer.echo(f"  Syncing folder: {folder_name}")

            try:
                select_info = client.select_folder(folder_name)

                # Search for messages
                # For simplicity, get recent messages (or all if first sync)
                search_criteria = ["ALL"]
                uids = client.search(search_criteria)

                if not uids:
                    folders_synced.append(folder_name)
                    continue

                # Apply limit to UIDs
                uids_to_fetch = list(uids)
                if limit and len(uids_to_fetch) > limit:
                    uids_to_fetch = uids_to_fetch[-limit:]  # Get most recent

                total_new += len(uids_to_fetch)

                # Batch fetch - get all messages in ONE command (fast!)
                # Fetch ENVELOPE for headers and BODY[TEXT] for body
                # Using RFC822 gets the full message which we can parse
                if not json_output:
                    typer.echo(f"    Fetching {len(uids_to_fetch)} messages...")

                fetch_data = client.fetch(uids_to_fetch, ['RFC822', 'FLAGS', 'INTERNALDATE'])

                # Process each fetched message
                for uid, data in fetch_data.items():
                    try:
                        raw_message = data.get(b'RFC822', b'')
                        internal_date = data.get(b'INTERNALDATE')
                        flags = data.get(b'FLAGS', [])

                        if not raw_message:
                            continue

                        total_bytes += len(raw_message)

                        # Parse with Python's email module (fast!)
                        msg = email.message_from_bytes(raw_message)

                        # Extract headers
                        def decode_header_value(value):
                            if not value:
                                return ""
                            decoded_parts = decode_header(value)
                            result = []
                            for part, charset in decoded_parts:
                                if isinstance(part, bytes):
                                    try:
                                        result.append(part.decode(charset or 'utf-8', errors='replace'))
                                    except:
                                        result.append(part.decode('utf-8', errors='replace'))
                                else:
                                    result.append(str(part))
                            return ' '.join(result)

                        subject = decode_header_value(msg.get('Subject', ''))
                        from_addr = decode_header_value(msg.get('From', ''))
                        to_addr = decode_header_value(msg.get('To', ''))
                        message_id = msg.get('Message-ID', f'<uid-{uid}@{descriptor.imap_host}>')
                        date_str = msg.get('Date')

                        # Parse date
                        date_parsed = None
                        if internal_date:
                            date_parsed = internal_date.isoformat()
                        elif date_str:
                            try:
                                date_parsed = parsedate_to_datetime(date_str).isoformat()
                            except:
                                pass

                        # Extract body text (fast, simple extraction)
                        body_text = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                if content_type == 'text/plain':
                                    try:
                                        payload = part.get_payload(decode=True)
                                        charset = part.get_content_charset() or 'utf-8'
                                        body_text = payload.decode(charset, errors='replace')
                                        break
                                    except:
                                        pass
                                elif content_type == 'text/html' and not body_text:
                                    try:
                                        payload = part.get_payload(decode=True)
                                        charset = part.get_content_charset() or 'utf-8'
                                        # Simple HTML to text - strip tags
                                        import re
                                        html_text = payload.decode(charset, errors='replace')
                                        body_text = re.sub(r'<[^>]+>', '', html_text)
                                        body_text = re.sub(r'\s+', ' ', body_text).strip()
                                    except:
                                        pass
                        else:
                            try:
                                payload = msg.get_payload(decode=True)
                                if payload:
                                    charset = msg.get_content_charset() or 'utf-8'
                                    body_text = payload.decode(charset, errors='replace')
                            except:
                                body_text = str(msg.get_payload())

                        # Create content and hash
                        content = f"{subject}\n\n{body_text}"
                        sha = hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()

                        # Create element
                        element = {
                            "sha256": sha,
                            "content": content,
                            "metadata": {
                                "source_id": source_id,
                                "source_type": "imap",
                                "source": f"imap://{email_address}/{folder_name}",
                                "uid": uid,
                                "message_id": message_id,
                                "subject": subject,
                                "sender": from_addr,
                                "recipient": to_addr,
                                "folder": folder_name,
                                "date": date_parsed,
                                "flags": [f.decode() if isinstance(f, bytes) else str(f) for f in flags],
                                "extractionTimestamp": datetime.now().isoformat(),
                                "schemaVersion": "v2",
                            }
                        }

                        # Write to parsed directory
                        safe_msg_id = str(uid)  # Use UID as safe identifier
                        output_file = parsed_dir / f"{sha[:16]}_{folder_name}_{safe_msg_id}.json"
                        with open(output_file, 'w') as f:
                            json.dump(element, f, indent=2, default=str)

                        files_processed += 1

                    except Exception as e:
                        files_failed += 1
                        errors.append(f"UID {uid}: {e}")

                folders_synced.append(folder_name)

            except Exception as e:
                errors.append(f"Folder {folder_name}: {e}")

        client.logout()

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "files_processed": files_processed,
            "total_new": total_new,
            "folders_synced": folders_synced,
            "errors": errors,
        }

    return {
        "status": "completed" if not errors else "completed_with_errors",
        "files_processed": files_processed,
        "files_failed": files_failed,
        "total_new": total_new,
        "total_bytes": total_bytes,
        "folders_synced": folders_synced,
        "errors": errors,
    }


def _process_synced_emails(
    mailbox_id: str,
    email_address: str,
    results: dict,
    workspace_path: Path,
    limit: Optional[int] = None,
) -> tuple:
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
        # Process new messages
        for msg in getattr(result, 'new_messages', []):
            # Check limit
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
                sha = hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()

                element = {
                    "sha256": sha,
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
                output_file = parsed_dir / f"{sha[:16]}_{source_id}_{msg_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(element, f, indent=2, default=str)

                files_processed += 1
                total_processed += 1

            except Exception as e:
                typer.echo(f"Warning: Failed to process message: {e}")
                files_failed += 1
                total_processed += 1  # Count failed as processed for limit purposes

        # Break outer loop if limit reached
        if limit is not None and total_processed >= limit:
            break

    return files_processed, files_failed


@imap_app.command("authenticate")
def imap_authenticate(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID to authenticate"),
    client_id: str = typer.Option(..., "--client-id", help="OAuth2 client ID"),
    client_secret: str = typer.Option(..., "--client-secret", help="OAuth2 client secret"),
    provider: str = typer.Option("gmail", "--provider", "-p", help="OAuth provider: gmail, office365"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON for Tauri IPC"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    """Authenticate an IMAP mailbox using OAuth2.

    Opens a browser for OAuth2 authentication and stores the tokens.
    This command must be run after adding an IMAP source with 'sources imap add'.
    """
    from ..ingestion.imap.oauth2_flow import OAuth2Flow, get_provider_config
    from ..ingestion.imap.credential_manager import CredentialManager

    # Load registry and get mailbox descriptor
    registry = _load_imap_registry(registry_path)

    try:
        descriptor = registry.get(mailbox_id)
    except FileNotFoundError:
        # Try to find by email address
        found = None
        for item in registry.list():
            if item.id == mailbox_id or item.email_address == mailbox_id:
                found = item
                break
        if not found:
            if json_output:
                print(json.dumps({"success": False, "error": f"Mailbox '{mailbox_id}' not found"}))
            else:
                typer.echo(f"❌ Mailbox '{mailbox_id}' not found")
            raise typer.Exit(code=1)
        descriptor = found

    # Get OAuth2 config for provider
    try:
        config = get_provider_config(provider, client_id, client_secret)
    except ValueError as e:
        if json_output:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            typer.echo(f"❌ {e}")
        raise typer.Exit(code=1)

    if not json_output:
        typer.echo(f"🔐 Starting OAuth2 authentication for {descriptor.email_address}...")
        typer.echo(f"   Provider: {provider}")
        typer.echo("\nA browser window will open for authentication.")
        typer.echo("Please sign in and grant access.\n")

    # Run OAuth2 flow
    try:
        oauth_flow = OAuth2Flow(config)
        tokens = oauth_flow.run_local_server_flow()

        if not json_output:
            typer.echo("✓ OAuth2 authentication successful!")

        # Store tokens using credential manager
        credential_manager = CredentialManager()

        stored_credential = credential_manager.store_oauth_tokens(
            credential_id=descriptor.credential_id,
            email_address=descriptor.email_address,
            tokens=tokens,
            provider=provider.lower(),
        )
        credential_id = stored_credential.credential_id

        if not json_output:
            typer.echo(f"✓ Tokens stored with credential ID: {credential_id}")
            typer.echo(f"\n📋 Next Steps:")
            typer.echo(f"   Run: futurnal sources imap sync {descriptor.id}")

        if json_output:
            print(json.dumps({
                "success": True,
                "mailbox_id": descriptor.id,
                "email": descriptor.email_address,
                "credential_id": credential_id,
                "provider": provider,
            }))

    except Exception as e:
        if json_output:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            typer.echo(f"❌ OAuth2 authentication failed: {e}")
        raise typer.Exit(code=1)


@obsidian_app.command("list")
def obsidian_list(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
    items = registry.list()
    if json_out:
        typer.echo(json.dumps([i.model_dump(mode="json") for i in items], indent=2))
        return
    if not items:
        typer.echo("No Obsidian vaults registered")
        return
    for i in items:
        # Use redaction policy to safely display path
        redaction_policy = i.build_redaction_policy()
        redacted_path = redaction_policy.apply(i.base_path).redacted
        typer.echo(f"- {i.id}: {redacted_path}")


@obsidian_app.command("show")
def obsidian_show(
    vault_id: str,
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
    descriptor = registry.get(vault_id)
    typer.echo(json.dumps(descriptor.model_dump(mode="json"), indent=2))


@obsidian_app.command("remove")
def obsidian_remove(
    vault_id: str,
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
) -> None:
    """Remove an Obsidian vault descriptor."""
    registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
    
    try:
        descriptor = registry.get(vault_id)
    except FileNotFoundError:
        typer.echo(f"❌ Vault '{vault_id}' not found")
        raise typer.Exit(code=1)
    
    vault_name = descriptor.name or f"vault-{descriptor.id[:8]}"
    
    if not yes:
        # Show confirmation prompt
        typer.echo(f"⚠️  About to remove Obsidian vault:")
        typer.echo(f"   Name: {vault_name}")
        # Use redaction policy to safely display path
        redaction_policy = descriptor.build_redaction_policy()
        redacted_path = redaction_policy.apply(descriptor.base_path).redacted
        typer.echo(f"   Path: {redacted_path}")
        typer.echo(f"   ID: {vault_id}")
        
        confirm = typer.confirm("Are you sure you want to remove this vault?")
        if not confirm:
            typer.echo("❌ Operation cancelled")
            raise typer.Exit(code=0)
    
    registry.remove(vault_id)
    typer.echo(f"✅ Removed Obsidian vault '{vault_name}' ({vault_id})")


@obsidian_app.command("to-local-source")
def obsidian_to_local_source(
    vault_id: str,
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
    max_workers: Optional[int] = typer.Option(None, help="Max concurrent workers"),
    schedule: str = typer.Option("@manual", help="Cron schedule or @manual/@interval"),
    priority: str = typer.Option("normal", help="Job priority (low/normal/high)"),
) -> None:
    """Convert Obsidian vault descriptor to LocalIngestionSource format."""
    registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
    descriptor = registry.get(vault_id)
    local_source = descriptor.to_local_source(
        max_workers=max_workers,
        schedule=schedule,
        priority=priority,
    )
    typer.echo(json.dumps(local_source.model_dump(mode="json"), indent=2))


@obsidian_app.command("report")
def obsidian_report(
    vault_id: str = typer.Argument(..., help="Vault ID to generate report for"),
    output_format: str = typer.Option("markdown", "--format", help="Output format: json, markdown"),
    output_path: Optional[Path] = typer.Option(None, "--output", help="Output file path"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    evaluation_hours: int = typer.Option(1, "--hours", help="Hours of metrics to evaluate"),
) -> None:
    """Generate comprehensive ingestion report for an Obsidian vault."""
    try:
        # Load vault registry and descriptor
        registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
        descriptor = registry.get(vault_id)
        if not descriptor:
            typer.echo(f"Vault {vault_id} not found in registry", err=True)
            raise typer.Exit(code=1)

        # Set up workspace and metrics collection
        workspace = _resolve_workspace(workspace_path)
        metrics_collector = create_metrics_collector()

        # Generate quality gate evaluation if we have metrics
        quality_result = None
        try:
            # Use vault-specific quality gate settings
            vault_settings = descriptor.quality_gate_settings
            config = QualityGateConfig(
                enable_strict_mode=vault_settings.strict_mode,
                max_error_rate=vault_settings.max_error_rate,
                max_critical_error_rate=vault_settings.max_critical_error_rate,
                max_parse_failure_rate=vault_settings.max_parse_failure_rate,
                max_broken_link_rate=vault_settings.max_broken_link_rate,
                min_throughput_events_per_second=vault_settings.min_throughput_events_per_second,
                max_avg_processing_time_seconds=vault_settings.max_avg_processing_time_seconds,
                min_consent_coverage_rate=vault_settings.min_consent_coverage_rate,
                min_asset_processing_success_rate=vault_settings.min_asset_processing_success_rate,
                max_quarantine_rate=vault_settings.max_quarantine_rate,
                evaluation_time_window_hours=evaluation_hours,
                require_minimum_sample_size=vault_settings.require_minimum_sample_size
            )

            if vault_settings.enable_quality_gates:
                evaluator = create_quality_gate_evaluator(config=config, metrics_collector=metrics_collector)
                quality_result = evaluator.evaluate_vault_quality(vault_id, metrics_collector)
            else:
                typer.echo("Quality gates disabled for this vault", err=True)
        except Exception as e:
            typer.echo(f"Warning: Could not evaluate quality gate: {e}", err=True)

        # Generate comprehensive report
        generator = create_report_generator()
        report = generator.generate_report(
            vault_id=vault_id,
            vault_name=descriptor.name,
            quality_gate_result=quality_result
        )

        # Format and output report
        if output_format.lower() == "json":
            formatter = create_json_formatter()
            content = formatter.format_report(report)
            if output_path:
                formatter.write_report(report, output_path)
                typer.echo(f"JSON report written to {output_path}")
            else:
                typer.echo(content)
        else:
            formatter = create_markdown_formatter()
            content = formatter.format_report(report)
            if output_path:
                formatter.write_report(report, output_path)
                typer.echo(f"Markdown report written to {output_path}")
            else:
                typer.echo(content)

        # Exit with appropriate code for CI/CD integration
        raise typer.Exit(code=report.exit_code)

    except Exception as e:
        typer.echo(f"Failed to generate report: {e}", err=True)
        raise typer.Exit(code=2)


@obsidian_app.command("quality-gate")
def obsidian_quality_gate(
    vault_id: str = typer.Argument(..., help="Vault ID to evaluate"),
    strict_mode: bool = typer.Option(False, "--strict", help="Treat warnings as failures"),
    output_format: str = typer.Option("summary", "--format", help="Output format: summary, json, markdown"),
    output_path: Optional[Path] = typer.Option(None, "--output", help="Output file path"),
    registry_path: Optional[Path] = typer.Option(None, help="Override registry directory"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    evaluation_hours: int = typer.Option(1, "--hours", help="Hours of metrics to evaluate"),
    max_error_rate: Optional[float] = typer.Option(None, help="Override max error rate threshold"),
    min_throughput: Optional[float] = typer.Option(None, help="Override min throughput threshold"),
) -> None:
    """Evaluate quality gate for an Obsidian vault with configurable thresholds."""
    try:
        # Load vault registry and descriptor
        registry = VaultRegistry(registry_root=registry_path) if registry_path else VaultRegistry()
        descriptor = registry.get(vault_id)
        if not descriptor:
            typer.echo(f"Vault {vault_id} not found in registry", err=True)
            raise typer.Exit(code=1)

        # Set up workspace and metrics collection
        workspace = _resolve_workspace(workspace_path)
        metrics_collector = create_metrics_collector()

        # Start with vault-specific quality gate settings
        vault_settings = descriptor.quality_gate_settings

        # Check if quality gates are enabled for this vault
        if not vault_settings.enable_quality_gates:
            typer.echo(f"Quality gates disabled for vault {vault_id}", err=True)
            typer.echo("Enable quality gates in vault configuration to use this feature", err=True)
            raise typer.Exit(code=1)

        # Configure quality gate with vault settings and custom overrides
        config = QualityGateConfig(
            enable_strict_mode=strict_mode or vault_settings.strict_mode,
            max_error_rate=max_error_rate or vault_settings.max_error_rate,
            max_critical_error_rate=vault_settings.max_critical_error_rate,
            max_parse_failure_rate=vault_settings.max_parse_failure_rate,
            max_broken_link_rate=vault_settings.max_broken_link_rate,
            min_throughput_events_per_second=min_throughput or vault_settings.min_throughput_events_per_second,
            min_critical_throughput_events_per_second=vault_settings.min_throughput_events_per_second * 0.5,
            max_avg_processing_time_seconds=vault_settings.max_avg_processing_time_seconds,
            max_critical_processing_time_seconds=vault_settings.max_avg_processing_time_seconds * 2,
            min_consent_coverage_rate=vault_settings.min_consent_coverage_rate,
            min_critical_consent_coverage_rate=vault_settings.min_consent_coverage_rate * 0.8,
            min_asset_processing_success_rate=vault_settings.min_asset_processing_success_rate,
            min_critical_asset_success_rate=vault_settings.min_asset_processing_success_rate * 0.8,
            max_quarantine_rate=vault_settings.max_quarantine_rate,
            max_critical_quarantine_rate=vault_settings.max_quarantine_rate * 2,
            evaluation_time_window_hours=evaluation_hours,
            require_minimum_sample_size=vault_settings.require_minimum_sample_size
        )

        # Apply additional custom thresholds if provided
        if max_error_rate is not None:
            config.max_critical_error_rate = max_error_rate * 2

        if min_throughput is not None:
            config.min_critical_throughput_events_per_second = min_throughput * 0.5

        # Evaluate quality gate
        evaluator = create_quality_gate_evaluator(config=config, metrics_collector=metrics_collector)
        result = evaluator.evaluate_vault_quality(vault_id, metrics_collector)

        # Generate output based on format
        if output_format.lower() == "json":
            # Generate full report with JSON output
            generator = create_report_generator()
            report = generator.generate_report(
                vault_id=vault_id,
                vault_name=descriptor.name,
                quality_gate_result=result
            )
            formatter = create_json_formatter()
            content = formatter.format_report(report)

            if output_path:
                formatter.write_report(report, output_path)
                typer.echo(f"JSON quality gate report written to {output_path}")
            else:
                typer.echo(content)

        elif output_format.lower() == "markdown":
            # Generate full report with Markdown output
            generator = create_report_generator()
            report = generator.generate_report(
                vault_id=vault_id,
                vault_name=descriptor.name,
                quality_gate_result=result
            )
            formatter = create_markdown_formatter()
            content = formatter.format_report(report)

            if output_path:
                formatter.write_report(report, output_path)
                typer.echo(f"Markdown quality gate report written to {output_path}")
            else:
                typer.echo(content)

        else:
            # Summary output for quick CI/CD feedback
            status_emoji = {
                "pass": "✅",
                "warn": "⚠️",
                "fail": "❌"
            }

            emoji = status_emoji.get(result.status.value, "ℹ️")
            typer.echo(f"{emoji} Quality Gate: {result.status.value.upper()}")
            typer.echo(f"Vault: {descriptor.name}")
            typer.echo(f"Evaluated: {result.evaluated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            typer.echo(f"Files processed: {result.total_files_processed:,}")
            typer.echo(f"Success rate: {((result.total_files_processed - result.total_files_failed) / max(result.total_files_processed, 1))*100:.1f}%")

            if result.has_failures:
                typer.echo(f"\n❌ CRITICAL ISSUES ({len(result.critical_issues)}):")
                for issue in result.critical_issues:
                    typer.echo(f"   • {issue}")

            if result.has_warnings:
                typer.echo(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
                for warning in result.warnings:
                    typer.echo(f"   • {warning}")

            if result.recommendations:
                typer.echo(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(result.recommendations[:5], 1):
                    typer.echo(f"   {i}. {rec}")

        # Exit with appropriate code for CI/CD integration
        exit_code = result.get_exit_code()

        if exit_code == 0:
            typer.echo(f"\n✅ Quality gate PASSED")
        elif exit_code == 1:
            typer.echo(f"\n⚠️  Quality gate PASSED with warnings")
        else:
            typer.echo(f"\n❌ Quality gate FAILED")

        raise typer.Exit(code=exit_code)

    except Exception as e:
        typer.echo(f"Failed to evaluate quality gate: {e}", err=True)
        raise typer.Exit(code=2)


app.add_typer(obsidian_app, name="obsidian")
app.add_typer(imap_app, name="imap")


@app.command("telemetry")
def show_telemetry(
    workspace_path: Optional[Path] = typer.Option(
        None,
        help="Path to ingestion workspace",
    ),
    raw: bool = typer.Option(False, "--raw", help="Emit raw JSON instead of formatted output"),
) -> None:
    """Display ingestion telemetry summary."""

    workspace = workspace_path or DEFAULT_WORKSPACE_PATH
    summary_path = workspace / TELEMETRY_DIR_NAME / TELEMETRY_SUMMARY_FILE
    if not summary_path.exists():
        typer.echo(f"Telemetry summary not found at {summary_path}")
        raise typer.Exit(code=1)

    summary = json.loads(summary_path.read_text())
    if raw:
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.echo("Telemetry Summary")
    typer.echo(f"Total jobs: {summary.get('total', 0)}")
    typer.echo(f"Succeeded: {summary.get('succeeded', 0)}")
    typer.echo(f"Failed: {summary.get('failed', 0)}")
    avg = summary.get("avg_duration", {})
    for status, duration in avg.items():
        typer.echo(f"Average duration ({status}): {duration:.2f}s")


schedule_app = typer.Typer(help="Manage ingestion schedules")
app.add_typer(schedule_app, name="schedule")


def _describe_schedule(source: LocalIngestionSource) -> str:
    if source.schedule == "@manual":
        return "manual"
    if source.schedule == "@interval" and source.interval_seconds:
        return f"every {int(source.interval_seconds)}s"
    return source.schedule


def _next_run(source: LocalIngestionSource) -> Optional[datetime]:
    now = datetime.utcnow()
    if source.paused:
        return None
    if source.schedule == "@manual":
        return None
    if source.schedule == "@interval" and source.interval_seconds:
        return now + timedelta(seconds=source.interval_seconds)
    if not source.schedule:
        return None
    try:
        iterator = croniter(source.schedule, now)
        return iterator.get_next(datetime)
    except (ValueError, TypeError):
        return None


@schedule_app.command("list")
def schedule_list(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    show_paused: bool = typer.Option(True, help="Include paused sources"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text"),
) -> None:
    sources = _load_sources(config_path)
    output = []
    for source in sources.values():
        if not show_paused and source.paused:
            continue
        entry: Dict[str, object] = {
            "name": source.name,
            "schedule": _describe_schedule(source),
            "priority": source.priority,
            "paused": source.paused,
        }
        next_run = _next_run(source)
        if next_run:
            entry["next_run"] = next_run.isoformat()
        output.append(entry)

    if json_output:
        typer.echo(json.dumps(output, indent=2))
        return

    if not output:
        typer.echo("No schedules configured")
        return

    typer.echo("Source schedules")
    for entry in output:
        line = f"- {entry['name']}: {entry['schedule']} (priority={entry['priority']})"
        if entry.get("next_run"):
            line += f" next={entry['next_run']}"
        if entry["paused"]:
            line += " [paused]"
        typer.echo(line)


def _validate_schedule_args(cron: Optional[str], interval: Optional[float], manual: bool) -> None:
    if manual:
        return
    if interval is not None:
        if interval <= 0:
            raise typer.BadParameter("Interval must be positive")
        LocalIngestionSource(
            name="__interval__",
            root_path=Path.cwd(),
            schedule="@interval",
            interval_seconds=interval,
        )
        return
    if cron:
        try:
            croniter(cron)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter("Invalid cron expression") from exc
        return
    raise typer.BadParameter("Provide --cron, --interval, or --manual")


@schedule_app.command("update")
def schedule_update(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    cron: Optional[str] = typer.Option(None, help="Cron expression"),
    interval: Optional[float] = typer.Option(None, help="Interval in seconds"),
    manual: bool = typer.Option(False, help="Switch to manual-only"),
) -> None:
    _validate_schedule_args(cron, interval, manual)

    def mutate(source: LocalIngestionSource) -> LocalIngestionSource:
        payload = source.model_copy()
        if manual:
            payload.schedule = "@manual"
            payload.interval_seconds = None
        elif interval is not None:
            payload.schedule = "@interval"
            payload.interval_seconds = interval
        elif cron:
            payload.schedule = cron
            payload.interval_seconds = None
        return payload

    updated = _update_source(config_path, source_name, mutate=mutate)
    next_run = _next_run(updated)
    next_repr = next_run.isoformat() if next_run else "manual"
    typer.echo(f"Updated schedule for {updated.name}; next run {next_repr}")


@schedule_app.command("remove")
def schedule_remove(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    updated = _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"schedule": "@manual", "interval_seconds": None}),
    )
    typer.echo(f"Removed schedule for {updated.name}; source is now manual-only")


@app.command("priority")
def set_priority(
    source_name: str = typer.Argument(..., help="Name of the source"),
    level: str = typer.Option("normal", help="Priority level: low|normal|high"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    allowed = {"low", "normal", "high"}
    normalized = level.lower()
    if normalized not in allowed:
        raise typer.BadParameter("Priority must be one of: low, normal, high")

    _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"priority": normalized}),
    )
    typer.echo(f"Set priority for {source_name} to {normalized}")


def _record_operator_event(
    workspace: Path,
    *,
    source: str,
    action: str,
    status: str,
    operator: Optional[str],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    logger = _audit_logger(workspace)
    event = AuditEvent(
        job_id=str(uuid.uuid4()),
        source=source,
        action=action,
        status=status,
        timestamp=datetime.utcnow(),
        operator_action=operator,
        metadata=metadata or {},
    )
    logger.record(event)


@app.command("pause")
def pause_source(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Workspace path for audit logging"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    updated = _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"paused": True}),
    )
    workspace = _resolve_workspace(workspace_path)
    _record_operator_event(
        workspace,
        source=updated.name,
        action="scheduler",
        status="paused",
        operator=operator,
        metadata={"schedule": _describe_schedule(updated)},
    )
    typer.echo(f"Paused automatic ingestion for {updated.name}")


@app.command("resume")
def resume_source(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Workspace path for audit logging"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    updated = _update_source(
        config_path,
        source_name,
        mutate=lambda src: src.model_copy(update={"paused": False}),
    )
    workspace = _resolve_workspace(workspace_path)
    _record_operator_event(
        workspace,
        source=updated.name,
        action="scheduler",
        status="resumed",
        operator=operator,
        metadata={"schedule": _describe_schedule(updated)},
    )
    typer.echo(f"Resumed automatic ingestion for {updated.name}")


@app.command("run")
def trigger_manual_run(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    force: bool = typer.Option(False, help="Allow manual run even when paused"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    sources = _load_sources(config_path)
    if source_name not in sources:
        raise typer.BadParameter(f"Source '{source_name}' not found")

    source = sources[source_name]
    if source.paused and not force:
        typer.echo("Source is paused; use --force to run anyway")
        raise typer.Exit(code=1)
    if not force and source.schedule == "@manual":
        typer.echo("Use schedule update or --force to enqueue a manual job")
        raise typer.Exit(code=1)

    queue = _load_job_queue(workspace)
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
    }
    job = IngestionJob(
        job_id=str(uuid.uuid4()),
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": source.name, "trigger": "manual"},
        priority=priority_map[source.priority],
        scheduled_for=datetime.utcnow(),
    )
    queue.enqueue(job)
    typer.echo(f"Enqueued manual ingestion job {job.job_id} for {source.name}")


@app.command("sync")
def sync_source(
    source_name: str = typer.Argument(..., help="Name of the source"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full resync (ignore file state)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Synchronously process files from a local source (does not require orchestrator)."""
    import time as time_module
    import sqlite3

    workspace = _resolve_workspace(workspace_path)
    sources = _load_sources(config_path)
    if source_name not in sources:
        raise typer.BadParameter(f"Source '{source_name}' not found")

    source = sources[source_name]

    # If force flag is set, clear the state for files from this source
    if force:
        state_db = workspace / "state" / "state.db"
        if state_db.exists():
            conn = sqlite3.connect(str(state_db))
            # Delete state entries for files under this source's root path
            root_prefix = str(source.root_path)
            conn.execute("DELETE FROM file_state WHERE path LIKE ?", (f"{root_prefix}%",))
            conn.commit()
            conn.close()
            if not json_output:
                typer.echo(f"Cleared file state for {source_name}")

    state_store = _load_state_store(workspace)
    connector = _load_connector(workspace, state_store)

    start_time = time_module.time()
    files_processed = 0
    files_failed = 0

    typer.echo(f"Syncing source '{source_name}' from {source.root_path}...")

    try:
        for element in connector.ingest(source):
            files_processed += 1
            if not json_output and files_processed % 10 == 0:
                typer.echo(f"  Processed {files_processed} elements...")
    except Exception as exc:
        files_failed += 1
        if not json_output:
            typer.echo(f"Error during sync: {exc}", err=True)

    # Run entity extraction on parsed files
    entities_extracted = 0
    if files_processed > 0:
        if not json_output:
            typer.echo("Extracting entities from documents...")
        try:
            from futurnal.pipeline.entity_extractor import EntityExtractor
            extractor = EntityExtractor(workspace)
            entities_extracted = extractor.process_all_parsed()
            if not json_output:
                typer.echo(f"  Extracted entities from {entities_extracted} documents")
        except Exception as exc:
            if not json_output:
                typer.echo(f"Warning: Entity extraction failed: {exc}", err=True)

    # Index documents to knowledge graph (ChromaDB + Neo4j) for GraphRAG search
    kg_indexed = 0
    if files_processed > 0:
        if not json_output:
            typer.echo("Indexing to knowledge graph (ChromaDB + Neo4j)...")
        try:
            from futurnal.pipeline.kg_indexer import KnowledgeGraphIndexer
            indexer = KnowledgeGraphIndexer(workspace_dir=workspace)
            index_stats = indexer.index_all_parsed()
            kg_indexed = index_stats.get("chroma_indexed", 0) + index_stats.get("neo4j_indexed", 0)
            if not json_output:
                typer.echo(
                    f"  Indexed {index_stats.get('chroma_indexed', 0)} to ChromaDB, "
                    f"{index_stats.get('neo4j_indexed', 0)} to Neo4j, "
                    f"{index_stats.get('entities_indexed', 0)} entities"
                )
            indexer.close()
        except Exception as exc:
            if not json_output:
                typer.echo(f"Warning: Knowledge graph indexing failed: {exc}", err=True)

    duration = time_module.time() - start_time

    if json_output:
        result = {
            "repo_id": source_name,
            "full_name": source_name,
            "status": "completed" if files_failed == 0 else "completed_with_errors",
            "files_synced": files_processed,
            "bytes_synced": 0,
            "bytes_synced_mb": 0.0,
            "duration_seconds": duration,
            "branches_synced": [],
            "error_message": None,
            "files_processed": files_processed,
            "files_failed": files_failed,
            "entities_extracted": entities_extracted,
            "kg_indexed": kg_indexed,
        }
        print(json.dumps(result))
    else:
        typer.echo(f"Sync complete: {files_processed} elements in {duration:.1f}s")


queue_app = typer.Typer(help="Inspect the job queue")
app.add_typer(queue_app, name="queue")


@queue_app.command("status")
def queue_status(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    limit: int = typer.Option(50, help="Maximum entries to display"),
    status: Optional[JobStatus] = typer.Option(None, help="Filter by job status"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    queue = _load_job_queue(workspace)
    entries = queue.snapshot(status=status, limit=limit)

    if json_output:
        typer.echo(json.dumps(entries, indent=2, default=str))
        return

    if not entries:
        typer.echo("Queue is empty")
        return

    typer.echo("Job queue")
    for entry in entries:
        payload = entry.get("payload", {})
        line = (
            f"- {entry['job_id']} :: {entry['status']} :: source={payload.get('source_name')}"
            f" attempts={entry['attempts']} priority={entry['priority']}"
        )
        if entry.get("scheduled_for"):
            line += f" scheduled_for={entry['scheduled_for']}"
        if entry.get("updated_at"):
            line += f" updated_at={entry['updated_at']}"
        if payload.get("error"):
            line += f" error={payload['error']}"
        typer.echo(line)


@app.command("audit")
def show_audit(
    workspace_path: Optional[Path] = typer.Option(
        None,
        help="Path to ingestion workspace",
    ),
    tail: int = typer.Option(20, min=1, help="Number of recent audit events to display"),
    verify: bool = typer.Option(False, help="Verify log chain integrity before printing"),
) -> None:
    """Print recent audit log entries."""

    workspace = workspace_path or DEFAULT_WORKSPACE_PATH
    audit_path = workspace / AUDIT_DIR_NAME / AUDIT_LOG_FILE
    if not audit_path.exists():
        typer.echo(f"Audit log not found at {audit_path}")
        raise typer.Exit(code=1)

    logger = _audit_logger(workspace)
    if verify and not logger.verify(path=audit_path):
        typer.echo("Audit log integrity check failed")
        raise typer.Exit(code=2)

    lines = audit_path.read_text().splitlines()
    for line in lines[-tail:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            typer.echo(line)
            continue
        typer.echo(
            f"[{payload.get('timestamp')}] {payload.get('source')} {payload.get('action')} {payload.get('status')}"
        )
        redacted_path = payload.get("redacted_path")
        if redacted_path:
            typer.echo(f"  path: {redacted_path} (hash={payload.get('path_hash')})")
        if payload.get("metadata"):
            typer.echo(f"  metadata: {json.dumps(payload['metadata'])}")


@app.command("remove")
def remove_source(
    source_id: str = typer.Argument(..., help="Source ID or name to remove"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Remove a configured source (local or Obsidian vault)."""
    
    # First try as local source
    sources = _load_sources(config_path)
    if source_id in sources:
        source = sources[source_id]
        
        if not yes:
            typer.echo(f"⚠️  About to remove local source:")
            typer.echo(f"   Name: {source.name}")
            # Use redaction to safely display path
            redacted = redact_path(source.root_path).redacted
            typer.echo(f"   Path: {redacted}")
            
            confirm = typer.confirm("Are you sure you want to remove this source?")
            if not confirm:
                typer.echo("❌ Operation cancelled")
                raise typer.Exit(code=0)
        
        sources.pop(source_id)
        _save_sources(config_path, sources.values())
        typer.echo(f"✅ Removed local source '{source_id}'")
        return
    
    # Try as Obsidian vault ID
    try:
        registry = VaultRegistry()
        descriptor = registry.get(source_id)
        
        vault_name = descriptor.name or f"vault-{descriptor.id[:8]}"
        
        if not yes:
            typer.echo(f"⚠️  About to remove Obsidian vault:")
            typer.echo(f"   Name: {vault_name}")
            # Use redaction policy to safely display path
            redaction_policy = descriptor.build_redaction_policy()
            redacted_path = redaction_policy.apply(descriptor.base_path).redacted
            typer.echo(f"   Path: {redacted_path}")
            typer.echo(f"   ID: {source_id}")
            
            confirm = typer.confirm("Are you sure you want to remove this vault?")
            if not confirm:
                typer.echo("❌ Operation cancelled")
                raise typer.Exit(code=0)
        
        registry.remove(source_id)
        typer.echo(f"✅ Removed Obsidian vault '{vault_name}' ({source_id})")
        
    except FileNotFoundError:
        typer.echo(f"❌ Source '{source_id}' not found (tried both local sources and Obsidian vaults)")
        raise typer.Exit(code=1)


@app.command("add")
def add_source(
    source_type: str = typer.Argument(..., help="Source type (e.g., 'obsidian')"),
    path: Path = typer.Option(..., "--path", exists=True, file_okay=False, help="Path to source"),
    name: Optional[str] = typer.Option(None, "--name", help="Human-readable source name"),
    icon: Optional[str] = typer.Option(None, "--icon", help="Emoji or path to icon"),
    redact_title: Optional[str] = typer.Option(None, "--redact-title", help="Comma-separated patterns to redact sensitive titles in logs"),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Add a new source (obsidian vault or other types)."""
    
    if source_type.lower() == "obsidian":
        # Handle Obsidian vault registration
        redact_patterns = [p.strip() for p in (redact_title or "").split(",") if p.strip()] if redact_title else None
        registry = VaultRegistry()
        descriptor = registry.register_path(path, name=name, icon=icon, extra_ignores=[], redact_title_patterns=redact_patterns)
        
        # Show network mount warning if applicable
        warning = descriptor.get_network_warning()
        if warning:
            typer.echo(f"⚠️  WARNING: {warning}", err=True)
        
        # Show empty vault warning if applicable
        empty_warning = descriptor.get_empty_vault_warning()
        if empty_warning:
            typer.echo(f"⚠️  WARNING: {empty_warning}", err=True)
        
        if json_out:
            typer.echo(json.dumps(descriptor.model_dump(mode="json"), indent=2))
        else:
            vault_name = descriptor.name or f"vault-{descriptor.id[:8]}"
            # Use redaction policy to safely display path
            redaction_policy = descriptor.build_redaction_policy()
            redacted_path = redaction_policy.apply(descriptor.base_path).redacted
            typer.echo(f"✅ Registered Obsidian vault '{vault_name}' at {redacted_path}")
            typer.echo(f"   Vault ID: {descriptor.id}")
            
            # Show next steps
            typer.echo("\n📋 Next Steps:")
            typer.echo("   1. Convert to ingestion source:")
            typer.echo(f"      futurnal sources obsidian to-local-source {descriptor.id}")
            typer.echo("   2. Start ingestion:")
            typer.echo(f"      futurnal sources run <source_name>")
            typer.echo("   3. View ingestion reports:")
            typer.echo("      futurnal sources telemetry")
            typer.echo("      futurnal sources audit")
    else:
        typer.echo(f"❌ Unsupported source type: {source_type}")
        typer.echo("Supported types: obsidian")
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect_source(
    source_id: str = typer.Argument(..., help="Source ID or name to inspect"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Inspect a configured source (local or Obsidian vault)."""
    
    # First try as local source
    sources = _load_sources(config_path)
    if source_id in sources:
        source = sources[source_id]
        typer.echo(json.dumps(source.model_dump(mode="json"), indent=2))
        return
    
    # Try as Obsidian vault ID
    try:
        registry = VaultRegistry()
        descriptor = registry.get(source_id)
        typer.echo(json.dumps(descriptor.model_dump(mode="json"), indent=2))
        
    except FileNotFoundError:
        typer.echo(f"❌ Source '{source_id}' not found (tried both local sources and Obsidian vaults)")
        raise typer.Exit(code=1)


quarantine_app = typer.Typer(help="Inspect and manage quarantined files")
app.add_typer(quarantine_app, name="quarantine")


@quarantine_app.command("list")
def quarantine_list(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    reason: Optional[str] = typer.Option(None, help="Filter by reason"),
    source: Optional[str] = typer.Option(None, help="Filter by source name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of entries"),
    summary: bool = typer.Option(False, help="Print aggregated summary"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    entries = list(iter_entries(_load_quarantine_dir(workspace)))

    if reason:
        entries = [entry for entry in entries if entry.reason == reason]
    if source:
        entries = [entry for entry in entries if entry.source == source]
    if limit is not None:
        entries = entries[:limit]

    if summary:
        counts = Counter(entry.reason for entry in entries)
        typer.echo("Quarantine Summary")
        typer.echo(f"Total entries: {len(entries)}")
        for reason_name, count in counts.items():
            typer.echo(f"  {reason_name}: {count}")
        return

    if not entries:
        typer.echo("No quarantine entries found")
        return

    for entry in entries:
        timestamp = entry.timestamp.isoformat() if entry.timestamp else "unknown"
        typer.echo(
            f"- {entry.identifier}: {entry.reason}"
            f" (source={entry.source}, retries={entry.retry_count}, timestamp={timestamp})"
        )
        if entry.redacted_path:
            typer.echo(f"    path: {entry.redacted_path} (hash={entry.path_hash})")


@quarantine_app.command("inspect")
def quarantine_inspect(
    entry_id: str,
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    raw: bool = typer.Option(False, help="Print raw JSON payload"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    entry_path = (_load_quarantine_dir(workspace) / f"{entry_id}.json")
    if not entry_path.exists():
        typer.echo(f"Quarantine entry {entry_id} not found")
        raise typer.Exit(code=1)

    entry = QuarantineEntry.from_file(entry_path)
    if raw:
        typer.echo(json.dumps(entry.to_dict(), indent=2))
        return

    typer.echo(f"id: {entry.identifier}")
    typer.echo(f"source: {entry.source}")
    typer.echo(f"reason: {entry.reason}")
    typer.echo(f"detail: {entry.detail}")
    typer.echo(f"timestamp: {entry.timestamp}")
    typer.echo(f"retries: {entry.retry_count}")
    typer.echo(f"redacted_path: {entry.redacted_path}")
    typer.echo(f"path_hash: {entry.path_hash}")


@quarantine_app.command("retry")
def quarantine_retry(
    entry_id: str,
    source_name: str = typer.Option(..., help="Name of the registered source"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
    dry_run: bool = typer.Option(False, help="Preview retry without executing"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    quarantine_dir = _load_quarantine_dir(workspace)
    entry_path = quarantine_dir / f"{entry_id}.json"

    if not entry_path.exists():
        typer.echo(f"Quarantine entry {entry_id} not found")
        raise typer.Exit(code=1)

    entry = update_entry(entry_path)
    if entry.retry_count >= MAX_RETRIES:
        typer.echo(f"Entry {entry_id} has reached retry limit ({MAX_RETRIES})")
        raise typer.Exit(code=1)

    source = _load_source(config_path, source_name)

    if dry_run:
        redacted = entry.redacted_path or redact_path(entry.path).redacted
        typer.echo(f"Dry run: would retry {redacted} for source {source.name}")
        return

    state_store = _load_state_store(workspace)
    try:
        connector = _load_connector(workspace, state_store)
        records = list(connector.ingest(source))
    finally:
        state_store.close()

    update_entry(
        entry_path,
        retry_count=entry.retry_count + 1,
        last_retry_at=datetime.utcnow().isoformat(),
    )

    if records:
        redacted = entry.redacted_path or redact_path(entry.path).redacted
        # Record audit event for successful retry
        _record_operator_event(
            workspace,
            source=source.name,
            action="quarantine",
            status="retry_succeeded",
            operator=operator,
            metadata={
                "entry_id": entry_id,
                "elements_ingested": len(records),
            },
        )
        # Archive the resolved quarantine entry and refresh telemetry summary
        archive_entry(entry_path, _load_archive_dir(workspace))
        write_summary(_load_quarantine_dir(workspace), workspace / TELEMETRY_DIR_NAME)
        typer.echo(f"Retried {redacted}; {len(records)} elements ingested and entry cleared")
    else:
        redacted = entry.redacted_path or redact_path(entry.path).redacted
        _record_operator_event(
            workspace,
            source=source.name,
            action="quarantine",
            status="retry_noop",
            operator=operator,
            metadata={"entry_id": entry_id},
        )
        # Keep entry; refresh telemetry summary to keep counts accurate
        write_summary(_load_quarantine_dir(workspace), workspace / TELEMETRY_DIR_NAME)
        typer.echo(f"Retried {redacted}; no elements produced. Entry remains for further action")


@quarantine_app.command("dismiss")
def quarantine_dismiss(
    entry_id: str,
    note: Optional[str] = typer.Option(None, help="Reason for dismissal"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    quarantine_dir = _load_quarantine_dir(workspace)
    entry_path = quarantine_dir / f"{entry_id}.json"

    if not entry_path.exists():
        typer.echo(f"Quarantine entry {entry_id} not found")
        raise typer.Exit(code=1)

    note_parts = [note for note in [note, operator] if note]
    if note_parts:
        append_note(entry_path, "; ".join(note_parts))

    # Record audit event and archive the entry
    _record_operator_event(
        workspace,
        source="local-sources",
        action="quarantine",
        status="dismissed",
        operator=operator,
        metadata={"entry_id": entry_id},
    )
    archive_entry(entry_path, _load_archive_dir(workspace))
    # Refresh telemetry summary after dismissal
    write_summary(_load_quarantine_dir(workspace), workspace / TELEMETRY_DIR_NAME)
    typer.echo(f"Dismissed and archived entry {entry_id}")


@quarantine_app.command("summary")
def quarantine_summary(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    summary = write_summary(_load_quarantine_dir(workspace), workspace / TELEMETRY_DIR_NAME)
    typer.echo(json.dumps(summary, indent=2))


consent_app = typer.Typer(help="Manage connector consent decisions")
app.add_typer(consent_app, name="consent")


@consent_app.command("grant")
def consent_grant(
    source_name: str = typer.Argument(..., help="Name of the registered source"),
    scope: str = typer.Option(CONSENT_SCOPE_LOCAL_EXTERNAL, help="Consent scope"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    duration_hours: Optional[int] = typer.Option(None, help="Optional duration for consent"),
    token: Optional[str] = typer.Option(None, help="Optional consent token"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    registry = _load_consent_registry(workspace)
    record = registry.grant(
        source=source_name,
        scope=scope,
        duration_hours=duration_hours,
        token=token,
        operator=operator,
    )
    audit = _audit_logger(workspace)
    audit.record_consent_event(
        job_id=str(uuid.uuid4()),
        source=source_name,
        scope=scope,
        granted=True,
        operator=operator,
        token_hash=record.token_hash,
    )
    typer.echo(f"Granted consent for {source_name}:{scope}")


@consent_app.command("revoke")
def consent_revoke(
    source_name: str = typer.Argument(..., help="Name of the registered source"),
    scope: str = typer.Option(CONSENT_SCOPE_LOCAL_EXTERNAL, help="Consent scope"),
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    operator: Optional[str] = typer.Option(None, help="Operator identifier"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    registry = _load_consent_registry(workspace)
    record = registry.revoke(source=source_name, scope=scope, operator=operator)
    audit = _audit_logger(workspace)
    audit.record_consent_event(
        job_id=str(uuid.uuid4()),
        source=source_name,
        scope=scope,
        granted=False,
        operator=operator,
        token_hash=record.token_hash,
    )
    typer.echo(f"Revoked consent for {source_name}:{scope}")


@consent_app.command("status")
def consent_status(
    workspace_path: Optional[Path] = typer.Option(None, help="Path to ingestion workspace"),
    active_only: bool = typer.Option(True, help="Show only active consents"),
) -> None:
    workspace = _resolve_workspace(workspace_path)
    registry = _load_consent_registry(workspace)
    records = list(registry.iter_active()) if active_only else list(registry.snapshot())
    if not records:
        typer.echo("No consent records found")
        return
    for record in records:
        typer.echo(
            f"- {record.source}:{record.scope} :: granted={record.granted}"
            f" expires={record.expires_at} operator={record.operator}"
        )


if __name__ == "__main__":
    app()



"""Descriptor and registry for Obsidian vaults.

This module defines the persistent metadata model for Obsidian vaults and a
file-based registry stored under the Futurnal workspace sources directory.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from enum import Enum

from filelock import FileLock
from pydantic import BaseModel, Field, field_validator

from futurnal import __version__ as FUTURNAL_VERSION
from ..local.config import LocalIngestionSource
from ...privacy.redaction import RedactionPolicy, redact_path

if TYPE_CHECKING:
    from ...privacy.redaction import RedactedPath


class PrivacyLevel(str, Enum):
    """Privacy levels for vault processing."""
    STRICT = "strict"       # Maximum privacy, minimal data exposure
    STANDARD = "standard"   # Balanced privacy with functionality
    PERMISSIVE = "permissive" # Reduced privacy for enhanced features


class ConsentScope(str, Enum):
    """Granular consent scopes for Obsidian operations."""
    VAULT_SCAN = "obsidian:vault:scan"
    CONTENT_ANALYSIS = "obsidian:vault:content_analysis"
    ASSET_EXTRACTION = "obsidian:vault:asset_extraction"
    LINK_GRAPH_ANALYSIS = "obsidian:vault:link_graph_analysis"
    CLOUD_MODELS = "obsidian:vault:cloud_models"
    METADATA_EXTRACTION = "obsidian:vault:metadata_extraction"


class VaultPrivacySettings(BaseModel):
    """Privacy configuration for an Obsidian vault."""

    privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.STANDARD,
        description="Overall privacy level for this vault"
    )

    required_consent_scopes: List[ConsentScope] = Field(
        default_factory=lambda: [ConsentScope.VAULT_SCAN],
        description="Consent scopes required for vault operations"
    )

    enable_content_redaction: bool = Field(
        default=True,
        description="Enable content redaction in logs and audit trails"
    )

    enable_path_anonymization: bool = Field(
        default=True,
        description="Enable path anonymization in logs"
    )

    tag_based_privacy_classification: bool = Field(
        default=False,
        description="Use note tags for automatic privacy classification"
    )

    privacy_tags: List[str] = Field(
        default_factory=lambda: ["private", "confidential", "personal"],
        description="Tags that trigger enhanced privacy protection"
    )

    audit_content_changes: bool = Field(
        default=True,
        description="Audit content changes (using checksums only)"
    )

    audit_link_changes: bool = Field(
        default=True,
        description="Audit link graph changes (anonymized)"
    )

    retain_audit_days: int = Field(
        default=90,
        description="Number of days to retain audit logs",
        ge=1,
        le=365
    )


class VaultQualityGateSettings(BaseModel):
    """Quality gate configuration for an Obsidian vault."""

    enable_quality_gates: bool = Field(
        default=True,
        description="Enable quality gate evaluation for this vault"
    )

    strict_mode: bool = Field(
        default=False,
        description="Treat warnings as failures in quality gate evaluation"
    )

    max_error_rate: float = Field(
        default=0.05,
        description="Maximum allowed error rate (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    max_critical_error_rate: float = Field(
        default=0.10,
        description="Critical error rate threshold that always fails (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    max_parse_failure_rate: float = Field(
        default=0.02,
        description="Maximum allowed parse failure rate (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    max_broken_link_rate: float = Field(
        default=0.03,
        description="Maximum allowed broken link rate (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    min_throughput_events_per_second: float = Field(
        default=1.0,
        description="Minimum required throughput in events per second",
        ge=0.0
    )

    max_avg_processing_time_seconds: float = Field(
        default=5.0,
        description="Maximum allowed average processing time in seconds",
        ge=0.0
    )

    min_consent_coverage_rate: float = Field(
        default=0.95,
        description="Minimum required consent coverage rate (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    min_asset_processing_success_rate: float = Field(
        default=0.90,
        description="Minimum required asset processing success rate (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    max_quarantine_rate: float = Field(
        default=0.02,
        description="Maximum allowed quarantine rate (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    evaluation_time_window_hours: int = Field(
        default=1,
        description="Time window in hours for metrics evaluation",
        ge=1,
        le=168  # 1 week max
    )

    require_minimum_sample_size: int = Field(
        default=10,
        description="Minimum number of events required for meaningful evaluation",
        ge=1
    )


# Conservative defaults that align with Obsidian conventions and should not be ingested
DEFAULT_OBSIDIAN_IGNORE_RULES: List[str] = [
    ".obsidian/**",
    ".trash/**",
    "Templates/**",
    "template/**",
]


def _normalize_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:
        # On permission errors still return absolute path
        return path.expanduser().absolute()


def _deterministic_vault_id(base_path: Path) -> str:
    normalized = str(_normalize_path(base_path))
    # UUID5 over a stable namespace to ensure determinism across runs
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"obsidian:{normalized}"))


def _machine_id_hash() -> str:
    try:
        node = uuid.getnode()
        host = socket.gethostname()
        payload = f"{node}:{host}:{platform.system()}:{platform.machine()}".encode()
        return sha256(payload).hexdigest()
    except Exception:
        return "unknown"


def _detect_network_mount(path: Path) -> Optional[str]:
    """Detect if path is on a network mount and return warning if so."""
    try:
        # Check if path is on a network filesystem
        if platform.system() == "Darwin":  # macOS
            # Check if path starts with /Volumes/ (common for network mounts)
            if str(path).startswith("/Volumes/"):
                return "Path appears to be on a network volume which may have high latency"
        elif platform.system() == "Linux":
            # Check /proc/mounts for network filesystems
            try:
                with open("/proc/mounts", "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            mount_point, fs_type = parts[1], parts[2]
                            if fs_type in ["nfs", "nfs4", "cifs", "smb", "smbfs"] and str(path).startswith(mount_point):
                                return f"Path is on {fs_type} network filesystem which may have high latency"
            except (FileNotFoundError, PermissionError):
                pass
        elif platform.system() == "Windows":
            # Check for UNC paths
            if str(path).startswith("\\\\"):
                return "Path appears to be a UNC network path which may have high latency"
    except Exception:
        # Don't fail registration due to network detection issues
        pass
    return None


class Provenance(BaseModel):
    os_user: str
    machine_id_hash: str
    tool_version: str


class ObsidianVaultDescriptor(BaseModel):
    """Persistent descriptor for an Obsidian vault."""

    id: str = Field(..., description="Deterministic vault identifier")
    name: Optional[str] = Field(default=None, description="Human-readable label")
    base_path: Path = Field(..., description="Absolute path to the Obsidian vault root")
    icon: Optional[str] = Field(default=None, description="Optional emoji or icon path")
    ignore_rules: List[str] = Field(default_factory=list, description="Pathspec ignore rules")
    redact_title_patterns: List[str] = Field(
        default_factory=list, description="Patterns to mask sensitive note titles in logs"
    )
    privacy_settings: VaultPrivacySettings = Field(
        default_factory=VaultPrivacySettings, description="Privacy configuration for this vault"
    )
    quality_gate_settings: VaultQualityGateSettings = Field(
        default_factory=VaultQualityGateSettings, description="Quality gate configuration for this vault"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance: Provenance

    @field_validator("base_path")
    @classmethod
    def _absolute(cls, value: Path) -> Path:  # type: ignore[override]
        p = _normalize_path(value)
        if not p.is_absolute():
            raise ValueError("base_path must be absolute")
        return p

    @classmethod
    def from_path(
        cls,
        base_path: Path,
        *,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        extra_ignores: Optional[Iterable[str]] = None,
        redact_title_patterns: Optional[Iterable[str]] = None,
        privacy_settings: Optional[VaultPrivacySettings] = None,
        quality_gate_settings: Optional[VaultQualityGateSettings] = None,
    ) -> "ObsidianVaultDescriptor":
        base_path = _normalize_path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Vault path does not exist: {base_path}")
        obsidian_dir = base_path / ".obsidian"
        if not obsidian_dir.exists():
            raise ValueError("Provided path does not look like an Obsidian vault (missing .obsidian/)")

        # Merge defaults + .futurnalignore + provided extra ignores
        rules: List[str] = list(DEFAULT_OBSIDIAN_IGNORE_RULES)
        ignore_file = base_path / ".futurnalignore"
        if ignore_file.exists():
            try:
                for line in ignore_file.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rules.append(line)
            except Exception:
                # Ignore unreadable ignore file; privacy-first
                pass
        if extra_ignores:
            rules.extend([r for r in extra_ignores if r])

        descriptor = cls(
            id=_deterministic_vault_id(base_path),
            name=name,
            base_path=base_path,
            icon=icon,
            ignore_rules=rules,
            redact_title_patterns=list(redact_title_patterns or []),
            privacy_settings=privacy_settings or VaultPrivacySettings(),
            quality_gate_settings=quality_gate_settings or VaultQualityGateSettings(),
            provenance=Provenance(
                os_user=getpass.getuser(),
                machine_id_hash=_machine_id_hash(),
                tool_version=FUTURNAL_VERSION,
            ),
        )
        return descriptor

    def to_local_source(
        self,
        *,
        max_workers: Optional[int] = None,
        max_files_per_batch: Optional[int] = None,
        scan_interval_seconds: Optional[float] = None,
        watcher_debounce_seconds: Optional[float] = None,
        schedule: str = "@manual",
        priority: str = "normal",
    ) -> LocalIngestionSource:
        """Convert to LocalIngestionSource for orchestrator integration."""

        # Use vault name or fallback to a descriptive name
        source_name = self.name or f"obsidian-{self.id[:8]}"

        # Determine privacy settings based on privacy level
        privacy_settings = self.privacy_settings
        allow_plaintext = privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE and not privacy_settings.enable_path_anonymization
        require_consent = len(privacy_settings.required_consent_scopes) > 1  # More than just basic vault scan

        # Build external processing scope from required consent scopes
        external_scopes = [scope.value for scope in privacy_settings.required_consent_scopes
                          if scope != ConsentScope.VAULT_SCAN]
        external_scope = ",".join(external_scopes) if external_scopes else "obsidian.external_processing"

        return LocalIngestionSource(
            name=source_name,
            root_path=self.base_path,
            include=[],  # Obsidian vaults typically include all markdown files
            exclude=self.ignore_rules,  # Use ignore_rules as exclude patterns
            follow_symlinks=False,  # Conservative default for Obsidian vaults
            ignore_file=None,  # Already processed into ignore_rules
            max_workers=max_workers,
            max_files_per_batch=max_files_per_batch,
            scan_interval_seconds=scan_interval_seconds,
            watcher_debounce_seconds=watcher_debounce_seconds,
            allow_plaintext_paths=allow_plaintext,
            require_external_processing_consent=require_consent,
            external_processing_scope=external_scope,
            schedule=schedule,
            priority=priority,
            paused=False,
        )

    def get_network_warning(self) -> Optional[str]:
        """Check if vault is on a network mount and return warning if so."""
        return _detect_network_mount(self.base_path)
    
    def get_empty_vault_warning(self) -> Optional[str]:
        """Check if vault appears to be empty (no .md files) and return warning if so."""
        try:
            # Check if there are any .md files in the vault, excluding ignored directories
            from pathspec import PathSpec
            
            # Create pathspec for ignore rules to exclude ignored directories
            spec = PathSpec.from_lines('gitwildmatch', self.ignore_rules)
            
            # Look for any .md files that aren't ignored
            for md_file in self.base_path.rglob('*.md'):
                relative_path = md_file.relative_to(self.base_path)
                if not spec.match_file(str(relative_path)):
                    return None  # Found at least one non-ignored .md file
            
            # Also check for .markdown files
            for md_file in self.base_path.rglob('*.markdown'):
                relative_path = md_file.relative_to(self.base_path)
                if not spec.match_file(str(relative_path)):
                    return None  # Found at least one non-ignored .markdown file
                    
            return "Vault appears to be empty (no .md files found)"
            
        except Exception:
            # If we can't check, don't warn (privacy-first approach)
            return None

    def build_redaction_policy(self, *, allow_plaintext: Optional[bool] = None) -> RedactionPolicy:
        """Build a redaction policy that respects vault privacy settings and title patterns."""
        import re

        # Use privacy settings to determine redaction behavior
        privacy_settings = self.privacy_settings
        if allow_plaintext is None:
            allow_plaintext = (
                privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE and
                not privacy_settings.enable_path_anonymization
            )

        class ObsidianRedactionPolicy(RedactionPolicy):
            def __init__(
                self,
                title_patterns: List[str],
                privacy_tags: List[str],
                tag_based_classification: bool,
                privacy_level: PrivacyLevel,
                **kwargs
            ):
                super().__init__(**kwargs)
                self.title_patterns = [re.compile(pattern) for pattern in title_patterns]
                self.privacy_tags = privacy_tags
                self.tag_based_classification = tag_based_classification
                self.privacy_level = privacy_level

            def apply(self, path: Path | str) -> "RedactedPath":
                from ...privacy.redaction import RedactedPath

                path_obj = Path(path)
                should_force_redaction = False

                # Check if this is a markdown file for enhanced privacy checks
                if path_obj.suffix.lower() in ['.md', '.markdown']:
                    stem = path_obj.stem

                    # Check title patterns
                    for pattern in self.title_patterns:
                        if pattern.search(stem):
                            should_force_redaction = True
                            break

                    # Check privacy tags if enabled
                    if self.tag_based_classification and not should_force_redaction:
                        for tag in self.privacy_tags:
                            if tag.lower() in stem.lower():
                                should_force_redaction = True
                                break

                # Apply strict redaction for sensitive content
                if should_force_redaction or self.privacy_level == PrivacyLevel.STRICT:
                    temp_policy = RedactionPolicy(
                        reveal_filename=False,
                        reveal_extension=self.reveal_extension if self.privacy_level != PrivacyLevel.STRICT else False,
                        allow_plaintext=False,  # Never allow plaintext for sensitive content
                        segment_hash_length=self.segment_hash_length,
                        path_hash_length=self.path_hash_length,
                    )
                    return temp_policy.apply(path)

                # Use standard redaction
                return super().apply(path)

        return ObsidianRedactionPolicy(
            title_patterns=self.redact_title_patterns,
            privacy_tags=privacy_settings.privacy_tags,
            tag_based_classification=privacy_settings.tag_based_privacy_classification,
            privacy_level=privacy_settings.privacy_level,
            allow_plaintext=allow_plaintext and privacy_settings.enable_path_anonymization,
            reveal_filename=privacy_settings.privacy_level != PrivacyLevel.STRICT,
            reveal_extension=privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE,
        )

    def get_required_consent_scopes(self) -> List[str]:
        """Get list of consent scope strings required for this vault."""
        return [scope.value for scope in self.privacy_settings.required_consent_scopes]

    def requires_consent_for_scope(self, scope: ConsentScope) -> bool:
        """Check if a specific consent scope is required for this vault."""
        return scope in self.privacy_settings.required_consent_scopes

    def get_audit_retention_days(self) -> int:
        """Get audit log retention period for this vault."""
        return self.privacy_settings.retain_audit_days


@dataclass
class VaultRegistry:
    """File-based registry for Obsidian vault descriptors."""

    registry_root: Path
    audit_logger: Optional[Any] = None  # AuditLogger type

    def __init__(self, registry_root: Optional[Path] = None, audit_logger: Optional[Any] = None) -> None:
        default_root = Path.home() / ".futurnal" / "sources" / "obsidian"
        self.registry_root = (registry_root or default_root).expanduser()
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger

    def _descriptor_path(self, vault_id: str) -> Path:
        return self.registry_root / f"{vault_id}.json"

    def _lock_path(self, vault_id: str) -> Path:
        return self.registry_root / f"{vault_id}.json.lock"

    def _log_vault_event(
        self,
        action: str,
        status: str,
        vault_descriptor: ObsidianVaultDescriptor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        operator: Optional[str] = None,
    ) -> None:
        """Log vault lifecycle events to audit logger."""
        if self.audit_logger is None:
            return

        try:
            # Import here to avoid circular dependencies
            from ...privacy.audit import AuditEvent

            # Build redaction policy from vault settings
            policy = vault_descriptor.build_redaction_policy(allow_plaintext=False)

            event_metadata = {
                "vault_id": vault_descriptor.id,
                "vault_name": vault_descriptor.name,
                "privacy_level": vault_descriptor.privacy_settings.privacy_level.value,
                "required_consent_scopes": [scope.value for scope in vault_descriptor.privacy_settings.required_consent_scopes],
                "redact_patterns_count": len(vault_descriptor.redact_title_patterns),
                "ignore_rules_count": len(vault_descriptor.ignore_rules),
                "created_at": vault_descriptor.created_at.isoformat(),
                "updated_at": vault_descriptor.updated_at.isoformat(),
                "tool_version": vault_descriptor.provenance.tool_version,
            }

            if metadata:
                event_metadata.update(metadata)

            # Redact the base path for privacy
            redacted_path = policy.apply(vault_descriptor.base_path)

            event = AuditEvent(
                job_id=f"vault_registry_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                source="obsidian_vault_registry",
                action=f"vault_{action}",
                status=status,
                timestamp=datetime.utcnow(),
                redacted_path=redacted_path.redacted,
                path_hash=redacted_path.path_hash,
                operator_action=operator,
                metadata=event_metadata,
            )

            self.audit_logger.record(event)

        except Exception as e:
            # Don't fail vault operations due to audit logging issues
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to log vault audit event: {e}")

    def _log_vault_privacy_change(
        self,
        vault_descriptor: ObsidianVaultDescriptor,
        previous_settings: Optional[VaultPrivacySettings],
        *,
        operator: Optional[str] = None,
    ) -> None:
        """Log privacy settings changes."""
        if self.audit_logger is None or previous_settings is None:
            return

        current_settings = vault_descriptor.privacy_settings

        # Detect what changed
        changes = {}
        if previous_settings.privacy_level != current_settings.privacy_level:
            changes["privacy_level"] = {
                "from": previous_settings.privacy_level.value,
                "to": current_settings.privacy_level.value,
            }

        if set(previous_settings.required_consent_scopes) != set(current_settings.required_consent_scopes):
            changes["consent_scopes"] = {
                "from": [scope.value for scope in previous_settings.required_consent_scopes],
                "to": [scope.value for scope in current_settings.required_consent_scopes],
            }

        if previous_settings.enable_content_redaction != current_settings.enable_content_redaction:
            changes["content_redaction"] = {
                "from": previous_settings.enable_content_redaction,
                "to": current_settings.enable_content_redaction,
            }

        if previous_settings.enable_path_anonymization != current_settings.enable_path_anonymization:
            changes["path_anonymization"] = {
                "from": previous_settings.enable_path_anonymization,
                "to": current_settings.enable_path_anonymization,
            }

        if changes:
            self._log_vault_event(
                "privacy_updated",
                "success",
                vault_descriptor,
                metadata={"privacy_changes": changes},
                operator=operator,
            )

    def register_path(
        self,
        base_path: Path,
        *,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        extra_ignores: Optional[Iterable[str]] = None,
        redact_title_patterns: Optional[Iterable[str]] = None,
        privacy_settings: Optional[VaultPrivacySettings] = None,
        quality_gate_settings: Optional[VaultQualityGateSettings] = None,
        operator: Optional[str] = None,
    ) -> ObsidianVaultDescriptor:
        descriptor = ObsidianVaultDescriptor.from_path(
            base_path,
            name=name,
            icon=icon,
            extra_ignores=extra_ignores,
            redact_title_patterns=redact_title_patterns,
            privacy_settings=privacy_settings,
            quality_gate_settings=quality_gate_settings,
        )
        return self.add_or_update(descriptor, operator=operator)

    def add_or_update(self, descriptor: ObsidianVaultDescriptor, *, operator: Optional[str] = None) -> ObsidianVaultDescriptor:
        path = self._descriptor_path(descriptor.id)
        lock = FileLock(str(self._lock_path(descriptor.id)))
        with lock:
            now = datetime.utcnow()
            is_update = path.exists()
            previous_settings = None

            if is_update:
                try:
                    existing = self.get(descriptor.id)
                    previous_settings = existing.privacy_settings

                    # Preserve created_at and provenance; update mutable fields
                    updated = existing.model_copy(update={
                        "name": descriptor.name or existing.name,
                        "base_path": descriptor.base_path,
                        "icon": descriptor.icon or existing.icon,
                        "ignore_rules": descriptor.ignore_rules or existing.ignore_rules,
                        "redact_title_patterns": descriptor.redact_title_patterns or existing.redact_title_patterns,
                        "privacy_settings": descriptor.privacy_settings or existing.privacy_settings,
                        "updated_at": now,
                    })
                    self._write(path, updated)

                    # Log update event
                    self._log_vault_event("updated", "success", updated, operator=operator)

                    # Log privacy changes if any
                    self._log_vault_privacy_change(updated, previous_settings, operator=operator)

                    return updated
                except Exception as e:
                    # If corrupt, overwrite with fresh descriptor but keep created_at
                    descriptor.created_at = now
                    self._log_vault_event("update_failed", "error", descriptor,
                                        metadata={"error": str(e)}, operator=operator)

            # New vault registration
            descriptor.updated_at = now
            if not descriptor.created_at:
                descriptor.created_at = now
            self._write(path, descriptor)

            # Log registration event
            if not is_update:
                self._log_vault_event("registered", "success", descriptor, operator=operator)

            return descriptor

    def get(self, vault_id: str) -> ObsidianVaultDescriptor:
        path = self._descriptor_path(vault_id)
        if not path.exists():
            raise FileNotFoundError(f"Vault {vault_id} not found")
        data = json.loads(path.read_text())
        return ObsidianVaultDescriptor.model_validate(data)

    def list(self) -> List[ObsidianVaultDescriptor]:
        items: List[ObsidianVaultDescriptor] = []
        for file in sorted(self.registry_root.glob("*.json")):
            try:
                items.append(ObsidianVaultDescriptor.model_validate(json.loads(file.read_text())))
            except Exception:
                # Skip malformed entries
                continue
        return items

    def list_vaults(self) -> List[ObsidianVaultDescriptor]:
        """Alias for list() for compatibility with existing code."""
        return self.list()

    def find_by_path(self, base_path: Path) -> Optional[ObsidianVaultDescriptor]:
        vid = _deterministic_vault_id(base_path)
        try:
            return self.get(vid)
        except FileNotFoundError:
            return None

    def remove(self, vault_id: str, *, operator: Optional[str] = None) -> None:
        lock = FileLock(str(self._lock_path(vault_id)))
        with lock:
            path = self._descriptor_path(vault_id)
            if path.exists():
                # Get descriptor for audit logging before removal
                try:
                    descriptor = self.get(vault_id)
                    path.unlink()

                    # Log removal event
                    self._log_vault_event("removed", "success", descriptor, operator=operator)

                except Exception as e:
                    # Log failed removal
                    try:
                        descriptor = self.get(vault_id)
                        self._log_vault_event("remove_failed", "error", descriptor,
                                            metadata={"error": str(e)}, operator=operator)
                    except:
                        # Can't even get descriptor, just log basic error
                        if self.audit_logger:
                            from ...privacy.audit import AuditEvent
                            event = AuditEvent(
                                job_id=f"vault_registry_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                                source="obsidian_vault_registry",
                                action="vault_remove_failed",
                                status="error",
                                timestamp=datetime.utcnow(),
                                metadata={"vault_id": vault_id, "error": str(e)},
                                operator_action=operator,
                            )
                            self.audit_logger.record(event)
                    raise

    def _write(self, path: Path, descriptor: ObsidianVaultDescriptor) -> None:
        payload = json.dumps(descriptor.model_dump(mode="json"), indent=2)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)


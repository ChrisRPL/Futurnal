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
from typing import Dict, Iterable, List, Optional

from filelock import FileLock
from pydantic import BaseModel, Field, field_validator

from futurnal import __version__ as FUTURNAL_VERSION
from ..local.config import LocalIngestionSource
from ...privacy.redaction import RedactionPolicy, redact_path


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
            allow_plaintext_paths=False,  # Privacy-first default
            require_external_processing_consent=True,  # Privacy-first default
            external_processing_scope="obsidian.external_processing",
            schedule=schedule,
            priority=priority,
            paused=False,
        )

    def get_network_warning(self) -> Optional[str]:
        """Check if vault is on a network mount and return warning if so."""
        return _detect_network_mount(self.base_path)

    def build_redaction_policy(self, *, allow_plaintext: bool = False) -> RedactionPolicy:
        """Build a redaction policy that respects redact_title_patterns."""
        import re
        
        class ObsidianRedactionPolicy(RedactionPolicy):
            def __init__(self, title_patterns: List[str], **kwargs):
                super().__init__(**kwargs)
                self.title_patterns = [re.compile(pattern) for pattern in title_patterns]
            
            def apply(self, path: Path | str) -> "RedactedPath":
                # Check if this is a note title that should be redacted
                path_obj = Path(path)
                if path_obj.suffix.lower() in ['.md', '.markdown']:
                    stem = path_obj.stem
                    for pattern in self.title_patterns:
                        if pattern.search(stem):
                            # Force redaction for sensitive titles
                            temp_policy = RedactionPolicy(
                                reveal_filename=False,
                                reveal_extension=self.reveal_extension,
                                allow_plaintext=False,  # Override plaintext for sensitive titles
                            )
                            return temp_policy.apply(path)
                
                # Use standard redaction
                return super().apply(path)
        
        return ObsidianRedactionPolicy(
            title_patterns=self.redact_title_patterns,
            allow_plaintext=allow_plaintext,
        )


@dataclass
class VaultRegistry:
    """File-based registry for Obsidian vault descriptors."""

    registry_root: Path

    def __init__(self, registry_root: Optional[Path] = None) -> None:
        default_root = Path.home() / ".futurnal" / "sources" / "obsidian"
        self.registry_root = (registry_root or default_root).expanduser()
        self.registry_root.mkdir(parents=True, exist_ok=True)

    def _descriptor_path(self, vault_id: str) -> Path:
        return self.registry_root / f"{vault_id}.json"

    def _lock_path(self, vault_id: str) -> Path:
        return self.registry_root / f"{vault_id}.json.lock"

    def register_path(
        self,
        base_path: Path,
        *,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        extra_ignores: Optional[Iterable[str]] = None,
        redact_title_patterns: Optional[Iterable[str]] = None,
    ) -> ObsidianVaultDescriptor:
        descriptor = ObsidianVaultDescriptor.from_path(
            base_path,
            name=name,
            icon=icon,
            extra_ignores=extra_ignores,
            redact_title_patterns=redact_title_patterns,
        )
        return self.add_or_update(descriptor)

    def add_or_update(self, descriptor: ObsidianVaultDescriptor) -> ObsidianVaultDescriptor:
        path = self._descriptor_path(descriptor.id)
        lock = FileLock(str(self._lock_path(descriptor.id)))
        with lock:
            now = datetime.utcnow()
            if path.exists():
                try:
                    existing = self.get(descriptor.id)
                    # Preserve created_at and provenance; update mutable fields
                    updated = existing.model_copy(update={
                        "name": descriptor.name or existing.name,
                        "base_path": descriptor.base_path,
                        "icon": descriptor.icon or existing.icon,
                        "ignore_rules": descriptor.ignore_rules or existing.ignore_rules,
                        "redact_title_patterns": descriptor.redact_title_patterns or existing.redact_title_patterns,
                        "updated_at": now,
                    })
                    self._write(path, updated)
                    return updated
                except Exception:
                    # If corrupt, overwrite with fresh descriptor but keep created_at
                    descriptor.created_at = now
            descriptor.updated_at = now
            if not descriptor.created_at:
                descriptor.created_at = now
            self._write(path, descriptor)
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

    def find_by_path(self, base_path: Path) -> Optional[ObsidianVaultDescriptor]:
        vid = _deterministic_vault_id(base_path)
        try:
            return self.get(vid)
        except FileNotFoundError:
            return None

    def remove(self, vault_id: str) -> None:
        lock = FileLock(str(self._lock_path(vault_id)))
        with lock:
            path = self._descriptor_path(vault_id)
            if path.exists():
                path.unlink()

    def _write(self, path: Path, descriptor: ObsidianVaultDescriptor) -> None:
        payload = json.dumps(descriptor.model_dump(mode="json"), indent=2)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)



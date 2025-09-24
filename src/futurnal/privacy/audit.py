"""Tamper-evident audit logging with privacy-aware payloads."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

from .redaction import RedactedPath, RedactionPolicy, redact_path


@dataclass(frozen=True)
class AuditEvent:
    """Represents a structured audit event."""

    job_id: str
    source: str
    action: str
    status: str
    timestamp: datetime
    sha256: Optional[str] = None
    redacted_path: Optional[str] = None
    path_hash: Optional[str] = None
    attempt: Optional[int] = None
    operator_action: Optional[str] = None
    consent_token_hash: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "job_id": self.job_id,
            "source": self.source,
            "action": self.action,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.sha256:
            payload["sha256"] = self.sha256
        if self.redacted_path:
            payload["redacted_path"] = self.redacted_path
        if self.path_hash:
            payload["path_hash"] = self.path_hash
        if self.attempt is not None:
            payload["attempt"] = self.attempt
        if self.operator_action:
            payload["operator_action"] = self.operator_action
        if self.consent_token_hash:
            payload["consent_token_hash"] = self.consent_token_hash
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class AuditLogger:
    """Writes append-only tamper-evident audit logs."""

    output_dir: Path
    filename: str = "audit.log"
    max_bytes: int = 5 * 1024 * 1024
    retention_days: int = 30
    review_dirname: str = "review"
    manifest_name: str = "audit_manifest.json"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / self.filename
        self._manifest_path = self.output_dir / self.manifest_name
        self._review_dir = self.output_dir / self.review_dirname
        self._review_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            self._manifest_path.write_text(json.dumps({"last_hash": None, "rotated": []}, indent=2))

    def record(self, event: AuditEvent) -> None:
        payload = event.to_payload()
        chain_payload = self._augment_with_chain(payload)
        self._append_line(chain_payload)
        self._emit_review_payload(chain_payload)
        self._rotate_if_needed()
        self._prune_old_logs()

    def record_file_event(
        self,
        *,
        job_id: str,
        source: str,
        action: str,
        status: str,
        path: Path | str,
        sha256: Optional[str] = None,
        attempt: Optional[int] = None,
        operator_action: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
        policy: Optional[RedactionPolicy] = None,
    ) -> None:
        redacted: RedactedPath = redact_path(path, policy=policy)
        event = AuditEvent(
            job_id=job_id,
            source=source,
            action=action,
            status=status,
            sha256=sha256,
            timestamp=datetime.utcnow(),
            redacted_path=redacted.redacted,
            path_hash=redacted.path_hash,
            attempt=attempt,
            operator_action=operator_action,
            metadata=metadata or {},
        )
        self.record(event)

    def record_consent_event(
        self,
        *,
        job_id: str,
        source: str,
        scope: str,
        granted: bool,
        operator: Optional[str],
        token_hash: Optional[str],
    ) -> None:
        status = "granted" if granted else "revoked"
        event = AuditEvent(
            job_id=job_id,
            source=source,
            action=f"consent:{scope}",
            status=status,
            timestamp=datetime.utcnow(),
            operator_action=operator,
            consent_token_hash=token_hash,
        )
        self.record(event)

    def verify(self, *, path: Optional[Path] = None) -> bool:
        target = path or self._path
        if not target.exists():
            return True
        previous_hash = self._load_manifest().get("previous_hash")
        for entry in _iter_json_lines(target):
            chain_prev = entry.get("chain_prev")
            if chain_prev != previous_hash:
                return False
            current_hash = entry.get("chain_hash")
            computed = _compute_chain_hash(entry)
            if current_hash != computed:
                return False
            previous_hash = current_hash
        return True

    def _augment_with_chain(self, payload: Dict[str, object]) -> Dict[str, object]:
        manifest = self._load_manifest()
        previous_hash = manifest.get("last_hash")
        augmented = dict(payload)
        augmented["chain_prev"] = previous_hash
        augmented["chain_hash"] = _compute_chain_hash(augmented)
        manifest["last_hash"] = augmented["chain_hash"]
        manifest["previous_hash"] = augmented["chain_hash"]
        self._save_manifest(manifest)
        return augmented

    def _append_line(self, payload: Dict[str, object]) -> None:
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _emit_review_payload(self, payload: Dict[str, object]) -> None:
        review_file = self._review_dir / f"{datetime.utcnow().date().isoformat()}.log"
        with review_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _rotate_if_needed(self) -> None:
        if not self._path.exists():
            return
        if self._path.stat().st_size < self.max_bytes:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        rotated_name = self.output_dir / f"audit-{timestamp}.log"
        os.replace(self._path, rotated_name)
        manifest = self._load_manifest()
        rotated = manifest.get("rotated", [])
        rotated.append({"path": rotated_name.name, "closed_at": datetime.utcnow().isoformat(), "hash": manifest.get("last_hash")})
        manifest["rotated"] = rotated
        self._save_manifest(manifest)

    def _prune_old_logs(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        for file_path in self.output_dir.glob("audit-*.log"):
            timestamp = _extract_timestamp(file_path.name)
            if timestamp and timestamp < cutoff:
                file_path.unlink(missing_ok=True)

    def _load_manifest(self) -> Dict[str, object]:
        return json.loads(self._manifest_path.read_text())

    def _save_manifest(self, manifest: Dict[str, object]) -> None:
        self._manifest_path.write_text(json.dumps(manifest, indent=2))


def _compute_chain_hash(payload: Dict[str, object]) -> str:
    from hashlib import sha256

    canonical = json.dumps({k: payload[k] for k in sorted(payload) if k not in {"chain_hash"}}, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    return digest


def _iter_json_lines(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_timestamp(filename: str) -> Optional[datetime]:
    try:
        stamp = filename.split("-")[1].split(".")[0]
        return datetime.strptime(stamp, "%Y%m%d%H%M%S")
    except (IndexError, ValueError):
        return None

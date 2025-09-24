"""Consent registry for managing privacy approvals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class ConsentRecord:
    """Represents a single consent decision."""

    source: str
    scope: str
    granted: bool
    timestamp: datetime
    expires_at: Optional[datetime]
    token_hash: Optional[str]
    operator: Optional[str]

    def is_active(self, *, reference: Optional[datetime] = None) -> bool:
        if not self.granted:
            return False
        ref = reference or datetime.utcnow()
        if self.expires_at and ref >= self.expires_at:
            return False
        return True


class ConsentRequiredError(RuntimeError):
    """Raised when an operation requires consent that has not been granted."""


class ConsentRegistry:
    """Filesystem-backed registry for consent decisions."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "consent.json"
        if not self._path.exists():
            self._path.write_text(json.dumps({"records": []}, indent=2))

    def grant(
        self,
        *,
        source: str,
        scope: str,
        operator: Optional[str] = None,
        duration_hours: Optional[int] = None,
        token: Optional[str] = None,
    ) -> ConsentRecord:
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=duration_hours) if duration_hours else None
        token_hash = _hash_optional(token) if token else None
        record = ConsentRecord(
            source=source,
            scope=scope,
            granted=True,
            timestamp=now,
            expires_at=expires_at,
            token_hash=token_hash,
            operator=operator,
        )
        data = self._load()
        normalized = _normalize_key(source, scope)
        data[normalized] = _record_to_dict(record)
        self._persist(data)
        return record

    def revoke(self, *, source: str, scope: str, operator: Optional[str] = None) -> ConsentRecord:
        now = datetime.utcnow()
        data = self._load()
        normalized = _normalize_key(source, scope)
        existing = data.get(normalized)
        token_hash = existing.get("token_hash") if existing else None
        record = ConsentRecord(
            source=source,
            scope=scope,
            granted=False,
            timestamp=now,
            expires_at=None,
            token_hash=token_hash,
            operator=operator,
        )
        data[normalized] = _record_to_dict(record)
        self._persist(data)
        return record

    def get(self, *, source: str, scope: str) -> Optional[ConsentRecord]:
        data = self._load()
        normalized = _normalize_key(source, scope)
        payload = data.get(normalized)
        if not payload:
            return None
        return _dict_to_record(payload)

    def require(self, *, source: str, scope: str) -> ConsentRecord:
        record = self.get(source=source, scope=scope)
        if record and record.is_active():
            return record
        raise ConsentRequiredError(f"Consent required for {source}:{scope}")

    def iter_active(self) -> Iterable[ConsentRecord]:
        data = self._load()
        for payload in data.values():
            record = _dict_to_record(payload)
            if record.is_active():
                yield record

    def snapshot(self) -> Iterable[ConsentRecord]:
        data = self._load()
        for payload in data.values():
            yield _dict_to_record(payload)

    def _load(self) -> Dict[str, dict]:
        raw = json.loads(self._path.read_text())
        return {entry["key"]: entry for entry in raw.get("records", [])}

    def _persist(self, records: Dict[str, dict]) -> None:
        ordered = sorted(records.values(), key=lambda item: item["timestamp"])
        payload = {"records": ordered}
        self._path.write_text(json.dumps(payload, indent=2))


def _hash_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    from hashlib import sha256

    return sha256(value.encode("utf-8")).hexdigest()


def _normalize_key(source: str, scope: str) -> str:
    return f"{source}:{scope}"


def _record_to_dict(record: ConsentRecord) -> dict:
    return {
        "key": _normalize_key(record.source, record.scope),
        "source": record.source,
        "scope": record.scope,
        "granted": record.granted,
        "timestamp": record.timestamp.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "token_hash": record.token_hash,
        "operator": record.operator,
    }


def _dict_to_record(payload: dict) -> ConsentRecord:
    return ConsentRecord(
        source=payload["source"],
        scope=payload["scope"],
        granted=bool(payload["granted"]),
        timestamp=datetime.fromisoformat(payload["timestamp"]),
        expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
        token_hash=payload.get("token_hash"),
        operator=payload.get("operator"),
    )

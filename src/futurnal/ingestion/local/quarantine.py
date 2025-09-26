"""Helpers for managing local connector quarantine entries."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from futurnal.privacy.redaction import build_policy, redact_path


QUARANTINE_SUMMARY_FILENAME = "quarantine_summary.json"
MAX_RETRY_ATTEMPTS = 3


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # `fromisoformat` handles the formats we generate.
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class QuarantineEntry:
    identifier: str
    path: str
    reason: str
    detail: str
    timestamp: Optional[datetime]
    retry_count: int
    last_retry_at: Optional[datetime]
    notes: List[str]
    source: Optional[str]
    file_path: Path
    redacted_path: str = field(default="")
    path_hash: Optional[str] = field(default=None)

    @classmethod
    def from_file(cls, path: Path) -> "QuarantineEntry":
        data = json.loads(path.read_text())
        return cls(
            identifier=path.stem,
            path=data.get("path", ""),
            reason=data.get("reason", "unknown"),
            detail=data.get("detail", ""),
            timestamp=_parse_datetime(data.get("timestamp")),
            retry_count=int(data.get("retry_count", 0)),
            last_retry_at=_parse_datetime(data.get("last_retry_at")),
            notes=list(data.get("notes", [])),
            source=data.get("source"),
            file_path=path,
            redacted_path=data.get("redacted_path", ""),
            path_hash=data.get("path_hash"),
        )

    def to_dict(self) -> Dict[str, object]:
        if not self.redacted_path:
            redacted = redact_path(self.path, policy=build_policy())
            self.redacted_path = redacted.redacted
            self.path_hash = redacted.path_hash
        return {
            "path": self.path,
            "reason": self.reason,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "notes": self.notes,
            "source": self.source,
            "redacted_path": self.redacted_path,
            "path_hash": self.path_hash,
        }


def iter_entries(quarantine_dir: Path) -> Iterable[QuarantineEntry]:
    if not quarantine_dir.exists():
        return []
    entries: List[QuarantineEntry] = []
    for file_path in sorted(quarantine_dir.glob("*.json")):
        try:
            entries.append(QuarantineEntry.from_file(file_path))
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def write_summary(quarantine_dir: Path, telemetry_dir: Path) -> Dict[str, object]:
    entries = list(iter_entries(quarantine_dir))
    now = datetime.now(timezone.utc)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    total = len(entries)
    by_reason = Counter(entry.reason for entry in entries)
    retry_pending = sum(1 for entry in entries if entry.retry_count < MAX_RETRY_ATTEMPTS)
    oldest_ts: Optional[datetime] = None
    if entries:
        timestamps = [entry.timestamp for entry in entries if entry.timestamp]
        if timestamps:
            oldest_ts = min(timestamps)

    oldest_age: Optional[float] = None
    if oldest_ts:
        reference = oldest_ts
        if oldest_ts.tzinfo is None:
            reference = oldest_ts.replace(tzinfo=timezone.utc)
        oldest_age = (now - reference).total_seconds()

    summary = {
        "total": total,
        "by_reason": dict(by_reason),
        "retry_pending": retry_pending,
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "oldest_timestamp": oldest_ts.isoformat() if oldest_ts else None,
        "oldest_age_seconds": oldest_age,
        "generated_at": now.isoformat(),
    }

    summary_path = telemetry_dir / QUARANTINE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def update_entry(path: Path, **changes: object) -> QuarantineEntry:
    entry = QuarantineEntry.from_file(path)
    for key, value in changes.items():
        if hasattr(entry, key):
            setattr(entry, key, value)
    data = entry.to_dict()
    path.write_text(json.dumps(data, indent=2))
    return entry


def append_note(path: Path, note: str) -> QuarantineEntry:
    entry = QuarantineEntry.from_file(path)
    entry.notes.append(note)
    path.write_text(json.dumps(entry.to_dict(), indent=2))
    return entry


def remove_entry(path: Path) -> None:
    if path.exists():
        path.unlink()


def archive_entry(path: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / path.name
    path.replace(destination)
    return destination


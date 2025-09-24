"""Helpers for creating quarantine entries in tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class QuarantinePayloadBuilder:
    directory: Path

    def write(
        self,
        identifier: str,
        *,
        path: str,
        reason: str,
        detail: str,
        source: str | None = None,
        timestamp: str = "2024-01-01T00:00:00",
        retry_count: int = 0,
        last_retry_at: str | None = None,
        notes: list[str] | None = None,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "path": path,
            "reason": reason,
            "detail": detail,
            "timestamp": timestamp,
            "retry_count": retry_count,
            "last_retry_at": last_retry_at,
            "notes": notes or [],
            "source": source,
        }
        file_path = self.directory / f"{identifier}.json"
        file_path.write_text(json.dumps(payload))
        return file_path


"""Local audit logging for ingestion operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class AuditLogger:
    """Writes append-only audit entries to a local JSONL file."""

    output_dir: Path
    filename: str = "audit.log"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / self.filename

    def record(self, *, job_id: str, status: str, source: str, detail: dict | None = None) -> None:
        payload = {
            "job_id": job_id,
            "status": status,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if detail:
            payload.update(detail)
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")



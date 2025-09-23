"""Simple telemetry recorder for ingestion jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class TelemetryRecorder:
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(self, job_id: str, duration: float, status: str) -> None:
        entry = {
            "job_id": job_id,
            "duration": duration,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        path = self.output_dir / "telemetry.log"
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")



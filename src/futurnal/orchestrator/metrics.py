"""Simple telemetry recorder for ingestion jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TelemetryRecorder:
    output_dir: Path
    metrics_file: str = "telemetry.log"
    summary_file: str = "telemetry_summary.json"
    _stats: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "succeeded": {"count": 0, "duration": 0.0, "files": 0, "bytes": 0.0},
            "failed": {"count": 0, "duration": 0.0, "files": 0, "bytes": 0.0},
        }
    )

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        job_id: str,
        duration: float,
        status: str,
        *,
        files_processed: Optional[int] = None,
        bytes_processed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        active_workers: Optional[int] = None,
        configured_workers: Optional[int] = None,
        queue_depth: Optional[int] = None,
        effective_throughput: Optional[float] = None,
    ) -> None:
        entry = {
            "job_id": job_id,
            "duration": duration,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if files_processed is not None:
            entry["files_processed"] = files_processed
        if bytes_processed is not None:
            entry["bytes_processed"] = bytes_processed
        metadata_payload: Dict[str, Any] = metadata.copy() if metadata else {}
        if active_workers is not None:
            metadata_payload["active_workers"] = active_workers
        if configured_workers is not None:
            metadata_payload["configured_workers"] = configured_workers
        if queue_depth is not None:
            metadata_payload["queue_depth"] = queue_depth
        if effective_throughput is not None:
            metadata_payload["effective_throughput_bps"] = effective_throughput
        if metadata_payload:
            entry["metadata"] = metadata_payload

        status_stats = self._stats.setdefault(
            status, {"count": 0, "duration": 0.0, "files": 0, "bytes": 0.0}
        )
        status_stats["count"] += 1
        status_stats["duration"] += duration
        if files_processed is not None:
            status_stats["files"] += files_processed
        if bytes_processed is not None:
            status_stats["bytes"] += bytes_processed

        path = self.output_dir / self.metrics_file
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")

        summary = self._build_summary()
        summary_path = self.output_dir / self.summary_file
        summary_path.write_text(json.dumps(summary, indent=2))

    def _build_summary(self) -> Dict[str, Any]:
        statuses: Dict[str, Any] = {}
        total_jobs = 0
        total_duration = 0.0
        total_files = 0
        total_bytes = 0.0
        for status, stats in self._stats.items():
            count = stats.get("count", 0)
            duration = stats.get("duration", 0.0)
            files = int(stats.get("files", 0))
            bytes_processed = float(stats.get("bytes", 0.0))
            avg_duration = duration / count if count else 0.0
            throughput = bytes_processed / duration if duration else 0.0
            statuses[status] = {
                "count": count,
                "files": files,
                "bytes": bytes_processed,
                "avg_duration": avg_duration,
                "throughput_bytes_per_second": throughput,
            }
            total_jobs += count
            total_duration += duration
            total_files += files
            total_bytes += bytes_processed

        overall_throughput = total_bytes / total_duration if total_duration else 0.0
        summary = {
            "overall": {
                "jobs": total_jobs,
                "files": total_files,
                "bytes": total_bytes,
                "avg_duration": total_duration / total_jobs if total_jobs else 0.0,
                "throughput_bytes_per_second": overall_throughput,
            },
            "statuses": statuses,
        }
        return summary



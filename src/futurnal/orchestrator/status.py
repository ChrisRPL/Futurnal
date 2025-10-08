"""Orchestrator status collection for operator dashboards."""

from __future__ import annotations

import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .queue import JobQueue
from .quarantine import QuarantineStore
from .source_control import PausedSourcesRegistry


def collect_status_report(
    *,
    workspace_path: Path,
    job_queue: Optional[JobQueue] = None,
    quarantine_store: Optional[QuarantineStore] = None,
    paused_sources_registry: Optional[PausedSourcesRegistry] = None,
) -> Dict[str, Any]:
    """Collect comprehensive orchestrator status for operator dashboards.

    Args:
        workspace_path: Path to Futurnal workspace directory
        job_queue: Optional JobQueue instance (will create if not provided)
        quarantine_store: Optional QuarantineStore instance (will create if not provided)
        paused_sources_registry: Optional PausedSourcesRegistry (will create if not provided)

    Returns:
        Status report dict with keys:
        - queue: Job queue metrics (pending, running, completed, failed, quarantined)
        - workers: Worker metrics (active, max, utilization)
        - system: System resource metrics (CPU, memory, disk)
        - throughput: Recent throughput metrics (files, bytes, rate)
        - sources: List of configured sources with status
    """
    workspace = Path(workspace_path)

    # Initialize components if not provided
    if job_queue is None:
        queue_db = workspace / "queue" / "jobs.db"
        if queue_db.exists():
            job_queue = JobQueue(queue_db)

    if quarantine_store is None:
        quarantine_db = workspace / "quarantine" / "quarantine.db"
        if quarantine_db.exists():
            quarantine_store = QuarantineStore(quarantine_db)

    if paused_sources_registry is None:
        paused_sources_file = workspace / "orchestrator" / "paused_sources.json"
        paused_sources_registry = PausedSourcesRegistry(paused_sources_file)

    # Collect queue metrics
    queue_metrics = _collect_queue_metrics(job_queue, quarantine_store)

    # Collect worker metrics
    worker_metrics = _collect_worker_metrics(workspace)

    # Collect system metrics
    system_metrics = _collect_system_metrics(workspace)

    # Collect throughput metrics
    throughput_metrics = _collect_throughput_metrics(workspace)

    # Collect source metrics
    source_metrics = _collect_source_metrics(workspace, paused_sources_registry)

    return {
        "queue": queue_metrics,
        "workers": worker_metrics,
        "system": system_metrics,
        "throughput": throughput_metrics,
        "sources": source_metrics,
    }


def _collect_queue_metrics(
    job_queue: Optional[JobQueue],
    quarantine_store: Optional[QuarantineStore],
) -> Dict[str, int]:
    """Collect job queue statistics.

    Args:
        job_queue: JobQueue instance
        quarantine_store: QuarantineStore instance

    Returns:
        Dict with pending, running, completed_24h, failed_24h, quarantined counts
    """
    if not job_queue:
        return {
            "pending": 0,
            "running": 0,
            "completed_24h": 0,
            "failed_24h": 0,
            "quarantined": 0,
        }

    since_24h = datetime.utcnow() - timedelta(days=1)

    metrics = {
        "pending": job_queue.pending_count(),
        "running": job_queue.running_count(),
        "completed_24h": job_queue.completed_count(since=since_24h),
        "failed_24h": job_queue.failed_count(since=since_24h),
        "quarantined": 0,
    }

    if quarantine_store:
        stats = quarantine_store.statistics()
        metrics["quarantined"] = stats["total_quarantined"]

    return metrics


def _collect_worker_metrics(workspace: Path) -> Dict[str, Any]:
    """Collect worker statistics from recent telemetry.

    Args:
        workspace: Workspace directory path

    Returns:
        Dict with active, max, utilization worker metrics
    """
    # Try to read latest telemetry for worker info
    telemetry_file = workspace / "telemetry" / "telemetry.log"

    active_workers = 0
    configured_workers = 0

    if telemetry_file.exists():
        try:
            # Read last few lines to get recent worker counts
            with telemetry_file.open("r") as f:
                lines = f.readlines()
                # Check last 100 lines for recent worker metadata
                for line in reversed(lines[-100:]):
                    try:
                        entry = json.loads(line.strip())
                        metadata = entry.get("metadata", {})
                        if "active_workers" in metadata:
                            active_workers = metadata["active_workers"]
                        if "configured_workers" in metadata:
                            configured_workers = metadata["configured_workers"]
                        if active_workers and configured_workers:
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue
        except (FileNotFoundError, IOError):
            pass

    # Fallback: estimate from system if no telemetry
    if configured_workers == 0:
        import os
        hardware_cpu_count = os.cpu_count() or 4
        configured_workers = max(1, min(hardware_cpu_count, 8))

    utilization = (active_workers / configured_workers * 100.0) if configured_workers > 0 else 0.0

    return {
        "active": active_workers,
        "max": configured_workers,
        "utilization": utilization,
    }


def _collect_system_metrics(workspace: Path) -> Dict[str, Any]:
    """Collect system resource metrics using psutil.

    Args:
        workspace: Workspace directory path

    Returns:
        Dict with CPU, memory, disk metrics
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(workspace))

    return {
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count() or 1,
        "memory_used_gb": memory.used / (1024 ** 3),
        "memory_total_gb": memory.total / (1024 ** 3),
        "memory_percent": memory.percent,
        "disk_free_gb": disk.free / (1024 ** 3),
        "disk_total_gb": disk.total / (1024 ** 3),
        "disk_percent": disk.percent,
    }


def _collect_throughput_metrics(workspace: Path) -> Dict[str, Any]:
    """Collect throughput statistics from recent telemetry.

    Args:
        workspace: Workspace directory path

    Returns:
        Dict with files, bytes, rate for last hour
    """
    telemetry_file = workspace / "telemetry" / "telemetry.log"

    total_files = 0
    total_bytes = 0
    total_duration = 0.0
    cutoff_time = datetime.utcnow() - timedelta(hours=1)

    if telemetry_file.exists():
        try:
            with telemetry_file.open("r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        timestamp_str = entry.get("timestamp")
                        if not timestamp_str:
                            continue

                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if timestamp < cutoff_time:
                            continue

                        # Only count succeeded jobs
                        if entry.get("status") != "succeeded":
                            continue

                        total_files += entry.get("files_processed", 0)
                        total_bytes += entry.get("bytes_processed", 0)
                        total_duration += entry.get("duration", 0.0)

                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
        except (FileNotFoundError, IOError):
            pass

    rate_bps = (total_bytes / total_duration) if total_duration > 0 else 0.0

    return {
        "files_last_hour": total_files,
        "bytes_last_hour": total_bytes,
        "rate_bytes_per_second": rate_bps,
    }


def _collect_source_metrics(
    workspace: Path,
    paused_sources_registry: PausedSourcesRegistry,
) -> List[Dict[str, Any]]:
    """Collect configured source information from workspace.

    Discovers sources by scanning workspace directories and combines
    with pause state and last run times from telemetry.

    Args:
        workspace: Workspace directory path
        paused_sources_registry: Pause state registry

    Returns:
        List of source dicts with name, type, status, last_run
    """
    sources: List[Dict[str, Any]] = []
    paused_set = set(paused_sources_registry.list_paused())

    # Scan for local file sources
    local_sources_dir = workspace / "sources" / "local"
    if local_sources_dir.exists():
        for source_dir in local_sources_dir.iterdir():
            if source_dir.is_dir():
                sources.append({
                    "name": source_dir.name,
                    "type": "local_files",
                    "status": "paused" if source_dir.name in paused_set else "active",
                    "last_run": None,  # TODO: extract from telemetry
                })

    # Scan for IMAP sources
    imap_sources_dir = workspace / "sources" / "imap"
    if imap_sources_dir.exists():
        for source_dir in imap_sources_dir.iterdir():
            if source_dir.is_dir():
                # Load descriptor to get email address
                descriptor_file = source_dir / "descriptor.json"
                if descriptor_file.exists():
                    try:
                        descriptor = json.loads(descriptor_file.read_text())
                        email = descriptor.get("email_address", source_dir.name)
                        source_name = f"imap-{source_dir.name[:8]}"
                        sources.append({
                            "name": email,
                            "type": "imap_mailbox",
                            "status": "paused" if source_name in paused_set else "active",
                            "last_run": None,
                        })
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Scan for GitHub sources
    github_sources_dir = workspace / "sources" / "github"
    if github_sources_dir.exists():
        for source_dir in github_sources_dir.iterdir():
            if source_dir.is_dir():
                # Load descriptor to get repo name
                descriptor_file = source_dir / "descriptor.json"
                if descriptor_file.exists():
                    try:
                        descriptor = json.loads(descriptor_file.read_text())
                        repo_name = descriptor.get("full_name", source_dir.name)
                        source_name = f"github-{source_dir.name[:8]}"
                        sources.append({
                            "name": repo_name,
                            "type": "github_repository",
                            "status": "paused" if source_name in paused_set else "active",
                            "last_run": None,
                        })
                    except (json.JSONDecodeError, KeyError):
                        pass

    return sources

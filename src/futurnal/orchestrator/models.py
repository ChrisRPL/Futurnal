"""Domain models for the ingestion orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional


class JobPriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()


class JobType(str, Enum):
    LOCAL_FILES = "local_files"
    OBSIDIAN_VAULT = "obsidian_vault"
    IMAP_MAILBOX = "imap_mailbox"
    GITHUB_REPOSITORY = "github_repository"

    # AGI Phase 6: Autonomous Insight Jobs
    INSIGHT_GENERATION = "insight_generation"
    CORRELATION_SCAN = "correlation_scan"
    CURIOSITY_SCAN = "curiosity_scan"
    LEARNING_UPDATE = "learning_update"


@dataclass(slots=True)
class IngestionJob:
    """Represents an ingestion job to be processed by the orchestrator."""

    job_id: str
    job_type: JobType
    payload: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    scheduled_for: Optional[datetime] = None



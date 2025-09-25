"""Configuration models for the local files connector."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from croniter import croniter, is_valid
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from pathspec import PathSpec


class LocalIngestionSource(BaseModel):
    """User-defined ingestion source description."""

    name: str = Field(..., description="Unique identifier for the source")
    root_path: Path = Field(..., description="Root directory for ingestion")
    include: List[str] = Field(default_factory=list, description="Glob patterns to include")
    exclude: List[str] = Field(default_factory=list, description="Glob patterns to exclude")
    follow_symlinks: bool = Field(False, description="Follow symlinks during traversal")
    ignore_file: Optional[Path] = Field(
        default=None,
        description="Optional path to .futurnalignore-like file with patterns",
    )
    max_workers: Optional[int] = Field(
        default=None,
        description="Upper bound on concurrent ingestion workers for this source",
        ge=1,
        le=32,
    )
    max_files_per_batch: Optional[int] = Field(
        default=None,
        description="Maximum number of files to ingest per worker batch",
        ge=1,
        le=1000,
    )
    scan_interval_seconds: Optional[float] = Field(
        default=None,
        description="Interval for periodic scans when file watcher is unavailable",
        gt=0.0,
        le=3600.0,
    )
    watcher_debounce_seconds: Optional[float] = Field(
        default=None,
        description="Minimum seconds between watcher-triggered job enqueues",
        ge=0.0,
        le=120.0,
    )
    allow_plaintext_paths: bool = Field(
        default=False,
        description="Expose plaintext paths to audit outputs instead of redacted values",
    )
    require_external_processing_consent: bool = Field(
        default=False,
        description="Require explicit consent before escalating files to external parsers",
    )
    external_processing_scope: str = Field(
        default="local.external_processing",
        description="Consent scope identifier for external processing decisions",
    )
    schedule: str = Field(
        default="@manual",
        description="Cron expression for scheduled ingestion or '@manual' for manual-only",
    )
    interval_seconds: Optional[float] = Field(
        default=None,
        description="Interval in seconds for '@interval' scheduled ingestion",
        gt=0.0,
        le=86400.0,
    )
    priority: str = Field(
        default="normal",
        description="Job priority for scheduled ingestion (low, normal, high)",
        pattern="^(low|normal|high)$",
    )
    paused: bool = Field(
        default=False,
        description="When true, suppress automatic ingestion for this source",
    )

    @field_validator("root_path")
    def _validate_root(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Ingestion root does not exist: {value}")
        if not value.is_dir():
            raise ValueError(f"Ingestion root must be a directory: {value}")
        return value.resolve()

    @model_validator(mode="after")
    def _validate_schedule(self):  # type: ignore[override]
        if self.schedule == "@manual":
            return self
        if self.schedule == "@interval":
            if self.interval_seconds is None:
                raise ValueError("interval_seconds is required when schedule is '@interval'")
            return self
        if not is_valid(self.schedule):
            raise ValueError(f"Invalid cron expression: {self.schedule}")
        self.interval_seconds = None
        return self

    def build_pathspec(self) -> PathSpec:
        patterns: List[str] = []
        if self.ignore_file and self.ignore_file.exists():
            patterns.extend(self.ignore_file.read_text().splitlines())
        patterns.extend(f"!{pattern}" for pattern in self.include)
        patterns.extend(self.exclude)
        return PathSpec.from_lines("gitwildmatch", patterns)

    def allows_plaintext(self) -> bool:
        return self.allow_plaintext_paths

    def workspace_subdir(self) -> Path:
        return Path(self.name)


class LocalIngestionConfig(RootModel[List[LocalIngestionSource]]):
    """Collection of local ingestion sources."""

    root: List[LocalIngestionSource]


def load_config_from_dict(data: dict) -> LocalIngestionConfig:
    """Instantiate configuration from a dictionary."""

    sources = data.get("sources", [])
    return LocalIngestionConfig.model_validate([LocalIngestionSource(**src) for src in sources])



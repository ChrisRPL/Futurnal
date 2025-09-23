"""Configuration models for the local files connector."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, RootModel, validator
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

    @validator("root_path")
    def _validate_root(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Ingestion root does not exist: {value}")
        if not value.is_dir():
            raise ValueError(f"Ingestion root must be a directory: {value}")
        return value.resolve()

    def build_pathspec(self) -> PathSpec:
        patterns: List[str] = []
        if self.ignore_file and self.ignore_file.exists():
            patterns.extend(self.ignore_file.read_text().splitlines())
        patterns.extend(f"!{pattern}" for pattern in self.include)
        patterns.extend(self.exclude)
        return PathSpec.from_lines("gitwildmatch", patterns)


class LocalIngestionConfig(RootModel[List[LocalIngestionSource]]):
    """Collection of local ingestion sources."""

    root: List[LocalIngestionSource]


def load_config_from_dict(data: dict) -> LocalIngestionConfig:
    """Instantiate configuration from a dictionary."""

    sources = data.get("sources", [])
    return LocalIngestionConfig.model_validate([LocalIngestionSource(**src) for src in sources])



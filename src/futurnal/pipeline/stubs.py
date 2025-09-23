"""Stubs for downstream normalization and storage pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NormalizationSink:
    """Writes ingested elements to structured handoff for downstream processing."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def handle(self, element: dict) -> None:
        filename = f"{element['sha256']}.json"
        path = self.output_dir / filename
        data = {
            "source": element["source"],
            "path": element["path"],
            "element_path": element["element_path"],
        }
        path.write_text(json.dumps(data, indent=2))



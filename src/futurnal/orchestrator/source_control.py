"""Source pause/resume state management for orchestrator."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional, Set


class PausedSourcesRegistry:
    """Thread-safe registry for tracking paused ingestion sources.

    Persists pause state to JSON file so CLI commands and orchestrator
    can coordinate source scheduling without IPC.
    """

    def __init__(self, registry_path: Path) -> None:
        """Initialize pause state registry.

        Args:
            registry_path: Path to paused_sources.json file
        """
        self._path = registry_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Initialize file if it doesn't exist
        if not self._path.exists():
            self._save(set())

    def is_paused(self, source_name: str) -> bool:
        """Check if source is currently paused.

        Args:
            source_name: Name of the ingestion source

        Returns:
            True if source is paused, False otherwise
        """
        paused = self._load()
        return source_name in paused

    def pause(self, source_name: str) -> None:
        """Pause a source (prevent scheduled jobs from being enqueued).

        Args:
            source_name: Name of the ingestion source to pause
        """
        with self._lock:
            paused = self._load()
            paused.add(source_name)
            self._save(paused)

    def resume(self, source_name: str) -> None:
        """Resume a paused source (allow scheduled jobs to be enqueued).

        Args:
            source_name: Name of the ingestion source to resume

        Raises:
            ValueError: If source is not currently paused
        """
        with self._lock:
            paused = self._load()
            if source_name not in paused:
                raise ValueError(f"Source {source_name} is not paused")
            paused.discard(source_name)
            self._save(paused)

    def list_paused(self) -> List[str]:
        """Get list of all currently paused sources.

        Returns:
            List of paused source names, sorted alphabetically
        """
        paused = self._load()
        return sorted(list(paused))

    def _load(self) -> Set[str]:
        """Load paused sources from JSON file.

        Returns:
            Set of paused source names
        """
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(data)
            return set()
        except (json.JSONDecodeError, FileNotFoundError):
            return set()

    def _save(self, paused: Set[str]) -> None:
        """Save paused sources to JSON file.

        Args:
            paused: Set of paused source names
        """
        data = sorted(list(paused))
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

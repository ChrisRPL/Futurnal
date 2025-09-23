"""Local Files Connector implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from unstructured.partition.auto import partition

from .config import LocalIngestionSource
from .scanner import FileSnapshot, detect_deletions, walk_directory
from .state import FileRecord, StateStore, compute_sha256

logger = logging.getLogger(__name__)


class LocalFilesConnector:
    """Connector responsible for ingesting local files into Futurnal pipelines."""

    def __init__(self, *, workspace_dir: Path | str, state_store: StateStore) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._parsed_dir = self._workspace_dir / "parsed"
        self._parsed_dir.mkdir(parents=True, exist_ok=True)
        self._state_store = state_store

    def crawl_source(self, source: LocalIngestionSource) -> List[FileRecord]:
        """Perform a crawl of the provided source returning updated records."""

        pathspec = source.build_pathspec()
        snapshots: List[FileRecord] = []
        current_paths: List[Path] = []

        for snapshot in walk_directory(
            source.root_path,
            include_spec=pathspec,
            follow_symlinks=source.follow_symlinks,
        ):
            current_paths.append(snapshot.path)
            record = self._process_snapshot(snapshot)
            if record:
                snapshots.append(record)

        self._handle_deletions(current_paths)
        return snapshots

    def ingest(self, source: LocalIngestionSource) -> Iterable[dict]:
        """Yield parsed document elements for the given source."""

        for record in self.crawl_source(source):
            logger.debug("Parsing %s", record.path)
            elements = partition(filename=str(record.path), strategy="fast", include_metadata=True)
            for element in elements:
                yield self._persist_element(source, record, element)

    def _persist_element(self, source: LocalIngestionSource, record: FileRecord, element) -> dict:
        storage_path = self._parsed_dir / f"{record.sha256}.json"
        storage_path.write_text(str(element))
        return {
            "source": source.name,
            "path": str(record.path),
            "sha256": record.sha256,
            "element_path": str(storage_path),
        }

    def _process_snapshot(self, snapshot: FileSnapshot) -> FileRecord | None:
        current_hash = compute_sha256(snapshot.path)
        existing = self._state_store.fetch(snapshot.path)
        if existing and existing.sha256 == current_hash and existing.mtime == snapshot.mtime:
            logger.debug("Skipping unchanged file %s", snapshot.path)
            return None

        record = FileRecord(
            path=snapshot.path,
            size=snapshot.size,
            mtime=snapshot.mtime,
            sha256=current_hash,
        )
        self._state_store.upsert(record)
        return record

    def _handle_deletions(self, current_paths: Iterable[Path]) -> None:
        previous_paths = [record.path for record in self._state_store.iter_all()]
        removed = detect_deletions(previous_paths, current_paths)
        for path in removed:
            logger.debug("Removing deleted file %s", path)
            self._state_store.remove(path)



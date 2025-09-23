"""Directory scanning utilities for the local connector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from pathspec import PathSpec


@dataclass(frozen=True)
class FileSnapshot:
    """Represents a snapshot of file metadata used for change detection."""

    path: Path
    size: int
    mtime: float


def walk_directory(
    root: Path,
    *,
    include_spec: PathSpec,
    follow_symlinks: bool = False,
) -> Iterator[FileSnapshot]:
    """Yield file snapshots under the root directory honoring ignore patterns."""

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        current_dir = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not include_spec.match_file(str((current_dir / name).relative_to(root)))
        ]

        for filename in filenames:
            path = current_dir / filename
            rel_path = str(path.relative_to(root))
            if include_spec.match_file(rel_path):
                continue
            stat = path.stat()
            yield FileSnapshot(path=path, size=stat.st_size, mtime=stat.st_mtime)


def detect_deletions(previous_paths: Iterable[Path], current_paths: Iterable[Path]) -> set[Path]:
    """Return set of paths that were present before but not in current snapshot."""

    previous = {Path(p).resolve() for p in previous_paths}
    current = {Path(p).resolve() for p in current_paths}
    return previous.difference(current)



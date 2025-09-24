"""Utility builders for Local Files Connector integration test fixtures."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class FixtureResult:
    """Container describing a generated fixture."""

    root: Path
    files: List[Path] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


def create_nested_fixture(base_dir: Path) -> FixtureResult:
    """Create nested directories with mixed file types and hidden files."""

    root = base_dir / "nested"
    root.mkdir(parents=True, exist_ok=True)

    docs_dir = root / "docs" / "deep"
    docs_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = root / ".hidden"
    hidden_dir.mkdir()

    files: List[Path] = []

    readme = root / "README.md"
    readme.write_text("# Futurnal Test\n\nThis is a root-level document.\n", encoding="utf-8")
    files.append(readme)

    deep_note = docs_dir / "deep_note.md"
    deep_note.write_text("Deeply nested note for integration tests.", encoding="utf-8")
    files.append(deep_note)

    binary = docs_dir / "attachment.bin"
    binary.write_bytes(os.urandom(512))
    files.append(binary)

    hidden_file = hidden_dir / ".hidden_note.txt"
    hidden_file.write_text("Hidden content should still be ingested.", encoding="utf-8")
    files.append(hidden_file)

    return FixtureResult(root=root, files=files)


def create_symlink_fixture(base_dir: Path) -> FixtureResult:
    """Create fixture with valid and broken symlinks."""

    root = base_dir / "symlinks"
    root.mkdir(parents=True, exist_ok=True)

    target = root / "target.txt"
    target.write_text("Symlink target content", encoding="utf-8")

    valid_link = root / "valid_link.txt"
    broken_link = root / "broken_link.txt"

    files = [target]
    metadata: Dict[str, object] = {"symlinks": []}

    if hasattr(os, "symlink"):
        try:
            os.symlink(target.name, valid_link)
            metadata["symlinks"].append(valid_link)
        except OSError:
            # Symlink creation may fail due to permissions (e.g., Windows without admin).
            pass
        try:
            os.symlink("non-existent.txt", broken_link)
            metadata["symlinks"].append(broken_link)
        except OSError:
            pass

    return FixtureResult(root=root, files=files, metadata=metadata)


def create_sparse_large_file_fixture(base_dir: Path, size_bytes: int = 100 * 1024 * 1024) -> FixtureResult:
    """Create fixture with a sparse file approximating ≥100 MB."""

    root = base_dir / "large"
    root.mkdir(parents=True, exist_ok=True)

    sparse_file = root / "large_file.bin"
    with sparse_file.open("wb") as fh:
        if size_bytes > 0:
            fh.seek(size_bytes - 1)
            fh.write(b"\0")

    return FixtureResult(root=root, files=[sparse_file], metadata={"size_bytes": size_bytes})


def create_permission_locked_fixture(base_dir: Path) -> FixtureResult:
    """Create fixture containing a file without read permissions."""

    root = base_dir / "permissions"
    root.mkdir(parents=True, exist_ok=True)

    readable = root / "readable.txt"
    readable.write_text("Readable content", encoding="utf-8")

    locked = root / "locked.txt"
    locked.write_text("This file will have permissions removed.", encoding="utf-8")
    locked.chmod(0)

    metadata = {"locked_file": locked, "reset_mode": stat.S_IRUSR | stat.S_IWUSR}

    return FixtureResult(root=root, files=[readable], metadata=metadata)


def create_concurrent_modification_fixture(base_dir: Path) -> FixtureResult:
    """Create fixture that can be mutated during tests to simulate concurrent edits."""

    root = base_dir / "concurrency"
    root.mkdir(parents=True, exist_ok=True)

    tracked = root / "tracked.md"
    tracked.write_text("Initial content", encoding="utf-8")

    return FixtureResult(root=root, files=[tracked], metadata={"mutable_file": tracked})



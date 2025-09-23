"""Tests for directory scanner utilities."""

from pathlib import Path

from pathspec import PathSpec

from futurnal.ingestion.local import scanner


def test_walk_directory_respects_ignore(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "keep.txt").write_text("keep")
    (root / "ignore.tmp").write_text("ignore")

    spec = PathSpec.from_lines("gitwildmatch", ["*.tmp"])
    results = list(scanner.walk_directory(root, include_spec=spec))

    assert len(results) == 1
    assert results[0].path.name == "keep.txt"


def test_detect_deletions(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    deleted = tmp_path / "deleted.txt"

    paths_before = [first, second, deleted]
    paths_after = [first, second]

    removed = scanner.detect_deletions(paths_before, paths_after)
    assert removed == {deleted.resolve()}



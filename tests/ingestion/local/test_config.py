"""Tests for local ingestion configuration models."""

from pathlib import Path

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource, load_config_from_dict


def test_root_path_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")

    with pytest.raises(ValueError):
        LocalIngestionSource(name="invalid", root_path=file_path)


def test_build_pathspec_with_ignore(tmp_path: Path) -> None:
    ignore_file = tmp_path / ".futurnalignore"
    ignore_file.write_text("*.tmp\n")
    root = tmp_path / "root"
    root.mkdir()

    source = LocalIngestionSource(name="test", root_path=root, ignore_file=ignore_file)
    pathspec = source.build_pathspec()

    assert pathspec.match_file("foo.tmp")
    assert not pathspec.match_file("foo.txt")


def test_load_config_from_dict(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()

    config = load_config_from_dict(
        {
            "sources": [
                {
                    "name": "docs",
                    "root_path": str(root),
                    "include": ["**/*.md"],
                    "max_workers": 4,
                    "max_files_per_batch": 50,
                    "scan_interval_seconds": 120.0,
                    "watcher_debounce_seconds": 1.0,
                    "schedule": "*/5 * * * *",
                    "priority": "high",
                }
            ]
        }
    )

    assert len(config.root) == 1
    assert config.root[0].name == "docs"
    assert config.root[0].max_workers == 4
    assert config.root[0].max_files_per_batch == 50
    assert config.root[0].scan_interval_seconds == 120.0
    assert config.root[0].watcher_debounce_seconds == 1.0
    assert config.root[0].schedule == "*/5 * * * *"
    assert config.root[0].priority == "high"



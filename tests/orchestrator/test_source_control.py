"""Tests for source pause/resume control."""

import pytest
from pathlib import Path
from futurnal.orchestrator.source_control import PausedSourcesRegistry


@pytest.fixture
def temp_registry(tmp_path: Path) -> PausedSourcesRegistry:
    """Create a temporary paused sources registry for testing."""
    registry_file = tmp_path / "paused_sources.json"
    return PausedSourcesRegistry(registry_file)


def test_initial_state_empty(temp_registry: PausedSourcesRegistry):
    """Test that initial registry state is empty."""
    assert not temp_registry.is_paused("any_source")
    assert temp_registry.list_paused() == []


def test_pause_source(temp_registry: PausedSourcesRegistry):
    """Test pausing a source."""
    temp_registry.pause("test_source")
    assert temp_registry.is_paused("test_source")
    assert "test_source" in temp_registry.list_paused()


def test_pause_multiple_sources(temp_registry: PausedSourcesRegistry):
    """Test pausing multiple sources."""
    temp_registry.pause("source1")
    temp_registry.pause("source2")
    temp_registry.pause("source3")

    assert temp_registry.is_paused("source1")
    assert temp_registry.is_paused("source2")
    assert temp_registry.is_paused("source3")

    paused = temp_registry.list_paused()
    assert len(paused) == 3
    assert set(paused) == {"source1", "source2", "source3"}
    # Should be sorted alphabetically
    assert paused == ["source1", "source2", "source3"]


def test_resume_source(temp_registry: PausedSourcesRegistry):
    """Test resuming a paused source."""
    temp_registry.pause("test_source")
    assert temp_registry.is_paused("test_source")

    temp_registry.resume("test_source")
    assert not temp_registry.is_paused("test_source")
    assert temp_registry.list_paused() == []


def test_resume_not_paused_raises_error(temp_registry: PausedSourcesRegistry):
    """Test that resuming a non-paused source raises ValueError."""
    with pytest.raises(ValueError, match="is not paused"):
        temp_registry.resume("never_paused")


def test_pause_idempotent(temp_registry: PausedSourcesRegistry):
    """Test that pausing the same source multiple times is safe."""
    temp_registry.pause("test_source")
    temp_registry.pause("test_source")  # Pause again

    assert temp_registry.is_paused("test_source")
    assert temp_registry.list_paused() == ["test_source"]


def test_persistence_across_instances(tmp_path: Path):
    """Test that pause state persists across registry instances."""
    registry_file = tmp_path / "paused_sources.json"

    # Create first instance and pause sources
    registry1 = PausedSourcesRegistry(registry_file)
    registry1.pause("source1")
    registry1.pause("source2")

    # Create second instance and verify state persisted
    registry2 = PausedSourcesRegistry(registry_file)
    assert registry2.is_paused("source1")
    assert registry2.is_paused("source2")
    assert set(registry2.list_paused()) == {"source1", "source2"}


def test_thread_safe_operations(temp_registry: PausedSourcesRegistry):
    """Test that concurrent operations are thread-safe."""
    import threading

    def pause_sources():
        for i in range(10):
            temp_registry.pause(f"source_{i}")

    def resume_sources():
        for i in range(10):
            try:
                temp_registry.resume(f"source_{i}")
            except ValueError:
                pass  # OK if not paused yet

    # Run concurrent operations
    threads = []
    for _ in range(3):
        t1 = threading.Thread(target=pause_sources)
        t2 = threading.Thread(target=resume_sources)
        threads.extend([t1, t2])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have consistent state (some sources paused, some not)
    paused = temp_registry.list_paused()
    assert isinstance(paused, list)
    assert all(isinstance(s, str) for s in paused)


def test_file_format_valid_json(tmp_path: Path):
    """Test that the persistence file is valid JSON."""
    import json

    registry_file = tmp_path / "paused_sources.json"
    registry = PausedSourcesRegistry(registry_file)

    registry.pause("source1")
    registry.pause("source2")

    # Verify file contains valid JSON
    with open(registry_file) as f:
        data = json.load(f)
        assert isinstance(data, list)
        assert set(data) == {"source1", "source2"}


def test_corrupted_file_recovery(tmp_path: Path):
    """Test that registry handles corrupted JSON file gracefully."""
    registry_file = tmp_path / "paused_sources.json"

    # Write corrupted JSON
    registry_file.write_text("not valid json {{{")

    # Should recover and start fresh
    registry = PausedSourcesRegistry(registry_file)
    assert registry.list_paused() == []

    # Should work normally after recovery
    registry.pause("test_source")
    assert registry.is_paused("test_source")


def test_missing_file_creation(tmp_path: Path):
    """Test that registry creates file if it doesn't exist."""
    registry_file = tmp_path / "paused_sources.json"
    assert not registry_file.exists()

    registry = PausedSourcesRegistry(registry_file)
    assert registry_file.exists()

    # Should be empty initially
    assert registry.list_paused() == []

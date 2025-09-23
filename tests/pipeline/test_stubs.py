"""Tests for pipeline stubs."""

from pathlib import Path

from futurnal.pipeline import NormalizationSink


def test_normalization_sink_writes_file(tmp_path: Path) -> None:
    sink = NormalizationSink(output_dir=tmp_path)
    element = {
        "source": "docs",
        "path": "/data/file.md",
        "sha256": "abc123",
        "element_path": "/tmp/element.json",
    }

    sink.handle(element)

    output_file = tmp_path / "abc123.json"
    assert output_file.exists()
    content = output_file.read_text()
    assert "docs" in content



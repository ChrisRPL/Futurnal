"""Tests for pipeline stubs."""

from pathlib import Path

from futurnal.pipeline import NormalizationSink


class StubPKGWriter:
    def __init__(self) -> None:
        self.documents = {}
        self.deleted = []

    def write_document(self, payload: dict) -> None:
        self.documents[payload["sha256"]] = payload

    def remove_document(self, sha256: str) -> None:
        self.deleted.append(sha256)


class StubVectorWriter:
    def __init__(self) -> None:
        self.embeddings = {}
        self.deleted = []

    def write_embedding(self, payload: dict) -> None:
        self.embeddings[payload["sha256"]] = payload

    def remove_embedding(self, sha256: str) -> None:
        self.deleted.append(sha256)


def test_normalization_sink_writes_payload(tmp_path: Path) -> None:
    pkg_writer = StubPKGWriter()
    vector_writer = StubVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)
    element_file = tmp_path / "element.json"
    element_file.write_text(
        """
{
  "text": "hello",
  "metadata": {
    "source": "docs",
    "path": "/data/file.md",
    "sha256": "abc123"
  }
}
"""
    )
    element = {
        "source": "docs",
        "path": "/data/file.md",
        "sha256": "abc123",
        "element_path": str(element_file),
    }

    sink.handle(element)

    assert pkg_writer.documents["abc123"]["source"] == "docs"
    assert vector_writer.embeddings["abc123"]["path"] == "/data/file.md"


def test_normalization_sink_handles_deletion() -> None:
    pkg_writer = StubPKGWriter()
    vector_writer = StubVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

    sink.handle_deletion({"sha256": "abc123"})

    assert "abc123" in pkg_writer.deleted
    assert "abc123" in vector_writer.deleted



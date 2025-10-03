"""Tests for pipeline stubs."""

from pathlib import Path

from futurnal.pipeline import NormalizationSink


class StubPKGWriter:
    def __init__(self) -> None:
        self.documents = {}
        self.deleted = []
        self.events = []

    def write_document(self, payload: dict) -> None:
        self.documents[payload["sha256"]] = payload

    def remove_document(self, sha256: str) -> None:
        self.deleted.append(sha256)

    def create_experiential_event(self, event_data: dict) -> None:
        self.events.append(event_data)


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


def test_normalization_sink_creates_experiential_events(tmp_path: Path) -> None:
    """Test that document ingestion creates experiential events for Phase 2."""
    pkg_writer = StubPKGWriter()
    vector_writer = StubVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

    element_file = tmp_path / "element.json"
    element_file.write_text(
        """
{
  "text": "hello world",
  "metadata": {
    "source": "obsidian",
    "path": "/vault/note.md",
    "sha256": "def456",
    "modified_at": "2025-01-01T12:00:00",
    "file_size": 1024
  }
}
"""
    )
    element = {
        "source": "obsidian",
        "path": "/vault/note.md",
        "sha256": "def456",
        "element_path": str(element_file),
    }

    sink.handle(element)

    # Verify document ingestion event was created
    assert len(pkg_writer.events) >= 1
    doc_event = pkg_writer.events[0]
    assert doc_event["event_type"] == "document_ingested"
    assert doc_event["event_id"] == "evt-doc-def456"
    assert doc_event["timestamp"] == "2025-01-01T12:00:00"
    assert doc_event["source_uri"] == "/vault/note.md"
    assert doc_event["context"]["source"] == "obsidian"
    assert doc_event["context"]["checksum"] == "def456"


def test_normalization_sink_creates_obsidian_note_events(tmp_path: Path) -> None:
    """Test that Obsidian notes create note_created, link_added, and tag_applied events."""
    pkg_writer = StubPKGWriter()
    vector_writer = StubVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

    element_file = tmp_path / "element.json"
    element_file.write_text(
        """
{
  "text": "note content",
  "metadata": {
    "source": "obsidian",
    "path": "/vault/project.md",
    "sha256": "xyz789",
    "modified_at": "2025-01-15T10:30:00",
    "futurnal": {
      "link_graph": {
        "note_nodes": [
          {
            "vault_id": "vault1",
            "note_id": "project",
            "title": "Project Notes",
            "checksum": "xyz789",
            "path": "/vault/project.md",
            "uri": "futurnal:note/vault1/project"
          }
        ],
        "link_relationships": [
          {
            "source_uri": "futurnal:note/vault1/project",
            "target_uri": "futurnal:note/vault1/reference",
            "relationship_type": "links_to",
            "display_text": "Reference Doc",
            "is_broken": false,
            "source_path": "/vault/project.md"
          }
        ],
        "tag_relationships": [
          {
            "note_uri": "futurnal:note/vault1/project",
            "tag_name": "#important",
            "tag_uri": "futurnal:tag/obsidian/important",
            "is_nested": false,
            "source_path": "/vault/project.md"
          }
        ]
      }
    }
  }
}
"""
    )
    element = {
        "source": "obsidian",
        "path": "/vault/project.md",
        "sha256": "xyz789",
        "element_path": str(element_file),
    }

    sink.handle(element)

    # Verify all event types were created
    assert len(pkg_writer.events) == 4  # document_ingested + note_created + link_added + tag_applied

    event_types = {event["event_type"] for event in pkg_writer.events}
    assert "document_ingested" in event_types
    assert "note_created" in event_types
    assert "link_added" in event_types
    assert "tag_applied" in event_types

    # Verify note_created event details
    note_event = next(e for e in pkg_writer.events if e["event_type"] == "note_created")
    assert note_event["context"]["vault_id"] == "vault1"
    assert note_event["context"]["note_id"] == "project"
    assert note_event["context"]["title"] == "Project Notes"

    # Verify link_added event details
    link_event = next(e for e in pkg_writer.events if e["event_type"] == "link_added")
    assert link_event["context"]["relationship_type"] == "links_to"
    assert link_event["context"]["target_uri"] == "futurnal:note/vault1/reference"

    # Verify tag_applied event details
    tag_event = next(e for e in pkg_writer.events if e["event_type"] == "tag_applied")
    assert tag_event["context"]["tag_name"] == "#important"
    assert tag_event["context"]["tag_uri"] == "futurnal:tag/obsidian/important"


def test_normalization_sink_backward_compatible_without_event_support(tmp_path: Path) -> None:
    """Test that sink gracefully handles PKG writers without event support."""
    # Create a minimal PKG writer without create_experiential_event method
    class LegacyPKGWriter:
        def __init__(self) -> None:
            self.documents = {}

        def write_document(self, payload: dict) -> None:
            self.documents[payload["sha256"]] = payload

        def remove_document(self, sha256: str) -> None:
            pass

    pkg_writer = LegacyPKGWriter()
    vector_writer = StubVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

    element_file = tmp_path / "element.json"
    element_file.write_text(
        """
{
  "text": "hello",
  "metadata": {
    "source": "local",
    "path": "/docs/file.md",
    "sha256": "legacy123"
  }
}
"""
    )
    element = {
        "source": "local",
        "path": "/docs/file.md",
        "sha256": "legacy123",
        "element_path": str(element_file),
    }

    # Should not raise even though PKG writer lacks event support
    sink.handle(element)

    # Verify document was still written
    assert "legacy123" in pkg_writer.documents



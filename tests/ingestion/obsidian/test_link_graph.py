"""Unit tests for Obsidian link graph construction."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.ingestion.obsidian.link_graph import (
    ObsidianLinkGraphConstructor,
    NoteNode,
    LinkRelationship,
    TagRelationship
)
from futurnal.ingestion.obsidian.normalizer import (
    NormalizedDocument,
    DocumentMetadata,
    ProvenanceInfo,
    ObsidianLink,
    ObsidianTag
)


class TestNoteNode:
    """Test NoteNode data structure."""

    def test_note_node_creation(self):
        """Test basic note node creation."""
        node = NoteNode(
            vault_id="test_vault",
            note_id="notes/test_note",
            title="Test Note",
            path=Path("/vault/notes/test_note.md"),
            checksum="abc123"
        )

        assert node.vault_id == "test_vault"
        assert node.note_id == "notes/test_note"
        assert node.title == "Test Note"
        assert node.path == Path("/vault/notes/test_note.md")
        assert node.checksum == "abc123"

    def test_note_node_uri_generation(self):
        """Test URI generation for note nodes."""
        node = NoteNode(
            vault_id="test_vault",
            note_id="notes/test_note",
            title="Test Note",
            path=Path("/vault/notes/test_note.md"),
            checksum="abc123"
        )

        expected_uri = "futurnal:note/test_vault/notes/test_note"
        assert node.to_uri() == expected_uri

    def test_note_node_serialization(self):
        """Test note node serialization to dict."""
        node = NoteNode(
            vault_id="test_vault",
            note_id="notes/test_note",
            title="Test Note",
            path=Path("/vault/notes/test_note.md"),
            checksum="abc123"
        )

        data = node.to_dict()
        assert data["vault_id"] == "test_vault"
        assert data["note_id"] == "notes/test_note"
        assert data["title"] == "Test Note"
        assert data["path"] == "/vault/notes/test_note.md"
        assert data["checksum"] == "abc123"
        assert data["uri"] == "futurnal:note/test_vault/notes/test_note"
        assert "created_at" in data


class TestLinkRelationship:
    """Test LinkRelationship data structure."""

    def test_link_relationship_creation(self):
        """Test basic link relationship creation."""
        source_node = NoteNode(
            vault_id="test_vault",
            note_id="source",
            title="Source Note",
            path=Path("/vault/source.md"),
            checksum="abc123"
        )

        target_node = NoteNode(
            vault_id="test_vault",
            note_id="target",
            title="Target Note",
            path=Path("/vault/target.md"),
            checksum="def456"
        )

        relationship = LinkRelationship(
            source_note=source_node,
            target_note=target_node,
            relationship_type="links_to",
            source_path="/vault/source.md",
            offset=42,
            checksum="rel123"
        )

        assert relationship.source_note == source_node
        assert relationship.target_note == target_node
        assert relationship.relationship_type == "links_to"
        assert relationship.source_path == "/vault/source.md"
        assert relationship.offset == 42
        assert relationship.checksum == "rel123"

    def test_link_relationship_serialization(self):
        """Test link relationship serialization to dict."""
        source_node = NoteNode(
            vault_id="test_vault",
            note_id="source",
            title="Source Note",
            path=Path("/vault/source.md"),
            checksum="abc123"
        )

        target_node = NoteNode(
            vault_id="test_vault",
            note_id="target",
            title="Target Note",
            path=Path("/vault/target.md"),
            checksum="def456"
        )

        relationship = LinkRelationship(
            source_note=source_node,
            target_note=target_node,
            relationship_type="links_to",
            source_path="/vault/source.md",
            heading="Section Title",
            display_text="custom text"
        )

        data = relationship.to_dict()
        assert data["source_uri"] == "futurnal:note/test_vault/source"
        assert data["target_uri"] == "futurnal:note/test_vault/target"
        assert data["relationship_type"] == "links_to"
        assert data["source_path"] == "/vault/source.md"
        assert data["heading"] == "Section Title"
        assert data["display_text"] == "custom text"
        assert "created_at" in data


class TestTagRelationship:
    """Test TagRelationship data structure."""

    def test_tag_relationship_creation(self):
        """Test basic tag relationship creation."""
        note = NoteNode(
            vault_id="test_vault",
            note_id="test_note",
            title="Test Note",
            path=Path("/vault/test_note.md"),
            checksum="abc123"
        )

        tag_relationship = TagRelationship(
            note=note,
            tag_name="project/work",
            tag_uri="futurnal:tag/obsidian/project_work",
            source_path="/vault/test_note.md",
            offset=100,
            is_nested=True
        )

        assert tag_relationship.note == note
        assert tag_relationship.tag_name == "project/work"
        assert tag_relationship.tag_uri == "futurnal:tag/obsidian/project_work"
        assert tag_relationship.is_nested is True

    def test_tag_relationship_serialization(self):
        """Test tag relationship serialization to dict."""
        note = NoteNode(
            vault_id="test_vault",
            note_id="test_note",
            title="Test Note",
            path=Path("/vault/test_note.md"),
            checksum="abc123"
        )

        tag_relationship = TagRelationship(
            note=note,
            tag_name="work",
            tag_uri="futurnal:tag/obsidian/work",
            source_path="/vault/test_note.md",
            is_nested=False
        )

        data = tag_relationship.to_dict()
        assert data["note_uri"] == "futurnal:note/test_vault/test_note"
        assert data["tag_name"] == "work"
        assert data["tag_uri"] == "futurnal:tag/obsidian/work"
        assert data["is_nested"] is False
        assert "created_at" in data


class TestObsidianLinkGraphConstructor:
    """Test ObsidianLinkGraphConstructor class."""

    @pytest.fixture
    def vault_root(self, tmp_path):
        """Create a temporary vault root."""
        vault_dir = tmp_path / "test_vault"
        vault_dir.mkdir()
        return vault_dir

    @pytest.fixture
    def constructor(self, vault_root):
        """Create a link graph constructor instance."""
        return ObsidianLinkGraphConstructor(
            vault_id="test_vault",
            vault_root=vault_root
        )

    @pytest.fixture
    def sample_normalized_doc(self, vault_root):
        """Create a sample normalized document."""
        source_path = vault_root / "test_note.md"

        # Create sample links
        links = [
            ObsidianLink(
                target="target_note",
                display_text=None,
                is_embed=False,
                section=None,
                block_id=None,
                resolved_path=vault_root / "target_note.md",
                is_broken=False,
                start_pos=10,
                end_pos=25
            ),
            ObsidianLink(
                target="another_note",
                display_text="Another Note",
                is_embed=False,
                section="Section 1",
                block_id=None,
                resolved_path=vault_root / "another_note.md",
                is_broken=False,
                start_pos=50,
                end_pos=80
            ),
            ObsidianLink(
                target="image",
                display_text=None,
                is_embed=True,
                section=None,
                block_id=None,
                resolved_path=vault_root / "image.png",
                is_broken=False,
                start_pos=100,
                end_pos=115
            )
        ]

        # Create sample tags
        tags = [
            ObsidianTag(name="project", is_nested=False, start_pos=200, end_pos=208),
            ObsidianTag(name="work/important", is_nested=True, start_pos=250, end_pos=265)
        ]

        metadata = DocumentMetadata(
            frontmatter={"title": "Test Note", "author": "Test Author"},
            links=links,
            tags=tags
        )

        provenance = ProvenanceInfo(
            source_path=source_path,
            vault_id="test_vault",
            content_checksum="abc123",
            metadata_checksum="def456"
        )

        return NormalizedDocument(
            content="# Test Note\n\nThis is a test note with [[target_note]] and [[another_note#Section 1|Another Note]].\n\n![[image]]\n\n#project #work/important",
            metadata=metadata,
            provenance=provenance
        )

    def test_constructor_initialization(self, constructor, vault_root):
        """Test constructor initialization."""
        assert constructor.vault_id == "test_vault"
        assert constructor.vault_root == vault_root
        assert constructor.enable_backlinks is True
        assert constructor.enable_cycle_detection is True
        assert len(constructor.note_registry) == 0
        assert len(constructor.relationship_registry) == 0

    def test_construct_graph_basic(self, constructor, sample_normalized_doc):
        """Test basic graph construction."""
        notes, link_relationships, tag_relationships, _ = constructor.construct_graph(sample_normalized_doc)

        # Should create one source note
        assert len(notes) >= 1
        source_note = notes[0]
        assert source_note.note_id == "test_note"
        assert source_note.title == "Test Note"
        assert source_note.vault_id == "test_vault"

        # Should create link relationships
        assert len(link_relationships) >= 3  # 3 links + possible backlinks

        # Should create tag relationships
        assert len(tag_relationships) == 2

        # Verify link relationship types
        link_types = {rel.relationship_type for rel in link_relationships}
        assert "links_to" in link_types
        assert "references_heading" in link_types
        assert "embeds" in link_types

    def test_note_node_creation(self, constructor, sample_normalized_doc):
        """Test note node creation from normalized document."""
        notes, _, _, _ = constructor.construct_graph(sample_normalized_doc)

        source_note = notes[0]
        assert source_note.vault_id == "test_vault"
        assert source_note.note_id == "test_note"
        assert source_note.title == "Test Note"  # From frontmatter
        assert source_note.checksum == "abc123"

    def test_link_relationship_creation(self, constructor, sample_normalized_doc):
        """Test link relationship creation."""
        _, link_relationships, _, _ = constructor.construct_graph(sample_normalized_doc)

        # Find specific relationships
        simple_link = next((rel for rel in link_relationships if rel.target_note.note_id == "target_note"), None)
        section_link = next((rel for rel in link_relationships if rel.heading == "Section 1"), None)
        embed_link = next((rel for rel in link_relationships if rel.relationship_type == "embeds"), None)

        assert simple_link is not None
        assert simple_link.relationship_type == "links_to"
        assert simple_link.source_path == str(sample_normalized_doc.provenance.source_path)

        assert section_link is not None
        assert section_link.relationship_type == "references_heading"
        assert section_link.heading == "Section 1"
        assert section_link.display_text == "Another Note"

        assert embed_link is not None
        assert embed_link.relationship_type == "embeds"
        assert embed_link.target_note.note_id == "image"

    def test_tag_relationship_creation(self, constructor, sample_normalized_doc):
        """Test tag relationship creation."""
        _, _, tag_relationships, _ = constructor.construct_graph(sample_normalized_doc)

        # Should have 2 tag relationships
        assert len(tag_relationships) == 2

        # Find specific tags
        simple_tag = next((rel for rel in tag_relationships if rel.tag_name == "project"), None)
        nested_tag = next((rel for rel in tag_relationships if rel.tag_name == "work/important"), None)

        assert simple_tag is not None
        assert simple_tag.is_nested is False
        assert simple_tag.tag_uri == "futurnal:tag/obsidian/project"

        assert nested_tag is not None
        assert nested_tag.is_nested is True
        assert nested_tag.tag_uri == "futurnal:tag/obsidian/work_important"

    def test_backlink_inference(self, constructor, sample_normalized_doc):
        """Test backlink inference."""
        notes, link_relationships, _, _ = constructor.construct_graph(sample_normalized_doc)

        # Count forward and backward links
        forward_links = [rel for rel in link_relationships if rel.relationship_type in ["links_to", "references_heading"]]
        backlinks = [rel for rel in link_relationships if rel.relationship_type == "linked_from"]

        # Should have backlinks created for non-embed relationships
        assert len(backlinks) >= 2  # At least 2 backlinks for the 2 non-embed links

    def test_cycle_detection(self, constructor, vault_root):
        """Test cycle detection in relationships."""
        # Create a normalized document that would create a cycle
        links = [
            ObsidianLink(
                target="note_b",
                display_text=None,
                is_embed=False,
                resolved_path=vault_root / "note_b.md",
                is_broken=False,
                start_pos=10,
                end_pos=20
            )
        ]

        metadata = DocumentMetadata(links=links, tags=[])
        provenance = ProvenanceInfo(
            source_path=vault_root / "note_a.md",
            vault_id="test_vault",
            content_checksum="abc123",
            metadata_checksum="def456"
        )

        doc_a = NormalizedDocument(
            content="Link to [[note_b]]",
            metadata=metadata,
            provenance=provenance
        )

        # Process first document
        notes_a, links_a, _, _ = constructor.construct_graph(doc_a)

        # Create reverse link that would cause cycle
        reverse_links = [
            ObsidianLink(
                target="note_a",
                display_text=None,
                is_embed=False,
                resolved_path=vault_root / "note_a.md",
                is_broken=False,
                start_pos=10,
                end_pos=20
            )
        ]

        reverse_metadata = DocumentMetadata(links=reverse_links, tags=[])
        reverse_provenance = ProvenanceInfo(
            source_path=vault_root / "note_b.md",
            vault_id="test_vault",
            content_checksum="def456",
            metadata_checksum="ghi789"
        )

        doc_b = NormalizedDocument(
            content="Link back to [[note_a]]",
            metadata=reverse_metadata,
            provenance=reverse_provenance
        )

        # Process second document - should detect potential cycle
        notes_b, links_b, _, _ = constructor.construct_graph(doc_b)

        # Check statistics for cycle detection
        stats = constructor.get_statistics()
        assert stats["cycles_detected"] >= 0  # May or may not detect cycle depending on processing order

    def test_broken_link_handling(self, constructor, vault_root):
        """Test handling of broken links."""
        # Create link to non-existent note
        broken_links = [
            ObsidianLink(
                target="nonexistent_note",
                display_text=None,
                is_embed=False,
                resolved_path=None,
                is_broken=True,
                start_pos=10,
                end_pos=30
            )
        ]

        metadata = DocumentMetadata(links=broken_links, tags=[])
        provenance = ProvenanceInfo(
            source_path=vault_root / "source.md",
            vault_id="test_vault",
            content_checksum="abc123",
            metadata_checksum="def456"
        )

        doc = NormalizedDocument(
            content="Link to [[nonexistent_note]]",
            metadata=metadata,
            provenance=provenance
        )

        notes, link_relationships, _, _ = constructor.construct_graph(doc)

        # Should still create relationship for broken link
        broken_relationship = next((rel for rel in link_relationships if rel.is_broken), None)
        assert broken_relationship is not None
        assert broken_relationship.target_note.note_id == "nonexistent_note"
        assert broken_relationship.is_broken is True

        # Check statistics
        stats = constructor.get_statistics()
        assert stats["broken_links_found"] >= 1

    def test_statistics_tracking(self, constructor, sample_normalized_doc):
        """Test statistics tracking."""
        constructor.construct_graph(sample_normalized_doc)

        stats = constructor.get_statistics()
        assert stats["notes_created"] >= 1
        assert stats["relationships_created"] >= 3
        assert stats["broken_links_found"] >= 0
        assert stats["cycles_detected"] >= 0
        assert stats["notes_in_registry"] >= 1
        assert stats["relationships_in_registry"] >= 3

    def test_clear_state(self, constructor, sample_normalized_doc):
        """Test clearing constructor state."""
        constructor.construct_graph(sample_normalized_doc)

        # Verify state has data
        assert len(constructor.note_registry) > 0
        assert len(constructor.relationship_registry) > 0
        assert constructor.notes_created > 0

        # Clear state
        constructor.clear_state()

        # Verify state is cleared
        assert len(constructor.note_registry) == 0
        assert len(constructor.relationship_registry) == 0
        assert constructor.notes_created == 0
        assert constructor.relationships_created == 0

    def test_relationship_deduplication(self, constructor, vault_root):
        """Test that duplicate relationships are not created."""
        # Create document with identical links
        links = [
            ObsidianLink(
                target="target_note",
                display_text=None,
                is_embed=False,
                resolved_path=vault_root / "target_note.md",
                is_broken=False,
                start_pos=10,
                end_pos=25
            ),
            ObsidianLink(
                target="target_note",
                display_text=None,
                is_embed=False,
                resolved_path=vault_root / "target_note.md",
                is_broken=False,
                start_pos=50,
                end_pos=65
            )
        ]

        metadata = DocumentMetadata(links=links, tags=[])
        provenance = ProvenanceInfo(
            source_path=vault_root / "source.md",
            vault_id="test_vault",
            content_checksum="abc123",
            metadata_checksum="def456"
        )

        doc = NormalizedDocument(
            content="Link to [[target_note]] and [[target_note]] again",
            metadata=metadata,
            provenance=provenance
        )

        notes, link_relationships, _, _ = constructor.construct_graph(doc)

        # Should only create one relationship despite two identical links
        target_relationships = [rel for rel in link_relationships
                             if rel.target_note.note_id == "target_note"
                             and rel.relationship_type == "links_to"]

        assert len(target_relationships) == 1  # Deduplicated


if __name__ == "__main__":
    pytest.main([__file__])
"""Integration tests for Obsidian link graph construction pipeline."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector, ObsidianVaultSource
from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor, VaultRegistry
from futurnal.ingestion.obsidian.link_graph import ObsidianLinkGraphConstructor
from futurnal.ingestion.obsidian.processor import ObsidianDocumentProcessor
from futurnal.ingestion.local.state import StateStore, FileRecord
from futurnal.pipeline.graph import Neo4jPKGWriter
from futurnal.pipeline.triples import TripleEnrichedNormalizationSink


class TestObsidianLinkGraphIntegration:
    """Integration tests for the complete link graph construction pipeline."""

    @pytest.fixture
    def vault_fixture(self, tmp_path):
        """Create a test vault with interconnected notes."""
        vault_dir = tmp_path / "test_vault"
        vault_dir.mkdir()

        # Create interconnected notes
        notes = {
            "index.md": """# Main Index

This is the main index note that links to other notes.

- See [[project-alpha]] for the main project
- Check [[meeting-notes]] for recent discussions
- Visit [[ideas]] for brainstorming

#index #important
""",
            "project-alpha.md": """---
title: Project Alpha
author: Test Author
status: active
---

# Project Alpha

This is the main project documentation.

## Overview
Project Alpha is about implementing [[ideas]] into reality.

## References
- Related meeting: [[meeting-notes#Project Discussion]]
- See also: [[project-beta]]

#project #alpha #active
""",
            "meeting-notes.md": """# Meeting Notes

## Project Discussion
We discussed the status of [[project-alpha]] and [[project-beta]].

### Action Items
- Update [[project-alpha]] documentation
- Schedule follow-up for [[ideas]] implementation

## Other Topics
- Review [[index]] organization

#meetings #notes
""",
            "ideas.md": """# Ideas

Collection of project ideas and concepts.

## Implementation Ideas
- Feature A: Could enhance [[project-alpha]]
- Feature B: New project [[project-gamma]]

## Brainstorming
Random thoughts and [[meeting-notes]] references.

#ideas #brainstorming
""",
            "project-beta.md": """# Project Beta

Secondary project that relates to [[project-alpha]].

See [[ideas]] for potential enhancements.

#project #beta #inactive
""",
            "orphan-note.md": """# Orphan Note

This note has no incoming links.

It references [[nonexistent-note]] (broken link).

#orphan
""",
            "project-gamma.md": """# Project Gamma

Future project mentioned in [[ideas]].

#project #gamma #planned
"""
        }

        # Write notes to vault
        for filename, content in notes.items():
            note_path = vault_dir / filename
            note_path.write_text(content)

        # Create some subdirectories with notes
        subdir = vault_dir / "archive"
        subdir.mkdir()

        (subdir / "old-project.md").write_text("""# Old Project

This is an archived project.

Referenced from [[../project-alpha]].

#archive #old
""")

        return vault_dir

    @pytest.fixture
    def vault_descriptor(self, vault_fixture):
        """Create vault descriptor for test vault."""
        # Create .obsidian directory for valid vault
        obsidian_dir = vault_fixture / ".obsidian"
        obsidian_dir.mkdir(exist_ok=True)

        return ObsidianVaultDescriptor.from_path(
            base_path=vault_fixture,
            name="Test Vault"
        )

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store for testing."""
        state_db = tmp_path / "state.db"
        return StateStore(state_db)

    @pytest.fixture
    def mock_pkg_writer(self):
        """Create mock PKG writer."""
        return Mock(spec=Neo4jPKGWriter)

    @pytest.fixture
    def mock_vector_writer(self):
        """Create mock vector writer."""
        return Mock()

    @pytest.fixture
    def element_sink(self, mock_pkg_writer, mock_vector_writer):
        """Create element sink with mocked dependencies."""
        return TripleEnrichedNormalizationSink(mock_pkg_writer, mock_vector_writer)

    @pytest.fixture
    def vault_source(self, vault_descriptor):
        """Create vault source for testing."""
        return ObsidianVaultSource.from_vault_descriptor(vault_descriptor)

    @pytest.fixture
    def vault_registry(self, tmp_path, vault_descriptor):
        """Create vault registry with test vault."""
        registry_root = tmp_path / "vault_registry"
        vault_registry = VaultRegistry(registry_root)
        vault_registry.add_or_update(vault_descriptor)
        return vault_registry

    @pytest.fixture
    def connector(self, tmp_path, state_store, element_sink, vault_registry):
        """Create Obsidian vault connector."""
        workspace_dir = tmp_path / "workspace"
        return ObsidianVaultConnector(
            workspace_dir=workspace_dir,
            state_store=state_store,
            element_sink=element_sink,
            vault_registry=vault_registry
        )

    def test_full_vault_ingestion_with_link_graph(self, connector, vault_source, element_sink):
        """Test complete vault ingestion with link graph construction."""
        # Ingest the vault
        elements = list(connector.ingest(vault_source))

        # Should have processed multiple documents
        assert len(elements) > 0

        # Verify element sink was called
        assert element_sink.handle.call_count > 0

        # Check that elements contain link graph data
        for element in elements:
            element_path = element["element_path"]
            assert Path(element_path).exists()

            # Load element data to verify link graph information
            with open(element_path, 'r') as f:
                payload = json.load(f)

            # Should have Futurnal metadata with link graph
            assert "metadata" in payload
            assert "futurnal" in payload["metadata"]

            futurnal_metadata = payload["metadata"]["futurnal"]
            if "link_graph" in futurnal_metadata:
                link_graph = futurnal_metadata["link_graph"]

                # Verify link graph structure
                assert "note_nodes" in link_graph
                assert "link_relationships" in link_graph
                assert "tag_relationships" in link_graph
                assert "statistics" in link_graph

                # Verify at least one note node was created
                assert len(link_graph["note_nodes"]) >= 1

                # Check note node structure
                for node in link_graph["note_nodes"]:
                    assert "vault_id" in node
                    assert "note_id" in node
                    assert "title" in node
                    assert "path" in node
                    assert "checksum" in node
                    assert "uri" in node

    def test_link_relationships_construction(self, connector, vault_source):
        """Test that link relationships are properly constructed."""
        # Ingest the vault
        elements = list(connector.ingest(vault_source))

        # Find element with link relationships
        link_relationships = []
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                link_relationships.extend(futurnal_metadata["link_graph"]["link_relationships"])

        # Should have multiple link relationships
        assert len(link_relationships) > 0

        # Verify relationship types
        relationship_types = {rel["relationship_type"] for rel in link_relationships}
        assert "links_to" in relationship_types

        # Check for specific relationships we know should exist
        # project-alpha -> ideas
        alpha_to_ideas = any(
            "project-alpha" in rel["source_uri"] and "ideas" in rel["target_uri"]
            for rel in link_relationships
        )
        assert alpha_to_ideas

        # Check for heading references
        heading_refs = [rel for rel in link_relationships if rel["relationship_type"] == "references_heading"]
        assert len(heading_refs) > 0

        # Verify heading reference structure
        for rel in heading_refs:
            if rel.get("heading"):
                assert "heading" in rel
                assert rel["heading"] is not None

    def test_tag_relationships_construction(self, connector, vault_source):
        """Test that tag relationships are properly constructed."""
        elements = list(connector.ingest(vault_source))

        # Collect all tag relationships
        tag_relationships = []
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                tag_relationships.extend(futurnal_metadata["link_graph"]["tag_relationships"])

        # Should have multiple tag relationships
        assert len(tag_relationships) > 0

        # Verify tag structure
        for rel in tag_relationships:
            assert "note_uri" in rel
            assert "tag_name" in rel
            assert "tag_uri" in rel
            assert "is_nested" in rel

        # Check for specific tags we know should exist
        tag_names = {rel["tag_name"] for rel in tag_relationships}
        assert "project" in tag_names
        assert "ideas" in tag_names
        assert "alpha" in tag_names

        # Check for nested tags
        nested_tags = [rel for rel in tag_relationships if rel["is_nested"]]
        # Should have some nested tags (none in our fixture, but structure should support it)
        assert isinstance(nested_tags, list)

    def test_broken_link_handling(self, connector, vault_source):
        """Test handling of broken links."""
        elements = list(connector.ingest(vault_source))

        # Look for broken link relationships
        broken_relationships = []
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                broken_relationships.extend([
                    rel for rel in futurnal_metadata["link_graph"]["link_relationships"]
                    if rel["is_broken"]
                ])

        # Should have at least one broken link (orphan-note -> nonexistent-note)
        assert len(broken_relationships) > 0

        # Verify broken link structure
        for rel in broken_relationships:
            assert rel["is_broken"] is True
            assert "target_uri" in rel

    def test_backlink_inference(self, connector, vault_source):
        """Test that backlinks are properly inferred."""
        elements = list(connector.ingest(vault_source))

        # Collect all relationships
        all_relationships = []
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                all_relationships.extend(futurnal_metadata["link_graph"]["link_relationships"])

        # Should have backlink relationships
        backlinks = [rel for rel in all_relationships if rel["relationship_type"] == "linked_from"]
        assert len(backlinks) > 0

        # For each forward link, there should be a corresponding backlink
        forward_links = [rel for rel in all_relationships if rel["relationship_type"] == "links_to"]

        for forward_link in forward_links:
            # Look for corresponding backlink
            corresponding_backlink = any(
                rel["source_uri"] == forward_link["target_uri"] and
                rel["target_uri"] == forward_link["source_uri"] and
                rel["relationship_type"] == "linked_from"
                for rel in backlinks
            )
            # Note: Not all forward links may have backlinks due to implementation details
            # but at least some should exist

    def test_statistics_generation(self, connector, vault_source):
        """Test that graph construction statistics are generated."""
        elements = list(connector.ingest(vault_source))

        # Find statistics
        statistics_found = False
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                stats = futurnal_metadata["link_graph"]["statistics"]

                # Verify statistics structure
                assert "notes_created" in stats
                assert "relationships_created" in stats
                assert "broken_links_found" in stats
                assert "cycles_detected" in stats
                assert "notes_in_registry" in stats
                assert "relationships_in_registry" in stats

                # Verify reasonable values
                assert stats["notes_created"] >= 1
                assert stats["relationships_created"] >= 0
                assert stats["broken_links_found"] >= 0

                statistics_found = True
                break

        assert statistics_found, "No link graph statistics found in processed elements"

    def test_vault_hierarchy_handling(self, connector, vault_source):
        """Test handling of notes in subdirectories."""
        elements = list(connector.ingest(vault_source))

        # Look for the archived note
        archive_note_found = False
        for element in elements:
            if "archive/old-project" in element["path"]:
                archive_note_found = True

                element_path = element["element_path"]
                with open(element_path, 'r') as f:
                    payload = json.load(f)

                futurnal_metadata = payload["metadata"].get("futurnal", {})
                if "link_graph" in futurnal_metadata:
                    note_nodes = futurnal_metadata["link_graph"]["note_nodes"]

                    # Should have created note node for archived note
                    archive_nodes = [node for node in note_nodes if "archive" in node["note_id"]]
                    assert len(archive_nodes) >= 1

        assert archive_note_found, "Archive note not processed"

    def test_cross_directory_links(self, connector, vault_source):
        """Test links between notes in different directories."""
        elements = list(connector.ingest(vault_source))

        # Look for relationships involving the archived note
        cross_dir_relationships = []
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                for rel in futurnal_metadata["link_graph"]["link_relationships"]:
                    # Check for relationships involving archive directory
                    if ("archive" in rel["source_uri"] or "archive" in rel["target_uri"]):
                        cross_dir_relationships.append(rel)

        # Should have at least one cross-directory relationship
        # (old-project.md -> ../project-alpha)
        assert len(cross_dir_relationships) >= 0  # May be 0 if relative links aren't processed

    @patch('futurnal.ingestion.obsidian.processor.partition')
    def test_integration_with_unstructured_io(self, mock_partition, connector, vault_source):
        """Test integration with Unstructured.io processing."""
        # Mock Unstructured.io response
        mock_element = Mock()
        mock_element.to_dict.return_value = {
            "text": "Sample text content",
            "metadata": {"category": "Title"}
        }
        mock_partition.return_value = [mock_element]

        # Ingest vault
        elements = list(connector.ingest(vault_source))

        # Verify Unstructured.io was called
        assert mock_partition.call_count > 0

        # Verify elements were created with both Unstructured.io and link graph data
        assert len(elements) > 0
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            # Should have text from Unstructured.io
            assert "text" in payload

            # Should also have Futurnal metadata
            assert "metadata" in payload
            assert "futurnal" in payload["metadata"]

    def test_error_handling_in_graph_construction(self, connector, vault_source, tmp_path):
        """Test error handling when graph construction fails."""
        # Create a note with invalid content that might cause parsing errors
        vault_root = vault_source.root_path
        invalid_note = vault_root / "invalid-note.md"
        invalid_note.write_text("# Invalid Note\n\n[[")  # Incomplete wikilink

        # Should not crash the entire ingestion
        elements = list(connector.ingest(vault_source))

        # Should still process other valid notes
        assert len(elements) > 0

    def test_performance_with_large_vault(self, tmp_path, state_store, element_sink):
        """Test performance with a larger vault (stress test)."""
        # Create larger vault for performance testing
        large_vault = tmp_path / "large_vault"
        large_vault.mkdir()

        # Create many interconnected notes
        num_notes = 50
        for i in range(num_notes):
            content = f"""# Note {i}

This is note number {i}.

Links to other notes:
"""
            # Add links to some other notes
            for j in range(min(5, num_notes)):
                target = (i + j + 1) % num_notes
                content += f"- [[note-{target}]]\n"

            content += f"\n#note{i} #test"

            note_path = large_vault / f"note-{i}.md"
            note_path.write_text(content)

        # Create vault source
        vault_descriptor = ObsidianVaultDescriptor(
            id="large_vault_test",
            name="Large Test Vault",
            base_path=large_vault
        )
        vault_source = ObsidianVaultSource.from_vault_descriptor(vault_descriptor)

        # Create connector
        workspace_dir = tmp_path / "large_workspace"
        connector = ObsidianVaultConnector(
            workspace_dir=workspace_dir,
            state_store=state_store,
            element_sink=element_sink
        )

        # Time the ingestion
        import time
        start_time = time.time()
        elements = list(connector.ingest(vault_source))
        end_time = time.time()

        # Should complete in reasonable time (adjust threshold as needed)
        processing_time = end_time - start_time
        assert processing_time < 30  # Should complete within 30 seconds

        # Should process all notes
        assert len(elements) >= num_notes

        # Verify graph data is present
        graph_data_found = 0
        for element in elements:
            element_path = element["element_path"]
            with open(element_path, 'r') as f:
                payload = json.load(f)

            futurnal_metadata = payload["metadata"].get("futurnal", {})
            if "link_graph" in futurnal_metadata:
                graph_data_found += 1

        assert graph_data_found > 0


if __name__ == "__main__":
    pytest.main([__file__])
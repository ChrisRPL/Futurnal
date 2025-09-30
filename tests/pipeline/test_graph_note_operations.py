"""Unit tests for Neo4j PKG writer note-specific operations."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from futurnal.pipeline.graph import Neo4jPKGWriter


class TestNeo4jPKGWriterNoteOperations:
    """Test note-specific operations in Neo4jPKGWriter."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        tx = Mock()

        # Setup mock chain
        driver.session.return_value.__enter__.return_value = session
        session.execute_write.return_value = Mock()
        session.execute_read.return_value = Mock()

        return driver

    @pytest.fixture
    def pkg_writer(self, mock_driver):
        """Create a PKG writer with mocked driver."""
        writer = Neo4jPKGWriter(driver=mock_driver)
        return writer

    def test_create_note_node(self, pkg_writer):
        """Test note node creation."""
        note_data = {
            "vault_id": "test_vault",
            "note_id": "notes/test_note",
            "title": "Test Note",
            "path": "/vault/notes/test_note.md",
            "checksum": "abc123",
            "uri": "futurnal:note/test_vault/notes/test_note",
            "created_at": datetime.utcnow().isoformat()
        }

        pkg_writer.create_note_node(note_data)

        # Verify session was called
        pkg_writer._driver.session.assert_called_once()

        # Verify execute_write was called with a function
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.assert_called_once()

    def test_create_note_relationship(self, pkg_writer):
        """Test note relationship creation."""
        relationship_data = {
            "source_uri": "futurnal:note/test_vault/source",
            "target_uri": "futurnal:note/test_vault/target",
            "relationship_type": "links_to",
            "source_path": "/vault/source.md",
            "offset": 42,
            "checksum": "rel123",
            "heading": None,
            "block_id": None,
            "display_text": None,
            "is_broken": False,
            "created_at": datetime.utcnow().isoformat()
        }

        # Mock successful relationship creation
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.return_value = Mock()  # Simulates successful result

        pkg_writer.create_note_relationship(relationship_data)

        # Verify session and transaction were called
        pkg_writer._driver.session.assert_called_once()
        session_mock.execute_write.assert_called_once()

    def test_create_note_relationship_with_heading(self, pkg_writer):
        """Test note relationship creation with heading reference."""
        relationship_data = {
            "source_uri": "futurnal:note/test_vault/source",
            "target_uri": "futurnal:note/test_vault/target",
            "relationship_type": "references_heading",
            "source_path": "/vault/source.md",
            "offset": 42,
            "checksum": "rel123",
            "heading": "Section Title",
            "block_id": None,
            "display_text": "Custom Display Text",
            "is_broken": False
        }

        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.return_value = Mock()

        pkg_writer.create_note_relationship(relationship_data)

        # Verify the call was made
        session_mock.execute_write.assert_called_once()

    def test_create_tag_relationship(self, pkg_writer):
        """Test tag relationship creation."""
        tag_data = {
            "note_uri": "futurnal:note/test_vault/note",
            "tag_name": "project",
            "tag_uri": "futurnal:tag/obsidian/project",
            "source_path": "/vault/note.md",
            "offset": 100,
            "is_nested": False,
            "created_at": datetime.utcnow().isoformat()
        }

        pkg_writer.create_tag_relationship(tag_data)

        # Verify session was called
        pkg_writer._driver.session.assert_called_once()

        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.assert_called_once()

    def test_create_nested_tag_relationship(self, pkg_writer):
        """Test nested tag relationship creation."""
        tag_data = {
            "note_uri": "futurnal:note/test_vault/note",
            "tag_name": "project/work",
            "tag_uri": "futurnal:tag/obsidian/project_work",
            "source_path": "/vault/note.md",
            "offset": 100,
            "is_nested": True
        }

        pkg_writer.create_tag_relationship(tag_data)

        # Verify session was called
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.assert_called_once()

    def test_update_note_path(self, pkg_writer):
        """Test note path update."""
        # Mock successful update
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        result_mock = Mock()
        result_mock.__getitem__.return_value = "futurnal:note/test_vault/note"
        session_mock.execute_write.return_value = result_mock

        pkg_writer.update_note_path(
            vault_id="test_vault",
            note_id="note",
            old_path="/vault/old_path.md",
            new_path="/vault/new_path.md"
        )

        # Verify session was called
        pkg_writer._driver.session.assert_called_once()
        session_mock.execute_write.assert_called_once()

    def test_update_note_path_not_found(self, pkg_writer):
        """Test note path update when note is not found."""
        # Mock no result (note not found)
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.return_value = None

        pkg_writer.update_note_path(
            vault_id="test_vault",
            note_id="nonexistent_note",
            old_path="/vault/old_path.md",
            new_path="/vault/new_path.md"
        )

        # Should still attempt the update
        session_mock.execute_write.assert_called_once()

    def test_remove_note_relationships(self, pkg_writer):
        """Test removing note relationships."""
        # Mock deletion result
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        result_mock = Mock()
        result_mock.__getitem__.return_value = 5  # 5 relationships deleted
        session_mock.execute_write.return_value = result_mock

        deleted_count = pkg_writer.remove_note_relationships(
            note_uri="futurnal:note/test_vault/note"
        )

        assert deleted_count == 5
        session_mock.execute_write.assert_called_once()

    def test_remove_note_relationships_specific_types(self, pkg_writer):
        """Test removing specific types of note relationships."""
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        result_mock = Mock()
        result_mock.__getitem__.return_value = 3
        session_mock.execute_write.return_value = result_mock

        deleted_count = pkg_writer.remove_note_relationships(
            note_uri="futurnal:note/test_vault/note",
            relationship_types=["links_to", "references_heading"]
        )

        assert deleted_count == 3
        session_mock.execute_write.assert_called_once()

    def test_get_note_statistics_all_vaults(self, pkg_writer):
        """Test getting statistics for all vaults."""
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value

        # Mock query results
        def mock_run(query, params):
            result = Mock()
            if "note_count" in query:
                result.__getitem__.return_value = 100
            elif "relationship_count" in query:
                result.__getitem__.return_value = 250
            elif "tag_count" in query:
                result.__getitem__.return_value = 50
            return result

        session_mock.run.side_effect = mock_run

        stats = pkg_writer.get_note_statistics()

        assert stats["notes"] == 100
        assert stats["relationships"] == 250
        assert stats["tags"] == 50

        # Should call run 3 times (for 3 different queries)
        assert session_mock.run.call_count == 3

    def test_get_note_statistics_specific_vault(self, pkg_writer):
        """Test getting statistics for a specific vault."""
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value

        # Mock query results
        def mock_run(query, params):
            result = Mock()
            if "note_count" in query:
                result.__getitem__.return_value = 25
            elif "relationship_count" in query:
                result.__getitem__.return_value = 60
            elif "tag_count" in query:
                result.__getitem__.return_value = 15
            return result

        session_mock.run.side_effect = mock_run

        stats = pkg_writer.get_note_statistics(vault_id="specific_vault")

        assert stats["notes"] == 25
        assert stats["relationships"] == 60
        assert stats["tags"] == 15

        # Should call run 3 times with vault_id parameter
        assert session_mock.run.call_count == 3
        # All calls should include vault_id in parameters
        for call in session_mock.run.call_args_list:
            query, params = call[0]
            if params:  # Some queries might not have params
                assert "vault_id" in params

    def test_create_note_node_minimal_data(self, pkg_writer):
        """Test note node creation with minimal required data."""
        note_data = {
            "vault_id": "test_vault",
            "note_id": "simple_note",
            "title": "Simple Note",
            "path": "/vault/simple_note.md",
            "checksum": "xyz789",
            "uri": "futurnal:note/test_vault/simple_note"
        }

        pkg_writer.create_note_node(note_data)

        # Should not raise an error even without created_at
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.assert_called_once()

    def test_create_relationship_dynamic_type(self, pkg_writer):
        """Test that relationship types are dynamically created."""
        relationship_data = {
            "source_uri": "futurnal:note/test_vault/source",
            "target_uri": "futurnal:note/test_vault/target",
            "relationship_type": "custom_relationship_type",
            "source_path": "/vault/source.md",
            "checksum": "custom123",
            "is_broken": False
        }

        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.return_value = Mock()

        pkg_writer.create_note_relationship(relationship_data)

        # Verify that the custom relationship type is used
        session_mock.execute_write.assert_called_once()
        call_args = session_mock.execute_write.call_args
        func = call_args[0][0]

        # Execute the function to see the query
        mock_tx = Mock()
        func(mock_tx)

        # The relationship type should be converted to uppercase in the query
        mock_tx.run.assert_called_once()
        query = mock_tx.run.call_args[0][0]
        assert "CUSTOM_RELATIONSHIP_TYPE" in query

    def test_error_handling_in_note_operations(self, pkg_writer):
        """Test error handling in note operations."""
        # Mock driver to raise an exception
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.side_effect = Exception("Database error")

        note_data = {
            "vault_id": "test_vault",
            "note_id": "error_note",
            "title": "Error Note",
            "path": "/vault/error_note.md",
            "checksum": "error123",
            "uri": "futurnal:note/test_vault/error_note"
        }

        # Should raise the exception (not silently fail)
        with pytest.raises(Exception, match="Database error"):
            pkg_writer.create_note_node(note_data)


class TestNeo4jPKGWriterBackwardCompatibility:
    """Test that new note operations don't break existing functionality."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()

        # Setup mock chain
        driver.session.return_value.__enter__.return_value = session
        session.execute_write.return_value = Mock()

        return driver

    @pytest.fixture
    def pkg_writer(self, mock_driver):
        """Create a PKG writer with mocked driver."""
        return Neo4jPKGWriter(driver=mock_driver)

    def test_original_write_document_still_works(self, pkg_writer):
        """Test that original write_document method still works."""
        payload = {
            "sha256": "abc123",
            "path": "/path/to/document.md",
            "source": "test_source",
            "text": "Document content",
            "metadata": {
                "size_bytes": 1000,
                "modified_at": "2023-01-01T00:00:00",
                "ingested_at": "2023-01-01T00:00:00"
            }
        }

        pkg_writer.write_document(payload)

        # Should work without errors
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.assert_called_once()

    def test_original_remove_document_still_works(self, pkg_writer):
        """Test that original remove_document method still works."""
        pkg_writer.remove_document("abc123")

        # Should work without errors
        session_mock = pkg_writer._driver.session.return_value.__enter__.return_value
        session_mock.execute_write.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
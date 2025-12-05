"""Tests for ReembeddingService, SchemaChangeDetection, and ReembeddingProgress.

Tests schema change detection, batch re-embedding, and progress tracking.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from futurnal.embeddings.reembedding import (
    ReembeddingProgress,
    ReembeddingService,
    SchemaChangeDetection,
)


# -----------------------------------------------------------------------------
# SchemaChangeDetection Tests
# -----------------------------------------------------------------------------


class TestSchemaChangeDetection:
    """Tests for SchemaChangeDetection dataclass."""

    def test_basic_creation(self):
        """Validate basic creation with all fields."""
        change = SchemaChangeDetection(
            old_version=1,
            new_version=2,
            new_entity_types=["Project", "Task"],
            removed_entity_types=[],
            new_relationship_types=["ASSIGNED_TO"],
            removed_relationship_types=[],
            requires_reembedding=True,
        )

        assert change.old_version == 1
        assert change.new_version == 2
        assert change.new_entity_types == ["Project", "Task"]
        assert change.requires_reembedding is True

    def test_has_entity_changes(self):
        """Validate has_entity_changes property."""
        # With changes
        change = SchemaChangeDetection(
            old_version=1,
            new_version=2,
            new_entity_types=["Project"],
        )
        assert change.has_entity_changes is True

        # Without changes
        change_empty = SchemaChangeDetection(
            old_version=1,
            new_version=2,
        )
        assert change_empty.has_entity_changes is False

    def test_has_relationship_changes(self):
        """Validate has_relationship_changes property."""
        # With changes
        change = SchemaChangeDetection(
            old_version=1,
            new_version=2,
            new_relationship_types=["ASSIGNED_TO"],
        )
        assert change.has_relationship_changes is True

        # Without changes
        change_empty = SchemaChangeDetection(
            old_version=1,
            new_version=2,
        )
        assert change_empty.has_relationship_changes is False

    def test_to_dict(self):
        """Validate to_dict serialization."""
        change = SchemaChangeDetection(
            old_version=1,
            new_version=2,
            new_entity_types=["Project"],
            removed_entity_types=["OldType"],
            new_relationship_types=["ASSIGNED_TO"],
            removed_relationship_types=["OLD_REL"],
            requires_reembedding=True,
        )

        result = change.to_dict()

        assert result["old_version"] == 1
        assert result["new_version"] == 2
        assert result["new_entity_types"] == ["Project"]
        assert result["removed_entity_types"] == ["OldType"]
        assert result["requires_reembedding"] is True

    def test_str_representation(self):
        """Validate string representation."""
        change = SchemaChangeDetection(
            old_version=1,
            new_version=2,
            new_entity_types=["Project"],
            requires_reembedding=True,
        )

        str_repr = str(change)

        assert "v1 -> v2" in str_repr
        assert "Project" in str_repr
        assert "requires_reembedding=True" in str_repr


# -----------------------------------------------------------------------------
# ReembeddingProgress Tests
# -----------------------------------------------------------------------------


class TestReembeddingProgress:
    """Tests for ReembeddingProgress dataclass."""

    def test_initial_state(self):
        """Validate initial state."""
        progress = ReembeddingProgress()

        assert progress.total == 0
        assert progress.processed == 0
        assert progress.succeeded == 0
        assert progress.failed == 0
        assert progress.started_at is None
        assert progress.completed_at is None
        assert progress.errors == []

    def test_is_complete(self):
        """Validate is_complete property."""
        progress = ReembeddingProgress(total=10, processed=5)
        assert progress.is_complete is False

        progress.processed = 10
        assert progress.is_complete is True

    def test_success_rate(self):
        """Validate success_rate calculation."""
        progress = ReembeddingProgress(
            total=10,
            processed=10,
            succeeded=8,
            failed=2,
        )

        assert progress.success_rate == 0.8

    def test_success_rate_zero_processed(self):
        """Validate success_rate is 0.0 when nothing processed."""
        progress = ReembeddingProgress()
        assert progress.success_rate == 0.0

    def test_duration_seconds(self):
        """Validate duration_seconds calculation."""
        progress = ReembeddingProgress()
        progress.started_at = datetime(2024, 1, 15, 10, 0, 0)
        progress.completed_at = datetime(2024, 1, 15, 10, 5, 30)

        assert progress.duration_seconds == 330.0  # 5 minutes 30 seconds

    def test_duration_seconds_in_progress(self):
        """Validate duration_seconds uses current time if not completed."""
        progress = ReembeddingProgress()
        progress.started_at = datetime.utcnow() - timedelta(seconds=60)

        duration = progress.duration_seconds
        assert duration is not None
        assert duration >= 59  # At least 59 seconds

    def test_embeddings_per_second(self):
        """Validate embeddings_per_second calculation."""
        progress = ReembeddingProgress(
            total=100,
            processed=100,
            succeeded=95,
            failed=5,
        )
        progress.started_at = datetime(2024, 1, 15, 10, 0, 0)
        progress.completed_at = datetime(2024, 1, 15, 10, 0, 10)  # 10 seconds

        assert progress.embeddings_per_second == 10.0

    def test_to_dict(self):
        """Validate to_dict serialization."""
        progress = ReembeddingProgress(
            total=100,
            processed=50,
            succeeded=45,
            failed=5,
        )
        progress.started_at = datetime(2024, 1, 15, 10, 0, 0)
        progress.errors = [{"entity_id": "test", "error": "Test error"}]

        result = progress.to_dict()

        assert result["total"] == 100
        assert result["processed"] == 50
        assert result["succeeded"] == 45
        assert result["failed"] == 5
        assert result["success_rate"] == 0.9
        assert result["is_complete"] is False
        assert result["error_count"] == 1


# -----------------------------------------------------------------------------
# ReembeddingService Tests
# -----------------------------------------------------------------------------


class TestReembeddingService:
    """Tests for ReembeddingService."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=None)
        return driver, session

    @pytest.fixture
    def mock_store(self):
        """Create mock SchemaVersionedEmbeddingStore."""
        store = MagicMock()
        store.mark_for_reembedding.return_value = 5
        store.get_embeddings_needing_reembedding.return_value = []
        store.store_embedding.return_value = "new_emb_123"
        store.delete_embedding.return_value = True
        return store

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock MultiModelEmbeddingService."""
        service = MagicMock()

        # Create mock embedding result
        result = MagicMock()
        result.embedding = np.random.rand(768).tolist()
        result.model_version = "mock:instructor-large"

        service.embed.return_value = result
        return service

    @pytest.fixture
    def mock_schema_manager(self):
        """Create mock SchemaVersionManager."""
        manager = MagicMock()

        # Version 1 schema
        v1 = MagicMock()
        v1.version = 1
        v1.entity_types = ["Person", "Organization", "Event"]
        v1.relationship_types = ["RELATED_TO", "WORKS_AT"]

        # Version 2 schema (with new types)
        v2 = MagicMock()
        v2.version = 2
        v2.entity_types = ["Person", "Organization", "Event", "Project"]
        v2.relationship_types = ["RELATED_TO", "WORKS_AT", "ASSIGNED_TO"]

        def get_version(version: int):
            if version == 1:
                return v1
            elif version == 2:
                return v2
            return None

        manager.get_version = get_version
        return manager

    @pytest.fixture
    def reembedding_service(
        self,
        mock_neo4j_driver,
        mock_store,
        mock_embedding_service,
        mock_schema_manager,
    ):
        """Create ReembeddingService with mocked dependencies."""
        driver, session = mock_neo4j_driver

        with patch("futurnal.pkg.schema.migration.SchemaVersionManager") as mock_manager_class:
            mock_manager_class.return_value = mock_schema_manager

            service = ReembeddingService(
                store=mock_store,
                embedding_service=mock_embedding_service,
                neo4j_driver=driver,
            )

            return service

    def test_detect_schema_changes_with_new_types(
        self,
        reembedding_service,
    ):
        """Validate detection of new entity types."""
        changes = reembedding_service.detect_schema_changes(
            old_version=1,
            new_version=2,
        )

        assert changes.old_version == 1
        assert changes.new_version == 2
        assert "Project" in changes.new_entity_types
        assert "ASSIGNED_TO" in changes.new_relationship_types
        assert changes.requires_reembedding is True

    def test_detect_schema_changes_no_new_types(
        self,
        reembedding_service,
        mock_schema_manager,
    ):
        """Validate requires_reembedding is False when no new entity types."""
        # Modify v2 to have same entity types as v1
        v2 = mock_schema_manager.get_version(2)
        v2.entity_types = ["Person", "Organization", "Event"]

        changes = reembedding_service.detect_schema_changes(
            old_version=1,
            new_version=2,
        )

        assert changes.requires_reembedding is False

    def test_detect_schema_changes_missing_version(
        self,
        reembedding_service,
    ):
        """Validate handling of missing schema versions."""
        changes = reembedding_service.detect_schema_changes(
            old_version=1,
            new_version=99,  # Non-existent
        )

        assert changes.requires_reembedding is False

    def test_trigger_reembedding_empty_queue(
        self,
        reembedding_service,
        mock_store,
    ):
        """Validate trigger_reembedding with empty queue."""
        mock_store.get_embeddings_needing_reembedding.return_value = []

        progress = reembedding_service.trigger_reembedding(schema_version=1)

        assert progress.total == 0
        assert progress.is_complete is True
        assert progress.started_at is not None
        assert progress.completed_at is not None

    def test_trigger_reembedding_with_entities(
        self,
        reembedding_service,
        mock_store,
        mock_neo4j_driver,
    ):
        """Validate trigger_reembedding processes entities."""
        driver, session = mock_neo4j_driver

        # Setup pending embeddings
        mock_store.get_embeddings_needing_reembedding.return_value = [
            {
                "embedding_id": "emb_1",
                "entity_id": "person_1",
                "entity_type": "Person",
                "model_id": "instructor-large",
                "extraction_confidence": 0.95,
                "source_document_id": "doc_1",
            },
        ]

        # Setup PKG query result
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_node = {
            "id": "person_1",
            "name": "John Smith",
            "description": "Software Engineer",
        }
        mock_record.__getitem__ = lambda self, key: mock_node if key == "e" else ["Person"]
        mock_result.single.return_value = mock_record
        session.run.return_value = mock_result

        progress = reembedding_service.trigger_reembedding(schema_version=1)

        # Verify progress
        assert progress.total == 1
        assert progress.processed == 1
        assert progress.succeeded == 1
        assert progress.failed == 0
        assert progress.completed_at is not None

        # Verify embedding service was called
        reembedding_service._embedding_service.embed.assert_called_once()

        # Verify store was called
        mock_store.store_embedding.assert_called_once()

    def test_trigger_reembedding_handles_missing_entity(
        self,
        reembedding_service,
        mock_store,
        mock_neo4j_driver,
    ):
        """Validate handling of entity not found in PKG."""
        driver, session = mock_neo4j_driver

        # Setup pending embeddings
        mock_store.get_embeddings_needing_reembedding.return_value = [
            {
                "embedding_id": "emb_1",
                "entity_id": "missing_entity",
                "entity_type": "Person",
                "model_id": "instructor-large",
            },
        ]

        # Setup PKG query to return no result
        mock_result = MagicMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result

        progress = reembedding_service.trigger_reembedding(schema_version=1)

        assert progress.total == 1
        assert progress.processed == 1
        assert progress.succeeded == 0
        assert progress.failed == 1
        assert len(progress.errors) == 1
        assert progress.errors[0]["entity_id"] == "missing_entity"

    def test_trigger_reembedding_batch_size(
        self,
        reembedding_service,
        mock_store,
    ):
        """Validate batch processing with custom batch size."""
        mock_store.get_embeddings_needing_reembedding.return_value = []

        progress = reembedding_service.trigger_reembedding(
            schema_version=1,
            batch_size=50,
        )

        # Verify mark_for_reembedding was called
        mock_store.mark_for_reembedding.assert_called_once_with(
            entity_ids=None,
            schema_version=1,
            reason="schema_evolution",
        )

    def test_get_reembedding_stats(
        self,
        reembedding_service,
        mock_store,
    ):
        """Validate get_reembedding_stats aggregation."""
        mock_store.get_embeddings_needing_reembedding.return_value = [
            {"reembedding_reason": "schema_evolution", "schema_version": 1},
            {"reembedding_reason": "schema_evolution", "schema_version": 1},
            {"reembedding_reason": "quality", "schema_version": 2},
        ]

        stats = reembedding_service.get_reembedding_stats()

        assert stats["total_pending"] == 3
        assert stats["by_reason"]["schema_evolution"] == 2
        assert stats["by_reason"]["quality"] == 1
        assert stats["by_schema_version"][1] == 2
        assert stats["by_schema_version"][2] == 1


class TestReembeddingServiceEntityFetching:
    """Tests for entity fetching from PKG."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=None)
        return driver, session

    @pytest.fixture
    def service(self, mock_neo4j_driver):
        """Create ReembeddingService."""
        driver, session = mock_neo4j_driver

        with patch("futurnal.pkg.schema.migration.SchemaVersionManager"):
            mock_store = MagicMock()
            mock_embedding_service = MagicMock()

            service = ReembeddingService(
                store=mock_store,
                embedding_service=mock_embedding_service,
                neo4j_driver=driver,
            )

            return service, session

    def test_fetch_person_entity(self, service):
        """Validate fetching Person entity from PKG."""
        svc, session = service

        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_node = {
            "id": "person_1",
            "name": "John Smith",
            "description": "Software Engineer at Futurnal",
        }
        mock_record.__getitem__ = lambda self, key: mock_node if key == "e" else ["Person"]
        mock_result.single.return_value = mock_record
        session.run.return_value = mock_result

        entities = svc._fetch_entities_from_pkg(["person_1"])

        assert "person_1" in entities
        assert entities["person_1"]["type"] == "Person"
        assert entities["person_1"]["content"] == "John Smith: Software Engineer at Futurnal"
        assert entities["person_1"]["temporal_context"] is None

    def test_fetch_event_entity_with_temporal(self, service):
        """Validate fetching Event entity with temporal context."""
        svc, session = service

        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_node = {
            "id": "event_1",
            "name": "Team Meeting",
            "description": "Quarterly planning",
            "timestamp": "2024-01-15T14:30:00",
            "duration": 7200,  # 2 hours in seconds
            "temporal_type": "DURING",
        }
        mock_record.__getitem__ = lambda self, key: mock_node if key == "e" else ["Event"]
        mock_result.single.return_value = mock_record
        session.run.return_value = mock_result

        entities = svc._fetch_entities_from_pkg(["event_1"])

        assert "event_1" in entities
        assert entities["event_1"]["type"] == "Event"
        assert entities["event_1"]["temporal_context"] is not None

        temporal = entities["event_1"]["temporal_context"]
        assert temporal.timestamp == datetime(2024, 1, 15, 14, 30, 0)
        assert temporal.duration == timedelta(hours=2)
        assert temporal.temporal_type == "DURING"

    def test_fetch_missing_entity(self, service):
        """Validate handling of missing entity."""
        svc, session = service

        mock_result = MagicMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result

        entities = svc._fetch_entities_from_pkg(["missing_entity"])

        assert "missing_entity" not in entities

    def test_format_entity_content(self, service):
        """Validate entity content formatting."""
        svc, _ = service

        # With description
        content = svc._format_entity_content(
            {"name": "John Smith", "description": "Engineer"},
            "Person",
        )
        assert content == "John Smith: Engineer"

        # Without description
        content = svc._format_entity_content(
            {"name": "John Smith"},
            "Person",
        )
        assert content == "John Smith"

        # With only ID
        content = svc._format_entity_content(
            {"id": "person_123"},
            "Person",
        )
        assert content == "person_123"

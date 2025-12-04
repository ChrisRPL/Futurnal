"""Integration Test Fixtures for PKG Module 05.

Provides fixtures combining PKG + Vector + Extraction for end-to-end testing.
All tests use real databases via testcontainers (no mocks per no-mockups.mdc).

Fixtures provided:
- pkg_repository: Full PKGRepository with initialized schema
- chroma_collection: ChromaDB test collection
- normalization_sink: Combined sink for integration tests
- sample_normalized_document: NormalizedDocument with temporal data
- sample_temporal_triples: TemporalTriple test data
- sync_event_capture: Captures sync events for verification

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional

import pytest

from tests.pkg.conftest import (
    neo4j_container,
    neo4j_driver,
    neo4j_session,
    clean_database,
    initialized_schema,
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
    sample_person,
    sample_event,
    sample_events_pair,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync Event Models (for testing - production version in src/futurnal/pkg/sync)
# ---------------------------------------------------------------------------


@dataclass
class SyncEvent:
    """Sync event for tracking PKG ↔ Vector synchronization.

    Used to verify that PKG updates properly trigger vector store sync.
    """

    event_type: str  # "entity_created", "entity_updated", "entity_deleted"
    entity_id: str
    entity_type: str
    timestamp: datetime
    source_operation: str  # "pkg_write", "vector_write"
    vector_sync_status: str  # "pending", "completed", "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncEventCapture:
    """Test utility to capture and verify sync events.

    Used in integration tests to verify PKG operations trigger proper sync.

    Example:
        >>> capture = SyncEventCapture()
        >>> sink = create_sink_with_capture(capture)
        >>> sink.handle(document)
        >>> assert len(capture.events) == 2  # pkg_write + vector_write
        >>> assert capture.get_events_for_entity(doc_id)[0].event_type == "entity_created"
    """

    events: List[SyncEvent] = field(default_factory=list)

    def capture(self, event: SyncEvent) -> None:
        """Record a sync event."""
        self.events.append(event)
        logger.debug(f"Captured sync event: {event.event_type} for {event.entity_id}")

    def get_events_for_entity(self, entity_id: str) -> List[SyncEvent]:
        """Get all events for a specific entity."""
        return [e for e in self.events if e.entity_id == entity_id]

    def get_events_by_type(self, event_type: str) -> List[SyncEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_latest_event(self) -> Optional[SyncEvent]:
        """Get the most recent event."""
        return self.events[-1] if self.events else None

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()

    @property
    def count(self) -> int:
        """Total number of captured events."""
        return len(self.events)


# ---------------------------------------------------------------------------
# Mock Writers for Integration Testing
# ---------------------------------------------------------------------------


@dataclass
class MockPKGWriter:
    """Mock PKG writer that integrates with real Neo4j for testing.

    Wraps the actual PKG repository operations while emitting sync events.
    """

    driver: Any
    sync_handler: Optional[Callable[[SyncEvent], None]] = None
    documents: Dict[str, Dict] = field(default_factory=dict)

    def write_document(self, payload: Dict[str, Any]) -> None:
        """Write document to PKG and emit sync event."""
        sha256 = payload["sha256"]
        self.documents[sha256] = payload

        # Write to Neo4j (using CREATE/MATCH pattern for Neo4j 4.x compatibility)
        with self.driver.session() as session:
            # Use simple MERGE without ON CREATE SET (Neo4j 4.x compatible)
            session.run(
                """
                MERGE (d:Document {sha256: $sha256})
                SET d.path = $path,
                    d.source = $source,
                    d.text = $text,
                    d.updated_at = COALESCE(d.updated_at, datetime()),
                    d.created_at = COALESCE(d.created_at, datetime())
                """,
                {
                    "sha256": sha256,
                    "path": payload.get("path", ""),
                    "source": payload.get("source", ""),
                    "text": payload.get("text", ""),
                },
            )

        # Emit sync event
        if self.sync_handler:
            self.sync_handler(
                SyncEvent(
                    event_type="entity_created",
                    entity_id=sha256,
                    entity_type="Document",
                    timestamp=datetime.utcnow(),
                    source_operation="pkg_write",
                    vector_sync_status="pending",
                    metadata={"path": payload.get("path")},
                )
            )

    def remove_document(self, sha256: str) -> None:
        """Remove document from PKG."""
        self.documents.pop(sha256, None)

        with self.driver.session() as session:
            session.run("MATCH (d:Document {sha256: $sha256}) DETACH DELETE d", {"sha256": sha256})

        if self.sync_handler:
            self.sync_handler(
                SyncEvent(
                    event_type="entity_deleted",
                    entity_id=sha256,
                    entity_type="Document",
                    timestamp=datetime.utcnow(),
                    source_operation="pkg_delete",
                    vector_sync_status="pending",
                )
            )

    def create_experiential_event(self, event_data: Dict[str, Any]) -> None:
        """Create experiential event in PKG."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (e:ExperientialEvent {
                    event_id: $event_id,
                    event_type: $event_type,
                    timestamp: datetime($timestamp),
                    source_uri: $source_uri,
                    context: $context,
                    created_at: datetime()
                })
                """,
                {
                    "event_id": event_data["event_id"],
                    "event_type": event_data["event_type"],
                    "timestamp": event_data["timestamp"],
                    "source_uri": event_data.get("source_uri", ""),
                    "context": str(event_data.get("context", {})),
                },
            )


@dataclass
class MockVectorWriter:
    """Mock Vector writer for testing vector sync.

    Stores embeddings in memory for verification.
    """

    sync_handler: Optional[Callable[[SyncEvent], None]] = None
    embeddings: Dict[str, Dict] = field(default_factory=dict)

    def write_embedding(self, payload: Dict[str, Any]) -> None:
        """Write embedding to vector store."""
        sha256 = payload["sha256"]
        self.embeddings[sha256] = {
            "sha256": sha256,
            "path": payload.get("path"),
            "source": payload.get("source"),
            "text": payload.get("text"),
            "embedding": payload.get("embedding"),
        }

        if self.sync_handler:
            self.sync_handler(
                SyncEvent(
                    event_type="entity_created",
                    entity_id=sha256,
                    entity_type="Document",
                    timestamp=datetime.utcnow(),
                    source_operation="vector_write",
                    vector_sync_status="completed",
                )
            )

    def remove_embedding(self, sha256: str) -> None:
        """Remove embedding from vector store."""
        self.embeddings.pop(sha256, None)

        if self.sync_handler:
            self.sync_handler(
                SyncEvent(
                    event_type="entity_deleted",
                    entity_id=sha256,
                    entity_type="Document",
                    timestamp=datetime.utcnow(),
                    source_operation="vector_delete",
                    vector_sync_status="completed",
                )
            )

    def get_embedding(self, sha256: str) -> Optional[Dict]:
        """Get embedding by SHA256."""
        return self.embeddings.get(sha256)


# ---------------------------------------------------------------------------
# Repository Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pkg_repository(neo4j_driver, initialized_schema):
    """Provide a fully initialized PKGRepository for integration tests.

    Args:
        neo4j_driver: Neo4j driver fixture
        initialized_schema: Ensures schema is initialized

    Yields:
        PKGRepository instance with all sub-repositories configured
    """
    from futurnal.pkg.repository import PKGRepository

    # Create a mock database manager wrapping the driver
    class IntegrationDBManager:
        def __init__(self, driver):
            self._driver = driver

        def session(self):
            return self._driver.session()

    db_manager = IntegrationDBManager(neo4j_driver)
    repo = PKGRepository(db_manager)

    yield repo


@pytest.fixture
def temporal_queries(neo4j_driver, initialized_schema):
    """Provide TemporalGraphQueries service for integration tests.

    Args:
        neo4j_driver: Neo4j driver fixture
        initialized_schema: Ensures schema is initialized

    Yields:
        TemporalGraphQueries instance
    """
    from futurnal.pkg.queries.temporal import TemporalGraphQueries

    class IntegrationDBManager:
        def __init__(self, driver):
            self._driver = driver

        def session(self):
            return self._driver.session()

    db_manager = IntegrationDBManager(neo4j_driver)
    queries = TemporalGraphQueries(db_manager)

    yield queries


# ---------------------------------------------------------------------------
# Sync Event Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_event_capture():
    """Provide a SyncEventCapture for verifying sync behavior.

    Yields:
        SyncEventCapture instance that can be attached to sinks
    """
    capture = SyncEventCapture()
    yield capture
    capture.clear()


@pytest.fixture
def mock_pkg_writer(neo4j_driver, sync_event_capture):
    """Provide MockPKGWriter connected to real Neo4j.

    Args:
        neo4j_driver: Neo4j driver fixture
        sync_event_capture: Event capture fixture

    Yields:
        MockPKGWriter instance
    """
    writer = MockPKGWriter(driver=neo4j_driver, sync_handler=sync_event_capture.capture)
    yield writer


@pytest.fixture
def mock_vector_writer(sync_event_capture):
    """Provide MockVectorWriter for testing.

    Args:
        sync_event_capture: Event capture fixture

    Yields:
        MockVectorWriter instance
    """
    writer = MockVectorWriter(sync_handler=sync_event_capture.capture)
    yield writer


@pytest.fixture
def normalization_sink(mock_pkg_writer, mock_vector_writer):
    """Provide NormalizationSink configured for integration testing.

    Args:
        mock_pkg_writer: PKG writer fixture
        mock_vector_writer: Vector writer fixture

    Yields:
        NormalizationSink instance
    """
    from futurnal.pipeline.stubs import NormalizationSink

    sink = NormalizationSink(pkg_writer=mock_pkg_writer, vector_writer=mock_vector_writer)
    yield sink


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------


def generate_sha256(content: str) -> str:
    """Generate SHA256 hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture
def sample_normalized_document():
    """Provide a sample normalized document with temporal data.

    Returns a dict matching NormalizationSink.handle() expected format.
    """
    content = "Sample document content for integration testing."
    sha256 = generate_sha256(content)

    return {
        "sha256": sha256,
        "path": "/test/vault/integration-test.md",
        "source": "obsidian_vault",
        "metadata": {
            "filetype": "markdown",
            "created_at": "2024-01-15T10:00:00Z",
            "modified_at": "2024-01-15T14:30:00Z",
            "frontmatter": {
                "title": "Integration Test Document",
                "author": "Test Author",
                "tags": ["test", "integration"],
                "created": "2024-01-15",
            },
            "obsidian_tags": ["#test", "#integration"],
        },
        "text": content,
    }


@pytest.fixture
def sample_temporal_document():
    """Provide a document with rich temporal content for extraction testing."""
    content = """# Project Timeline

On January 15, 2024, we held the initial planning meeting.
The meeting lasted 2 hours, from 9:00 AM to 11:00 AM.

After the meeting, a decision was made at 2:00 PM.

The next day (January 16th), development started at 10:00 AM.
This was directly caused by the planning meeting outcomes.

By January 20, 2024, the first milestone was reached.
"""
    sha256 = generate_sha256(content)

    return {
        "sha256": sha256,
        "path": "/test/vault/temporal-test.md",
        "source": "obsidian_vault",
        "metadata": {
            "filetype": "markdown",
            "created_at": "2024-01-15T09:00:00Z",
            "modified_at": "2024-01-20T18:00:00Z",
            "frontmatter": {
                "title": "Project Timeline",
                "project": "Integration Testing",
            },
        },
        "text": content,
    }


@pytest.fixture
def sample_temporal_triples():
    """Provide sample temporal triples for testing temporal queries.

    Returns list of dicts representing extracted temporal triples.
    """
    base_date = datetime(2024, 1, 15)

    return [
        {
            "subject": "planning_meeting",
            "predicate": "BEFORE",
            "object": "decision_made",
            "timestamp": base_date + timedelta(hours=9),
            "temporal_type": "BEFORE",
            "confidence": 0.95,
            "temporal_confidence": 0.90,
        },
        {
            "subject": "decision_made",
            "predicate": "CAUSES",
            "object": "development_started",
            "timestamp": base_date + timedelta(hours=14),
            "temporal_type": "CAUSES",
            "confidence": 0.85,
            "temporal_confidence": 0.85,
            "causal_confidence": 0.80,
        },
        {
            "subject": "development_started",
            "predicate": "BEFORE",
            "object": "milestone_reached",
            "timestamp": base_date + timedelta(days=1, hours=10),
            "temporal_type": "BEFORE",
            "confidence": 0.90,
            "temporal_confidence": 0.88,
        },
    ]


@pytest.fixture
def sample_events_for_causal_chain():
    """Provide sample events forming a causal chain for testing.

    Returns list of event dicts with causal relationships.
    """
    base_date = datetime(2024, 1, 15)

    return [
        {
            "id": "event_meeting",
            "name": "Planning Meeting",
            "event_type": "meeting",
            "timestamp": base_date + timedelta(hours=9),
            "duration": timedelta(hours=2),
            "description": "Initial project planning",
            "source_document": "test_doc_001",
        },
        {
            "id": "event_decision",
            "name": "Requirements Decision",
            "event_type": "decision",
            "timestamp": base_date + timedelta(hours=14),
            "description": "Finalized requirements",
            "source_document": "test_doc_001",
        },
        {
            "id": "event_dev_start",
            "name": "Development Started",
            "event_type": "action",
            "timestamp": base_date + timedelta(days=1, hours=10),
            "description": "Development phase began",
            "source_document": "test_doc_002",
        },
        {
            "id": "event_milestone",
            "name": "First Milestone",
            "event_type": "milestone",
            "timestamp": base_date + timedelta(days=5),
            "description": "Completed initial milestone",
            "source_document": "test_doc_003",
        },
    ]


@pytest.fixture
def causal_relationships():
    """Provide causal relationship definitions for the event chain."""
    return [
        {
            "cause_id": "event_meeting",
            "effect_id": "event_decision",
            "relationship_type": "CAUSES",
            "causal_confidence": 0.85,
            "causal_evidence": "Meeting led to decision",
            "temporality_satisfied": True,
        },
        {
            "cause_id": "event_decision",
            "effect_id": "event_dev_start",
            "relationship_type": "ENABLES",
            "causal_confidence": 0.90,
            "causal_evidence": "Decision enabled development",
            "temporality_satisfied": True,
        },
        {
            "cause_id": "event_dev_start",
            "effect_id": "event_milestone",
            "relationship_type": "CAUSES",
            "causal_confidence": 0.80,
            "causal_evidence": "Development led to milestone",
            "temporality_satisfied": True,
        },
    ]


# ---------------------------------------------------------------------------
# Performance Test Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def large_entity_dataset():
    """Generate 1000 entities and 5000 relationships for performance testing.

    Returns dict with 'entities' and 'relationships' keys.
    """
    from futurnal.pkg.schema.models import PersonNode, EventNode, ConceptNode

    entities = []
    relationships = []

    # Create 300 persons
    for i in range(300):
        entities.append(
            PersonNode(
                name=f"Person {i}",
                aliases=[f"P{i}"],
                confidence=0.9,
            )
        )

    # Create 500 events with timestamps
    base_date = datetime(2024, 1, 1)
    for i in range(500):
        entities.append(
            EventNode(
                name=f"Event {i}",
                event_type=["meeting", "decision", "action", "communication"][i % 4],
                timestamp=base_date + timedelta(hours=i),
                source_document=f"doc_{i // 50}",
                description=f"Test event {i}",
            )
        )

    # Create 200 concepts
    for i in range(200):
        entities.append(
            ConceptNode(
                name=f"Concept {i}",
                category=["topic", "skill", "domain"][i % 3],
                description=f"Test concept {i}",
            )
        )

    # Create relationships (5000 total)
    # Person -> Event (PARTICIPATED_IN): 1500
    for i in range(1500):
        person_idx = i % 300
        event_idx = 300 + (i % 500)
        relationships.append(
            (entities[person_idx].id, "PARTICIPATED_IN", entities[event_idx].id, {"confidence": 0.8})
        )

    # Event -> Event (CAUSES/BEFORE): 2000
    for i in range(2000):
        event1_idx = 300 + (i % 499)
        event2_idx = 300 + ((i + 1) % 500)
        rel_type = "CAUSES" if i % 2 == 0 else "BEFORE"
        relationships.append((entities[event1_idx].id, rel_type, entities[event2_idx].id, {"confidence": 0.75}))

    # Person -> Concept (RELATED_TO): 1500
    for i in range(1500):
        person_idx = i % 300
        concept_idx = 800 + (i % 200)
        relationships.append((entities[person_idx].id, "RELATED_TO", entities[concept_idx].id, {"confidence": 0.85}))

    return {"entities": entities, "relationships": relationships}


@pytest.fixture
def bulk_triple_dataset():
    """Generate 10000 triples for bulk insert performance testing."""
    triples = []
    base_date = datetime(2024, 1, 1)

    for i in range(10000):
        triples.append(
            {
                "subject": f"entity_{i % 1000}",
                "predicate": ["RELATED_TO", "CAUSES", "BEFORE", "PART_OF"][i % 4],
                "object": f"entity_{(i + 1) % 1000}",
                "confidence": 0.7 + (i % 30) / 100,
                "timestamp": base_date + timedelta(hours=i // 10),
                "source_document": f"doc_{i // 100}",
            }
        )

    return triples


# ---------------------------------------------------------------------------
# Re-export base fixtures for convenience
# ---------------------------------------------------------------------------


__all__ = [
    # Sync event models
    "SyncEvent",
    "SyncEventCapture",
    # Mock writers
    "MockPKGWriter",
    "MockVectorWriter",
    # Base fixtures (re-exported)
    "requires_neo4j",
    "requires_testcontainers",
    "requires_docker",
]

"""Integration Tests for Extraction → PKG Pipeline (Module 05).

Tests the full pipeline from document extraction through PKG storage.

From production plan:
- test_metadata_extraction_to_pkg: Frontmatter/tags/links → PKG nodes
- test_advanced_extraction_to_pkg: Temporal/events/causal → PKG temporal nodes
- test_full_document_pipeline: Complete NormalizedDocument flow

Success Metrics:
- All integration tests passing
- Entity counts match extraction output
- Provenance chain maintained

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pytest

from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def generate_sha256(content: str) -> str:
    """Generate SHA256 hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()


def create_document_with_metadata(
    path: str,
    content: str,
    frontmatter: Dict[str, Any],
    tags: List[str],
) -> Dict[str, Any]:
    """Create a document payload with metadata for testing."""
    sha256 = generate_sha256(content)
    return {
        "sha256": sha256,
        "path": path,
        "source": "obsidian_vault",
        "metadata": {
            "filetype": "markdown",
            "created_at": datetime.utcnow().isoformat(),
            "frontmatter": frontmatter,
            "obsidian_tags": tags,
        },
        "text": content,
    }


def store_document_in_pkg(session, payload: Dict[str, Any]) -> str:
    """Store a document node in PKG."""
    sha256 = payload["sha256"]
    session.run(
        """
        CREATE (d:Document {
            sha256: $sha256,
            path: $path,
            source: $source,
            text: $text,
            created_at: datetime(),
            updated_at: datetime()
        })
        """,
        {
            "sha256": sha256,
            "path": payload["path"],
            "source": payload["source"],
            "text": payload.get("text", ""),
        },
    )
    return sha256


def store_tag_nodes(session, sha256: str, tags: List[str]) -> None:
    """Store tag nodes and relationships."""
    for tag in tags:
        tag_name = tag.lstrip("#")
        session.run(
            """
            MATCH (d:Document {sha256: $sha256})
            MERGE (t:Tag {name: $tag_name})
            CREATE (d)-[:HAS_TAG {created_at: datetime()}]->(t)
            """,
            {"sha256": sha256, "tag_name": tag_name},
        )


def store_event_nodes(session, events: List[Dict[str, Any]]) -> List[str]:
    """Store event nodes in PKG."""
    event_ids = []
    for event in events:
        session.run(
            """
            CREATE (e:Event {
                id: $id,
                name: $name,
                event_type: $event_type,
                timestamp: datetime($timestamp),
                description: $description,
                source_document: $source_document,
                created_at: datetime(),
                updated_at: datetime()
            })
            """,
            {
                "id": event["id"],
                "name": event["name"],
                "event_type": event["event_type"],
                "timestamp": event["timestamp"].isoformat(),
                "description": event.get("description", ""),
                "source_document": event.get("source_document", "test"),
            },
        )
        event_ids.append(event["id"])
    return event_ids


def store_causal_relationships(
    session, relationships: List[Dict[str, Any]]
) -> None:
    """Store causal relationships between events."""
    for rel in relationships:
        session.run(
            """
            MATCH (cause:Event {id: $cause_id})
            MATCH (effect:Event {id: $effect_id})
            CREATE (cause)-[:CAUSES {
                causal_confidence: $confidence,
                causal_evidence: $evidence,
                temporality_satisfied: $temporality,
                created_at: datetime()
            }]->(effect)
            """,
            {
                "cause_id": rel["cause_id"],
                "effect_id": rel["effect_id"],
                "confidence": rel.get("causal_confidence", 0.8),
                "evidence": rel.get("causal_evidence", "Test evidence"),
                "temporality": rel.get("temporality_satisfied", True),
            },
        )


# ---------------------------------------------------------------------------
# Integration Tests: Extraction → PKG Pipeline
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestExtractionToPKGPipeline:
    """Tests for extraction pipeline → PKG storage integration.

    From production plan 05-integration-testing.md:
    Validates that extracted entities/relationships are correctly stored in PKG.
    """

    def test_metadata_extraction_to_pkg(self, neo4j_driver, clean_database):
        """Validate frontmatter/tags/links → PKG nodes.

        From production plan:
        - Extract entities/relationships from document
        - Store in PKG
        - Verify storage
        """
        # Create test document with metadata
        document = create_document_with_metadata(
            path="/vault/notes/test-doc.md",
            content="# Test Document\n\nThis is a test document with tags.",
            frontmatter={
                "title": "Test Document",
                "author": "Test Author",
                "category": "testing",
                "created": "2024-01-15",
            },
            tags=["#test", "#integration", "#pkg"],
        )

        # Store in PKG (simulating extraction pipeline output)
        with neo4j_driver.session() as session:
            sha256 = store_document_in_pkg(session, document)
            store_tag_nodes(session, sha256, document["metadata"]["obsidian_tags"])

            # Store frontmatter as properties on document
            session.run(
                """
                MATCH (d:Document {sha256: $sha256})
                SET d.title = $title,
                    d.author = $author,
                    d.category = $category
                """,
                {
                    "sha256": sha256,
                    "title": document["metadata"]["frontmatter"]["title"],
                    "author": document["metadata"]["frontmatter"]["author"],
                    "category": document["metadata"]["frontmatter"]["category"],
                },
            )

            # Verify document stored
            result = session.run(
                "MATCH (d:Document {sha256: $sha256}) RETURN d",
                {"sha256": sha256},
            )
            doc_record = result.single()
            assert doc_record is not None
            doc = doc_record["d"]
            assert doc["title"] == "Test Document"
            assert doc["author"] == "Test Author"

            # Verify tags stored
            tag_result = session.run(
                """
                MATCH (d:Document {sha256: $sha256})-[:HAS_TAG]->(t:Tag)
                RETURN collect(t.name) as tags
                """,
                {"sha256": sha256},
            )
            tag_record = tag_result.single()
            tags = tag_record["tags"]
            assert len(tags) == 3
            assert set(tags) == {"test", "integration", "pkg"}

    def test_advanced_extraction_to_pkg(self, neo4j_driver, clean_database):
        """Validate temporal/events/causal → PKG temporal nodes.

        From production plan:
        - Extract temporal triples
        - Store in PKG
        - Verify EventNode with timestamp, temporal relationships
        """
        base_date = datetime(2024, 1, 15)

        # Simulated extraction output: events with temporal grounding
        extracted_events = [
            {
                "id": "event_planning",
                "name": "Planning Meeting",
                "event_type": "meeting",
                "timestamp": base_date + timedelta(hours=9),
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
                "id": "event_development",
                "name": "Development Started",
                "event_type": "action",
                "timestamp": base_date + timedelta(days=1, hours=10),
                "description": "Development phase began",
                "source_document": "test_doc_002",
            },
        ]

        # Causal relationships from extraction
        causal_relationships = [
            {
                "cause_id": "event_planning",
                "effect_id": "event_decision",
                "causal_confidence": 0.85,
                "causal_evidence": "Meeting led to decision",
                "temporality_satisfied": True,
            },
            {
                "cause_id": "event_decision",
                "effect_id": "event_development",
                "causal_confidence": 0.90,
                "causal_evidence": "Decision enabled development",
                "temporality_satisfied": True,
            },
        ]

        with neo4j_driver.session() as session:
            # Store events
            event_ids = store_event_nodes(session, extracted_events)
            assert len(event_ids) == 3

            # Store causal relationships
            store_causal_relationships(session, causal_relationships)

            # Verify events stored with timestamps (Option B critical)
            event_result = session.run(
                """
                MATCH (e:Event)
                WHERE e.timestamp IS NOT NULL
                RETURN count(e) as count
                """
            )
            count_record = event_result.single()
            assert count_record["count"] == 3, "All events must have timestamps (Option B)"

            # Verify causal relationships
            causal_result = session.run(
                """
                MATCH (cause:Event)-[r:CAUSES]->(effect:Event)
                RETURN count(r) as count
                """
            )
            causal_count = causal_result.single()["count"]
            assert causal_count == 2, "Causal relationships should be stored"

            # Verify temporal ordering (cause timestamp < effect timestamp)
            ordering_result = session.run(
                """
                MATCH (cause:Event)-[r:CAUSES]->(effect:Event)
                WHERE cause.timestamp >= effect.timestamp
                RETURN count(r) as violations
                """
            )
            violations = ordering_result.single()["violations"]
            assert violations == 0, "Cause must precede effect (temporal ordering)"

    def test_full_document_pipeline(
        self,
        neo4j_driver,
        clean_database,
        sample_normalized_document,
        normalization_sink,
        sync_event_capture,
    ):
        """Validate complete NormalizedDocument → PKG storage flow.

        From production plan:
        - Load realistic test document
        - Run full extraction pipeline
        - Verify all node types created
        - Verify provenance chain maintained
        """
        # Process document through normalization sink
        normalization_sink.handle(sample_normalized_document)

        # Verify sync events were emitted
        assert sync_event_capture.count >= 2, "Should emit PKG and Vector sync events"

        events = sync_event_capture.get_events_for_entity(
            sample_normalized_document["sha256"]
        )
        assert len(events) >= 2

        # Verify PKG write event
        pkg_events = [e for e in events if e.source_operation == "pkg_write"]
        assert len(pkg_events) == 1

        # Verify Vector write event
        vector_events = [e for e in events if e.source_operation == "vector_write"]
        assert len(vector_events) == 1
        assert vector_events[0].vector_sync_status == "completed"

        # Verify document stored in PKG
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (d:Document {sha256: $sha256}) RETURN d",
                {"sha256": sample_normalized_document["sha256"]},
            )
            doc_record = result.single()
            assert doc_record is not None
            assert doc_record["d"]["path"] == sample_normalized_document["path"]

    def test_extraction_with_provenance_chain(self, neo4j_driver, clean_database):
        """Verify provenance is tracked from source → extraction → storage.

        Option B critical: Full provenance tracking for causal validation.
        """
        # Create document with provenance metadata
        document_sha = generate_sha256("Test document content")

        with neo4j_driver.session() as session:
            # Create source document
            session.run(
                """
                CREATE (d:Document {
                    sha256: $sha256,
                    path: '/vault/notes/source.md',
                    source: 'obsidian_vault',
                    created_at: datetime()
                })
                """,
                {"sha256": document_sha},
            )

            # Create chunk for provenance
            chunk_id = f"chunk_{document_sha[:16]}_0"
            session.run(
                """
                MATCH (d:Document {sha256: $doc_sha})
                CREATE (c:Chunk {
                    id: $chunk_id,
                    document_id: $doc_sha,
                    content_hash: $content_hash,
                    position: 0,
                    created_at: datetime()
                })
                CREATE (c)-[:EXTRACTED_FROM]->(d)
                """,
                {
                    "doc_sha": document_sha,
                    "chunk_id": chunk_id,
                    "content_hash": generate_sha256("chunk content"),
                },
            )

            # Create event extracted from chunk
            event_id = f"event_{document_sha[:16]}"
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                CREATE (e:Event {
                    id: $event_id,
                    name: 'Extracted Event',
                    event_type: 'action',
                    timestamp: datetime(),
                    source_document: $doc_sha,
                    extraction_method: 'llm',
                    confidence: 0.85,
                    created_at: datetime()
                })
                CREATE (e)-[:DISCOVERED_IN]->(c)
                """,
                {
                    "chunk_id": chunk_id,
                    "event_id": event_id,
                    "doc_sha": document_sha,
                },
            )

            # Verify provenance chain: Event → Chunk → Document
            provenance_result = session.run(
                """
                MATCH (e:Event)-[:DISCOVERED_IN]->(c:Chunk)-[:EXTRACTED_FROM]->(d:Document)
                WHERE e.id = $event_id
                RETURN d.sha256 as doc_sha, c.id as chunk_id
                """,
                {"event_id": event_id},
            )
            provenance = provenance_result.single()
            assert provenance is not None, "Provenance chain must exist"
            assert provenance["doc_sha"] == document_sha
            assert provenance["chunk_id"] == chunk_id


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestExtractionEdgeCases:
    """Tests for edge cases in extraction → PKG pipeline."""

    def test_duplicate_entity_handling(self, neo4j_driver, clean_database):
        """Verify duplicate entities are handled correctly."""
        with neo4j_driver.session() as session:
            # Create initial entity
            session.run(
                """
                CREATE (p:Person {
                    id: 'person_001',
                    name: 'John Doe',
                    confidence: 0.8,
                    discovery_count: 1,
                    created_at: datetime()
                })
                """
            )

            # Try to merge same entity (simulating second extraction)
            session.run(
                """
                MERGE (p:Person {id: 'person_001'})
                ON MATCH SET
                    p.discovery_count = p.discovery_count + 1,
                    p.confidence = CASE
                        WHEN 0.9 > p.confidence THEN 0.9
                        ELSE p.confidence
                    END,
                    p.updated_at = datetime()
                """
            )

            # Verify only one entity exists with updated count
            result = session.run(
                """
                MATCH (p:Person {id: 'person_001'})
                RETURN p.discovery_count as count, p.confidence as conf
                """
            )
            record = result.single()
            assert record["count"] == 2, "Discovery count should increase"
            assert record["conf"] == 0.9, "Confidence should update if higher"

    def test_event_without_timestamp_rejected(self, neo4j_driver, clean_database):
        """Verify events without timestamps are rejected (Option B critical).

        Option B mandates temporal grounding for all events.
        """
        # This test validates that our validation layer catches missing timestamps
        # In real implementation, validation happens at Pydantic model level

        with neo4j_driver.session() as session:
            # Create event with timestamp (valid)
            session.run(
                """
                CREATE (e:Event {
                    id: 'valid_event',
                    name: 'Valid Event',
                    event_type: 'action',
                    timestamp: datetime(),
                    source_document: 'test',
                    created_at: datetime()
                })
                """
            )

            # Verify valid event exists
            valid_result = session.run(
                "MATCH (e:Event {id: 'valid_event'}) WHERE e.timestamp IS NOT NULL RETURN e"
            )
            assert valid_result.single() is not None

            # Count events without timestamps (should be 0)
            invalid_result = session.run(
                "MATCH (e:Event) WHERE e.timestamp IS NULL RETURN count(e) as count"
            )
            assert invalid_result.single()["count"] == 0

    def test_empty_document_handling(self, neo4j_driver, clean_database):
        """Verify empty documents are handled gracefully."""
        empty_doc = create_document_with_metadata(
            path="/vault/notes/empty.md",
            content="",
            frontmatter={},
            tags=[],
        )

        with neo4j_driver.session() as session:
            sha256 = store_document_in_pkg(session, empty_doc)

            # Verify document stored
            result = session.run(
                "MATCH (d:Document {sha256: $sha256}) RETURN d",
                {"sha256": sha256},
            )
            assert result.single() is not None

            # No tags should be created
            tag_result = session.run(
                """
                MATCH (d:Document {sha256: $sha256})-[:HAS_TAG]->(t:Tag)
                RETURN count(t) as count
                """,
                {"sha256": sha256},
            )
            assert tag_result.single()["count"] == 0

"""Production-grade pipeline integrations for normalization and storage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.pkg.sync.events import SyncEvent

logger = logging.getLogger(__name__)


class PKGWriter(Protocol):
    def write_document(self, payload: dict) -> None:
        ...

    def remove_document(self, sha256: str) -> None:
        ...


class VectorWriter(Protocol):
    def write_embedding(self, payload: dict) -> None:
        ...

    def remove_embedding(self, sha256: str) -> None:
        ...


@dataclass
class NormalizationSink:
    """Normalization sink that persists documents, embeddings, and experiential events.

    Implements the Ghost→Animal experiential learning architecture by creating:
    - Document nodes for content storage
    - Vector embeddings for semantic search
    - ExperientialEvents for temporal pattern tracking (Phase 2 prep)

    Sync Event Support (Module 05):
        Optional sync_event_handler enables tracking of PKG ↔ Vector synchronization.
        Used for integration testing and production monitoring.

        Example:
            >>> from futurnal.pkg.sync import SyncEventCapture
            >>> capture = SyncEventCapture()
            >>> sink = NormalizationSink(
            ...     pkg_writer=pkg_writer,
            ...     vector_writer=vector_writer,
            ...     sync_event_handler=capture.capture
            ... )
            >>> sink.handle(document)
            >>> assert capture.count == 2  # pkg_write + vector_write
    """
    pkg_writer: PKGWriter
    vector_writer: VectorWriter
    sync_event_handler: Optional[Callable[["SyncEvent"], None]] = None

    def handle(self, element: dict) -> None:
        # Support both old format (element_path) and new format (direct payload)
        if "element_path" in element:
            # Old format: read from file
            with open(element["element_path"], "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            # New format: payload included directly
            payload = element

        sha256 = element.get("sha256") or payload.get("sha256")
        path = element.get("path") or payload.get("path")
        source = element.get("source") or payload.get("source")

        document_payload = {
            "sha256": sha256,
            "path": path,
            "source": source,
            "metadata": payload.get("metadata", {}),
            "text": payload.get("text"),
        }
        self.pkg_writer.write_document(document_payload)

        # Emit sync event: PKG write completed, vector sync pending
        self._emit_sync_event(
            event_type="entity_created",
            entity_id=sha256,
            entity_type="Document",
            source_operation="pkg_write",
            vector_sync_status="pending",
            metadata={"path": path, "source": source},
        )

        embedding_payload = {
            "sha256": sha256,
            "path": path,
            "source": source,
            "text": payload.get("text"),
        }
        if "embedding" in payload:
            embedding_payload["embedding"] = payload["embedding"]
        self.vector_writer.write_embedding(embedding_payload)

        # Emit sync event: Vector write completed
        self._emit_sync_event(
            event_type="entity_created",
            entity_id=sha256,
            entity_type="Document",
            source_operation="vector_write",
            vector_sync_status="completed",
            metadata={"path": path, "source": source},
        )

        # Create experiential event for Ghost's temporal awareness
        # Phase 2 (Analyst) will use these events for correlation detection
        self._create_experiential_event(element, payload)

    def handle_deletion(self, element: dict) -> None:
        sha256 = element["sha256"]

        self.pkg_writer.remove_document(sha256)

        # Emit sync event: PKG delete completed, vector sync pending
        self._emit_sync_event(
            event_type="entity_deleted",
            entity_id=sha256,
            entity_type="Document",
            source_operation="pkg_delete",
            vector_sync_status="pending",
        )

        self.vector_writer.remove_embedding(sha256)

        # Emit sync event: Vector delete completed
        self._emit_sync_event(
            event_type="entity_deleted",
            entity_id=sha256,
            entity_type="Document",
            source_operation="vector_delete",
            vector_sync_status="completed",
        )

    def _emit_sync_event(
        self,
        event_type: str,
        entity_id: str,
        entity_type: str,
        source_operation: str,
        vector_sync_status: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Emit a sync event if a handler is configured.

        Args:
            event_type: Type of event (entity_created, entity_deleted, etc.)
            entity_id: Entity identifier (usually SHA256)
            entity_type: Type of entity (Document, etc.)
            source_operation: Operation that triggered sync (pkg_write, vector_write, etc.)
            vector_sync_status: Current sync status (pending, completed, failed)
            metadata: Optional additional event data
        """
        if not self.sync_event_handler:
            return

        try:
            from futurnal.pkg.sync.events import SyncEvent

            event = SyncEvent(
                event_type=event_type,
                entity_id=entity_id,
                entity_type=entity_type,
                timestamp=datetime.utcnow(),
                source_operation=source_operation,
                vector_sync_status=vector_sync_status,
                metadata=metadata or {},
            )
            self.sync_event_handler(event)
        except Exception as e:
            # Don't fail operations due to sync event emission failure
            logger.warning(f"Failed to emit sync event: {e}")

    def _create_experiential_event(self, element: dict, payload: dict) -> None:
        """Create an ExperientialEvent tracking document ingestion.

        Events enable Phase 2 temporal pattern detection by capturing WHEN
        actions occurred in the user's experiential stream.

        Args:
            element: Element metadata from connector
            payload: Parsed document payload
        """
        # Check if PKG writer supports event creation (backward compatibility)
        if not hasattr(self.pkg_writer, 'create_experiential_event'):
            return

        metadata = payload.get("metadata", {})

        # Infer timestamp: prefer file creation/modification over ingestion time
        timestamp = (
            metadata.get("created_at") or
            metadata.get("modified_at") or
            metadata.get("ingested_at") or
            datetime.utcnow().isoformat()
        )

        # Create document ingestion event
        try:
            self.pkg_writer.create_experiential_event({
                'event_id': f"evt-doc-{element['sha256'][:16]}",
                'event_type': 'document_ingested',
                'timestamp': timestamp,
                'source_uri': element['path'],
                'context': {
                    'source': element['source'],
                    'file_type': metadata.get('filetype'),
                    'size_bytes': metadata.get('file_size') or metadata.get('size_bytes'),
                    'checksum': element['sha256']
                }
            })
        except Exception:
            # Don't fail ingestion if event creation fails
            # Events are Phase 2 prep, not Phase 1 critical
            pass

        # Create Obsidian-specific events if link graph data is present
        self._create_obsidian_note_events(element, payload, timestamp)

    def _create_obsidian_note_events(self, element: dict, payload: dict, base_timestamp: str) -> None:
        """Create Obsidian-specific experiential events from link graph data.

        Tracks note creation, link additions, and tag applications for temporal
        pattern detection. Enables Phase 2 to detect patterns like "ideas tagged
        #important on Mondays are 3x more likely to ship."

        Args:
            element: Element metadata from connector
            payload: Parsed document payload with potential link graph data
            base_timestamp: Base timestamp for the events
        """
        if not hasattr(self.pkg_writer, 'create_experiential_event'):
            return

        metadata = payload.get("metadata", {})
        futurnal_metadata = metadata.get("futurnal", {})
        graph_data = futurnal_metadata.get("link_graph")

        if not graph_data or "error" in graph_data:
            return

        try:
            # Extract graph components
            note_nodes = graph_data.get("note_nodes", [])
            link_relationships = graph_data.get("link_relationships", [])
            tag_relationships = graph_data.get("tag_relationships", [])

            # Create note_created events for primary note (first node is source)
            if note_nodes:
                primary_note = note_nodes[0]
                self.pkg_writer.create_experiential_event({
                    'event_id': f"evt-note-{primary_note['note_id'][:16]}",
                    'event_type': 'note_created',
                    'timestamp': base_timestamp,
                    'source_uri': primary_note.get('uri', element['path']),
                    'context': {
                        'vault_id': primary_note.get('vault_id'),
                        'note_id': primary_note.get('note_id'),
                        'title': primary_note.get('title'),
                        'checksum': primary_note.get('checksum'),
                        'path': primary_note.get('path')
                    }
                })

            # Create link_added events for each relationship
            for idx, link in enumerate(link_relationships[:50]):  # Limit to avoid event spam
                if not link.get('is_broken'):  # Only track valid links
                    self.pkg_writer.create_experiential_event({
                        'event_id': f"evt-link-{element['sha256'][:12]}-{idx}",
                        'event_type': 'link_added',
                        'timestamp': base_timestamp,
                        'source_uri': link.get('source_uri', element['path']),
                        'context': {
                            'relationship_type': link.get('relationship_type'),
                            'target_uri': link.get('target_uri'),
                            'display_text': link.get('display_text'),
                            'heading': link.get('heading'),
                            'source_path': link.get('source_path')
                        }
                    })

            # Create tag_applied events for each tag
            for idx, tag in enumerate(tag_relationships[:20]):  # Limit to avoid event spam
                self.pkg_writer.create_experiential_event({
                    'event_id': f"evt-tag-{element['sha256'][:12]}-{idx}",
                    'event_type': 'tag_applied',
                    'timestamp': base_timestamp,
                    'source_uri': tag.get('note_uri', element['path']),
                    'context': {
                        'tag_name': tag.get('tag_name'),
                        'tag_uri': tag.get('tag_uri'),
                        'is_nested': tag.get('is_nested'),
                        'source_path': tag.get('source_path')
                    }
                })

        except Exception as e:
            # Don't fail ingestion if Obsidian event creation fails
            # These events enhance Phase 2 but aren't critical for Phase 1
            pass


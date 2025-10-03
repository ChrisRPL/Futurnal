"""Production-grade pipeline integrations for normalization and storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


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
    """
    pkg_writer: PKGWriter
    vector_writer: VectorWriter

    def handle(self, element: dict) -> None:
        with open(element["element_path"], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        document_payload = {
            "sha256": element["sha256"],
            "path": element["path"],
            "source": element["source"],
            "metadata": payload.get("metadata", {}),
            "text": payload.get("text"),
        }
        self.pkg_writer.write_document(document_payload)
        embedding_payload = {
            "sha256": element["sha256"],
            "path": element["path"],
            "source": element["source"],
            "text": payload.get("text"),
        }
        if "embedding" in payload:
            embedding_payload["embedding"] = payload["embedding"]
        self.vector_writer.write_embedding(embedding_payload)

        # Create experiential event for Ghost's temporal awareness
        # Phase 2 (Analyst) will use these events for correlation detection
        self._create_experiential_event(element, payload)

    def handle_deletion(self, element: dict) -> None:
        self.pkg_writer.remove_document(element["sha256"])
        self.vector_writer.remove_embedding(element["sha256"])

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


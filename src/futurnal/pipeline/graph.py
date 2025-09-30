"""Graph storage integration for the Personal Knowledge Graph (PKG).

Creates a thin wrapper around the Neo4j driver with settings models that keep
URI validation consistent with the broader storage configuration. The writer is
used by ingestion sinks to project document metadata into the PKG in lockstep
with the vector index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator

from neo4j import GraphDatabase
from neo4j import Driver

logger = logging.getLogger(__name__)


class Neo4jSettings(BaseModel):
    uri: str = Field(...)
    username: str = Field(...)
    password: SecretStr = Field(...)
    database: Optional[str] = Field(default=None)
    encrypted: bool = Field(default=False)

    @field_validator("uri")
    def _validate_uri(cls, value: str) -> str:
        if not value.startswith("bolt://") and not value.startswith("neo4j://"):
            raise ValueError("URI must start with bolt:// or neo4j://")
        return value


@dataclass
class Neo4jPKGWriter:
    """Persists document metadata into an embedded Neo4j instance."""

    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    encrypted: bool = False
    driver: Optional[Driver] = None

    def __post_init__(self) -> None:
        if self.driver:
            self._driver = self.driver
            return
        if not self.uri or self.username is None or self.password is None:
            raise ValueError("Neo4jPKGWriter requires uri, username, and password when driver not provided")
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            encrypted=self.encrypted,
        )

    def close(self) -> None:
        self._driver.close()

    def write_document(self, payload: Dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        parameters = {
            "sha": payload["sha256"],
            "path": payload.get("path"),
            "source": payload.get("source"),
            "text": payload.get("text", ""),
            "metadata": metadata,
            "size_bytes": payload.get("metadata", {}).get("size_bytes"),
            "modified_at": payload.get("metadata", {}).get("modified_at"),
            "ingested_at": payload.get("metadata", {}).get("ingested_at"),
        }

        def _upsert(tx):
            tx.run(
                """
                MERGE (d:Document {sha256: $sha})
                ON CREATE SET d.created_at = datetime()
                SET d.path = $path,
                    d.source = $source,
                    d.text = $text,
                    d.metadata = $metadata,
                    d.size_bytes = $size_bytes,
                    d.modified_at = CASE WHEN $modified_at IS NULL THEN d.modified_at ELSE datetime($modified_at) END,
                    d.ingested_at = CASE WHEN $ingested_at IS NULL THEN d.ingested_at ELSE datetime($ingested_at) END,
                    d.updated_at = datetime()
                WITH d
                MERGE (s:Source {name: $source})
                SET s.updated_at = datetime()
                MERGE (d)-[:BELONGS_TO]->(s)
                """,
                parameters,
            )

        with self._driver.session(database=self.database) as session:
            session.execute_write(_upsert)

    def remove_document(self, sha256: str) -> None:
        def _delete(tx):
            tx.run(
                """
                MATCH (d:Document {sha256: $sha})
                DETACH DELETE d
                """,
                {"sha": sha256},
            )

        with self._driver.session(database=self.database) as session:
            session.execute_write(_delete)

    def create_note_node(self, note_data: Dict[str, Any]) -> None:
        """Create or update a Note node in the PKG.

        Args:
            note_data: Dictionary containing note properties
                - vault_id: Vault identifier
                - note_id: Note identifier within vault
                - title: Note title
                - path: File path
                - checksum: Content checksum
                - uri: Note URI
        """
        parameters = {
            "vault_id": note_data["vault_id"],
            "note_id": note_data["note_id"],
            "title": note_data["title"],
            "path": note_data["path"],
            "checksum": note_data["checksum"],
            "uri": note_data["uri"],
            "created_at": note_data.get("created_at"),
        }

        def _upsert_note(tx):
            tx.run(
                """
                MERGE (n:Note {vault_id: $vault_id, note_id: $note_id})
                ON CREATE SET
                    n.created_at = CASE WHEN $created_at IS NULL THEN datetime() ELSE datetime($created_at) END,
                    n.uri = $uri
                SET
                    n.title = $title,
                    n.path = $path,
                    n.checksum = $checksum,
                    n.updated_at = datetime()
                WITH n
                MERGE (v:Vault {id: $vault_id})
                SET v.updated_at = datetime()
                MERGE (n)-[:IN_VAULT]->(v)
                """,
                parameters,
            )

        with self._driver.session(database=self.database) as session:
            session.execute_write(_upsert_note)
            logger.debug(f"Created/updated note node: {parameters['uri']}")

    def create_note_relationship(self, relationship_data: Dict[str, Any]) -> None:
        """Create a relationship between two notes.

        Args:
            relationship_data: Dictionary containing relationship properties
                - source_uri: Source note URI
                - target_uri: Target note URI
                - relationship_type: Type of relationship
                - source_path: Original file path for provenance
                - offset: Character offset in source file
                - checksum: Relationship checksum for deduplication
                - heading: Section heading (for references_heading)
                - block_id: Block ID (for block references)
                - display_text: Original display text
                - is_broken: Whether target exists
        """
        parameters = {
            "source_uri": relationship_data["source_uri"],
            "target_uri": relationship_data["target_uri"],
            "relationship_type": relationship_data["relationship_type"],
            "source_path": relationship_data["source_path"],
            "offset": relationship_data.get("offset"),
            "checksum": relationship_data["checksum"],
            "heading": relationship_data.get("heading"),
            "block_id": relationship_data.get("block_id"),
            "display_text": relationship_data.get("display_text"),
            "is_broken": relationship_data["is_broken"],
            "created_at": relationship_data.get("created_at"),
        }

        def _create_relationship(tx):
            # Dynamic relationship type creation
            rel_type = parameters["relationship_type"].upper()

            query = f"""
            MATCH (source:Note {{uri: $source_uri}})
            MATCH (target:Note {{uri: $target_uri}})
            MERGE (source)-[r:{rel_type} {{checksum: $checksum}}]->(target)
            ON CREATE SET
                r.created_at = CASE WHEN $created_at IS NULL THEN datetime() ELSE datetime($created_at) END,
                r.source_path = $source_path,
                r.offset = $offset,
                r.heading = $heading,
                r.block_id = $block_id,
                r.display_text = $display_text,
                r.is_broken = $is_broken
            SET
                r.updated_at = datetime()
            RETURN r
            """

            result = tx.run(query, parameters)
            return result.single()

        with self._driver.session(database=self.database) as session:
            result = session.execute_write(_create_relationship)
            if result:
                logger.debug(f"Created {parameters['relationship_type']} relationship: {parameters['source_uri']} -> {parameters['target_uri']}")
            else:
                logger.warning(f"Failed to create relationship (nodes may not exist): {parameters['source_uri']} -> {parameters['target_uri']}")

    def create_tag_relationship(self, tag_data: Dict[str, Any]) -> None:
        """Create a relationship between a note and a tag.

        Args:
            tag_data: Dictionary containing tag relationship properties
                - note_uri: Note URI
                - tag_name: Tag name
                - tag_uri: Tag URI
                - source_path: Original file path
                - offset: Character offset
                - is_nested: Whether tag is nested
        """
        parameters = {
            "note_uri": tag_data["note_uri"],
            "tag_name": tag_data["tag_name"],
            "tag_uri": tag_data["tag_uri"],
            "source_path": tag_data["source_path"],
            "offset": tag_data.get("offset"),
            "is_nested": tag_data["is_nested"],
            "created_at": tag_data.get("created_at"),
        }

        def _create_tag_relationship(tx):
            tx.run(
                """
                MATCH (n:Note {uri: $note_uri})
                MERGE (t:Tag {uri: $tag_uri})
                ON CREATE SET
                    t.name = $tag_name,
                    t.is_nested = $is_nested,
                    t.created_at = CASE WHEN $created_at IS NULL THEN datetime() ELSE datetime($created_at) END
                SET
                    t.updated_at = datetime()
                MERGE (n)-[r:HAS_TAG]->(t)
                ON CREATE SET
                    r.source_path = $source_path,
                    r.offset = $offset,
                    r.created_at = CASE WHEN $created_at IS NULL THEN datetime() ELSE datetime($created_at) END
                SET
                    r.updated_at = datetime()
                """,
                parameters,
            )

        with self._driver.session(database=self.database) as session:
            session.execute_write(_create_tag_relationship)
            logger.debug(f"Created tag relationship: {parameters['note_uri']} -> {parameters['tag_name']}")

    def update_note_path(self, vault_id: str, note_id: str, old_path: str, new_path: str) -> None:
        """Update note path when file is renamed or moved.

        Args:
            vault_id: Vault identifier
            note_id: Note identifier
            old_path: Previous file path
            new_path: New file path
        """
        parameters = {
            "vault_id": vault_id,
            "note_id": note_id,
            "old_path": old_path,
            "new_path": new_path,
        }

        def _update_path(tx):
            result = tx.run(
                """
                MATCH (n:Note {vault_id: $vault_id, note_id: $note_id})
                SET n.path = $new_path,
                    n.previous_path = $old_path,
                    n.path_updated_at = datetime(),
                    n.updated_at = datetime()
                RETURN n.uri as uri
                """,
                parameters,
            )
            return result.single()

        with self._driver.session(database=self.database) as session:
            result = session.execute_write(_update_path)
            if result:
                logger.info(f"Updated note path: {result['uri']} from {old_path} to {new_path}")
            else:
                logger.warning(f"Note not found for path update: {vault_id}/{note_id}")

    def remove_note_relationships(self, note_uri: str, relationship_types: Optional[List[str]] = None) -> int:
        """Remove relationships for a note (useful for re-processing).

        Args:
            note_uri: Note URI
            relationship_types: Optional list of relationship types to remove (default: all)

        Returns:
            Number of relationships removed
        """
        parameters = {"note_uri": note_uri}

        def _remove_relationships(tx):
            if relationship_types:
                # Remove specific relationship types
                type_conditions = " OR ".join([f"type(r) = '{rt.upper()}'" for rt in relationship_types])
                query = f"""
                MATCH (n:Note {{uri: $note_uri}})-[r]-()
                WHERE {type_conditions}
                DELETE r
                RETURN count(r) as deleted_count
                """
            else:
                # Remove all relationships
                query = """
                MATCH (n:Note {uri: $note_uri})-[r]-()
                DELETE r
                RETURN count(r) as deleted_count
                """

            result = tx.run(query, parameters)
            return result.single()["deleted_count"]

        with self._driver.session(database=self.database) as session:
            deleted_count = session.execute_write(_remove_relationships)
            logger.debug(f"Removed {deleted_count} relationships for note: {note_uri}")
            return deleted_count

    def get_note_statistics(self, vault_id: Optional[str] = None) -> Dict[str, int]:
        """Get statistics about notes and relationships in the graph.

        Args:
            vault_id: Optional vault ID to filter statistics

        Returns:
            Dictionary with graph statistics
        """
        parameters = {"vault_id": vault_id} if vault_id else {}

        def _get_stats(tx):
            if vault_id:
                notes_query = "MATCH (n:Note {vault_id: $vault_id}) RETURN count(n) as note_count"
                relationships_query = """
                MATCH (n1:Note {vault_id: $vault_id})-[r]-(n2:Note {vault_id: $vault_id})
                RETURN count(r) as relationship_count
                """
                tags_query = """
                MATCH (n:Note {vault_id: $vault_id})-[:HAS_TAG]->(t:Tag)
                RETURN count(DISTINCT t) as tag_count
                """
            else:
                notes_query = "MATCH (n:Note) RETURN count(n) as note_count"
                relationships_query = "MATCH (n1:Note)-[r]-(n2:Note) RETURN count(r) as relationship_count"
                tags_query = "MATCH (n:Note)-[:HAS_TAG]->(t:Tag) RETURN count(DISTINCT t) as tag_count"

            stats = {}

            # Get note count
            result = tx.run(notes_query, parameters)
            stats["notes"] = result.single()["note_count"]

            # Get relationship count
            result = tx.run(relationships_query, parameters)
            stats["relationships"] = result.single()["relationship_count"]

            # Get tag count
            result = tx.run(tags_query, parameters)
            stats["tags"] = result.single()["tag_count"]

            return stats

        with self._driver.session(database=self.database) as session:
            return session.execute_read(_get_stats)



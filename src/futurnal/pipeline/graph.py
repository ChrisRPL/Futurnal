"""Graph storage integration for the Personal Knowledge Graph (PKG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, SecretStr

from neo4j import GraphDatabase
from neo4j import Driver


class Neo4jSettings(BaseModel):
    uri: str = Field(...)
    username: str = Field(...)
    password: SecretStr = Field(...)
    database: Optional[str] = Field(default=None)
    encrypted: bool = Field(default=False)


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



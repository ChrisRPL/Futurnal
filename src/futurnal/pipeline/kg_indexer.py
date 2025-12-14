"""Knowledge Graph Indexer for GraphRAG integration.

Indexes parsed documents from the ingestion pipeline into:
1. ChromaDB - for vector similarity search
2. Neo4j - for graph traversal and relationship queries

This bridges the gap between document ingestion and the hybrid search system,
enabling GraphRAG-powered search over user's personal knowledge.

Usage:
    from futurnal.pipeline.kg_indexer import KnowledgeGraphIndexer

    indexer = KnowledgeGraphIndexer(workspace_dir)
    indexed_count = indexer.index_all_parsed()
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KnowledgeGraphIndexer:
    """Indexes parsed documents into ChromaDB and Neo4j for GraphRAG search.

    Reads JSON documents from the workspace parsed/ and entities/ directories
    and indexes them into the knowledge graph stores:

    - ChromaDB: Vector embeddings for semantic similarity search
    - Neo4j: Document nodes, entity nodes, and relationships for graph traversal

    This enables the hybrid search system to find relevant content through both
    vector similarity and graph-based relevance.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        chromadb_path: Optional[Path] = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            workspace_dir: Path to ingestion workspace (default: ~/.futurnal/workspace)
            chromadb_path: Path to ChromaDB storage (default: workspace/chromadb)
            neo4j_uri: Neo4j connection URI
            neo4j_auth: Tuple of (username, password), None for no auth
        """
        self._workspace = workspace_dir or self._default_workspace()
        self._parsed_dir = self._workspace / "parsed"
        self._entities_dir = self._workspace / "entities"
        self._imap_dir = self._workspace / "imap"

        # ChromaDB settings - use same path as EmbeddingServiceConfig for consistency
        # Default is ~/.futurnal/embeddings (same as embedding service)
        self._chromadb_path = chromadb_path or (Path.home() / ".futurnal" / "embeddings")

        # Neo4j settings
        self._neo4j_uri = neo4j_uri
        self._neo4j_auth = neo4j_auth

        # Lazy-loaded clients
        self._chroma_client = None
        self._chroma_collection = None
        self._neo4j_driver = None

    def _default_workspace(self) -> Path:
        """Get default workspace path."""
        return Path.home() / ".futurnal" / "workspace"

    def _get_chroma_collection(self):
        """Get or create ChromaDB collection.

        Uses the same collection as the embedding service (futurnal-entities-v1.0)
        to ensure GraphRAG search can find indexed documents.
        """
        if self._chroma_collection is not None:
            return self._chroma_collection

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._chromadb_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self._chromadb_path))

            # Use sentence-transformers for embeddings (same as embedding service)
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Use the same collection name as SchemaVersionedEmbeddingStore
            # to ensure GraphRAG search can find these documents
            collection_name = "futurnal-entities-v1.0"

            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "Static entity embeddings",
                    "schema_version": "1.0",
                    "entity_type": "static_entity",
                },
                embedding_function=embedding_fn,
            )

            logger.info(f"Connected to ChromaDB collection '{collection_name}' at {self._chromadb_path}")
            return self._chroma_collection

        except ImportError:
            logger.warning("ChromaDB not installed - vector indexing disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return None

    def _get_neo4j_driver(self):
        """Get or create Neo4j driver."""
        if self._neo4j_driver is not None:
            return self._neo4j_driver

        try:
            from neo4j import GraphDatabase

            if self._neo4j_auth:
                self._neo4j_driver = GraphDatabase.driver(
                    self._neo4j_uri,
                    auth=self._neo4j_auth,
                )
            else:
                self._neo4j_driver = GraphDatabase.driver(
                    self._neo4j_uri,
                    auth=None,
                )

            # Test connection
            with self._neo4j_driver.session() as session:
                session.run("RETURN 1")

            logger.info(f"Connected to Neo4j at {self._neo4j_uri}")
            return self._neo4j_driver

        except ImportError:
            logger.warning("neo4j package not installed - graph indexing disabled")
            return None
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            return None

    def _compute_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Compute a unique document ID from content and metadata."""
        # Use source path or generate from content hash
        source_path = metadata.get("path") or metadata.get("source_file") or ""
        if source_path:
            return hashlib.sha256(source_path.encode()).hexdigest()[:16]
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_text_content(self, doc: Dict[str, Any]) -> str:
        """Extract searchable text content from a parsed document."""
        parts = []

        # Get main text content
        if "text" in doc:
            parts.append(doc["text"])
        elif "content" in doc:
            parts.append(doc["content"])
        elif "elements" in doc:
            # Handle unstructured.io format
            for element in doc["elements"]:
                if isinstance(element, dict) and "text" in element:
                    parts.append(element["text"])

        # Include metadata fields that are searchable
        metadata = doc.get("metadata", {})
        if "title" in metadata:
            parts.insert(0, metadata["title"])
        if "subject" in metadata:
            parts.insert(0, metadata["subject"])
        if "filename" in metadata:
            parts.insert(0, metadata["filename"])

        return "\n".join(filter(None, parts))

    def index_document_to_chroma(self, doc: Dict[str, Any], doc_id: str) -> bool:
        """Index a single document to ChromaDB.

        Args:
            doc: Parsed document dictionary
            doc_id: Unique document identifier

        Returns:
            True if indexed successfully
        """
        collection = self._get_chroma_collection()
        if collection is None:
            return False

        try:
            content = self._extract_text_content(doc)
            if not content or len(content.strip()) < 10:
                logger.debug(f"Skipping document {doc_id}: insufficient content")
                return False

            metadata = doc.get("metadata", {})

            # Prepare metadata for ChromaDB (must be flat primitives)
            chroma_metadata = {
                "source": str(metadata.get("source", "unknown")),
                "source_type": str(metadata.get("source_type", "document")),
                "path": str(metadata.get("path", "")),
                "entity_type": "Document",
                "indexed_at": datetime.utcnow().isoformat(),
                # Required for SchemaAwareRetrieval filtering
                "schema_version": 1,
                "needs_reembedding": False,
            }

            # Add optional metadata
            if "title" in metadata:
                chroma_metadata["title"] = str(metadata["title"])[:200]
            if "filename" in metadata:
                chroma_metadata["filename"] = str(metadata["filename"])
            if "ingested_at" in metadata:
                chroma_metadata["ingested_at"] = str(metadata["ingested_at"])

            # Store the full content for retrieval
            chroma_metadata["content"] = content[:5000]  # Limit for metadata storage

            collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[chroma_metadata],
            )

            logger.debug(f"Indexed document {doc_id} to ChromaDB")
            return True

        except Exception as e:
            logger.warning(f"Failed to index document {doc_id} to ChromaDB: {e}")
            return False

    def index_document_to_neo4j(self, doc: Dict[str, Any], doc_id: str) -> bool:
        """Index a single document to Neo4j.

        Args:
            doc: Parsed document dictionary
            doc_id: Unique document identifier

        Returns:
            True if indexed successfully
        """
        driver = self._get_neo4j_driver()
        if driver is None:
            return False

        try:
            content = self._extract_text_content(doc)
            metadata = doc.get("metadata", {})

            def _create_document_node(tx):
                # Create Document node
                tx.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    ON CREATE SET d.created_at = datetime()
                    SET d.content = $content,
                        d.source = $source,
                        d.source_type = $source_type,
                        d.path = $path,
                        d.title = $title,
                        d.indexed_at = datetime(),
                        d.updated_at = datetime()
                    WITH d
                    MERGE (s:Source {name: $source})
                    SET s.updated_at = datetime()
                    MERGE (d)-[:BELONGS_TO]->(s)
                    """,
                    {
                        "doc_id": doc_id,
                        "content": content[:10000],  # Limit content size
                        "source": metadata.get("source", "unknown"),
                        "source_type": metadata.get("source_type", "document"),
                        "path": metadata.get("path", ""),
                        "title": metadata.get("title", metadata.get("filename", "")),
                    },
                )

            with driver.session() as session:
                session.execute_write(_create_document_node)

            logger.debug(f"Indexed document {doc_id} to Neo4j")
            return True

        except Exception as e:
            logger.warning(f"Failed to index document {doc_id} to Neo4j: {e}")
            return False

    def index_entity_to_stores(self, entity: Dict[str, Any], doc_id: str) -> Tuple[bool, bool]:
        """Index an extracted entity to both stores.

        Args:
            entity: Entity dictionary with name, type, etc.
            doc_id: Parent document ID

        Returns:
            Tuple of (chroma_success, neo4j_success)
        """
        entity_name = entity.get("name", "")
        entity_type = entity.get("type", "Entity")
        entity_id = f"entity_{hashlib.sha256(f'{entity_type}:{entity_name}'.encode()).hexdigest()[:12]}"

        chroma_success = False
        neo4j_success = False

        # Index to ChromaDB
        collection = self._get_chroma_collection()
        if collection is not None:
            try:
                content = f"{entity_type}: {entity_name}"
                if "description" in entity:
                    content += f". {entity['description']}"

                collection.upsert(
                    ids=[entity_id],
                    documents=[content],
                    metadatas=[{
                        "entity_type": entity_type,
                        "entity_name": entity_name,
                        "parent_doc_id": doc_id,
                        "content": content,
                        "indexed_at": datetime.utcnow().isoformat(),
                        # Required for SchemaAwareRetrieval filtering
                        "schema_version": 1,
                        "needs_reembedding": False,
                    }],
                )
                chroma_success = True
            except Exception as e:
                logger.debug(f"Failed to index entity {entity_name} to ChromaDB: {e}")

        # Index to Neo4j
        driver = self._get_neo4j_driver()
        if driver is not None:
            try:
                def _create_entity_node(tx):
                    # Create entity node with dynamic label based on type
                    label = entity_type.replace(" ", "_")
                    tx.run(
                        f"""
                        MERGE (e:{label} {{name: $name}})
                        ON CREATE SET e.created_at = datetime()
                        SET e.entity_type = $entity_type,
                            e.updated_at = datetime()
                        WITH e
                        MATCH (d:Document {{id: $doc_id}})
                        MERGE (d)-[:MENTIONS]->(e)
                        """,
                        {
                            "name": entity_name,
                            "entity_type": entity_type,
                            "doc_id": doc_id,
                        },
                    )

                with driver.session() as session:
                    session.execute_write(_create_entity_node)
                neo4j_success = True
            except Exception as e:
                logger.debug(f"Failed to index entity {entity_name} to Neo4j: {e}")

        return chroma_success, neo4j_success

    def index_all_parsed(self) -> Dict[str, int]:
        """Index all parsed documents from the workspace.

        Reads all JSON files from parsed/, imap/, and entities/ directories
        and indexes them to ChromaDB and Neo4j.

        Returns:
            Dictionary with indexing statistics
        """
        stats = {
            "documents_found": 0,
            "chroma_indexed": 0,
            "neo4j_indexed": 0,
            "entities_indexed": 0,
            "errors": 0,
        }

        # Process parsed documents
        all_docs: List[Tuple[Path, Dict[str, Any]]] = []

        for source_dir in [self._parsed_dir, self._imap_dir]:
            if source_dir.exists():
                for json_file in source_dir.glob("*.json"):
                    try:
                        with open(json_file) as f:
                            doc = json.load(f)
                        all_docs.append((json_file, doc))
                    except Exception as e:
                        logger.debug(f"Failed to read {json_file}: {e}")
                        stats["errors"] += 1

        stats["documents_found"] = len(all_docs)
        logger.info(f"Found {len(all_docs)} documents to index")

        # Index each document
        for json_file, doc in all_docs:
            metadata = doc.get("metadata", {})
            doc_id = self._compute_doc_id(
                self._extract_text_content(doc),
                metadata,
            )

            # Index to ChromaDB
            if self.index_document_to_chroma(doc, doc_id):
                stats["chroma_indexed"] += 1

            # Index to Neo4j
            if self.index_document_to_neo4j(doc, doc_id):
                stats["neo4j_indexed"] += 1

        # Process extracted entities
        if self._entities_dir.exists():
            for json_file in self._entities_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        entity_doc = json.load(f)

                    doc_id = entity_doc.get("document_id", json_file.stem)
                    entities = entity_doc.get("entities", [])

                    for entity in entities:
                        chroma_ok, neo4j_ok = self.index_entity_to_stores(entity, doc_id)
                        if chroma_ok or neo4j_ok:
                            stats["entities_indexed"] += 1

                except Exception as e:
                    logger.debug(f"Failed to process entities from {json_file}: {e}")
                    stats["errors"] += 1

        logger.info(
            f"Indexing complete: {stats['chroma_indexed']} to ChromaDB, "
            f"{stats['neo4j_indexed']} to Neo4j, {stats['entities_indexed']} entities"
        )

        return stats

    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics.

        Returns:
            Dictionary with ChromaDB and Neo4j statistics
        """
        stats = {
            "chromadb": {"connected": False, "document_count": 0},
            "neo4j": {"connected": False, "node_count": 0, "relationship_count": 0},
        }

        # ChromaDB stats
        collection = self._get_chroma_collection()
        if collection is not None:
            try:
                stats["chromadb"]["connected"] = True
                stats["chromadb"]["document_count"] = collection.count()
            except Exception:
                pass

        # Neo4j stats
        driver = self._get_neo4j_driver()
        if driver is not None:
            try:
                with driver.session() as session:
                    result = session.run(
                        """
                        MATCH (n) WITH count(n) as nodes
                        MATCH ()-[r]->() WITH nodes, count(r) as rels
                        RETURN nodes, rels
                        """
                    )
                    record = result.single()
                    if record:
                        stats["neo4j"]["connected"] = True
                        stats["neo4j"]["node_count"] = record["nodes"]
                        stats["neo4j"]["relationship_count"] = record["rels"]
            except Exception:
                pass

        return stats

    def close(self) -> None:
        """Close connections to ChromaDB and Neo4j."""
        if self._neo4j_driver:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass
            self._neo4j_driver = None

        # ChromaDB persistent client doesn't need explicit close
        self._chroma_client = None
        self._chroma_collection = None


def index_workspace_documents(
    workspace_dir: Optional[Path] = None,
    neo4j_uri: str = "bolt://localhost:7687",
) -> Dict[str, int]:
    """Convenience function to index all workspace documents.

    Args:
        workspace_dir: Path to workspace (default: ~/.futurnal/workspace)
        neo4j_uri: Neo4j connection URI

    Returns:
        Indexing statistics
    """
    indexer = KnowledgeGraphIndexer(
        workspace_dir=workspace_dir,
        neo4j_uri=neo4j_uri,
    )
    try:
        return indexer.index_all_parsed()
    finally:
        indexer.close()

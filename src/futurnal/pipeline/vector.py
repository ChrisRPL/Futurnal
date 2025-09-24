"""Vector storage integration backed by ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:  # pragma: no cover - import is optional for tests
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.utils import embedding_functions
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ChromaDB dependency is required for ChromaVectorWriter") from exc


@dataclass
class ChromaVectorWriter:
    """Writes embeddings into a Chroma persistent collection."""

    persist_directory: Path
    collection_name: str = "futurnal-ingestion"
    embedding_model: str = "all-MiniLM-L6-v2"
    client: Optional[ClientAPI] = None
    embedding_function: Optional[embedding_functions.EmbeddingFunction] = None

    def __post_init__(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = self.client or chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = self._ensure_collection()

    def _ensure_collection(self) -> Collection:
        embedding_fn = self.embedding_function or embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"source": "futurnal"},
            embedding_function=embedding_fn,
        )

    def write_embedding(self, payload: Dict[str, Any]) -> None:
        sha = payload["sha256"]
        text = payload.get("text")
        embedding: Optional[Sequence[float]] = payload.get("embedding")
        metadata = {
            "path": payload.get("path"),
            "source": payload.get("source"),
        }

        if not text and embedding is None:
            raise ValueError("Vector payload requires either text for embedding or precomputed embedding")

        kwargs: Dict[str, Any] = {
            "ids": [sha],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
            kwargs["documents"] = [text or ""]
        else:
            kwargs["documents"] = [text]

        self._collection.upsert(**kwargs)

    def remove_embedding(self, sha256: str) -> None:
        self._collection.delete(ids=[sha256])

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()



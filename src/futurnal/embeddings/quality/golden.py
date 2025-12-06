"""Golden Embeddings Manager.

Manages golden/reference embeddings for quality comparison.
Golden embeddings are high-quality, manually validated embeddings
used as reference points for quality scoring.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class GoldenEmbeddingsManager:
    """Manages golden/reference embeddings for quality comparison.

    Golden embeddings are high-quality, manually validated embeddings
    used as reference points for quality scoring. They enable:
    - Quality measurement via cosine similarity
    - Validation of new embeddings
    - Quality baseline establishment

    Storage format: JSON files per entity type with metadata.

    Example:
        manager = GoldenEmbeddingsManager(
            storage_path=Path("~/.futurnal/golden_embeddings")
        )

        # Add a validated golden embedding
        manager.add_golden_embedding(
            entity_type="Person",
            embedding=np.array([0.1, 0.2, ...]),
            metadata={"entity_name": "Reference Person", "validated_by": "human"},
        )

        # Get golden embeddings for comparison
        goldens = manager.get_golden_embeddings("Person")
    """

    # Default storage subdirectory
    DEFAULT_SUBDIR = "golden_embeddings"

    # Maximum golden embeddings per entity type
    MAX_GOLDENS_PER_TYPE = 50

    def __init__(
        self,
        storage_path: Optional[Path] = None,
    ) -> None:
        """Initialize golden embeddings manager.

        Args:
            storage_path: Path to storage directory. If None, uses default.
        """
        if storage_path is None:
            storage_path = Path.home() / ".futurnal" / self.DEFAULT_SUBDIR
        self._storage_path = Path(storage_path).expanduser()
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()

        # Cache of loaded embeddings: entity_type -> list of (embedding, metadata)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

        # Load existing golden embeddings
        self._load_all()

        logger.info(
            f"Initialized GoldenEmbeddingsManager at {self._storage_path} "
            f"with {sum(len(v) for v in self._cache.values())} golden embeddings"
        )

    @property
    def supported_types(self) -> Set[str]:
        """Entity types with golden embeddings available.

        Returns:
            Set of entity type names with at least one golden embedding.
        """
        with self._lock:
            return set(self._cache.keys())

    def get_golden_embeddings(self, entity_type: str) -> List[np.ndarray]:
        """Get golden embeddings for entity type.

        Args:
            entity_type: Type of entity (e.g., "Person", "Event")

        Returns:
            List of numpy arrays, empty if no golden embeddings exist.
        """
        with self._lock:
            entries = self._cache.get(entity_type, [])
            return [np.array(entry["embedding"]) for entry in entries]

    def get_golden_embeddings_with_metadata(
        self,
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """Get golden embeddings with their metadata.

        Args:
            entity_type: Type of entity (e.g., "Person", "Event")

        Returns:
            List of dicts with 'embedding' and 'metadata' keys.
        """
        with self._lock:
            return list(self._cache.get(entity_type, []))

    def add_golden_embedding(
        self,
        entity_type: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a validated golden embedding.

        Args:
            entity_type: Type of entity (e.g., "Person", "Event")
            embedding: The embedding vector (will be normalized)
            metadata: Optional metadata about this golden embedding

        Returns:
            Unique ID for the added golden embedding

        Raises:
            ValueError: If max golden embeddings reached for this type
        """
        if metadata is None:
            metadata = {}

        # Normalize embedding
        embedding = self._normalize(embedding)

        # Generate unique ID
        golden_id = f"{entity_type}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        with self._lock:
            # Initialize list if needed
            if entity_type not in self._cache:
                self._cache[entity_type] = []

            # Check capacity
            if len(self._cache[entity_type]) >= self.MAX_GOLDENS_PER_TYPE:
                raise ValueError(
                    f"Maximum golden embeddings ({self.MAX_GOLDENS_PER_TYPE}) "
                    f"reached for type '{entity_type}'"
                )

            # Add to cache
            entry = {
                "id": golden_id,
                "embedding": embedding.tolist(),
                "metadata": {
                    **metadata,
                    "added_at": datetime.utcnow().isoformat(),
                },
            }
            self._cache[entity_type].append(entry)

            # Persist to disk
            self._save_type(entity_type)

        logger.info(f"Added golden embedding {golden_id} for type '{entity_type}'")
        return golden_id

    def remove_golden_embedding(
        self,
        entity_type: str,
        golden_id: str,
    ) -> bool:
        """Remove a golden embedding by ID.

        Args:
            entity_type: Type of entity
            golden_id: ID of golden embedding to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if entity_type not in self._cache:
                return False

            original_len = len(self._cache[entity_type])
            self._cache[entity_type] = [
                entry
                for entry in self._cache[entity_type]
                if entry.get("id") != golden_id
            ]

            if len(self._cache[entity_type]) < original_len:
                self._save_type(entity_type)
                logger.info(f"Removed golden embedding {golden_id}")
                return True

            return False

    def get_count(self, entity_type: Optional[str] = None) -> int:
        """Get count of golden embeddings.

        Args:
            entity_type: If provided, count for this type only.
                         If None, return total count.

        Returns:
            Number of golden embeddings.
        """
        with self._lock:
            if entity_type is not None:
                return len(self._cache.get(entity_type, []))
            return sum(len(entries) for entries in self._cache.values())

    def clear(self, entity_type: Optional[str] = None) -> int:
        """Clear golden embeddings.

        Args:
            entity_type: If provided, clear only this type.
                         If None, clear all types.

        Returns:
            Number of embeddings removed.
        """
        with self._lock:
            if entity_type is not None:
                count = len(self._cache.get(entity_type, []))
                if entity_type in self._cache:
                    del self._cache[entity_type]
                    self._delete_type_file(entity_type)
                return count
            else:
                count = sum(len(entries) for entries in self._cache.values())
                for et in list(self._cache.keys()):
                    self._delete_type_file(et)
                self._cache.clear()
                return count

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length.

        Args:
            embedding: Input embedding vector

        Returns:
            L2-normalized embedding
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _get_type_file_path(self, entity_type: str) -> Path:
        """Get file path for entity type storage.

        Args:
            entity_type: Type of entity

        Returns:
            Path to JSON file for this type
        """
        # Sanitize entity type for filename
        safe_name = entity_type.replace("/", "_").replace("\\", "_")
        return self._storage_path / f"{safe_name}.json"

    def _save_type(self, entity_type: str) -> None:
        """Save golden embeddings for a type to disk.

        Args:
            entity_type: Type of entity to save
        """
        file_path = self._get_type_file_path(entity_type)
        entries = self._cache.get(entity_type, [])

        with open(file_path, "w") as f:
            json.dump(
                {
                    "entity_type": entity_type,
                    "updated_at": datetime.utcnow().isoformat(),
                    "count": len(entries),
                    "embeddings": entries,
                },
                f,
                indent=2,
            )

        logger.debug(f"Saved {len(entries)} golden embeddings for '{entity_type}'")

    def _load_all(self) -> None:
        """Load all golden embeddings from storage."""
        for file_path in self._storage_path.glob("*.json"):
            try:
                self._load_type_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to load golden embeddings from {file_path}: {e}")

    def _load_type_file(self, file_path: Path) -> None:
        """Load golden embeddings from a single file.

        Args:
            file_path: Path to JSON file
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        entity_type = data.get("entity_type", file_path.stem)
        entries = data.get("embeddings", [])

        self._cache[entity_type] = entries
        logger.debug(f"Loaded {len(entries)} golden embeddings for '{entity_type}'")

    def _delete_type_file(self, entity_type: str) -> None:
        """Delete storage file for entity type.

        Args:
            entity_type: Type of entity
        """
        file_path = self._get_type_file_path(entity_type)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted golden embeddings file for '{entity_type}'")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about golden embeddings.

        Returns:
            Dictionary with counts per type and totals.
        """
        with self._lock:
            return {
                "total_count": sum(len(v) for v in self._cache.values()),
                "type_count": len(self._cache),
                "counts_by_type": {
                    entity_type: len(entries)
                    for entity_type, entries in self._cache.items()
                },
                "storage_path": str(self._storage_path),
            }

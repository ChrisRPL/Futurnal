"""Base class for all embedders.

Provides common functionality:
- Model access via ModelManager
- L2 normalization
- Result construction
- Timing utilities

All embedders should inherit from BaseEmbedder to ensure consistent
behavior and interface.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from futurnal.embeddings.exceptions import EmbeddingGenerationError
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import EmbeddingEntityType, EmbeddingResult


class BaseEmbedder(ABC):
    """Abstract base class for all embedding generators.

    Provides:
    - Consistent interface across embedder types
    - Common utilities (normalization, timing, result construction)
    - Model access via ModelManager

    Subclasses must implement:
    - embed(): Generate embeddings for specific entity type
    - entity_type: Property returning the entity type this embedder handles
    """

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize the embedder.

        Args:
            model_manager: Manager for loading and accessing embedding models
        """
        self._model_manager = model_manager

    @abstractmethod
    def embed(self, **kwargs: Any) -> EmbeddingResult:
        """Generate embedding for an entity.

        Subclasses implement specific embedding logic based on entity type.

        Returns:
            EmbeddingResult with the generated embedding and metadata
        """
        pass

    @property
    @abstractmethod
    def entity_type(self) -> EmbeddingEntityType:
        """Return the entity type this embedder handles.

        Returns:
            EmbeddingEntityType enum value
        """
        pass

    def _normalize_l2(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize an embedding vector.

        Normalization ensures embeddings have unit length, which:
        - Makes cosine similarity equivalent to dot product
        - Improves numerical stability
        - Enables efficient similarity computations

        Args:
            embedding: Raw embedding vector

        Returns:
            L2 normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _build_result(
        self,
        embedding: np.ndarray,
        model_id: str,
        generation_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        temporal_context_encoded: bool = False,
        causal_context_encoded: bool = False,
    ) -> EmbeddingResult:
        """Build a standardized embedding result.

        Args:
            embedding: The generated embedding vector
            model_id: ID of the model used
            generation_time_ms: Time taken to generate (milliseconds)
            metadata: Additional metadata to include
            temporal_context_encoded: Whether temporal context was encoded
            causal_context_encoded: Whether causal context was encoded

        Returns:
            EmbeddingResult with all fields populated
        """
        return EmbeddingResult(
            embedding=embedding.tolist(),
            entity_type=self.entity_type,
            model_version=self._model_manager.get_model_version(model_id),
            embedding_dimension=len(embedding),
            generation_time_ms=generation_time_ms,
            metadata=metadata or {},
            temporal_context_encoded=temporal_context_encoded,
            causal_context_encoded=causal_context_encoded,
        )

    def _encode_text(
        self,
        text: str,
        model_id: str,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Encode text using specified model.

        Args:
            text: Text to encode
            model_id: ID of the model to use
            instruction: Optional instruction for Instructor models

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingGenerationError: If encoding fails
        """
        try:
            model = self._model_manager.get_model(model_id)

            # Check if model supports instruction parameter
            if instruction is not None and hasattr(model, "encode"):
                # Try to pass instruction (for Instructor wrapper)
                try:
                    embedding = model.encode(text, instruction=instruction)
                except TypeError:
                    # Model doesn't accept instruction parameter
                    embedding = model.encode(text)
            else:
                embedding = model.encode(text)

            return np.array(embedding)

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to encode text with model {model_id}: {e}"
            ) from e

    def _encode_batch(
        self,
        texts: list[str],
        model_id: str,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Encode multiple texts in a batch.

        More efficient than encoding texts individually.

        Args:
            texts: List of texts to encode
            model_id: ID of the model to use
            instruction: Optional instruction for Instructor models

        Returns:
            Array of embedding vectors

        Raises:
            EmbeddingGenerationError: If encoding fails
        """
        if not texts:
            return np.array([])

        try:
            model = self._model_manager.get_model(model_id)

            if instruction is not None and hasattr(model, "encode"):
                try:
                    embeddings = model.encode(texts, instruction=instruction)
                except TypeError:
                    embeddings = model.encode(texts)
            else:
                embeddings = model.encode(texts)

            return np.array(embeddings)

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to encode batch with model {model_id}: {e}"
            ) from e


class TimingContext:
    """Context manager for timing operations.

    Usage:
        with TimingContext() as timer:
            # do work
        print(f"Took {timer.elapsed_ms} ms")
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self._end - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self._end - self._start

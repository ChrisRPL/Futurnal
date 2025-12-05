"""Model Manager for embedding model lifecycle management.

Handles:
- Lazy loading of embedding models
- Model caching and unloading
- Device selection (CPU/GPU)
- Fallback to lightweight models
- Schema version tracking

Option B Compliance:
- Models are FROZEN (no fine-tuning, loaded as-is)
- Schema versioning for re-embedding support
- Memory-efficient lazy loading
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from futurnal.embeddings.config import (
    EmbeddingServiceConfig,
    ModelConfig,
    ModelType,
)
from futurnal.embeddings.exceptions import ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models.

    Defines the interface that all embedding models must support.
    """

    def encode(
        self,
        sentences: Any,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Encode sentences to embeddings."""
        ...


class ModelManager:
    """Manages embedding models with lazy loading and caching.

    Features:
    - Lazy model initialization (load on first use)
    - Memory-efficient model caching
    - Automatic fallback to lightweight models
    - Schema version tracking for re-embedding

    Option B Compliance:
    - Models are FROZEN (loaded pre-trained, no fine-tuning)
    - Supports experiential learning via token priors (not model updates)

    Example:
        config = EmbeddingServiceConfig()
        manager = ModelManager(config)

        # Get model (loads lazily)
        model = manager.get_model("content-instructor")

        # Encode text
        embedding = model.encode("Hello world")
    """

    def __init__(self, config: EmbeddingServiceConfig) -> None:
        """Initialize the ModelManager.

        Args:
            config: Service configuration with model settings
        """
        self._config = config
        self._models: Dict[str, EmbeddingModel] = {}
        self._model_versions: Dict[str, str] = {}
        self._model_configs: Dict[str, ModelConfig] = {}

        # Register configured models
        self._register_model(config.content_model)
        self._register_model(config.temporal_model)

    def _register_model(self, model_config: ModelConfig) -> None:
        """Register a model configuration for lazy loading.

        Args:
            model_config: Configuration for the model
        """
        self._model_configs[model_config.model_id] = model_config
        logger.debug(f"Registered model config: {model_config.model_id}")

    def get_model(self, model_id: str) -> EmbeddingModel:
        """Get or load a model by ID.

        Implements lazy loading - model is only loaded on first access.

        Args:
            model_id: Unique identifier for the model

        Returns:
            The loaded embedding model

        Raises:
            ModelNotFoundError: If model_id is not registered
            ModelLoadError: If model fails to load
        """
        if model_id not in self._model_configs:
            raise ModelNotFoundError(
                f"Model '{model_id}' not registered. "
                f"Available models: {list(self._model_configs.keys())}"
            )

        if model_id not in self._models:
            self._load_model(model_id)

        return self._models[model_id]

    def _load_model(self, model_id: str) -> None:
        """Load a model into memory.

        Tries to load the configured model, with fallback to lightweight
        alternatives if the primary model fails.

        Args:
            model_id: ID of the model to load
        """
        config = self._model_configs[model_id]
        model_type = config.model_type

        logger.info(f"Loading model: {model_id} ({model_type.value})")

        # Try Instructor model first (for instructor model types)
        if model_type in (ModelType.INSTRUCTOR_LARGE, ModelType.INSTRUCTOR_XL):
            if self._try_load_instructor(model_id, config):
                return

            # Fallback to sentence-transformers if configured
            if self._config.fallback_to_lightweight:
                logger.warning(
                    f"Instructor model unavailable, falling back to "
                    f"sentence-transformers for {model_id}"
                )
                self._load_sentence_transformer(model_id, config)
                return

            raise ModelLoadError(
                f"Failed to load Instructor model {model_type.value} "
                "and fallback is disabled"
            )

        # Load sentence-transformer models
        self._load_sentence_transformer(model_id, config)

    def _try_load_instructor(self, model_id: str, config: ModelConfig) -> bool:
        """Try to load an Instructor model.

        Args:
            model_id: ID to register the model under
            config: Model configuration

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            from InstructorEmbedding import INSTRUCTOR

            model = INSTRUCTOR(config.model_type.value, device=config.device)
            self._models[model_id] = _InstructorModelWrapper(
                model, config.instruction
            )
            self._model_versions[model_id] = f"instructor:{config.model_type.value}"
            logger.info(f"Loaded Instructor model: {config.model_type.value}")
            return True

        except ImportError:
            logger.debug("InstructorEmbedding package not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to load Instructor model: {e}")
            return False

    def _load_sentence_transformer(
        self, model_id: str, config: ModelConfig
    ) -> None:
        """Load a SentenceTransformer model.

        Args:
            model_id: ID to register the model under
            config: Model configuration

        Raises:
            ModelLoadError: If model fails to load
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                config.model_type.value,
                device=config.device,
            )
            self._models[model_id] = model
            self._model_versions[model_id] = f"st:{config.model_type.value}"
            logger.info(f"Loaded SentenceTransformer: {config.model_type.value}")

        except ImportError as e:
            raise ModelLoadError(
                "sentence-transformers package not installed. "
                "Please install it with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load SentenceTransformer model "
                f"{config.model_type.value}: {e}"
            ) from e

    def unload_model(self, model_id: str) -> None:
        """Unload a model to free memory.

        Args:
            model_id: ID of the model to unload
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Unloaded model: {model_id}")

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        model_ids = list(self._models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        logger.info("Unloaded all models")

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded.

        Args:
            model_id: ID of the model to check

        Returns:
            True if model is loaded, False otherwise
        """
        return model_id in self._models

    def get_model_version(self, model_id: str) -> str:
        """Get version string for a model.

        Used for schema versioning to track which model generated embeddings.

        Args:
            model_id: ID of the model

        Returns:
            Version string (e.g., "st:all-MiniLM-L6-v2")
        """
        if model_id in self._model_versions:
            return self._model_versions[model_id]

        # Return config info even if not loaded
        if model_id in self._model_configs:
            return f"unloaded:{self._model_configs[model_id].model_type.value}"

        return "unknown"

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model.

        Args:
            model_id: ID of the model

        Returns:
            Model configuration or None if not registered
        """
        return self._model_configs.get(model_id)

    @property
    def schema_version(self) -> str:
        """Current schema version from configuration."""
        return self._config.schema_version

    @property
    def loaded_models(self) -> list[str]:
        """List of currently loaded model IDs."""
        return list(self._models.keys())

    @property
    def registered_models(self) -> list[str]:
        """List of all registered model IDs."""
        return list(self._model_configs.keys())


class _InstructorModelWrapper:
    """Wrapper for Instructor models to match SentenceTransformer interface.

    Instructor models expect input as [[instruction, text]] format.
    This wrapper handles the conversion automatically.
    """

    def __init__(
        self,
        model: Any,
        default_instruction: Optional[str] = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            model: The Instructor model
            default_instruction: Default instruction to use if not provided
        """
        self._model = model
        self._default_instruction = default_instruction or "Represent the text:"

    def encode(
        self,
        sentences: Any,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Encode sentences using Instructor model.

        Args:
            sentences: Text or list of texts to encode
            batch_size: Batch size for inference
            show_progress_bar: Whether to show progress
            instruction: Task-specific instruction (overrides default)
            **kwargs: Additional arguments

        Returns:
            Embedding(s) as numpy array
        """
        instr = instruction or self._default_instruction

        # Handle single string
        if isinstance(sentences, str):
            inputs = [[instr, sentences]]
        else:
            # Handle list of strings
            inputs = [[instr, s] for s in sentences]

        return self._model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

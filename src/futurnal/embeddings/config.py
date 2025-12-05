"""Configuration for the embedding service.

Defines configuration classes for:
- Model selection and parameters
- Storage configuration
- Performance constraints

Supports on-device execution with:
- CPU preference for memory efficiency
- Configurable batch sizes
- Lazy loading for memory management

Option B Compliance:
- Models are FROZEN (no fine-tuning configuration)
- Schema versioning enabled by default
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class ModelType(str, Enum):
    """Available embedding model types.

    Models selected for balance between quality and on-device performance.
    """

    # Task-specific embeddings - SOTA on MTEB leaderboard
    INSTRUCTOR_LARGE = "hkunlp/instructor-large"
    INSTRUCTOR_XL = "hkunlp/instructor-xl"

    # Lightweight general-purpose embeddings
    MINILM_L6_V2 = "all-MiniLM-L6-v2"
    MINILM_L12_V2 = "all-MiniLM-L12-v2"
    MPNET_BASE_V2 = "all-mpnet-base-v2"

    # Code-specific embeddings (future enhancement)
    CODEBERT_BASE = "microsoft/codebert-base"
    CODESAGE_SMALL = "codesage/codesage-small"

    # Lightweight alternatives for resource-constrained environments
    PARAPHRASE_MINILM = "paraphrase-MiniLM-L6-v2"


@dataclass
class ModelConfig:
    """Configuration for a single embedding model.

    Attributes:
        model_id: Unique identifier for this model configuration
        model_type: The model to load (from ModelType enum)
        embedding_dimension: Output dimension of the model
        max_sequence_length: Maximum input length (tokens)
        device: Compute device ("cpu" or "cuda")
        batch_size: Batch size for inference
        cache_embeddings: Whether to cache computed embeddings
        instruction: Task-specific instruction (for Instructor models)
    """

    model_id: str
    model_type: ModelType
    embedding_dimension: int
    max_sequence_length: int = 512
    device: str = "cpu"  # Prefer CPU for on-device execution
    batch_size: int = 32
    cache_embeddings: bool = True
    instruction: Optional[str] = None  # For Instructor models


@dataclass
class EmbeddingServiceConfig:
    """Top-level configuration for the embedding service.

    Provides sensible defaults for on-device execution while allowing
    customization for different deployment scenarios.
    """

    # Model configurations
    content_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model_id="content-instructor",
            model_type=ModelType.INSTRUCTOR_LARGE,
            embedding_dimension=768,
            max_sequence_length=512,
            device="cpu",
            instruction="Represent the text for retrieval:",
        )
    )

    temporal_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model_id="temporal-minilm",
            model_type=ModelType.MINILM_L6_V2,
            embedding_dimension=384,
            max_sequence_length=256,
            device="cpu",
        )
    )

    # Storage configuration
    persist_directory: Path = field(
        default_factory=lambda: Path.home() / ".futurnal" / "embeddings"
    )
    collection_prefix: str = "futurnal"

    # Schema versioning (Option B requirement)
    schema_version: str = "1.0"

    # Default fusion weights
    default_fusion_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "content": 0.6,
            "temporal": 0.3,
            "causal": 0.1,
        }
    )

    # Performance constraints for on-device execution
    max_memory_mb: int = 4096  # 4GB max memory for models
    prefer_cpu: bool = True  # Prefer CPU for on-device execution
    enable_model_caching: bool = True  # Cache loaded models

    # Fallback configuration
    fallback_to_lightweight: bool = True  # Fallback to MiniLM if Instructor fails

    def get_persist_path(self) -> Path:
        """Get or create the persistence directory."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        return self.persist_directory

    def get_collection_name(self, entity_type: str) -> str:
        """Generate collection name for an entity type.

        Args:
            entity_type: Type of entity (events, entities, sequences)

        Returns:
            Collection name like "futurnal-events-v1.0"
        """
        return f"{self.collection_prefix}-{entity_type}-v{self.schema_version}"


# Pre-defined configurations for common scenarios


def get_default_config() -> EmbeddingServiceConfig:
    """Get default configuration for on-device execution."""
    return EmbeddingServiceConfig()


def get_lightweight_config() -> EmbeddingServiceConfig:
    """Get lightweight configuration for resource-constrained environments.

    Uses smaller models for faster inference and lower memory usage.
    """
    return EmbeddingServiceConfig(
        content_model=ModelConfig(
            model_id="content-minilm",
            model_type=ModelType.MINILM_L12_V2,
            embedding_dimension=384,
            max_sequence_length=256,
            device="cpu",
        ),
        temporal_model=ModelConfig(
            model_id="temporal-minilm",
            model_type=ModelType.MINILM_L6_V2,
            embedding_dimension=384,
            max_sequence_length=128,
            device="cpu",
        ),
        max_memory_mb=2048,
    )


def get_high_quality_config() -> EmbeddingServiceConfig:
    """Get high-quality configuration for better embedding quality.

    Uses larger models when GPU is available.
    """
    return EmbeddingServiceConfig(
        content_model=ModelConfig(
            model_id="content-instructor-xl",
            model_type=ModelType.INSTRUCTOR_XL,
            embedding_dimension=768,
            max_sequence_length=512,
            device="cuda",
            instruction="Represent the text for semantic similarity:",
        ),
        temporal_model=ModelConfig(
            model_id="temporal-mpnet",
            model_type=ModelType.MPNET_BASE_V2,
            embedding_dimension=768,
            max_sequence_length=384,
            device="cuda",
        ),
        prefer_cpu=False,
        max_memory_mb=8192,
    )

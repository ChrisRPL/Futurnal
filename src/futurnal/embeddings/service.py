"""Multi-Model Embedding Service.

Main orchestration service for the multi-model embedding architecture.
Provides unified API for embedding generation across all entity types.

Option B Compliance:
- Models are FROZEN (no fine-tuning, pre-trained only)
- Temporal context REQUIRED for events
- Schema version tracked in results
- On-device execution preferred

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/02-multi-model-architecture.md
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from futurnal.embeddings.base import TimingContext
from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.exceptions import (
    BatchProcessingError,
    EmbeddingError,
    EmbeddingGenerationError,
    RoutingError,
)
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.metrics import EmbeddingMetrics
from futurnal.embeddings.models import (
    EmbeddingResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.registry import ModelRegistry
from futurnal.embeddings.request import EmbeddingRequest, BatchEmbeddingRequest
from futurnal.embeddings.router import ModelRouter

logger = logging.getLogger(__name__)


class MultiModelEmbeddingService:
    """Orchestrates embedding generation across multiple specialized models.

    The main entry point for the multi-model embedding architecture.
    Routes requests to appropriate models based on entity type and
    manages the embedding generation pipeline.

    Features:
        - Automatic model routing based on entity type
        - Single and batch embedding APIs
        - Performance metrics tracking
        - Memory management (model loading/unloading)
        - Option B compliance enforcement

    Option B Compliance:
        - Models are FROZEN (no fine-tuning, loaded pre-trained)
        - Temporal context REQUIRED for Event type (validated in EmbeddingRequest)
        - Schema version tracked in all EmbeddingResult instances
        - On-device execution preferred (CPU by default)

    Example:
        # Initialize service
        service = MultiModelEmbeddingService()

        # Single embedding (static entity)
        result = service.embed(
            entity_type="Person",
            content="John Smith, Software Engineer",
        )
        print(f"Embedding dimension: {result.embedding_dimension}")

        # Single embedding (temporal event)
        from datetime import datetime
        result = service.embed(
            entity_type="Event",
            content="Team Meeting: Quarterly planning",
            temporal_context=TemporalEmbeddingContext(
                timestamp=datetime(2024, 1, 15, 14, 30),
            ),
        )

        # Batch embedding
        requests = [
            EmbeddingRequest(entity_type="Person", content="John Smith"),
            EmbeddingRequest(
                entity_type="Event",
                content="Meeting",
                temporal_context=TemporalEmbeddingContext(timestamp=datetime.now()),
            ),
        ]
        results = service.embed_batch(requests)

        # Get metrics
        metrics = service.get_metrics()
        print(f"Total embeddings: {metrics['total_embeddings']}")
    """

    def __init__(
        self,
        config: Optional[EmbeddingServiceConfig] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        """Initialize the multi-model embedding service.

        Args:
            config: Service configuration (uses defaults if None)
            registry: Model registry (uses default registry if None)
        """
        self._config = config or EmbeddingServiceConfig()
        self._registry = registry or ModelRegistry()
        self._model_manager = ModelManager(self._config)
        self._router = ModelRouter(
            registry=self._registry,
            model_manager=self._model_manager,
            config=self._config,
        )
        self._metrics = EmbeddingMetrics()

        logger.info(
            f"Initialized MultiModelEmbeddingService with "
            f"{len(self._registry.registered_models)} registered models, "
            f"supporting {len(self._registry.supported_entity_types)} entity types"
        )

    def embed(
        self,
        entity_type: str,
        content: str,
        temporal_context: Optional[TemporalEmbeddingContext] = None,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Generate embedding for a single entity.

        Automatically routes to appropriate model based on entity type.

        Args:
            entity_type: Type of entity (Person, Organization, Event, etc.)
            content: Content to embed
            temporal_context: Temporal context (REQUIRED for Event type)
            entity_id: Optional PKG entity ID for provenance
            entity_name: Optional entity name for metadata
            metadata: Additional metadata

        Returns:
            EmbeddingResult with embedding vector and metadata

        Raises:
            ValueError: If Event type without temporal_context
            ModelNotFoundError: If no model for entity type
            EmbeddingGenerationError: If embedding fails

        Example:
            # Static entity
            result = service.embed(
                entity_type="Person",
                content="John Smith, Software Engineer",
            )

            # Temporal event (requires temporal_context)
            result = service.embed(
                entity_type="Event",
                content="Team Meeting",
                temporal_context=TemporalEmbeddingContext(
                    timestamp=datetime.now(),
                ),
            )
        """
        # Create validated request (enforces Option B temporal requirement)
        request = EmbeddingRequest(
            entity_type=entity_type,
            content=content,
            entity_id=entity_id,
            entity_name=entity_name,
            temporal_context=temporal_context,
            metadata=metadata or {},
        )

        return self._process_request(request)

    def embed_batch(
        self,
        requests: List[EmbeddingRequest],
        fail_fast: bool = True,
    ) -> List[EmbeddingResult]:
        """Embed multiple entities in a batch.

        Groups requests by entity type for efficient processing.

        Args:
            requests: List of embedding requests
            fail_fast: If True, stop on first error. If False, continue
                and raise BatchProcessingError with partial results.

        Returns:
            List of EmbeddingResults in same order as requests

        Raises:
            BatchProcessingError: If any embedding fails (contains partial results)
            ValueError: If requests list is empty

        Example:
            requests = [
                EmbeddingRequest(entity_type="Person", content="John"),
                EmbeddingRequest(entity_type="Organization", content="Acme Inc"),
            ]
            results = service.embed_batch(requests)
            assert len(results) == 2
        """
        if not requests:
            return []

        # Track results and errors
        results: List[Optional[EmbeddingResult]] = [None] * len(requests)
        errors: List[tuple] = []  # (index, exception)
        failed_indices: List[int] = []

        # Group by entity type for potential batching optimization
        batch_request = BatchEmbeddingRequest(requests=requests, fail_fast=fail_fast)
        grouped = batch_request.group_by_entity_type()

        self._metrics.record_batch_operation(len(requests))

        # Process each group
        for entity_type, group in grouped.items():
            for request in group:
                # Find original index
                idx = requests.index(request)

                try:
                    result = self._process_request(request)
                    results[idx] = result
                except EmbeddingError as e:
                    logger.error(f"Failed to embed request {idx}: {e}")
                    errors.append((idx, e))
                    failed_indices.append(idx)

                    # Record error metric
                    model_id = "unknown"
                    try:
                        model_id = self._registry.get_model_for_entity_type(
                            entity_type
                        ).model_id
                    except Exception:
                        pass

                    self._metrics.record_embedding(
                        model_id=model_id,
                        entity_type=entity_type,
                        latency_ms=0,
                        vector_dimension=0,
                        success=False,
                    )

                    if fail_fast:
                        raise BatchProcessingError(
                            f"Batch processing failed at index {idx}: {e}",
                            successful_results=[r for r in results if r is not None],
                            failed_indices=failed_indices,
                            errors=errors,
                        ) from e

        # Check for failures in non-fail-fast mode
        if errors:
            raise BatchProcessingError(
                f"Batch processing completed with {len(errors)} errors",
                successful_results=[r for r in results if r is not None],
                failed_indices=failed_indices,
                errors=errors,
            )

        return [r for r in results if r is not None]

    def _process_request(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Process a single embedding request.

        Internal method that handles routing and embedding generation.

        Args:
            request: Validated embedding request

        Returns:
            EmbeddingResult with embedding and metadata
        """
        with TimingContext() as timer:
            # Route to model and embedder
            model_id, embedder = self._router.route_request(request)

            # Get registered model info for metadata
            registered_model = self._registry.get_model_by_id(model_id)

            # Generate embedding based on entity type
            try:
                if request.entity_type == "Event":
                    # Use temporal event embedder
                    result = embedder.embed(
                        event_name=request.get_effective_name(),
                        event_description=request.content,
                        temporal_context=request.temporal_context,
                    )
                else:
                    # Use static entity embedder
                    result = embedder.embed(
                        entity_type=request.entity_type,
                        entity_name=request.get_effective_name(),
                        entity_description=request.content,
                        properties=request.metadata,
                    )
            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Failed to generate embedding for {request.entity_type}: {e}"
                ) from e

        # Record metrics
        self._metrics.record_embedding(
            model_id=model_id,
            entity_type=request.entity_type,
            latency_ms=timer.elapsed_ms,
            vector_dimension=result.embedding_dimension,
            success=True,
        )

        logger.debug(
            f"Generated embedding for {request.entity_type} "
            f"using {model_id} in {timer.elapsed_ms:.2f}ms "
            f"(dim={result.embedding_dimension})"
        )

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics summary.

        Returns:
            Dictionary with comprehensive metrics including:
            - total_embeddings: Total successful embeddings
            - total_errors: Total errors
            - error_rate: Overall error rate
            - entity_type_distribution: Counts per entity type
            - model_metrics: Per-model statistics
            - uptime_seconds: Service uptime
        """
        return self._metrics.get_summary()

    def get_latency_summary(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics per model.

        Returns:
            Dict mapping model_id to latency stats (avg, p50, p95, p99)
        """
        return self._metrics.get_latency_summary()

    def unload_all_models(self) -> None:
        """Unload all models to free memory.

        Useful for resource management when embedding generation
        is paused or system resources are needed elsewhere.
        """
        self._model_manager.unload_all()
        logger.info("Unloaded all models from memory")

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded.

        Args:
            model_id: ID of the model to check

        Returns:
            True if model is in memory, False otherwise
        """
        return self._model_manager.is_loaded(model_id)

    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types.

        Returns:
            Sorted list of entity type strings
        """
        return self._router.get_supported_entity_types()

    def get_model_for_entity_type(self, entity_type: str) -> Optional[str]:
        """Get model ID for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            Model ID if found, None otherwise
        """
        model = self._registry.get_model_for_entity_type(entity_type)
        return model.model_id if model else None

    @property
    def registry(self) -> ModelRegistry:
        """Access the model registry."""
        return self._registry

    @property
    def router(self) -> ModelRouter:
        """Access the model router."""
        return self._router

    @property
    def model_manager(self) -> ModelManager:
        """Access the model manager."""
        return self._model_manager

    @property
    def schema_version(self) -> str:
        """Current schema version from configuration."""
        return self._config.schema_version

    @property
    def config(self) -> EmbeddingServiceConfig:
        """Access service configuration."""
        return self._config

    def close(self) -> None:
        """Clean up resources.

        Unloads all models and performs cleanup.
        Should be called when the service is no longer needed.
        """
        self.unload_all_models()
        logger.info("Closed MultiModelEmbeddingService")

    def __enter__(self) -> "MultiModelEmbeddingService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        metrics = self._metrics.get_summary()
        return (
            f"MultiModelEmbeddingService("
            f"models={len(self._registry.registered_models)}, "
            f"entity_types={len(self._registry.supported_entity_types)}, "
            f"embeddings={metrics['total_embeddings']})"
        )

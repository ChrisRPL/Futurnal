Summary: Implement multi-model architecture with specialized embedding models for different entity types and routing logic.

# 02 · Multi-Model Architecture

## Purpose
Design and implement a multi-model embedding architecture that selects and routes to specialized models based on entity type, optimizing embedding quality for static entities, temporal events, code, and documents.

**Criticality**: HIGH - Ensures optimal embedding quality across diverse entity types

## Scope
- Model selection strategy based on entity type
- Specialized models for entities, events, code, documents
- Model routing and orchestration logic
- On-device execution optimization
- Model quantization and performance tuning

## Requirements Alignment
- **Option B Requirement**: "Specialized embeddings for different knowledge types"
- **Performance Target**: <2s embedding latency on consumer hardware
- **Privacy Guarantee**: On-device by default, cloud escalation requires consent
- **Enables**: High-quality embeddings for all entity types

## Component Design

### Model Registry

```python
from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional, List


class EmbeddingModel(BaseModel):
    """Embedding model configuration."""
    model_id: str
    model_type: str  # "instructor", "sentence-transformer", "code-bert"
    model_path: str  # Local path or HuggingFace ID
    entity_types: List[str]  # Entity types this model handles
    vector_dimension: int
    max_sequence_length: int
    quantized: bool = False
    memory_mb: int  # Expected memory usage
    avg_latency_ms: float  # Benchmarked latency


class ModelRegistry:
    """Registry of available embedding models."""

    def __init__(self):
        self.models: Dict[str, EmbeddingModel] = {}
        self._load_default_models()

    def _load_default_models(self):
        """Load default model configurations."""
        # Entity embedding model
        self.register_model(EmbeddingModel(
            model_id="instructor-large-entity",
            model_type="instructor",
            model_path="hkunlp/instructor-large",
            entity_types=["Person", "Organization", "Concept"],
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,  # Quantized for on-device
            memory_mb=800,
            avg_latency_ms=150
        ))

        # Event embedding model (temporal-aware)
        self.register_model(EmbeddingModel(
            model_id="instructor-temporal-event",
            model_type="instructor",
            model_path="hkunlp/instructor-large",  # Same base, different prompt
            entity_types=["Event"],
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,
            memory_mb=800,
            avg_latency_ms=150
        ))

        # Code embedding model
        self.register_model(EmbeddingModel(
            model_id="codebert-code",
            model_type="code-bert",
            model_path="microsoft/codebert-base",
            entity_types=["CodeEntity"],
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,
            memory_mb=600,
            avg_latency_ms=120
        ))

        # Document embedding model
        self.register_model(EmbeddingModel(
            model_id="instructor-document",
            model_type="instructor",
            model_path="hkunlp/instructor-large",
            entity_types=["Document"],
            vector_dimension=768,
            max_sequence_length=2048,  # Longer context
            quantized=True,
            memory_mb=1200,
            avg_latency_ms=300
        ))

    def register_model(self, model: EmbeddingModel):
        """Register a new model."""
        self.models[model.model_id] = model

    def get_model_for_entity_type(
        self,
        entity_type: str
    ) -> Optional[EmbeddingModel]:
        """Get best model for entity type."""
        for model in self.models.values():
            if entity_type in model.entity_types:
                return model
        return None
```

### Model Router

```python
class ModelRouter:
    """
    Routes embedding requests to appropriate models.

    Strategy:
    - Entity type determines model selection
    - Resource constraints influence quantization
    - Performance targets guide batching
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.loaded_models: Dict[str, Any] = {}

    def route_embedding_request(
        self,
        entity_type: str,
        content: str,
        temporal_context: Optional[TemporalEmbeddingContext] = None
    ) -> str:
        """
        Route embedding request to appropriate model.

        Returns: model_id to use for embedding
        """
        # Get model for entity type
        model = self.registry.get_model_for_entity_type(entity_type)

        if model is None:
            raise ValueError(f"No model registered for entity type: {entity_type}")

        return model.model_id

    def get_model_instance(self, model_id: str):
        """
        Get or load model instance.

        Implements lazy loading and caching.
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        # Load model
        model_config = self.registry.models[model_id]
        model_instance = self._load_model(model_config)

        # Cache
        self.loaded_models[model_id] = model_instance

        return model_instance

    def _load_model(self, config: EmbeddingModel):
        """Load model from configuration."""
        if config.model_type == "instructor":
            from InstructorEmbedding import INSTRUCTOR
            model = INSTRUCTOR(config.model_path)

        elif config.model_type == "code-bert":
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(config.model_path)
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            return {"model": model, "tokenizer": tokenizer}

        elif config.model_type == "sentence-transformer":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.model_path)

        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        # Quantize if configured
        if config.quantized:
            model = self._quantize_model(model)

        return model

    def _quantize_model(self, model):
        """
        Apply quantization for on-device efficiency.

        Reduces memory footprint and improves latency.
        """
        # PyTorch dynamic quantization
        import torch
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized
```

### Multi-Model Embedding Service

```python
class MultiModelEmbeddingService:
    """
    Orchestrates embedding generation across multiple models.

    Responsibilities:
    - Route requests to appropriate models
    - Batch requests for efficiency
    - Handle model loading and unloading
    - Track performance metrics
    """

    def __init__(self, registry: ModelRegistry, router: ModelRouter):
        self.registry = registry
        self.router = router
        self.metrics = EmbeddingMetrics()

    def embed(
        self,
        entity_type: str,
        content: str,
        temporal_context: Optional[TemporalEmbeddingContext] = None,
        metadata: Optional[Dict] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for entity.

        Automatically routes to appropriate model.
        """
        import time

        start = time.time()

        # Route to model
        model_id = self.router.route_embedding_request(
            entity_type,
            content,
            temporal_context
        )

        # Get model instance
        model = self.router.get_model_instance(model_id)

        # Generate embedding based on entity type
        if entity_type == "Event":
            # Use temporal-aware embedding
            embedding = self._embed_event(
                model,
                content,
                temporal_context
            )
        elif entity_type in ["Person", "Organization", "Concept"]:
            # Use static entity embedding
            embedding = self._embed_entity(model, content)
        elif entity_type == "CodeEntity":
            # Use code embedding
            embedding = self._embed_code(model, content)
        elif entity_type == "Document":
            # Use document embedding
            embedding = self._embed_document(model, content)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

        elapsed = time.time() - start

        # Track metrics
        self.metrics.record_embedding(
            model_id=model_id,
            entity_type=entity_type,
            latency_ms=elapsed * 1000,
            vector_dimension=len(embedding)
        )

        return EmbeddingResult(
            embedding=embedding,
            model_id=model_id,
            entity_type=entity_type,
            latency_ms=elapsed * 1000,
            metadata=metadata or {}
        )

    def embed_batch(
        self,
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResult]:
        """
        Batch embed multiple entities.

        Groups by model for efficiency.
        """
        # Group requests by model
        grouped = {}
        for req in requests:
            model_id = self.router.route_embedding_request(
                req.entity_type,
                req.content,
                req.temporal_context
            )
            if model_id not in grouped:
                grouped[model_id] = []
            grouped[model_id].append(req)

        # Process each group
        results = []
        for model_id, group in grouped.items():
            model = self.router.get_model_instance(model_id)

            # Batch encode
            contents = [req.content for req in group]
            embeddings = model.encode(contents, batch_size=32)

            # Create results
            for req, emb in zip(group, embeddings):
                results.append(EmbeddingResult(
                    embedding=emb,
                    model_id=model_id,
                    entity_type=req.entity_type,
                    latency_ms=0,  # Not tracked in batch
                    metadata=req.metadata or {}
                ))

        return results

    def _embed_event(
        self,
        model,
        content: str,
        temporal_context: TemporalEmbeddingContext
    ) -> np.ndarray:
        """Embed temporal event with context."""
        # Use TemporalEventEmbedder from module 01
        from temporal_aware_embeddings import TemporalEventEmbedder

        embedder = TemporalEventEmbedder(model, temporal_encoder=None)
        return embedder.embed_event(
            event_name=content.split(":")[0],
            event_description=content,
            temporal_context=temporal_context
        )

    def _embed_entity(self, model, content: str) -> np.ndarray:
        """Embed static entity."""
        instruction = "Represent the knowledge entity for retrieval:"
        return model.encode([[instruction, content]])[0]

    def _embed_code(self, model, content: str) -> np.ndarray:
        """Embed code entity."""
        tokenizer = model["tokenizer"]
        model_instance = model["model"]

        tokens = tokenizer(content, return_tensors="pt", truncation=True)
        output = model_instance(**tokens)

        # Use mean pooling
        embedding = output.last_hidden_state.mean(dim=1).detach().numpy()[0]
        return embedding

    def _embed_document(self, model, content: str) -> np.ndarray:
        """Embed full document."""
        instruction = "Represent the document for retrieval:"
        return model.encode([[instruction, content]])[0]
```

## Implementation Details

### Week 2: Model Selection & Benchmarking

**Deliverable**: Model registry with benchmarked models

1. **Evaluate embedding models**:
   - Instructor-large for entities and events
   - CodeBERT for code
   - Sentence-BERT for documents
   - Benchmark on reference hardware

2. **Quantization strategy**:
   - PyTorch dynamic quantization
   - Memory footprint reduction
   - Latency vs quality trade-offs

3. **Model registry implementation**:
   - Model configuration management
   - Lazy loading strategy
   - Resource constraint handling

### Week 3: Multi-Model Orchestration

**Deliverable**: Working multi-model embedding service

1. **Implement `ModelRouter`**:
   - Entity type to model mapping
   - Dynamic model loading
   - Performance monitoring

2. **Implement `MultiModelEmbeddingService`**:
   - Single and batch embedding APIs
   - Model-specific embedding strategies
   - Metrics tracking

3. **Performance optimization**:
   - Batch processing
   - Model caching
   - GPU utilization (if available)

## Testing Strategy

```python
class TestMultiModelArchitecture:
    def test_model_routing(self):
        """Validate model routing by entity type."""
        registry = ModelRegistry()
        router = ModelRouter(registry)

        # Entity should route to entity model
        entity_model = router.route_embedding_request("Person", "John Doe")
        assert entity_model == "instructor-large-entity"

        # Event should route to event model
        event_model = router.route_embedding_request("Event", "Meeting")
        assert event_model == "instructor-temporal-event"

        # Code should route to code model
        code_model = router.route_embedding_request("CodeEntity", "def foo():")
        assert code_model == "codebert-code"

    def test_batch_processing(self):
        """Validate batch embedding efficiency."""
        service = MultiModelEmbeddingService(registry, router)

        requests = [
            EmbeddingRequest("Person", "John Doe"),
            EmbeddingRequest("Person", "Jane Smith"),
            EmbeddingRequest("Event", "Meeting"),
        ]

        results = service.embed_batch(requests)

        assert len(results) == 3
        assert all(r.embedding is not None for r in results)

    def test_latency_targets(self):
        """Validate embedding latency meets targets."""
        service = MultiModelEmbeddingService(registry, router)

        result = service.embed("Person", "John Doe")

        # <2s target for single embedding
        assert result.latency_ms < 2000
```

## Success Metrics

- ✅ Model routing functional for all entity types
- ✅ Embedding latency <2s for single requests
- ✅ Batch processing >100 entities/minute
- ✅ Memory footprint <2GB with all models loaded
- ✅ On-device execution for all models

## Dependencies

- Temporal-aware embeddings (01-temporal-aware-embeddings.md)
- Base models (Instructor, CodeBERT)
- PyTorch/TensorFlow runtime
- Quantization libraries

## Next Steps

After multi-model architecture complete:
1. Integrate with schema-versioned storage (03-schema-versioned-storage.md)
2. Connect to PKG synchronization (04-pkg-synchronization.md)
3. Enable quality evolution tracking (05-quality-evolution.md)

**This module ensures optimal embedding quality across all entity types with on-device performance.**

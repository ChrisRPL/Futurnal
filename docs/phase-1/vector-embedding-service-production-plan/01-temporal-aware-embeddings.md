Summary: Implement temporal-aware embedding strategies that preserve temporal context for events and support causal pattern matching.

# 01 · Temporal-Aware Embedding Strategy

## Purpose
Design and implement embedding strategies that preserve temporal semantics for events, distinguish temporal from static entities, and optimize embeddings for Phase 2 correlation detection and Phase 3 causal inference.

**Criticality**: CRITICAL - Foundation for temporal search and causal pattern matching

## Scope
- Event vs entity embedding differentiation
- Temporal context preservation techniques
- Event sequence embeddings for causal patterns
- Temporal semantic encoding
- Integration with temporal extraction pipeline

## Requirements Alignment
- **Option B Requirement**: "Embeddings must preserve temporal context for correlation detection"
- **Phase 2 Foundation**: Event embeddings optimized for correlation pattern matching
- **Phase 3 Foundation**: Causal relationship embeddings for causal inference
- **Enables**: Temporal search, correlation detection, causal hypothesis exploration

## Component Design

### Temporal Entity Types

```python
from enum import Enum
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List


class EmbeddingEntityType(str, Enum):
    """Types of entities requiring different embedding strategies."""
    STATIC_ENTITY = "static_entity"      # Person, Organization, Concept
    TEMPORAL_EVENT = "temporal_event"     # Event with timestamp
    TEMPORAL_RELATIONSHIP = "temporal_relationship"  # BEFORE/AFTER/CAUSES
    CODE_ENTITY = "code_entity"          # Code snippets
    DOCUMENT = "document"                # Full documents


class TemporalEmbeddingContext(BaseModel):
    """Temporal context for event embeddings."""
    timestamp: datetime
    duration: Optional[timedelta] = None
    temporal_type: Optional[str] = None  # BEFORE/AFTER/DURING/CAUSES
    event_sequence: List[str] = []       # Sequence of related events
    causal_chain: List[str] = []         # Causal predecessors/successors
```

### Event Embedding Strategy

```python
class TemporalEventEmbedder:
    """
    Specialized embedder for temporal events.

    Key difference from static entity embeddings:
    - Incorporates temporal context into embedding
    - Optimized for temporal similarity and causal pattern matching
    """

    def __init__(self, base_model, temporal_encoder):
        self.base_model = base_model  # e.g., Instructor-large
        self.temporal_encoder = temporal_encoder

    def embed_event(
        self,
        event_name: str,
        event_description: str,
        temporal_context: TemporalEmbeddingContext
    ) -> np.ndarray:
        """
        Generate embedding for temporal event.

        Strategy:
        1. Encode event content (what happened)
        2. Encode temporal context (when it happened)
        3. Encode causal context (what led to it/what it led to)
        4. Combine with weighted fusion
        """
        # Content embedding
        content = f"{event_name}: {event_description}"
        content_embedding = self.base_model.encode(content)

        # Temporal context embedding
        temporal_text = self._format_temporal_context(temporal_context)
        temporal_embedding = self.temporal_encoder.encode(temporal_text)

        # Causal context embedding (if available)
        causal_embedding = None
        if temporal_context.causal_chain:
            causal_text = self._format_causal_chain(temporal_context.causal_chain)
            causal_embedding = self.base_model.encode(causal_text)

        # Weighted fusion
        combined = self._fuse_embeddings(
            content_embedding,
            temporal_embedding,
            causal_embedding,
            weights=[0.6, 0.3, 0.1]  # Tunable
        )

        return combined

    def _format_temporal_context(
        self,
        context: TemporalEmbeddingContext
    ) -> str:
        """
        Format temporal context for embedding.

        Example: "Event occurred on 2024-01-15, lasted 2 hours,
                  happened BEFORE Meeting on 2024-01-16"
        """
        parts = [f"occurred on {context.timestamp.isoformat()}"]

        if context.duration:
            parts.append(f"lasted {context.duration}")

        if context.temporal_type:
            parts.append(f"temporal relationship: {context.temporal_type}")

        return ", ".join(parts)

    def _format_causal_chain(self, causal_chain: List[str]) -> str:
        """Format causal chain for embedding."""
        if not causal_chain:
            return ""

        return f"causal context: {' → '.join(causal_chain)}"

    def _fuse_embeddings(
        self,
        content: np.ndarray,
        temporal: np.ndarray,
        causal: Optional[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Fuse multiple embeddings with weights.

        This ensures temporal and causal context influence
        the final embedding for better pattern matching.
        """
        if causal is None:
            # Only content + temporal
            combined = (
                weights[0] * content +
                weights[1] * temporal
            )
        else:
            combined = (
                weights[0] * content +
                weights[1] * temporal +
                weights[2] * causal
            )

        # Normalize
        return combined / np.linalg.norm(combined)
```

### Event Sequence Embeddings

```python
class EventSequenceEmbedder:
    """
    Embed sequences of events for causal pattern matching.

    Critical for Phase 2 correlation detection.
    """

    def __init__(self, event_embedder, sequence_model):
        self.event_embedder = event_embedder
        self.sequence_model = sequence_model  # e.g., LSTM or Transformer

    def embed_event_sequence(
        self,
        events: List[Event],
        temporal_contexts: List[TemporalEmbeddingContext]
    ) -> np.ndarray:
        """
        Generate embedding for sequence of events.

        Preserves:
        - Temporal ordering
        - Causal relationships
        - Event semantics

        Optimized for:
        - Correlation detection (Phase 2)
        - Causal pattern matching (Phase 3)
        """
        # Embed each event with temporal context
        event_embeddings = []
        for event, context in zip(events, temporal_contexts):
            embedding = self.event_embedder.embed_event(
                event.name,
                event.description,
                context
            )
            event_embeddings.append(embedding)

        # Encode sequence with temporal ordering
        sequence_embedding = self.sequence_model.encode_sequence(
            event_embeddings,
            preserve_order=True
        )

        return sequence_embedding

    def find_similar_sequences(
        self,
        query_sequence: List[Event],
        embedding_store,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find similar event sequences.

        Use case:
        - "Find patterns similar to: Meeting → Decision → Publication"
        - Supports correlation detection in Phase 2
        """
        query_embedding = self.embed_event_sequence(query_sequence, [])

        similar = embedding_store.search_sequences(
            query_embedding,
            top_k=top_k
        )

        return similar
```

### Static Entity Embeddings

```python
class StaticEntityEmbedder:
    """
    Standard embedder for static entities (Person, Organization, Concept).

    Different from event embeddings:
    - No temporal context
    - Focused on semantic content
    - Optimized for entity linking and disambiguation
    """

    def __init__(self, base_model):
        self.base_model = base_model  # e.g., Instructor-large

    def embed_entity(
        self,
        entity_type: str,
        entity_name: str,
        entity_description: str,
        properties: dict
    ) -> np.ndarray:
        """Generate embedding for static entity."""
        # Format entity content
        content = self._format_entity_content(
            entity_type,
            entity_name,
            entity_description,
            properties
        )

        # Generate embedding
        embedding = self.base_model.encode(content)

        return embedding

    def _format_entity_content(
        self,
        entity_type: str,
        name: str,
        description: str,
        properties: dict
    ) -> str:
        """Format entity for embedding."""
        parts = [f"{entity_type}: {name}"]

        if description:
            parts.append(description)

        # Add key properties
        for key, value in properties.items():
            if key not in ["id", "created_at", "updated_at"]:
                parts.append(f"{key}: {value}")

        return ". ".join(parts)
```

## Implementation Details

### Week 1: Temporal Embedding Strategy Design

**Deliverable**: Temporal-aware embedding architecture

1. **Design temporal context encoding**:
   - Timestamp encoding schemes (absolute vs relative)
   - Duration encoding
   - Temporal relationship encoding (BEFORE/AFTER/DURING)

2. **Design embedding fusion strategy**:
   - Weighted combination of content + temporal + causal
   - Normalization strategies
   - Similarity metrics for temporal embeddings

3. **Research temporal embedding models**:
   - Survey SOTA temporal embedding approaches
   - Evaluate models for on-device execution
   - Benchmark temporal similarity accuracy

### Week 2: Event Embedding Implementation

**Deliverable**: Working event embedder with temporal context

1. **Implement `TemporalEventEmbedder`**:
   - Content embedding generation
   - Temporal context encoding
   - Causal chain encoding
   - Weighted fusion

2. **Implement `EventSequenceEmbedder`**:
   - Sequence embedding generation
   - Temporal ordering preservation
   - Causal pattern optimization

3. **Implement `StaticEntityEmbedder`**:
   - Standard entity embedding
   - Entity linking optimization

## Testing Strategy

```python
class TestTemporalEmbeddings:
    def test_temporal_similarity(self):
        """
        Validate temporal embeddings preserve temporal semantics.

        Events closer in time should have higher similarity.
        """
        event1 = create_event("Meeting", timestamp=datetime(2024, 1, 1))
        event2 = create_event("Meeting", timestamp=datetime(2024, 1, 2))
        event3 = create_event("Meeting", timestamp=datetime(2024, 6, 1))

        embedder = TemporalEventEmbedder(base_model, temporal_encoder)

        emb1 = embedder.embed_event(event1.name, event1.description,
                                    create_temporal_context(event1))
        emb2 = embedder.embed_event(event2.name, event2.description,
                                    create_temporal_context(event2))
        emb3 = embedder.embed_event(event3.name, event3.description,
                                    create_temporal_context(event3))

        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)

        # Events 1 day apart should be more similar than 5 months apart
        assert sim_12 > sim_13

    def test_causal_pattern_matching(self):
        """Validate event sequences support causal pattern matching."""
        sequence1 = [
            create_event("Meeting", timestamp=datetime(2024, 1, 1)),
            create_event("Decision", timestamp=datetime(2024, 1, 2)),
            create_event("Publication", timestamp=datetime(2024, 1, 3))
        ]

        sequence2 = [
            create_event("Meeting", timestamp=datetime(2024, 2, 1)),
            create_event("Decision", timestamp=datetime(2024, 2, 2)),
            create_event("Publication", timestamp=datetime(2024, 2, 3))
        ]

        sequence_embedder = EventSequenceEmbedder(event_embedder, sequence_model)

        emb1 = sequence_embedder.embed_event_sequence(sequence1, [])
        emb2 = sequence_embedder.embed_event_sequence(sequence2, [])

        similarity = cosine_similarity(emb1, emb2)

        # Similar causal patterns should have high similarity
        assert similarity > 0.8

    def test_event_vs_entity_distinction(self):
        """Validate events and entities have different embeddings."""
        event = create_event("Meeting with John", timestamp=datetime(2024, 1, 1))
        entity = create_person_entity("John", "Software Engineer")

        event_embedder = TemporalEventEmbedder(base_model, temporal_encoder)
        entity_embedder = StaticEntityEmbedder(base_model)

        event_emb = event_embedder.embed_event(
            event.name,
            event.description,
            create_temporal_context(event)
        )

        entity_emb = entity_embedder.embed_entity(
            "Person",
            entity.name,
            entity.description,
            {}
        )

        similarity = cosine_similarity(event_emb, entity_emb)

        # Events and entities should be distinguishable
        assert similarity < 0.6
```

## Success Metrics

- ✅ Temporal similarity accuracy >80% (events closer in time more similar)
- ✅ Causal pattern matching >75% accuracy
- ✅ Event vs entity distinction clear (similarity <0.6)
- ✅ Event sequence embeddings preserve temporal ordering
- ✅ Integration with temporal extraction pipeline complete

## Dependencies

- Temporal extraction pipeline (01-temporal-extraction.md from entity-relationship extraction)
- PKG temporal metadata
- Base embedding models (Instructor-large, CodeBERT)
- Temporal encoding models

## Next Steps

After temporal-aware embeddings complete:
1. Integrate with multi-model architecture (02-multi-model-architecture.md)
2. Connect to schema-versioned storage (03-schema-versioned-storage.md)
3. Enable PKG synchronization (04-pkg-synchronization.md)

**This module enables Phase 2 correlation detection and Phase 3 causal inference through temporal-aware embeddings.**

"""Static Entity Embedder for Person, Organization, Concept entities.

Generates standard semantic embeddings for entities without temporal context.
These entities represent persistent concepts rather than time-bound events.

Key differences from temporal event embeddings:
- No temporal context encoding
- Focused purely on semantic content
- Optimized for entity linking and disambiguation

Use cases:
- Person entities (people mentioned in documents)
- Organization entities (companies, institutions)
- Concept entities (topics, themes, ideas)

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from futurnal.embeddings.base import BaseEmbedder, TimingContext
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import EmbeddingEntityType, EmbeddingResult


class StaticEntityEmbedder(BaseEmbedder):
    """Embedder for static entities without temporal context.

    Used for embedding persistent entities that don't have temporal grounding:
    - Person: Named individuals
    - Organization: Companies, institutions, groups
    - Concept: Abstract ideas, topics, themes

    These embeddings focus purely on semantic content to enable:
    - Entity linking (connecting mentions to known entities)
    - Disambiguation (distinguishing "Apple" company from "apple" fruit)
    - Similarity search (finding related entities)

    Example:
        manager = ModelManager(EmbeddingServiceConfig())
        embedder = StaticEntityEmbedder(manager)

        result = embedder.embed(
            entity_type="Person",
            entity_name="John Smith",
            entity_description="Software Engineer at Futurnal",
            properties={"role": "Lead Developer", "team": "Backend"},
        )
    """

    DEFAULT_CONTENT_MODEL = "content-instructor"

    def __init__(
        self,
        model_manager: ModelManager,
        content_model_id: Optional[str] = None,
    ) -> None:
        """Initialize the static entity embedder.

        Args:
            model_manager: Manager for loading embedding models
            content_model_id: ID of content embedding model (default: content-instructor)
        """
        super().__init__(model_manager)
        self._content_model_id = content_model_id or self.DEFAULT_CONTENT_MODEL

    @property
    def entity_type(self) -> EmbeddingEntityType:
        """Return entity type for static entities."""
        return EmbeddingEntityType.STATIC_ENTITY

    def embed(
        self,
        entity_type: str,
        entity_name: str,
        entity_description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Generate embedding for a static entity.

        Creates a semantic embedding based on entity type, name, description,
        and relevant properties.

        Args:
            entity_type: Type of entity (Person, Organization, Concept)
            entity_name: Name of the entity
            entity_description: Optional description
            properties: Additional properties to include in embedding

        Returns:
            EmbeddingResult with semantic embedding
        """
        with TimingContext() as timer:
            # Format entity content
            content = self._format_entity_content(
                entity_type,
                entity_name,
                entity_description,
                properties or {},
            )

            # Generate embedding with appropriate instruction
            instruction = self._get_instruction_for_type(entity_type)
            embedding = self._encode_text(
                content,
                self._content_model_id,
                instruction=instruction,
            )

            # L2 normalize
            normalized = self._normalize_l2(embedding)

        return self._build_result(
            embedding=normalized,
            model_id=self._content_model_id,
            generation_time_ms=timer.elapsed_ms,
            metadata={
                "entity_type": entity_type,
                "entity_name": entity_name,
                "has_description": bool(entity_description),
                "property_count": len(properties) if properties else 0,
            },
            temporal_context_encoded=False,
            causal_context_encoded=False,
        )

    def embed_batch(
        self,
        entities: list[Dict[str, Any]],
    ) -> list[EmbeddingResult]:
        """Embed multiple entities in a batch.

        More efficient than embedding entities individually.

        Args:
            entities: List of entity dicts with keys:
                - entity_type: str
                - entity_name: str
                - entity_description: str (optional)
                - properties: dict (optional)

        Returns:
            List of EmbeddingResults
        """
        if not entities:
            return []

        with TimingContext() as timer:
            # Format all entity content
            contents = [
                self._format_entity_content(
                    e.get("entity_type", "Entity"),
                    e.get("entity_name", ""),
                    e.get("entity_description", ""),
                    e.get("properties", {}),
                )
                for e in entities
            ]

            # Batch encode
            embeddings = self._encode_batch(
                contents,
                self._content_model_id,
                instruction="Represent the entity for retrieval:",
            )

        # Build results
        avg_time = timer.elapsed_ms / len(entities)
        results = []
        for i, (entity, embedding) in enumerate(zip(entities, embeddings)):
            normalized = self._normalize_l2(embedding)
            result = self._build_result(
                embedding=normalized,
                model_id=self._content_model_id,
                generation_time_ms=avg_time,
                metadata={
                    "entity_type": entity.get("entity_type", "Entity"),
                    "entity_name": entity.get("entity_name", ""),
                    "batch_index": i,
                },
                temporal_context_encoded=False,
                causal_context_encoded=False,
            )
            results.append(result)

        return results

    def _format_entity_content(
        self,
        entity_type: str,
        name: str,
        description: str,
        properties: Dict[str, Any],
    ) -> str:
        """Format entity information for embedding.

        Creates a coherent text representation including type, name,
        description, and relevant properties.

        Args:
            entity_type: Type of entity
            name: Entity name
            description: Entity description
            properties: Additional properties

        Returns:
            Formatted content string
        """
        parts = [f"{entity_type}: {name}"]

        if description and description.strip():
            parts.append(description.strip())

        # Add relevant properties (exclude metadata fields)
        excluded_keys = {"id", "created_at", "updated_at", "embedding", "embedding_id"}
        for key, value in properties.items():
            if key.lower() not in excluded_keys and value is not None:
                parts.append(f"{key}: {value}")

        return ". ".join(parts)

    def _get_instruction_for_type(self, entity_type: str) -> str:
        """Get embedding instruction appropriate for entity type.

        Different entity types benefit from different instructions.

        Args:
            entity_type: Type of entity

        Returns:
            Instruction string for embedding model
        """
        type_lower = entity_type.lower()

        if type_lower == "person":
            return "Represent this person for entity linking:"
        elif type_lower == "organization":
            return "Represent this organization for entity linking:"
        elif type_lower == "concept":
            return "Represent this concept for semantic similarity:"
        else:
            return "Represent this entity for retrieval:"

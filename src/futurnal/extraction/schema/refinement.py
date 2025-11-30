"""
Schema Refinement Engine

Quality assessment and schema refinement logic based on
extraction performance and reflection mechanisms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from futurnal.extraction.schema.models import EntityType, RelationshipType

if TYPE_CHECKING:
    from futurnal.extraction.schema.evolution import ExtractionResult
    from futurnal.extraction.schema.models import SchemaDiscovery


class ReflectionTrigger:
    """Determine when reflection should occur based on various conditions."""

    def __init__(
        self,
        interval: int = 100,
        low_success_threshold: float = 0.7,
        high_novel_threshold: float = 0.1,
    ):
        """
        Initialize reflection trigger.
        
        Args:
            interval: Number of documents between automatic reflections
            low_success_threshold: Success rate below which to trigger reflection
            high_novel_threshold: Novel pattern rate above which to trigger reflection
        """
        self.interval = interval
        self.low_success_threshold = low_success_threshold
        self.high_novel_threshold = high_novel_threshold

    def should_reflect(
        self,
        documents_processed: int,
        recent_results: List[ExtractionResult],
    ) -> bool:
        """
        Trigger reflection if:
        1. Interval reached (every N documents)
        2. Success rate drops below threshold
        3. Many novel patterns discovered
        
        Args:
            documents_processed: Total documents processed
            recent_results: Recent extraction results
            
        Returns:
            bool: True if reflection should be triggered
        """
        # Trigger at regular intervals
        if documents_processed % self.interval == 0:
            return True

        # Need sufficient samples
        if len(recent_results) < 20:
            return False

        # Trigger if success rate is low
        success_rate = (
            sum(1 for r in recent_results if r.success) / len(recent_results)
        )
        if success_rate < self.low_success_threshold:
            return True

        # Trigger if many novel patterns
        novel_rate = (
            sum(1 for r in recent_results if r.has_novel_pattern)
            / len(recent_results)
        )
        if novel_rate > self.high_novel_threshold:
            return True

        return False


class SchemaRefinementEngine:
    """Refine schema based on reflection and quality assessment."""

    def refine_entity_types(
        self,
        current_types: Dict[str, EntityType],
        discoveries: List[SchemaDiscovery],
        min_discovery_count: int = 10,
    ) -> Dict[str, EntityType]:
        """
        Refine entity types:
        - Add frequently discovered new types
        - Remove rarely seen types
        - Update confidence scores
        
        Args:
            current_types: Current entity types
            discoveries: Discovered schema elements
            min_discovery_count: Minimum discoveries to add a new type
            
        Returns:
            Dict[str, EntityType]: Refined entity types
        """
        refined = current_types.copy()
        newly_added = set()

        # Add new types with sufficient evidence
        for discovery in discoveries:
            if (
                discovery.element_type == "entity"
                and len(discovery.examples) >= min_discovery_count
            ):
                refined[discovery.name] = EntityType(
                    name=discovery.name,
                    description=discovery.description,
                    examples=discovery.examples,
                    confidence=discovery.confidence,
                    properties={},
                    aliases=[],
                )
                newly_added.add(discovery.name)

        # Remove types with low discovery count
        # Keep: 1) seed types (confidence 1.0), 2) types with sufficient discoveries, 3) newly added types
        refined = {
            name: entity
            for name, entity in refined.items()
            if entity.confidence == 1.0  # Seed types
            or entity.discovery_count >= min_discovery_count  # Established types
            or name in newly_added  # Newly discovered types being added now
        }

        return refined

    def refine_relationship_types(
        self,
        current_types: Dict[str, RelationshipType],
        discoveries: List[SchemaDiscovery],
        min_discovery_count: int = 5,
    ) -> Dict[str, RelationshipType]:
        """
        Refine relationship types:
        - Add frequently discovered new types
        - Remove rarely seen types
        - Update confidence scores
        
        Args:
            current_types: Current relationship types
            discoveries: Discovered schema elements
            min_discovery_count: Minimum discoveries to add a new type
            
        Returns:
            Dict[str, RelationshipType]: Refined relationship types
        """
        refined = current_types.copy()

        # Add new types with sufficient evidence
        # Note: Discovery would need to include subject/object types
        # This is a simplified placeholder

        # Remove types with low discovery count
        # (Keep seed types with confidence 1.0)
        refined = {
            name: rel
            for name, rel in refined.items()
            if rel.discovery_count >= min_discovery_count or rel.confidence == 1.0
        }

        return refined

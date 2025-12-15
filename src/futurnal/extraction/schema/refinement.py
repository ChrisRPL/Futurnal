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
        - Validate subject/object type compatibility

        Args:
            current_types: Current relationship types
            discoveries: Discovered schema elements
            min_discovery_count: Minimum discoveries to add a new type

        Returns:
            Dict[str, RelationshipType]: Refined relationship types
        """
        refined = current_types.copy()
        newly_added = set()

        # Add new types with sufficient evidence
        for discovery in discoveries:
            if (
                discovery.element_type == "relationship"
                and len(discovery.examples) >= min_discovery_count
            ):
                rel_name = discovery.name

                # Parse subject/object types from description if available
                # Description format: "Relationship 'verb' between [types] and [types]"
                subject_types = self._parse_types_from_description(
                    discovery.description, "subject"
                )
                object_types = self._parse_types_from_description(
                    discovery.description, "object"
                )

                # Infer temporal and causal properties
                is_temporal = self._infer_temporal(discovery)
                is_causal = self._infer_causal(discovery)

                refined[rel_name] = RelationshipType(
                    name=rel_name,
                    description=discovery.description,
                    examples=discovery.examples,
                    confidence=discovery.confidence,
                    subject_types=subject_types or ["Entity"],
                    object_types=object_types or ["Entity"],
                    temporal=is_temporal,
                    causal=is_causal,
                    properties={},
                    discovery_count=len(discovery.examples),
                )
                newly_added.add(rel_name)

        # Remove types with low discovery count
        # Keep: 1) seed types (confidence 1.0), 2) types with sufficient discoveries, 3) newly added types
        refined = {
            name: rel
            for name, rel in refined.items()
            if rel.confidence == 1.0  # Seed types
            or rel.discovery_count >= min_discovery_count  # Established types
            or name in newly_added  # Newly discovered types being added now
        }

        return refined

    def _parse_types_from_description(
        self, description: str, type_position: str
    ) -> List[str]:
        """
        Parse entity types from discovery description.

        Args:
            description: Discovery description
            type_position: "subject" or "object"

        Returns:
            List of entity type names
        """
        # Description format: "Relationship 'verb' between [types] and [types]"
        # or: "Relationship 'verb' between ['Type1', 'Type2'] and ['Type3']"
        import re

        try:
            # Look for "between X and Y" pattern
            match = re.search(r"between\s+(\[.+?\])\s+and\s+(\[.+?\])", description)
            if match:
                if type_position == "subject":
                    types_str = match.group(1)
                else:
                    types_str = match.group(2)

                # Parse the list-like string
                # Remove brackets and quotes, split by comma
                types_str = types_str.strip("[]")
                types = [t.strip().strip("'\"") for t in types_str.split(",")]
                return [t for t in types if t]  # Filter empty strings
        except Exception:
            pass

        return []

    def _infer_temporal(self, discovery: SchemaDiscovery) -> bool:
        """
        Infer if relationship has temporal grounding.

        Args:
            discovery: Schema discovery

        Returns:
            True if relationship appears to be temporal
        """
        temporal_keywords = [
            "when", "during", "after", "before", "while",
            "timestamp", "temporal", "time", "event",
        ]
        combined = f"{discovery.description} {' '.join(discovery.examples)}".lower()
        return any(kw in combined for kw in temporal_keywords)

    def _infer_causal(self, discovery: SchemaDiscovery) -> bool:
        """
        Infer if relationship is causal.

        Args:
            discovery: Schema discovery

        Returns:
            True if relationship appears to be causal
        """
        causal_keywords = [
            "caused", "led to", "resulted", "enabled", "triggered",
            "prevented", "because", "therefore", "consequently",
            "causal", "effect", "impact",
        ]
        combined = f"{discovery.description} {' '.join(discovery.examples)}".lower()
        return any(kw in combined for kw in causal_keywords)

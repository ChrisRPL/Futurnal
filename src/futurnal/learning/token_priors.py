"""
Token Prior Store for Experiential Learning

Implements Training-Free GRPO experiential knowledge storage as token priors
(natural language patterns) that enhance extraction prompts without model updates.

Research Foundation:
- Training-Free GRPO (2510.08191v1): Experiential knowledge as token priors
  instead of parameter updates
- TOTAL (2510.07499v1): Thought templates with textual gradients

CRITICAL CONSTRAINT: All knowledge stored as TEXT (natural language), NOT
numerical weights. Ghost model MUST remain frozen.

Option B Compliance:
- Ghost model frozen (verified by test)
- Experiential knowledge as token priors (natural language strings)
- No gradient computation or parameter updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.extraction.schema.models import ThoughtTemplate
    from futurnal.learning.context_gate import SemanticContextGate

logger = logging.getLogger(__name__)


# Defaults
DEFAULT_PRIOR_CAPACITY = 100
MIN_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class EntityTypePrior:
    """Frequency-based prior for entity types.

    Stored as TEXT (natural language), NOT numerical weights.
    This is how experiential knowledge guides extraction without model updates.

    Example context_pattern:
        "Person entities appear frequently in personal notes, often as
        proper nouns followed by action verbs"

    Research Reference:
    - Training-Free GRPO (2510.08191v1) Section 2: "Experiential knowledge
      serves as the learned token prior"
    """

    entity_type: str
    frequency: int = 0
    confidence: float = 0.5
    context_pattern: str = ""  # Natural language description
    examples: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_success(self) -> None:
        """Record successful use of this prior."""
        self.success_count += 1
        self.frequency += 1
        self._update_confidence()
        self.updated_at = datetime.utcnow()

    def update_failure(self) -> None:
        """Record failed use of this prior."""
        self.failure_count += 1
        self._update_confidence()
        self.updated_at = datetime.utcnow()

    def _update_confidence(self) -> None:
        """Update confidence based on success/failure ratio."""
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = self.success_count / total

    def to_natural_language(self) -> str:
        """Convert to natural language for prompt injection."""
        examples_str = ", ".join(self.examples[:3]) if self.examples else "none yet"
        return (
            f"- {self.entity_type} entities (frequency: {self.frequency}, "
            f"confidence: {self.confidence:.0%})\n"
            f"  Pattern: {self.context_pattern or 'No pattern identified yet'}\n"
            f"  Examples: {examples_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "entity_type": self.entity_type,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "context_pattern": self.context_pattern,
            "examples": self.examples,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class RelationTypePrior:
    """Frequency-based prior for relationship types.

    Captures patterns about how entities relate to each other, stored as
    natural language to guide extraction without model updates.

    Research Reference:
    - Training-Free GRPO (2510.08191v1): Token priors for relationship extraction
    """

    relation_type: str
    frequency: int = 0
    confidence: float = 0.5
    subject_types: List[str] = field(default_factory=list)
    object_types: List[str] = field(default_factory=list)
    context_pattern: str = ""  # Natural language description
    examples: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_success(self) -> None:
        """Record successful use of this prior."""
        self.success_count += 1
        self.frequency += 1
        self._update_confidence()
        self.updated_at = datetime.utcnow()

    def update_failure(self) -> None:
        """Record failed use of this prior."""
        self.failure_count += 1
        self._update_confidence()
        self.updated_at = datetime.utcnow()

    def _update_confidence(self) -> None:
        """Update confidence based on success/failure ratio."""
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = self.success_count / total

    def to_natural_language(self) -> str:
        """Convert to natural language for prompt injection."""
        subjects = ", ".join(self.subject_types[:3]) if self.subject_types else "various"
        objects = ", ".join(self.object_types[:3]) if self.object_types else "various"
        examples_str = "; ".join(self.examples[:2]) if self.examples else "none yet"
        return (
            f"- {self.relation_type}: {subjects} -> {objects} "
            f"(confidence: {self.confidence:.0%})\n"
            f"  Pattern: {self.context_pattern or 'No pattern identified yet'}\n"
            f"  Examples: {examples_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "relation_type": self.relation_type,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "subject_types": self.subject_types,
            "object_types": self.object_types,
            "context_pattern": self.context_pattern,
            "examples": self.examples,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class TemporalPatternPrior:
    """Frequency-based prior for temporal patterns.

    CRITICAL for Phase 3 causal inference. Captures temporal expression
    patterns that are common in the user's knowledge base.

    Pattern types:
    - "explicit_date": YYYY-MM-DD, January 15, 2024
    - "relative_time": yesterday, last week, 3 days ago
    - "duration": for 2 hours, during the meeting
    - "causal_sequence": because, led to, resulted in

    Research Reference:
    - Training-Free GRPO (2510.08191v1): Token priors for temporal extraction
    - Time-R1 (2505.13508v2): Temporal reasoning patterns
    """

    pattern_type: str  # "explicit_date", "relative_time", "duration", "causal_sequence"
    frequency: int = 0
    confidence: float = 0.5
    extraction_guidance: str = ""  # Natural language extraction guidance
    examples: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_success(self) -> None:
        """Record successful use of this prior."""
        self.success_count += 1
        self.frequency += 1
        self._update_confidence()
        self.updated_at = datetime.utcnow()

    def update_failure(self) -> None:
        """Record failed use of this prior."""
        self.failure_count += 1
        self._update_confidence()
        self.updated_at = datetime.utcnow()

    def _update_confidence(self) -> None:
        """Update confidence based on success/failure ratio."""
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = self.success_count / total

    def to_natural_language(self) -> str:
        """Convert to natural language for prompt injection."""
        examples_str = ", ".join(self.examples[:3]) if self.examples else "none yet"
        return (
            f"- {self.pattern_type} (frequency: {self.frequency}, "
            f"confidence: {self.confidence:.0%})\n"
            f"  Guidance: {self.extraction_guidance or 'No guidance yet'}\n"
            f"  Examples: {examples_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "extraction_guidance": self.extraction_guidance,
            "examples": self.examples,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class TokenPriorStore:
    """Training-Free GRPO experiential knowledge storage.

    CRITICAL CONSTRAINT: All knowledge stored as TEXT (natural language),
    NOT numerical weights. Ghost model MUST remain frozen.

    This class is the core of experiential learning - it stores patterns
    learned from successful extractions and injects them as token priors
    into prompts, guiding the Ghost model without parameter updates.

    Research Reference:
    - Training-Free GRPO (2510.08191v1) Section 2: "Experiential knowledge
      serves as the learned token prior"

    Example:
        >>> store = TokenPriorStore()
        >>> store.update_from_extraction(result, success=True)
        >>> context = store.generate_prompt_context()
        >>> enhanced_prompt = f"{context}\\n\\n{base_prompt}"

    Option B Compliance:
    - Ghost model frozen (no parameter updates)
    - All priors are natural language strings
    - No gradient computation
    """

    def __init__(
        self,
        capacity: int = DEFAULT_PRIOR_CAPACITY,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        context_gate: Optional["SemanticContextGate"] = None,
    ):
        """Initialize Token Prior Store.

        Args:
            capacity: Maximum priors to store per category
            min_confidence: Minimum confidence for priors to be included in prompts
            context_gate: Optional SemanticContextGate for query-aware filtering
                         (AGI Phase 2 enhancement)
        """
        self.capacity = capacity
        self.min_confidence = min_confidence
        self._context_gate = context_gate

        # Prior storage (keyed by type/pattern name)
        self.entity_priors: Dict[str, EntityTypePrior] = {}
        self.relation_priors: Dict[str, RelationTypePrior] = {}
        self.temporal_priors: Dict[str, TemporalPatternPrior] = {}

        # Metadata
        self.total_updates = 0
        self.created_at = datetime.utcnow()

    def update_from_extraction(
        self,
        extraction_result: Any,
        success: bool,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        temporal_patterns: Optional[List[str]] = None,
    ) -> None:
        """Update priors based on extraction result.

        Per Training-Free GRPO: Distill semantic advantages into knowledge
        stored as natural language token priors.

        Args:
            extraction_result: Result from extraction (can have entities, relations)
            success: Whether this extraction was successful
            entity_types: Explicit entity types to update (if not in result)
            relation_types: Explicit relation types to update
            temporal_patterns: Explicit temporal patterns to update
        """
        self.total_updates += 1

        # Extract types from result if not provided
        if entity_types is None:
            entity_types = self._extract_entity_types(extraction_result)
        if relation_types is None:
            relation_types = self._extract_relation_types(extraction_result)
        if temporal_patterns is None:
            temporal_patterns = self._extract_temporal_patterns(extraction_result)

        # Update entity priors
        for entity_type in entity_types:
            self._update_entity_prior(entity_type, success)

        # Update relation priors
        for relation_type in relation_types:
            self._update_relation_prior(relation_type, success)

        # Update temporal priors
        for pattern in temporal_patterns:
            self._update_temporal_prior(pattern, success)

        # Prune if over capacity
        self._prune_to_capacity()

        logger.debug(
            f"Updated priors (success={success}): "
            f"{len(entity_types)} entity, {len(relation_types)} relation, "
            f"{len(temporal_patterns)} temporal"
        )

    def _extract_entity_types(self, result: Any) -> List[str]:
        """Extract entity types from extraction result."""
        if result is None:
            return []

        # Try different attribute names
        for attr in ["entities", "entity_types", "discovered_entities"]:
            entities = getattr(result, attr, None)
            if entities:
                if isinstance(entities, dict):
                    return list(entities.keys())
                elif isinstance(entities, list):
                    return [
                        getattr(e, "type", getattr(e, "entity_type", str(e)))
                        for e in entities
                    ]
        return []

    def _extract_relation_types(self, result: Any) -> List[str]:
        """Extract relation types from extraction result."""
        if result is None:
            return []

        for attr in ["relations", "relationships", "relationship_types"]:
            relations = getattr(result, attr, None)
            if relations:
                if isinstance(relations, dict):
                    return list(relations.keys())
                elif isinstance(relations, list):
                    return [
                        getattr(r, "type", getattr(r, "relation_type", str(r)))
                        for r in relations
                    ]
        return []

    def _extract_temporal_patterns(self, result: Any) -> List[str]:
        """Extract temporal pattern types from extraction result.

        Enhanced to detect various temporal patterns including:
        - Temporal markers (explicit_date, relative_time, etc.)
        - Temporal relations (BEFORE, AFTER, CAUSES, etc.)
        - Events with timestamps
        - Causal relationships
        """
        if result is None:
            return []

        patterns = []

        # Check for temporal markers
        markers = getattr(result, "temporal_markers", [])
        for marker in markers:
            marker_type = getattr(marker, "temporal_type", None)
            if marker_type:
                patterns.append(str(marker_type.value if hasattr(marker_type, "value") else marker_type))

        # Check for temporal relations (BEFORE, AFTER, CAUSES, etc.)
        temporal_relations = getattr(result, "temporal_relations", [])
        if not temporal_relations:
            # Try alternate attribute names
            temporal_relations = getattr(result, "relationships", [])
            temporal_relations = [
                r for r in temporal_relations
                if getattr(r, "relation_type", "").upper() in
                ("BEFORE", "AFTER", "DURING", "CAUSES", "ENABLES", "PREVENTS", "TRIGGERS", "TEMPORAL")
            ]

        for rel in temporal_relations:
            rel_type = getattr(rel, "relation_type", getattr(rel, "type", None))
            if rel_type:
                rel_type_str = str(rel_type.value if hasattr(rel_type, "value") else rel_type)
                # Map to causal_sequence pattern type for causal relations
                if rel_type_str.upper() in ("CAUSES", "ENABLES", "PREVENTS", "TRIGGERS"):
                    patterns.append("causal_sequence")
                else:
                    patterns.append(f"temporal_{rel_type_str.lower()}")

        # Check for events with timestamps
        events = getattr(result, "events", [])
        if events:
            patterns.append("event_extraction")
            # Check if events have explicit timestamps
            for event in events:
                timestamp = getattr(event, "timestamp", getattr(event, "event_timestamp", None))
                if timestamp:
                    patterns.append("explicit_date")
                    break

        # Check for causal candidates (from extraction pipeline)
        causal_candidates = getattr(result, "causal_candidates", [])
        if causal_candidates:
            patterns.append("causal_sequence")

        # Deduplicate patterns
        return list(set(patterns))

    def _update_entity_prior(self, entity_type: str, success: bool) -> None:
        """Update or create entity type prior."""
        if entity_type not in self.entity_priors:
            self.entity_priors[entity_type] = EntityTypePrior(
                entity_type=entity_type,
                context_pattern=f"{entity_type} entities appear in this knowledge base",
            )

        prior = self.entity_priors[entity_type]
        if success:
            prior.update_success()
        else:
            prior.update_failure()

    def _update_relation_prior(self, relation_type: str, success: bool) -> None:
        """Update or create relation type prior."""
        if relation_type not in self.relation_priors:
            self.relation_priors[relation_type] = RelationTypePrior(
                relation_type=relation_type,
                context_pattern=f"{relation_type} relationships connect entities",
            )

        prior = self.relation_priors[relation_type]
        if success:
            prior.update_success()
        else:
            prior.update_failure()

    def _update_temporal_prior(self, pattern_type: str, success: bool) -> None:
        """Update or create temporal pattern prior."""
        if pattern_type not in self.temporal_priors:
            guidance = self._get_default_temporal_guidance(pattern_type)
            self.temporal_priors[pattern_type] = TemporalPatternPrior(
                pattern_type=pattern_type,
                extraction_guidance=guidance,
            )

        prior = self.temporal_priors[pattern_type]
        if success:
            prior.update_success()
        else:
            prior.update_failure()

    def _get_default_temporal_guidance(self, pattern_type: str) -> str:
        """Get default extraction guidance for temporal pattern type."""
        guidance_map = {
            "explicit_date": "Look for dates in YYYY-MM-DD or month-day-year formats",
            "relative_time": "Look for relative expressions like 'yesterday', 'last week', 'ago'",
            "duration": "Look for duration expressions like 'for 2 hours', 'during the meeting'",
            "causal_sequence": "Look for causal markers like 'because', 'led to', 'resulted in'",
            "explicit": "Look for explicit timestamp expressions",
            "inferred": "Infer timestamps from context and surrounding text",
            "event_extraction": "Extract events with their temporal grounding",
        }
        return guidance_map.get(pattern_type, f"Extract {pattern_type} temporal patterns")

    def _prune_to_capacity(self) -> None:
        """Prune priors to stay within capacity limits."""
        for prior_dict in [self.entity_priors, self.relation_priors, self.temporal_priors]:
            if len(prior_dict) > self.capacity:
                # Sort by confidence and keep top capacity
                sorted_items = sorted(
                    prior_dict.items(),
                    key=lambda x: x[1].confidence,
                    reverse=True,
                )
                # Keep only top capacity items
                prior_dict.clear()
                for key, value in sorted_items[:self.capacity]:
                    prior_dict[key] = value

    def generate_prompt_context(
        self,
        query: Optional[str] = None,
        document_type: Optional[str] = None,
        target_entity_types: Optional[List[str]] = None,
        include_temporal: bool = True,
        max_priors: int = 10,
    ) -> str:
        """Generate natural language context for prompt injection.

        This is the TOKEN PRIOR - injected into prompt to guide extraction.
        The Ghost model uses this context without any parameter changes.

        AGI Phase 2 Enhancement:
        - When query is provided and context_gate is configured, only
          priors relevant to the query are included
        - Prevents irrelevant priors from polluting context

        Args:
            query: Optional user query for relevance filtering (Phase 2)
            document_type: Optional filter by document type
            target_entity_types: Optional specific entity types to include
            include_temporal: Whether to include temporal priors
            max_priors: Maximum priors per category to include

        Returns:
            Natural language context string for prompt injection

        Example output:
            "## Learned Patterns

            When extracting from personal notes:
            - Person entities appear frequently (85% confidence)
            - Date expressions often use relative formats
            - Causal relationships follow 'because', 'led to' patterns"
        """
        sections = []

        # Header
        sections.append("## Learned Patterns from Experience\n")
        sections.append("The following patterns have been learned from previous extractions:\n")

        # Use context gate for query-aware filtering if available (AGI Phase 2)
        if query and self._context_gate:
            return self._generate_query_aware_context(
                query=query,
                include_temporal=include_temporal,
                max_priors=max_priors,
            )

        # Standard filtering (no query or no context gate)
        # Entity type priors
        entity_priors = self._get_high_confidence_priors(
            self.entity_priors,
            target_types=target_entity_types,
            max_count=max_priors,
        )
        if entity_priors:
            sections.append("\n### Entity Types")
            for prior in entity_priors:
                sections.append(prior.to_natural_language())

        # Relation type priors
        relation_priors = self._get_high_confidence_priors(
            self.relation_priors,
            max_count=max_priors,
        )
        if relation_priors:
            sections.append("\n### Relationship Types")
            for prior in relation_priors:
                sections.append(prior.to_natural_language())

        # Temporal pattern priors
        if include_temporal:
            temporal_priors = self._get_high_confidence_priors(
                self.temporal_priors,
                max_count=max_priors,
            )
            if temporal_priors:
                sections.append("\n### Temporal Patterns")
                for prior in temporal_priors:
                    sections.append(prior.to_natural_language())

        # Add note about experience-based guidance
        if len(sections) > 2:  # Header + note = 2
            sections.append(
                "\n### Extraction Guidance\n"
                "Use these learned patterns to guide your extraction. "
                "Prioritize patterns with higher confidence scores."
            )

        return "\n".join(sections)

    def _generate_query_aware_context(
        self,
        query: str,
        include_temporal: bool = True,
        max_priors: int = 10,
    ) -> str:
        """Generate query-aware context using SemanticContextGate.

        AGI Phase 2: Filter priors by relevance to user query.

        Args:
            query: User's natural language query
            include_temporal: Whether to include temporal priors
            max_priors: Maximum priors per category

        Returns:
            Natural language context with only relevant priors
        """
        sections = []
        sections.append("## Relevant Patterns for Your Query\n")
        sections.append(f"Based on your query, these learned patterns are most relevant:\n")

        # Filter all categories
        filtered = self._context_gate.filter_all_prior_categories(
            query=query,
            entity_priors=self.entity_priors,
            relation_priors=self.relation_priors,
            temporal_priors=self.temporal_priors if include_temporal else {},
            top_k_per_category=max_priors,
        )

        # Entity priors
        if filtered.get("entity"):
            sections.append("\n### Entity Types (Relevant to Query)")
            for prior, relevance in filtered["entity"]:
                sections.append(f"{prior.to_natural_language()} [relevance: {relevance:.0%}]")

        # Relation priors
        if filtered.get("relation"):
            sections.append("\n### Relationship Types (Relevant to Query)")
            for prior, relevance in filtered["relation"]:
                sections.append(f"{prior.to_natural_language()} [relevance: {relevance:.0%}]")

        # Temporal priors
        if include_temporal and filtered.get("temporal"):
            sections.append("\n### Temporal Patterns (Relevant to Query)")
            for prior, relevance in filtered["temporal"]:
                sections.append(f"{prior.to_natural_language()} [relevance: {relevance:.0%}]")

        # Total priors included
        total_included = sum(len(v) for v in filtered.values())
        total_available = len(self.entity_priors) + len(self.relation_priors)
        if include_temporal:
            total_available += len(self.temporal_priors)

        if total_included > 0:
            sections.append(
                f"\n### Context Gate Applied\n"
                f"Included {total_included} of {total_available} available priors "
                f"based on query relevance."
            )
        else:
            sections.append(
                "\n### No Highly Relevant Priors\n"
                "No priors passed the relevance threshold for this query. "
                "Consider exploring related topics to build relevant experience."
            )

        logger.debug(
            f"Query-aware context: {total_included}/{total_available} priors included"
        )

        return "\n".join(sections)

    def _get_high_confidence_priors(
        self,
        prior_dict: Dict[str, Any],
        target_types: Optional[List[str]] = None,
        max_count: int = 10,
    ) -> List[Any]:
        """Get high-confidence priors sorted by confidence."""
        priors = list(prior_dict.values())

        # Filter by target types if specified
        if target_types:
            priors = [
                p for p in priors
                if getattr(p, "entity_type", getattr(p, "relation_type", getattr(p, "pattern_type", "")))
                in target_types
            ]

        # Filter by minimum confidence
        priors = [p for p in priors if p.confidence >= self.min_confidence]

        # Sort by confidence and return top N
        return sorted(priors, key=lambda p: p.confidence, reverse=True)[:max_count]

    def integrate_with_thought_template(
        self,
        template: ThoughtTemplate,
    ) -> str:
        """Combine token priors with thought template.

        Per TOTAL: Templates + Experience = Enhanced reasoning.

        Args:
            template: ThoughtTemplate to combine with priors

        Returns:
            Combined prompt context with template and priors
        """
        sections = []

        # Add template reasoning scaffold
        sections.append("## Reasoning Template\n")
        sections.append(f"### {template.name}\n")
        sections.append(template.pattern)
        sections.append("")

        # Add experiential knowledge
        sections.append(self.generate_prompt_context())

        return "\n".join(sections)

    def prune_low_confidence_priors(
        self,
        min_confidence: Optional[float] = None,
    ) -> int:
        """Remove priors below confidence threshold.

        Args:
            min_confidence: Minimum confidence to keep (defaults to self.min_confidence)

        Returns:
            Number of priors removed
        """
        threshold = min_confidence or self.min_confidence
        removed = 0

        for prior_dict in [self.entity_priors, self.relation_priors, self.temporal_priors]:
            to_remove = [
                key for key, prior in prior_dict.items()
                if prior.confidence < threshold
            ]
            for key in to_remove:
                del prior_dict[key]
                removed += 1

        logger.info(f"Pruned {removed} low-confidence priors (threshold: {threshold})")
        return removed

    def add_example(
        self,
        prior_type: str,
        prior_name: str,
        example: str,
    ) -> bool:
        """Add an example to a specific prior.

        Args:
            prior_type: "entity", "relation", or "temporal"
            prior_name: Name of the prior (entity type, relation type, pattern type)
            example: Example string to add

        Returns:
            True if example was added, False if prior not found
        """
        prior_dict = {
            "entity": self.entity_priors,
            "relation": self.relation_priors,
            "temporal": self.temporal_priors,
        }.get(prior_type)

        if prior_dict is None or prior_name not in prior_dict:
            return False

        prior = prior_dict[prior_name]
        if example not in prior.examples:
            prior.examples.append(example)
            prior.examples = prior.examples[:10]  # Keep max 10 examples
            prior.updated_at = datetime.utcnow()
        return True

    def update_context_pattern(
        self,
        prior_type: str,
        prior_name: str,
        pattern: str,
    ) -> bool:
        """Update the context pattern for a specific prior.

        Args:
            prior_type: "entity", "relation", or "temporal"
            prior_name: Name of the prior
            pattern: New natural language pattern

        Returns:
            True if pattern was updated, False if prior not found
        """
        prior_dict = {
            "entity": self.entity_priors,
            "relation": self.relation_priors,
            "temporal": self.temporal_priors,
        }.get(prior_type)

        if prior_dict is None or prior_name not in prior_dict:
            return False

        prior = prior_dict[prior_name]
        if hasattr(prior, "context_pattern"):
            prior.context_pattern = pattern
        elif hasattr(prior, "extraction_guidance"):
            prior.extraction_guidance = pattern
        prior.updated_at = datetime.utcnow()
        return True

    def export_as_natural_language(self) -> str:
        """Export all priors as natural language document.

        CRITICAL: This is how knowledge transfers - as text, not weights.

        Returns:
            Complete natural language document of all learned patterns
        """
        sections = [
            "# Experiential Knowledge Export",
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Total updates: {self.total_updates}",
            "",
        ]

        # Entity priors
        sections.append("## Entity Type Patterns")
        sections.append(f"Total: {len(self.entity_priors)} patterns\n")
        for prior in sorted(self.entity_priors.values(), key=lambda p: -p.confidence):
            sections.append(prior.to_natural_language())
            sections.append("")

        # Relation priors
        sections.append("\n## Relationship Patterns")
        sections.append(f"Total: {len(self.relation_priors)} patterns\n")
        for prior in sorted(self.relation_priors.values(), key=lambda p: -p.confidence):
            sections.append(prior.to_natural_language())
            sections.append("")

        # Temporal priors
        sections.append("\n## Temporal Patterns")
        sections.append(f"Total: {len(self.temporal_priors)} patterns\n")
        for prior in sorted(self.temporal_priors.values(), key=lambda p: -p.confidence):
            sections.append(prior.to_natural_language())
            sections.append("")

        return "\n".join(sections)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stored priors.

        Returns:
            Dict with prior statistics
        """
        return {
            "entity_prior_count": len(self.entity_priors),
            "relation_prior_count": len(self.relation_priors),
            "temporal_prior_count": len(self.temporal_priors),
            "total_priors": (
                len(self.entity_priors) +
                len(self.relation_priors) +
                len(self.temporal_priors)
            ),
            "total_updates": self.total_updates,
            "capacity": self.capacity,
            "min_confidence": self.min_confidence,
            "high_confidence_entity": sum(
                1 for p in self.entity_priors.values()
                if p.confidence >= self.min_confidence
            ),
            "high_confidence_relation": sum(
                1 for p in self.relation_priors.values()
                if p.confidence >= self.min_confidence
            ),
            "high_confidence_temporal": sum(
                1 for p in self.temporal_priors.values()
                if p.confidence >= self.min_confidence
            ),
            "created_at": self.created_at.isoformat(),
        }

    def clear(self) -> None:
        """Clear all stored priors."""
        self.entity_priors.clear()
        self.relation_priors.clear()
        self.temporal_priors.clear()
        self.total_updates = 0
        logger.info("Cleared all token priors")

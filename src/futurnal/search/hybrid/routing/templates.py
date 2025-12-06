"""Query Understanding Templates.

Provides thought templates for query understanding that evolve via
textual gradients (TOTAL framework).

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Template Types:
- Temporal Query Template: Analyzing time-based queries
- Causal Query Template: Analyzing causation queries
- Lookup Query Template: Analyzing entity/fact lookup queries
- Exploratory Query Template: Analyzing broad exploration queries

Integration:
- Uses patterns from extraction/schema/templates.py
- Connects to TemplateRefinementEngine for evolution
- Templates evolve based on search quality feedback

Option B Compliance:
- Templates are token priors, not model parameters
- Evolution via textual gradients, not fine-tuning
- Templates improve through use and feedback
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional

from pydantic import BaseModel, Field

from futurnal.search.hybrid.types import QueryIntent

if TYPE_CHECKING:
    from futurnal.extraction.schema.templates import TemplateDatabase

logger = logging.getLogger(__name__)


class QueryTemplate(BaseModel):
    """Thought template for query understanding.

    Extends the pattern from extraction/schema/templates.py
    to the query understanding domain.

    Attributes:
        template_id: Unique identifier
        name: Human-readable name
        intent_type: Which query intent this template serves
        pattern: The template pattern/instructions
        version: Version number for evolution tracking
        success_rate: Measured success rate (0-1)
        created_at: When template was created
        last_updated: When template was last modified
    """

    template_id: str = Field(description="Unique template identifier")
    name: str = Field(description="Human-readable name")
    intent_type: QueryIntent = Field(description="Target query intent")
    pattern: str = Field(description="Template pattern/instructions")
    version: int = Field(default=1, description="Version number")
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Measured success rate",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp",
    )


# Seed templates for each intent type
SEED_TEMPLATES = {
    QueryIntent.TEMPORAL: QueryTemplate(
        template_id="temporal_query_v1",
        name="Temporal Query Understanding",
        intent_type=QueryIntent.TEMPORAL,
        pattern="""# Temporal Query Analysis

1. Identify time references (dates, relative expressions)
   - Explicit dates: "January 2024", "2024-01-15"
   - Relative: "last week", "before the meeting"
   - Ranges: "between X and Y"

2. Determine time scope
   - Point in time: specific moment
   - Range: period between two points
   - Relative: before/after reference event

3. Extract temporal relationships to search for
   - BEFORE/AFTER relationships
   - DURING relationships
   - Temporal proximity

4. Consider temporal ordering requirements
   - Chronological vs reverse chronological
   - Event sequences""",
        version=1,
    ),
    QueryIntent.CAUSAL: QueryTemplate(
        template_id="causal_query_v1",
        name="Causal Query Understanding",
        intent_type=QueryIntent.CAUSAL,
        pattern="""# Causal Query Analysis

1. Identify the effect/outcome being questioned
   - What happened? (the effect)
   - What is the user trying to understand?

2. Determine causal depth
   - Direct cause: immediate predecessor
   - Causal chain: multiple steps back
   - Root cause: ultimate origin

3. Consider temporal precedence requirements
   - Cause must precede effect
   - Time window for causation

4. Look for intervention/counterfactual framing
   - "What if X hadn't happened?"
   - "How could we prevent Y?"

5. Bradford Hill criteria preparation
   - Strength of association
   - Consistency
   - Temporality""",
        version=1,
    ),
    QueryIntent.LOOKUP: QueryTemplate(
        template_id="lookup_query_v1",
        name="Lookup Query Understanding",
        intent_type=QueryIntent.LOOKUP,
        pattern="""# Lookup Query Analysis

1. Identify the target entity or fact
   - Named entity: person, organization, place
   - Concept: abstract idea, definition
   - Fact: specific piece of information

2. Determine entity type
   - Person: names, roles, relationships
   - Organization: companies, teams, groups
   - Concept: ideas, definitions, explanations
   - Event: specific occurrences

3. Extract identifying attributes
   - Name variations
   - Associated context
   - Disambiguating details

4. Consider disambiguation if ambiguous
   - Multiple entities with same name
   - Context clues for resolution""",
        version=1,
    ),
    QueryIntent.EXPLORATORY: QueryTemplate(
        template_id="exploratory_query_v1",
        name="Exploratory Query Understanding",
        intent_type=QueryIntent.EXPLORATORY,
        pattern="""# Exploratory Query Analysis

1. Identify the topic/domain of interest
   - Broad area of exploration
   - Key concepts mentioned

2. Determine exploration breadth vs depth
   - Breadth: survey many related topics
   - Depth: deep dive into specific area

3. Consider related entities and relationships
   - Connected concepts
   - Associated people/organizations
   - Related events

4. Plan for multi-hop traversal if needed
   - Direct connections
   - Secondary connections
   - Semantic similarity expansion""",
        version=1,
    ),
}


class QueryTemplateDatabase:
    """Manage query understanding templates.

    Templates evolve via textual gradients (TOTAL framework).

    Features:
    - Seed templates for each intent type
    - Template selection based on intent
    - Template evolution via feedback
    - Connection to extraction templates

    Example:
        template_db = QueryTemplateDatabase()

        # Select template for intent
        template = template_db.select_template(QueryIntent.TEMPORAL)

        # Update template based on feedback
        template_db.update_template(
            intent=QueryIntent.TEMPORAL,
            new_pattern="...",
            feedback="Better handling of relative dates"
        )
    """

    def __init__(
        self,
        parent_template_db: Optional["TemplateDatabase"] = None,
    ):
        """Initialize query template database.

        Args:
            parent_template_db: Parent template database for composition
        """
        self.templates: Dict[QueryIntent, QueryTemplate] = {}
        self.parent_db = parent_template_db
        self._load_seed_templates()

        logger.info(
            f"QueryTemplateDatabase initialized with {len(self.templates)} templates"
        )

    def _load_seed_templates(self):
        """Initialize with core query templates."""
        for intent, template in SEED_TEMPLATES.items():
            self.templates[intent] = template.model_copy()

    def select_template(self, intent: QueryIntent) -> QueryTemplate:
        """Select template for intent type.

        Args:
            intent: Query intent

        Returns:
            Best matching template
        """
        template = self.templates.get(intent)

        if template is None:
            # Fallback to exploratory template
            logger.warning(f"No template for {intent}, using exploratory")
            template = self.templates[QueryIntent.EXPLORATORY]

        return template

    def get_template(self, template_id: str) -> Optional[QueryTemplate]:
        """Get template by ID.

        Args:
            template_id: Template identifier

        Returns:
            Template if found, None otherwise
        """
        for template in self.templates.values():
            if template.template_id == template_id:
                return template
        return None

    def update_template(
        self,
        intent: QueryIntent,
        new_pattern: str,
        feedback: str,
    ):
        """Update template based on textual gradient.

        Called when GRPO identifies better query understanding patterns.

        Args:
            intent: Intent to update
            new_pattern: New pattern text
            feedback: Feedback explaining the change
        """
        current = self.templates.get(intent)

        if current is None:
            logger.warning(f"No template found for {intent}")
            return

        # Create new version
        new_version = current.version + 1
        new_template = QueryTemplate(
            template_id=f"{intent.value}_query_v{new_version}",
            name=current.name,
            intent_type=intent,
            pattern=new_pattern,
            version=new_version,
            success_rate=0.0,  # Reset for new version
            last_updated=datetime.utcnow(),
        )

        self.templates[intent] = new_template

        logger.info(
            f"Updated template for {intent}: v{current.version} -> v{new_version}"
        )
        logger.debug(f"Feedback: {feedback}")

    def record_success(self, intent: QueryIntent, success: bool):
        """Record template usage success/failure.

        Updates success rate for template evolution decisions.

        Args:
            intent: Intent that was used
            success: Whether usage was successful
        """
        template = self.templates.get(intent)

        if template is None:
            return

        # Simple exponential moving average
        alpha = 0.1
        current_rate = template.success_rate
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate

        # Update template
        template.success_rate = new_rate
        template.last_updated = datetime.utcnow()

    def get_all_templates(self) -> Dict[QueryIntent, QueryTemplate]:
        """Get all templates.

        Returns:
            Dictionary mapping intent to template
        """
        return self.templates.copy()

    def get_template_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all templates.

        Returns:
            Dictionary mapping template_id to statistics
        """
        stats = {}
        for intent, template in self.templates.items():
            stats[template.template_id] = {
                "intent": intent.value,
                "version": template.version,
                "success_rate": template.success_rate,
                "created_at": template.created_at.isoformat(),
                "last_updated": template.last_updated.isoformat(),
            }
        return stats

    def reset_to_seed(self):
        """Reset all templates to seed versions.

        Useful for testing or when templates have degraded.
        """
        self.templates.clear()
        self._load_seed_templates()
        logger.info("Reset templates to seed versions")

    def compose_with_extraction_template(
        self,
        intent: QueryIntent,
        extraction_template_name: str,
    ) -> str:
        """Compose query template with extraction template.

        Allows combining query understanding with extraction patterns.

        Args:
            intent: Query intent
            extraction_template_name: Name of extraction template

        Returns:
            Composed template pattern
        """
        query_template = self.select_template(intent)
        composed = query_template.pattern

        if self.parent_db:
            try:
                # Try to get extraction template
                extraction_template = self.parent_db.select_template(
                    extraction_template_name
                )
                if extraction_template:
                    composed += f"\n\n# Extraction Pattern\n{extraction_template.pattern}"
            except Exception as e:
                logger.debug(f"Could not compose with extraction template: {e}")

        return composed

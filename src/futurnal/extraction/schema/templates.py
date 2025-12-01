"""
Thought Template System (TOTAL Framework)

Implements evolving thought templates using textual gradient refinement
for reusable "how to think" scaffolds that improve through use.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from futurnal.extraction.schema.models import (
    TextualGradient,
    ThoughtTemplate,
    TemplateStats,
)


class LLMClient(Protocol):
    """Protocol for LLM interactions."""

    def extract(self, prompt: str) -> Any:
        """Run extraction on a prompt."""
        ...

    def introspect(self, prompt: str) -> str:
        """
        Run introspection prompt (for textual gradients).

        Returns:
            JSON string with decision, feedback, proposed_change, confidence
        """
        ...


class TemplateDatabase:
    """
    Manage thought template collection.

    Provides 10+ seed templates, selection logic, and composition.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize template database.

        Args:
            storage_path: Path to store templates (default: ~/.futurnal/templates/)
        """
        self.templates: Dict[str, ThoughtTemplate] = {}
        self.storage_path = storage_path or Path.home() / ".futurnal" / "templates"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.load_seed_templates()
        self._load_from_disk()

    def load_seed_templates(self):
        """
        Initialize with 10+ seed templates.

        Production plan requirement: "Template database operational with 10+ seed templates"
        """
        seed_templates = [
            self._create_entity_recognition_template(),
            self._create_relationship_extraction_template(),
            self._create_temporal_reasoning_template(),
            self._create_event_detection_template(),
            self._create_causal_inference_template(),
            self._create_person_entity_template(),
            self._create_organization_entity_template(),
            self._create_concept_entity_template(),
            self._create_works_at_relationship_template(),
            self._create_created_relationship_template(),
            self._create_multi_phase_extraction_template(),
        ]

        for template in seed_templates:
            self.templates[template.template_id] = template

    def _create_entity_recognition_template(self) -> ThoughtTemplate:
        """General entity extraction template."""
        return ThoughtTemplate(
            template_id="entity_recognition_v1",
            name="Entity Recognition",
            description="Identify and extract entities from text",
            pattern="""# Entity Recognition

1. **Read text carefully** to identify noun phrases
2. **Identify entity types**:
   - Person: Names of people, pronouns referring to people
   - Organization: Companies, institutions, groups
   - Concept: Abstract ideas, topics, themes
   - Document: Referenced documents, files, resources

3. **Extract with metadata**:
   - Entity text (exact span from source)
   - Entity type classification
   - Confidence score (0.0-1.0)
   - Context (surrounding text for disambiguation)

4. **Validate**:
   - Check for ambiguity (same text, different meanings)
   - Verify entity type is appropriate
   - Ensure confidence reflects extraction quality""",
            version=1,
            composable_with=["relationship_extraction", "temporal_reasoning"]
        )

    def _create_relationship_extraction_template(self) -> ThoughtTemplate:
        """General relationship extraction template."""
        return ThoughtTemplate(
            template_id="relationship_extraction_v1",
            name="Relationship Extraction",
            description="Extract relationships between entities",
            pattern="""# Relationship Extraction

1. **Identify entity pairs** that appear related in text
2. **Determine relationship type**:
   - works_at: Employment relationships
   - created: Creation/authorship relationships
   - related_to: General association
   - mentions: One entity mentions another

3. **Extract relationship direction**:
   - Subject: Source entity
   - Predicate: Relationship type
   - Object: Target entity

4. **Add metadata**:
   - Confidence score
   - Temporal grounding (if applicable)
   - Causal nature (if applicable)

5. **Validate consistency**:
   - Check relationship makes semantic sense
   - Verify entity types are compatible with relationship type""",
            version=1,
            composable_with=["entity_recognition", "temporal_reasoning"]
        )

    def _create_temporal_reasoning_template(self) -> ThoughtTemplate:
        """Temporal relationship detection template."""
        return ThoughtTemplate(
            template_id="temporal_reasoning_v1",
            name="Temporal Relationship Detection",
            description="Detect temporal relationships between events",
            pattern="""# Temporal Reasoning

1. **Identify temporal markers**:
   - Explicit: ISO 8601 (2024-01-15), dates, times (9:00 AM)
   - Relative: "yesterday", "last week", "2 days ago"
   - Implicit: "during the meeting", "after lunch"

2. **Determine temporal order**:
   - BEFORE: A finishes before B starts
   - AFTER: A starts after B finishes
   - DURING: A occurs within B's timespan
   - SIMULTANEOUS: A and B occur at same time

3. **Look for causal language**:
   - "caused", "led to", "resulted in" → CAUSES
   - "enabled", "allowed", "made possible" → ENABLES
   - "prevented", "blocked" → PREVENTS
   - "triggered", "initiated" → TRIGGERS

4. **Validate temporal consistency**:
   - Check for contradictions
   - Verify timestamps make logical sense
   - Flag impossible temporal relationships

5. **Ground in timestamps** when possible:
   - Convert relative expressions to absolute timestamps
   - Associate events with temporal metadata""",
            version=1,
            composable_with=["event_detection", "causal_inference"]
        )

    def _create_event_detection_template(self) -> ThoughtTemplate:
        """Event entity identification template."""
        return ThoughtTemplate(
            template_id="event_detection_v1",
            name="Event Detection",
            description="Identify event entities with temporal grounding",
            pattern="""# Event Detection

1. **Identify event mentions**:
   - Verbs indicating actions (met, decided, created)
   - Nominalizations (meeting, decision, creation)
   - Event-indicating nouns (conference, deadline, milestone)

2. **Extract event properties**:
   - Event type (meeting, decision, creation, etc.)
   - Participants (entities involved)
   - Location (where it occurred)
   - Duration (how long)

3. **Temporal grounding (CRITICAL)**:
   - Timestamp (when event occurred)
   - Duration (start and end times if available)
   - Temporal relationships to other events

4. **Validate event extraction**:
   - Ensure event has temporal metadata
   - Check participants are valid entities
   - Verify event type is appropriate""",
            version=1,
            composable_with=["temporal_reasoning", "entity_recognition"]
        )

    def _create_causal_inference_template(self) -> ThoughtTemplate:
        """Event-event causal candidate detection template."""
        return ThoughtTemplate(
            template_id="causal_inference_v1",
            name="Causal Inference",
            description="Detect event-event causal relationship candidates",
            pattern="""# Causal Inference

1. **Identify event pairs** with potential causal relationship
2. **Look for causal indicators**:
   - Explicit: "caused", "led to", "resulted in", "because"
   - Enabling: "enabled", "allowed", "made possible"
   - Preventing: "prevented", "blocked", "stopped"
   - Triggering: "triggered", "initiated", "started"

3. **Assess temporal precedence**:
   - Cause must precede effect temporally
   - Check for temporal consistency

4. **Extract causal metadata**:
   - Causal relationship type (CAUSES/ENABLES/PREVENTS/TRIGGERS)
   - Confidence score
   - Supporting evidence from text

5. **Mark as causal candidate** (not definitive):
   - Phase 1 identifies candidates
   - Phase 3 will validate with Bradford Hill criteria
   - Store for future causal validation""",
            version=1,
            composable_with=["event_detection", "temporal_reasoning"]
        )

    def _create_person_entity_template(self) -> ThoughtTemplate:
        """Person-specific entity extraction template."""
        return ThoughtTemplate(
            template_id="person_entity_v1",
            name="Person Entity Extraction",
            description="Extract person entities with attributes",
            pattern="""# Person Entity Extraction

1. **Identify person mentions**:
   - Proper names (John Smith, Dr. Jane Doe)
   - Pronouns (he, she, they) with clear antecedents
   - Titles (CEO, Professor, Mr./Ms.)

2. **Extract person attributes**:
   - Full name
   - Title/role
   - Affiliations (organizations)
   - Contact information (if present)

3. **Resolve coreferences**:
   - Link pronouns to names
   - Handle multiple mentions of same person
   - Merge duplicate entities

4. **Validate person extraction**:
   - Ensure name is plausible
   - Check title/role is appropriate
   - Verify affiliations are valid organizations""",
            version=1,
            composable_with=["entity_recognition", "works_at_relationship"]
        )

    def _create_organization_entity_template(self) -> ThoughtTemplate:
        """Organization-specific entity extraction template."""
        return ThoughtTemplate(
            template_id="organization_entity_v1",
            name="Organization Entity Extraction",
            description="Extract organization entities with attributes",
            pattern="""# Organization Entity Extraction

1. **Identify organization mentions**:
   - Company names (Apple Inc., University of X)
   - Institutions (hospitals, schools, government)
   - Groups (teams, committees, boards)

2. **Extract organization attributes**:
   - Official name
   - Type (company, institution, group)
   - Industry/sector
   - Location

3. **Handle organization variations**:
   - Acronyms (MIT, NASA)
   - Informal names (Apple vs Apple Inc.)
   - Subsidiaries and parent companies

4. **Validate organization extraction**:
   - Ensure name is complete
   - Check type is appropriate
   - Verify not confused with person or location""",
            version=1,
            composable_with=["entity_recognition", "works_at_relationship"]
        )

    def _create_concept_entity_template(self) -> ThoughtTemplate:
        """Concept-specific entity extraction template."""
        return ThoughtTemplate(
            template_id="concept_entity_v1",
            name="Concept Entity Extraction",
            description="Extract abstract concept entities",
            pattern="""# Concept Entity Extraction

1. **Identify concept mentions**:
   - Abstract ideas (freedom, justice, innovation)
   - Topics (machine learning, philosophy, economics)
   - Themes (collaboration, sustainability, growth)

2. **Extract concept properties**:
   - Concept name
   - Category/domain
   - Related concepts
   - Defining characteristics

3. **Distinguish from concrete entities**:
   - Concepts are abstract, not physical
   - May be referenced indirectly
   - Often have multiple interpretations

4. **Validate concept extraction**:
   - Ensure concept is actually abstract
   - Check not confused with event or person
   - Verify concept name is meaningful""",
            version=1,
            composable_with=["entity_recognition", "relationship_extraction"]
        )

    def _create_works_at_relationship_template(self) -> ThoughtTemplate:
        """Employment relationship extraction template."""
        return ThoughtTemplate(
            template_id="works_at_relationship_v1",
            name="Works-At Relationship Extraction",
            description="Extract employment relationships between persons and organizations",
            pattern="""# Works-At Relationship Extraction

1. **Identify employment indicators**:
   - Explicit: "works at", "employed by", "employee of"
   - Implicit: "CEO of", "Professor at", "member of"

2. **Extract relationship**:
   - Subject: Person entity
   - Predicate: works_at
   - Object: Organization entity

3. **Add employment metadata**:
   - Role/title
   - Start date (if mentioned)
   - End date (if mentioned)
   - Employment type (full-time, part-time, contractor)

4. **Validate works_at relationship**:
   - Subject must be Person
   - Object must be Organization
   - Role should be plausible""",
            version=1,
            composable_with=["person_entity", "organization_entity"]
        )

    def _create_created_relationship_template(self) -> ThoughtTemplate:
        """Creation/authorship relationship extraction template."""
        return ThoughtTemplate(
            template_id="created_relationship_v1",
            name="Created Relationship Extraction",
            description="Extract creation/authorship relationships",
            pattern="""# Created Relationship Extraction

1. **Identify creation indicators**:
   - Explicit: "created", "authored", "wrote", "designed", "built"
   - Implicit: "by X", "X's work", "invention of X"

2. **Extract relationship**:
   - Subject: Creator (Person or Organization)
   - Predicate: created
   - Object: Created entity (Document, Concept, Product)

3. **Add creation metadata**:
   - Creation date (if mentioned)
   - Creation location (if mentioned)
   - Collaborators (if multiple creators)

4. **Validate created relationship**:
   - Subject must be capable of creation
   - Object must be something that can be created
   - Temporal consistency if dates present""",
            version=1,
            composable_with=["person_entity", "event_detection"]
        )

    def _create_multi_phase_extraction_template(self) -> ThoughtTemplate:
        """Multi-phase extraction composition example."""
        return ThoughtTemplate(
            template_id="multi_phase_extraction_v1",
            name="Multi-Phase Extraction",
            description="Composed template for complex multi-phase extraction",
            pattern="""# Multi-Phase Extraction

## Phase 1: Entity Extraction
[Apply entity_recognition template]

## Phase 2: Temporal Grounding
[Apply temporal_reasoning template]

## Phase 3: Relationship Extraction
[Apply relationship_extraction template]

## Phase 4: Event Detection
[Apply event_detection template]

## Phase 5: Causal Candidate Identification
[Apply causal_inference template]

**Note**: This is a composition example showing how templates can be combined
for complex extraction tasks.""",
            version=1,
            composable_with=[]  # This is already a composition
        )

    def select_template(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ThoughtTemplate]:
        """
        Select best template for task.

        Strategy:
        1. Exact match by template_id
        2. Keyword match in name/description
        3. Performance-based (highest success_rate)

        Args:
            task: Task description or template_id
            context: Optional context for selection

        Returns:
            Best matching template, or None if no match
        """
        # Strategy 1: Exact match by template_id
        if task in self.templates:
            return self.templates[task]

        # Strategy 2: Keyword match
        task_lower = task.lower()
        keyword_matches = []

        for template in self.templates.values():
            # Check if keywords from task appear in template name/description
            if any(word in template.name.lower() or word in template.description.lower()
                   for word in task_lower.split()):
                keyword_matches.append(template)

        if not keyword_matches:
            return None

        # Strategy 3: Performance-based selection among keyword matches
        # Sort by success rate, then confidence
        keyword_matches.sort(
            key=lambda t: (
                t.performance_stats.success_rate(),
                t.performance_stats.average_confidence
            ),
            reverse=True
        )

        return keyword_matches[0]

    def compose_templates(
        self,
        template_ids: List[str],
        strategy: str = "sequential"
    ) -> str:
        """
        Compose multiple templates for complex tasks.

        Args:
            template_ids: List of template IDs to compose
            strategy: Composition strategy (sequential, parallel, conditional)

        Returns:
            Composed template pattern

        Raises:
            ValueError: If template_id not found or strategy unknown
        """
        # Validate all template IDs exist
        templates = []
        for template_id in template_ids:
            if template_id not in self.templates:
                raise ValueError(f"Template '{template_id}' not found in database")
            templates.append(self.templates[template_id])

        if strategy == "sequential":
            return self._compose_sequential(templates)
        elif strategy == "parallel":
            return self._compose_parallel(templates)
        elif strategy == "conditional":
            return self._compose_conditional(templates)
        else:
            raise ValueError(f"Unknown composition strategy: {strategy}")

    def _compose_sequential(self, templates: List[ThoughtTemplate]) -> str:
        """Compose templates sequentially (Step 1, Step 2, ...)."""
        composed = "# Multi-Step Reasoning\n\n"

        for i, template in enumerate(templates, 1):
            composed += f"## Step {i}: {template.name}\n\n"
            composed += f"{template.pattern}\n\n"

        return composed.strip()

    def _compose_parallel(self, templates: List[ThoughtTemplate]) -> str:
        """Compose templates to apply in parallel."""
        composed = "# Parallel Analysis\n\n"
        composed += "Apply the following reasoning patterns simultaneously:\n\n"

        for i, template in enumerate(templates, 1):
            composed += f"## Analysis {i}: {template.name}\n\n"
            composed += f"{template.pattern}\n\n"

        return composed.strip()

    def _compose_conditional(self, templates: List[ThoughtTemplate]) -> str:
        """Compose templates with conditional application."""
        composed = "# Conditional Reasoning\n\n"
        composed += "Apply templates based on document type:\n\n"

        for i, template in enumerate(templates, 1):
            composed += f"## If condition {i} met: {template.name}\n\n"
            composed += f"{template.pattern}\n\n"

        return composed.strip()

    def add_template(self, template: ThoughtTemplate):
        """
        Add or update template in database.

        Args:
            template: Template to add
        """
        self.templates[template.template_id] = template
        self._save_to_disk(template)

    def remove_template(self, template_id: str):
        """
        Remove template from database (DISCARD decision).

        Args:
            template_id: Template to remove
        """
        if template_id in self.templates:
            del self.templates[template_id]
            # Remove from disk
            template_path = self.storage_path / f"{template_id}.json"
            if template_path.exists():
                template_path.unlink()

    def _save_to_disk(self, template: ThoughtTemplate):
        """Save template to disk as JSON."""
        template_path = self.storage_path / f"{template.template_id}.json"
        with open(template_path, "w") as f:
            json.dump(template.model_dump(), f, indent=2, default=str)

    def _load_from_disk(self):
        """Load templates from disk."""
        if not self.storage_path.exists():
            return

        for template_file in self.storage_path.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    template = ThoughtTemplate(**data)
                    # Don't overwrite seed templates
                    if template.template_id not in self.templates:
                        self.templates[template.template_id] = template
            except Exception:
                # Skip invalid templates
                continue


class TemplateRefinementEngine:
    """
    Refine templates via textual gradients (TOTAL framework).

    Uses LLM introspection to generate KEEP/FIX/DISCARD decisions.
    """

    def __init__(self, llm: LLMClient, template_db: TemplateDatabase):
        """
        Initialize refinement engine.

        Args:
            llm: LLM client for introspection
            template_db: Template database to refine
        """
        self.llm = llm
        self.template_db = template_db

    def analyze_template_performance(
        self,
        template: ThoughtTemplate,
        recent_results: List[Any]  # Should be ExtractionResult
    ) -> TextualGradient:
        """
        LLM introspects template performance.

        Process:
        1. Build introspection prompt with template + results
        2. LLM generates KEEP/FIX/DISCARD decision
        3. Parse response into TextualGradient

        Args:
            template: Template to analyze
            recent_results: Recent extraction results using this template

        Returns:
            TextualGradient with refinement decision
        """
        # Build introspection prompt
        prompt = self._build_introspection_prompt(template, recent_results)

        # Get LLM introspection
        response = self.llm.introspect(prompt)

        # Parse textual gradient
        gradient = self._parse_textual_gradient(template.template_id, response)

        return gradient

    def _build_introspection_prompt(
        self,
        template: ThoughtTemplate,
        results: List[Any]
    ) -> str:
        """
        Construct LLM introspection prompt.

        Args:
            template: Template to analyze
            results: Recent extraction results

        Returns:
            Introspection prompt for LLM
        """
        # Calculate statistics
        success_rate = template.performance_stats.success_rate()
        avg_confidence = template.performance_stats.average_confidence
        usage_count = template.performance_stats.usage_count

        # Format recent results (sample up to 5)
        results_sample = results[:5] if len(results) > 5 else results
        formatted_results = []

        for i, result in enumerate(results_sample, 1):
            success_marker = "✓" if getattr(result, "success", True) else "✗"
            confidence = getattr(result, "confidence", 0.0)
            content = getattr(result, "content", "N/A")
            formatted_results.append(
                f"{i}. Success {success_marker} | Confidence: {confidence:.2f} | \"{content}\""
            )

        results_text = "\n".join(formatted_results)

        prompt = f"""# Template Performance Analysis

**Template**: {template.name} (v{template.version})
**Pattern**:
{template.pattern}

**Performance Statistics**:
- Usage count: {usage_count}
- Success rate: {success_rate:.1%}
- Average confidence: {avg_confidence:.2f}

**Recent Results** ({len(results_sample)} samples):
{results_text}

**Task**: Analyze if this template is working well.

**Output ONE of**:
1. KEEP - Template works well (minor refinement suggested)
2. FIX - Template has issues (provide revised template)
3. DISCARD - Template is fundamentally flawed

**Format** (JSON):
{{
  "decision": "KEEP|FIX|DISCARD",
  "feedback": "detailed reasoning for decision",
  "proposed_change": "new template text (if FIX, else null)",
  "confidence": 0.0-1.0
}}
"""

        return prompt

    def _parse_textual_gradient(
        self,
        template_id: str,
        response: str
    ) -> TextualGradient:
        """
        Parse JSON response into TextualGradient.

        Args:
            template_id: Template being refined
            response: JSON string from LLM

        Returns:
            TextualGradient parsed from response

        Raises:
            ValueError: If response is invalid JSON or missing required fields
        """
        try:
            data = json.loads(response)

            return TextualGradient(
                template_id=template_id,
                decision=data["decision"],
                feedback=data["feedback"],
                proposed_change=data.get("proposed_change"),
                confidence=data["confidence"],
                based_on_results=[],  # Could track result IDs
                created_at=datetime.utcnow()
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid textual gradient response: {e}")

    def apply_textual_gradient(
        self,
        template: ThoughtTemplate,
        gradient: TextualGradient
    ) -> Optional[ThoughtTemplate]:
        """
        Apply refinement based on textual gradient.

        Returns:
        - KEEP: Original template (stats updated)
        - FIX: New template version (v+1)
        - DISCARD: None (template removed)

        Args:
            template: Template to refine
            gradient: Textual gradient with decision

        Returns:
            Refined template or None if DISCARD
        """
        if gradient.decision == "DISCARD":
            return None

        if gradient.decision == "KEEP":
            # Update evolution history, no version change
            template.last_updated = datetime.utcnow()
            return template

        if gradient.decision == "FIX" and gradient.proposed_change:
            # Create new version
            new_template = ThoughtTemplate(
                template_id=f"{template.name.lower().replace(' ', '_')}_v{template.version + 1}",
                name=template.name,
                description=template.description,
                pattern=gradient.proposed_change,
                version=template.version + 1,
                performance_stats=TemplateStats(),  # Reset stats for new version
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                parent_version=template.template_id,
                composable_with=template.composable_with
            )

            return new_template

        return template

    def evolve_templates(
        self,
        extraction_results: Dict[str, List[Any]]
    ) -> Dict[str, TextualGradient]:
        """
        Batch evolution for all templates.

        Args:
            extraction_results: Map of template_id to list of results

        Returns:
            Map of template_id to textual gradient
        """
        gradients = {}

        for template_id, results in extraction_results.items():
            if template_id in self.template_db.templates:
                template = self.template_db.templates[template_id]
                gradient = self.analyze_template_performance(template, results)
                gradients[template_id] = gradient

                # Apply gradient
                evolved = self.apply_textual_gradient(template, gradient)

                if evolved is None:
                    # DISCARD
                    self.template_db.remove_template(template_id)
                elif evolved.version > template.version:
                    # FIX - add new version
                    self.template_db.add_template(evolved)
                else:
                    # KEEP - update existing
                    self.template_db.add_template(evolved)

        return gradients

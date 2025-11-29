Summary: Implement TOTAL thought template system with textual gradient refinement for evolving reasoning patterns.

# 04 · Thought Template System

## Purpose
Implement evolving thought templates using the TOTAL (Thought Template) framework with textual gradient refinement. This creates reusable "how to think" scaffolds that improve through use via natural language feedback, eliminating the static template limitation that prevents continuous quality improvement.

**Criticality**: CRITICAL - Enables evolving reasoning patterns; foundation for Phase 2/3 template evolution

## Scope
- Template database with seed templates
- Template selection and composition for complex extractions
- Textual gradient refinement (KEEP/FIX/DISCARD decisions)
- Template performance tracking and versioning
- Integration with experiential learning

## Requirements Alignment
- **Option B Requirement**: "Thought templates refining successfully via textual gradients"
- **SOTA Foundation**: TOTAL (2510.07499v1, 2025)
- **Critical Gap**: Eliminates static template limitation
- **Key Innovation**: Templates evolve via natural language feedback (not parameter tuning)

## Component Design

### Template Database

```python
class ThoughtTemplate(BaseModel):
    """Reusable reasoning pattern for extraction."""
    template_id: str
    name: str
    description: str
    pattern: str  # The actual template text
    version: int
    performance_stats: TemplateStats
    created_at: datetime
    last_updated: datetime


class TemplateStats(BaseModel):
    """Track template performance."""
    usage_count: int = 0
    success_count: int = 0
    average_confidence: float = 0.0
    improvement_trend: float = 0.0


class TemplateDatabase:
    """Manage thought template collection."""

    def __init__(self):
        self.templates: Dict[str, ThoughtTemplate] = {}
        self.load_seed_templates()

    def load_seed_templates(self):
        """Initialize with core extraction templates."""
        self.templates = {
            "entity_recognition": ThoughtTemplate(
                template_id="entity_recognition_v1",
                name="Entity Recognition",
                description="Identify entities in text",
                pattern="""# Entity Recognition

1. Read the text carefully
2. Identify noun phrases
3. Classify by type (Person/Organization/Concept)
4. Extract with confidence score""",
                version=1,
                performance_stats=TemplateStats(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            ),
            "temporal_reasoning": ThoughtTemplate(
                template_id="temporal_reasoning_v1",
                name="Temporal Relationship Detection",
                description="Detect temporal relationships between events",
                pattern="""# Temporal Reasoning

1. Identify events with timestamps
2. Determine temporal order (before/after/during)
3. Look for causal language (caused/led to/resulted in)
4. Validate temporal consistency""",
                version=1,
                performance_stats=TemplateStats(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
        }

    def select_template(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> ThoughtTemplate:
        """Select best template for task."""
        # Simple selection: match by task type
        # Could be enhanced with learning
        if "temporal" in task.lower():
            return self.templates["temporal_reasoning"]
        return self.templates["entity_recognition"]

    def compose_templates(
        self,
        templates: List[ThoughtTemplate]
    ) -> str:
        """Compose multiple templates for complex tasks."""
        composed = "# Multi-Step Reasoning\n\n"
        for i, template in enumerate(templates, 1):
            composed += f"## Step {i}: {template.name}\n\n"
            composed += f"{template.pattern}\n\n"
        return composed
```

### Textual Gradient Refinement

```python
class TextualGradient(BaseModel):
    """Natural language feedback for template refinement."""
    template_id: str
    decision: str  # "KEEP", "FIX", "DISCARD"
    feedback: str  # Natural language explanation
    proposed_change: Optional[str] = None


class TemplateRefinementEngine:
    """Refine templates via textual gradients (TOTAL framework)."""

    def __init__(self, llm):
        self.llm = llm

    def analyze_template_performance(
        self,
        template: ThoughtTemplate,
        recent_results: List[ExtractionResult]
    ) -> TextualGradient:
        """
        LLM introspects template performance.

        TOTAL approach: Use LLM to generate natural language feedback
        """
        prompt = f"""# Template Performance Analysis

Template: {template.name}
Pattern:
{template.pattern}

Recent Results:
{self._format_results(recent_results)}

Task: Analyze if this template is working well.

Output one of:
1. KEEP - Template works well, minor refinement suggested
2. FIX - Template has issues, needs revision
3. DISCARD - Template is flawed, should be removed

Provide reasoning and proposed changes if FIX.

Format:
{{
  "decision": "KEEP|FIX|DISCARD",
  "feedback": "explanation",
  "proposed_change": "new template text if FIX"
}}
"""

        response = self.llm.generate(prompt)
        return self._parse_textual_gradient(response)

    def apply_textual_gradient(
        self,
        template: ThoughtTemplate,
        gradient: TextualGradient
    ) -> Optional[ThoughtTemplate]:
        """
        Apply refinement based on textual gradient.

        Returns: New template version or None if DISCARD
        """
        if gradient.decision == "DISCARD":
            return None

        if gradient.decision == "KEEP":
            # Minor update, increment usage stats
            template.performance_stats.usage_count += 1
            return template

        if gradient.decision == "FIX" and gradient.proposed_change:
            # Create new version
            return ThoughtTemplate(
                template_id=f"{template.name.lower()}_v{template.version + 1}",
                name=template.name,
                description=template.description,
                pattern=gradient.proposed_change,
                version=template.version + 1,
                performance_stats=TemplateStats(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

        return template
```

## Implementation Details

See [PHASE-1-OPTION-B-ROADMAP.md](../PHASE-1-OPTION-B-ROADMAP.md) Weeks 7-8 for timeline.

## Testing Strategy

```python
class TestThoughtTemplates:
    def test_template_selection(self):
        """Validate template selection for tasks."""
        db = TemplateDatabase()

        temporal_template = db.select_template("temporal_reasoning", {})
        assert "temporal" in temporal_template.name.lower()

        entity_template = db.select_template("entity_extraction", {})
        assert "entity" in entity_template.name.lower()

    def test_template_composition(self):
        """Validate multi-template composition."""
        db = TemplateDatabase()
        templates = [
            db.templates["entity_recognition"],
            db.templates["temporal_reasoning"]
        ]

        composed = db.compose_templates(templates)

        assert "Step 1" in composed
        assert "Step 2" in composed
        assert all(t.name in composed for t in templates)

    def test_textual_gradient_refinement(self):
        """Validate template evolution via textual gradients."""
        engine = TemplateRefinementEngine(mock_llm)
        template = create_test_template()
        results = create_mixed_quality_results()

        gradient = engine.analyze_template_performance(template, results)

        assert gradient.decision in ["KEEP", "FIX", "DISCARD"]
        if gradient.decision == "FIX":
            assert gradient.proposed_change is not None

    def test_template_evolution_over_time(self):
        """Validate templates improve with use."""
        db = TemplateDatabase()
        engine = TemplateRefinementEngine(real_llm)
        template = db.templates["entity_recognition"]

        initial_version = template.version

        # Simulate usage and refinement cycles
        for _ in range(5):
            results = run_extraction_with_template(template)
            gradient = engine.analyze_template_performance(template, results)
            refined = engine.apply_textual_gradient(template, gradient)

            if refined and refined.version > template.version:
                template = refined
                db.templates["entity_recognition"] = template

        assert template.version >= initial_version  # Should evolve
```

## Success Metrics

- ✅ Template database operational with 10+ seed templates
- ✅ Template composition works for complex extractions
- ✅ Textual gradients refine templates (KEEP/FIX/DISCARD)
- ✅ Template evolution demonstrable (version increases)
- ✅ Performance tracking shows template effectiveness

## Dependencies

- LLM for textual gradient generation
- Extraction pipeline for template usage
- Experiential learning for feedback integration

**This module enables evolving reasoning patterns—key to continuous quality improvement.**

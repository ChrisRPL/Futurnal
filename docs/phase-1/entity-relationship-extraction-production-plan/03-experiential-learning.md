Summary: Implement Training-Free GRPO for Ghost→Animal evolution via experiential knowledge as token priors without parameter updates.

# 03 · Experiential Learning Foundation

## Purpose
Implement Training-Free GRPO (Group Relative Policy Optimization) to enable Ghost→Animal AI evolution through experiential knowledge accumulation—without fine-tuning the base model. This eliminates the passive feedback limitation and creates a self-improving extraction system that learns from user's unique data patterns while preserving the Ghost model's general capabilities.

**Criticality**: CRITICAL - Core mechanism for Ghost→Animal evolution; enables Phase 2 proactive insights

## Scope
- Experiential knowledge storage as natural language token priors
- Semantic advantage extraction via LLM introspection
- Rollout generation and group comparison
- World State Model for quality tracking
- Multi-epoch learning with knowledge pruning
- Integration with extraction pipeline

## Requirements Alignment
- **Option B Requirement**: "Extraction quality improves through use via Training-Free GRPO"
- **SOTA Foundation**: Training-Free GRPO (2510.08191v1, 2025)
- **Critical Gap**: Eliminates passive feedback limitation
- **Key Innovation**: Ghost + Experiential Knowledge = Animal behavior

## Component Design

### Training-Free GRPO Core

```python
from typing import List, Dict, Tuple
from pydantic import BaseModel


class ExperientialKnowledge(BaseModel):
    """Natural language patterns learned from experience."""
    pattern_id: str
    description: str
    context: str
    success_count: int = 0
    failure_count: int = 0
    confidence: float
    examples: List[str]
    created_at: datetime


class SemanticAdvantage(BaseModel):
    """Advantage signal from LLM introspection."""
    better_approach: str
    worse_approach: str
    reasoning: str
    confidence: float


class TrainingFreeGRPO:
    """
    Lightweight evolution without parameter updates.

    Key Innovation: Experiential knowledge = token priors (not parameters)
    """

    def __init__(
        self,
        llm,
        knowledge_capacity: int = 20,  # Keep top N patterns
        rollout_group_size: int = 4
    ):
        self.llm = llm
        self.experiential_knowledge: List[ExperientialKnowledge] = []
        self.knowledge_capacity = knowledge_capacity
        self.rollout_group_size = rollout_group_size

    def generate_rollouts(
        self,
        document: Document,
        base_prompt: str
    ) -> List[ExtractionResult]:
        """
        Generate K rollouts for same document.

        Ghost model runs K times on same input → K different extractions
        """
        rollouts = []
        for i in range(self.rollout_group_size):
            # Add experiential knowledge as context (token priors)
            prompt = self._build_prompt_with_experience(base_prompt, document)

            # Run Ghost model (no parameter changes)
            result = self.llm.extract(prompt)
            rollouts.append(result)

        return rollouts

    def extract_semantic_advantages(
        self,
        rollouts: List[ExtractionResult],
        ground_truth: Optional[ExtractionResult] = None
    ) -> List[SemanticAdvantage]:
        """
        LLM introspection: compare rollouts to discover better approaches.

        Training-Free GRPO innovation: Use LLM to generate advantages
        """
        advantages = []

        # Rank rollouts by quality
        ranked = self._rank_rollouts_by_quality(rollouts, ground_truth)

        # Compare best vs worst
        for i in range(len(ranked) // 2):
            better = ranked[i]
            worse = ranked[-(i+1)]

            # Ask LLM: "Why is extraction A better than extraction B?"
            advantage = self._llm_introspect_advantage(better, worse)
            if advantage.confidence > 0.7:
                advantages.append(advantage)

        return advantages

    def update_experiential_knowledge(
        self,
        advantages: List[SemanticAdvantage]
    ):
        """
        Update knowledge base from semantic advantages.

        Key: Store as natural language (token priors), not parameters
        """
        for advantage in advantages:
            knowledge = ExperientialKnowledge(
                pattern_id=self._generate_pattern_id(advantage),
                description=advantage.better_approach,
                context=advantage.reasoning,
                confidence=advantage.confidence,
                examples=[],
                created_at=datetime.utcnow()
            )

            self._add_to_knowledge_base(knowledge)

        # Prune to capacity
        self._prune_knowledge_base()

    def _build_prompt_with_experience(
        self,
        base_prompt: str,
        document: Document
    ) -> str:
        """
        Inject experiential knowledge as token priors.

        This is how Ghost + Experience = Animal
        """
        if not self.experiential_knowledge:
            return base_prompt

        experience_context = "## Learned Patterns\n\n"
        for knowledge in self.experiential_knowledge[:5]:  # Top 5
            experience_context += f"- {knowledge.description}\n"
            experience_context += f"  Context: {knowledge.context}\n"
            experience_context += f"  Success rate: {knowledge.success_count}/{knowledge.success_count + knowledge.failure_count}\n\n"

        return f"{experience_context}\n\n{base_prompt}"
```

### World State Model

```python
class WorldStateModel:
    """Track extraction quality trajectory and identify patterns."""

    def __init__(self):
        self.quality_history: List[QualityMetrics] = []
        self.success_patterns: List[str] = []
        self.failure_patterns: List[str] = []

    def assess_extraction_trajectory(
        self,
        recent_extractions: List[ExtractionResult]
    ) -> Dict[str, float]:
        """
        Assess if quality is improving over time.

        Metrics:
        - Precision trend
        - Confidence trend
        - Success rate trend
        """
        if len(recent_extractions) < 10:
            return {"insufficient_data": True}

        metrics = {
            "current_precision": self._compute_precision(recent_extractions[-10:]),
            "previous_precision": self._compute_precision(recent_extractions[-20:-10]),
            "improvement": 0.0
        }

        metrics["improvement"] = metrics["current_precision"] - metrics["previous_precision"]

        return metrics

    def generate_curriculum(
        self,
        pending_documents: List[Document]
    ) -> List[Document]:
        """
        Order documents by learning value.

        Priority:
        1. Documents with patterns similar to recent failures
        2. Documents with novel patterns
        3. Documents similar to recent successes
        """
        scored = []
        for doc in pending_documents:
            score = self._compute_learning_value(doc)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]
```

## Implementation Details

See [PHASE-1-OPTION-B-ROADMAP.md](../PHASE-1-OPTION-B-ROADMAP.md) Weeks 9-12 for timeline.

## Testing Strategy

```python
class TestTrainingFreeGRPO:
    def test_rollout_generation(self):
        """Validate K rollouts generated for same input."""
        grpo = TrainingFreeGRPO(mock_llm, rollout_group_size=4)
        document = load_test_document()

        rollouts = grpo.generate_rollouts(document, "Extract entities")

        assert len(rollouts) == 4
        # Rollouts should differ (sampling variation)
        assert len(set(str(r) for r in rollouts)) > 1

    def test_semantic_advantage_extraction(self):
        """Validate LLM introspection produces advantages."""
        grpo = TrainingFreeGRPO(mock_llm)
        rollouts = create_mock_rollouts_with_quality_variance()

        advantages = grpo.extract_semantic_advantages(rollouts)

        assert len(advantages) > 0
        assert all(a.confidence > 0.7 for a in advantages)

    def test_quality_improvement_over_time(self):
        """Critical: Validate quality improves with experience."""
        grpo = TrainingFreeGRPO(real_llm)
        documents = load_diverse_corpus(50)

        precision_epoch1 = measure_precision(grpo, documents[:25])
        # Learn from epoch 1
        for doc in documents[:25]:
            rollouts = grpo.generate_rollouts(doc, base_prompt)
            advantages = grpo.extract_semantic_advantages(rollouts)
            grpo.update_experiential_knowledge(advantages)

        precision_epoch2 = measure_precision(grpo, documents[25:])

        assert precision_epoch2 > precision_epoch1, "Quality must improve"
```

## Success Metrics

- ✅ Extraction quality improves measurably over 50+ documents
- ✅ Ghost model parameters remain frozen (verified)
- ✅ Experiential knowledge stored as natural language
- ✅ Semantic advantages generated with >0.7 confidence
- ✅ Knowledge pruning maintains top N patterns

## Dependencies

- LLM for rollout generation and introspection
- Extraction pipeline for integration
- Quality metrics from determinism tests

**This module enables Ghost→Animal evolution—the core innovation of Option B.**

# Step 06: Experiential Learning Foundation

## Status: TODO

## Objective

Implement the foundation for experiential learning where the Ghost learns from user data without model fine-tuning. This is THE core differentiator that enables Ghost→Animal evolution.

## Research Foundation

### Primary Papers:

#### SEAgent (2508.04700v2) - CRITICAL
**Key Innovation**: Self-evolving through experiential learning
**Technical Approach**:
- World State Model for step-wise trajectory assessment
- Curriculum Generator: simple → medium → complex
- GRPO (Group Relative Policy Optimization) on successes
- Adversarial imitation on failures

### Option B Principles (`.cursor/rules/option-b-principles.mdc`):
> "Ghost model FROZEN - No parameter updates"
> "Learning via token priors (not fine-tuning)"
> "Training-free GRPO framework"

## Implementation Tasks

### 1. World State Model

**New File**: `src/futurnal/learning/world_state.py`

```python
"""
World State Model - Extraction Quality Assessment

Per SEAgent: Assess task completion quality step-wise.
"""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ExtractionState:
    """State of extraction for a document."""
    document_id: str
    entity_count: int
    relation_count: int
    temporal_coverage: float  # 0-1
    confidence_avg: float
    consistency_score: float  # No temporal contradictions

class WorldStateModel:
    """Assess extraction quality without parameter updates."""

    def assess_extraction(self, state: ExtractionState) -> float:
        """
        Compute quality score for extraction.

        Returns 0-1 quality score.
        """
        scores = [
            min(state.entity_count / 10, 1.0) * 0.2,  # Entity coverage
            min(state.relation_count / 5, 1.0) * 0.2,  # Relation coverage
            state.temporal_coverage * 0.3,  # Temporal awareness
            state.confidence_avg * 0.15,  # Confidence
            state.consistency_score * 0.15,  # Consistency
        ]
        return sum(scores)

    def record_trajectory(self, document_id: str, success: bool, quality: float):
        """Record extraction trajectory for learning."""
        # Store in experiential memory (not model parameters)
        pass
```

### 2. Curriculum Generator

**New File**: `src/futurnal/learning/curriculum.py`

```python
"""
Curriculum Generator - Progressive Document Complexity

Per SEAgent: Order tasks from simple → complex.
"""

class CurriculumGenerator:
    """Generate document processing curriculum."""

    def order_documents(self, documents: List[dict]) -> List[dict]:
        """
        Order documents by estimated complexity.

        Complexity factors:
        - Document length
        - Entity density
        - Temporal expression count
        - Cross-reference count
        """
        return sorted(documents, key=self._estimate_complexity)

    def _estimate_complexity(self, doc: dict) -> float:
        """Estimate document processing complexity."""
        length_score = min(len(doc.get('content', '')) / 10000, 1.0)
        # Additional heuristics...
        return length_score
```

### 3. Token Priors Storage

**New File**: `src/futurnal/learning/token_priors.py`

```python
"""
Token Priors - Learning Without Parameter Updates

Per Option B: Ghost model FROZEN, learning through token priors.
"""

from typing import Dict, List

class TokenPriorStore:
    """Store learned token priors for prompt enhancement."""

    def __init__(self):
        self.entity_priors: Dict[str, float] = {}  # Entity type → frequency
        self.relation_priors: Dict[str, float] = {}  # Relation type → frequency
        self.temporal_priors: Dict[str, float] = {}  # Temporal pattern → frequency

    def update_from_extraction(self, extraction_result: dict):
        """Update priors from successful extraction."""
        for entity in extraction_result.get('entities', []):
            entity_type = entity.get('type', 'Unknown')
            self.entity_priors[entity_type] = self.entity_priors.get(entity_type, 0) + 1

    def get_prompt_context(self) -> str:
        """Generate prompt context from priors."""
        top_entities = sorted(self.entity_priors.items(), key=lambda x: -x[1])[:10]
        return f"Common entity types in this knowledge base: {', '.join(e[0] for e in top_entities)}"
```

### 4. Quality Feedback Integration

Integrate with search quality feedback from Step 02.

## Success Criteria (From Quality Gates)

- [ ] Ghost model parameters unchanged after learning
- [ ] Experiential knowledge stored as token priors
- [ ] Quality progression demonstrable over 50+ documents
- [ ] World State Model assesses extraction quality
- [ ] Curriculum orders documents by complexity

## Files to Create

- `src/futurnal/learning/world_state.py`
- `src/futurnal/learning/curriculum.py`
- `src/futurnal/learning/token_priors.py`
- `src/futurnal/learning/__init__.py`

## Dependencies

- **Step 04**: Temporal extraction (quality assessment needs temporal coverage)
- **Step 05**: Schema evolution (learning interacts with schema)

## Next Step

Proceed to **Step 07: Causal Structure Preparation**.

## Research References

1. **SEAgent**: `docs/phase-1/papers/converted/2508.04700v2.md`
2. **Option B Principles**: `.cursor/rules/option-b-principles.mdc`
3. **Experiential Learning Rules**: `.cursor/rules/experiential-learning.mdc`

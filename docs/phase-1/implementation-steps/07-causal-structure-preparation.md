# Step 07: Causal Structure Preparation

## Status: TODO

## Objective

Prepare the PKG structure for Phase 3 causal inference by ensuring causal-candidate relationships are properly marked and temporal ordering is preserved.

## Research Foundation

### Primary Papers:

#### CausalRAG (ACL 2025) - CRITICAL
**Key Innovation**: Integrates causal graphs into RAG
**Requirement**: Causal structure must be prepared in Phase 1

#### Temporal KG Extrapolation (IJCAI 2024)
**Key Innovation**: Causal subhistory identification
**Requirement**: Temporal ordering preserved for causal inference

## Implementation Tasks

### 1. Causal Relationship Taxonomy

**File**: `src/futurnal/extraction/causal/models.py`

```python
"""Causal relationship taxonomy for Phase 3."""

from enum import Enum

class CausalRelationType(Enum):
    """Types of causal relationships per CausalRAG."""
    CAUSES = "causes"      # A directly causes B
    ENABLES = "enables"    # A enables B (necessary but not sufficient)
    PREVENTS = "prevents"  # A prevents B
    TRIGGERS = "triggers"  # A triggers B (sufficient condition)
    CORRELATES = "correlates"  # A correlates with B (not yet validated)
```

### 2. Causal Candidate Detection

**File**: `src/futurnal/extraction/causal/relationship_detector.py`

Mark relationships as causal candidates based on:
- Temporal ordering (cause must precede effect)
- Language patterns ("caused by", "led to", "resulted in")
- Event-Event relationships

### 3. Bradford Hill Criteria Metadata

Prepare metadata for Phase 3 Bradford Hill validation:
- Temporal precedence
- Strength of association
- Consistency
- Specificity

### 4. Graph Optimization

Ensure PKG indexes support causal queries:
```cypher
// Causal chain traversal
MATCH path = (start:Event)-[:CAUSES*1..5]->(end:Event)
WHERE start.id = $start_id
RETURN path
```

## Success Criteria

- [ ] Causal relationship taxonomy defined
- [ ] Causal candidates marked in PKG
- [ ] Temporal ordering preserved (cause before effect)
- [ ] Bradford Hill metadata prepared
- [ ] Causal queries performant (<100ms)

## Files to Create/Modify

- `src/futurnal/extraction/causal/models.py` - Taxonomy
- `src/futurnal/extraction/causal/relationship_detector.py` - Detection
- `src/futurnal/extraction/causal/bradford_hill_prep.py` - Metadata

## Dependencies

- **Step 04**: Temporal extraction (temporal ordering)
- **Step 05**: Schema evolution (Event types)

## Next Step

Proceed to **Step 08: Frontend Intelligence Integration**.

## Research References

1. **CausalRAG**: `docs/phase-1/SOTA_RESEARCH_SUMMARY.md` (Paper #4)
2. **Causal-Copilot**: `docs/phase-1/papers/converted/2504.13263v2.md`

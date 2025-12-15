# Step 07: Causal Structure Preparation

## Status: COMPLETE

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

## Implementation Summary

### 1. Causal Relationship Taxonomy

**File**: `src/futurnal/extraction/causal/models.py`

Implemented 6 causal relationship types (exceeds spec requirement of 5):

```python
class CausalRelationshipType(str, Enum):
    CAUSES = "causes"              # Strong causal claim (A causes B)
    ENABLES = "enables"            # Prerequisite relationship (A enables B)
    PREVENTS = "prevents"          # Blocking relationship (A prevents B)
    TRIGGERS = "triggers"          # Immediate causation (A triggers B)
    LEADS_TO = "leads_to"          # Indirect causation (A leads to B)
    CONTRIBUTES_TO = "contributes_to"  # Partial causation (A contributes to B)
```

### 2. Causal Candidate Detection

**File**: `src/futurnal/extraction/causal/relationship_detector.py`

`CausalRelationshipDetector` class implements:
- Temporal ordering validation (cause must precede effect)
- 40+ causal language pattern indicators
- LLM-based causal evidence extraction
- Confidence assessment (threshold: 0.6)
- Relationship type inference from evidence

### 3. Bradford Hill Criteria Metadata

**File**: `src/futurnal/extraction/causal/models.py` & `bradford_hill_prep.py`

All 9 Bradford Hill criteria structure prepared:

```python
class BradfordHillCriteria(BaseModel):
    temporality: bool              # Criterion 1: VALIDATED in Phase 1
    strength: Optional[float]      # Criterion 2: Phase 3
    dose_response: Optional[bool]  # Criterion 3: Phase 3
    consistency: Optional[float]   # Criterion 4: Phase 3
    plausibility: Optional[str]    # Criterion 5: Phase 3
    coherence: Optional[bool]      # Criterion 6: Phase 3
    experiment_possible: Optional[bool]  # Criterion 7: Phase 3
    analogy: Optional[str]         # Criterion 8: Phase 3
    specificity: Optional[bool]    # Criterion 9: Phase 3
```

### 4. Graph Optimization

**Files**: `src/futurnal/pkg/schema/constraints.py` & `src/futurnal/pkg/queries/temporal.py`

Neo4j indexes for causal queries:
- `event_timestamp_index` - Temporal ordering queries
- `event_type_index` - Event type filtering
- `event_timestamp_type_index` - Composite queries

Causal chain query support in `TemporalGraphQueries.query_causal_chain()`:
```cypher
MATCH path = (start:Event {id: $start_id})-[:CAUSES|ENABLES|TRIGGERS*1..{max_hops}]->(end:Event)
RETURN path, confidences, evidence, depth
```

### 5. PKG Integration Bridge

**File**: `src/futurnal/extraction/causal/pkg_integration.py`

`CausalPKGIntegration` class bridges extraction and PKG layers:
- Transforms `CausalCandidate` to `CausalRelationshipProps`
- Maps extraction relationship types to PKG types
- Supports single and bulk storage operations

## Success Criteria

- [x] Causal relationship taxonomy defined (`models.py:48-65` - 6 types)
- [x] Causal candidates marked in PKG (`is_causal_candidate` flag)
- [x] Temporal ordering preserved (100% validation in detector)
- [x] Bradford Hill metadata prepared (9 criteria structure)
- [x] Causal queries performant (<100ms with Neo4j indexes)

## Test Coverage

**105 tests** in `tests/extraction/causal/`:
- `test_models.py` - Model validation (16 tests)
- `test_event_extractor.py` - Event extraction (24 tests)
- `test_relationship_detector.py` - Causal detection (29 tests)
- `test_bradford_hill_prep.py` - Bradford Hill preparation (8 tests)
- `test_causal_integration.py` - End-to-end workflow (5 tests)
- `test_causal_performance.py` - Performance benchmarks (6 tests)
- `test_pkg_integration.py` - PKG bridge (17 tests)

## Files Implemented

| File | Purpose | Lines |
|------|---------|-------|
| `models.py` | Taxonomy, CausalCandidate, BradfordHillCriteria | ~266 |
| `relationship_detector.py` | Causal candidate detection | ~504 |
| `event_extractor.py` | Event extraction with temporal grounding | ~389 |
| `bradford_hill_prep.py` | Bradford Hill criteria preparation | ~133 |
| `pkg_integration.py` | PKG storage bridge | ~200 |
| `__init__.py` | Module exports | ~40 |

## Dependencies

- **Step 04**: Temporal extraction (temporal ordering) - SATISFIED
- **Step 05**: Schema evolution (Event types) - SATISFIED

## Research Alignment

### CausalRAG Compliance:
- [x] Causal graphs constructed separately from semantic graphs
- [x] Explicit causal relationship types (not generic "related")
- [x] Temporal ordering enforced (cause before effect)

### Bradford Hill Compliance:
- [x] Criterion 1 (Temporality): Validated in Phase 1
- [x] Criteria 2-9: Structure prepared, fields nullable for Phase 3

### Option B Compliance:
- [x] No hardcoded event types (seed schema, discoverable)
- [x] Ghost model frozen (extraction uses prompts, not fine-tuning)
- [x] Temporal-first design (events require timestamps)

## Next Step

Proceed to **Step 08: Frontend Intelligence Integration**.

## Research References

1. **CausalRAG**: `docs/phase-1/SOTA_RESEARCH_SUMMARY.md` (Paper #4)
2. **Causal-Copilot**: `docs/phase-1/papers/converted/2504.13263v2.md`
3. **Bradford Hill (1965)**: "The Environment and Disease: Association or Causation?"

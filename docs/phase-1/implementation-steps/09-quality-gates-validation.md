# Step 09: Quality Gates Validation

## Status: TODO

## Objective

Validate all quality gates defined in `.cursor/rules/quality-gates.mdc` before marking Phase 1 complete.

## Quality Gates to Validate

### 1. Temporal Extraction Quality
```python
def test_temporal_accuracy():
    """Validate >85% temporal extraction accuracy."""
    accuracy = measure_temporal_accuracy(test_corpus)
    assert accuracy > 0.85

def test_temporal_consistency():
    """Validate 100% temporal consistency."""
    events = extract_events(test_corpus)
    assert validate_temporal_consistency(events)
```

### 2. Schema Evolution Quality
```python
def test_schema_evolution_alignment():
    """Validate >90% schema alignment."""
    alignment = compute_semantic_alignment(evolved_schema, manual_schema)
    assert alignment > 0.90
```

### 3. Extraction Quality
```python
def test_extraction_precision():
    """Validate precision ≥0.8."""
    precision = measure_precision(extraction_pipeline, labeled_corpus)
    assert precision >= 0.8

def test_extraction_recall():
    """Validate recall ≥0.7."""
    recall = measure_recall(extraction_pipeline, labeled_corpus)
    assert recall >= 0.7
```

### 4. Performance Quality
```python
def test_throughput_target():
    """Validate >5 docs/sec throughput."""
    throughput = measure_throughput(extraction_pipeline, 100)
    assert throughput > 5.0

def test_memory_usage():
    """Validate <2GB memory usage."""
    memory_used = measure_memory_usage(extraction_pipeline)
    assert memory_used < 2.0  # GB

def test_search_latency():
    """Validate <1s search latency."""
    latency = measure_search_latency()
    assert latency < 1.0  # seconds
```

### 5. Experiential Learning Quality
```python
def test_ghost_model_frozen():
    """Validate Ghost model parameters unchanged."""
    params_before = get_model_parameters(ghost_model)
    run_experiential_learning(ghost_model, 100)
    params_after = get_model_parameters(ghost_model)
    assert params_before == params_after

def test_quality_progression():
    """Validate quality improves over documents."""
    quality_0_10 = measure_quality(docs[0:10])
    quality_40_50 = measure_quality(docs[40:50])
    assert quality_40_50 > quality_0_10
```

### 6. Causal Structure Quality
```python
def test_event_extraction_accuracy():
    """Validate >80% event extraction accuracy."""
    accuracy = measure_event_extraction_accuracy(test_corpus)
    assert accuracy > 0.80

def test_temporal_ordering_validation():
    """Ensure cause always precedes effect."""
    causal_candidates = extract_causal_candidates(test_corpus)
    assert all(c.temporal_ordering_valid for c in causal_candidates)
```

## Validation Process

1. **Create test corpus** with labeled data
2. **Run all quality gate tests**
3. **Document results** in quality report
4. **Fix any failures** before proceeding
5. **Re-validate** after fixes

## Success Criteria

All gates must be GREEN:
- [ ] Temporal accuracy >85%
- [ ] Schema alignment >90%
- [ ] Extraction precision ≥0.8
- [ ] Extraction recall ≥0.7
- [ ] Throughput >5 docs/sec
- [ ] Memory <2GB
- [ ] Search latency <1s
- [ ] Ghost model frozen
- [ ] Quality progression demonstrated
- [ ] Event extraction >80%
- [ ] Causal ordering 100% valid

## Files to Create

- `tests/quality_gates/test_temporal.py`
- `tests/quality_gates/test_schema.py`
- `tests/quality_gates/test_extraction.py`
- `tests/quality_gates/test_performance.py`
- `tests/quality_gates/test_experiential.py`
- `tests/quality_gates/test_causal.py`

## Dependencies

- **Steps 01-08**: All features complete

## Next Step

Proceed to **Step 10: Production Readiness**.

## Research References

- Quality Gates: `.cursor/rules/quality-gates.mdc`

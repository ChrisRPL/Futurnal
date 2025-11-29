Summary: Comprehensive integration testing for full extraction pipeline with quality validation and production readiness verification.

# 06 · Integration Testing & Production Readiness

## Purpose
Validate the complete entity-relationship extraction pipeline from normalized documents through temporal extraction, schema evolution, experiential learning, thought templates, and causal structure preparation to PKG storage. Ensure production readiness with comprehensive integration tests, performance benchmarks, and quality validation.

## Scope
- End-to-end pipeline integration tests
- Multi-document learning progression validation
- Quality improvement measurement over time
- Error recovery and quarantine workflows
- Performance benchmarks and optimization
- Production deployment readiness

## Requirements Alignment
- **Option B Requirement**: "Production-ready extraction pipeline"
- **Quality Targets**: Precision ≥0.8, Temporal accuracy ≥0.85, Schema alignment ≥0.90
- **Production Gates**: All 10 quality gates validated

## Test Suites

### 1. End-to-End Integration Tests

```python
class TestFullPipeline:
    """End-to-end extraction pipeline tests."""

    def test_normalized_document_to_pkg(self):
        """Validate full pipeline: normalization → extraction → PKG."""
        # Setup
        doc = load_test_document("obsidian_sample.md")
        pipeline = create_extraction_pipeline()

        # Execute full pipeline
        normalized = normalize_document(doc)
        temporal_triples = extract_temporal_info(normalized)
        entities = extract_entities(normalized)
        relationships = extract_relationships(normalized, entities)
        events = extract_events(normalized, temporal_triples)
        causal_candidates = detect_causal_relationships(events)

        # Store in PKG
        pkg.store_triples(entities + relationships + events + causal_candidates)

        # Validate
        assert pkg.count_entities() > 0
        assert pkg.count_relationships() > 0
        assert pkg.count_events() > 0
        assert all(t.timestamp for t in temporal_triples)

    def test_multi_document_learning(self):
        """Validate learning progression across documents."""
        docs = load_test_corpus(50)
        pipeline = create_extraction_pipeline_with_learning()

        # Measure quality at different stages
        precision_0_10 = measure_precision(pipeline, docs[0:10])
        precision_10_20 = measure_precision(pipeline, docs[10:20])
        precision_40_50 = measure_precision(pipeline, docs[40:50])

        # Quality should improve
        assert precision_40_50 > precision_0_10
        assert precision_10_20 > precision_0_10

### 2. Quality Validation Tests

```python
class TestQualityTargets:
    """Validate quality metrics meet targets."""

    def test_temporal_accuracy_target(self):
        """Validate >85% temporal accuracy."""
        docs = load_temporally_labeled_corpus(100)
        pipeline = create_extraction_pipeline()

        accuracy = measure_temporal_accuracy(pipeline, docs)

        assert accuracy > 0.85, f"Temporal accuracy {accuracy} below 85%"

    def test_schema_evolution_alignment(self):
        """Validate >90% schema alignment."""
        manual_schema = load_manual_schema()
        docs = load_diverse_corpus(200)

        evolved_schema = run_schema_evolution(docs)

        alignment = compute_semantic_alignment(evolved_schema, manual_schema)

        assert alignment > 0.90, f"Schema alignment {alignment} below 90%"

    def test_extraction_precision_target(self):
        """Validate ≥0.8 precision."""
        docs = load_labeled_corpus(100)
        pipeline = create_extraction_pipeline()

        precision = measure_extraction_precision(pipeline, docs)

        assert precision >= 0.8, f"Precision {precision} below 0.8"
```

### 3. Error Recovery Tests

```python
class TestErrorRecovery:
    """Validate error handling and quarantine workflows."""

    def test_malformed_document_quarantine(self):
        """Ensure malformed documents go to quarantine."""
        doc = create_malformed_document()
        pipeline = create_extraction_pipeline()

        result = pipeline.process(doc)

        assert result.status == "quarantined"
        assert result.error_message is not None
        assert result.retry_count == 0

    def test_low_confidence_flagging(self):
        """Ensure low-confidence extractions flagged."""
        doc = create_ambiguous_document()
        pipeline = create_extraction_pipeline()

        result = pipeline.process(doc)

        if result.confidence < 0.7:
            assert result.needs_review == True

    def test_quarantine_retry_workflow(self):
        """Validate quarantine retry mechanisms."""
        doc = create_temporarily_failing_document()
        pipeline = create_extraction_pipeline()

        # First attempt - quarantine
        result1 = pipeline.process(doc)
        assert result1.status == "quarantined"

        # Retry after template improvement
        pipeline.update_templates()
        result2 = pipeline.retry_quarantined(doc.id)

        assert result2.status == "success" or result2.retry_count > 0
```

### 4. Performance Benchmarks

```python
class TestPerformance:
    """Performance and scalability tests."""

    def test_throughput_target(self):
        """Validate >5 documents/second throughput."""
        docs = load_test_corpus(100)
        pipeline = create_extraction_pipeline()

        start = time.time()
        for doc in docs:
            pipeline.process(doc)
        elapsed = time.time() - start

        throughput = len(docs) / elapsed
        assert throughput > 5, f"Throughput {throughput} below 5 docs/sec"

    def test_memory_usage(self):
        """Validate memory usage <2GB."""
        large_doc = create_large_document(size_mb=10)
        pipeline = create_extraction_pipeline()

        memory_before = get_memory_usage()
        pipeline.process(large_doc)
        memory_after = get_memory_usage()

        memory_used = (memory_after - memory_before) / (1024 ** 3)  # GB
        assert memory_used < 2, f"Memory usage {memory_used}GB exceeds 2GB"
```

## Production Readiness Checklist

```python
class ProductionReadinessValidation:
    """Comprehensive production readiness validation."""

    def validate_all_gates(self) -> bool:
        """Validate all 10 quality gates."""

        gates = {
            "temporal_extraction": self.validate_temporal_accuracy() > 0.85,
            "schema_evolution": self.validate_schema_alignment() > 0.90,
            "extraction_quality": self.validate_precision() >= 0.8,
            "experiential_learning": self.validate_quality_improvement(),
            "thought_templates": self.validate_template_evolution(),
            "causal_structure": self.validate_event_extraction() > 0.80,
            "performance": self.validate_throughput() > 5,
            "privacy": self.validate_privacy_compliance(),
            "error_handling": self.validate_quarantine_workflow(),
            "integration": self.validate_end_to_end_pipeline()
        }

        return all(gates.values())
```

## Success Metrics

- ✅ All integration tests passing
- ✅ Quality targets met (precision ≥0.8, temporal ≥0.85, schema ≥0.90)
- ✅ Learning improvement demonstrable
- ✅ Performance targets met (>5 docs/sec, <2GB memory)
- ✅ Error recovery functional
- ✅ Production deployment ready

## Dependencies

- All previous modules (01-05)
- Quality gates testing infrastructure
- PKG storage operational
- Orchestrator integration

**This module validates production readiness and ensures all Option B requirements are met.**

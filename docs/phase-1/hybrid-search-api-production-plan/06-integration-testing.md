Summary: Comprehensive integration testing for hybrid search API with relevance validation and production readiness verification.

# 06 · Integration Testing & Production Readiness

## Purpose
Validate the complete hybrid search API from query routing through temporal/causal retrieval, schema-aware fusion, and result ranking to ensure production readiness with relevance targets.

**Criticality**: CRITICAL - Production deployment gate

## Scope
- End-to-end query tests
- Relevance metrics validation (MRR, precision@5)
- Performance benchmarks
- Intent classification accuracy
- Production deployment readiness

## Requirements Alignment
- **Option B Requirement**: "Production-ready search API"
- **Quality Targets**: MRR >0.7, precision@5 >0.8, <1s latency
- **Production Gates**: All 8 quality gates validated

## Test Suites

### 1. End-to-End Query Tests

```python
class TestFullHybridSearch:
    """End-to-end hybrid search tests."""

    def test_temporal_query_flow(self):
        """Validate temporal query routing and execution."""
        api = create_hybrid_search_api()

        query = "What happened between January and March 2024?"

        results = api.search(query, top_k=10)

        assert len(results) > 0
        assert all("timestamp" in r for r in results)

    def test_causal_query_flow(self):
        """Validate causal query routing and execution."""
        api = create_hybrid_search_api()

        query = "What led to the product launch decision?"

        results = api.search(query, top_k=10)

        assert len(results) > 0
        # Verify causal chain present

    def test_exploratory_query_flow(self):
        """Validate exploratory query flow."""
        api = create_hybrid_search_api()

        query = "Tell me about machine learning projects"

        results = api.search(query, top_k=10)

        assert len(results) > 0
```

### 2. Relevance Metrics

```python
class TestRelevanceMetrics:
    """Validate relevance quality."""

    def test_mean_reciprocal_rank(self):
        """Validate MRR >0.7."""
        api = create_hybrid_search_api()
        test_queries = load_golden_query_set()

        mrr = compute_mrr(api, test_queries)

        assert mrr > 0.7

    def test_precision_at_5(self):
        """Validate precision@5 >0.8."""
        api = create_hybrid_search_api()
        test_queries = load_golden_query_set()

        precision = compute_precision_at_k(api, test_queries, k=5)

        assert precision > 0.8
```

### 3. Performance Benchmarks

```python
class TestPerformance:
    """Performance validation."""

    def test_latency_target(self):
        """Validate <1s latency for 95% queries."""
        api = create_hybrid_search_api()

        latencies = []
        for i in range(100):
            start = time.time()
            api.search(f"Query {i}", top_k=10)
            latencies.append(time.time() - start)

        p95 = np.percentile(latencies, 95)

        assert p95 < 1.0
```

## Production Readiness Checklist

```python
class ProductionReadinessValidation:
    """Production readiness validation."""

    def validate_all_gates(self) -> Dict[str, bool]:
        gates = {
            "temporal_queries": self.validate_temporal_functional(),
            "causal_retrieval": self.validate_causal_functional(),
            "schema_aware": self.validate_schema_adaptation(),
            "intent_classification": self.validate_intent_accuracy() > 0.85,
            "relevance_mrr": self.validate_mrr() > 0.7,
            "relevance_precision": self.validate_precision() > 0.8,
            "performance_latency": self.validate_latency() < 1.0,
            "integration": self.validate_end_to_end()
        }
        return gates
```

## Success Metrics

- ✅ All end-to-end tests passing
- ✅ MRR >0.7, precision@5 >0.8
- ✅ Latency <1s for 95% queries
- ✅ Intent classification >85% accuracy
- ✅ Integration complete
- ✅ Production ready

## Dependencies

- All previous modules (01-05)
- Golden query set for relevance testing
- Performance testing infrastructure

**This module validates production readiness for hybrid search API.**

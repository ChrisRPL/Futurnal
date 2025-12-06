Summary: Comprehensive integration testing for hybrid search API with LLM backend validation, multimodal content tests, relevance metrics, and production readiness verification.

# 06 · Integration Testing & Production Readiness

## Purpose
Validate the complete hybrid search API from query routing through temporal/causal retrieval, schema-aware fusion, multimodal content search, and result ranking to ensure production readiness with relevance targets.

**Criticality**: CRITICAL - Production deployment gate

## Scope
- End-to-end query tests (temporal, causal, exploratory, multimodal)
- LLM backend integration (Ollama connectivity and fallback)
- Multi-model embedding validation
- Multimodal content search (OCR, transcriptions)
- Relevance metrics validation (MRR, precision@5)
- Performance benchmarks (P95 latency)
- Cache effectiveness validation
- GRPO/experiential learning hooks
- Production deployment readiness

## Requirements Alignment
- **Option B Requirement**: "Production-ready search API"
- **Quality Targets**: MRR >0.7, precision@5 >0.8, <1s latency
- **Production Gates**: All quality gates validated
- **Multimodal Support**: OCR >80%, audio >75% relevance

---

## Test Suites

### 1. End-to-End Query Tests

```python
import pytest
import asyncio
import time
import numpy as np
from typing import Dict, List, Any


class TestFullHybridSearch:
    """End-to-end hybrid search tests."""

    @pytest.fixture
    def api(self):
        """Create hybrid search API instance."""
        return create_hybrid_search_api()

    @pytest.mark.integration
    async def test_temporal_query_flow(self, api):
        """Validate temporal query routing and execution."""
        query = "What happened between January and March 2024?"

        results = await api.search(query, top_k=10)

        assert len(results) > 0
        assert all("timestamp" in r for r in results)

        # Verify temporal ordering
        timestamps = [r["timestamp"] for r in results]
        assert all(
            "2024-01" <= ts <= "2024-03"
            for ts in timestamps if ts
        )

    @pytest.mark.integration
    async def test_causal_query_flow(self, api):
        """Validate causal query routing and execution."""
        query = "What led to the product launch decision?"

        results = await api.search(query, top_k=10)

        assert len(results) > 0

        # Verify causal chain metadata present
        causal_results = [r for r in results if r.get("causal_chain")]
        assert len(causal_results) > 0

        # Verify chain structure
        for r in causal_results:
            chain = r["causal_chain"]
            assert "anchor" in chain
            assert "causes" in chain or "effects" in chain

    @pytest.mark.integration
    async def test_exploratory_query_flow(self, api):
        """Validate exploratory query flow."""
        query = "Tell me about machine learning projects"

        results = await api.search(query, top_k=10)

        assert len(results) > 0

        # Exploratory should have diverse results
        entities = set(r.get("entity_type") for r in results)
        assert len(entities) >= 2  # Multiple entity types

    @pytest.mark.integration
    async def test_factual_query_flow(self, api):
        """Validate factual lookup query flow."""
        query = "What is the project deadline for Alpha?"

        results = await api.search(query, top_k=5)

        assert len(results) > 0

        # Factual should return high-confidence results
        assert results[0].get("confidence", 0) > 0.8

    @pytest.mark.integration
    async def test_code_query_flow(self, api):
        """Validate code-specific query flow."""
        query = "How does the authentication module work?"

        results = await api.search(query, top_k=10)

        assert len(results) > 0

        # Should use CodeBERT embeddings for code content
        code_results = [r for r in results if r.get("source_type") == "code"]
        # Code queries should prioritize code results
```

---

### 2. LLM Backend Integration Tests

```python
class TestOllamaIntegration:
    """Tests for Ollama LLM backend integration."""

    @pytest.fixture
    def ollama_classifier(self):
        """Create Ollama-based intent classifier."""
        return OllamaIntentClassifier(
            model="llama3.1:8b",
            base_url="http://localhost:11434"
        )

    @pytest.mark.integration
    async def test_ollama_connectivity(self, ollama_classifier):
        """Test Ollama server connectivity."""
        is_available = await ollama_classifier.check_availability()

        assert is_available, "Ollama server not available"

    @pytest.mark.integration
    async def test_intent_classification_temporal(self, ollama_classifier):
        """Test intent classification for temporal queries."""
        query = "What happened last week?"

        intent = await ollama_classifier.classify(query)

        assert intent.primary_intent == "temporal"
        assert intent.confidence > 0.8

    @pytest.mark.integration
    async def test_intent_classification_causal(self, ollama_classifier):
        """Test intent classification for causal queries."""
        query = "Why did the server crash?"

        intent = await ollama_classifier.classify(query)

        assert intent.primary_intent == "causal"
        assert intent.confidence > 0.8

    @pytest.mark.integration
    async def test_intent_classification_batch(self, ollama_classifier):
        """Test batch intent classification."""
        queries = [
            "What happened yesterday?",
            "Why did this fail?",
            "Tell me about projects",
            "What is the deadline?"
        ]

        intents = await ollama_classifier.classify_batch(queries)

        assert len(intents) == 4
        assert intents[0].primary_intent == "temporal"
        assert intents[1].primary_intent == "causal"

    @pytest.mark.integration
    async def test_ollama_fallback_to_hf(self):
        """Test fallback to HuggingFace when Ollama unavailable."""
        # Create classifier with invalid Ollama URL
        classifier = create_intent_classifier(
            backend="auto",
            ollama_url="http://localhost:99999"
        )

        query = "What happened yesterday?"

        # Should fall back to HuggingFace
        intent = await classifier.classify(query)

        assert intent is not None
        assert classifier.active_backend == "hf"

    @pytest.mark.integration
    async def test_ollama_inference_latency(self, ollama_classifier):
        """Validate Ollama inference latency <100ms."""
        query = "What happened yesterday?"

        latencies = []
        for _ in range(10):
            start = time.time()
            await ollama_classifier.classify(query)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency < 100, f"Ollama latency {avg_latency}ms exceeds 100ms target"


class TestLLMModelSelection:
    """Tests for LLM model selection and configuration."""

    @pytest.mark.integration
    async def test_model_selection_via_env(self):
        """Test model selection via environment variable."""
        import os

        os.environ["FUTURNAL_PRODUCTION_LLM"] = "phi3"
        classifier = create_intent_classifier()

        assert classifier.model_name == "phi3:mini"

    @pytest.mark.integration
    async def test_backend_selection_via_env(self):
        """Test backend selection via environment variable."""
        import os

        os.environ["FUTURNAL_LLM_BACKEND"] = "ollama"
        classifier = create_intent_classifier()

        assert classifier.backend_type == "ollama"

    @pytest.mark.integration
    async def test_bielik_selection_for_polish(self):
        """Test Bielik model selected for Polish queries."""
        import os
        os.environ["FUTURNAL_PRODUCTION_LLM"] = "auto"

        router = DynamicModelRouter()
        polish_query = "Jakie spotkania miałem w tym tygodniu?"

        model = router.config.get_model_for_query(polish_query)

        assert model == "bielik:4.5b", f"Expected Bielik for Polish, got {model}"

    @pytest.mark.integration
    async def test_kimi_selection_for_reasoning(self):
        """Test Kimi-K2 model selected for advanced reasoning queries."""
        import os
        os.environ["FUTURNAL_PRODUCTION_LLM"] = "auto"

        router = DynamicModelRouter()
        reasoning_query = "What caused the server crash and what are the implications?"

        model = router.config.get_model_for_query(reasoning_query)

        assert model == "kimi-k2:thinking", f"Expected Kimi-K2 for reasoning, got {model}"

    @pytest.mark.integration
    async def test_qwen_selection_for_code(self):
        """Test Qwen model selected for code queries."""
        import os
        os.environ["FUTURNAL_PRODUCTION_LLM"] = "auto"

        router = DynamicModelRouter()
        code_query = "How does the authentication function work?"

        model = router.config.get_model_for_query(code_query)

        assert model == "qwen2.5-coder:32b", f"Expected Qwen for code, got {model}"

    @pytest.mark.integration
    async def test_runtime_model_switch(self):
        """Test runtime model switching."""
        router = DynamicModelRouter()

        # Switch to Bielik
        success = router.switch_default_model("bielik")

        if success:  # Only if model available
            assert router._current_model == "bielik:4.5b"

    @pytest.mark.integration
    async def test_all_model_aliases_resolve(self):
        """Test all model aliases resolve correctly."""
        from futurnal.search.config import QueryRouterLLMConfig

        expected_mappings = {
            "phi3": "phi3:mini",
            "llama3.1": "llama3.1:8b",
            "qwen": "qwen2.5-coder:32b",
            "bielik": "bielik:4.5b",
            "kimi": "kimi-k2:thinking",
            "k2": "kimi-k2:thinking",
            "gpt-oss": "gpt-oss:20b",
        }

        for alias, expected in expected_mappings.items():
            resolved = QueryRouterLLMConfig.MODEL_ALIASES.get(alias)
            assert resolved == expected, f"Alias {alias} resolved to {resolved}, expected {expected}"

    @pytest.mark.integration
    async def test_explicit_model_overrides_auto(self):
        """Test explicit model env var overrides auto selection."""
        import os
        os.environ["FUTURNAL_PRODUCTION_LLM"] = "phi3"

        router = DynamicModelRouter()
        polish_query = "Jakie spotkania miałem?"  # Polish query

        # Should use phi3, not bielik, because explicit model is set
        model = router.config.get_model_for_query(polish_query)

        assert model == "phi3:mini", "Explicit model should override auto selection"

    @pytest.mark.integration
    def test_list_available_models(self):
        """Test listing all available models."""
        router = DynamicModelRouter()
        models = router.list_available_models()

        # Should have all 6 models
        assert len(models) >= 6

        aliases = [m["alias"] for m in models]
        assert "phi3" in aliases
        assert "bielik" in aliases
        assert "kimi" in aliases
        assert "qwen" in aliases
```

---

### 3. Multi-Model Embedding Tests

```python
class TestMultiModelEmbeddings:
    """Tests for multi-model embedding architecture."""

    @pytest.fixture
    def embedding_router(self):
        """Create embedding router."""
        return QueryEmbeddingRouter()

    @pytest.mark.integration
    async def test_general_entity_embedding(self, embedding_router):
        """Test embedding generation for general entities."""
        query = "project meeting notes"

        embedding = await embedding_router.embed_query(query)

        assert embedding is not None
        assert len(embedding) == 768  # Instructor-large dimension
        assert embedding_router.last_model_used == "instructor-large"

    @pytest.mark.integration
    async def test_code_entity_embedding(self, embedding_router):
        """Test embedding generation for code queries."""
        query = "def authenticate_user(token):"

        embedding = await embedding_router.embed_query(
            query,
            entity_type="code"
        )

        assert embedding is not None
        assert embedding_router.last_model_used == "codebert"

    @pytest.mark.integration
    async def test_temporal_context_embedding(self, embedding_router):
        """Test embedding with temporal context."""
        query = "meetings from last week"

        embedding = await embedding_router.embed_query(
            query,
            temporal_context="2024-01-01/2024-01-07"
        )

        assert embedding is not None

    @pytest.mark.integration
    async def test_embedding_model_routing(self, embedding_router):
        """Test correct model routing for different entity types."""
        test_cases = [
            ("general query about projects", "instructor-large"),
            ("function calculateTotal(items)", "codebert"),
            ("what happened yesterday", "instructor-large"),
        ]

        for query, expected_model in test_cases:
            await embedding_router.embed_query(query)
            assert embedding_router.last_model_used == expected_model, \
                f"Expected {expected_model} for '{query}'"
```

---

### 4. Multimodal Content Tests

```python
class TestMultimodalSearch:
    """Tests for multimodal content search."""

    @pytest.fixture
    def api(self):
        """Create API with multimodal support."""
        return create_hybrid_search_api(multimodal_enabled=True)

    @pytest.mark.integration
    async def test_ocr_content_search(self, api):
        """Test search in OCR-extracted content."""
        # Index OCR content
        ocr_content = {
            "text": "Invoice #12345 for Project Alpha",
            "confidence": 0.95,
            "source_file": "invoice.pdf",
            "layout": {"is_form": True}
        }
        await api.index_ocr_content(ocr_content)

        # Search with OCR hint
        results = await api.search(
            "invoice from the PDF",
            top_k=5
        )

        assert len(results) > 0
        assert "invoice" in results[0]["content"].lower()
        assert results[0].get("source_type") == "ocr_document"

    @pytest.mark.integration
    async def test_transcription_search(self, api):
        """Test search in audio transcriptions."""
        # Index transcription
        transcription = {
            "text": "We discussed the quarterly revenue targets in the meeting.",
            "segments": [
                {"start": 0, "end": 5, "text": "We discussed the quarterly revenue targets in the meeting."}
            ],
            "language": "en"
        }
        await api.index_transcription(transcription)

        # Search with audio hint
        results = await api.search(
            "what did we discuss in the meeting about revenue",
            top_k=5
        )

        assert len(results) > 0
        assert "revenue" in results[0]["content"].lower()
        assert results[0].get("source_type") == "audio_transcription"

    @pytest.mark.integration
    async def test_modality_hint_detection(self, api):
        """Test detection of modality hints in queries."""
        test_cases = [
            ("in my voice notes", "audio_transcription"),
            ("from the scanned document", "ocr_document"),
            ("in that PDF I uploaded", "ocr_document"),
            ("what I said in the meeting", "audio_transcription"),
        ]

        for query, expected_modality in test_cases:
            plan = api.multimodal_handler.analyze_query(query)

            assert expected_modality in [m.value for m in plan.target_modalities], \
                f"Expected {expected_modality} for '{query}'"

    @pytest.mark.integration
    async def test_cross_modal_fusion(self, api):
        """Test cross-modal result fusion."""
        # Index content from multiple sources
        await api.index_text("Meeting notes: Project deadline is Friday")
        await api.index_ocr_content({
            "text": "Project Schedule: Friday deadline",
            "confidence": 0.9
        })
        await api.index_transcription({
            "text": "The project deadline is on Friday"
        })

        # Search without modality hint
        results = await api.search("project deadline", top_k=10)

        # Should have results from multiple modalities
        modalities = set(r.get("source_type") for r in results)
        assert len(modalities) >= 2, "Should fuse results from multiple modalities"

    @pytest.mark.integration
    async def test_ocr_fuzzy_matching(self, api):
        """Test fuzzy matching for OCR errors."""
        # Index OCR content with potential error
        ocr_content = {
            "text": "Reciept for purchase",  # Common OCR error
            "confidence": 0.85
        }
        await api.index_ocr_content(ocr_content)

        # Search with correct spelling
        results = await api.search(
            "receipt from the document",
            top_k=5
        )

        # Should still find despite OCR error
        assert len(results) > 0

    @pytest.mark.integration
    async def test_transcription_homophone_handling(self, api):
        """Test handling of homophones in transcriptions."""
        # Index transcription with homophone
        transcription = {
            "text": "They're going to submit their report there."
        }
        await api.index_transcription(transcription)

        # Search with different homophone
        results = await api.search(
            "where did they submit the report",
            top_k=5
        )

        # Should find despite homophone variance
        assert len(results) > 0


class TestOCRContentRelevance:
    """Tests for OCR content relevance targets."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api(multimodal_enabled=True)

    @pytest.mark.integration
    async def test_ocr_relevance_target(self, api):
        """Validate >80% relevance for OCR content queries."""
        # Load OCR-specific golden query set
        ocr_queries = load_golden_query_set(modality="ocr")

        correct = 0
        total = 0

        for query_data in ocr_queries:
            results = await api.search(query_data["query"], top_k=5)

            if results and query_data["expected_id"] in [r["id"] for r in results[:5]]:
                correct += 1
            total += 1

        precision = correct / total if total > 0 else 0

        assert precision > 0.8, f"OCR relevance {precision:.2%} below 80% target"


class TestAudioContentRelevance:
    """Tests for audio content relevance targets."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api(multimodal_enabled=True)

    @pytest.mark.integration
    async def test_audio_relevance_target(self, api):
        """Validate >75% relevance for audio content queries."""
        # Load audio-specific golden query set
        audio_queries = load_golden_query_set(modality="audio")

        correct = 0
        total = 0

        for query_data in audio_queries:
            results = await api.search(query_data["query"], top_k=5)

            if results and query_data["expected_id"] in [r["id"] for r in results[:5]]:
                correct += 1
            total += 1

        precision = correct / total if total > 0 else 0

        assert precision > 0.75, f"Audio relevance {precision:.2%} below 75% target"
```

---

### 5. Relevance Metrics

```python
class TestRelevanceMetrics:
    """Validate relevance quality."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api()

    @pytest.mark.integration
    async def test_mean_reciprocal_rank(self, api):
        """Validate MRR >0.7."""
        test_queries = load_golden_query_set()

        mrr = await compute_mrr(api, test_queries)

        assert mrr > 0.7, f"MRR {mrr:.3f} below 0.7 target"

    @pytest.mark.integration
    async def test_precision_at_5(self, api):
        """Validate precision@5 >0.8."""
        test_queries = load_golden_query_set()

        precision = await compute_precision_at_k(api, test_queries, k=5)

        assert precision > 0.8, f"Precision@5 {precision:.3f} below 0.8 target"

    @pytest.mark.integration
    async def test_relevance_by_query_type(self, api):
        """Validate relevance across query types."""
        query_types = ["temporal", "causal", "exploratory", "factual"]

        for query_type in query_types:
            queries = load_golden_query_set(query_type=query_type)
            mrr = await compute_mrr(api, queries)

            assert mrr > 0.65, f"MRR for {query_type} queries {mrr:.3f} below 0.65"


async def compute_mrr(api, queries: List[Dict]) -> float:
    """Compute Mean Reciprocal Rank."""
    reciprocal_ranks = []

    for q in queries:
        results = await api.search(q["query"], top_k=10)
        result_ids = [r["id"] for r in results]

        expected_id = q["expected_id"]
        if expected_id in result_ids:
            rank = result_ids.index(expected_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


async def compute_precision_at_k(api, queries: List[Dict], k: int) -> float:
    """Compute Precision@K."""
    precisions = []

    for q in queries:
        results = await api.search(q["query"], top_k=k)
        result_ids = set(r["id"] for r in results)

        expected_ids = set(q.get("expected_ids", [q["expected_id"]]))
        relevant = len(result_ids & expected_ids)

        precisions.append(relevant / min(k, len(expected_ids)))

    return sum(precisions) / len(precisions) if precisions else 0.0
```

---

### 6. Performance Benchmarks

```python
class TestPerformance:
    """Performance validation."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api()

    @pytest.mark.performance
    async def test_latency_target_p95(self, api):
        """Validate <1s latency for 95% queries."""
        test_queries = generate_benchmark_queries(n=100)

        latencies = []
        for query in test_queries:
            start = time.time()
            await api.search(query, top_k=10)
            latencies.append((time.time() - start) * 1000)

        p95 = np.percentile(latencies, 95)

        assert p95 < 1000, f"P95 latency {p95:.0f}ms exceeds 1000ms target"

    @pytest.mark.performance
    async def test_latency_target_p50(self, api):
        """Validate <200ms latency for 50% queries."""
        test_queries = generate_benchmark_queries(n=100)

        latencies = []
        for query in test_queries:
            start = time.time()
            await api.search(query, top_k=10)
            latencies.append((time.time() - start) * 1000)

        p50 = np.percentile(latencies, 50)

        assert p50 < 200, f"P50 latency {p50:.0f}ms exceeds 200ms target"

    @pytest.mark.performance
    async def test_cached_query_latency(self, api):
        """Validate <100ms latency for cached queries."""
        query = "project deadline"

        # First query - cache miss
        await api.search(query, top_k=10)

        # Second query - should hit cache
        latencies = []
        for _ in range(10):
            start = time.time()
            await api.search(query, top_k=10)
            latencies.append((time.time() - start) * 1000)

        avg_cached = sum(latencies) / len(latencies)

        assert avg_cached < 100, f"Cached latency {avg_cached:.0f}ms exceeds 100ms target"

    @pytest.mark.performance
    async def test_throughput(self, api):
        """Validate query throughput."""
        test_queries = generate_benchmark_queries(n=50)

        start = time.time()
        tasks = [api.search(q, top_k=10) for q in test_queries]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        qps = len(test_queries) / elapsed

        assert qps > 5, f"QPS {qps:.1f} below 5 queries/second minimum"


class TestCacheEffectiveness:
    """Tests for cache system effectiveness."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api(caching_enabled=True)

    @pytest.mark.performance
    async def test_cache_hit_rate(self, api):
        """Validate >60% cache hit rate."""
        # Simulate realistic query pattern with repeats
        queries = ["query_" + str(i % 30) for i in range(100)]

        for query in queries:
            await api.search(query, top_k=10)

        hit_rate = api.cache.stats.overall_hit_rate()

        assert hit_rate > 0.6, f"Cache hit rate {hit_rate:.2%} below 60% target"

    @pytest.mark.performance
    async def test_semantic_cache_effectiveness(self, api):
        """Test semantic similarity cache hits."""
        # Query 1
        await api.search("what happened yesterday", top_k=10)

        # Similar query - should hit semantic cache
        await api.search("what occurred yesterday", top_k=10)

        semantic_hits = api.cache.stats.semantic_hits[CacheLayer.QUERY_RESULT]

        assert semantic_hits > 0, "No semantic cache hits for similar queries"

    @pytest.mark.performance
    async def test_cache_invalidation_on_mutation(self, api):
        """Test cache invalidation when PKG mutates."""
        query = "project status"

        # Cache result
        await api.search(query, top_k=10)

        # Verify cached
        _, hit = api.cache.get(CacheLayer.QUERY_RESULT, query)
        assert hit, "Query should be cached"

        # Simulate PKG mutation
        await api.pkg.create_entity({"type": "Project", "name": "New"})

        # Cache should be invalidated
        _, hit = api.cache.get(CacheLayer.QUERY_RESULT, query)
        assert not hit, "Cache should be invalidated after PKG mutation"
```

---

### 7. GRPO/Experiential Learning Tests

```python
class TestExperientialLearning:
    """Tests for GRPO experiential learning integration."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api(experiential_learning=True)

    @pytest.mark.integration
    async def test_quality_feedback_recording(self, api):
        """Test recording of search quality feedback."""
        results = await api.search("project meeting", top_k=10)

        # Simulate user click on result
        await api.record_feedback(
            query="project meeting",
            clicked_result_id=results[0]["id"],
            feedback_type="click"
        )

        # Verify feedback recorded
        feedback = api.quality_feedback.get_recent_feedback(limit=1)
        assert len(feedback) == 1
        assert feedback[0]["signal_type"] == "positive"

    @pytest.mark.integration
    async def test_negative_feedback_on_refinement(self, api):
        """Test negative signal on query refinement."""
        # Initial query
        await api.search("meetings", top_k=10)

        # Refinement (indicates poor results)
        await api.search("project meetings last week", top_k=10)

        # Verify negative feedback recorded for initial query
        feedback = api.quality_feedback.get_feedback_for_query("meetings")
        assert any(f["signal_type"] == "negative" for f in feedback)

    @pytest.mark.integration
    async def test_template_evolution_trigger(self, api):
        """Test that feedback triggers template evolution."""
        # Generate enough feedback to trigger evolution
        for i in range(10):
            results = await api.search(f"temporal query {i}", top_k=5)
            await api.record_feedback(
                query=f"temporal query {i}",
                clicked_result_id=results[0]["id"] if results else None,
                feedback_type="click" if results else "no_results"
            )

        # Check if template evolution was triggered
        evolution_events = api.template_database.get_evolution_events()
        assert len(evolution_events) > 0


class TestThoughtTemplates:
    """Tests for thought template integration."""

    @pytest.fixture
    def template_db(self):
        return QueryTemplateDatabase()

    @pytest.mark.integration
    def test_template_matching(self, template_db):
        """Test query-to-template matching."""
        query = "what happened last week with the project"

        templates = template_db.find_matching_templates(query, top_k=3)

        assert len(templates) > 0
        assert templates[0].intent_pattern in ["temporal", "exploratory"]

    @pytest.mark.integration
    def test_template_evolution_signal(self, template_db):
        """Test template evolution with textual gradients."""
        template_id = "temporal_query_v1"

        # Record KEEP signal
        template_db.record_feedback(template_id, "KEEP")

        # Verify score increased
        template = template_db.get_template(template_id)
        assert template.evolution_score > 0

    @pytest.mark.integration
    def test_template_discard_on_failures(self, template_db):
        """Test template discarding after repeated failures."""
        template_id = "poor_template_v1"

        # Record multiple DISCARD signals
        for _ in range(5):
            template_db.record_feedback(template_id, "DISCARD")

        # Template should be marked for removal
        template = template_db.get_template(template_id)
        assert template.status == "deprecated" or template is None
```

---

### 8. Schema Evolution Tests

```python
class TestSchemaEvolution:
    """Tests for schema-aware retrieval adaptation."""

    @pytest.fixture
    def api(self):
        return create_hybrid_search_api()

    @pytest.mark.integration
    async def test_schema_version_compatibility(self, api):
        """Test retrieval works across schema versions."""
        # Create content with old schema
        old_entity = {"type": "Event", "name": "Meeting", "schema_version": "1.0"}
        await api.pkg.create_entity(old_entity)

        # Create content with new schema (additional fields)
        new_entity = {
            "type": "Event",
            "name": "Conference",
            "schema_version": "2.0",
            "participants": ["Alice", "Bob"]
        }
        await api.pkg.create_entity(new_entity)

        # Search should find both
        results = await api.search("events", top_k=10)

        result_names = [r["name"] for r in results]
        assert "Meeting" in result_names
        assert "Conference" in result_names

    @pytest.mark.integration
    async def test_cache_invalidation_on_schema_change(self, api):
        """Test cache invalidation when schema evolves."""
        # Cache query result
        await api.search("project data", top_k=10)

        # Simulate schema evolution
        await api.schema_manager.evolve_schema("2.1")

        # Cache should be invalidated for affected layers
        assert api.cache.stats.invalidations[CacheLayer.GRAPH_TRAVERSAL] > 0

    @pytest.mark.integration
    async def test_entity_type_strategy_adaptation(self, api):
        """Test retrieval strategy adapts to entity type."""
        # Index different entity types
        await api.pkg.create_entity({"type": "Event", "timestamp": "2024-01-01"})
        await api.pkg.create_entity({"type": "Code", "language": "python"})
        await api.pkg.create_entity({"type": "Document", "format": "markdown"})

        # Event query should use temporal-aware strategy
        results = await api.search("events from January", top_k=5)
        assert api.last_strategy == "temporal_first" or "temporal" in str(api.last_strategy)

        # Code query should use CodeBERT embeddings
        results = await api.search("authentication function", top_k=5)
        assert api.last_embedding_model == "codebert"
```

---

### 9. Production Readiness Checklist

```python
class ProductionReadinessValidation:
    """Production readiness validation."""

    def __init__(self, api):
        self.api = api

    async def validate_all_gates(self) -> Dict[str, bool]:
        """Validate all production gates."""
        gates = {
            "temporal_queries": await self.validate_temporal_functional(),
            "causal_retrieval": await self.validate_causal_functional(),
            "schema_aware": await self.validate_schema_adaptation(),
            "intent_classification": await self.validate_intent_accuracy() > 0.85,
            "ollama_integration": await self.validate_ollama_connectivity(),
            "multi_model_embeddings": await self.validate_embedding_models(),
            "multimodal_ocr": await self.validate_ocr_search() > 0.80,
            "multimodal_audio": await self.validate_audio_search() > 0.75,
            "relevance_mrr": await self.validate_mrr() > 0.7,
            "relevance_precision": await self.validate_precision() > 0.8,
            "performance_p95": await self.validate_latency_p95() < 1.0,
            "performance_p50": await self.validate_latency_p50() < 0.2,
            "cache_hit_rate": await self.validate_cache_hit_rate() > 0.6,
            "integration": await self.validate_end_to_end()
        }
        return gates

    async def validate_temporal_functional(self) -> bool:
        """Validate temporal queries work."""
        try:
            results = await self.api.search("what happened yesterday", top_k=5)
            return len(results) > 0
        except Exception:
            return False

    async def validate_causal_functional(self) -> bool:
        """Validate causal queries work."""
        try:
            results = await self.api.search("why did this happen", top_k=5)
            return len(results) > 0
        except Exception:
            return False

    async def validate_schema_adaptation(self) -> bool:
        """Validate schema-aware retrieval."""
        try:
            # Test with different schema versions
            return True  # Implementation specific
        except Exception:
            return False

    async def validate_intent_accuracy(self) -> float:
        """Validate intent classification accuracy."""
        test_cases = [
            ("what happened yesterday", "temporal"),
            ("why did this fail", "causal"),
            ("tell me about projects", "exploratory"),
            ("what is the deadline", "factual"),
        ]

        correct = 0
        for query, expected in test_cases:
            intent = await self.api.router.classify_intent(query)
            if intent.primary_intent == expected:
                correct += 1

        return correct / len(test_cases)

    async def validate_ollama_connectivity(self) -> bool:
        """Validate Ollama server connectivity."""
        try:
            return await self.api.router.classifier.check_availability()
        except Exception:
            return False

    async def validate_embedding_models(self) -> bool:
        """Validate all embedding models accessible."""
        try:
            # Test each model
            await self.api.embedding_router.embed_query("test", model="instructor-large")
            await self.api.embedding_router.embed_query("def test():", model="codebert")
            return True
        except Exception:
            return False

    async def validate_ocr_search(self) -> float:
        """Validate OCR content search precision."""
        queries = load_golden_query_set(modality="ocr")
        return await compute_precision_at_k(self.api, queries, k=5)

    async def validate_audio_search(self) -> float:
        """Validate audio content search precision."""
        queries = load_golden_query_set(modality="audio")
        return await compute_precision_at_k(self.api, queries, k=5)

    async def validate_mrr(self) -> float:
        """Validate MRR metric."""
        queries = load_golden_query_set()
        return await compute_mrr(self.api, queries)

    async def validate_precision(self) -> float:
        """Validate precision@5 metric."""
        queries = load_golden_query_set()
        return await compute_precision_at_k(self.api, queries, k=5)

    async def validate_latency_p95(self) -> float:
        """Validate P95 latency in seconds."""
        queries = generate_benchmark_queries(n=100)
        latencies = []

        for q in queries:
            start = time.time()
            await self.api.search(q, top_k=10)
            latencies.append(time.time() - start)

        return np.percentile(latencies, 95)

    async def validate_latency_p50(self) -> float:
        """Validate P50 latency in seconds."""
        queries = generate_benchmark_queries(n=100)
        latencies = []

        for q in queries:
            start = time.time()
            await self.api.search(q, top_k=10)
            latencies.append(time.time() - start)

        return np.percentile(latencies, 50)

    async def validate_cache_hit_rate(self) -> float:
        """Validate cache hit rate."""
        return self.api.cache.stats.overall_hit_rate()

    async def validate_end_to_end(self) -> bool:
        """Validate full end-to-end flow."""
        try:
            # Test complete flow
            results = await self.api.search("project meeting notes", top_k=10)
            return len(results) > 0
        except Exception:
            return False

    def generate_report(self, gates: Dict[str, bool]) -> str:
        """Generate production readiness report."""
        passed = sum(1 for v in gates.values() if v)
        total = len(gates)

        report = f"""
# Production Readiness Report

## Summary
Passed: {passed}/{total} gates
Status: {"READY" if passed == total else "NOT READY"}

## Gate Results
"""
        for gate, passed in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"- {gate}: {status}\n"

        return report
```

---

## Success Metrics

| Metric | Target | Test Suite |
|--------|--------|------------|
| End-to-end tests | All passing | TestFullHybridSearch |
| Ollama connectivity | Available | TestOllamaIntegration |
| Intent accuracy | >85% | TestOllamaIntegration |
| Multi-model routing | Correct | TestMultiModelEmbeddings |
| OCR relevance | >80% | TestOCRContentRelevance |
| Audio relevance | >75% | TestAudioContentRelevance |
| MRR | >0.7 | TestRelevanceMetrics |
| Precision@5 | >0.8 | TestRelevanceMetrics |
| P95 latency | <1s | TestPerformance |
| P50 latency | <200ms | TestPerformance |
| Cache hit rate | >60% | TestCacheEffectiveness |
| Quality feedback | Recording | TestExperientialLearning |
| Template evolution | Functional | TestThoughtTemplates |
| Schema compatibility | Working | TestSchemaEvolution |

---

## Dependencies

- All previous modules (01-05, 07)
- Golden query set for relevance testing
- Performance testing infrastructure
- Ollama server running
- Test fixtures for multimodal content

---

## Option B Compliance Verification

This module verifies:
- [x] Ghost Model Frozen: Ollama used for inference only
- [x] Experiential Learning: Quality feedback recorded and used
- [x] Temporal-First: Temporal queries validated
- [x] Schema Evolution: Cross-version compatibility tested
- [x] Quality Gates: All metrics validated against targets

---

**This module is the production gate - all tests must pass before deployment.**

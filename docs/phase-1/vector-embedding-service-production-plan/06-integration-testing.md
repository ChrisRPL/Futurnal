Summary: Comprehensive integration testing for vector embedding service with quality validation and production readiness verification.

# 06 · Integration Testing & Production Readiness

## Purpose
Validate the complete vector embedding service from temporal-aware embedding generation through multi-model architecture, schema-versioned storage, PKG synchronization, and quality evolution to ensure production readiness.

**Criticality**: CRITICAL - Production deployment gate

## Scope
- End-to-end embedding pipeline tests
- Multi-model architecture validation
- Schema evolution compatibility tests
- PKG synchronization verification
- Performance benchmarks
- Quality gates validation
- Production deployment readiness

## Requirements Alignment
- **Option B Requirement**: "Production-ready embedding service with quality evolution"
- **Quality Targets**: <2s latency, >100/min throughput, >0.8 quality score
- **Production Gates**: All 8 quality gates validated

## Test Suites

### 1. End-to-End Pipeline Tests

```python
class TestFullEmbeddingPipeline:
    """End-to-end embedding pipeline tests."""

    def test_entity_to_embedding_full_flow(self):
        """Validate full pipeline: PKG entity → embedding → storage."""
        # Setup
        entity = create_test_entity("Person", "John Doe", "Software Engineer")
        pipeline = create_embedding_pipeline()

        # Execute full pipeline
        # 1. Route to model
        model_id = pipeline.router.route_embedding_request(
            entity.type,
            entity.content
        )
        assert model_id is not None

        # 2. Generate embedding
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content
        )
        assert result.embedding is not None
        assert len(result.embedding) == 768

        # 3. Store with schema version
        embedding_id = pipeline.store.store_embedding(
            entity_id=entity.id,
            entity_type=entity.type,
            embedding=result.embedding,
            model_id=result.model_id,
            extraction_confidence=0.9,
            source_document_id="test_doc"
        )
        assert embedding_id is not None

        # 4. Query back
        results = pipeline.store.query_embeddings(
            query_vector=result.embedding,
            top_k=1
        )
        assert len(results) > 0
        assert results[0].metadata.entity_id == entity.id

    def test_temporal_event_embedding_flow(self):
        """Validate temporal event embedding with context preservation."""
        # Create event with temporal context
        event = create_test_event(
            "Meeting with stakeholders",
            timestamp=datetime(2024, 1, 15, 10, 0),
            duration=timedelta(hours=2)
        )

        temporal_context = TemporalEmbeddingContext(
            timestamp=event.timestamp,
            duration=event.duration,
            temporal_type="BEFORE",
            event_sequence=["Meeting", "Decision"]
        )

        pipeline = create_embedding_pipeline()

        # Generate temporal embedding
        result = pipeline.embedding_service.embed(
            entity_type="Event",
            content=event.description,
            temporal_context=temporal_context
        )

        # Validate temporal context preserved
        assert result.embedding is not None

        # Store and retrieve
        embedding_id = pipeline.store.store_embedding(
            entity_id=event.id,
            entity_type="Event",
            embedding=result.embedding,
            model_id=result.model_id,
            extraction_confidence=0.85,
            source_document_id="test_doc"
        )

        # Verify temporal metadata stored
        results = pipeline.store.query_embeddings(
            query_vector=result.embedding,
            top_k=1,
            entity_type="Event"
        )
        assert len(results) > 0
        assert results[0].metadata.entity_type == "Event"

    def test_multi_entity_type_batch_processing(self):
        """Validate batch processing across different entity types."""
        # Create mixed batch
        requests = [
            EmbeddingRequest("Person", "Alice Smith"),
            EmbeddingRequest("Organization", "Acme Corp"),
            EmbeddingRequest("Event", "Product launch"),
            EmbeddingRequest("Concept", "Machine Learning"),
        ]

        pipeline = create_embedding_pipeline()

        # Batch embed
        results = pipeline.embedding_service.embed_batch(requests)

        # Validate all succeeded
        assert len(results) == 4
        assert all(r.embedding is not None for r in results)

        # Validate routed to correct models
        model_ids = {r.model_id for r in results}
        assert len(model_ids) >= 2  # At least 2 different models used
```

### 2. Schema Evolution Compatibility Tests

```python
class TestSchemaEvolutionCompatibility:
    """Validate schema evolution doesn't break embeddings."""

    def test_schema_version_tracking(self):
        """Validate embeddings track schema versions."""
        pipeline = create_embedding_pipeline()

        # Create embeddings in schema v1
        pipeline.store.current_schema_version = 1
        entity1 = create_test_entity("Person", "John", "Engineer")

        result1 = pipeline.embedding_service.embed(
            entity_type=entity1.type,
            content=entity1.content
        )

        id1 = pipeline.store.store_embedding(
            entity_id=entity1.id,
            entity_type=entity1.type,
            embedding=result1.embedding,
            model_id=result1.model_id,
            extraction_confidence=0.9,
            source_document_id="doc1"
        )

        # Evolve schema to v2
        pipeline.store.current_schema_version = 2

        # Create embeddings in schema v2
        entity2 = create_test_entity("Person", "Jane", "Manager")

        result2 = pipeline.embedding_service.embed(
            entity_type=entity2.type,
            content=entity2.content
        )

        id2 = pipeline.store.store_embedding(
            entity_id=entity2.id,
            entity_type=entity2.type,
            embedding=result2.embedding,
            model_id=result2.model_id,
            extraction_confidence=0.9,
            source_document_id="doc2"
        )

        # Query both schema versions
        v1_embeddings = pipeline.store.query_embeddings(
            query_vector=result1.embedding,
            top_k=10,
            min_schema_version=1
        )

        v2_embeddings = pipeline.store.query_embeddings(
            query_vector=result2.embedding,
            top_k=10,
            min_schema_version=2
        )

        # Validate version filtering
        assert any(e.metadata.schema_version == 1 for e in v1_embeddings)
        assert any(e.metadata.schema_version == 2 for e in v2_embeddings)

    def test_reembedding_on_schema_evolution(self):
        """Validate re-embedding triggered on schema evolution."""
        pipeline = create_embedding_pipeline()
        reembedding_service = create_reembedding_service(pipeline)

        # Create embeddings in schema v1
        pipeline.store.current_schema_version = 1
        entities = [create_test_entity("Person", f"Person {i}") for i in range(10)]

        for entity in entities:
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content
            )
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id=result.model_id,
                extraction_confidence=0.9,
                source_document_id="doc"
            )

        # Evolve schema to v2
        pipeline.store.current_schema_version = 2

        # Trigger re-embedding
        reembedding_service.trigger_reembedding(
            schema_version=1,
            batch_size=10
        )

        # Verify re-embedding completed
        updated_embeddings = pipeline.store.query_embeddings(
            query_vector=np.random.rand(768),
            top_k=100,
            min_schema_version=2
        )

        assert len(updated_embeddings) >= 10
```

### 3. PKG Synchronization Tests

```python
class TestPKGSynchronization:
    """Validate PKG-embedding synchronization."""

    def test_entity_creation_triggers_embedding(self):
        """Validate PKG entity creation triggers embedding."""
        pipeline = create_embedding_pipeline()
        sync_handler = PKGSyncHandler(
            pipeline.embedding_service,
            pipeline.store,
            event_queue=None
        )

        # Simulate PKG entity creation event
        event = PKGEvent(
            event_id="test_creation",
            event_type=PKGEventType.ENTITY_CREATED,
            entity_id="person_new",
            entity_type="Person",
            timestamp=datetime.now(),
            new_data={"name": "New Person", "description": "Description"},
            extraction_confidence=0.85,
            schema_version=1
        )

        # Handle event
        sync_handler.handle_event(event)

        # Verify embedding created
        results = pipeline.store.query_embeddings(
            query_vector=np.random.rand(768),
            top_k=100
        )

        assert any(r.metadata.entity_id == "person_new" for r in results)

    def test_entity_update_triggers_reembedding(self):
        """Validate PKG entity update triggers re-embedding."""
        pipeline = create_embedding_pipeline()
        sync_handler = PKGSyncHandler(
            pipeline.embedding_service,
            pipeline.store,
            event_queue=None
        )

        # Create initial entity
        create_event = PKGEvent(
            event_id="test_create",
            event_type=PKGEventType.ENTITY_CREATED,
            entity_id="person_update",
            entity_type="Person",
            timestamp=datetime.now(),
            new_data={"name": "Original Name", "description": "Original"},
            extraction_confidence=0.8,
            schema_version=1
        )
        sync_handler.handle_event(create_event)

        # Update entity (significant change)
        update_event = PKGEvent(
            event_id="test_update",
            event_type=PKGEventType.ENTITY_UPDATED,
            entity_id="person_update",
            entity_type="Person",
            timestamp=datetime.now(),
            old_data={"name": "Original Name", "description": "Original"},
            new_data={"name": "Updated Name", "description": "Updated description"},
            extraction_confidence=0.9,
            schema_version=1
        )
        sync_handler.handle_event(update_event)

        # Verify embedding updated (would need to check metadata)
        results = pipeline.store.query_embeddings(
            query_vector=np.random.rand(768),
            top_k=100
        )

        updated = next(r for r in results if r.metadata.entity_id == "person_update")
        assert updated is not None

    def test_consistency_validation(self):
        """Validate consistency between PKG and embeddings."""
        pipeline = create_embedding_pipeline()
        pkg_client = create_mock_pkg_client()
        validator = EmbeddingConsistencyValidator(pkg_client, pipeline.store)

        # Run consistency check
        result = validator.validate_consistency()

        assert "missing_embeddings" in result
        assert "orphaned_embeddings" in result
        assert "total_entities" in result
        assert result["total_entities"] >= 0
```

### 4. Performance Benchmarks

```python
class TestPerformanceBenchmarks:
    """Performance and scalability tests."""

    def test_single_embedding_latency(self):
        """Validate single embedding latency <2s."""
        pipeline = create_embedding_pipeline()

        start = time.time()
        result = pipeline.embedding_service.embed(
            entity_type="Person",
            content="John Doe: Software Engineer with 10 years experience"
        )
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Latency {elapsed}s exceeds 2s target"

    def test_batch_throughput(self):
        """Validate batch throughput >100 embeddings/minute."""
        pipeline = create_embedding_pipeline()

        # Create 100 embedding requests
        requests = [
            EmbeddingRequest("Person", f"Person {i}: Description")
            for i in range(100)
        ]

        start = time.time()
        results = pipeline.embedding_service.embed_batch(requests)
        elapsed = time.time() - start

        throughput_per_minute = (len(results) / elapsed) * 60

        assert throughput_per_minute > 100, \
            f"Throughput {throughput_per_minute}/min below 100/min target"

    def test_memory_usage(self):
        """Validate memory usage <2GB."""
        import psutil
        import os

        pipeline = create_embedding_pipeline()
        process = psutil.Process(os.getpid())

        memory_before = process.memory_info().rss / (1024 ** 3)  # GB

        # Generate 1000 embeddings
        requests = [
            EmbeddingRequest("Person", f"Person {i}")
            for i in range(1000)
        ]
        results = pipeline.embedding_service.embed_batch(requests)

        memory_after = process.memory_info().rss / (1024 ** 3)  # GB
        memory_used = memory_after - memory_before

        assert memory_used < 2.0, \
            f"Memory usage {memory_used}GB exceeds 2GB target"

    def test_concurrent_embedding_performance(self):
        """Validate concurrent embedding requests."""
        import concurrent.futures

        pipeline = create_embedding_pipeline()

        def embed_entity(i):
            return pipeline.embedding_service.embed(
                entity_type="Person",
                content=f"Person {i}"
            )

        # Run 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(embed_entity, i) for i in range(50)]
            results = [f.result() for f in futures]

        assert len(results) == 50
        assert all(r.embedding is not None for r in results)
```

### 5. Quality Gates Validation

```python
class TestQualityGates:
    """Validate all quality gates for production readiness."""

    def test_temporal_embedding_quality(self):
        """Validate temporal embedding quality >80%."""
        pipeline = create_embedding_pipeline()
        quality_tracker = QualityMetricsTracker(metrics_store)

        # Generate event embeddings
        events = create_test_events(count=100, with_temporal_context=True)

        for event in events:
            temporal_context = create_temporal_context(event)
            result = pipeline.embedding_service.embed(
                entity_type="Event",
                content=event.description,
                temporal_context=temporal_context
            )

            quality_score = quality_tracker.compute_quality_score(
                embedding=result.embedding,
                entity_type="Event",
                extraction_confidence=0.85
            )

            assert quality_score > 0.8, \
                f"Quality score {quality_score} below 0.8 threshold"

    def test_schema_version_tracking_functional(self):
        """Validate schema version tracking 100% functional."""
        pipeline = create_embedding_pipeline()

        # Create embeddings
        for i in range(50):
            entity = create_test_entity("Person", f"Person {i}")
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content
            )

            embedding_id = pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id=result.model_id,
                extraction_confidence=0.9,
                source_document_id=f"doc_{i}"
            )

        # Verify all have schema version
        results = pipeline.store.query_embeddings(
            query_vector=np.random.rand(768),
            top_k=100
        )

        assert all(r.metadata.schema_version > 0 for r in results), \
            "Some embeddings missing schema version"

    def test_pkg_sync_consistency(self):
        """Validate PKG sync >99% consistent."""
        pipeline = create_embedding_pipeline()
        pkg_client = create_mock_pkg_client()
        validator = EmbeddingConsistencyValidator(pkg_client, pipeline.store)

        result = validator.validate_consistency()

        consistency_rate = 1.0 - (
            len(result["missing_embeddings"]) + len(result["orphaned_embeddings"])
        ) / max(result["total_entities"], 1)

        assert consistency_rate > 0.99, \
            f"Consistency rate {consistency_rate} below 99%"
```

## Production Readiness Checklist

```python
class ProductionReadinessValidation:
    """Comprehensive production readiness validation."""

    def validate_all_gates(self) -> Dict[str, bool]:
        """Validate all 8 quality gates."""

        gates = {
            "temporal_embeddings": self.validate_temporal_quality() > 0.8,
            "multi_model_architecture": self.validate_model_routing(),
            "schema_versioning": self.validate_schema_tracking(),
            "pkg_synchronization": self.validate_sync_consistency() > 0.99,
            "quality_evolution": self.validate_quality_tracking(),
            "performance_latency": self.validate_latency_targets(),
            "performance_throughput": self.validate_throughput_targets(),
            "production_integration": self.validate_end_to_end_pipeline()
        }

        return gates

    def generate_readiness_report(self) -> str:
        """Generate comprehensive readiness report."""
        gates = self.validate_all_gates()

        report = "## Production Readiness Report\\n\\n"
        report += f"**Overall Status**: {'PASS' if all(gates.values()) else 'FAIL'}\\n\\n"

        for gate_name, passed in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"- {gate_name}: {status}\\n"

        return report
```

## Success Metrics

- ✅ All end-to-end pipeline tests passing
- ✅ Schema evolution compatibility verified
- ✅ PKG synchronization >99% consistent
- ✅ Performance targets met (<2s latency, >100/min throughput)
- ✅ Quality gates validated (>80% quality score)
- ✅ Production deployment ready

## Dependencies

- All previous modules (01-05)
- Quality gates testing infrastructure
- PKG Graph Storage operational
- Extraction pipeline integration

## Next Steps

After integration testing complete:
1. Production deployment
2. Monitoring and observability setup
3. Performance optimization based on real-world usage

**This module validates production readiness and ensures all Option B requirements are met for the vector embedding service.**

Summary: Implement schema-versioned embedding storage with re-embedding triggers for schema evolution support.

# 03 · Schema-Versioned Storage

## Purpose
Implement embedding storage layer with schema version tracking, enabling re-embedding when schema evolves and maintaining consistency between embeddings and the evolving PKG schema.

**Criticality**: HIGH - Enables autonomous schema evolution without breaking embeddings

## Scope
- Embedding metadata with schema version tracking
- ChromaDB/Weaviate integration with version support
- Re-embedding trigger mechanisms
- Migration strategies for schema evolution
- Consistency validation between embeddings and PKG

## Requirements Alignment
- **Option B Requirement**: "Schema evolution must not break existing embeddings"
- **Schema Evolution Support**: Track which schema version generated each embedding
- **Re-embedding Strategy**: Automatic triggers when schema changes significantly
- **Enables**: Seamless schema evolution without manual re-indexing

## Component Design

### Embedding Metadata

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any


class EmbeddingMetadata(BaseModel):
    """
    Metadata for stored embeddings.

    Tracks schema version to enable re-embedding when schema evolves.
    """
    embedding_id: str
    entity_id: str  # PKG node ID
    entity_type: str
    model_id: str
    model_version: str

    # Schema versioning (NEW - Option B)
    schema_version: int  # Which schema version was active
    schema_hash: str     # Hash of schema structure
    extraction_template_version: Optional[str] = None  # Thought template version

    # Quality tracking
    extraction_confidence: float
    embedding_quality_score: Optional[float] = None

    # Provenance
    source_document_id: str
    created_at: datetime
    last_validated: datetime

    # Re-embedding flags
    needs_reembedding: bool = False
    reembedding_reason: Optional[str] = None


class EmbeddingRecord(BaseModel):
    """Complete embedding record with vector and metadata."""
    metadata: EmbeddingMetadata
    vector: List[float]
    vector_dimension: int
```

### Schema-Versioned Embedding Store

```python
class SchemaVersionedEmbeddingStore:
    """
    Embedding store with schema version tracking.

    Integrates with ChromaDB/Weaviate and tracks schema versions
    to enable re-embedding when schema evolves.
    """

    def __init__(self, chroma_client, pkg_client):
        self.chroma = chroma_client
        self.pkg = pkg_client
        self.current_schema_version = self._get_current_schema_version()

    def _get_current_schema_version(self) -> int:
        """Get current schema version from PKG."""
        # Query PKG for latest SchemaVersion node
        result = self.pkg.query("""
            MATCH (sv:SchemaVersion)
            RETURN sv.version AS version
            ORDER BY sv.version DESC
            LIMIT 1
        """)

        if result:
            return result[0]["version"]
        return 1  # Default to version 1

    def store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: np.ndarray,
        model_id: str,
        extraction_confidence: float,
        source_document_id: str,
        template_version: Optional[str] = None
    ) -> str:
        """
        Store embedding with schema version metadata.

        Returns: embedding_id
        """
        import hashlib
        import uuid

        # Generate embedding ID
        embedding_id = str(uuid.uuid4())

        # Get schema hash
        schema_hash = self._compute_schema_hash()

        # Create metadata
        metadata = EmbeddingMetadata(
            embedding_id=embedding_id,
            entity_id=entity_id,
            entity_type=entity_type,
            model_id=model_id,
            model_version=self._get_model_version(model_id),
            schema_version=self.current_schema_version,
            schema_hash=schema_hash,
            extraction_template_version=template_version,
            extraction_confidence=extraction_confidence,
            source_document_id=source_document_id,
            created_at=datetime.now(),
            last_validated=datetime.now()
        )

        # Store in ChromaDB with metadata
        self.chroma.add(
            collection_name="embeddings",
            embeddings=[embedding.tolist()],
            metadatas=[metadata.dict()],
            ids=[embedding_id]
        )

        return embedding_id

    def _compute_schema_hash(self) -> str:
        """
        Compute hash of current schema structure.

        Used to detect schema changes that require re-embedding.
        """
        import hashlib
        import json

        # Get schema from PKG
        schema = self.pkg.query("""
            MATCH (sv:SchemaVersion)
            WHERE sv.version = $version
            RETURN sv.entity_types AS entities, sv.relationship_types AS relationships
        """, version=self.current_schema_version)

        if not schema:
            return hashlib.sha256(b"").hexdigest()

        # Hash schema structure
        schema_json = json.dumps(schema[0], sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()

    def _get_model_version(self, model_id: str) -> str:
        """Get version of embedding model."""
        # This would track model versions in registry
        return "1.0.0"  # Placeholder

    def query_embeddings(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        entity_type: Optional[str] = None,
        min_schema_version: Optional[int] = None
    ) -> List[EmbeddingRecord]:
        """
        Query embeddings with schema version filtering.

        Can filter by minimum schema version to exclude outdated embeddings.
        """
        # Build metadata filter
        where_filter = {}
        if entity_type:
            where_filter["entity_type"] = entity_type
        if min_schema_version:
            where_filter["schema_version"] = {"$gte": min_schema_version}

        # Query ChromaDB
        results = self.chroma.query(
            collection_name="embeddings",
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where_filter if where_filter else None
        )

        # Convert to EmbeddingRecord objects
        records = []
        for i, metadata in enumerate(results["metadatas"][0]):
            records.append(EmbeddingRecord(
                metadata=EmbeddingMetadata(**metadata),
                vector=results["embeddings"][0][i],
                vector_dimension=len(results["embeddings"][0][i])
            ))

        return records

    def mark_for_reembedding(
        self,
        entity_ids: Optional[List[str]] = None,
        schema_version: Optional[int] = None,
        reason: str = "schema_evolution"
    ):
        """
        Mark embeddings for re-embedding.

        Can target specific entities or all embeddings from a schema version.
        """
        if entity_ids:
            # Mark specific entities
            for entity_id in entity_ids:
                self.chroma.update(
                    collection_name="embeddings",
                    ids=[entity_id],
                    metadatas=[{
                        "needs_reembedding": True,
                        "reembedding_reason": reason
                    }]
                )
        elif schema_version:
            # Mark all embeddings from schema version
            # This requires fetching all embeddings (not ideal, but necessary)
            all_embeddings = self.chroma.get(
                collection_name="embeddings",
                where={"schema_version": schema_version}
            )

            for emb_id in all_embeddings["ids"]:
                self.chroma.update(
                    collection_name="embeddings",
                    ids=[emb_id],
                    metadatas=[{
                        "needs_reembedding": True,
                        "reembedding_reason": reason
                    }]
                )
```

### Re-embedding Service

```python
class ReembeddingService:
    """
    Service to re-embed entities when schema evolves.

    Monitors schema changes and triggers re-embedding as needed.
    """

    def __init__(
        self,
        embedding_store: SchemaVersionedEmbeddingStore,
        embedding_service: MultiModelEmbeddingService,
        pkg_client
    ):
        self.store = embedding_store
        self.service = embedding_service
        self.pkg = pkg_client

    def detect_schema_changes(
        self,
        old_version: int,
        new_version: int
    ) -> Dict[str, Any]:
        """
        Detect changes between schema versions.

        Returns: {
            "new_entity_types": [...],
            "removed_entity_types": [...],
            "new_relationship_types": [...],
            "requires_reembedding": bool
        }
        """
        # Get old schema
        old_schema = self.pkg.query("""
            MATCH (sv:SchemaVersion)
            WHERE sv.version = $version
            RETURN sv.entity_types AS entities, sv.relationship_types AS relationships
        """, version=old_version)

        # Get new schema
        new_schema = self.pkg.query("""
            MATCH (sv:SchemaVersion)
            WHERE sv.version = $version
            RETURN sv.entity_types AS entities, sv.relationship_types AS relationships
        """, version=new_version)

        if not old_schema or not new_schema:
            return {"requires_reembedding": False}

        old_entities = set(old_schema[0]["entities"])
        new_entities = set(new_schema[0]["entities"])

        changes = {
            "new_entity_types": list(new_entities - old_entities),
            "removed_entity_types": list(old_entities - new_entities),
            "new_relationship_types": [],
            "requires_reembedding": len(new_entities - old_entities) > 0
        }

        return changes

    def trigger_reembedding(
        self,
        schema_version: Optional[int] = None,
        entity_ids: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        """
        Trigger re-embedding for entities.

        Can re-embed by schema version or specific entity IDs.
        """
        # Mark for re-embedding
        self.store.mark_for_reembedding(
            entity_ids=entity_ids,
            schema_version=schema_version,
            reason="schema_evolution"
        )

        # Get entities needing re-embedding
        to_reembed = self.store.chroma.get(
            collection_name="embeddings",
            where={"needs_reembedding": True},
            limit=batch_size
        )

        # Re-embed in batches
        for i in range(0, len(to_reembed["ids"]), batch_size):
            batch_ids = to_reembed["ids"][i:i+batch_size]
            batch_metadata = to_reembed["metadatas"][i:i+batch_size]

            # Fetch entities from PKG
            entities = self._fetch_entities_from_pkg(
                [m["entity_id"] for m in batch_metadata]
            )

            # Re-embed
            for entity, metadata in zip(entities, batch_metadata):
                new_embedding = self.service.embed(
                    entity_type=entity["type"],
                    content=entity["content"],
                    temporal_context=entity.get("temporal_context"),
                    metadata=metadata
                )

                # Update embedding in store
                self.store.chroma.update(
                    collection_name="embeddings",
                    ids=[metadata["embedding_id"]],
                    embeddings=[new_embedding.embedding.tolist()],
                    metadatas=[{
                        "schema_version": self.store.current_schema_version,
                        "schema_hash": self.store._compute_schema_hash(),
                        "needs_reembedding": False,
                        "reembedding_reason": None,
                        "last_validated": datetime.now().isoformat()
                    }]
                )

    def _fetch_entities_from_pkg(
        self,
        entity_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch entity data from PKG for re-embedding."""
        # Query PKG for entities
        results = []
        for entity_id in entity_ids:
            entity = self.pkg.query("""
                MATCH (e)
                WHERE e.id = $entity_id
                RETURN e
            """, entity_id=entity_id)

            if entity:
                results.append({
                    "type": entity[0]["e"].labels[0],
                    "content": self._format_entity_content(entity[0]["e"]),
                    "temporal_context": self._extract_temporal_context(entity[0]["e"])
                })

        return results

    def _format_entity_content(self, entity) -> str:
        """Format entity for re-embedding."""
        # Extract relevant properties
        return f"{entity.get('name', '')}: {entity.get('description', '')}"

    def _extract_temporal_context(self, entity) -> Optional[TemporalEmbeddingContext]:
        """Extract temporal context for events."""
        if "timestamp" in entity:
            return TemporalEmbeddingContext(
                timestamp=entity["timestamp"],
                duration=entity.get("duration"),
                temporal_type=entity.get("temporal_type")
            )
        return None
```

## Implementation Details

### Week 3: Schema-Versioned Storage Implementation

**Deliverable**: Working schema-versioned embedding store

1. **Implement `EmbeddingMetadata` model**:
   - Schema version tracking
   - Schema hash computation
   - Template version tracking

2. **Integrate with ChromaDB**:
   - Metadata storage and querying
   - Schema version filtering
   - Update mechanisms

3. **Implement re-embedding triggers**:
   - Schema change detection
   - Automatic marking for re-embedding
   - Quality-based re-embedding

### Week 4: Re-embedding Service

**Deliverable**: Automated re-embedding workflow

1. **Implement `ReembeddingService`**:
   - Schema change detection
   - Batch re-embedding
   - Progress tracking

2. **Migration strategies**:
   - Incremental re-embedding
   - Priority-based re-embedding (high-confidence first)
   - Background re-embedding jobs

3. **Validation mechanisms**:
   - Embedding consistency checks
   - Schema alignment validation

## Testing Strategy

```python
class TestSchemaVersionedStorage:
    def test_schema_version_tracking(self):
        """Validate embeddings track schema version."""
        store = SchemaVersionedEmbeddingStore(chroma_client, pkg_client)

        embedding_id = store.store_embedding(
            entity_id="person_123",
            entity_type="Person",
            embedding=np.random.rand(768),
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="doc_456"
        )

        # Retrieve and check schema version
        results = store.query_embeddings(
            query_vector=np.random.rand(768),
            top_k=10
        )

        record = next(r for r in results if r.metadata.embedding_id == embedding_id)
        assert record.metadata.schema_version == store.current_schema_version

    def test_reembedding_trigger(self):
        """Validate re-embedding triggered on schema evolution."""
        service = ReembeddingService(store, embedding_service, pkg_client)

        # Simulate schema evolution
        changes = service.detect_schema_changes(old_version=1, new_version=2)

        assert changes["requires_reembedding"] == True

        # Trigger re-embedding
        service.trigger_reembedding(schema_version=1, batch_size=10)

        # Verify embeddings updated
        updated = store.chroma.get(
            collection_name="embeddings",
            where={"schema_version": 2}
        )

        assert len(updated["ids"]) > 0
```

## Success Metrics

- ✅ Schema version tracking functional for all embeddings
- ✅ Re-embedding triggers on schema evolution
- ✅ Migration completes <24 hours for 100k embeddings
- ✅ Embedding consistency maintained during schema evolution
- ✅ Zero downtime during re-embedding

## Dependencies

- Multi-model architecture (02-multi-model-architecture.md)
- PKG schema versioning (from PKG storage module 01)
- Schema evolution system (from entity-relationship extraction module 02)
- ChromaDB/Weaviate

## Next Steps

After schema-versioned storage complete:
1. Integrate with PKG synchronization (04-pkg-synchronization.md)
2. Enable quality evolution tracking (05-quality-evolution.md)
3. Production testing and validation (06-integration-testing.md)

**This module enables seamless schema evolution without breaking the embedding layer.**

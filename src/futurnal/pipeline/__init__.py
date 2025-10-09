"""Pipelines for normalization and storage handoff."""

from .graph import Neo4jPKGWriter
from .stubs import NormalizationSink
from .vector import ChromaVectorWriter
from .triples import (
    SemanticTriple,
    Entity,
    MetadataTripleExtractor,
    TripleEnrichedNormalizationSink,
)
from .models import (
    DocumentFormat,
    ChunkingStrategy,
    SchemaVersion,
    NormalizedMetadata,
    DocumentChunk,
    NormalizedDocument,
    NormalizedDocumentV1,
    compute_content_hash,
    compute_chunk_hash,
    generate_chunk_id,
)

__all__ = [
    "NormalizationSink",
    "Neo4jPKGWriter",
    "ChromaVectorWriter",
    "SemanticTriple",
    "Entity",
    "MetadataTripleExtractor",
    "TripleEnrichedNormalizationSink",
    # Normalized document schema
    "DocumentFormat",
    "ChunkingStrategy",
    "SchemaVersion",
    "NormalizedMetadata",
    "DocumentChunk",
    "NormalizedDocument",
    "NormalizedDocumentV1",
    "compute_content_hash",
    "compute_chunk_hash",
    "generate_chunk_id",
]



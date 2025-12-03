"""Schema Evolution Semantic Alignment Validation (>90% target).

This module implements REAL semantic alignment measurement using LOCAL LLM
to validate that autonomously evolved schemas align with manually curated
reference schemas.

PRODUCTION GATE: >90% semantic alignment (AutoSchemaKG benchmark)
"""

import pytest
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from futurnal.extraction.local_llm_client import get_test_llm_client, LLMClient
from futurnal.extraction.schema import (
    create_seed_schema,
    SchemaEvolutionEngine,
    ExtractionPhase,
)
from futurnal.extraction.schema.models import EntityType, RelationshipType
from futurnal.extraction.schema.evolution import Document

logger = logging.getLogger(__name__)


# ==============================================================================
# Manual Reference Schema (Ground Truth)
# ==============================================================================

@dataclass
class ManualSchema:
    """Manually curated reference schema for validation."""
    entity_types: List[str]
    relationship_types: List[str]
    descriptions: Dict[str, str]


def load_manual_reference_schema() -> ManualSchema:
    """Load manually curated reference schema.

    This represents the "ideal" schema that autonomous evolution should discover.
    Based on common knowledge graph patterns and AutoSchemaKG benchmarks.
    """
    return ManualSchema(
        entity_types=[
            # Core entity types (Phase 1)
            "Person",
            "Organization",
            "Concept",
            "Location",

            # Event entities (Phase 2)
            "Event",
            "Meeting",
            "Decision",
            "Publication",

            # Temporal/Causal (Phase 3)
            "Action",
            "StateChange",
        ],
        relationship_types=[
            # Entity-Entity relationships (Phase 1)
            "works_at",
            "employed_by",
            "member_of",
            "located_in",
            "part_of",
            "related_to",
            "created",
            "authored",

            # Entity-Event relationships (Phase 2)
            "participated_in",
            "attended",
            "organized",
            "occurred_at",

            # Event-Event (Causal) relationships (Phase 3)
            "causes",
            "enables",
            "prevents",
            "triggers",
            "leads_to",
            "followed_by",
        ],
        descriptions={
            "Person": "Individual human being",
            "Organization": "Company, institution, or group",
            "Concept": "Abstract idea, topic, or domain",
            "Event": "Time-bound occurrence or happening",
            "works_at": "Employment relationship",
            "causes": "Causal relationship between events",
        }
    )


# ==============================================================================
# Semantic Similarity Calculation (LLM-based)
# ==============================================================================

def calculate_semantic_similarity(
    text1: str,
    text2: str,
    llm: LLMClient
) -> float:
    """Calculate semantic similarity between two texts using LOCAL LLM.

    Args:
        text1: First text
        text2: Second text
        llm: Local LLM client

    Returns:
        Similarity score [0.0, 1.0]
    """
    prompt = f"""Compare the semantic similarity of these two concepts:

Concept A: "{text1}"
Concept B: "{text2}"

Rate their semantic similarity on a scale of 0.0 to 1.0, where:
- 1.0 = Identical or synonymous meaning
- 0.8-0.9 = Very similar, minor differences
- 0.6-0.7 = Related but distinct concepts
- 0.4-0.5 = Somewhat related
- 0.0-0.3 = Unrelated or opposite

Respond with ONLY a single number between 0.0 and 1.0."""

    response = llm.generate(prompt, max_new_tokens=10, temperature=0.0)

    # Parse similarity score
    try:
        score = float(response.strip())
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except ValueError:
        logger.warning(f"Failed to parse similarity score: {response}")
        return 0.0


def compute_schema_alignment(
    evolved_types: List[str],
    reference_types: List[str],
    llm: LLMClient,
    similarity_threshold: float = 0.7
) -> Tuple[float, Dict[str, any]]:
    """Compute semantic alignment between evolved and reference schemas.

    Uses LOCAL LLM to measure semantic similarity between type names.

    Args:
        evolved_types: Types discovered by autonomous evolution
        reference_types: Manually curated reference types
        llm: Local LLM for similarity measurement
        similarity_threshold: Minimum similarity to count as match

    Returns:
        (alignment_score, detailed_metrics)
    """
    if not reference_types:
        return 1.0 if not evolved_types else 0.0, {}

    matches = 0
    matched_ref = set()
    similarity_scores = []

    logger.info(f"\nComputing alignment:")
    logger.info(f"  Evolved types: {len(evolved_types)}")
    logger.info(f"  Reference types: {len(reference_types)}")

    # For each evolved type, find best match in reference
    for evolved in evolved_types:
        best_similarity = 0.0
        best_match = None

        for ref in reference_types:
            if ref in matched_ref:
                continue

            similarity = calculate_semantic_similarity(evolved, ref, llm)
            similarity_scores.append(similarity)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = ref

        if best_similarity >= similarity_threshold:
            matches += 1
            if best_match:
                matched_ref.add(best_match)
                logger.info(f"  ✓ '{evolved}' → '{best_match}' (sim: {best_similarity:.2f})")
        else:
            logger.info(f"  ✗ '{evolved}' (best: {best_similarity:.2f})")

    # Calculate alignment metrics
    precision = matches / len(evolved_types) if evolved_types else 0.0
    recall = len(matched_ref) / len(reference_types) if reference_types else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Alignment score is F1 (balance of precision and recall)
    alignment = f1

    return alignment, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "evolved_count": len(evolved_types),
        "reference_count": len(reference_types),
        "matched_reference": len(matched_ref),
        "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    }


# ==============================================================================
# PRODUCTION GATE: Schema Evolution Semantic Alignment (>90%)
# ==============================================================================

@pytest.mark.production_readiness
@pytest.mark.slow  # Uses real LLM inference
def test_schema_evolution_semantic_alignment_gate(local_llm: LLMClient = None):
    """PRODUCTION GATE: Schema evolution semantic alignment must exceed 90%.

    Validates that autonomously evolved schema aligns with manually curated
    reference schema using REAL LOCAL LLM semantic similarity.

    Target: >90% alignment (AutoSchemaKG benchmark)
    """
    logger.info("=" * 80)
    logger.info("PRODUCTION GATE: Schema Evolution Semantic Alignment")
    logger.info("=" * 80)

    # Get local LLM for similarity measurement
    if local_llm is None:
        logger.info("Loading LOCAL LLM for semantic alignment...")
        local_llm = get_test_llm_client(fast=True)

    # Load manual reference schema
    reference = load_manual_reference_schema()
    logger.info(f"\nReference schema loaded:")
    logger.info(f"  Entity types: {len(reference.entity_types)}")
    logger.info(f"  Relationship types: {len(reference.relationship_types)}")

    # Create seed schema and evolution engine
    seed = create_seed_schema()
    engine = SchemaEvolutionEngine(seed)

    # Simulate schema evolution with sample documents
    # In production, this would use real document corpus
    logger.info("\nSimulating schema evolution...")
    documents = [
        Document(f"Sample document {i} with entities and relationships", f"doc{i}")
        for i in range(200)  # Target from spec: 200+ documents
    ]

    # Evolve through all phases
    logger.info("Phase 1: Entity-Entity extraction...")
    phase1_schema = engine.induce_schema_from_documents(
        documents, ExtractionPhase.ENTITY_ENTITY
    )

    logger.info("Phase 2: Entity-Event extraction...")
    phase2_schema = engine.induce_schema_from_documents(
        documents, ExtractionPhase.ENTITY_EVENT
    )

    logger.info("Phase 3: Event-Event extraction...")
    phase3_schema = engine.induce_schema_from_documents(
        documents, ExtractionPhase.EVENT_EVENT
    )

    # Get evolved types
    evolved_entity_types = list(phase3_schema.entity_types.keys())
    evolved_rel_types = list(phase3_schema.relationship_types.keys())

    logger.info(f"\nEvolved schema:")
    logger.info(f"  Entity types: {len(evolved_entity_types)}")
    logger.info(f"  Relationship types: {len(evolved_rel_types)}")

    # Compute semantic alignment for entities
    logger.info("\n" + "-" * 80)
    logger.info("Entity Type Alignment:")
    logger.info("-" * 80)
    entity_alignment, entity_metrics = compute_schema_alignment(
        evolved_entity_types,
        reference.entity_types,
        local_llm,
        similarity_threshold=0.7
    )

    # Compute semantic alignment for relationships
    logger.info("\n" + "-" * 80)
    logger.info("Relationship Type Alignment:")
    logger.info("-" * 80)
    rel_alignment, rel_metrics = compute_schema_alignment(
        evolved_rel_types,
        reference.relationship_types,
        local_llm,
        similarity_threshold=0.7
    )

    # Overall alignment (weighted average)
    overall_alignment = (entity_alignment + rel_alignment) / 2

    # Report results
    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC ALIGNMENT RESULTS:")
    logger.info("=" * 80)
    logger.info(f"Entity Type Alignment: {entity_alignment:.2%}")
    logger.info(f"  Precision: {entity_metrics['precision']:.2%}")
    logger.info(f"  Recall: {entity_metrics['recall']:.2%}")
    logger.info(f"  F1: {entity_metrics['f1']:.2%}")
    logger.info(f"\nRelationship Type Alignment: {rel_alignment:.2%}")
    logger.info(f"  Precision: {rel_metrics['precision']:.2%}")
    logger.info(f"  Recall: {rel_metrics['recall']:.2%}")
    logger.info(f"  F1: {rel_metrics['f1']:.2%}")
    logger.info(f"\n{'='*80}")
    logger.info(f"OVERALL SEMANTIC ALIGNMENT: {overall_alignment:.2%}")
    logger.info("=" * 80)

    # PRODUCTION GATE: Must exceed 90% alignment
    assert overall_alignment > 0.90, (
        f"Schema semantic alignment {overall_alignment:.2%} "
        f"does not meet production gate threshold of 90%"
    )

    logger.info("\n✅ PRODUCTION GATE: PASSED")

    return overall_alignment, entity_metrics, rel_metrics


@pytest.mark.production_readiness
@pytest.mark.slow
def test_entity_type_coverage():
    """Validate evolved schema covers core entity types."""
    logger.info("Validating entity type coverage...")

    seed = create_seed_schema()
    engine = SchemaEvolutionEngine(seed)

    documents = [Document(f"Doc {i}", f"doc{i}") for i in range(50)]

    # Phase 1 should discover core entity types
    schema = engine.induce_schema_from_documents(
        documents, ExtractionPhase.ENTITY_ENTITY
    )

    # Must have at least core types
    core_types = {"Person", "Organization", "Concept"}
    evolved_types = set(schema.entity_types.keys())

    coverage = len(core_types & evolved_types) / len(core_types)
    logger.info(f"Core type coverage: {coverage:.2%}")

    assert coverage >= 0.8, f"Core entity type coverage {coverage:.2%} below 80%"


@pytest.mark.production_readiness
@pytest.mark.slow
def test_relationship_type_richness():
    """Validate evolved schema has rich relationship vocabulary."""
    logger.info("Validating relationship type richness...")

    seed = create_seed_schema()
    engine = SchemaEvolutionEngine(seed)

    documents = [Document(f"Doc {i}", f"doc{i}") for i in range(100)]

    schema = engine.induce_schema_from_documents(
        documents, ExtractionPhase.ENTITY_ENTITY
    )

    # Should discover multiple relationship types
    num_rel_types = len(schema.relationship_types)
    logger.info(f"Discovered {num_rel_types} relationship types")

    assert num_rel_types >= 5, (
        f"Only {num_rel_types} relationship types discovered, "
        f"expected at least 5 for rich schema"
    )


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def local_llm() -> LLMClient:
    """Real LOCAL LLM for semantic similarity measurement."""
    logger.info("Loading LOCAL LLM for schema alignment tests...")
    client = get_test_llm_client(fast=True)
    logger.info("LOCAL LLM loaded")
    return client

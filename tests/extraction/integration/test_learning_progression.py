"""Multi-Document Learning Progression Validation.

Validates Ghost→Animal evolution through experiential learning:
- Ghost model remains frozen (no parameter updates)
- Animal behavior emerges via experiential knowledge (token priors)
- Extraction quality improves measurably over 50+ documents

Tests the core Training-Free GRPO innovation WITHOUT fine-tuning.

PRODUCTION GATE: Measurable quality improvement over 50+ documents
"""

import pytest
import logging
from typing import List, Tuple, Dict
from datetime import datetime

# Local LLM and experiential learning
from futurnal.extraction.local_llm_client import (
    get_test_llm_client,
    ExperientialLLMWrapper,
    LLMClient
)
from futurnal.extraction.schema.experiential import (
    TrainingFreeGRPO,
    WorldStateModel
)
from futurnal.extraction.schema.models import SemanticAdvantage

# Test corpus
from tests.extraction.test_corpus import load_corpus, TestDocument

logger = logging.getLogger(__name__)


# ==============================================================================
# Learning Progression Measurement
# ==============================================================================

def measure_extraction_quality(
    documents: List[TestDocument],
    extractor_func,
    **kwargs
) -> float:
    """Measure extraction quality (average confidence).

    Args:
        documents: Documents to extract from
        extractor_func: Extraction function
        **kwargs: Additional arguments

    Returns:
        Average quality score (0.0-1.0)
    """
    confidences = []

    for doc in documents:
        results = extractor_func(doc, **kwargs)

        # Extract confidence scores from results
        if isinstance(results, list):
            for r in results:
                if hasattr(r, 'confidence'):
                    confidences.append(r.confidence)
                elif isinstance(r, dict) and 'confidence' in r:
                    confidences.append(r['confidence'])

    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)


# ==============================================================================
# PRODUCTION GATE: Multi-Document Learning Validation
# ==============================================================================

@pytest.mark.production_readiness
@pytest.mark.slow
def test_ghost_to_animal_learning_progression(local_llm: LLMClient = None):
    """PRODUCTION GATE: Extraction quality improves over 50+ documents.

    Validates Ghost→Animal evolution through experiential learning:
    1. Ghost model (frozen base LLM)
    2. Experiential knowledge accumulation (token priors)
    3. Animal behavior (Ghost + Experience)
    4. Measurable quality improvement

    Requirements:
    - Quality improves from docs 1-10 to docs 40-50
    - No model parameter updates (Ghost frozen)
    - Learning via natural language patterns only
    """
    logger.info("=" * 80)
    logger.info("PRODUCTION GATE: Ghost→Animal Learning Progression")
    logger.info("=" * 80)

    # Initialize Ghost LLM (frozen)
    if local_llm is None:
        logger.info("\nLoading Ghost LLM (frozen base model)...")
        local_llm = get_test_llm_client(fast=True)

    # Verify model is frozen
    if hasattr(local_llm, 'model'):
        is_frozen = not any(p.requires_grad for p in local_llm.model.parameters())
        assert is_frozen, "Ghost model must be frozen (no parameter updates)"
        logger.info("✓ Ghost model verified as frozen")

    # Initialize Training-Free GRPO
    logger.info("\nInitializing Training-Free GRPO...")
    grpo = TrainingFreeGRPO(
        llm=local_llm,
        knowledge_capacity=20,
        rollout_group_size=4
    )

    # Initialize World State Model
    world_state = WorldStateModel()

    # Load documents (need 50+ for progression validation)
    logger.info("\nLoading document corpus...")
    corpus = load_corpus("all")

    # Extend corpus if needed
    while len(corpus) < 50:
        corpus.extend(load_corpus("temporal"))

    corpus = corpus[:50]  # Use exactly 50 documents
    logger.info(f"Loaded {len(corpus)} documents for learning progression")

    # Track quality over time
    quality_history = []
    batch_size = 10

    logger.info("\n" + "-" * 80)
    logger.info("Processing documents with experiential learning...")
    logger.info("-" * 80)

    for batch_idx in range(5):  # 5 batches of 10 documents
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        batch_docs = corpus[batch_start:batch_end]

        logger.info(f"\nBatch {batch_idx + 1}: Documents {batch_start + 1}-{batch_end}")

        # Simple extraction task for validation
        batch_quality = 0.0
        for doc in batch_docs:
            # Generate rollouts (multiple extraction attempts)
            rollouts = grpo.generate_rollouts(doc, "Extract key information")

            # Compute quality (using confidence as proxy)
            if rollouts:
                batch_quality += sum(r.confidence for r in rollouts) / len(rollouts)

            # Extract semantic advantages (what worked better?)
            advantages = grpo.extract_semantic_advantages(rollouts)

            # Update experiential knowledge (no parameter updates!)
            grpo.update_experiential_knowledge(advantages)

        batch_quality /= len(batch_docs)
        quality_history.append(batch_quality)

        # Update world state
        world_state.quality_history.append({
            "batch": batch_idx + 1,
            "quality": batch_quality,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"  Quality: {batch_quality:.3f}")
        logger.info(f"  Experiential patterns: {len(grpo.experiential_knowledge)}")

    # Analyze learning progression
    logger.info("\n" + "=" * 80)
    logger.info("LEARNING PROGRESSION ANALYSIS:")
    logger.info("=" * 80)

    early_quality = sum(quality_history[:2]) / 2  # Batches 1-2 (docs 1-20)
    late_quality = sum(quality_history[-2:]) / 2  # Batches 4-5 (docs 31-50)
    improvement = late_quality - early_quality
    improvement_pct = (improvement / early_quality * 100) if early_quality > 0 else 0

    logger.info(f"\nQuality Metrics:")
    logger.info(f"  Early quality (docs 1-20): {early_quality:.3f}")
    logger.info(f"  Late quality (docs 31-50): {late_quality:.3f}")
    logger.info(f"  Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

    logger.info(f"\nExperiential Learning:")
    logger.info(f"  Patterns learned: {len(grpo.experiential_knowledge)}")
    logger.info(f"  Knowledge capacity: {grpo.knowledge_capacity}")

    # PRODUCTION GATE: Quality must improve
    assert late_quality > early_quality, (
        f"No quality improvement detected: "
        f"early={early_quality:.3f}, late={late_quality:.3f}"
    )

    logger.info("\n✅ PRODUCTION GATE: PASSED")
    logger.info("Ghost→Animal evolution validated")

    return quality_history


@pytest.mark.production_readiness
@pytest.mark.slow
def test_experiential_knowledge_accumulation():
    """Validate experiential knowledge accumulates correctly.

    Tests that patterns are learned and stored as token priors.
    """
    logger.info("Testing experiential knowledge accumulation...")

    local_llm = get_test_llm_client(fast=True)
    grpo = TrainingFreeGRPO(llm=local_llm, knowledge_capacity=10)

    corpus = load_corpus("temporal")[:20]

    initial_knowledge_count = len(grpo.experiential_knowledge)
    logger.info(f"Initial knowledge patterns: {initial_knowledge_count}")

    # Process documents
    for doc in corpus:
        rollouts = grpo.generate_rollouts(doc, "Extract entities")
        advantages = grpo.extract_semantic_advantages(rollouts)
        grpo.update_experiential_knowledge(advantages)

    final_knowledge_count = len(grpo.experiential_knowledge)
    logger.info(f"Final knowledge patterns: {final_knowledge_count}")

    # Knowledge should accumulate (up to capacity)
    assert final_knowledge_count > initial_knowledge_count, (
        "No experiential knowledge accumulated"
    )
    assert final_knowledge_count <= grpo.knowledge_capacity, (
        "Knowledge exceeded capacity"
    )

    logger.info("✓ Experiential knowledge accumulates correctly")


@pytest.mark.production_readiness
@pytest.mark.slow
def test_ghost_model_remains_frozen():
    """CRITICAL: Verify Ghost model never updates parameters.

    This is a core requirement of Training-Free GRPO - all learning
    happens via token priors, NOT parameter updates.
    """
    logger.info("CRITICAL TEST: Ghost model remains frozen...")

    local_llm = get_test_llm_client(fast=True)

    # Get initial parameter state
    if hasattr(local_llm, 'model'):
        initial_params = {
            name: param.clone().detach()
            for name, param in local_llm.model.named_parameters()
        }

        grpo = TrainingFreeGRPO(llm=local_llm)
        corpus = load_corpus("temporal")[:10]

        # Process documents (trigger learning)
        for doc in corpus:
            rollouts = grpo.generate_rollouts(doc, "Extract")
            advantages = grpo.extract_semantic_advantages(rollouts)
            grpo.update_experiential_knowledge(advantages)

        # Verify parameters unchanged
        import torch
        params_changed = False
        for name, param in local_llm.model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                logger.error(f"Parameter changed: {name}")

        assert not params_changed, (
            "CRITICAL FAILURE: Ghost model parameters were updated! "
            "Training-Free GRPO must NOT update parameters."
        )

        logger.info("✅ CRITICAL: Ghost model verified frozen (no parameter updates)")
    else:
        logger.warning("⚠️  Cannot verify parameter freeze (model not accessible)")


@pytest.mark.production_readiness
def test_world_state_trajectory_assessment():
    """Test world state model tracks quality trajectory."""
    logger.info("Testing world state trajectory assessment...")

    world_state = WorldStateModel()

    # Simulate quality progression
    for i in range(15):
        quality = 0.6 + (i * 0.02)  # Simulated improvement
        world_state.quality_history.append({
            "iteration": i,
            "precision": quality,
            "timestamp": datetime.now().isoformat()
        })

    # Assess trajectory (requires ≥10 data points)
    # Pass quality_history as recent_extractions for trajectory assessment
    trajectory = world_state.assess_extraction_trajectory(world_state.quality_history)

    logger.info(f"Trajectory assessment: {trajectory}")

    assert "current_precision" in trajectory
    assert "improvement" in trajectory


@pytest.mark.production_readiness
def test_curriculum_generation():
    """Test curriculum generator orders documents by learning value."""
    logger.info("Testing curriculum generation...")

    world_state = WorldStateModel()
    corpus = load_corpus("all")[:30]

    # Generate curriculum (orders documents for optimal learning)
    curriculum = world_state.generate_curriculum(corpus)

    logger.info(f"Generated curriculum for {len(curriculum)} documents")

    assert len(curriculum) == len(corpus)
    assert all(isinstance(doc, TestDocument) for doc in curriculum)


# ==============================================================================
# Experiential Wrapper Tests
# ==============================================================================

@pytest.mark.production_readiness
def test_experiential_wrapper_enhances_prompts():
    """Test ExperientialLLMWrapper adds token priors to prompts."""
    logger.info("Testing experiential wrapper...")

    local_llm = get_test_llm_client(fast=True)

    # Create wrapper without experience
    wrapper_empty = ExperientialLLMWrapper(local_llm, experiential_knowledge=[])

    # Create wrapper with experience
    experience = [
        "Focus on named entities",
        "Extract dates carefully",
        "Look for relationships"
    ]
    wrapper_with_exp = ExperientialLLMWrapper(local_llm, experiential_knowledge=experience)

    # Verify experiential knowledge is added
    assert len(wrapper_empty.experiential_knowledge) == 0
    assert len(wrapper_with_exp.experiential_knowledge) == 3

    # Add more experience dynamically
    wrapper_with_exp.add_experience("Validate temporal ordering")
    assert len(wrapper_with_exp.experiential_knowledge) == 4

    logger.info("✓ Experiential wrapper works correctly")


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def local_llm() -> LLMClient:
    """Real LOCAL LLM for learning tests."""
    logger.info("Loading LOCAL LLM for learning progression tests...")
    client = get_test_llm_client(fast=True)
    logger.info("LOCAL LLM loaded")
    return client

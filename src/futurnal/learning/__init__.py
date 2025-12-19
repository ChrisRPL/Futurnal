"""
Experiential Learning Foundation Module

Implements Ghost to Animal evolution via experiential learning without model
fine-tuning. This is THE core differentiator that enables personal AI evolution.

CRITICAL CONSTRAINT: Ghost model MUST remain FROZEN.
All learning via token priors (natural language), NOT parameter updates.

Research Foundation:
- SEAgent (2508.04700v2): World State Model, Curriculum Generator, GRPO
- Training-Free GRPO (2510.08191v1): Token priors instead of parameters
- TOTAL (2510.07499v1): Thought templates with textual gradients

Option B Compliance:
- Ghost model frozen (verified by test)
- Experiential knowledge as token priors (natural language strings)
- Training-Free GRPO (no gradient computation)
- Local-only processing (no cloud)
- Quality improvement >5% over 50 documents
"""

from futurnal.learning.world_state import (
    QualityMetrics,
    ExtractionTrajectory,
    WorldStateAssessor,
)
from futurnal.learning.curriculum import (
    DocumentComplexity,
    CurriculumGenerator,
)
from futurnal.learning.token_priors import (
    EntityTypePrior,
    RelationTypePrior,
    TemporalPatternPrior,
    TokenPriorStore,
)
from futurnal.learning.integration import (
    ExperientialLearningPipeline,
)
from futurnal.learning.reflective_reasoning import (
    ReasoningStrategy,
    ReasoningOutcome,
    StrategyWeight,
    AdaptivePolicyOptimizer,
    ReasoningStep,
    ReasoningTrace,
    ReflectiveReasoner,
)

__all__ = [
    # World State
    "QualityMetrics",
    "ExtractionTrajectory",
    "WorldStateAssessor",
    # Curriculum
    "DocumentComplexity",
    "CurriculumGenerator",
    # Token Priors
    "EntityTypePrior",
    "RelationTypePrior",
    "TemporalPatternPrior",
    "TokenPriorStore",
    # Integration
    "ExperientialLearningPipeline",
    # Reflective Reasoning (MM-HELIX AHPO)
    "ReasoningStrategy",
    "ReasoningOutcome",
    "StrategyWeight",
    "AdaptivePolicyOptimizer",
    "ReasoningStep",
    "ReasoningTrace",
    "ReflectiveReasoner",
]

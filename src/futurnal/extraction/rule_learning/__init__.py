"""
LeSR-Style Automatic Rule Learning.

Implements LLM-enhanced symbolic reasoning for knowledge base completion:
- Subgraph pattern extraction
- LLM-powered rule generation
- Rule validation and refinement
- Continuous rule evolution

Based on LeSR paper (2501.01246v1).
"""

from .patterns import (
    SubgraphPattern,
    PatternExtractor,
    PatternMatcher,
)
from .rules import (
    ExtractionRule,
    RuleProposer,
    RuleValidator,
    RuleStore,
)
from .learner import (
    RuleLearner,
    LearningConfig,
    LearningMetrics,
)

__all__ = [
    "SubgraphPattern",
    "PatternExtractor",
    "PatternMatcher",
    "ExtractionRule",
    "RuleProposer",
    "RuleValidator",
    "RuleStore",
    "RuleLearner",
    "LearningConfig",
    "LearningMetrics",
]

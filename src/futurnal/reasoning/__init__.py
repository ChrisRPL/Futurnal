"""Reasoning module - Explicit causal boundary abstraction.

Research Foundation:
- arxiv:2510.07231 - LLM causal benchmarks (57.6% max accuracy)
- arxiv:2503.00237 - Agentic AI Needs Systems Theory

This module provides the CausalBoundary class that enforces
explicit separation between LLM pattern detection and causal validation.

Three-Layer Architecture:
1. Pattern Detection (LLM) -> correlation_confidence
2. Causal Validation (Algorithms) -> causal_confidence
3. Human Verification (ICDA) -> verified_confidence

Option B Compliance:
- Ghost model FROZEN
- Learning through token priors only
- No parameter updates
"""

from futurnal.reasoning.causal_boundary import (
    CausalBoundary,
    CausalBoundaryResult,
    ConfidenceType,
    create_causal_boundary,
)

__all__ = [
    "CausalBoundary",
    "CausalBoundaryResult",
    "ConfidenceType",
    "create_causal_boundary",
]

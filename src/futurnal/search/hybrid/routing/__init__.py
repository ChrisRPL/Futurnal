"""Query Routing & Orchestration Module.

Implements intelligent query routing with LLM-based intent classification,
multi-strategy orchestration, GRPO experiential learning hooks, and
thought template integration.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Components:
- QueryRouterLLMConfig: LLM backend configuration
- OllamaIntentClassifier: Fast intent classification via Ollama (<100ms)
- HuggingFaceIntentClassifier: Fallback classifier (<500ms)
- DynamicModelRouter: Runtime model switching
- QueryRouter: Main orchestrator for multi-strategy retrieval
- SearchQualityFeedback: GRPO integration for experiential learning
- QueryTemplateDatabase: Query understanding templates

Option B Compliance:
- Ghost model FROZEN: No parameter updates (model selection only)
- Experiential learning via token priors (SemanticAdvantage patterns)
- Templates evolve via textual gradients, not fine-tuning
- Temporal-first design: TEMPORAL queries prioritized
"""

from __future__ import annotations

from futurnal.search.hybrid.routing.config import (
    LLMBackendType,
    QueryRouterLLMConfig,
)
from futurnal.search.hybrid.routing.intent_classifier import (
    IntentClassifierLLM,
    OllamaIntentClassifier,
    HuggingFaceIntentClassifier,
    get_intent_classifier,
)
from futurnal.search.hybrid.routing.model_router import DynamicModelRouter
from futurnal.search.hybrid.routing.orchestrator import QueryRouter
from futurnal.search.hybrid.routing.feedback import (
    SearchQualitySignal,
    SearchQualityFeedback,
)
from futurnal.search.hybrid.routing.templates import (
    QueryTemplate,
    QueryTemplateDatabase,
)

__all__ = [
    # Configuration
    "LLMBackendType",
    "QueryRouterLLMConfig",
    # Intent Classification
    "IntentClassifierLLM",
    "OllamaIntentClassifier",
    "HuggingFaceIntentClassifier",
    "get_intent_classifier",
    # Model Routing
    "DynamicModelRouter",
    # Orchestration
    "QueryRouter",
    # GRPO Feedback
    "SearchQualitySignal",
    "SearchQualityFeedback",
    # Templates
    "QueryTemplate",
    "QueryTemplateDatabase",
]

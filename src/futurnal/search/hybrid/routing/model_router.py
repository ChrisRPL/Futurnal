"""Dynamic Model Router for Query Routing.

Enables runtime switching between LLM models based on query characteristics.

Features:
- Per-query model selection based on language/complexity
- Hot-swapping default model without restart
- Model availability checking
- Cached model clients for performance

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Model Selection Logic:
- Polish queries -> Bielik 4.5B
- Advanced reasoning -> Kimi-K2-Thinking
- Code queries -> Qwen 2.5 Coder
- Default -> Llama 3.1 8B

Option B Compliance:
- Ghost model FROZEN: Model switching is selection, not training
- Experiential knowledge applies to all models equally
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from futurnal.search.hybrid.routing.config import (
    LLMBackendType,
    QueryRouterLLMConfig,
)
from futurnal.search.hybrid.routing.intent_classifier import (
    IntentClassifierLLM,
    OllamaIntentClassifier,
    HuggingFaceIntentClassifier,
)

logger = logging.getLogger(__name__)


class DynamicModelRouter:
    """Enables runtime switching between LLM models.

    Supports:
    - Per-query model selection based on language/complexity
    - Hot-swapping default model without restart
    - Model availability checking
    - Caching of loaded model clients

    Environment Variables:
    - FUTURNAL_PRODUCTION_LLM: Default model (llama3.1|phi3|qwen|bielik|kimi|gpt-oss|auto)
    - FUTURNAL_LLM_BACKEND: Backend type (ollama|hf|auto)

    Example:
        router = DynamicModelRouter()

        # Auto-select based on query
        client = router.get_client_for_query("Co to jest machine learning?")
        # Returns Bielik client for Polish query

        client = router.get_client_for_query("Why did the system fail?")
        # Returns Kimi-K2 client for reasoning query
    """

    def __init__(self, config: Optional[QueryRouterLLMConfig] = None):
        """Initialize dynamic model router.

        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config or QueryRouterLLMConfig()
        self._model_clients: Dict[str, IntentClassifierLLM] = {}
        self._current_model: str = self.config.get_model_name()

        logger.info(f"DynamicModelRouter initialized with default model: {self._current_model}")

    def get_client_for_query(
        self,
        query: str,
        detected_language: Optional[str] = None,
        force_model: Optional[str] = None,
    ) -> IntentClassifierLLM:
        """Get appropriate LLM client for a query.

        Selects model based on query characteristics unless forced.

        Args:
            query: User query
            detected_language: Detected language (e.g., "pl", "en")
            force_model: Override model selection

        Returns:
            LLM client instance for the selected model
        """
        if force_model:
            model = self.config.MODEL_ALIASES.get(force_model, force_model)
            logger.debug(f"Forced model selection: {model}")
        else:
            model = self.config.get_model_for_query(query, detected_language)
            logger.debug(f"Auto-selected model for query: {model}")

        return self._get_or_create_client(model)

    def get_current_client(self) -> IntentClassifierLLM:
        """Get client for current default model.

        Returns:
            Client for the current default model
        """
        return self._get_or_create_client(self._current_model)

    def switch_default_model(self, model_name: str) -> bool:
        """Switch the default model at runtime.

        Args:
            model_name: Model alias or full name (e.g., "kimi", "bielik")

        Returns:
            True if switch successful, False otherwise
        """
        resolved = self.config.MODEL_ALIASES.get(model_name, model_name)

        # Verify model is available
        if not self._check_model_available(resolved):
            logger.warning(f"Model {resolved} not available, switch failed")
            return False

        old_model = self._current_model
        self._current_model = resolved
        logger.info(f"Switched default model: {old_model} -> {resolved}")

        return True

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their characteristics.

        Returns:
            List of model info dictionaries
        """
        models = [
            {
                "alias": "phi3",
                "model": "phi3:mini",
                "vram": "4GB",
                "use_case": "Fast testing, CI/CD",
                "language": "en",
                "available": self._check_model_available("phi3:mini"),
            },
            {
                "alias": "llama3.1",
                "model": "llama3.1:8b-instruct-q4_0",
                "vram": "8GB",
                "use_case": "Production default",
                "language": "en",
                "available": self._check_model_available("llama3.1:8b-instruct-q4_0"),
            },
            {
                "alias": "qwen",
                "model": "qwen2.5-coder:32b-instruct-q4_0",
                "vram": "16GB",
                "use_case": "Code queries",
                "language": "en",
                "available": self._check_model_available("qwen2.5-coder:32b-instruct-q4_0"),
            },
            {
                "alias": "bielik",
                "model": "bielik:4.5b-instruct-q4_0",
                "vram": "5GB",
                "use_case": "Polish language",
                "language": "pl",
                "available": self._check_model_available("bielik:4.5b-instruct-q4_0"),
            },
            {
                "alias": "kimi",
                "model": "kimi-k2:thinking",
                "vram": "16GB",
                "use_case": "Advanced reasoning",
                "language": "en",
                "available": self._check_model_available("kimi-k2:thinking"),
            },
            {
                "alias": "gpt-oss",
                "model": "gpt-oss:20b-derestricted",
                "vram": "12GB",
                "use_case": "Unrestricted content",
                "language": "en",
                "available": self._check_model_available("gpt-oss:20b-derestricted"),
            },
        ]

        return models

    def get_model_info(self, model_alias: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.

        Args:
            model_alias: Model alias (e.g., "kimi", "bielik")

        Returns:
            Model info dictionary or None if not found
        """
        models = self.list_available_models()
        for model in models:
            if model["alias"] == model_alias:
                return model
        return None

    def detect_language(self, query: str) -> str:
        """Detect query language.

        Simple heuristic-based detection. For production use,
        consider using a dedicated language detection library.

        Args:
            query: Query string

        Returns:
            ISO language code (e.g., "pl", "en")
        """
        if self.config._is_polish_query(query):
            return "pl"
        return "en"

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity for model selection.

        Args:
            query: Query string

        Returns:
            Complexity analysis with recommendations
        """
        analysis = {
            "is_polish": self.config._is_polish_query(query),
            "requires_reasoning": self.config._requires_advanced_reasoning(query),
            "is_code_related": self.config._is_code_query(query),
            "recommended_model": None,
            "reasoning": None,
        }

        if analysis["is_polish"]:
            analysis["recommended_model"] = "bielik"
            analysis["reasoning"] = "Polish language detected"
        elif analysis["requires_reasoning"]:
            analysis["recommended_model"] = "kimi"
            analysis["reasoning"] = "Advanced reasoning required"
        elif analysis["is_code_related"]:
            analysis["recommended_model"] = "qwen"
            analysis["reasoning"] = "Code-related query"
        else:
            analysis["recommended_model"] = "llama3.1"
            analysis["reasoning"] = "Default production model"

        return analysis

    def _get_or_create_client(self, model: str) -> IntentClassifierLLM:
        """Get or create cached client for model.

        Args:
            model: Ollama model name

        Returns:
            Cached or new client instance
        """
        if model not in self._model_clients:
            backend = self.config.backend or self.config.get_backend()

            if backend == LLMBackendType.OLLAMA:
                self._model_clients[model] = OllamaIntentClassifier(
                    model=model,
                    base_url=self.config.get_ollama_url(),
                    timeout=self.config.timeout_seconds,
                )
            else:
                # Map Ollama model to HuggingFace
                hf_model = self._ollama_to_hf_model(model)
                self._model_clients[model] = HuggingFaceIntentClassifier(
                    model_name=hf_model
                )

            logger.debug(f"Created new client for model: {model}")

        return self._model_clients[model]

    def _ollama_to_hf_model(self, ollama_model: str) -> str:
        """Map Ollama model name to HuggingFace name.

        Args:
            ollama_model: Ollama model name

        Returns:
            HuggingFace model name
        """
        # Reverse lookup in model map
        for hf_name, ollama_name in self.config.OLLAMA_MODEL_MAP.items():
            if ollama_name == ollama_model or ollama_model in ollama_name:
                return hf_name

        # Default to Phi-3 for unknown models
        return "microsoft/Phi-3-mini-4k-instruct"

    def _check_model_available(self, model: str) -> bool:
        """Check if model is available in Ollama.

        Args:
            model: Model name to check

        Returns:
            True if available
        """
        backend = self.config.backend or self.config.get_backend()

        if backend == LLMBackendType.HUGGINGFACE:
            # HuggingFace models are always "available" (downloaded on demand)
            return True

        try:
            response = requests.get(
                f"{self.config.get_ollama_url()}/api/tags",
                timeout=2,
            )
            if response.status_code == 200:
                available = response.json().get("models", [])
                model_base = model.split(":")[0]
                return any(
                    m.get("name", "").startswith(model_base)
                    for m in available
                )
        except Exception as e:
            logger.debug(f"Model availability check failed: {e}")

        return False

    def clear_cache(self):
        """Clear cached model clients.

        Useful when switching backends or for memory management.
        """
        self._model_clients.clear()
        logger.info("Cleared model client cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached model clients.

        Returns:
            Cache statistics dictionary
        """
        return {
            "cached_models": list(self._model_clients.keys()),
            "cache_size": len(self._model_clients),
            "current_default": self._current_model,
        }

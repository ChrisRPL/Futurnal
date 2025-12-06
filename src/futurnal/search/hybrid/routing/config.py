"""LLM Backend Configuration for Query Routing.

Provides configuration for LLM-based intent classification with support for:
- Ollama backend (recommended, 800x speedup over HuggingFace)
- HuggingFace fallback for environments without Ollama
- Dynamic model selection based on query characteristics
- Runtime model switching without restart

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Environment Variables:
- FUTURNAL_LLM_BACKEND: Backend selection (ollama|hf|auto)
- FUTURNAL_PRODUCTION_LLM: Model selection (llama3.1|phi3|qwen|bielik|kimi|auto)
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

Option B Compliance:
- Ghost model FROZEN: Model selection, not fine-tuning
- Supports multiple specialized models for different query types
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LLMBackendType(str, Enum):
    """LLM backend selection.

    Values:
        OLLAMA: Ollama C++ backend (recommended, 10-100x faster)
        HUGGINGFACE: HuggingFace transformers fallback
        AUTO: Auto-detect best available backend
    """

    OLLAMA = "ollama"
    HUGGINGFACE = "hf"
    AUTO = "auto"


@dataclass
class QueryRouterLLMConfig:
    """LLM configuration for query routing.

    Manages backend selection, model configuration, and environment
    variable handling for the query routing system.

    Uses same patterns as entity-relationship-extraction for consistency.

    Attributes:
        backend: LLM backend type
        model: Model name or alias
        ollama_base_url: Ollama server URL
        timeout_seconds: Request timeout

    Class Attributes:
        BACKEND_ENV: Environment variable for backend selection
        MODEL_ENV: Environment variable for model selection
        OLLAMA_URL_ENV: Environment variable for Ollama URL
    """

    # Environment variable names
    BACKEND_ENV: str = field(default="FUTURNAL_LLM_BACKEND", init=False)
    MODEL_ENV: str = field(default="FUTURNAL_PRODUCTION_LLM", init=False)
    OLLAMA_URL_ENV: str = field(default="OLLAMA_BASE_URL", init=False)

    # Default values
    DEFAULT_OLLAMA_URL: str = field(default="http://localhost:11434", init=False)
    DEFAULT_MODEL: str = field(default="llama3.1", init=False)

    # Latency targets (milliseconds)
    OLLAMA_LATENCY_TARGET_MS: int = field(default=100, init=False)
    HF_LATENCY_TARGET_MS: int = field(default=500, init=False)

    # Instance configuration
    backend: Optional[LLMBackendType] = None
    model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    timeout_seconds: int = 5

    # Model recommendations by use case
    INTENT_MODELS: Dict[str, str] = field(
        default_factory=lambda: {
            "fast": "phi3",           # Phi-3 Mini 3.8B (~4GB VRAM) - CI/CD, testing
            "production": "llama3.1", # Llama 3.1 8B (~8GB VRAM) - recommended default
            "advanced": "qwen",       # Qwen 2.5 32B (~16GB VRAM) - complex queries
            "reasoning": "kimi",      # Kimi-K2-Thinking (~16GB) - advanced reasoning
            "polish": "bielik",       # Bielik 4.5B (~5GB) - Polish language
            "unrestricted": "gpt-oss",# GPT-OSS-20B (~12GB) - unrestricted content
        },
        init=False,
    )

    # Short name aliases mapping to Ollama model names
    MODEL_ALIASES: Dict[str, str] = field(
        default_factory=lambda: {
            # Fast models
            "phi3": "phi3:mini",
            "fast": "phi3:mini",
            # Production models
            "llama3.1": "llama3.1:8b-instruct-q4_0",
            "llama": "llama3.3:70b-instruct-q4_0",
            # Specialized models
            "qwen": "qwen2.5-coder:32b-instruct-q4_0",
            "bielik": "bielik:4.5b-instruct-q4_0",
            "kimi": "kimi-k2:thinking",
            "k2": "kimi-k2:thinking",
            "gpt-oss": "gpt-oss:20b-derestricted",
            # Auto-selection
            "auto": "auto",
        },
        init=False,
    )

    # HuggingFace model name mapping
    OLLAMA_MODEL_MAP: Dict[str, str] = field(
        default_factory=lambda: {
            "microsoft/Phi-3-mini-4k-instruct": "phi3:mini",
            "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b-instruct-q4_0",
            "meta-llama/Llama-3.3-70B-Instruct": "llama3.3:70b-instruct-q4_0",
            "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen2.5-coder:32b-instruct-q4_0",
            "speakleash/Bielik-4.5B-v3.0-Instruct": "bielik:4.5b-instruct-q4_0",
            "moonshotai/Kimi-K2-Thinking": "kimi-k2:thinking",
            "ArliAI/gpt-oss-20b-Derestricted": "gpt-oss:20b-derestricted",
        },
        init=False,
    )

    # Polish language indicators
    POLISH_INDICATORS: tuple = field(
        default_factory=lambda: (
            "co", "jak", "kiedy", "gdzie", "dlaczego", "który",
            "czym", "jaki", "ile", "czy", "oraz", "jest", "był",
            "będzie", "mogę", "można", "projekt", "spotkanie",
        ),
        init=False,
    )

    # Advanced reasoning patterns
    REASONING_PATTERNS: tuple = field(
        default_factory=lambda: (
            "why did", "what caused", "analyze", "compare",
            "explain the relationship", "what are the implications",
            "how does this affect", "what would happen if",
            "reasoning", "logic", "hypothesis", "Bradford Hill",
            "causal chain", "root cause", "consequence",
        ),
        init=False,
    )

    # Code-related patterns
    CODE_PATTERNS: tuple = field(
        default_factory=lambda: (
            "function", "class", "def ", "import", "code",
            "implementation", "algorithm", "bug", "error",
            "syntax", "api", "method", "variable", "parameter",
        ),
        init=False,
    )

    def __post_init__(self):
        """Initialize from environment if not explicitly set."""
        if self.backend is None:
            self.backend = self.get_backend()
        if self.model is None:
            self.model = self.get_model_name()
        if self.ollama_base_url is None:
            self.ollama_base_url = os.getenv(
                self.OLLAMA_URL_ENV,
                self.DEFAULT_OLLAMA_URL,
            )

    def get_backend(self) -> LLMBackendType:
        """Get configured backend with auto-detection.

        Returns:
            LLMBackendType based on environment or auto-detection
        """
        backend_str = os.getenv(self.BACKEND_ENV, "auto")

        if backend_str == "auto":
            return self._auto_detect_backend()

        try:
            return LLMBackendType(backend_str)
        except ValueError:
            logger.warning(
                f"Unknown backend '{backend_str}', falling back to auto-detection"
            )
            return self._auto_detect_backend()

    def _auto_detect_backend(self) -> LLMBackendType:
        """Auto-detect best available backend.

        Checks Ollama availability with a quick health check.

        Returns:
            OLLAMA if available, otherwise HUGGINGFACE
        """
        try:
            import requests

            ollama_url = os.getenv(self.OLLAMA_URL_ENV, self.DEFAULT_OLLAMA_URL)
            response = requests.get(
                f"{ollama_url}/api/tags",
                timeout=1,
            )
            if response.status_code == 200:
                logger.info("Ollama detected, using fast backend")
                return LLMBackendType.OLLAMA
        except Exception:
            pass

        logger.info("Ollama not available, using HuggingFace backend")
        return LLMBackendType.HUGGINGFACE

    def get_model_name(self) -> str:
        """Get configured model name.

        Resolves environment variable to Ollama model name.

        Returns:
            Ollama-compatible model name
        """
        model = os.getenv(self.MODEL_ENV, self.DEFAULT_MODEL)

        # Check if it's a use-case alias
        if model in self.INTENT_MODELS:
            model = self.INTENT_MODELS[model]

        # Resolve to Ollama model name
        if model in self.MODEL_ALIASES:
            model = self.MODEL_ALIASES[model]

        return model

    def get_model_for_query(
        self,
        query: str,
        detected_language: Optional[str] = None,
    ) -> str:
        """Select optimal model based on query characteristics.

        Implements dynamic model switching:
        - Polish queries -> Bielik 4.5B
        - Complex reasoning -> Kimi-K2-Thinking
        - Code queries -> Qwen 2.5 Coder
        - Default -> Llama 3.1 8B

        Args:
            query: User query string
            detected_language: ISO language code if known

        Returns:
            Ollama model name to use
        """
        # Check if explicit model is set via environment
        explicit_model = os.getenv(self.MODEL_ENV)
        if explicit_model and explicit_model != "auto":
            return self.MODEL_ALIASES.get(explicit_model, explicit_model)

        # Language-based selection
        if detected_language == "pl" or self._is_polish_query(query):
            logger.debug("Polish query detected, selecting Bielik")
            return self.MODEL_ALIASES["bielik"]

        # Complexity-based selection
        if self._requires_advanced_reasoning(query):
            logger.debug("Complex reasoning query, selecting Kimi-K2")
            return self.MODEL_ALIASES["kimi"]

        # Code-related queries
        if self._is_code_query(query):
            logger.debug("Code-related query, selecting Qwen")
            return self.MODEL_ALIASES["qwen"]

        # Default to production model
        return self.MODEL_ALIASES.get("llama3.1", "llama3.1:8b-instruct-q4_0")

    def _is_polish_query(self, query: str) -> bool:
        """Detect if query is in Polish.

        Uses keyword matching for common Polish words.

        Args:
            query: Query string

        Returns:
            True if likely Polish
        """
        query_lower = query.lower()
        polish_count = sum(
            1 for word in self.POLISH_INDICATORS
            if word in query_lower
        )
        return polish_count >= 2

    def _requires_advanced_reasoning(self, query: str) -> bool:
        """Detect if query requires advanced reasoning (Kimi-K2).

        Looks for causal, analytical, and hypothesis-related patterns.

        Args:
            query: Query string

        Returns:
            True if requires advanced reasoning
        """
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.REASONING_PATTERNS)

    def _is_code_query(self, query: str) -> bool:
        """Detect if query is code-related.

        Args:
            query: Query string

        Returns:
            True if code-related
        """
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.CODE_PATTERNS)

    def get_ollama_url(self) -> str:
        """Get Ollama server URL.

        Returns:
            Ollama base URL
        """
        return self.ollama_base_url or os.getenv(
            self.OLLAMA_URL_ENV,
            self.DEFAULT_OLLAMA_URL,
        )

    def to_dict(self) -> Dict[str, str]:
        """Export configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "backend": self.backend.value if self.backend else "auto",
            "model": self.model or self.DEFAULT_MODEL,
            "ollama_base_url": self.get_ollama_url(),
            "timeout_seconds": str(self.timeout_seconds),
        }

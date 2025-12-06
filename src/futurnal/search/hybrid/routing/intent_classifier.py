"""Intent Classification for Query Routing.

Provides LLM-based intent classification to route queries to optimal
retrieval strategies.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Components:
- IntentClassifierLLM: Protocol for intent classifiers
- OllamaIntentClassifier: Fast classification via Ollama (<100ms target)
- HuggingFaceIntentClassifier: Fallback classifier (<500ms target)
- get_intent_classifier: Factory function with auto-detection

Intent Types:
- lookup: Specific entity/fact lookup
- exploratory: Broad exploration
- temporal: Time-based queries
- causal: Causation queries

Option B Compliance:
- Ghost model FROZEN: Classification only, no fine-tuning
- Supports temporal-first design
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import requests

from futurnal.search.hybrid.routing.config import (
    LLMBackendType,
    QueryRouterLLMConfig,
)

logger = logging.getLogger(__name__)


# Classification prompt template
INTENT_CLASSIFICATION_PROMPT = """Classify the query intent. Respond in JSON only.

Query: "{query}"

Intent types:
- lookup: Specific entity or fact lookup ("What is X?", "Who is Y?")
- exploratory: Broad exploration or discovery ("Tell me about...", "What do I know about...")
- temporal: Time-based query ("when", "before", "after", "between dates")
- causal: Causation query ("why", "what caused", "what led to", "because")

Output format:
{{"intent": "lookup|exploratory|temporal|causal", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

JSON:"""


@runtime_checkable
class IntentClassifierLLM(Protocol):
    """Protocol for intent classification LLM.

    Defines the interface that all intent classifiers must implement.
    Both OllamaIntentClassifier and HuggingFaceIntentClassifier
    implement this protocol.
    """

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent.

        Args:
            query: User query string

        Returns:
            Dictionary with:
                intent: "lookup" | "exploratory" | "temporal" | "causal"
                confidence: 0.0 to 1.0
                reasoning: Brief explanation
        """
        ...


class BaseIntentClassifier(ABC):
    """Abstract base class for intent classifiers.

    Provides common functionality for parsing and validating
    intent classification results.
    """

    VALID_INTENTS = {"lookup", "exploratory", "temporal", "causal"}
    DEFAULT_INTENT = "exploratory"
    DEFAULT_CONFIDENCE = 0.5

    @abstractmethod
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent."""
        pass

    def _build_classification_prompt(self, query: str) -> str:
        """Build intent classification prompt.

        Args:
            query: User query

        Returns:
            Formatted prompt for LLM
        """
        return INTENT_CLASSIFICATION_PROMPT.format(query=query)

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured intent.

        Handles JSON parsing with fallback for text responses.

        Args:
            response: Raw LLM response

        Returns:
            Parsed intent dictionary
        """
        try:
            # Clean up response - handle markdown code blocks
            cleaned = response.strip()
            if "```" in cleaned:
                # Extract content between code blocks
                parts = cleaned.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        cleaned = part
                        break

            # Find JSON object in response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                cleaned = cleaned[start:end]

            result = json.loads(cleaned)

            # Validate and normalize intent
            intent = result.get("intent", "").lower()
            if intent not in self.VALID_INTENTS:
                intent = self.DEFAULT_INTENT

            # Ensure confidence is float in valid range
            confidence = float(result.get("confidence", self.DEFAULT_CONFIDENCE))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "intent": intent,
                "confidence": confidence,
                "reasoning": result.get("reasoning", "Classified by LLM"),
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"JSON parsing failed: {e}, falling back to text extraction")
            return self._extract_intent_from_text(response)

    def _extract_intent_from_text(self, response: str) -> Dict[str, Any]:
        """Extract intent from text response when JSON parsing fails.

        Args:
            response: Raw LLM response

        Returns:
            Best-effort intent extraction
        """
        response_lower = response.lower()

        # Check for intent keywords in order of specificity
        if "temporal" in response_lower or "time" in response_lower:
            return {
                "intent": "temporal",
                "confidence": 0.6,
                "reasoning": "Extracted 'temporal' from text response",
            }
        if "causal" in response_lower or "cause" in response_lower:
            return {
                "intent": "causal",
                "confidence": 0.6,
                "reasoning": "Extracted 'causal' from text response",
            }
        if "lookup" in response_lower or "specific" in response_lower:
            return {
                "intent": "lookup",
                "confidence": 0.6,
                "reasoning": "Extracted 'lookup' from text response",
            }

        # Default fallback
        return {
            "intent": self.DEFAULT_INTENT,
            "confidence": self.DEFAULT_CONFIDENCE,
            "reasoning": "Default fallback - could not parse response",
        }


class OllamaIntentClassifier(BaseIntentClassifier):
    """Fast intent classification via Ollama.

    Achieves <100ms latency for intent classification using
    Ollama's optimized C++ backend.

    Uses the same infrastructure as OllamaLLMClient in
    extraction/ollama_client.py for consistency.

    Attributes:
        model: Ollama model name
        base_url: Ollama server URL
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model: str = "llama3.1:8b-instruct-q4_0",
        base_url: str = "http://localhost:11434",
        timeout: int = 5,
    ):
        """Initialize Ollama intent classifier.

        Args:
            model: Ollama model name (default: llama3.1:8b)
            base_url: Ollama server URL
            timeout: Request timeout in seconds (short for fast classification)
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        logger.info(f"Initialized OllamaIntentClassifier with model: {model}")

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent with Ollama backend.

        Target latency: <100ms

        Args:
            query: User query string

        Returns:
            Intent classification result
        """
        start_time = time.time()
        prompt = self._build_classification_prompt(query)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for classification
                        "num_predict": 100,  # Short response
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = self._parse_intent_response(response.json()["response"])

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Intent classification: {result['intent']} "
                f"(confidence: {result['confidence']:.2f}, latency: {latency_ms:.1f}ms)"
            )

            # Add latency to result for monitoring
            result["latency_ms"] = latency_ms

            return result

        except requests.exceptions.Timeout:
            logger.warning(f"Ollama request timed out after {self.timeout}s")
            return self._fallback_classification(query)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama request failed: {e}")
            return self._fallback_classification(query)

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Fallback classification when Ollama fails.

        Uses keyword-based heuristics.

        Args:
            query: User query

        Returns:
            Heuristic-based classification
        """
        query_lower = query.lower()

        # Temporal keywords
        temporal_keywords = ["when", "before", "after", "during", "between", "date", "time"]
        if any(kw in query_lower for kw in temporal_keywords):
            return {
                "intent": "temporal",
                "confidence": 0.7,
                "reasoning": "Keyword-based fallback: temporal keywords detected",
            }

        # Causal keywords
        causal_keywords = ["why", "cause", "because", "led to", "result", "effect"]
        if any(kw in query_lower for kw in causal_keywords):
            return {
                "intent": "causal",
                "confidence": 0.7,
                "reasoning": "Keyword-based fallback: causal keywords detected",
            }

        # Lookup keywords
        lookup_keywords = ["what is", "who is", "define", "find", "get"]
        if any(kw in query_lower for kw in lookup_keywords):
            return {
                "intent": "lookup",
                "confidence": 0.7,
                "reasoning": "Keyword-based fallback: lookup keywords detected",
            }

        # Default to exploratory
        return {
            "intent": "exploratory",
            "confidence": 0.5,
            "reasoning": "Keyword-based fallback: defaulting to exploratory",
        }


class HuggingFaceIntentClassifier(BaseIntentClassifier):
    """Fallback intent classifier using HuggingFace transformers.

    Slower than Ollama (~500ms) but works without Ollama installation.
    Uses lazy model loading to avoid startup overhead.

    Uses patterns from QuantizedLocalLLM in extraction/local_llm_client.py

    Attributes:
        model_name: HuggingFace model name
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    ):
        """Initialize HuggingFace intent classifier.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._pipeline = None

        logger.info(f"Initialized HuggingFaceIntentClassifier with model: {model_name}")

    def _load_model(self):
        """Lazy load model on first use.

        Defers heavy model loading until actually needed.
        """
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            import torch

            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            logger.info(f"Loading HuggingFace model on {device}")

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                max_new_tokens=100,
            )

            logger.info("HuggingFace model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify intent using HuggingFace model.

        Target latency: <500ms

        Args:
            query: User query string

        Returns:
            Intent classification result
        """
        start_time = time.time()

        try:
            self._load_model()

            prompt = self._build_classification_prompt(query)

            outputs = self._pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )

            generated_text = outputs[0]["generated_text"]
            # Extract only the generated part (after the prompt)
            response = generated_text[len(prompt):].strip()

            result = self._parse_intent_response(response)

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"HF Intent classification: {result['intent']} "
                f"(confidence: {result['confidence']:.2f}, latency: {latency_ms:.1f}ms)"
            )

            result["latency_ms"] = latency_ms

            return result

        except Exception as e:
            logger.error(f"HuggingFace classification failed: {e}")
            return self._fallback_classification(query)

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Same fallback logic as Ollama classifier."""
        return OllamaIntentClassifier._fallback_classification(self, query)


def get_intent_classifier(
    config: Optional[QueryRouterLLMConfig] = None,
) -> IntentClassifierLLM:
    """Factory function to get appropriate intent classifier.

    Auto-detects Ollama availability with HuggingFace fallback.

    Uses patterns from get_test_llm_client() in local_llm_client.py

    Args:
        config: Optional configuration. If None, uses default config.

    Returns:
        Intent classifier instance (Ollama or HuggingFace)
    """
    if config is None:
        config = QueryRouterLLMConfig()

    backend = config.backend or config.get_backend()
    model = config.model or config.get_model_name()

    if backend == LLMBackendType.OLLAMA:
        return OllamaIntentClassifier(
            model=model,
            base_url=config.get_ollama_url(),
            timeout=config.timeout_seconds,
        )

    elif backend == LLMBackendType.HUGGINGFACE:
        # Map Ollama model name to HuggingFace name
        hf_model = model
        for hf_name, ollama_name in config.OLLAMA_MODEL_MAP.items():
            if ollama_name == model or model in ollama_name:
                hf_model = hf_name
                break

        return HuggingFaceIntentClassifier(model_name=hf_model)

    else:
        # AUTO: Try Ollama first, fallback to HuggingFace
        try:
            response = requests.get(
                f"{config.get_ollama_url()}/api/tags",
                timeout=1,
            )
            if response.status_code == 200:
                logger.info("Auto-detected Ollama, using fast backend")
                return OllamaIntentClassifier(
                    model=model,
                    base_url=config.get_ollama_url(),
                    timeout=config.timeout_seconds,
                )
        except Exception:
            pass

        logger.info("Ollama not available, using HuggingFace fallback")
        return HuggingFaceIntentClassifier(
            model_name="microsoft/Phi-3-mini-4k-instruct"
        )


def ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is installed and running.

    Convenience function for quick availability check.

    Args:
        base_url: Ollama server URL

    Returns:
        True if Ollama is available
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

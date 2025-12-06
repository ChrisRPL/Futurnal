"""Tests for Intent Classification.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Success Metrics:
- Intent classification accuracy >85%
- Intent classification latency <100ms (Ollama)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from futurnal.search.hybrid.routing.config import (
    LLMBackendType,
    QueryRouterLLMConfig,
)
from futurnal.search.hybrid.routing.intent_classifier import (
    BaseIntentClassifier,
    HuggingFaceIntentClassifier,
    OllamaIntentClassifier,
    get_intent_classifier,
    ollama_available,
)


class TestBaseIntentClassifier:
    """Tests for BaseIntentClassifier."""

    def test_valid_intents(self):
        """Valid intents are defined."""
        assert "lookup" in BaseIntentClassifier.VALID_INTENTS
        assert "exploratory" in BaseIntentClassifier.VALID_INTENTS
        assert "temporal" in BaseIntentClassifier.VALID_INTENTS
        assert "causal" in BaseIntentClassifier.VALID_INTENTS

    def test_default_intent(self):
        """Default intent is exploratory."""
        assert BaseIntentClassifier.DEFAULT_INTENT == "exploratory"


class TestOllamaIntentClassifier:
    """Tests for OllamaIntentClassifier."""

    def test_initialization(self):
        """Classifier initializes correctly."""
        classifier = OllamaIntentClassifier(
            model="phi3:mini",
            base_url="http://localhost:11434",
            timeout=5,
        )

        assert classifier.model == "phi3:mini"
        assert classifier.base_url == "http://localhost:11434"
        assert classifier.timeout == 5

    def test_classification_prompt_building(self):
        """Classification prompt is built correctly."""
        classifier = OllamaIntentClassifier()
        prompt = classifier._build_classification_prompt("What happened yesterday?")

        assert "Query:" in prompt
        assert "What happened yesterday?" in prompt
        assert "lookup" in prompt
        assert "exploratory" in prompt
        assert "temporal" in prompt
        assert "causal" in prompt
        assert "JSON" in prompt

    def test_response_parsing_valid_json(self):
        """Valid JSON response is parsed correctly."""
        classifier = OllamaIntentClassifier()

        response = '{"intent": "temporal", "confidence": 0.9, "reasoning": "time keywords"}'
        result = classifier._parse_intent_response(response)

        assert result["intent"] == "temporal"
        assert result["confidence"] == 0.9
        assert "time keywords" in result["reasoning"]

    def test_response_parsing_with_markdown(self):
        """JSON in markdown code block is parsed correctly."""
        classifier = OllamaIntentClassifier()

        response = '```json\n{"intent": "causal", "confidence": 0.85, "reasoning": "why keyword"}\n```'
        result = classifier._parse_intent_response(response)

        assert result["intent"] == "causal"
        assert result["confidence"] == 0.85

    def test_response_parsing_invalid_json(self):
        """Invalid JSON falls back to text extraction."""
        classifier = OllamaIntentClassifier()

        response = "This query is about temporal events"
        result = classifier._parse_intent_response(response)

        assert result["intent"] == "temporal"
        assert result["confidence"] < 1.0

    def test_response_parsing_invalid_intent(self):
        """Invalid intent falls back to default."""
        classifier = OllamaIntentClassifier()

        response = '{"intent": "invalid_type", "confidence": 0.9}'
        result = classifier._parse_intent_response(response)

        assert result["intent"] == "exploratory"

    def test_fallback_classification_temporal(self):
        """Fallback correctly detects temporal queries."""
        classifier = OllamaIntentClassifier()

        result = classifier._fallback_classification("When did the meeting happen?")
        assert result["intent"] == "temporal"
        assert "fallback" in result["reasoning"].lower()

    def test_fallback_classification_causal(self):
        """Fallback correctly detects causal queries."""
        classifier = OllamaIntentClassifier()

        result = classifier._fallback_classification("Why did the project fail?")
        assert result["intent"] == "causal"

    def test_fallback_classification_lookup(self):
        """Fallback correctly detects lookup queries."""
        classifier = OllamaIntentClassifier()

        result = classifier._fallback_classification("What is the API endpoint?")
        assert result["intent"] == "lookup"

    def test_fallback_classification_default(self):
        """Fallback defaults to exploratory."""
        classifier = OllamaIntentClassifier()

        result = classifier._fallback_classification("Tell me more")
        assert result["intent"] == "exploratory"

    def test_classify_intent_with_mock(self, mock_ollama_generate):
        """Classification works with mocked Ollama."""
        classifier = OllamaIntentClassifier()

        result = classifier.classify_intent("What happened in January?")

        assert result["intent"] == "temporal"
        assert result["confidence"] == 0.9
        assert "latency_ms" in result


class TestHuggingFaceIntentClassifier:
    """Tests for HuggingFaceIntentClassifier."""

    def test_initialization(self):
        """Classifier initializes correctly."""
        classifier = HuggingFaceIntentClassifier(
            model_name="microsoft/Phi-3-mini-4k-instruct"
        )

        assert classifier.model_name == "microsoft/Phi-3-mini-4k-instruct"
        assert classifier._model is None
        assert classifier._pipeline is None

    def test_fallback_classification(self):
        """Fallback classification works."""
        classifier = HuggingFaceIntentClassifier()

        result = classifier._fallback_classification("When was the meeting?")
        assert result["intent"] == "temporal"


class TestGetIntentClassifier:
    """Tests for get_intent_classifier factory function."""

    def test_returns_ollama_when_available(self, mock_ollama_available):
        """Returns OllamaIntentClassifier when Ollama is available."""
        config = QueryRouterLLMConfig(backend=LLMBackendType.OLLAMA)
        classifier = get_intent_classifier(config)

        assert isinstance(classifier, OllamaIntentClassifier)

    def test_returns_huggingface_when_configured(self):
        """Returns HuggingFaceIntentClassifier when configured."""
        config = QueryRouterLLMConfig(backend=LLMBackendType.HUGGINGFACE)
        classifier = get_intent_classifier(config)

        assert isinstance(classifier, HuggingFaceIntentClassifier)

    def test_auto_detection_ollama_available(self, mock_ollama_available):
        """Auto-detection returns Ollama when available."""
        config = QueryRouterLLMConfig(backend=LLMBackendType.AUTO)
        classifier = get_intent_classifier(config)

        assert isinstance(classifier, OllamaIntentClassifier)

    def test_auto_detection_ollama_unavailable(self, mock_ollama_unavailable):
        """Auto-detection returns HuggingFace when Ollama unavailable."""
        config = QueryRouterLLMConfig(backend=LLMBackendType.AUTO)
        classifier = get_intent_classifier(config)

        assert isinstance(classifier, HuggingFaceIntentClassifier)


class TestOllamaAvailable:
    """Tests for ollama_available utility function."""

    def test_returns_true_when_available(self, mock_ollama_available):
        """Returns True when Ollama responds."""
        assert ollama_available() is True

    def test_returns_false_when_unavailable(self, mock_ollama_unavailable):
        """Returns False when Ollama doesn't respond."""
        assert ollama_available() is False


class TestIntentClassificationAccuracy:
    """Tests for intent classification accuracy.

    Target: >85% accuracy per production plan.
    """

    @pytest.fixture
    def classifier_with_mock(self, mock_ollama_generate):
        """Classifier with mocked backend."""
        return OllamaIntentClassifier()

    def test_temporal_queries_classification(
        self, sample_temporal_queries, mock_ollama_generate
    ):
        """Temporal queries are classified correctly.

        Note: With real LLM, accuracy should be >85%.
        Mock always returns temporal, so 100% here.
        """
        classifier = OllamaIntentClassifier()

        correct = 0
        for query in sample_temporal_queries:
            result = classifier.classify_intent(query)
            if result["intent"] == "temporal":
                correct += 1

        accuracy = correct / len(sample_temporal_queries)
        assert accuracy >= 0.8, f"Temporal accuracy {accuracy} below 80%"

    def test_classification_returns_confidence(self, mock_ollama_generate):
        """Classification includes confidence score."""
        classifier = OllamaIntentClassifier()

        result = classifier.classify_intent("When was the meeting?")

        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

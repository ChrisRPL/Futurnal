"""Tests for Query Router LLM Configuration.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from futurnal.search.hybrid.routing.config import (
    LLMBackendType,
    QueryRouterLLMConfig,
)


class TestLLMBackendType:
    """Tests for LLMBackendType enum."""

    def test_backend_values(self):
        """All expected backend types exist."""
        assert LLMBackendType.OLLAMA.value == "ollama"
        assert LLMBackendType.HUGGINGFACE.value == "hf"
        assert LLMBackendType.AUTO.value == "auto"

    def test_backend_from_string(self):
        """Backend can be created from string."""
        assert LLMBackendType("ollama") == LLMBackendType.OLLAMA
        assert LLMBackendType("hf") == LLMBackendType.HUGGINGFACE
        assert LLMBackendType("auto") == LLMBackendType.AUTO


class TestQueryRouterLLMConfig:
    """Tests for QueryRouterLLMConfig."""

    def test_default_initialization(self):
        """Config initializes with defaults."""
        config = QueryRouterLLMConfig()

        assert config.DEFAULT_MODEL == "llama3.1"
        assert config.DEFAULT_OLLAMA_URL == "http://localhost:11434"
        assert config.OLLAMA_LATENCY_TARGET_MS == 100
        assert config.HF_LATENCY_TARGET_MS == 500

    def test_explicit_backend_setting(self):
        """Explicit backend setting works."""
        config = QueryRouterLLMConfig(backend=LLMBackendType.OLLAMA)
        assert config.backend == LLMBackendType.OLLAMA

        config = QueryRouterLLMConfig(backend=LLMBackendType.HUGGINGFACE)
        assert config.backend == LLMBackendType.HUGGINGFACE

    def test_explicit_model_setting(self):
        """Explicit model setting works."""
        config = QueryRouterLLMConfig(model="phi3:mini")
        assert config.model == "phi3:mini"

    def test_model_aliases(self, default_config):
        """Model aliases resolve correctly."""
        aliases = default_config.MODEL_ALIASES

        assert "phi3" in aliases
        assert "llama3.1" in aliases
        assert "qwen" in aliases
        assert "bielik" in aliases
        assert "kimi" in aliases
        assert "gpt-oss" in aliases

    def test_intent_models(self, default_config):
        """Intent models are defined."""
        models = default_config.INTENT_MODELS

        assert models["fast"] == "phi3"
        assert models["production"] == "llama3.1"
        assert models["advanced"] == "qwen"
        assert models["reasoning"] == "kimi"
        assert models["polish"] == "bielik"

    @patch.dict(os.environ, {"FUTURNAL_LLM_BACKEND": "ollama"})
    def test_backend_from_environment(self):
        """Backend reads from environment variable."""
        config = QueryRouterLLMConfig()
        assert config.get_backend() == LLMBackendType.OLLAMA

    @patch.dict(os.environ, {"FUTURNAL_LLM_BACKEND": "hf"})
    def test_hf_backend_from_environment(self):
        """HuggingFace backend from environment."""
        config = QueryRouterLLMConfig()
        assert config.get_backend() == LLMBackendType.HUGGINGFACE

    @patch.dict(os.environ, {"FUTURNAL_PRODUCTION_LLM": "phi3"})
    def test_model_from_environment(self):
        """Model reads from environment variable."""
        config = QueryRouterLLMConfig()
        model = config.get_model_name()
        assert "phi3" in model

    def test_polish_query_detection(self, default_config):
        """Polish query detection works."""
        # Polish queries
        assert default_config._is_polish_query("Co to jest machine learning?")
        assert default_config._is_polish_query("Kiedy było spotkanie?")
        assert default_config._is_polish_query("Jak to działa oraz dlaczego?")

        # Non-Polish queries
        assert not default_config._is_polish_query("What is machine learning?")
        assert not default_config._is_polish_query("When was the meeting?")

    def test_reasoning_query_detection(self, default_config):
        """Advanced reasoning detection works."""
        # Reasoning queries
        assert default_config._requires_advanced_reasoning("Why did the project fail?")
        assert default_config._requires_advanced_reasoning("What caused the delay?")
        assert default_config._requires_advanced_reasoning("Analyze the root cause")
        assert default_config._requires_advanced_reasoning("What are the implications?")

        # Non-reasoning queries
        assert not default_config._requires_advanced_reasoning("What is the status?")
        assert not default_config._requires_advanced_reasoning("Find the document")

    def test_code_query_detection(self, default_config):
        """Code query detection works."""
        # Code queries
        assert default_config._is_code_query("How does the function work?")
        assert default_config._is_code_query("What is the class structure?")
        assert default_config._is_code_query("Find the bug in the code")
        assert default_config._is_code_query("Show me the API implementation")

        # Non-code queries
        assert not default_config._is_code_query("What happened yesterday?")
        assert not default_config._is_code_query("Who attended the meeting?")

    def test_model_for_polish_query(self, default_config):
        """Polish queries route to Bielik model."""
        model = default_config.get_model_for_query("Co to jest machine learning?")
        assert "bielik" in model

    def test_model_for_reasoning_query(self, default_config):
        """Reasoning queries route to Kimi model."""
        model = default_config.get_model_for_query("Why did the system fail?")
        assert "kimi" in model

    def test_model_for_code_query(self, default_config):
        """Code queries route to Qwen model."""
        model = default_config.get_model_for_query("How does the function work?")
        assert "qwen" in model

    def test_model_for_default_query(self, default_config):
        """Default queries route to Llama model."""
        model = default_config.get_model_for_query("What is the status?")
        assert "llama" in model

    def test_to_dict(self, default_config):
        """Config exports to dictionary."""
        data = default_config.to_dict()

        assert "backend" in data
        assert "model" in data
        assert "ollama_base_url" in data
        assert "timeout_seconds" in data

    def test_get_ollama_url(self, default_config):
        """Ollama URL retrieval works."""
        url = default_config.get_ollama_url()
        assert url.startswith("http")
        assert "11434" in url

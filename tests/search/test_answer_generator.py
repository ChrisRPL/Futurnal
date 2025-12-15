"""Tests for AnswerGenerator module.

Step 02: LLM Answer Generation
Production Plan Reference:
docs/phase-1/p1-production-steps.md

Research Foundation:
- CausalRAG (ACL 2025): Causal-aware generation with citations
- LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.search.answer_generator import (
    ANSWER_MODELS,
    DEFAULT_MODEL,
    AnswerGenerator,
    AnswerGeneratorConfig,
    GeneratedAnswer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def answer_config() -> AnswerGeneratorConfig:
    """Default answer generator configuration for testing."""
    return AnswerGeneratorConfig(
        model_name="llama3.1:8b-instruct-q4_0",
        temperature=0.3,
        max_tokens=512,
        context_limit=10,
    )


@pytest.fixture
def mock_ollama_pool() -> MagicMock:
    """Mock OllamaConnectionPool for testing."""
    pool = MagicMock()
    pool.generate = AsyncMock(return_value="This is a test answer based on the context. [Source: doc1.md]")
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    pool.is_initialized = True
    pool.is_healthy = True
    return pool


@pytest.fixture
def sample_context() -> List[Dict[str, Any]]:
    """Sample context for testing - matches expected API format."""
    return [
        {
            "content": "The quarterly review meeting discussed project progress and upcoming milestones.",
            "metadata": {
                "source": "meetings/q1-review.md",
                "label": "Q1 Review Notes",
            },
        },
        {
            "content": "Project Alpha aims to improve user experience through AI-powered features.",
            "metadata": {
                "source": "docs/project-alpha-spec.md",
                "label": "Project Alpha Spec",
            },
        },
        {
            "content": "Decision: Proceed with Phase 2 implementation. Approved by stakeholders.",
            "metadata": {
                "source": "decisions/phase2-approval.md",
            },
        },
    ]


@pytest.fixture
def sample_graph_context() -> Dict[str, Any]:
    """Sample graph context for testing CausalRAG integration."""
    return {
        "relationships": [
            {"type": "authored_by", "from_entity": "Q1 Review", "to_entity": "John"},
            {"type": "mentions", "from_entity": "Q1 Review", "to_entity": "Project Alpha"},
        ],
        "related_entities": [
            {"name": "Project Alpha", "id": "proj_1"},
            {"name": "John Smith", "id": "person_1"},
        ],
    }


# ---------------------------------------------------------------------------
# AnswerGeneratorConfig Tests
# ---------------------------------------------------------------------------


class TestAnswerGeneratorConfig:
    """Tests for AnswerGeneratorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AnswerGeneratorConfig()
        assert config.model_name == DEFAULT_MODEL
        assert config.temperature == 0.3
        assert config.max_tokens == 512
        assert config.context_limit == 10
        assert config.include_graph_context is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AnswerGeneratorConfig(
            model_name="phi3:mini",
            temperature=0.5,
            max_tokens=256,
            context_limit=5,
            include_graph_context=False,
        )
        assert config.model_name == "phi3:mini"
        assert config.temperature == 0.5
        assert config.max_tokens == 256
        assert config.context_limit == 5
        assert config.include_graph_context is False


# ---------------------------------------------------------------------------
# GeneratedAnswer Tests
# ---------------------------------------------------------------------------


class TestGeneratedAnswer:
    """Tests for GeneratedAnswer dataclass."""

    def test_creation(self):
        """Test GeneratedAnswer creation."""
        answer = GeneratedAnswer(
            answer="Test answer text",
            sources=["doc1.md", "doc2.md"],
            model="llama3.1:8b-instruct-q4_0",
            tokens_generated=50,
            generation_time_ms=1500,
        )
        assert answer.answer == "Test answer text"
        assert len(answer.sources) == 2
        assert answer.model == "llama3.1:8b-instruct-q4_0"
        assert answer.tokens_generated == 50
        assert answer.generation_time_ms == 1500

    def test_defaults(self):
        """Test GeneratedAnswer defaults."""
        answer = GeneratedAnswer(answer="Test")
        assert answer.sources == []
        assert answer.model == ""
        assert answer.tokens_generated == 0
        assert answer.generation_time_ms == 0.0


# ---------------------------------------------------------------------------
# ANSWER_MODELS Registry Tests
# ---------------------------------------------------------------------------


class TestAnswerModels:
    """Tests for ANSWER_MODELS registry."""

    def test_all_models_present(self):
        """Test that all expected models are in registry."""
        expected_models = [
            "phi3:mini",
            "llama3.1:8b-instruct-q4_0",
            "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0",
            "qwen2.5:7b-instruct",
            "mistral:7b-instruct",
        ]
        for model_id in expected_models:
            assert model_id in ANSWER_MODELS, f"Model {model_id} missing from registry"

    def test_model_info_complete(self):
        """Test that all models have required fields."""
        required_fields = ["label", "vram", "hint"]
        for model_id, info in ANSWER_MODELS.items():
            for field in required_fields:
                assert field in info, f"Model {model_id} missing {field}"

    def test_default_model_in_registry(self):
        """Test that default model is in the registry."""
        assert DEFAULT_MODEL in ANSWER_MODELS


# ---------------------------------------------------------------------------
# AnswerGenerator Tests
# ---------------------------------------------------------------------------


class TestAnswerGenerator:
    """Tests for AnswerGenerator class."""

    @pytest.mark.asyncio
    async def test_initialization_creates_pool(self, answer_config: AnswerGeneratorConfig):
        """Test that initialization creates pool if not provided."""
        with patch("futurnal.search.answer_generator.OllamaConnectionPool") as MockPool:
            mock_pool = MagicMock()
            mock_pool.initialize = AsyncMock()
            mock_pool.is_initialized = True
            MockPool.return_value = mock_pool

            generator = AnswerGenerator(config=answer_config)
            await generator.initialize()

            MockPool.assert_called_once()
            mock_pool.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_uses_provided_pool(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
    ):
        """Test that provided pool is used."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)
        await generator.initialize()

        # Should not create new pool
        assert generator._pool is mock_ollama_pool

    @pytest.mark.asyncio
    async def test_generate_answer_basic(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
        sample_context: List[Dict[str, Any]],
    ):
        """Test basic answer generation."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="What is Project Alpha?",
            context=sample_context,
        )

        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0
        assert result.model == answer_config.model_name
        assert result.generation_time_ms > 0
        mock_ollama_pool.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_assembly_includes_graph(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
        sample_context: List[Dict[str, Any]],
        sample_graph_context: Dict[str, Any],
    ):
        """Test that graph context is included in assembled context."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)

        context = generator._assemble_context(sample_context, sample_graph_context)

        # Should include content from results
        assert "quarterly review meeting" in context.lower()
        assert "project alpha" in context.lower()

        # Should include graph relationships (CausalRAG)
        assert "knowledge graph" in context.lower() or "related" in context.lower()
        assert "john" in context.lower()  # From relationships

    @pytest.mark.asyncio
    async def test_context_limit_respected(
        self,
        mock_ollama_pool: MagicMock,
    ):
        """Test that context is limited to config.context_limit docs."""
        config = AnswerGeneratorConfig(context_limit=2)
        generator = AnswerGenerator(config=config, pool=mock_ollama_pool)

        # Create more docs than limit with unique content
        many_docs = [
            {"content": f"UNIQUE_CONTENT_{i}_HERE", "metadata": {"source": f"doc{i}.md"}}
            for i in range(10)
        ]

        context = generator._assemble_context(many_docs)

        # Should only have first 2 docs content
        assert "UNIQUE_CONTENT_0_HERE" in context
        assert "UNIQUE_CONTENT_1_HERE" in context
        assert "UNIQUE_CONTENT_2_HERE" not in context  # Should be cut off

    def test_extract_sources_unique(
        self,
        answer_config: AnswerGeneratorConfig,
    ):
        """Test source extraction returns unique sources."""
        generator = AnswerGenerator(config=answer_config)

        # Add duplicate source
        context = [
            {"content": "A", "metadata": {"source": "doc.md", "label": "Doc"}},
            {"content": "B", "metadata": {"source": "doc.md", "label": "Doc"}},  # Duplicate
            {"content": "C", "metadata": {"source": "other.md"}},
        ]

        sources = generator._extract_sources(context)

        # Should have unique sources
        assert len(sources) == 2
        assert "Doc" in sources
        assert "other.md" in sources

    @pytest.mark.asyncio
    async def test_prompt_includes_system_rules(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
        sample_context: List[Dict[str, Any]],
    ):
        """Test that system prompt enforces hallucination prevention."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)
        await generator.initialize()

        await generator.generate_answer(
            query="Test query",
            context=sample_context,
        )

        # Check the call to generate
        call_args = mock_ollama_pool.generate.call_args

        # System prompt should contain anti-hallucination rules
        system_prompt = call_args.kwargs.get("system", "")
        assert "context" in system_prompt.lower()
        assert "source" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_model_override(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
        sample_context: List[Dict[str, Any]],
    ):
        """Test that model can be overridden per request."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="Test query",
            context=sample_context,
            model="phi3:mini",  # Override default model
        )

        call_args = mock_ollama_pool.generate.call_args
        assert call_args.kwargs.get("model") == "phi3:mini"
        assert result.model == "phi3:mini"

    @pytest.mark.asyncio
    async def test_empty_context_returns_answer(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
    ):
        """Test handling of empty context."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="Test query",
            context=[],
        )

        # Should still return an answer (LLM was called)
        assert isinstance(result, GeneratedAnswer)
        assert result.answer is not None

    def test_get_available_models(self):
        """Test static method returns model registry."""
        models = AnswerGenerator.get_available_models()
        assert models == ANSWER_MODELS

    def test_get_default_model(self):
        """Test static method returns default model."""
        default = AnswerGenerator.get_default_model()
        assert default == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Stream Generation Tests
# ---------------------------------------------------------------------------


class TestStreamGeneration:
    """Tests for streaming answer generation."""

    @pytest.mark.asyncio
    async def test_stream_answer_yields_tokens(
        self,
        answer_config: AnswerGeneratorConfig,
        sample_context: List[Dict[str, Any]],
    ):
        """Test that stream_answer yields tokens."""
        mock_pool = MagicMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.is_initialized = True

        # Mock stream_generate to be an async generator
        async def mock_stream_generate(*args, **kwargs):
            tokens = ["This ", "is ", "a ", "test ", "answer."]
            for token in tokens:
                yield token

        mock_pool.stream_generate = mock_stream_generate

        generator = AnswerGenerator(config=answer_config, pool=mock_pool)
        await generator.initialize()

        tokens = []
        async for token in generator.stream_answer(
            query="Test query",
            context=sample_context,
        ):
            tokens.append(token)

        assert len(tokens) == 5
        assert "".join(tokens) == "This is a test answer."


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in AnswerGenerator."""

    @pytest.mark.asyncio
    async def test_pool_connection_error(
        self,
        answer_config: AnswerGeneratorConfig,
        sample_context: List[Dict[str, Any]],
    ):
        """Test graceful handling of pool connection errors."""
        from futurnal.search.hybrid.performance.ollama_pool import OllamaConnectionError

        mock_pool = MagicMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.is_initialized = True
        mock_pool.generate = AsyncMock(side_effect=OllamaConnectionError("Connection failed"))

        generator = AnswerGenerator(config=answer_config, pool=mock_pool)
        await generator.initialize()

        # Should not raise, but return error response
        result = await generator.generate_answer(
            query="Test query",
            context=sample_context,
        )

        assert isinstance(result, GeneratedAnswer)
        # Answer should indicate error
        assert "couldn't generate" in result.answer.lower()
        assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_uninitialized_generator_auto_initializes(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
        sample_context: List[Dict[str, Any]],
    ):
        """Test that calling generate_answer auto-initializes if needed."""
        mock_ollama_pool.is_initialized = False

        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)

        # Don't call initialize, let generate_answer handle it
        result = await generator.generate_answer(
            query="Test query",
            context=sample_context,
        )

        # Should have auto-initialized
        mock_ollama_pool.initialize.assert_called()
        assert isinstance(result, GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_close_cleans_up(
        self,
        answer_config: AnswerGeneratorConfig,
        mock_ollama_pool: MagicMock,
    ):
        """Test that close properly cleans up resources."""
        generator = AnswerGenerator(config=answer_config, pool=mock_ollama_pool)
        await generator.initialize()

        assert generator._initialized is True

        await generator.close()

        mock_ollama_pool.close.assert_called_once()
        assert generator._initialized is False

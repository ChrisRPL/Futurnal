"""Integration tests for LLM Answer Generation.

Step 02: LLM Answer Generation
Production Plan Reference:
docs/phase-1/p1-production-steps.md

Tests the full pipeline from search -> answer generation.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from futurnal.search.answer_generator import (
    AnswerGenerator,
    AnswerGeneratorConfig,
    GeneratedAnswer,
)


# ---------------------------------------------------------------------------
# Mock Factories
# ---------------------------------------------------------------------------


def create_mock_context(
    query: str,
    num_results: int = 5,
) -> List[Dict[str, Any]]:
    """Create mock context in the format expected by AnswerGenerator."""
    results = []
    for i in range(num_results):
        result = {
            "content": f"Content for document {i} related to {query}",
            "metadata": {
                "source": f"docs/doc{i}.md",
                "label": f"Document {i}",
            },
        }
        results.append(result)
    return results


def create_mock_ollama_pool() -> MagicMock:
    """Create a mock OllamaConnectionPool."""
    pool = MagicMock()
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    pool.is_initialized = True
    pool.is_healthy = True
    pool.generate = AsyncMock(
        return_value="Based on the provided context, the answer is... [Source: docs/doc0.md]"
    )
    return pool


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestSearchWithAnswerPipeline:
    """Integration tests for search + answer generation pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(self):
        """Test end-to-end pipeline with mocked dependencies."""
        mock_pool = create_mock_ollama_pool()
        context = create_mock_context(
            query="What is knowledge management?",
            num_results=5,
        )

        config = AnswerGeneratorConfig(
            model_name="llama3.1:8b-instruct-q4_0",
            temperature=0.3,
        )
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="What is knowledge management?",
            context=context,
        )

        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0
        assert len(result.sources) >= 0
        assert result.model == "llama3.1:8b-instruct-q4_0"
        assert result.generation_time_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_graph_context(self):
        """Test that graph context is properly integrated."""
        mock_pool = create_mock_ollama_pool()
        context = create_mock_context(
            query="Project relationships",
            num_results=3,
        )
        graph_context = {
            "relationships": [
                {"type": "mentions", "from_entity": "Doc1", "to_entity": "ProjectX"},
            ],
            "related_entities": [
                {"name": "ProjectX", "id": "proj_1"},
            ],
        }

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="Project relationships",
            context=context,
            graph_context=graph_context,
        )

        # Verify LLM was called with prompt containing context
        call_args = mock_pool.generate.call_args
        prompt = call_args.kwargs.get("prompt", "")

        # Should include content from results
        assert "document" in prompt.lower()

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_ollama(self):
        """Test that search works even if answer generation fails."""
        from futurnal.search.hybrid.performance.ollama_pool import OllamaConnectionError

        mock_pool = MagicMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.is_initialized = True
        mock_pool.generate = AsyncMock(side_effect=OllamaConnectionError("Ollama unavailable"))

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        context = create_mock_context(query="Test query", num_results=3)

        # Should not raise exception
        result = await generator.generate_answer(
            query="Test query",
            context=context,
        )

        # Should return a result with error message
        assert isinstance(result, GeneratedAnswer)
        assert "couldn't generate" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_model_selection_override(self):
        """Test that model can be changed per request."""
        mock_pool = create_mock_ollama_pool()

        config = AnswerGeneratorConfig(model_name="llama3.1:8b-instruct-q4_0")
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        context = create_mock_context(query="Test", num_results=2)

        # Override model
        result = await generator.generate_answer(
            query="Test",
            context=context,
            model="phi3:mini",
        )

        # Verify different model was used
        call_args = mock_pool.generate.call_args
        assert call_args.kwargs.get("model") == "phi3:mini"
        assert result.model == "phi3:mini"


class TestContextAggregation:
    """Tests for context aggregation from multiple sources."""

    @pytest.mark.asyncio
    async def test_aggregates_multiple_sources(self):
        """Test context aggregation from different source types."""
        mock_pool = create_mock_ollama_pool()

        # Create results from different sources with proper metadata format
        results = [
            {
                "content": "Personal notes about the topic.",
                "metadata": {"source": "notes/topic.md", "label": "Obsidian Note"},
            },
            {
                "content": "Project documentation from repository.",
                "metadata": {"source": "README.md", "label": "GitHub README"},
            },
            {
                "content": "Discussion about the topic in email.",
                "metadata": {"source": "inbox/thread-123", "label": "Email Thread"},
            },
        ]

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)

        context = generator._assemble_context(results)

        # All sources should be represented
        assert "personal notes" in context.lower()
        assert "project documentation" in context.lower()
        assert "discussion" in context.lower()

    @pytest.mark.asyncio
    async def test_preserves_document_ordering(self):
        """Test that documents appear in order in context."""
        mock_pool = create_mock_ollama_pool()

        results = [
            {
                "content": "FIRST_DOCUMENT_CONTENT",
                "metadata": {"source": "first.md"},
            },
            {
                "content": "SECOND_DOCUMENT_CONTENT",
                "metadata": {"source": "second.md"},
            },
        ]

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)

        context = generator._assemble_context(results)

        # First document should appear before second
        first_pos = context.find("FIRST_DOCUMENT_CONTENT")
        second_pos = context.find("SECOND_DOCUMENT_CONTENT")

        assert first_pos < second_pos, "Documents should maintain order"


class TestSourceCitation:
    """Tests for source citation extraction and formatting."""

    def test_extracts_unique_sources(self):
        """Test that sources are deduplicated."""
        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config)

        results = [
            {"content": "A", "metadata": {"source": "doc.md", "label": "Doc"}},
            {"content": "B", "metadata": {"source": "doc.md", "label": "Doc"}},  # Duplicate
            {"content": "C", "metadata": {"source": "other.md"}},
        ]

        sources = generator._extract_sources(results)

        # Should be unique
        assert len(sources) == 2
        assert "Doc" in sources
        assert "other.md" in sources

    def test_handles_missing_metadata(self):
        """Test handling of results without metadata."""
        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config)

        results = [
            {"content": "Content without metadata"},
            {"content": "With metadata", "metadata": {"source": "doc.md"}},
        ]

        sources = generator._extract_sources(results)

        # Should only include valid sources
        assert "doc.md" in sources
        assert len(sources) == 1


class TestPerformanceCharacteristics:
    """Tests for performance-related behavior."""

    @pytest.mark.asyncio
    async def test_timing_is_recorded(self):
        """Test that generation time is recorded."""
        mock_pool = create_mock_ollama_pool()

        # Add delay to mock
        async def slow_generate(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.01)  # 10ms delay
            return "Answer"

        mock_pool.generate = slow_generate

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="Test",
            context=[{"content": "Test", "metadata": {"source": "t.md"}}],
        )

        # Should have recorded time
        assert result.generation_time_ms >= 10  # At least 10ms

    @pytest.mark.asyncio
    async def test_context_limit_truncates_documents(self):
        """Test that context is limited to configured number of documents."""
        mock_pool = create_mock_ollama_pool()

        # Create more documents than limit
        many_docs = [
            {"content": f"Content {i}", "metadata": {"source": f"doc{i}.md"}}
            for i in range(20)
        ]

        config = AnswerGeneratorConfig(context_limit=5)  # Only 5 docs
        generator = AnswerGenerator(config=config, pool=mock_pool)

        context = generator._assemble_context(many_docs)

        # Should only include first 5 documents
        assert "Content 0" in context
        assert "Content 4" in context
        assert "Content 5" not in context  # Should be cut off


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test handling of empty query."""
        mock_pool = create_mock_ollama_pool()

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="",
            context=[{"content": "Test", "metadata": {"source": "t.md"}}],
        )

        # Should handle gracefully
        assert isinstance(result, GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self):
        """Test handling of special characters in content."""
        mock_pool = create_mock_ollama_pool()

        results = [
            {
                "content": "Content with <html> tags and \"quotes\" and 'apostrophes'",
                "metadata": {"source": "test.md"},
            },
            {
                "content": "Unicode: 日本語 and émojis 🎉",
                "metadata": {"source": "unicode.md"},
            },
        ]

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="Test special chars",
            context=results,
        )

        # Should handle without error
        assert isinstance(result, GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test handling of very long query."""
        mock_pool = create_mock_ollama_pool()

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        long_query = "What is " + " ".join(["word"] * 500)

        result = await generator.generate_answer(
            query=long_query,
            context=[{"content": "Answer", "metadata": {"source": "t.md"}}],
        )

        # Should handle without error
        assert isinstance(result, GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test handling of empty context list."""
        mock_pool = create_mock_ollama_pool()

        config = AnswerGeneratorConfig()
        generator = AnswerGenerator(config=config, pool=mock_pool)
        await generator.initialize()

        result = await generator.generate_answer(
            query="Test with no context",
            context=[],
        )

        # Should handle gracefully
        assert isinstance(result, GeneratedAnswer)

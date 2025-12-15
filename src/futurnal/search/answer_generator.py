"""Answer Generation Module - LLM-Powered Response Synthesis.

Research Foundation:
- CausalRAG (ACL 2025): Causal-aware generation from graph context
- LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach with rule grounding

Production Plan Reference:
docs/phase-1/implementation-steps/02-llm-answer-generation.md

Option B Compliance:
- Ghost model FROZEN - Ollama inference only, no fine-tuning
- Context-grounded generation prevents hallucination
- Local-first processing on localhost:11434
- Experiential learning via prompt refinement (not parameter updates)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from futurnal.search.hybrid.performance.ollama_pool import (
    OllamaConnectionPool,
    OllamaConnectionConfig,
    OllamaConnectionError,
)

logger = logging.getLogger(__name__)


# Available models from LLM_MODEL_REGISTRY.md
ANSWER_MODELS = {
    "phi3:mini": {
        "label": "Phi-3 Mini",
        "vram": "4GB",
        "hint": "Fast",
    },
    "llama3.1:8b-instruct-q4_0": {
        "label": "Llama 3.1 8B",
        "vram": "8GB",
        "hint": "Balanced",
    },
    "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0": {
        "label": "Bielik 4.5B",
        "vram": "5GB",
        "hint": "Polish",
    },
    "qwen2.5:7b-instruct": {
        "label": "Qwen 2.5 7B",
        "vram": "8GB",
        "hint": "Quality",
    },
    "mistral:7b-instruct": {
        "label": "Mistral 7B",
        "vram": "8GB",
        "hint": "Reasoning",
    },
}

DEFAULT_MODEL = "llama3.1:8b-instruct-q4_0"


@dataclass
class AnswerGeneratorConfig:
    """Configuration for answer generation.

    Attributes:
        model_name: Ollama model name from ANSWER_MODELS registry
        temperature: Sampling temperature (lower = more factual)
        max_tokens: Maximum tokens to generate
        context_limit: Maximum documents to include in context
        include_graph_context: Whether to include graph relationships
    """

    model_name: str = DEFAULT_MODEL
    temperature: float = 0.3  # Lower for factual grounding
    max_tokens: int = 512
    context_limit: int = 10  # Max documents in context
    include_graph_context: bool = True


@dataclass
class GeneratedAnswer:
    """Result of answer generation.

    Attributes:
        answer: The generated answer text
        sources: List of source documents used
        model: Model name that generated the answer
        tokens_generated: Approximate token count
        generation_time_ms: Time taken to generate
    """

    answer: str
    sources: List[str] = field(default_factory=list)
    model: str = ""
    tokens_generated: int = 0
    generation_time_ms: float = 0.0


class AnswerGenerator:
    """Generate synthesized answers from retrieved context.

    Per CausalRAG paper: Uses graph context for causal-aware generation.
    Per LLM-Enhanced Symbolic: Combines LLM with structured context.

    Example:
        >>> generator = AnswerGenerator()
        >>> await generator.initialize()
        >>> context = [{"content": "Python is a language", "metadata": {"source": "wiki.md"}}]
        >>> answer = await generator.generate_answer("What is Python?", context)
        >>> print(answer.answer)
        >>> await generator.close()
    """

    SYSTEM_PROMPT = '''You are Futurnal's knowledge assistant. Your role is to synthesize
answers from the user's personal knowledge graph.

CRITICAL RULES:
1. ONLY use information from the provided context - NO external knowledge
2. Always cite sources using [Source: filename] notation
3. If the context doesn't contain the answer, say "I couldn't find this in your knowledge"
4. Be concise but comprehensive
5. Preserve the user's terminology and concepts from their notes

FORMAT:
- Start with a direct answer to the question
- Support with evidence from context
- End with source citations'''

    def __init__(
        self,
        config: Optional[AnswerGeneratorConfig] = None,
        pool: Optional[OllamaConnectionPool] = None,
    ) -> None:
        """Initialize answer generator.

        Args:
            config: Optional generation configuration
            pool: Optional pre-existing connection pool
        """
        self.config = config or AnswerGeneratorConfig()
        self._pool = pool
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool.

        Creates the Ollama connection pool if not provided.
        """
        if self._initialized:
            return

        if self._pool is None:
            self._pool = OllamaConnectionPool(
                OllamaConnectionConfig(
                    request_timeout=60.0,  # Longer for generation
                )
            )

        await self._pool.initialize()
        self._initialized = True
        logger.info(
            "AnswerGenerator initialized with model %s",
            self.config.model_name,
        )

    async def close(self) -> None:
        """Clean up resources."""
        if self._pool:
            await self._pool.close()
        self._initialized = False
        logger.info("AnswerGenerator closed")

    async def generate_answer(
        self,
        query: str,
        context: List[Dict[str, Any]],
        graph_context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> GeneratedAnswer:
        """Generate a complete answer (non-streaming).

        Args:
            query: User's question
            context: Retrieved document snippets from search
            graph_context: Related entities and relationships from PKG
            model: Optional model override (uses config.model_name if None)

        Returns:
            GeneratedAnswer with synthesized answer and metadata
        """
        if not self._initialized:
            await self.initialize()

        start = time.time()
        model_name = model or self.config.model_name

        context_text = self._assemble_context(context, graph_context)
        prompt = self._build_prompt(query, context_text)

        try:
            response = await self._pool.generate(  # type: ignore
                model=model_name,
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
            )
        except OllamaConnectionError as e:
            logger.error(f"Answer generation failed: {e}")
            return GeneratedAnswer(
                answer="I couldn't generate an answer. Please try again.",
                sources=[],
                model=model_name,
                generation_time_ms=(time.time() - start) * 1000,
            )

        sources = self._extract_sources(context)
        generation_time = (time.time() - start) * 1000

        logger.info(
            "Generated answer in %.0fms using %s",
            generation_time,
            model_name,
        )

        return GeneratedAnswer(
            answer=response,
            sources=sources,
            model=model_name,
            tokens_generated=len(response.split()),  # Approximate
            generation_time_ms=generation_time,
        )

    async def stream_answer(
        self,
        query: str,
        context: List[Dict[str, Any]],
        graph_context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream answer generation for real-time UI updates.

        Yields tokens as they are generated for progressive display.

        Args:
            query: User's question
            context: Retrieved document snippets
            graph_context: Related entities and relationships
            model: Optional model override

        Yields:
            Individual tokens/chunks from LLM response
        """
        if not self._initialized:
            await self.initialize()

        model_name = model or self.config.model_name
        context_text = self._assemble_context(context, graph_context)
        prompt = self._build_prompt(query, context_text)

        try:
            async for chunk in self._pool.stream_generate(  # type: ignore
                model=model_name,
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
            ):
                yield chunk
        except OllamaConnectionError as e:
            logger.error(f"Streaming answer generation failed: {e}")
            yield "I couldn't generate an answer. Please try again."

    def _build_prompt(self, query: str, context_text: str) -> str:
        """Build the generation prompt.

        Args:
            query: User's question
            context_text: Assembled context from documents and graph

        Returns:
            Formatted prompt string
        """
        return f'''Based on the following context from my personal knowledge:

{context_text}

Question: {query}

Provide a comprehensive answer with source citations:'''

    def _assemble_context(
        self,
        context: List[Dict[str, Any]],
        graph_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Assemble context from retrieval results.

        Per CausalRAG: Include graph relationships for causal awareness.

        Args:
            context: Document snippets from search
            graph_context: Graph relationships and entities

        Returns:
            Formatted context string for LLM prompt
        """
        parts: List[str] = []

        # Add document snippets (limited to config.context_limit)
        for i, doc in enumerate(context[: self.config.context_limit], 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            # Use label if available for better citation
            label = metadata.get("label", source)
            content = doc.get("content", "")[:500]  # Truncate long content
            parts.append(f"[Document {i} - {label}]\n{content}\n")

        # Add graph context if available (CausalRAG influence)
        if graph_context and self.config.include_graph_context:
            relationships = graph_context.get("relationships", [])
            if relationships:
                parts.append("\n[Related Concepts and Connections from Knowledge Graph]")
                for rel in relationships[:5]:
                    rel_type = rel.get("type", rel.get("rel_type", "related_to"))
                    from_entity = rel.get("from_entity", rel.get("source", "?"))
                    to_entity = rel.get("to_entity", rel.get("target", "?"))
                    parts.append(f"- {from_entity} --[{rel_type}]--> {to_entity}")

            # Include related entities for context
            related_entities = graph_context.get("related_entities", [])
            if related_entities:
                entity_names = [
                    e.get("name", e.get("id", "?"))
                    for e in related_entities[:5]
                ]
                parts.append(f"\n[Related Entities: {', '.join(entity_names)}]")

        return "\n".join(parts)

    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source names from context.

        Args:
            context: Document snippets from search

        Returns:
            Deduplicated list of source names
        """
        sources: List[str] = []
        seen: Set[str] = set()

        for doc in context:
            metadata = doc.get("metadata", {})
            source = metadata.get("source")
            label = metadata.get("label")
            name = label or source

            if name and name not in seen:
                sources.append(name)
                seen.add(name)

        return sources

    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, str]]:
        """Get available models for answer generation.

        Returns:
            Dict of model_id -> {label, vram, hint}
        """
        return ANSWER_MODELS.copy()

    @staticmethod
    def get_default_model() -> str:
        """Get the default model name.

        Returns:
            Default model identifier
        """
        return DEFAULT_MODEL

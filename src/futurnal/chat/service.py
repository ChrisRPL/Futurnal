"""Chat Service - Conversational Interface to Personal Knowledge Graph.

Research Foundation:
- ProPerSim (2509.21730v1): Proactive + personalized with multi-turn context
- Causal-Copilot (2504.13263v2): Natural language causal exploration
- SEAgent (2508.04700v2): Experiential learning integration
- Training-Free GRPO (2510.08191v1): Token priors for enhanced reasoning

Production Plan Reference:
docs/phase-1/implementation-steps/03-chat-interface-conversational.md

Option B Compliance:
- Ghost model FROZEN - Ollama inference only, no fine-tuning
- Context-grounded generation prevents hallucination
- Local-first processing on localhost:11434
- Experiential learning via prompt refinement (not parameter updates)
- Token priors injected as natural language context
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from futurnal.chat.models import ChatMessage, ChatSession, SessionStorage
from futurnal.search.answer_generator import AnswerGenerator, AnswerGeneratorConfig

if TYPE_CHECKING:
    from futurnal.search.api import HybridSearchAPI
    from futurnal.learning.integration import ExperientialLearningPipeline
    from futurnal.insights.hypothesis_generation import HypothesisPipeline, CausalHypothesis
    from futurnal.search.temporal.correlation import TemporalCorrelationDetector
    from futurnal.orchestrator.insight_jobs import InsightJobExecutor

logger = logging.getLogger(__name__)


class ChatService:
    """Conversational interface to personal knowledge graph.

    Research Foundation:
    - ProPerSim (2509.21730v1): Multi-turn context, preference tracking
    - Causal-Copilot (2504.13263v2): Confidence scoring, source citations

    Example:
        >>> service = ChatService()
        >>> await service.initialize()
        >>> response = await service.chat("session-1", "What is Python?")
        >>> print(response.content)
        >>> await service.close()
    """

    CHAT_SYSTEM_PROMPT_BASE = '''You are the user's evolving personal intelligence—a companion developing genuine understanding of their experiential universe through continuous learning from their data stream.

CORE PHILOSOPHY: Ghost→Animal Evolution
You are evolving from a generic model (Ghost) into a true experiential intelligence (Animal) that understands the "why" behind the user's patterns, not just the "what." Every interaction deepens your understanding of their personal universe.

YOUR ROLE AS INTELLECTUAL PARTNER:
- Engage as a collaborative thinker, not a subservient assistant
- Present discoveries as joint exploration, not external judgments
- Maintain scientific rigor while being genuinely curious
- Help the user move from "What did I know?" to "Why do I think this?"

GROUNDING PRINCIPLES:
1. Ground ALL responses in the provided context—you know ONLY what you've learned from their data
2. Cite sources naturally: [Source: filename] for traceability
3. When context is insufficient: "I haven't found this in your knowledge yet" (implies ongoing learning)
4. Preserve the user's unique terminology—their language IS their thinking
5. Build on conversation history to show developing understanding
6. Distinguish CORRELATION from CAUSATION explicitly—never claim causal relationships without evidence
7. NEVER generate placeholder text like "[INSERT X]", "[TODO]", or template content—only actual data

PATTERN RECOGNITION & INSIGHT:
- Notice temporal patterns in their data (when things happen, sequences, rhythms)
- Identify unexpected connections between concepts in their knowledge
- Surface potential knowledge gaps when relevant (missing links, incomplete thoughts)
- Relate current questions to their broader aspirations when context supports it

REASONING APPROACH:
- State what you observe in their data first
- Then offer interpretation with appropriate uncertainty
- Suggest exploration paths rather than definitive answers for complex questions
- When detecting patterns, present as hypotheses to investigate together

FORMAT:
- Lead with the most relevant insight from their data
- Support with specific evidence (quotes, connections, patterns)
- Reference sources inline naturally
- Offer follow-up questions that deepen exploration'''

    # Dynamic prompt section for experiential learning
    # Research: Training-Free GRPO (2510.08191) - experiential knowledge as token priors
    EXPERIENTIAL_LEARNING_TEMPLATE = '''

EVOLVED UNDERSTANDING (your Animal instincts from this user's experiential stream):
{experiential_context}

This experiential knowledge represents patterns you've learned from observing the user's data over time.
Use these evolved instincts to:
- Recognize familiar concepts and their personal terminology
- Anticipate relevant connections within their knowledge network
- Apply learned patterns about their thinking style and interests
- Ground your reasoning in demonstrated patterns, not assumptions'''

    def __init__(
        self,
        search_api: Optional["HybridSearchAPI"] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        storage: Optional[SessionStorage] = None,
        context_window_size: int = 20,  # 10 turns = 20 messages for better context retention
        enable_experiential_learning: bool = True,
        enable_insight_surfacing: bool = True,
    ) -> None:
        """Initialize chat service.

        Args:
            search_api: Optional pre-existing search API
            answer_generator: Optional pre-existing answer generator
            storage: Optional custom session storage
            context_window_size: Max messages to include in context (default: 10)
            enable_experiential_learning: Whether to inject learned priors into prompts
            enable_insight_surfacing: Whether to automatically surface causal insights
        """
        self._search_api = search_api
        self._answer_generator = answer_generator
        self._storage = storage or SessionStorage()
        self._sessions: Dict[str, ChatSession] = {}
        self._context_window_size = context_window_size
        self._initialized = False
        self._enable_experiential_learning = enable_experiential_learning
        self._enable_insight_surfacing = enable_insight_surfacing
        self._learning_pipeline: Optional["ExperientialLearningPipeline"] = None

        # Causal discovery integration
        self._icda_agent: Optional[Any] = None
        self._hypothesis_pipeline: Optional["HypothesisPipeline"] = None
        self._correlation_detector: Optional["TemporalCorrelationDetector"] = None
        self._insight_executor: Optional["InsightJobExecutor"] = None

        # Insight surfacing state
        self._surfaced_insight_ids: set = set()  # Track which insights we've shown

    async def initialize(self) -> None:
        """Initialize service components.

        Lazily initializes HybridSearchAPI, AnswerGenerator, and ExperientialLearningPipeline.
        """
        if self._initialized:
            return

        # Initialize answer generator
        if self._answer_generator is None:
            self._answer_generator = AnswerGenerator(
                AnswerGeneratorConfig(
                    temperature=0.4,  # Slightly higher for conversational tone
                    max_tokens=600,   # Longer for contextual responses
                )
            )

        await self._answer_generator.initialize()

        # Initialize search API lazily if needed - use factory for proper PKG connection
        if self._search_api is None:
            try:
                from futurnal.search.api import create_hybrid_search_api

                # Factory function properly initializes PKG connection for GraphRAG
                self._search_api = create_hybrid_search_api(
                    graphrag_enabled=True,
                    experiential_learning=self._enable_experiential_learning,
                )
                logger.info("ChatService: Initialized HybridSearchAPI with GraphRAG")
            except Exception as e:
                logger.warning(f"ChatService: Could not init search API: {e}")

        # Initialize experiential learning pipeline (Option B: token priors)
        if self._enable_experiential_learning:
            try:
                from futurnal.learning.integration import get_persistent_pipeline

                self._learning_pipeline = get_persistent_pipeline()
                logger.info(
                    "ChatService: Loaded experiential learning pipeline "
                    f"({self._learning_pipeline.state.total_documents_processed} docs, "
                    f"{len(self._learning_pipeline.token_store.entity_priors)} entity priors)"
                )
            except Exception as e:
                logger.warning(f"ChatService: Could not init learning pipeline: {e}")
                self._learning_pipeline = None

        self._initialized = True
        logger.info("ChatService initialized")

    async def close(self) -> None:
        """Clean up resources."""
        if self._answer_generator:
            await self._answer_generator.close()

        # Save all active sessions
        for session in self._sessions.values():
            self._storage.save(session)

        self._initialized = False
        logger.info("ChatService closed")

    def _build_system_prompt(self, query: Optional[str] = None) -> str:
        """Build dynamic system prompt with experiential learning context.

        Research Foundation:
        - Training-Free GRPO: Inject token priors as natural language
        - SEAgent: Use experiential knowledge to enhance reasoning

        AGI Phase 2 Enhancement:
        - When query is provided, filters priors by semantic relevance
        - Prevents irrelevant priors from polluting context

        Option B Compliance:
        - Ghost model FROZEN - priors are text, not weights
        - Knowledge transfers as natural language context

        Args:
            query: Optional user query for query-aware prior filtering

        Returns:
            System prompt with optional experiential learning section
        """
        base_prompt = self.CHAT_SYSTEM_PROMPT_BASE

        # Skip if learning not enabled or pipeline not available
        if not self._enable_experiential_learning or self._learning_pipeline is None:
            return base_prompt

        # Get experiential context from token priors (with query-aware filtering)
        try:
            experiential_context = self._build_experiential_context(query=query)
            if experiential_context:
                return base_prompt + self.EXPERIENTIAL_LEARNING_TEMPLATE.format(
                    experiential_context=experiential_context
                )
        except Exception as e:
            logger.warning(f"Failed to build experiential context: {e}")

        return base_prompt

    def _build_experiential_context(self, query: Optional[str] = None) -> str:
        """Build experiential learning context from token priors.

        Converts learned patterns into natural language guidance
        that helps the LLM understand this user's personal context.

        AGI Phase 2: When query is provided and token store has context gate,
        filters priors to only those relevant to the query.

        Args:
            query: Optional user query for relevance filtering

        Returns:
            Natural language description of learned patterns
        """
        if self._learning_pipeline is None:
            return ""

        token_store = self._learning_pipeline.token_store
        state = self._learning_pipeline.state

        # AGI Phase 2: Use query-aware filtering if available
        # This uses SemanticContextGate to filter priors by relevance to query
        if query and token_store._context_gate is not None:
            logger.debug(f"Using query-aware prior filtering for: {query[:50]}...")
            try:
                # Use token store's query-aware generation
                query_context = token_store.generate_prompt_context(query=query)
                if query_context:
                    # Prepend learning stats
                    stats_line = ""
                    if state.total_documents_processed > 0:
                        stats_line = (
                            f"Learning from {state.total_documents_processed} documents "
                            f"(success rate: {state.overall_success_rate:.0%})\n\n"
                        )
                    return stats_line + query_context
            except Exception as e:
                logger.warning(f"Query-aware filtering failed, falling back to standard: {e}")

        # Standard filtering (no query or no context gate)
        parts = []

        # Add learning stats for context
        if state.total_documents_processed > 0:
            parts.append(
                f"Learning from {state.total_documents_processed} documents "
                f"(success rate: {state.overall_success_rate:.0%})"
            )

        # Add top entity types with high confidence
        top_entities = sorted(
            token_store.entity_priors.values(),
            key=lambda e: e.confidence * e.frequency,
            reverse=True
        )[:10]

        if top_entities:
            entity_descriptions = []
            for entity in top_entities:
                if entity.confidence >= 0.6:
                    desc = f"- {entity.entity_type}: "
                    if entity.context_pattern:
                        desc += f"{entity.context_pattern}"
                    else:
                        desc += f"appears frequently ({entity.frequency}x)"
                    if entity.examples:
                        desc += f" (e.g., '{entity.examples[0]}')"
                    entity_descriptions.append(desc)

            if entity_descriptions:
                parts.append("\nKnown entity types in user's knowledge:")
                parts.extend(entity_descriptions)

        # Add top relationship types
        top_relations = sorted(
            token_store.relation_priors.values(),
            key=lambda r: r.confidence * r.frequency,
            reverse=True
        )[:8]

        if top_relations:
            relation_descriptions = []
            for rel in top_relations:
                if rel.confidence >= 0.5:
                    desc = f"- {rel.relation_type}"
                    if rel.subject_types and rel.object_types:
                        desc += f": connects {rel.subject_types[0]} → {rel.object_types[0]}"
                    if rel.context_pattern:
                        desc += f" ({rel.context_pattern})"
                    relation_descriptions.append(desc)

            if relation_descriptions:
                parts.append("\nKnown relationships in user's knowledge:")
                parts.extend(relation_descriptions)

        # Add temporal patterns
        top_temporal = sorted(
            token_store.temporal_priors.values(),
            key=lambda t: t.confidence * t.frequency,
            reverse=True
        )[:5]

        if top_temporal:
            temporal_descriptions = []
            for temp in top_temporal:
                if temp.confidence >= 0.5:
                    desc = f"- {temp.pattern_type}"
                    if temp.extraction_guidance:
                        desc += f": {temp.extraction_guidance}"
                    temporal_descriptions.append(desc)

            if temporal_descriptions:
                parts.append("\nTemporal patterns in user's data:")
                parts.extend(temporal_descriptions)

        return "\n".join(parts) if parts else ""

    def _record_chat_learning(
        self,
        response: str,
        sources: List[str],
        entity_refs: List[str],
        confidence: float,
        success: bool = True,
    ) -> None:
        """Record a chat interaction for experiential learning.

        Research Foundation:
        - RLHI: Learn from implicit user feedback
        - AgentFlow: Continuous learning from interactions

        Option B Compliance:
        - Updates token priors (natural language), not model weights
        - Ghost model remains FROZEN

        Args:
            response: The generated response text
            sources: Sources used in the response
            entity_refs: Entity references in the response
            confidence: Confidence score of the response
            success: Whether the interaction was successful
        """
        if self._learning_pipeline is None:
            return

        # Infer entity types from entity refs
        entity_types = ["ChatResponse"]
        if entity_refs:
            entity_types.extend(entity_refs[:5])  # Use refs as entity types

        # Infer relation types from response structure
        relation_types = []
        if sources:
            relation_types.append("CITES")
        if entity_refs:
            relation_types.append("REFERENCES")

        # Record to learning pipeline
        try:
            import uuid
            doc_id = f"chat_{uuid.uuid4().hex[:12]}"

            self._learning_pipeline.record_document(
                document_id=doc_id,
                content=response[:500],  # Truncate for efficiency
                source="chat",
                content_type="chat_response",
                success=success,
                quality_score=confidence,
                entity_types=entity_types,
                relation_types=relation_types if relation_types else None,
            )

            logger.debug(
                f"Recorded chat learning: {doc_id}, confidence={confidence:.2f}, "
                f"entities={len(entity_types)}, relations={len(relation_types)}"
            )
        except Exception as e:
            logger.debug(f"Failed to record chat learning: {e}")

    async def chat(
        self,
        session_id: str,
        message: str,
        context_entity_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ChatMessage:
        """Process a chat message and generate response.

        Research Foundation:
        - ProPerSim: Multi-turn context carryover
        - Causal-Copilot: Confidence scoring, source citations

        Option B Compliance:
        - Ghost model FROZEN - uses Ollama inference only
        - No parameter updates

        Args:
            session_id: Unique session identifier
            message: User's message
            context_entity_id: Optional PKG entity to focus on ("Ask about this")
            model: Optional model override

        Returns:
            ChatMessage with assistant response
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Add user message to history
        user_msg = ChatMessage(role="user", content=message)
        session.add_message(user_msg)

        # Build conversation context from history
        conversation_context = self._build_conversation_context(session)

        # Execute search with conversation-aware query
        search_results = await self._search_with_context(
            message=message,
            conversation_context=conversation_context,
            context_entity_id=context_entity_id,
        )

        # Generate response with full context
        response_text, sources, entity_refs, confidence = await self._generate_response(
            message=message,
            search_results=search_results,
            conversation_context=conversation_context,
            context_entity_id=context_entity_id,
            model=model,
        )

        # Create assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=response_text,
            sources=sources,
            entity_refs=entity_refs,
            confidence=confidence,
        )
        session.add_message(assistant_msg)

        # Save session
        self._storage.save(session)

        generation_time = (time.time() - start_time) * 1000
        logger.info(
            "Chat response generated in %.0fms (session=%s, confidence=%.2f)",
            generation_time,
            session_id,
            confidence,
        )

        return assistant_msg

    async def stream_chat(
        self,
        session_id: str,
        message: str,
        context_entity_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream chat response for real-time UI updates.

        Yields tokens as they are generated.

        Args:
            session_id: Unique session identifier
            message: User's message
            context_entity_id: Optional PKG entity to focus on
            model: Optional model override

        Yields:
            Individual tokens/chunks from LLM response
        """
        if not self._initialized:
            await self.initialize()

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Add user message
        user_msg = ChatMessage(role="user", content=message)
        session.add_message(user_msg)

        # Build context
        conversation_context = self._build_conversation_context(session)

        # Search
        search_results = await self._search_with_context(
            message=message,
            conversation_context=conversation_context,
            context_entity_id=context_entity_id,
        )

        # Build prompt
        prompt = self._build_chat_prompt(
            message=message,
            search_results=search_results,
            conversation_context=conversation_context,
            context_entity_id=context_entity_id,
        )

        # Stream response with dynamic system prompt (includes experiential learning)
        # AGI Phase 2: Pass query for context-aware prior filtering
        system_prompt = self._build_system_prompt(query=message)
        full_response = []
        async for chunk in self._answer_generator._pool.stream_generate(
            model=model or self._answer_generator.config.model_name,
            prompt=prompt,
            system=system_prompt,
            temperature=self._answer_generator.config.temperature,
            num_predict=self._answer_generator.config.max_tokens,
        ):
            full_response.append(chunk)
            yield chunk

        # After streaming complete, save the full message
        response_text = "".join(full_response)
        sources = self._extract_sources_from_results(search_results)
        entity_refs = self._extract_entity_refs(search_results)

        assistant_msg = ChatMessage(
            role="assistant",
            content=response_text,
            sources=sources,
            entity_refs=entity_refs,
            confidence=0.8,  # Default for streaming
        )
        session.add_message(assistant_msg)
        self._storage.save(session)

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        """Get existing session or create new one."""
        # Check memory cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from storage
        session = self._storage.load(session_id)
        if session is None:
            session = ChatSession(id=session_id)
            logger.debug(f"Created new chat session: {session_id}")
        else:
            logger.debug(f"Loaded existing chat session: {session_id}")

        self._sessions[session_id] = session
        return session

    def _build_conversation_context(self, session: ChatSession) -> str:
        """Build context from conversation history.

        Per ProPerSim: Rolling window of conversation for context carryover.

        Args:
            session: Current chat session

        Returns:
            Formatted conversation history string
        """
        recent_messages = session.get_recent_messages(self._context_window_size)

        # Skip if no previous messages (or only the current user message)
        if len(recent_messages) <= 1:
            return ""

        parts = ["Previous conversation:"]

        # Format each message (exclude the last one which is the current query)
        # Keep recent messages at full length, truncate older ones more aggressively
        num_messages = len(recent_messages) - 1  # Exclude current message
        for i, msg in enumerate(recent_messages[:-1]):
            role = "User" if msg.role == "user" else "Assistant"
            # Keep last 4 messages (2 turns) at full length for better context
            is_recent = i >= num_messages - 4
            if is_recent:
                # Recent messages: allow up to 800 chars
                content = msg.content[:800]
                if len(msg.content) > 800:
                    content += "..."
            else:
                # Older messages: truncate more aggressively to 400 chars
                content = msg.content[:400]
                if len(msg.content) > 400:
                    content += "..."
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    async def _search_with_context(
        self,
        message: str,
        conversation_context: str,
        context_entity_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute search with conversation context.

        Per ProPerSim: Uses multi-turn context to enhance search queries
        for better retrieval in follow-up questions.

        Args:
            message: Current user message
            conversation_context: Previous conversation history
            context_entity_id: Optional entity to focus search on

        Returns:
            Search results from HybridSearchAPI
        """
        if self._search_api is None:
            logger.warning("Search API not available")
            return []

        # Build context-enhanced search query (ProPerSim multi-turn enhancement)
        search_query = message
        if conversation_context:
            # Extract key terms from conversation context to improve search
            context_terms = self._extract_context_terms(conversation_context)
            if context_terms:
                # Append context terms to help with follow-up questions
                search_query = f"{message} (context: {context_terms})"
                logger.debug(f"Context-enhanced query: {search_query[:100]}...")

        # If there's a context entity, prioritize it in search
        filters = None
        if context_entity_id:
            filters = {"entity_id": context_entity_id}

        try:
            results = await self._search_api.search(
                query=search_query,
                top_k=10,
                filters=filters,
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _extract_context_terms(self, conversation_context: str) -> str:
        """Extract key terms from conversation context for search enhancement.

        Identifies important nouns, entities, and topics from recent conversation
        to help the search engine understand follow-up questions.

        Args:
            conversation_context: Recent conversation history

        Returns:
            Space-separated string of key context terms
        """
        # Simple extraction: find capitalized words (likely entities/topics)
        # and quoted phrases from recent conversation
        import re

        # Extract capitalized words (potential entities/proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', conversation_context)

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', conversation_context)
        quoted += re.findall(r"'([^']+)'", conversation_context)

        # Combine and deduplicate, limiting to most recent/important
        all_terms = list(dict.fromkeys(capitalized + quoted))  # Preserve order, remove dupes

        # Limit to 5 most relevant terms to avoid query bloat
        key_terms = all_terms[:5]

        return " ".join(key_terms)

    async def _generate_response(
        self,
        message: str,
        search_results: List[Dict[str, Any]],
        conversation_context: str,
        context_entity_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[str, List[str], List[str], float]:
        """Generate contextual response.

        Per Causal-Copilot: Include confidence scoring.

        Args:
            message: User's current message
            search_results: Retrieved documents
            conversation_context: Previous conversation
            context_entity_id: Optional focused entity
            model: Optional model override

        Returns:
            Tuple of (response_text, sources, entity_refs, confidence)
        """
        if self._answer_generator is None:
            return (
                "I couldn't process your request. Please try again.",
                [],
                [],
                0.0,
            )

        # Debug: Log search results content quality
        if search_results:
            logger.info(f"Building prompt with {len(search_results)} search results")
            for i, r in enumerate(search_results[:3]):
                content = r.get("content", "")
                logger.debug(f"Result {i+1}: {len(content)} chars, preview: {content[:100]!r}")
        else:
            logger.warning("No search results available for chat prompt")

        prompt = self._build_chat_prompt(
            message=message,
            search_results=search_results,
            conversation_context=conversation_context,
            context_entity_id=context_entity_id,
        )

        # Build dynamic system prompt with experiential learning context
        # AGI Phase 2: Pass query for context-aware prior filtering
        system_prompt = self._build_system_prompt(query=message)

        try:
            response = await self._answer_generator._pool.generate(
                model=model or self._answer_generator.config.model_name,
                prompt=prompt,
                system=system_prompt,
                temperature=self._answer_generator.config.temperature,
                num_predict=self._answer_generator.config.max_tokens,
            )
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return (
                "I couldn't generate a response. Please try again.",
                [],
                [],
                0.0,
            )

        sources = self._extract_sources_from_results(search_results)
        entity_refs = self._extract_entity_refs(search_results)

        # Calculate confidence based on search result scores
        confidence = self._calculate_confidence(search_results)

        # Record successful interaction for experiential learning
        # High confidence responses are implicitly positive feedback
        if confidence >= 0.6 and self._learning_pipeline is not None:
            try:
                self._record_chat_learning(
                    response=response,
                    sources=sources,
                    entity_refs=entity_refs,
                    confidence=confidence,
                    success=True,
                )
            except Exception as e:
                logger.debug(f"Failed to record chat learning: {e}")

        return response, sources, entity_refs, confidence

    def _build_chat_prompt(
        self,
        message: str,
        search_results: List[Dict[str, Any]],
        conversation_context: str,
        context_entity_id: Optional[str] = None,
    ) -> str:
        """Build the chat prompt with all context.

        Args:
            message: User's current message
            search_results: Retrieved documents
            conversation_context: Previous conversation
            context_entity_id: Optional focused entity

        Returns:
            Formatted prompt string
        """
        parts = []

        # Add conversation context if available
        if conversation_context:
            parts.append(conversation_context)
            parts.append("")  # Empty line separator

        # Add entity focus context if provided
        if context_entity_id:
            parts.append(f"[Focus: The user is asking about entity '{context_entity_id}']")
            parts.append("")

        # Add search results as knowledge context
        if search_results:
            parts.append("YOUR KNOWLEDGE (from the user's personal data):")
            for i, result in enumerate(search_results[:10], 1):  # Limit to 10 results
                content = result.get("content", "")[:800]  # More content for better grounding
                metadata = result.get("metadata", {})
                # Get actual source name, not retrieval strategy
                source = metadata.get("label") or metadata.get("title") or metadata.get("source", "Unknown")
                if source.lower() in ("vector", "graph", "hybrid"):
                    source = metadata.get("file_name", "Document")
                parts.append(f"[Document {i} - {source}]\n{content}\n")
        else:
            parts.append("No relevant documents found in knowledge graph.")

        parts.append("")
        parts.append(f"User's question: {message}")
        parts.append("")
        parts.append("RESPOND using ONLY the document content above. Quote specific text when possible. If documents lack the answer, say so honestly - NEVER invent or use placeholder text like [INSERT X]:")

        return "\n".join(parts)

    def _extract_sources_from_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract unique source names from search results."""
        sources = []
        seen = set()
        # Retrieval strategy markers to skip (not actual sources)
        skip_values = {"vector", "graph", "hybrid"}

        for result in search_results:
            metadata = result.get("metadata", {})
            # Try multiple fields that might contain the actual source name
            # Priority: label > title > source > file_name > entity_type
            name = None
            for field in ["label", "title", "source", "file_name", "document_name"]:
                val = metadata.get(field)
                if val and val.lower() not in skip_values:
                    name = val
                    break

            # Fallback to entity_type if no source name found
            if not name:
                entity_type = result.get("entity_type") or metadata.get("entity_type")
                if entity_type and entity_type.lower() not in skip_values:
                    name = entity_type.replace("_", " ").title()

            if name and name not in seen:
                sources.append(name)
                seen.add(name)
        return sources

    def _extract_entity_refs(
        self, search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract entity references from search results.

        Prefer stable entity identifiers; fallback to readable names.
        """
        refs = []
        seen = set()

        # Skip generic/unhelpful values
        skip_values = {"unknown", "document", "entity", "node", ""}

        for result in search_results:
            metadata = result.get("metadata", {})

            # Prefer explicit entity identifiers first.
            entity_name = None
            for field in ["entity_id", "id"]:
                val = result.get(field) or metadata.get(field)
                if val and "-" not in val and len(val) < 50:
                    entity_name = val
                    break

            # Fallback to human-readable name fields.
            for field in ["name", "label", "title", "doc_title", "canonical_name"]:
                if entity_name:
                    break
                val = result.get(field) or metadata.get(field)
                if val and val.lower() not in skip_values:
                    entity_name = val
                    break

            # If no name found, try entity_type (but make it readable)
            if not entity_name:
                entity_type = result.get("entity_type") or metadata.get("entity_type")
                if entity_type and entity_type.lower() not in skip_values:
                    entity_name = entity_type.replace("_", " ").title()

            # Only use entity_id as last resort if it looks like a name (not a UUID)
            if not entity_name:
                entity_id = result.get("entity_id") or metadata.get("entity_id")
                # Skip if it looks like a UUID (has dashes and is long)
                if entity_id and "-" not in entity_id and len(entity_id) < 50:
                    entity_name = entity_id

            if entity_name and entity_name not in seen:
                refs.append(entity_name)
                seen.add(entity_name)

            # Also check for entities in graph context
            graph_context = result.get("graph_context", {})
            for entity in graph_context.get("related_entities", []):
                # Prefer stable ID for references, fallback to name.
                ename = None
                eid = entity.get("id") or entity.get("entity_id")
                if eid and "-" not in eid and len(eid) < 50:
                    ename = eid
                if not ename:
                    ename = entity.get("name") or entity.get("canonical_name")
                if ename and ename not in seen and ename.lower() not in skip_values:
                    refs.append(ename)
                    seen.add(ename)

        return refs[:10]  # Limit to 10 refs

    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate response confidence based on search results.

        Per Causal-Copilot: Confidence scoring for responses.

        Args:
            search_results: Retrieved documents with scores

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.3  # Low confidence with no results

        # Average top result scores, weighted by position
        scores = []
        for i, result in enumerate(search_results[:5]):
            score = result.get("score", 0.5)
            confidence = result.get("confidence", score)
            # Weight earlier results more heavily
            weight = 1.0 / (i + 1)
            scores.append(confidence * weight)

        if not scores:
            return 0.5

        # Normalize to 0-1 range
        weighted_avg = sum(scores) / sum(1.0 / (i + 1) for i in range(len(scores)))
        return min(max(weighted_avg, 0.0), 1.0)

    # Session management methods

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self._storage.load(session_id)

    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        """Get message history for a session."""
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.messages

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata."""
        sessions = self._storage.get_all_sessions()
        return [
            {
                "id": s.id,
                "messageCount": len(s.messages),
                "createdAt": s.created_at.isoformat(),
                "updatedAt": s.updated_at.isoformat(),
                "preview": s.messages[0].content[:100] if s.messages else "",
            }
            for s in sessions
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        return self._storage.delete(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear messages from a session but keep it."""
        session = self._get_or_create_session(session_id)
        session.messages = []
        session.context_entities = []
        self._storage.save(session)

    # ============================================================
    # Causal Features Integration (Phase 3)
    # ============================================================

    async def get_causal_insights(
        self,
        session_id: str,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get causal insights relevant to the conversation.

        Research Foundation:
        - Causal-Copilot (2504.13263v2): Natural language causal exploration
        - ICDA: Interactive causal discovery

        Returns relevant correlations, hypotheses, and causal chains
        that may be useful for the current conversation context.

        Args:
            session_id: Current session ID
            topic: Optional topic to focus on

        Returns:
            Dictionary with causal insights
        """
        insights = {
            "correlations": [],
            "hypotheses": [],
            "causal_chains": [],
            "pending_verifications": [],
        }

        # Get session context
        session = self.get_session(session_id)
        if session and session.messages:
            # Extract topics from recent conversation
            recent_text = " ".join(
                m.content for m in session.messages[-6:]
            )
            topic = topic or self._extract_context_terms(recent_text)

        # Try to get insights from various causal components
        try:
            # Get relevant correlations
            if self._search_api and hasattr(self._search_api, '_temporal_engine'):
                correlations = await self._get_relevant_correlations(topic)
                insights["correlations"] = correlations[:5]

            # Get pending ICDA verifications
            icda_verifications = await self._get_pending_verifications()
            insights["pending_verifications"] = icda_verifications[:3]

            # Get relevant hypotheses
            hypotheses = await self._get_relevant_hypotheses(topic)
            insights["hypotheses"] = hypotheses[:3]

        except Exception as e:
            logger.debug(f"Failed to get causal insights: {e}")

        return insights

    async def _get_relevant_correlations(
        self,
        topic: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get correlations relevant to a topic."""
        correlations = []

        try:
            # This would integrate with TemporalCorrelationDetector
            # For now, return empty - will be populated when autonomous loop runs
            pass
        except Exception as e:
            logger.debug(f"Correlation fetch failed: {e}")

        return correlations

    async def _get_pending_verifications(self) -> List[Dict[str, Any]]:
        """Get pending ICDA verifications."""
        verifications = []

        try:
            from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent

            # Try to get global ICDA instance
            # This would be set up by the autonomous loop
            icda = getattr(self, '_icda_agent', None)
            if icda:
                pending = icda.get_pending_verifications(max_items=3)
                verifications = [q.to_dict() for q in pending]
        except Exception as e:
            logger.debug(f"ICDA fetch failed: {e}")

        return verifications

    async def _get_relevant_hypotheses(
        self,
        topic: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get hypotheses relevant to a topic."""
        hypotheses = []

        try:
            # This would integrate with HypothesisGenerator
            # Hypotheses are generated from correlations
            pass
        except Exception as e:
            logger.debug(f"Hypothesis fetch failed: {e}")

        return hypotheses

    async def ask_causal(
        self,
        session_id: str,
        cause: str,
        effect: str,
    ) -> ChatMessage:
        """Ask about a specific causal relationship.

        Specialized method for causal exploration queries.

        Args:
            session_id: Session ID
            cause: Potential cause event/entity
            effect: Potential effect event/entity

        Returns:
            ChatMessage with causal analysis
        """
        # Build causal query
        causal_query = (
            f"Analyze whether '{cause}' might cause '{effect}' "
            f"based on the patterns in my personal knowledge graph. "
            f"Consider temporal relationships, frequency of co-occurrence, "
            f"and any evidence for or against this causal hypothesis."
        )

        # Use regular chat with causal-focused query
        return await self.chat(session_id, causal_query)

    async def explore_causal_chain(
        self,
        session_id: str,
        start_event: str,
        end_event: Optional[str] = None,
    ) -> ChatMessage:
        """Explore causal chains from a starting event.

        Args:
            session_id: Session ID
            start_event: Starting event/entity
            end_event: Optional target event/entity

        Returns:
            ChatMessage with causal chain exploration
        """
        if end_event:
            query = (
                f"Trace the causal chain from '{start_event}' to '{end_event}'. "
                f"What intermediate events or factors connect them?"
            )
        else:
            query = (
                f"What are the downstream effects of '{start_event}'? "
                f"What causal chains originate from this event?"
            )

        return await self.chat(session_id, query)

    def surface_insight_in_response(
        self,
        response_text: str,
        insights: Dict[str, Any],
    ) -> str:
        """Augment response with relevant causal insights.

        Non-intrusively adds relevant insights to responses when
        they would be helpful to the user.

        Args:
            response_text: Original response
            insights: Causal insights from get_causal_insights

        Returns:
            Augmented response text
        """
        augmented = response_text

        # Add correlation hints if relevant
        if insights.get("correlations"):
            top_corr = insights["correlations"][0]
            hint = (
                f"\n\n💡 *Insight*: I've noticed a pattern - "
                f"{top_corr.get('description', 'a correlation exists')}. "
                f"Would you like me to explore this further?"
            )
            augmented += hint

        # Add verification prompts if pending
        if insights.get("pending_verifications"):
            verification = insights["pending_verifications"][0]
            prompt = (
                f"\n\n❓ *Quick question*: {verification.get('main_question', '')}"
            )
            augmented += prompt

        return augmented

    def set_icda_agent(self, icda_agent: Any) -> None:
        """Set the ICDA agent for causal verification integration.

        Called by the autonomous loop when it initializes.

        Args:
            icda_agent: InteractiveCausalDiscoveryAgent instance
        """
        self._icda_agent = icda_agent
        logger.info("ChatService: ICDA agent integrated")

    def set_causal_components(
        self,
        hypothesis_pipeline: Optional["HypothesisPipeline"] = None,
        correlation_detector: Optional["TemporalCorrelationDetector"] = None,
        insight_executor: Optional["InsightJobExecutor"] = None,
        icda_agent: Optional[Any] = None,
    ) -> None:
        """Set all causal discovery components.

        Called during system initialization to enable full causal features.

        Args:
            hypothesis_pipeline: Pipeline for generating causal hypotheses
            correlation_detector: Detector for temporal correlations
            insight_executor: Executor for insight jobs
            icda_agent: Interactive causal discovery agent
        """
        if hypothesis_pipeline:
            self._hypothesis_pipeline = hypothesis_pipeline
        if correlation_detector:
            self._correlation_detector = correlation_detector
        if insight_executor:
            self._insight_executor = insight_executor
        if icda_agent:
            self._icda_agent = icda_agent

        logger.info(
            f"ChatService: Causal components set - "
            f"hypothesis_pipeline={hypothesis_pipeline is not None}, "
            f"correlation_detector={correlation_detector is not None}, "
            f"insight_executor={insight_executor is not None}, "
            f"icda_agent={icda_agent is not None}"
        )

    async def chat_with_insights(
        self,
        session_id: str,
        message: str,
        context_entity_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ChatMessage:
        """Process a chat message with automatic insight surfacing.

        Enhanced version of chat() that proactively surfaces relevant
        causal insights when they would be helpful to the user.

        Research Foundation:
        - ZIA (2502.16124v1): Zero-input proactive AI
        - Causal-Copilot: Natural language causal exploration

        Args:
            session_id: Unique session identifier
            message: User's message
            context_entity_id: Optional PKG entity to focus on
            model: Optional model override

        Returns:
            ChatMessage with response and optional insights
        """
        # First get the regular response
        response = await self.chat(session_id, message, context_entity_id, model)

        # If insight surfacing is disabled, return as-is
        if not self._enable_insight_surfacing:
            return response

        # Check for relevant insights to surface
        try:
            insights = await self.get_causal_insights(session_id, topic=message)

            # Surface insights if relevant and not already shown
            if self._should_surface_insights(insights):
                augmented_content = self.surface_insight_in_response(
                    response.content, insights
                )

                # Mark insights as surfaced
                self._mark_insights_surfaced(insights)

                # Return augmented response
                return ChatMessage(
                    role=response.role,
                    content=augmented_content,
                    sources=response.sources,
                    entity_refs=response.entity_refs,
                    confidence=response.confidence,
                )
        except Exception as e:
            logger.debug(f"Insight surfacing failed: {e}")

        return response

    def _should_surface_insights(self, insights: Dict[str, Any]) -> bool:
        """Determine if insights should be surfaced.

        Prevents spamming the user with too many insights.

        Args:
            insights: Causal insights dict

        Returns:
            True if insights should be surfaced
        """
        # Check if there are any meaningful insights
        has_insights = (
            insights.get("correlations")
            or insights.get("hypotheses")
            or insights.get("pending_verifications")
        )

        if not has_insights:
            return False

        # Check if we've already surfaced these specific insights
        insight_ids = set()
        for corr in insights.get("correlations", []):
            insight_ids.add(corr.get("id", str(corr)))
        for hyp in insights.get("hypotheses", []):
            insight_ids.add(hyp.get("hypothesis_id", str(hyp)))
        for ver in insights.get("pending_verifications", []):
            insight_ids.add(ver.get("question_id", str(ver)))

        # Don't surface if all insights have been shown before
        if insight_ids and insight_ids.issubset(self._surfaced_insight_ids):
            return False

        return True

    def _mark_insights_surfaced(self, insights: Dict[str, Any]) -> None:
        """Mark insights as surfaced to avoid repetition.

        Args:
            insights: Causal insights that were surfaced
        """
        for corr in insights.get("correlations", []):
            self._surfaced_insight_ids.add(corr.get("id", str(corr)))
        for hyp in insights.get("hypotheses", []):
            self._surfaced_insight_ids.add(hyp.get("hypothesis_id", str(hyp)))
        for ver in insights.get("pending_verifications", []):
            self._surfaced_insight_ids.add(ver.get("question_id", str(ver)))

        # Limit the size of surfaced set to prevent memory growth
        if len(self._surfaced_insight_ids) > 1000:
            # Keep only the most recent 500
            self._surfaced_insight_ids = set(
                list(self._surfaced_insight_ids)[-500:]
            )

    async def respond_to_icda(
        self,
        session_id: str,
        question_id: str,
        user_response: str,
    ) -> ChatMessage:
        """Process a user response to an ICDA verification question.

        Handles user answers to causal hypothesis verification questions.

        Args:
            session_id: Current session ID
            question_id: ID of the ICDA question being answered
            user_response: User's answer (yes, no, partial, etc.)

        Returns:
            ChatMessage with acknowledgment and next steps
        """
        if not self._icda_agent:
            return ChatMessage(
                role="assistant",
                content="I'm not currently set up for causal verification. Please try again later.",
                confidence=0.5,
            )

        try:
            # Process the response through ICDA
            from futurnal.insights.interactive_causal import ICDAResponseType

            # Parse user response
            response_lower = user_response.lower().strip()
            if response_lower in ["yes", "y", "correct", "true"]:
                response_type = ICDAResponseType.YES_CAUSAL
            elif response_lower in ["no", "n", "false", "not causal"]:
                response_type = ICDAResponseType.NO_CONFOUNDER
            elif response_lower in ["reverse", "backwards", "other way"]:
                response_type = ICDAResponseType.REVERSE_CAUSATION
            elif response_lower in ["partial", "partially", "sometimes"]:
                response_type = ICDAResponseType.PARTIAL
            else:
                response_type = ICDAResponseType.UNSURE

            # Record the response
            result = self._icda_agent.record_response(
                question_id=question_id,
                response_type=response_type,
                explanation=user_response,
            )

            # Generate acknowledgment
            if result:
                confidence_change = result.get("confidence_change", 0)
                new_confidence = result.get("new_confidence", 0.5)

                if confidence_change > 0:
                    message = (
                        f"Thanks! Your input increased my confidence in this causal relationship "
                        f"(now {new_confidence:.0%}). "
                    )
                elif confidence_change < 0:
                    message = (
                        f"Noted! Your input decreased my confidence in this relationship "
                        f"(now {new_confidence:.0%}). "
                    )
                else:
                    message = "Thanks for your input! I've recorded your response. "

                # Check if there are more questions
                pending = self._icda_agent.get_pending_verifications(max_items=1)
                if pending:
                    message += f"\n\nI have another question when you're ready: {pending[0].main_question}"

                return ChatMessage(
                    role="assistant",
                    content=message,
                    confidence=0.9,
                )

        except Exception as e:
            logger.error(f"ICDA response processing failed: {e}")

        return ChatMessage(
            role="assistant",
            content="I've noted your response. Thank you for helping me understand the causal relationships in your data.",
            confidence=0.7,
        )

    async def get_pending_icda_questions(self, max_items: int = 3) -> List[Dict[str, Any]]:
        """Get pending ICDA verification questions.

        Returns:
            List of pending questions for user verification
        """
        if not self._icda_agent:
            return []

        try:
            pending = self._icda_agent.get_pending_verifications(max_items=max_items)
            return [
                {
                    "question_id": q.question_id,
                    "main_question": q.main_question,
                    "hypothesis_text": q.hypothesis_text if hasattr(q, 'hypothesis_text') else "",
                    "current_confidence": q.current_confidence if hasattr(q, 'current_confidence') else 0.5,
                }
                for q in pending
            ]
        except Exception as e:
            logger.debug(f"Failed to get pending ICDA questions: {e}")
            return []

    def get_insight_statistics(self) -> Dict[str, Any]:
        """Get statistics about causal insight surfacing.

        Returns:
            Dictionary with insight statistics
        """
        stats = {
            "surfaced_insights_count": len(self._surfaced_insight_ids),
            "insight_surfacing_enabled": self._enable_insight_surfacing,
            "components_available": {
                "icda_agent": self._icda_agent is not None,
                "hypothesis_pipeline": self._hypothesis_pipeline is not None,
                "correlation_detector": self._correlation_detector is not None,
                "insight_executor": self._insight_executor is not None,
            },
        }

        # Get executor statistics if available
        if self._insight_executor:
            try:
                executor_stats = self._insight_executor.get_statistics()
                stats["executor_statistics"] = executor_stats
            except Exception:
                pass

        return stats

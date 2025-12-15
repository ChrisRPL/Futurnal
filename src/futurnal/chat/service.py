"""Chat Service - Conversational Interface to Personal Knowledge Graph.

Research Foundation:
- ProPerSim (2509.21730v1): Proactive + personalized with multi-turn context
- Causal-Copilot (2504.13263v2): Natural language causal exploration

Production Plan Reference:
docs/phase-1/implementation-steps/03-chat-interface-conversational.md

Option B Compliance:
- Ghost model FROZEN - Ollama inference only, no fine-tuning
- Context-grounded generation prevents hallucination
- Local-first processing on localhost:11434
- Experiential learning via prompt refinement (not parameter updates)
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

    CHAT_SYSTEM_PROMPT = '''You are Futurnal's conversational knowledge assistant. Your role is to help
users explore and understand their personal knowledge graph through natural dialogue.

CRITICAL RULES:
1. ONLY use information from the provided context - NO external knowledge
2. Always cite sources using [Source: filename] notation
3. If the context doesn't contain the answer, say "I couldn't find this in your knowledge"
4. Be conversational but concise
5. Preserve the user's terminology and concepts from their notes
6. Reference previous conversation when relevant
7. If asked about something mentioned earlier, use the conversation history

FORMAT:
- Start with a direct answer to the question
- Support with evidence from context
- Reference sources inline
- If building on previous conversation, acknowledge it naturally'''

    def __init__(
        self,
        search_api: Optional["HybridSearchAPI"] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        storage: Optional[SessionStorage] = None,
        context_window_size: int = 10,  # 5 turns = 10 messages
    ) -> None:
        """Initialize chat service.

        Args:
            search_api: Optional pre-existing search API
            answer_generator: Optional pre-existing answer generator
            storage: Optional custom session storage
            context_window_size: Max messages to include in context (default: 10)
        """
        self._search_api = search_api
        self._answer_generator = answer_generator
        self._storage = storage or SessionStorage()
        self._sessions: Dict[str, ChatSession] = {}
        self._context_window_size = context_window_size
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize service components.

        Lazily initializes HybridSearchAPI and AnswerGenerator if not provided.
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

        # Initialize search API lazily if needed
        if self._search_api is None:
            try:
                from futurnal.search.api import HybridSearchAPI

                self._search_api = HybridSearchAPI()
                logger.info("ChatService: Initialized HybridSearchAPI")
            except Exception as e:
                logger.warning(f"ChatService: Could not init search API: {e}")

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

        # Stream response
        full_response = []
        async for chunk in self._answer_generator._pool.stream_generate(
            model=model or self._answer_generator.config.model_name,
            prompt=prompt,
            system=self.CHAT_SYSTEM_PROMPT,
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
        for msg in recent_messages[:-1]:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages for context
            content = msg.content[:300]
            if len(msg.content) > 300:
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

        # Build search query - optionally enhanced with context
        search_query = message

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

        prompt = self._build_chat_prompt(
            message=message,
            search_results=search_results,
            conversation_context=conversation_context,
            context_entity_id=context_entity_id,
        )

        try:
            response = await self._answer_generator._pool.generate(
                model=model or self._answer_generator.config.model_name,
                prompt=prompt,
                system=self.CHAT_SYSTEM_PROMPT,
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
            parts.append("Relevant knowledge from personal knowledge graph:")
            for i, result in enumerate(search_results[:8], 1):  # Limit to 8 results
                content = result.get("content", "")[:400]
                source = result.get("metadata", {}).get("source", "Unknown")
                parts.append(f"[Document {i} - {source}]\n{content}\n")
        else:
            parts.append("No relevant documents found in knowledge graph.")

        parts.append("")
        parts.append(f"User's question: {message}")
        parts.append("")
        parts.append("Provide a helpful, conversational response:")

        return "\n".join(parts)

    def _extract_sources_from_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract unique source names from search results."""
        sources = []
        seen = set()
        for result in search_results:
            source = result.get("metadata", {}).get("source")
            label = result.get("metadata", {}).get("label")
            name = label or source
            if name and name not in seen:
                sources.append(name)
                seen.add(name)
        return sources

    def _extract_entity_refs(
        self, search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract entity references from search results."""
        refs = []
        seen = set()
        for result in search_results:
            # Check for entity_id in result or metadata
            entity_id = result.get("entity_id") or result.get("metadata", {}).get(
                "entity_id"
            )
            if entity_id and entity_id not in seen:
                refs.append(entity_id)
                seen.add(entity_id)

            # Also check for entities in graph context
            graph_context = result.get("graph_context", {})
            for entity in graph_context.get("related_entities", []):
                eid = entity.get("id") or entity.get("name")
                if eid and eid not in seen:
                    refs.append(eid)
                    seen.add(eid)

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

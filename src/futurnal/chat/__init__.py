"""Chat Module - Conversational Interface to Personal Knowledge Graph.

Research Foundation:
- ProPerSim (2509.21730v1): Proactive + personalized AI
- Causal-Copilot (2504.13263v2): Natural language causal exploration

Production Plan Reference:
docs/phase-1/implementation-steps/03-chat-interface-conversational.md

This module provides:
- ChatService: Main conversational interface
- ChatMessage: Message data structure
- ChatSession: Conversation session management
- SessionStorage: Persistent storage for sessions

Example:
    >>> from futurnal.chat import ChatService
    >>> service = ChatService()
    >>> await service.initialize()
    >>> response = await service.chat("session-1", "What is Python?")
    >>> print(response.content)
"""

from futurnal.chat.models import (
    ChatMessage,
    ChatSession,
    SessionStorage,
)
from futurnal.chat.service import ChatService
from futurnal.chat.health import (
    check_intelligence_health,
    ComponentHealth,
    IntelligenceHealthReport,
)

__all__ = [
    # Core Chat
    "ChatService",
    "ChatMessage",
    "ChatSession",
    "SessionStorage",
    # Health Checks
    "check_intelligence_health",
    "ComponentHealth",
    "IntelligenceHealthReport",
]

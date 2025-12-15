"""Chat Models - Data structures for conversational interface.

Research Foundation:
- ProPerSim (2509.21730v1): Session tracking and preference learning
- Causal-Copilot (2504.13263v2): Confidence scoring in responses

Production Plan Reference:
docs/phase-1/implementation-steps/03-chat-interface-conversational.md

Option B Compliance:
- Ghost model FROZEN - no parameter updates
- Experiential learning via token priors (prepared for Phase 2)
- Local-only processing
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ChatMessage:
    """A message in a conversation.

    Research Foundation:
    - ProPerSim: Session tracking with timestamps
    - Causal-Copilot: Confidence scoring per response

    Attributes:
        role: Message sender ('user' or 'assistant')
        content: Message text content
        timestamp: When the message was created
        sources: Document sources used for response
        entity_refs: PKG entity IDs referenced in response
        confidence: Response confidence score (0.0-1.0)
    """

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)
    entity_refs: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Causal-Copilot confidence scoring

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources,
            "entity_refs": self.entity_refs,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sources=data.get("sources", []),
            entity_refs=data.get("entity_refs", []),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ChatSession:
    """A conversation session with context.

    Research Foundation:
    - ProPerSim: Multi-turn context carryover
    - Maintains conversation state for contextual responses

    Attributes:
        id: Unique session identifier
        messages: List of messages in the session
        context_entities: Accumulated PKG entities from conversation
        created_at: Session creation timestamp
        updated_at: Last activity timestamp
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[ChatMessage] = field(default_factory=list)
    context_entities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, message: ChatMessage) -> None:
        """Add a message and update timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now()

        # Accumulate entity references from assistant messages
        if message.role == "assistant" and message.entity_refs:
            for ref in message.entity_refs:
                if ref not in self.context_entities:
                    self.context_entities.append(ref)

    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages (for context window).

        Per ProPerSim: Rolling window of conversation history.

        Args:
            count: Maximum messages to return (default 10 = 5 turns)

        Returns:
            Most recent messages, up to count
        """
        return self.messages[-count:] if len(self.messages) > count else self.messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "context_entities": self.context_entities,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            messages=[ChatMessage.from_dict(m) for m in data.get("messages", [])],
            context_entities=data.get("context_entities", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class SessionStorage:
    """Persist chat sessions to JSON files.

    Storage location: ~/.futurnal/chat/sessions/

    Simple file-based storage for Phase 1.
    SQLite can be added in Phase 2 if needed.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """Initialize session storage.

        Args:
            base_path: Optional custom path. Defaults to ~/.futurnal/chat/sessions/
        """
        if base_path is None:
            base_path = Path.home() / ".futurnal" / "chat" / "sessions"
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path for a session file."""
        return self.base_path / f"{session_id}.json"

    def save(self, session: ChatSession) -> None:
        """Save session to file."""
        path = self._session_path(session.id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load(self, session_id: str) -> Optional[ChatSession]:
        """Load session from file."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ChatSession.from_dict(data)

    def delete(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return [p.stem for p in self.base_path.glob("*.json")]

    def get_all_sessions(self) -> List[ChatSession]:
        """Load all sessions."""
        sessions = []
        for session_id in self.list_sessions():
            session = self.load(session_id)
            if session:
                sessions.append(session)
        # Sort by most recent first
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

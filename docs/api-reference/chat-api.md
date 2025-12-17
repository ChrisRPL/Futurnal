# Chat API Reference

`ChatService` - Conversational interface to your personal knowledge graph.

## Overview

ChatService provides multi-turn conversational access to your knowledge:
- Grounded responses (only from your data)
- Source citations
- Session management
- Experiential learning integration

## Quick Example

```python
from futurnal.chat.service import ChatService
from futurnal.search.api import create_hybrid_search_api

async def main():
    # Create dependencies
    search_api = await create_hybrid_search_api()

    # Create chat service
    chat = ChatService(search_api=search_api)
    await chat.initialize()

    # Have a conversation
    response = await chat.chat("session-1", "What are my main projects?")
    print(response.content)

    response = await chat.chat("session-1", "Tell me more about the first one")
    print(response.content)

    # Cleanup
    await chat.close()
```

## ChatService Class

### Constructor

```python
class ChatService:
    def __init__(
        self,
        search_api: Optional[HybridSearchAPI] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        storage: Optional[SessionStorage] = None,
        context_window_size: int = 10,
        enable_experiential_learning: bool = True,
    )
```

**Parameters**:
- `search_api`: Search API for context retrieval (optional, created if None)
- `answer_generator`: LLM answer generator (optional, created if None)
- `storage`: Session storage backend (optional, in-memory if None)
- `context_window_size`: Max messages to include in context (default: 10)
- `enable_experiential_learning`: Inject learned priors into prompts (default: True)

**Example**:
```python
chat = ChatService(
    search_api=my_search_api,
    context_window_size=20,
    enable_experiential_learning=True,
)
```

### Methods

#### `initialize`

```python
async def initialize(self) -> None
```

Initialize chat service and dependencies.

**Example**:
```python
chat = ChatService()
await chat.initialize()
```

#### `chat`

```python
async def chat(
    self,
    session_id: str,
    message: str,
    *,
    stream: bool = False,
) -> Union[ChatMessage, AsyncIterator[str]]
```

Send a message and get a response.

**Parameters**:
- `session_id`: Unique session identifier
- `message`: User message
- `stream`: Stream response tokens (default: False)

**Returns**: `ChatMessage` or async iterator of tokens

**Example**:
```python
# Full response
response = await chat.chat("session-1", "What is Python?")
print(response.content)
print(f"Sources: {response.sources}")

# Streaming response
async for token in await chat.chat("session-1", "Tell me more", stream=True):
    print(token, end="", flush=True)
```

#### `chat_with_context`

```python
async def chat_with_context(
    self,
    session_id: str,
    message: str,
    *,
    additional_context: Optional[str] = None,
    temporal_focus: Optional[str] = None,
) -> ChatMessage
```

Chat with additional context or temporal focus.

**Parameters**:
- `session_id`: Session identifier
- `message`: User message
- `additional_context`: Extra context to include
- `temporal_focus`: Time period to emphasize

**Returns**: `ChatMessage`

**Example**:
```python
response = await chat.chat_with_context(
    "session-1",
    "What progress did I make?",
    temporal_focus="last week",
    additional_context="Focus on the API project"
)
```

#### `get_session`

```python
async def get_session(
    self,
    session_id: str,
) -> Optional[ChatSession]
```

Get a chat session by ID.

**Parameters**:
- `session_id`: Session identifier

**Returns**: `ChatSession` or None

**Example**:
```python
session = await chat.get_session("session-1")
if session:
    print(f"Messages: {len(session.messages)}")
    print(f"Created: {session.created_at}")
```

#### `list_sessions`

```python
async def list_sessions(
    self,
    *,
    limit: int = 10,
    offset: int = 0,
) -> List[ChatSession]
```

List chat sessions.

**Parameters**:
- `limit`: Maximum sessions to return
- `offset`: Pagination offset

**Returns**: List of `ChatSession`

**Example**:
```python
sessions = await chat.list_sessions(limit=5)
for session in sessions:
    print(f"{session.id}: {len(session.messages)} messages")
```

#### `clear_session`

```python
async def clear_session(
    self,
    session_id: str,
) -> None
```

Clear a session's message history.

**Parameters**:
- `session_id`: Session to clear

**Example**:
```python
await chat.clear_session("session-1")
```

#### `delete_session`

```python
async def delete_session(
    self,
    session_id: str,
) -> None
```

Delete a session entirely.

**Parameters**:
- `session_id`: Session to delete

**Example**:
```python
await chat.delete_session("session-1")
```

#### `export_session`

```python
async def export_session(
    self,
    session_id: str,
    format: str = "json",
) -> str
```

Export session history.

**Parameters**:
- `session_id`: Session to export
- `format`: Export format ("json" or "markdown")

**Returns**: Formatted session data

**Example**:
```python
# JSON export
json_data = await chat.export_session("session-1", format="json")

# Markdown export
markdown = await chat.export_session("session-1", format="markdown")
```

#### `close`

```python
async def close(self) -> None
```

Close service and cleanup resources.

**Example**:
```python
await chat.close()
```

## ChatMessage Class

```python
@dataclass
class ChatMessage:
    id: str                          # Unique message ID
    session_id: str                  # Parent session ID
    role: str                        # "user" or "assistant"
    content: str                     # Message content
    timestamp: datetime              # Message timestamp
    sources: List[str]               # Source citations
    confidence: float                # Response confidence (0.0 - 1.0)
    entity_references: List[str]     # Referenced entities
    metadata: Dict[str, Any]         # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
```

**Example**:
```python
response = await chat.chat("session-1", "What is Python?")

print(f"Content: {response.content}")
print(f"Confidence: {response.confidence:.2f}")
print(f"Sources: {response.sources}")
print(f"Entities: {response.entity_references}")
```

## ChatSession Class

```python
@dataclass
class ChatSession:
    id: str                          # Session ID
    created_at: datetime             # Creation time
    updated_at: datetime             # Last update time
    messages: List[ChatMessage]      # Message history
    metadata: Dict[str, Any]         # Session metadata
```

**Example**:
```python
session = await chat.get_session("session-1")

print(f"Created: {session.created_at}")
print(f"Messages: {len(session.messages)}")

for msg in session.messages[-5:]:  # Last 5 messages
    print(f"[{msg.role}]: {msg.content[:100]}...")
```

## SessionStorage Class

```python
class SessionStorage:
    """In-memory session storage (default)."""

    async def save(self, session: ChatSession) -> None: ...
    async def get(self, session_id: str) -> Optional[ChatSession]: ...
    async def list(self, limit: int, offset: int) -> List[ChatSession]: ...
    async def delete(self, session_id: str) -> None: ...
```

For persistent storage, implement your own:

```python
class SQLiteSessionStorage(SessionStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def save(self, session: ChatSession) -> None:
        # Save to SQLite
        ...

# Use custom storage
chat = ChatService(storage=SQLiteSessionStorage("sessions.db"))
```

## System Prompts

ChatService uses configurable system prompts:

```python
# Default system prompt
ChatService.CHAT_SYSTEM_PROMPT_BASE = '''
You are Futurnal's conversational knowledge assistant...
'''

# Experiential learning template
ChatService.EXPERIENTIAL_LEARNING_TEMPLATE = '''
EXPERIENTIAL KNOWLEDGE (learned from this user's data patterns):
{experiential_context}
'''
```

## Error Handling

```python
from futurnal.chat.errors import (
    ChatError,
    SessionNotFoundError,
    ContextRetrievalError,
    GenerationError,
)

try:
    response = await chat.chat("session-1", "question")
except SessionNotFoundError:
    print("Session not found, creating new one")
except ContextRetrievalError as e:
    print(f"Could not retrieve context: {e}")
except GenerationError as e:
    print(f"LLM generation failed: {e}")
except ChatError as e:
    print(f"Chat error: {e}")
```

## Streaming Responses

```python
async def stream_chat():
    response_text = ""

    async for token in await chat.chat("session-1", "question", stream=True):
        response_text += token
        print(token, end="", flush=True)

    print()  # Newline after streaming
    return response_text
```

## Multi-Turn Context

ChatService automatically maintains conversation context:

```python
# Turn 1
await chat.chat("session-1", "What are my Python projects?")

# Turn 2 - references turn 1 automatically
await chat.chat("session-1", "Which one is most active?")

# Turn 3 - continues context
await chat.chat("session-1", "Show me recent changes to it")
```

The `context_window_size` parameter controls how many messages are included.

## Experiential Learning Integration

When `enable_experiential_learning=True`, ChatService injects learned patterns:

```python
# Token priors are injected as natural language context
# Example experiential context:
"""
EXPERIENTIAL KNOWLEDGE:
- Entity patterns: User frequently discusses "FastAPI" in context of "API development"
- Relationship patterns: "Python" connects to "data analysis" and "web development"
- Temporal patterns: User typically writes about projects on Monday mornings
"""
```

This improves personalization without model fine-tuning (Option B compliance).

## Performance Tips

1. **Reuse sessions**: Sessions cache context
2. **Limit context**: Smaller `context_window_size` = faster
3. **Stream long responses**: Better UX with streaming
4. **Cleanup old sessions**: Delete unused sessions

```python
# Cleanup sessions older than 7 days
sessions = await chat.list_sessions(limit=100)
for session in sessions:
    if session.updated_at < datetime.now() - timedelta(days=7):
        await chat.delete_session(session.id)
```

## Research Foundation

ChatService implements research from:
- ProPerSim (2509.21730v1): Proactive + personalized assistants
- Causal-Copilot (2504.13263v2): Natural language causal exploration
- SEAgent (2508.04700v2): Experiential learning integration

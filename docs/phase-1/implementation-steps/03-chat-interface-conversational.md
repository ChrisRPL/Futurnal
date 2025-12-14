# Step 03: Chat Interface & Conversational AI

## Status: TODO

## Objective

Create a chat interface that allows users to have conversations with their personal knowledge graph. This transforms Futurnal from a search tool into an interactive knowledge assistant.

## Research Foundation

### Primary Papers:

#### ProPerSim (2509.21730v1) - Proactive and Personalized AI
**Key Innovation**: Combines proactivity AND personalization with continual learning
**Application**: Chat interface that learns user preferences over time

#### Causal-Copilot (2504.13263v2) - Autonomous Causal Analysis Agent
**Key Innovation**: Natural language interface for causal exploration
**Application**: Users can ask "why" questions about their data

### Research Insight from FUTURNAL_CONCEPT.md:
> "Phase 3: The Guide → The Emergent Animal Brain: The user engages in **conversational causal exploration**, guided by the Animal's sophisticated world model."
> - Section 3.2

While full conversational causal exploration is Phase 3, **basic Q&A chat is essential for Phase 1** to demonstrate the Ghost's value.

## Current State Analysis

### What Exists:
1. **Command Palette** - `desktop/src/components/search/CommandPalette.tsx`
2. **Search API** - Can handle queries
3. **Answer Generation** - Added in Step 02

### What's Missing:
- No conversation history
- No context carryover between queries
- No chat-style UI
- No "Ask about this" on graph nodes

## Implementation Tasks

### 1. Create Chat Backend Service

**New File**: `src/futurnal/chat/service.py`

```python
"""
Chat Service - Conversational Interface to PKG

Research Foundation:
- ProPerSim (2509.21730v1): Proactive + personalized
- Causal-Copilot (2504.13263v2): Natural language causal exploration
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from futurnal.search.api import HybridSearchAPI
from futurnal.search.answer_generator import AnswerGenerator

@dataclass
class ChatMessage:
    """A message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)
    entity_refs: List[str] = field(default_factory=list)  # PKG entities referenced

@dataclass
class ChatSession:
    """A conversation session with context."""
    id: str
    messages: List[ChatMessage] = field(default_factory=list)
    context_entities: List[str] = field(default_factory=list)  # Accumulated context

class ChatService:
    """Conversational interface to personal knowledge graph."""

    def __init__(self):
        self.search_api = HybridSearchAPI()
        self.answer_generator = AnswerGenerator()
        self.sessions: Dict[str, ChatSession] = {}

    async def chat(
        self,
        session_id: str,
        message: str,
        context_entity_id: Optional[str] = None,
    ) -> ChatMessage:
        """
        Process a chat message and generate response.

        Args:
            session_id: Unique session identifier
            message: User's message
            context_entity_id: Optional PKG entity to focus on (for "Ask about this")
        """
        # Get or create session
        session = self._get_or_create_session(session_id)

        # Add user message to history
        user_msg = ChatMessage(role='user', content=message)
        session.messages.append(user_msg)

        # Build context from conversation history + optional entity
        context = await self._build_conversation_context(session, context_entity_id)

        # Search with conversation context
        search_results = await self.search_api.search_with_answer(
            query=message,
            top_k=10,
            generate_answer=False,  # We'll generate with full context
        )

        # Generate response with full conversation context
        response = await self._generate_response(
            message=message,
            search_results=search_results['results'],
            conversation_context=context,
            session=session,
        )

        # Create assistant message
        assistant_msg = ChatMessage(
            role='assistant',
            content=response,
            sources=search_results['sources'],
            entity_refs=self._extract_entity_refs(search_results['results']),
        )
        session.messages.append(assistant_msg)

        # Update session context entities
        session.context_entities.extend(assistant_msg.entity_refs)

        return assistant_msg

    async def _build_conversation_context(
        self,
        session: ChatSession,
        entity_id: Optional[str],
    ) -> str:
        """Build context from conversation history."""
        parts = []

        # Add conversation history (last 5 turns)
        recent_messages = session.messages[-10:]  # 5 turns = 10 messages
        if recent_messages:
            parts.append("Previous conversation:")
            for msg in recent_messages:
                role = "User" if msg.role == 'user' else "Assistant"
                parts.append(f"{role}: {msg.content[:200]}")

        # Add entity context if provided
        if entity_id:
            entity_context = await self._get_entity_context(entity_id)
            parts.append(f"\nFocus entity context:\n{entity_context}")

        return "\n".join(parts)

    async def _generate_response(
        self,
        message: str,
        search_results: List[Dict],
        conversation_context: str,
        session: ChatSession,
    ) -> str:
        """Generate contextual response."""
        prompt = f"""You are having a conversation about the user's personal knowledge.

{conversation_context}

Current context from knowledge graph:
{self._format_search_results(search_results)}

User's current question: {message}

Respond naturally, referencing previous conversation where relevant.
Cite sources as [Source: filename].
If you don't have enough information, ask clarifying questions."""

        return await self.answer_generator.generate_answer(
            query=message,
            context=search_results,
        )

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(id=session_id)
        return self.sessions[session_id]
```

### 2. Create Chat API Endpoints

**File**: `src/futurnal/chat/api.py`

```python
"""Chat API endpoints for Tauri IPC."""

from futurnal.chat.service import ChatService, ChatMessage

chat_service = ChatService()

async def send_message(
    session_id: str,
    message: str,
    context_entity_id: str | None = None,
) -> dict:
    """Handle chat message from frontend."""
    response = await chat_service.chat(
        session_id=session_id,
        message=message,
        context_entity_id=context_entity_id,
    )
    return {
        'content': response.content,
        'sources': response.sources,
        'entity_refs': response.entity_refs,
        'timestamp': response.timestamp.isoformat(),
    }

async def get_session_history(session_id: str) -> list:
    """Get conversation history for session."""
    session = chat_service.sessions.get(session_id)
    if not session:
        return []
    return [
        {
            'role': msg.role,
            'content': msg.content,
            'sources': msg.sources,
            'timestamp': msg.timestamp.isoformat(),
        }
        for msg in session.messages
    ]
```

### 3. Create Chat UI Component

**New File**: `desktop/src/components/chat/ChatInterface.tsx`

```tsx
/**
 * ChatInterface - Conversational interface to PKG
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Proactive + personalized
 * - Causal-Copilot: Natural language exploration
 */

import { useState, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { Send, Loader2, Link2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useGraphStore } from '@/stores/graphStore';
import { Markdown } from '@/components/ui/markdown';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  entityRefs?: string[];
  timestamp: string;
}

interface ChatInterfaceProps {
  sessionId: string;
  contextEntityId?: string;  // For "Ask about this" feature
  onEntityClick?: (entityId: string) => void;
}

export function ChatInterface({
  sessionId,
  contextEntityId,
  onEntityClick,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
    }]);

    setIsLoading(true);
    try {
      const response = await invoke<ChatMessage>('send_chat_message', {
        sessionId,
        message: userMessage,
        contextEntityId,
      });

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.content,
        sources: response.sources,
        entityRefs: response.entityRefs,
        timestamp: response.timestamp,
      }]);
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-black">
      {/* Context indicator */}
      {contextEntityId && (
        <div className="px-4 py-2 border-b border-white/10 text-sm text-white/60">
          <Link2 className="inline-block w-4 h-4 mr-2" />
          Discussing: {contextEntityId}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              'max-w-[80%] p-3 rounded-lg',
              msg.role === 'user'
                ? 'ml-auto bg-white/10 text-white'
                : 'mr-auto bg-white/5 text-white/90'
            )}
          >
            <Markdown>{msg.content}</Markdown>
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-2 text-xs text-white/40">
                Sources: {msg.sources.filter(Boolean).join(', ')}
              </div>
            )}
            {msg.entityRefs && msg.entityRefs.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {msg.entityRefs.map(ref => (
                  <button
                    key={ref}
                    onClick={() => onEntityClick?.(ref)}
                    className="text-xs px-1.5 py-0.5 bg-white/5 rounded hover:bg-white/10"
                  >
                    {ref}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex items-center gap-2 text-white/40">
            <Loader2 className="w-4 h-4 animate-spin" />
            Thinking...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about your knowledge..."
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white placeholder-white/30 focus:outline-none focus:border-white/20"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="px-4 py-2 bg-white text-black rounded-lg hover:bg-white/90 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 4. Add "Ask About This" to Graph

**File**: `desktop/src/pages/Graph.tsx`

```tsx
// Add chat panel state
const [chatOpen, setChatOpen] = useState(false);
const [chatContextEntity, setChatContextEntity] = useState<string | null>(null);

// Add "Ask about this" to node context menu
const handleAskAbout = (nodeId: string) => {
  setChatContextEntity(nodeId);
  setChatOpen(true);
};

// In node click handler or context menu
<button onClick={() => handleAskAbout(node.id)}>
  Ask about this
</button>

// Render chat panel
{chatOpen && (
  <div className="absolute right-0 top-0 bottom-0 w-96 border-l border-white/10">
    <ChatInterface
      sessionId={`graph-${Date.now()}`}
      contextEntityId={chatContextEntity}
      onEntityClick={(id) => highlightNode(id)}
    />
  </div>
)}
```

### 5. Add Chat to Dashboard

**File**: `desktop/src/pages/Dashboard.tsx`

```tsx
// Add chat panel to dashboard
<section className="chat-section">
  <h2>Ask Your Knowledge</h2>
  <ChatInterface
    sessionId="dashboard-main"
    onEntityClick={(id) => navigate(`/graph?highlight=${id}`)}
  />
</section>
```

## Success Criteria

### Functional:
- [ ] Multi-turn conversation with context carryover
- [ ] "Ask about this" feature on graph nodes
- [ ] Source citations in responses
- [ ] Entity references link to graph

### Quality:
- [ ] Responses are contextually aware
- [ ] Conversation history influences answers
- [ ] Entity focus provides relevant context

### UX:
- [ ] Smooth typing and sending
- [ ] Loading states visible
- [ ] Chat integrates with graph navigation

## Files to Create/Modify

### Backend:
- **NEW**: `src/futurnal/chat/service.py` - Chat service
- **NEW**: `src/futurnal/chat/api.py` - API endpoints
- **NEW**: `src/futurnal/chat/__init__.py` - Package init

### Frontend:
- **NEW**: `desktop/src/components/chat/ChatInterface.tsx` - Chat UI
- **NEW**: `desktop/src/components/chat/index.ts` - Export
- `desktop/src/pages/Graph.tsx` - Add "Ask about this"
- `desktop/src/pages/Dashboard.tsx` - Add chat panel

### Tauri:
- Add IPC commands for chat messages

## Dependencies

- **Step 01**: Intelligent search (for retrieval)
- **Step 02**: Answer generation (for responses)

## Next Step

After implementing chat interface, proceed to **Step 04: Temporal Extraction**.

## Research References

1. **ProPerSim**: `docs/phase-1/papers/converted/2509.21730v1.md`
2. **Causal-Copilot**: `docs/phase-1/papers/converted/2504.13263v2.md`
3. **FUTURNAL_CONCEPT.md**: Section 3.2 (conversational exploration)

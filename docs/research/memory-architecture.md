# Memory Architecture

## Overview

Futurnal implements a three-tier memory architecture aligned with
cognitive science principles and 2025 SOTA research.

## Research Alignment

| Research Term | Futurnal Component | Purpose | Max Entries |
|--------------|-------------------|---------|-------------|
| Working Memory | EvolvingMemoryBuffer | Active context | 50 |
| Episodic Memory | Insight Cache | Recent events | 200 |
| Semantic Memory | TokenPriorStore | Long-term patterns | 100/category |

## Research Foundation

### H-MEM (arxiv:2507.22925)

Three-tier hierarchical memory for long-term reasoning:
- Short-term (Working): Immediate context
- Medium-term (Episodic): Recent experiences
- Long-term (Semantic): Consolidated knowledge

### A-MEM (arxiv:2502.12110)

Zettelkasten-inspired self-organizing memory:
- Dynamic linking between memories
- Emergent organization
- Semantic clustering

### MemR3 (arxiv:2512.20237)

Reflective reasoning for memory retrieval:
- Query-aware filtering
- Relevance scoring
- Context gating

## Implementation

### TokenPriorStore (Semantic Memory)

```python
from futurnal.learning.token_priors import TokenPriorStore

store = TokenPriorStore(capacity=100, min_confidence=0.5)

# Update from extraction
store.update_from_extraction(result, success=True)

# Generate context for prompts
context = store.generate_prompt_context(query="meetings last week")
```

### EvolvingMemoryBuffer (Working Memory)

```python
from futurnal.agents.memory_buffer import EvolvingMemoryBuffer, MemoryEntry

buffer = EvolvingMemoryBuffer(max_entries=50)

# Add entry
entry = MemoryEntry(
    entry_type=MemoryEntryType.HYPOTHESIS,
    content="Monday meetings correlate with productivity",
    priority=MemoryPriority.NORMAL,
)
buffer.add_entry(entry)
```

### HierarchicalMemory (Unified Interface)

```python
from futurnal.memory import HierarchicalMemory, create_hierarchical_memory

memory = create_hierarchical_memory()

# Add to appropriate tier
memory.add_to_working_memory("Current hypothesis...")
memory.add_to_episodic_memory("Meeting occurred", "meeting")

# Consolidate episodic to semantic
memory.consolidate_episodic_to_semantic()
```

## Memory Flow

```
User Interaction
      ↓
┌─────────────────────────────────────┐
│ Working Memory (EvolvingMemoryBuffer)│
│ - Current session context            │
│ - Priority-based retention           │
│ - Max 50 entries                     │
└─────────────────────────────────────┘
      ↓ (session end)
┌─────────────────────────────────────┐
│ Episodic Memory (Insight Cache)      │
│ - Recent experiences                 │
│ - Event-indexed                      │
│ - 30-day retention                   │
└─────────────────────────────────────┘
      ↓ (consolidation)
┌─────────────────────────────────────┐
│ Semantic Memory (TokenPriorStore)    │
│ - Long-term patterns                 │
│ - Entity/relation/temporal priors    │
│ - Persists across sessions           │
└─────────────────────────────────────┘
```

## Option B Compliance

All memory components maintain strict Option B compliance:
- **Ghost model FROZEN** - No parameter updates
- **Natural language storage** - All priors are text
- **Local-only** - No cloud synchronization

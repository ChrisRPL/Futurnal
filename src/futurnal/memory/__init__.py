"""Memory module - Hierarchical memory architecture.

Research Foundation:
- H-MEM (arxiv:2507.22925): Three-tier hierarchical memory for long-term reasoning
- A-MEM (arxiv:2502.12110): Zettelkasten-inspired self-organizing memory
- MemR³ (arxiv:2512.20237): Reflective reasoning for memory retrieval

Memory Tier Mapping:
┌──────────────────────────────────────────────────────────────────┐
│ Research Term   │ Futurnal Component      │ Purpose             │
├──────────────────────────────────────────────────────────────────┤
│ Working Memory  │ EvolvingMemoryBuffer    │ Active context (50) │
│ Episodic Memory │ Insight Cache           │ Recent events (200) │
│ Semantic Memory │ TokenPriorStore         │ Long-term (100/cat) │
└──────────────────────────────────────────────────────────────────┘

Option B Compliance:
- Ghost model FROZEN
- All memory stored as natural language (token priors)
- No parameter updates
"""

from futurnal.memory.hierarchical_memory import (
    HierarchicalMemory,
    MemoryTierStats,
    create_hierarchical_memory,
)

__all__ = [
    "HierarchicalMemory",
    "MemoryTierStats",
    "create_hierarchical_memory",
]

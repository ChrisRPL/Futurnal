# Self-Evolution Approach

## Research Positioning

Futurnal's learning architecture aligns with the "Survey of Self-Evolving Agents"
(arxiv:2507.21046) taxonomy, implementing **Context Evolution** rather than
Model Evolution.

## Taxonomy Alignment

```
Self-Evolution Dimensions (arxiv:2507.21046):
├── What to Evolve
│   ├── Model (NOT us - Ghost frozen)
│   ├── Context ← OUR APPROACH (Token Priors)
│   ├── Tool
│   └── Architecture
├── When to Evolve
│   ├── Intra-test-time ← OUR APPROACH (per-session learning)
│   └── Inter-test-time ← OUR APPROACH (cross-session consolidation)
└── How to Evolve
    ├── Reward-based ← OUR APPROACH (user feedback)
    ├── Imitation
    └── Population-based
```

## Why Context Evolution?

### Advantages

1. **Privacy-preserving** - No model updates means no data leakage
2. **Local-first** - Works entirely on-device
3. **Interpretable** - All knowledge is natural language
4. **Reversible** - Priors can be reviewed and removed
5. **No API dependency** - Works with any LLM

### Trade-offs Accepted

1. **Bounded improvement** - Learning capped by prior capacity
2. **Transfer limitations** - Knowledge doesn't transfer across contexts
3. **Slower adaptation** - Multiple examples needed for pattern learning

## Implementation Details

### Token Prior Store (`src/futurnal/learning/token_priors.py`)

- Stores experiential knowledge as natural language
- Max 100 priors per category (entity, relation, temporal)
- Confidence-based pruning
- Query-aware filtering via SemanticContextGate

### Evolving Memory Buffer (`src/futurnal/agents/memory_buffer.py`)

- Working memory for active sessions
- Priority-based retention (max 50 entries)
- Automatic compression of old entries

### Hierarchical Memory (`src/futurnal/memory/hierarchical_memory.py`)

- Aligns with H-MEM three-tier model
- Consolidation from episodic to semantic
- Unified interface for all memory tiers

## Option B Compliance

- Ghost model FROZEN (no parameter updates)
- All knowledge stored as natural language (token priors)
- No cloud model updates
- Quality gates validated

## Research References

- **Training-Free GRPO** (arxiv:2510.08191v1): Token priors for experiential learning
- **A-MEM** (arxiv:2502.12110): Zettelkasten-inspired memory organization
- **H-MEM** (arxiv:2507.22925): Hierarchical memory for long-term reasoning
- **Self-Evolving Agents Survey** (arxiv:2507.21046): Taxonomy we adopt

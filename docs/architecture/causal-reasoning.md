# Causal Reasoning Architecture

## Overview

Futurnal implements a three-layer causal architecture that explicitly separates
LLM pattern detection from causal validation. This addresses the fundamental
limitation that LLMs cannot perform genuine causal reasoning (best: 57.6% on
causal benchmarks).

## Research Foundation

- **arxiv:2510.07231** - Benchmarking LLM Causal Reasoning (Oct 2025)
- **arxiv:2503.00237** - Agentic AI Needs Systems Theory (Mar 2025)
- **Bradford Hill (1965)** - Original criteria for distinguishing association from causation

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUTURNAL CAUSAL ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: Pattern Detection (LLM-Powered)                       │
│  ├── TemporalCorrelationDetector: Statistical co-occurrence     │
│  ├── HypothesisGenerator: LLM-based mechanism proposals         │
│  └── Output: correlation_confidence (NOT causal)                │
│                           ↓                                      │
│  LAYER 2: Causal Validation (Structured Algorithms)             │
│  ├── BradfordHillValidator: 9-criteria scoring                  │
│  ├── TemporalOrderingValidator: Cause precedes effect           │
│  ├── DOTSCausalOrdering: Temporal precedence ordering           │
│  └── Output: causal_confidence (algorithm-validated)            │
│                           ↓                                      │
│  LAYER 3: Human Verification (User-in-Loop)                     │
│  ├── InteractiveCausalDiscoveryAgent: Question generation       │
│  ├── Evidence presentation with both confidence types           │
│  └── Output: verified_confidence (human-validated)              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### CausalBoundary (`src/futurnal/reasoning/causal_boundary.py`)

The explicit abstraction layer that enforces LLM/causal separation:

```python
from futurnal.reasoning import CausalBoundary, create_causal_boundary

boundary = create_causal_boundary(llm_client=client)
result = await boundary.process_correlation(correlation)

# result.correlation_confidence - LLM-derived (Layer 1)
# result.causal_confidence - Algorithm-validated (Layer 2)
# result.verified_confidence - Human-validated (Layer 3)
```

### Bradford-Hill Criteria (`src/futurnal/search/causal/bradford_hill.py`)

Scores potential causal relationships on 9 criteria:

1. **Strength** - Strong associations more likely causal
2. **Consistency** - Repeated observation across contexts
3. **Specificity** - Specific cause leads to specific effect
4. **Temporality** - Cause precedes effect (100% enforced)
5. **Biological Gradient** - Dose-response relationship
6. **Plausibility** - Mechanism is logically plausible
7. **Coherence** - Doesn't conflict with known facts
8. **Experiment** - Experimental evidence supports causation
9. **Analogy** - Similar relationships are known to be causal

### DOTS Ordering (`src/futurnal/search/temporal/dots_ordering.py`)

Establishes causal ordering before structure learning:

- Computes pairwise temporal precedence
- Builds precedence DAG
- Filters correlations by causal order
- Improves accuracy from 63% to ~80% (DOTS paper results)

## Confidence Types

| Type | Source | Meaning |
|------|--------|---------|
| `correlation_confidence` | LLM (Layer 1) | Pattern detected, NOT validated |
| `causal_confidence` | Algorithms (Layer 2) | Bradford-Hill validated |
| `verified_confidence` | Human (Layer 3) | User-confirmed |

## Option B Compliance

This architecture maintains strict Option B compliance:

- **Ghost model FROZEN** - No parameter updates
- **Learning via token priors** - All knowledge stored as natural language
- **No gradient computation** - Algorithms are rule-based

## Usage Guidelines

### DO:
- Use `correlation_confidence` for pattern detection results
- Use `causal_confidence` only after Bradford-Hill validation
- Show both confidence types in UI
- Submit low-confidence hypotheses to ICDA for verification

### DON'T:
- Present LLM-derived patterns as "causal"
- Skip Bradford-Hill validation for causal claims
- Use `causal_confidence` without algorithm validation

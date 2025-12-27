Summary: Research integration sprint addressing 2025 SOTA findings to strengthen Futurnal's scientific foundation and competitive moat.

# Phase 2.5 — Research Integration Sprint

## Purpose
Phase 2.5 is a focused integration sprint that strengthens Futurnal's core features with 2025 SOTA research findings. This is NOT about adding new features—it's about making existing features scientifically rigorous and defensibly innovative.

**Duration**: 2-3 weeks
**Goal**: Align implementation with cutting-edge research, address known limitations, establish competitive moat

---

## 1. Hybrid Causal Architecture (Critical Priority)

### The Problem
2025 research confirms LLMs cannot perform genuine causal reasoning:
- Best LLM achieves only 57.6% accuracy on causal benchmarks (arxiv:2510.07231)
- LLMs limited to "level-1 causal reasoning" (CausalProbe 2024)
- "LLMs lack true causal reasoning" (Agentic AI Needs Systems Theory, 2025)

Our current implementation uses LLMs for hypothesis generation, which is correct—but we need explicit architectural separation.

### The Solution: Explicit Causal Boundary

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUTURNAL CAUSAL ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: Pattern Detection (LLM-Powered)                       │
│  ├── Correlation identification in experiential data            │
│  ├── Hypothesis generation from temporal patterns               │
│  └── Natural language insight formulation                       │
│                           ↓                                      │
│  LAYER 2: Causal Validation (Structured Algorithms)             │
│  ├── Bradford Hill criteria scoring                             │
│  ├── Statistical temporal tests (Granger, Transfer Entropy)     │
│  └── Confounding factor detection                               │
│                           ↓                                      │
│  LAYER 3: Human Verification (User-in-Loop)                     │
│  ├── ICDA conversational exploration                            │
│  ├── Evidence presentation with confidence levels               │
│  └── Final judgment preserved for user                          │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

**File**: `src/futurnal/reasoning/causal_boundary.py`
```python
class CausalBoundary:
    """
    Explicit separation between LLM pattern detection and causal validation.

    Research basis:
    - arxiv:2510.07231 (LLM causal benchmarks 2025)
    - arxiv:2503.00237 (Agentic AI Systems Theory)
    """

    def __init__(self):
        self.llm_detector = PatternDetector()  # Level 1: Correlations
        self.causal_validator = BradfordHillValidator()  # Level 2: Causation
        self.evidence_presenter = EvidencePresenter()  # Level 3: User

    def process_hypothesis(self, hypothesis: Hypothesis) -> CausalResult:
        # LLM generates hypothesis (correlation-based)
        patterns = self.llm_detector.find_patterns(hypothesis)

        # Structured algorithms validate causality
        validation = self.causal_validator.validate(patterns)

        # Present with explicit uncertainty
        return self.evidence_presenter.format(
            hypothesis=hypothesis,
            patterns=patterns,
            validation=validation,
            llm_confidence=patterns.confidence,
            causal_confidence=validation.bradford_hill_score,
        )
```

**Update existing files**:
- `src/futurnal/insights/emergent_insights.py` - Add causal boundary integration
- `src/futurnal/analysis/hypothesis_generator.py` - Explicitly label as "correlation-based"
- `src/futurnal/cli/insights.py` - Show both correlation and causal confidence

### Documentation Update
Add to `docs/architecture/causal-reasoning.md`:
- Diagram showing the three-layer architecture
- Explicit statement: "LLMs identify correlations; structured algorithms validate causation"
- Reference to 2025 benchmarks proving this approach is necessary

---

## 2. Memory Architecture Alignment (A-MEM/H-MEM)

### Research Alignment
Our Token Prior system aligns with 2025 SOTA memory research:
- **A-MEM** (arxiv:2502.12110): Zettelkasten-inspired self-organizing memory
- **H-MEM** (arxiv:2507.22925): Hierarchical memory for long-term reasoning
- **MemR³** (arxiv:2512.20237): Reflective reasoning for memory retrieval

### Adopt Standard Terminology
Rename internal concepts to align with research community:

| Current Name | Research Term | Purpose |
|--------------|---------------|---------|
| Token Priors | Experiential Memory | Long-term learned patterns |
| Evolving Memory Buffer | Working Memory | Active reasoning context |
| Insight Cache | Episodic Memory | Recent discoveries |

### Implementation Tasks

**File**: `src/futurnal/memory/hierarchical_memory.py`
```python
class HierarchicalMemory:
    """
    Three-tier memory architecture aligned with H-MEM (arxiv:2507.22925).

    Tiers:
    1. Working Memory - Active context window (LLM context)
    2. Episodic Memory - Recent experiences and insights
    3. Semantic Memory - Consolidated experiential patterns (Token Priors)
    """

    def __init__(self):
        self.working = WorkingMemory(max_tokens=8000)
        self.episodic = EpisodicMemory(retention_days=30)
        self.semantic = SemanticMemory()  # Token Priors

    def consolidate(self):
        """
        Consolidation process inspired by hippocampal memory formation.
        Moves significant episodic memories to semantic storage.
        """
        significant_episodes = self.episodic.get_significant()
        for episode in significant_episodes:
            self.semantic.integrate(episode)
```

**Update existing files**:
- `src/futurnal/learning/token_priors.py` - Add docstring referencing A-MEM
- `src/futurnal/agents/memory_buffer.py` - Rename to align with H-MEM terminology

### Why This Matters
- Positions Futurnal in academic discourse
- Makes architecture defensible in technical discussions
- Enables future publication opportunities

---

## 3. Temporal Causal Discovery Enhancement

### Integrate DOTS Algorithm
DOTS (arxiv:2510.24639, Oct 2025) achieves F1 0.81 vs 0.63 baseline for temporal causal discovery.

### Implementation Tasks

**File**: `src/futurnal/search/temporal/dots_ordering.py`
```python
class DOTSCausalOrdering:
    """
    Causal ordering for temporal structure learning.
    Based on DOTS (arxiv:2510.24639).

    Key insight: Establish causal ordering before structure learning
    to reduce search space and improve accuracy.
    """

    def compute_causal_order(
        self,
        events: List[ExperientialEvent],
        max_lag: int = 72  # hours
    ) -> CausalOrder:
        # 1. Compute pairwise temporal precedence
        precedence = self._compute_precedence_matrix(events)

        # 2. Apply DOTS ordering algorithm
        order = self._dots_order(precedence)

        # 3. Prune implausible causal directions
        return self._prune_with_domain_knowledge(order)
```

**Enhance existing correlation detector**:
- `src/futurnal/search/temporal/correlation.py` - Add DOTS pre-ordering step
- `src/futurnal/analysis/correlation_detector.py` - Improve accuracy with causal ordering

### Expected Improvement
- Correlation detection accuracy: 63% → ~80%
- Fewer false positive causal claims
- More defensible insights

---

## 4. Self-Evolution Documentation

### Align with Survey Taxonomy
The "Survey of Self-Evolving Agents" (arxiv:2507.21046) provides a taxonomy we should explicitly adopt:

```
Self-Evolution Dimensions:
├── What to Evolve
│   ├── Model (NOT us - frozen ghost)
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

### Documentation Task
Create `docs/research/self-evolution-approach.md`:
- Position Futurnal's approach within the taxonomy
- Explain why "Context Evolution" is chosen over "Model Evolution"
- Reference Option B architecture and its advantages

---

## 5. PyWhy Integration (Optional Enhancement)

### Consideration
PyWhy is the leading open-source causal ML ecosystem. Integration could provide:
- Statistical validation of causal claims
- Community-standard causal graph algorithms
- Credibility with technical users

### Assessment Needed
Before implementing, evaluate:
- Does it add meaningful capability beyond Bradford Hill?
- Is the dependency worth the complexity?
- Does it work well in local-first architecture?

### Recommendation
**Defer to Phase 4** unless:
- Users request stronger statistical backing
- We pursue academic publication

---

## Validation Criteria

Phase 2.5 is complete when:

1. **Causal Boundary Documented**
   - [ ] Architecture diagram in docs
   - [ ] Code comments reference research papers
   - [ ] UI explicitly shows "correlation" vs "causal" confidence

2. **Memory Terminology Aligned**
   - [ ] Token Priors documented as "Semantic Memory" (A-MEM aligned)
   - [ ] Memory buffer uses H-MEM three-tier model
   - [ ] Docstrings reference 2025 papers

3. **Temporal Discovery Enhanced**
   - [ ] DOTS ordering integrated (or documented why not)
   - [ ] Correlation accuracy measured against baseline
   - [ ] False positive rate reduced

4. **Research Position Clear**
   - [ ] `docs/research/` directory with position papers
   - [ ] Key differentiators explicitly stated
   - [ ] Publication pathway identified

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/futurnal/reasoning/causal_boundary.py` | Explicit LLM/causal separation |
| `src/futurnal/memory/hierarchical_memory.py` | H-MEM aligned architecture |
| `src/futurnal/search/temporal/dots_ordering.py` | DOTS algorithm integration |
| `docs/architecture/causal-reasoning.md` | Causal architecture documentation |
| `docs/research/self-evolution-approach.md` | Research positioning |
| `docs/research/memory-architecture.md` | Memory system documentation |

## Files to Update

| File | Change |
|------|--------|
| `src/futurnal/learning/token_priors.py` | Add A-MEM reference in docstring |
| `src/futurnal/insights/emergent_insights.py` | Integrate causal boundary |
| `src/futurnal/search/temporal/correlation.py` | Add DOTS pre-ordering |
| `CLAUDE.md` | Add research references section |

---

## Research References

### Primary Papers (2025)

1. **LLM Causal Limitations**
   - arxiv:2510.07231 - Benchmarking LLM Causal Reasoning (Oct 2025)
   - arxiv:2503.00237 - Agentic AI Needs Systems Theory (Mar 2025)

2. **Memory Architecture**
   - arxiv:2502.12110 - A-MEM: Agentic Memory (Feb 2025)
   - arxiv:2507.22925 - H-MEM: Hierarchical Memory (Jul 2025)
   - arxiv:2512.20237 - MemR³: Reflective Memory (Dec 2025)

3. **Self-Evolving Agents**
   - arxiv:2507.21046 - Survey of Self-Evolving Agents (Jul 2025)
   - arxiv:2504.20073 - RAGEN: Multi-Turn RL Self-Evolution (Apr 2025)

4. **Temporal Causal Discovery**
   - arxiv:2510.24639 - DOTS: Causal Ordering (Oct 2025)
   - arxiv:2507.09439 - DyCAST-Net: Sparse Attention (Jul 2025)

---

## Anti-Patterns to Avoid

1. **Over-Engineering**
   - Don't implement PyWhy just because it exists
   - Don't add complexity without clear user benefit
   - Keep the local-first, simple architecture

2. **Academic Creep**
   - Research alignment is for credibility, not complexity
   - Users don't need to know the paper references
   - Focus on outcomes, not methodology pride

3. **Premature Optimization**
   - DOTS is optional if current accuracy is acceptable
   - Memory renaming is documentation, not refactoring
   - Causal boundary is architectural clarity, not new features

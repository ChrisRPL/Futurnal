# Phase 2 & Phase 3 Architecture Notes

**Purpose**: Preserve architecture vision for Phase 2 (Analyst) and Phase 3 (Guide) to be used when creating detailed markdown folder plans after Phase 1 completion.

**Status**: PLANNING - Not for implementation yet
**Created**: Based on Critical Implementation Trilogy research and Option B planning

---

## Phase 2: Analyst (Proactive Intelligence)

**Timeline**: Months 6-10 (after Phase 1 completion)
**Goal**: Autonomous correlation detection and proactive pattern recognition

### Core Architecture: AgentFlow 4-Module System

```
┌─────────────────────────────────────────────────────────────┐
│                   ANALYST AGENT                             │
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────┐│
│  │ PLANNER  │ →  │ EXECUTOR  │ →  │ VERIFIER │ →  │ GEN  ││
│  │          │    │           │    │          │    │      ││
│  │ Generate │    │ Query PKG │    │ Validate │    │ Synth││
│  │ hypothes │    │ for evid  │    │ correlat │    │ insig││
│  │ about    │    │ ence      │    │ ion      │    │ ht   ││
│  │ correlat │    │           │    │          │    │      ││
│  │ ions     │    │           │    │          │    │      ││
│  └────┬─────┘    └─────┬─────┘    └────┬─────┘    └──────┘│
│       │                │                 │                  │
│       └────────────────┴─────────────────┴──> Memory       │
│                                              (Evolving)     │
└─────────────────────────────────────────────────────────────┘
```

### Modules

#### 1. Correlation Planner
**Purpose**: Generate hypotheses about temporal correlations from PKG
**Input**: PKG schema, memory buffer, temporal data
**Output**: Correlation hypothesis + query strategy

**Key Capabilities**:
- Temporal pattern identification (A occurs before B consistently)
- Frequency analysis (how often do A and B co-occur?)
- Hypothesis generation from thought templates
- Query strategy design for evidence gathering

**Implementation** (from Critical Trilogy):
- Uses TOTAL thought templates for correlation patterns
- Trained via Flow-GRPO on successful correlation discoveries
- Templates evolve: "correlation detection" → "stronger correlation detection"

#### 2. PKG Executor
**Purpose**: Execute queries against PKG to gather evidence
**Input**: Hypothesis + query strategy
**Output**: Query results (entities, events, temporal relationships)

**Key Capabilities**:
- Natural language → Cypher query translation
- Temporal query execution (events in time range, causal chains)
- Result aggregation and structuring
- Caching for repeated patterns

#### 3. Correlation Verifier
**Purpose**: Validate if evidence confirms/refutes correlation
**Input**: Hypothesis + evidence + memory context
**Output**: CONFIRMED / REFUTED / INCONCLUSIVE / EXHAUSTED

**Key Capabilities**:
- Statistical significance checking
- Temporal consistency validation
- Confounder identification
- Confidence calibration

#### 4. Insight Generator
**Purpose**: Synthesize user-facing insights from confirmed correlations
**Input**: Confirmed correlation + evidence + memory
**Output**: User-facing insight with suggestions for Phase 3 exploration

**Key Capabilities**:
- Natural language insight generation
- Evidence summarization
- Causal exploration suggestions
- Actionable recommendations

### Evolving Memory

**Purpose**: Deterministic state tracking with bounded context growth

**Structure**:
```python
class MemoryBuffer:
    entries: List[str]  # Turn-by-turn memory
    max_entries: int = 50

    def compress_old_entries(self):
        """Summarize old entries to keep context bounded."""
```

### Training: Flow-GRPO

**Key Innovation** (from AgentFlow paper):
- Broadcasts final reward to ALL turns (multi-turn → single-turn RL)
- In-the-flow optimization (learns while running)
- Sparse reward handling (success/failure of correlation discovery)

**Training Data**:
- Labeled correlation cases (100-200)
- User feedback on discovered correlations
- Quality metrics from user interactions

### Phase 1 → Phase 2 Transition

**What Phase 1 Provides**:
- ✅ Temporal data (timestamps, temporal relationships)
- ✅ Thought templates (evolved for extraction, reusable for correlation)
- ✅ Experiential learning foundation (Training-Free GRPO patterns)
- ✅ PKG with temporal queries enabled

**What Phase 2 Adds**:
- AgentFlow 4-module system
- Flow-GRPO for planner training
- Correlation-specific thought templates
- Proactive notification system

**No Refactoring Needed** - Phase 1 built with Phase 2 in mind

---

## Phase 3: Guide (Sophisticated Reasoning)

**Timeline**: Months 11-15 (after Phase 2 completion)
**Goal**: Interactive causal discovery and hypothesis validation

### Core Architecture: AgentFlow for Causal Reasoning

```
┌─────────────────────────────────────────────────────────────┐
│                   GUIDE AGENT                               │
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────┐│
│  │ CAUSAL   │ →  │ CAUSAL    │ →  │ CAUSAL   │ →  │ GEN  ││
│  │ PLANNER  │    │ EXECUTOR  │    │ VALIDATOR│    │      ││
│  │          │    │           │    │          │    │      ││
│  │ Design   │    │ Execute   │    │ Validate │    │ Guide││
│  │ causal   │    │ causal    │    │ via      │    │ user ││
│  │ tests    │    │ queries/  │    │ Bradford │    │ throu││
│  │ (Bradford│    │ intervent │    │ Hill     │    │ gh   ││
│  │ Hill)    │    │ ions      │    │ criteria │    │ disco││
│  │          │    │           │    │          │    │ very ││
│  └────┬─────┘    └─────┬─────┘    └────┬─────┘    └──────┘│
│       │                │                 │                  │
│       └────────────────┴─────────────────┴──> Memory +     │
│                                              User Input    │
└─────────────────────────────────────────────────────────────┘
```

### Modules

#### 1. Causal Hypothesis Planner
**Purpose**: Design causal tests using Bradford Hill criteria
**Input**: Correlation (from Phase 2) + PKG state + user input
**Output**: Causal test plan (which criterion to validate next)

**Bradford Hill Criteria**:
1. **Temporality**: Does cause precede effect?
2. **Strength**: How strong is the association?
3. **Dose-Response**: More cause → more effect?
4. **Consistency**: Replicable across contexts?
5. **Plausibility**: Mechanistically sound?
6. **Coherence**: Fits existing knowledge?
7. **Experiment**: Can we test via intervention?
8. **Analogy**: Similar to known causal patterns?

**Key Capabilities**:
- Select next criterion to test
- Design PKG queries to gather causal evidence
- Propose counterfactual tests
- Suggest interventions (if possible)

#### 2. Causal Query Executor
**Purpose**: Execute causal queries and gather evidence
**Input**: Causal test plan + PKG state
**Output**: Causal evidence (temporal sequences, dose-response data, etc.)

**Key Capabilities**:
- Causal chain traversal (A → B → C)
- Temporal precedence validation
- Strength quantification
- Counterfactual reasoning support

#### 3. Causal Validator
**Purpose**: Validate causal claim using Bradford Hill criteria
**Input**: Causal test results + memory context
**Output**: CAUSAL / NOT_CAUSAL / UNCERTAIN + confidence + reasoning

**Key Capabilities**:
- Multi-criteria validation (check all 8 Bradford Hill criteria)
- Confidence aggregation across criteria
- Alternative explanation generation
- Causal strength quantification

#### 4. Causal Insight Generator
**Purpose**: Guide user through causal discovery process
**Input**: Validation results + exploration history
**Output**: Interactive guidance (next steps, explanations, conclusions)

**Key Capabilities**:
- User-friendly causal explanations
- Interactive hypothesis refinement
- Evidence visualization
- Actionable insights from causal understanding

### User Interaction

**Interactive Mode**:
```python
async def explore_causality(correlation: Insight, user_interaction: bool = True):
    """
    Multi-turn causal exploration with user guidance.

    Flow:
    1. Planner proposes causal test
    2. [USER INPUT] User approves/refines test
    3. Executor gathers evidence
    4. Validator assesses causality
    5. Generator explains findings
    6. [USER INPUT] User guides next step
    7. Repeat until CAUSAL/NOT_CAUSAL determined
    """
```

### Training: Dual GRPO

**Flow-GRPO** (AgentFlow):
- Train planner on successful causal discoveries
- Sparse reward: did we determine causality correctly?
- Multi-turn → single-turn via broadcasting

**Training-Free GRPO** (Critical Trilogy):
- Learn from user's causal reasoning
- Semantic advantages from user guidance
- No parameter updates (preserve Ghost)

### Phase 2 → Phase 3 Transition

**What Phase 2 Provides**:
- ✅ Correlation discovery (starting points for causal exploration)
- ✅ Temporal data (enables temporality criterion)
- ✅ AgentFlow architecture (reuse for causal reasoning)
- ✅ Thought templates (evolve for causal patterns)

**What Phase 3 Adds**:
- Bradford Hill criteria validation
- Causal chain reasoning
- Interactive dialogue system
- Counterfactual reasoning
- Intervention suggestion

**No Refactoring Needed** - Phase 2 built with Phase 3 in mind

---

## Cross-Phase Technical Stack

### Consistent Across All Phases

**LLM**: Llama-3.1 8B / Qwen3-8B (quantized, on-device)
**Learning**: Training-Free GRPO (no parameter updates)
**Templates**: TOTAL thought templates (evolving via textual gradients)
**Memory**: Evolving memory buffer (bounded context)
**Privacy**: Local-only processing, experiential knowledge stored locally

### Phase-Specific Additions

| Component | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| **Focus** | Extraction | Correlation | Causation |
| **Architecture** | Pipeline | AgentFlow (4-module) | AgentFlow (causal) |
| **Templates** | Extraction patterns | Correlation patterns | Causal patterns |
| **Queries** | Entity/relationship | Temporal patterns | Causal chains |
| **Output** | PKG triples | Correlation insights | Causal understanding |
| **User Role** | Passive (PKG builds) | Notified (insights) | Active (guided) |

---

## Implementation Strategy for Future Phases

### When to Create Phase 2 Plans

**Trigger**: Phase 1 production deployment complete
**Timeline**: Month 6 (after 5-month Phase 1)

**Markdown Folder Structure**:
```
docs/phase-2/
├── overview.md
├── feature-correlation-detection.md
├── feature-agentflow-architecture.md
├── feature-thought-template-evolution.md
├── feature-flow-grpo-training.md
├── feature-proactive-notifications.md
└── correlation-detection-production-plan/
    ├── README.md
    ├── 01-agentflow-modules.md
    ├── 02-flow-grpo-implementation.md
    ├── 03-correlation-templates.md
    ├── 04-notification-system.md
    └── 05-integration-testing.md
```

### When to Create Phase 3 Plans

**Trigger**: Phase 2 production deployment complete
**Timeline**: Month 11 (after 4-month Phase 2)

**Markdown Folder Structure**:
```
docs/phase-3/
├── overview.md
├── feature-causal-discovery.md
├── feature-bradford-hill-validation.md
├── feature-interactive-dialogue.md
├── feature-counterfactual-reasoning.md
├── feature-intervention-suggestion.md
└── causal-discovery-production-plan/
    ├── README.md
    ├── 01-causal-planner.md
    ├── 02-bradford-hill-validation.md
    ├── 03-interactive-dialogue.md
    ├── 04-counterfactual-reasoning.md
    └── 05-production-deployment.md
```

---

## Key Architectural Decisions Preserved

### 1. No Breaking Changes Between Phases
- Phase 1 builds foundations for Phase 2/3
- No refactoring required for phase transitions
- AgentFlow patterns established in Phase 1 extraction

### 2. Consistent Learning Paradigm
- Training-Free GRPO across all phases
- Thought templates evolve continuously
- Experiential knowledge accumulates (never discarded)

### 3. Privacy-First Throughout
- All learning local
- No parameter updates (Ghost remains frozen)
- Experiential knowledge stays on-device
- Cloud escalation optional with consent

### 4. Progressive Capability Growth
- Phase 1: Reactive (builds PKG)
- Phase 2: Proactive (detects patterns)
- Phase 3: Interactive (guides understanding)

---

## Success Metrics Preview

### Phase 2 Success Criteria
- ✅ Autonomous correlation detection running daily
- ✅ >50% user "valuable" rating on insights
- ✅ Correlation accuracy >70%
- ✅ Notification relevance >60%

### Phase 3 Success Criteria
- ✅ Causal discovery conversations in ≥3 validated scenarios
- ✅ Bradford Hill validation >80% accuracy
- ✅ User satisfaction with causal guidance >70%
- ✅ Production-ready causal exploration

---

**These notes ensure Phase 1 implementation builds the correct foundations for Phase 2 and Phase 3, avoiding technical debt and enabling smooth Ghost→Animal evolution across all three phases.**

**When Phase 1 is complete, use these notes to create detailed Phase 2 plans. When Phase 2 is complete, use these notes to create detailed Phase 3 plans.**

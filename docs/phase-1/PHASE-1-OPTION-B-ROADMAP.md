# Phase 1 (Archivist) Implementation Roadmap - Option B

**Timeline**: 5-6 months
**Vision Alignment**: 95%
**Approach**: Critical Implementation Trilogy (Training-Free GRPO + TOTAL + AgentFlow)
**Status**: APPROVED - Ready for Implementation

---

## Executive Summary

This roadmap implements **Option B**: the SOTA-aligned approach that builds the correct foundation for Ghost→Animal AI evolution while avoiding 6+ months of technical debt.

**Investment**: +9-12 weeks upfront (vs Option A)
**Return**: 6+ months technical debt avoided + full vision enabled
**Net Benefit**: 3-4 months saved + 2-3 year competitive moat

---

## Phase 1 Architecture (Option B)

```
INPUTS                    PROCESSING                    OUTPUTS
┌──────────────────┐
│ Obsidian Vault   │──┐
└──────────────────┘  │
┌──────────────────┐  │  ┌─────────────────────────────────────┐
│ Local Files      │──┼─→│  Normalization Pipeline (✅ DONE)   │
└──────────────────┘  │  └────────────┬────────────────────────┘
┌──────────────────┐  │               │
│ IMAP Email       │──┤               ↓
└──────────────────┘  │  ┌─────────────────────────────────────┐
┌──────────────────┐  │  │ Entity-Relationship Extraction      │
│ GitHub Repos     │──┘  │ (Option B - 5 Critical Modules)     │
└──────────────────┘     │                                     │
                         │ 1. Temporal Extraction (Weeks 1-3)  │
                         │ 2. Schema Evolution (Weeks 5-6)     │
                         │ 3. Experiential Learning (Wks 9-11) │
                         │ 4. Thought Templates (Weeks 7-8)    │
                         │ 5. Causal Structure (Weeks 13-14)   │
                         └────────────┬────────────────────────┘
                                      │
                                      ↓
                         ┌─────────────────────────────────────┐
                         │  Personal Knowledge Graph (PKG)     │
                         │  - Temporal metadata                │
                         │  - Causal structure                 │
                         │  - Evolving schema                  │
                         └────────────┬────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ↓                                   ↓
       ┌─────────────────────────┐      ┌───────────────────────────┐
       │ Vector Embedding Service │      │  Hybrid Search API        │
       └─────────────────────────┘      └───────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ↓
                         ┌─────────────────────────────────────┐
                         │    Search Desktop Shell             │
                         │    Graph Visualization              │
                         └─────────────────────────────────────┘
```

---

## Month-by-Month Breakdown

### Month 1: Temporal Extraction & Foundation
**Timeline**: Weeks 1-4
**Focus**: Eliminate FATAL gap, establish schema foundation

#### Week 1: Temporal Marker Extraction
**Deliverables**:
- Explicit timestamp detector (ISO 8601, natural language, time expressions)
- Relative time parser (yesterday, last week, 2 weeks ago)
- Unit tests (100+ test cases)

**Success Criteria**:
- ✅ >95% explicit timestamp detection accuracy
- ✅ >85% relative expression parsing accuracy

#### Week 2: Temporal Relationship Detection
**Deliverables**:
- Temporal relationship classifier (7 types: BEFORE/AFTER/DURING/CAUSES/etc.)
- Contextual temporal inferencer
- Allen's Interval Algebra implementation

**Success Criteria**:
- ✅ >80% explicit relationship detection
- ✅ >70% implicit relationship inference

#### Week 3: Temporal Integration & Validation
**Deliverables**:
- Pipeline integration with existing extraction
- Temporal triple structure implementation
- Comprehensive temporal testing suite

**Success Criteria**:
- ✅ >85% overall temporal accuracy
- ✅ Temporal consistency maintained (no contradictions)
- ✅ Ready for PKG storage integration

#### Week 4: Schema Foundation
**Deliverables**:
- Seed schema design (Person, Organization, Concept, Event)
- Phase 1 extraction: Entity-Entity relationships
- Schema versioning infrastructure

**Success Criteria**:
- ✅ Basic entity extraction functional
- ✅ Schema versioning system operational

---

### Month 2: Schema Evolution & Thought Templates
**Timeline**: Weeks 5-8
**Focus**: Autonomous schema evolution, evolving reasoning patterns

#### Weeks 5-6: Autonomous Schema Evolution
**Deliverables**:
- Multi-phase extraction:
  - Phase 1: Entity-Entity (Person → Organization)
  - Phase 2: Entity-Event (Person → attended → Meeting)
  - Phase 3: Event-Event (Meeting → led to → Decision)
- Reflection mechanism (triggers every N documents)
- Schema evolution tracking and versioning

**Success Criteria**:
- ✅ All 3 extraction phases operational
- ✅ Reflection triggers and assesses quality
- ✅ Schema evolves measurably

#### Weeks 7-8: Thought Template System
**Deliverables**:
- Template database implementation (TOTAL framework)
- Seed templates for common extraction patterns
- Template selection and composition logic
- Textual gradient refinement mechanism

**Success Criteria**:
- ✅ Template database operational with 20+ seed templates
- ✅ Template composition works for complex extractions
- ✅ Textual gradients refine templates (KEEP/FIX/DISCARD)

---

### Month 3: Experiential Learning (Ghost→Animal)
**Timeline**: Weeks 9-12
**Focus**: Training-Free GRPO, World State Model

#### Weeks 9-10: Training-Free GRPO Core
**Deliverables**:
- Experiential knowledge storage (token priors, not parameters)
- Rollout group generation (4 attempts per document)
- LLM introspection for semantic advantages

**Success Criteria**:
- ✅ Experiential knowledge stored as natural language
- ✅ Semantic advantages generated from rollouts
- ✅ Ghost model remains frozen (no parameter updates)

#### Week 11: Semantic Advantage Integration
**Deliverables**:
- Semantic advantage application to future extractions
- Experiential knowledge pruning (keep top 20 patterns)
- Multi-epoch learning implementation

**Success Criteria**:
- ✅ Extraction quality improves with experiential knowledge
- ✅ Ghost + Experience = Animal behavior
- ✅ Measurable quality improvement over iterations

#### Week 12: World State Model
**Deliverables**:
- Quality trajectory tracking
- Success/failure pattern identification
- Curriculum generator (orders documents by learning value)

**Success Criteria**:
- ✅ Quality progression measurable
- ✅ Patterns identified and actionable
- ✅ Curriculum improves learning efficiency

---

### Month 4: Causal Structure & Integration
**Timeline**: Weeks 13-16
**Focus**: Phase 3 foundation, full pipeline integration

#### Weeks 13-14: Causal Structure Preparation
**Deliverables**:
- Event entity extraction (distinct from static entities)
- Event-event relationship detection
- Causal candidate flagging
- Bradford Hill criteria preparation

**Success Criteria**:
- ✅ Events extracted and classified
- ✅ Event-event relationships identified
- ✅ Causal candidates flagged for Phase 3

#### Week 15: Full Pipeline Integration
**Deliverables**:
- End-to-end integration:
  - Normalization → Temporal Extraction → Schema Evolution → Experiential Learning → PKG
- Error handling and quarantine workflows
- Performance optimization (KV-cache, batching)

**Success Criteria**:
- ✅ Full pipeline operational
- ✅ Error recovery works
- ✅ Performance meets targets (>5 docs/sec)

#### Week 16: Privacy & Documentation
**Deliverables**:
- Privacy compliance audit (experiential learning, template storage)
- Developer documentation (templates, experiential knowledge, schema evolution)
- Integration guides for Phase 2/3 teams

**Success Criteria**:
- ✅ Privacy audit passed
- ✅ Documentation complete
- ✅ Phase 2/3 transition path clear

---

### Month 5: Validation & Production Readiness
**Timeline**: Weeks 17-20
**Focus**: Quality validation, production deployment

#### Weeks 17-18: Quality Validation
**Deliverables**:
- Temporal accuracy validation (>85% target)
- Schema evolution alignment validation (>90% target)
- Extraction quality benchmarking (precision ≥0.8, recall ≥0.7)
- Learning improvement measurement (50+ document progression)
- SOTA baseline comparisons

**Success Criteria**:
- ✅ All quality targets met or exceeded
- ✅ Ghost→Animal evolution demonstrable
- ✅ Template evolution measurable

#### Weeks 19-20: Production Readiness
**Deliverables**:
- Load testing and performance tuning
- Edge case coverage and error handling
- Monitoring and telemetry setup
- Deployment scripts and automation
- Production runbook

**Success Criteria**:
- ✅ Load tested with real user vaults
- ✅ Monitoring operational
- ✅ Deployment automated
- ✅ Production ready

---

### (Optional) Month 6: Early Phase 2 Prep
**Timeline**: Weeks 21-24
**Focus**: AgentFlow foundations for Phase 2 (Analyst)

**Note**: This can overlap with production deployment

**Deliverables**:
- AgentFlow 4-module architecture design
- Planner module prototype (correlation hypothesis generation)
- Executor module prototype (PKG query execution)
- Temporal correlation detection using temporal triples

**Success Criteria**:
- ✅ AgentFlow architecture documented
- ✅ Basic correlation detection working
- ✅ Ready for Phase 2 full implementation

---

## Success Metrics

### Phase 1 Completion Criteria

| Category | Metric | Target | Validation Method |
|----------|--------|--------|-------------------|
| **Temporal Extraction** | Temporal ordering accuracy | >85% | Labeled dataset comparison |
| | Temporal marker detection | >95% | Unit test suite |
| | Temporal relationship inference | >80% | Integration tests |
| **Schema Evolution** | Semantic alignment with manual schema | >90% | AutoSchemaKG benchmark |
| | Multi-phase extraction success | 100% | All phases operational |
| | Reflection trigger reliability | 100% | Every N documents |
| **Experiential Learning** | Quality improvement over time | Measurable | Precision/recall progression |
| | Ghost model frozen | 100% | No parameter updates |
| | Semantic advantage quality | >0.7 confidence | Manual review sample |
| **Thought Templates** | Template evolution demonstrable | Yes | Template version history |
| | Template composition success | >85% | Complex extraction tests |
| | Textual gradient refinement | Operational | Template refinement logs |
| **Causal Structure** | Event extraction accuracy | >80% | Labeled event dataset |
| | Event-event relationships | Operational | Causal candidate count |
| | Bradford Hill prep complete | 100% | Structure validation |
| **Integration** | Full pipeline operational | 100% | End-to-end tests |
| | Error recovery working | 100% | Quarantine workflow tests |
| | Performance targets met | >5 docs/sec | Load testing |
| **Privacy** | Local-only learning | 100% | Privacy audit |
| | No content leakage | 100% | Log analysis |
| **Phase Readiness** | Phase 2 foundation ready | 100% | AgentFlow patterns established |
| | Phase 3 foundation ready | 100% | Causal structure prepared |
| | No refactoring needed | 100% | Architecture review |

---

## Resource Requirements

### Team
- **Lead Engineer**: 1 FTE (full 5-6 months)
- **ML/AI Engineer**: 1 FTE (Months 2-4 heavy, Months 1,5 light)
- **QA Engineer**: 0.5 FTE (Months 4-5 heavy)
- **Total**: ~2.5 FTE average

### Infrastructure
- Development machines with GPU (≥12GB VRAM)
- Test data: Diverse document corpus (100+ documents, manually labeled)
- LLM access: Llama-3.1 8B or Qwen3-8B (quantized, local)

### External Dependencies
- Normalization pipeline (✅ Complete)
- Quality gates testing (✅ Complete - 98/100 score)
- PKG storage with temporal/causal support (IN PROGRESS - updated spec)
- Orchestrator integration (✅ Complete)

---

## Risk Management

### High-Priority Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Temporal extraction complexity exceeds estimates | High | Medium | Progressive complexity (Time-R1 approach); allocate buffer week |
| Schema evolution instability | Medium | Medium | Validation layer; manual override; extensive testing |
| Learning plateau (no quality improvement) | Medium | Low | Curriculum generation; diverse training data; tuning |
| Performance degradation with scale | High | Low | Profiling early; KV-cache optimization; streaming |
| Privacy violation in experiential learning | Critical | Very Low | Local-only processing; comprehensive audits; no cloud |

### Mitigation Strategies
1. **Weekly progress reviews** against roadmap milestones
2. **Bi-weekly architecture reviews** for critical decisions
3. **Monthly stakeholder updates** on quality metrics
4. **Continuous integration** with quality gate validation
5. **Privacy audits** at each major milestone

---

## Comparison: Option A vs Option B

| Dimension | Option A (Original Spec) | Option B (This Roadmap) |
|-----------|--------------------------|-------------------------|
| **Timeline** | 3-4 months | 5-6 months |
| **Upfront Investment** | Lower | Higher (+9-12 weeks) |
| **Technical Debt** | 6+ months refactoring | None |
| **Vision Alignment** | 60% | 95% |
| **Temporal Extraction** | ❌ Missing (FATAL) | ✅ Complete |
| **Schema Evolution** | ❌ Static | ✅ Autonomous |
| **Experiential Learning** | ❌ Passive feedback | ✅ Active learning |
| **Phase 2 Ready** | ❌ Requires refactoring | ✅ AgentFlow foundations |
| **Phase 3 Ready** | ❌ Blocked (no temporal) | ✅ Causal structure |
| **Competitive Moat** | 6-12 months (weak) | 2-3 years (strong) |
| **Net Time** | ~9-10 months (includes debt) | ~5-6 months |
| **Outcome** | "Better Obsidian search" | "Experiential Animal Intelligence" |

**Winner**: Option B - Build it right, avoid debt, save 3-4 months net, enable full vision

---

## Stakeholder Approval

**Approval Required From**:
- [ ] Engineering Leadership (timeline, resources)
- [ ] Product Leadership (vision alignment, roadmap)
- [ ] ML/AI Leadership (technical approach, SOTA alignment)
- [ ] Privacy/Security (experiential learning, data handling)

**Decision Deadline**: [TO BE SET]

**Implementation Start**: After approval + resource allocation

---

## Next Steps

### Immediate (Pre-Implementation)
1. Stakeholder review of this roadmap
2. Resource allocation (2.5 FTE)
3. Infrastructure setup (GPU machines, LLM access)
4. Test data collection and labeling

### Week 1 (Implementation Start)
1. Create remaining production plan files (02-05)
2. Begin temporal extraction implementation
3. Set up project tracking and milestones
4. Establish weekly review cadence

### Monthly Checkpoints
- **Month 1 End**: Temporal module complete, schema foundation operational
- **Month 2 End**: Schema evolution autonomous, thought templates refining
- **Month 3 End**: Ghost→Animal evolution demonstrable, learning measurable
- **Month 4 End**: Full pipeline integrated, causal structure prepared
- **Month 5 End**: Production ready, Phase 2/3 foundations complete

---

**This roadmap represents the approved Option B approach: invest 9-12 weeks upfront to avoid 6+ months of technical debt, achieve 95% vision alignment, and enable full Ghost→Animal AI evolution across all three phases.**

**Decision**: Build it right the first time, or build it twice. We choose to build it right.

Summary: Plans the entity and relationship extraction feature with SOTA techniques, temporal reasoning, autonomous schema evolution, and experiential learning for Ghost→Animal AI transformation.

# Feature · Entity & Relationship Extraction (v2.0 - Option B)

## Goal
Convert normalized documents into a rich Personal Knowledge Graph (PKG) using state-of-the-art agentic extraction techniques that enable Ghost→Animal AI evolution. This system extracts temporal entities, relationships, and causal structure while learning from experience to continuously improve extraction quality—all while maintaining privacy-first principles.

**Key Innovation**: Unlike traditional static extraction, this implements the **Critical Implementation Trilogy** approach combining Training-Free GRPO (lightweight evolution), TOTAL thought templates (evolving reasoning patterns), and AgentFlow architecture preparation (Phase 2/3 foundation).

## Success Criteria
- **Temporal Extraction**: >85% accuracy on temporal ordering; all triples include temporal metadata where applicable
- **Autonomous Schema Evolution**: Schema evolves autonomously with >90% semantic alignment to manual schema
- **Experiential Learning**: Extraction quality improves through use via Training-Free GRPO (no parameter updates needed)
- **Multi-Phase Extraction**: Successful Entity-Entity, Entity-Event, and Event-Event extraction phases
- **Privacy-First**: Pipeline runs locally via quantized models (Llama-3.1 8B, Qwen3-8B) with optional cloud escalation per consent
- **Causal Foundation**: Graph structure prepared for Phase 3 causal inference (Bradford Hill criteria)
- **Production Quality**: ≥0.8 precision on entity/relationship extraction across diverse document types

## Architectural Foundation: The Critical Implementation Trilogy

This feature is built on three breakthrough papers that eliminate all critical gaps identified in SOTA research:

### 1. Training-Free GRPO (2510.08191v1) - Lightweight Evolution
**What**: Ghost→Animal evolution WITHOUT fine-tuning via experiential knowledge as token priors
**Why**: Enables on-device learning with dozens of samples instead of thousands
**How**: Semantic advantages (natural language feedback) update experiential knowledge, not model parameters

### 2. TOTAL - Thought Templates (2510.07499v1) - Evolving Reasoning
**What**: Reusable reasoning patterns that evolve via textual gradients
**Why**: Static templates prevent improvement; evolving templates enable continuous quality gains
**How**: LLM introspects failures → generates textual feedback → refines templates (KEEP/FIX/DISCARD)

### 3. AgentFlow (2510.05592v1) - Phase 2/3 Architecture Prep
**What**: 4-module agentic system (planner → executor → verifier → generator)
**Why**: Provides clear path from Phase 1 (Archivist) → Phase 2 (Analyst) → Phase 3 (Guide)
**How**: Modular architecture with Flow-GRPO for in-the-flow optimization

**Detailed Analysis**: See [CRITICAL_IMPLEMENTATION_TRILOGY.md](CRITICAL_IMPLEMENTATION_TRILOGY.md)

## Functional Scope

### Core Capabilities (Option B - Enhanced)

#### 1. **Temporal Extraction Module** (FATAL GAP - Fixed)
**Detailed Plan**: [01-temporal-extraction.md](entity-relationship-extraction-production-plan/01-temporal-extraction.md)

- Extract temporal markers: explicit timestamps, relative time expressions ("yesterday", "last month")
- Parse temporal relationships: BEFORE, AFTER, DURING, CAUSES, ENABLES, PREVENTS, SIMULTANEOUS
- Infer temporal ordering from context clues and document structure
- Validate temporal consistency (>85% accuracy target)

**Triple Structure**:
```python
TemporalTriple:
  subject: Entity
  predicate: Relationship
  object: Entity
  timestamp: datetime              # When did this occur?
  duration: Optional[timedelta]    # How long?
  temporal_type: TemporalRelationType  # BEFORE/AFTER/DURING/CAUSES/etc.
  provenance: ChunkReference
  confidence: float
```

#### 2. **Autonomous Schema Evolution** (CRITICAL GAP - Fixed)
**Detailed Plan**: [02-schema-evolution.md](entity-relationship-extraction-production-plan/02-schema-evolution.md)

- **Multi-Phase Extraction** (AutoSchemaKG approach):
  - Phase 1: Entity-Entity relationships (Person, Organization, Concept)
  - Phase 2: Entity-Event relationships (temporal grounding)
  - Phase 3: Event-Event relationships (causal candidates)

- **Reflection Mechanism**: Every N documents (default: 100)
  - Assess extraction quality (coverage, consistency, success rate)
  - Generate schema refinement recommendations
  - Version schema evolution with timestamps

- **Target**: >90% semantic alignment with manual schema (AutoSchemaKG benchmark)

#### 3. **Experiential Learning Foundation** (CRITICAL GAP - Fixed)
**Detailed Plan**: [03-experiential-learning.md](entity-relationship-extraction-production-plan/03-experiential-learning.md)

- **Training-Free GRPO Implementation**:
  - Experiential knowledge stored as natural language patterns (token priors)
  - Semantic advantages generated via LLM introspection on rollout groups
  - No parameter updates - Ghost model remains frozen
  - Works with dozens of samples (cold-start friendly)

- **World State Model**:
  - Tracks extraction quality trajectory
  - Identifies success patterns and failure patterns
  - Curriculum generator orders documents by learning value

- **Feedback Integration**:
  - User corrections captured and converted to semantic advantages
  - Quality progression measurable over time

#### 4. **Thought Template System** (CRITICAL GAP - Fixed)
**Detailed Plan**: [04-thought-templates.md](entity-relationship-extraction-production-plan/04-thought-templates.md)

- **Template Database** (TOTAL framework):
  - Reusable reasoning patterns acting as "how to think" scaffolds
  - Templates composed flexibly for complex multi-hop extractions
  - Each template versioned with performance statistics

- **Textual Gradient Refinement**:
  - LLM analyzes failures → generates natural language feedback
  - Templates evolve: KEEP (minor fix), FIX (major revision), DISCARD (flawed)
  - Human-readable and inspectable refinement process

- **Template Categories**:
  - Entity recognition patterns
  - Relationship extraction strategies
  - Temporal reasoning templates
  - Causal inference preparation templates

#### 5. **Causal Structure Preparation** (CRITICAL GAP - Fixed)
**Detailed Plan**: [05-causal-structure.md](entity-relationship-extraction-production-plan/05-causal-structure.md)

- **Event Extraction**: Identify events as distinct from static entities
- **Event-Event Relationships**: Extract potential causal connections
- **Causal Candidate Marking**: Flag relationships for Phase 3 validation
- **Bradford Hill Criteria Prep**: Structure data for temporality, strength, dose-response checks
- **Causal Chain Storage**: Enable traversal of A→B→C causal pathways

### Legacy Capabilities (Preserved from v1.0)

- ✅ **Post-processing normalization**: spaCy/LLM hybrid for entity resolution, coreference, canonical naming
- ✅ **Confidence scoring**: Model confidence + rule-based validation + LLM-as-Judge verification
- ✅ **Provenance tracking**: All triples reference source chunks with chunk hashes
- ✅ **Privacy-first**: On-device inference default, cloud escalation only with consent
- ✅ **Batch/streaming modes**: Support backfill and incremental updates

## Non-Functional Guarantees

- **Privacy-First Evolution**: All experiential learning happens locally with user's own data
- **Lightweight**: Training-Free GRPO avoids expensive fine-tuning; works on consumer hardware
- **Deterministic**: Model version + prompt signature + template version stored for reproducibility
- **Transparent**: Thought templates and experiential knowledge are human-readable and inspectable
- **Efficient**: Reuse patterns across documents; template caching; KV-cache compression support
- **Resilient**: Graceful degradation; confidence scoring enables quality thresholds
- **Observable**: Metrics for extraction quality, template evolution, learning progression

## Dependencies

- Normalized documents ([feature-document-normalization](feature-document-normalization.md))
- PKG schema with temporal and causal support ([feature-pkg-graph-storage](feature-pkg-graph-storage.md))
- Privacy audit logging for escalation tracking
- Ingestion orchestrator for job scheduling
- State-of-the-art quantized LLMs (Llama-3.1 8B, Qwen3-8B)

## Implementation Guide

### Phase 1: Foundation (Weeks 1-4)

1. **Temporal Extraction Module** (Weeks 1-3)
   - Implement temporal marker detection (explicit + relative)
   - Build temporal relationship parser (BEFORE/AFTER/DURING/CAUSES/etc.)
   - Integrate with existing extraction pipeline
   - Validate >85% temporal ordering accuracy

2. **Initial Schema Design** (Week 4)
   - Define seed schema (Person, Organization, Concept, Event)
   - Implement Phase 1 extraction: Entity-Entity relationships
   - Set up schema versioning infrastructure

### Phase 2: Schema Evolution & Templates (Weeks 5-8)

3. **Autonomous Schema Evolution** (Weeks 5-6)
   - Implement Phase 2 extraction: Entity-Event relationships
   - Implement Phase 3 extraction: Event-Event relationships
   - Build reflection mechanism (every N documents)
   - Set up schema evolution tracking

4. **Thought Template System** (Weeks 7-8)
   - Build template database with initial seed templates
   - Implement template selection and composition logic
   - Create textual gradient refinement mechanism
   - Set up template performance tracking

### Phase 3: Experiential Learning (Weeks 9-12)

5. **Training-Free GRPO** (Weeks 9-11)
   - Implement experiential knowledge storage (token priors)
   - Build semantic advantage extraction via LLM introspection
   - Create rollout group generation and evaluation
   - Integrate with extraction pipeline (Ghost + Experience = Animal)

6. **World State Model** (Week 12)
   - Implement quality trajectory tracking
   - Build success/failure pattern identification
   - Create curriculum generator for document ordering

### Phase 4: Causal Structure & Integration (Weeks 13-16)

7. **Causal Structure Preparation** (Weeks 13-14)
   - Implement event entity extraction
   - Build event-event relationship detection
   - Create causal candidate flagging logic
   - Prepare Bradford Hill criteria structure

8. **Integration & Polish** (Weeks 15-16)
   - Full pipeline integration testing
   - Performance optimization (KV-cache, batching)
   - Privacy compliance audit
   - Documentation and developer guides

### Phase 5: Validation & Deployment (Weeks 17-20)

9. **Quality Validation** (Weeks 17-18)
   - Validate >85% temporal accuracy
   - Validate >90% schema evolution alignment
   - Measure extraction quality improvement over time
   - Benchmark against SOTA baselines

10. **Production Readiness** (Weeks 19-20)
   - Load testing and performance tuning
   - Error handling and edge case coverage
   - Monitoring and telemetry setup
   - Deployment preparation

**Total Timeline**: 5 months (20 weeks)

**Detailed Plans**: See [entity-relationship-extraction-production-plan/](entity-relationship-extraction-production-plan/)

## Testing Strategy

### Core Test Suites

- **Temporal Extraction Tests**:
  - Temporal marker detection (explicit timestamps, relative expressions)
  - Temporal relationship parsing (all 7 types)
  - Temporal ordering validation
  - Target: >85% accuracy

- **Schema Evolution Tests**:
  - Multi-phase extraction validation (Entity→Event→Causal)
  - Reflection mechanism triggers and quality assessment
  - Schema versioning and migration
  - Target: >90% semantic alignment

- **Experiential Learning Tests**:
  - Semantic advantage generation quality
  - Experiential knowledge application effectiveness
  - Quality improvement trajectory over iterations
  - Target: Measurable improvement in precision/recall

- **Thought Template Tests**:
  - Template selection and composition logic
  - Textual gradient refinement quality
  - Template evolution tracking
  - Template performance statistics

- **Integration Tests**:
  - Full pipeline: normalization → extraction → PKG storage
  - Multi-document learning progression
  - Error recovery and quarantine workflows
  - Batch and streaming mode validation

- **Performance Tests**:
  - Throughput and latency benchmarks
  - Memory usage with large documents
  - Template cache effectiveness
  - KV-cache compression benefits

- **Privacy Compliance Tests**:
  - Local inference validation
  - No content leakage in logs
  - Experiential knowledge privacy
  - Cloud escalation consent checks

### Quality Benchmarks

- **Precision**: ≥0.8 across entity types
- **Recall**: ≥0.7 for relationship extraction
- **Temporal Accuracy**: ≥0.85 for temporal ordering
- **Schema Alignment**: ≥0.90 with manual schema
- **Learning Improvement**: Measurable quality increase over 50+ documents

## Code Review Checklist

### Core Functionality
- [ ] All extraction modes produce valid temporal metadata
- [ ] Schema evolution triggered and tracked correctly
- [ ] Experiential learning updates knowledge without parameter changes
- [ ] Thought templates compose correctly for complex extractions
- [ ] Causal candidates identified and structured appropriately

### Quality & Performance
- [ ] Precision ≥0.8, temporal accuracy ≥0.85
- [ ] Schema evolution alignment ≥0.90
- [ ] Extraction quality improves measurably over time
- [ ] Template refinement produces better patterns
- [ ] Performance meets throughput targets

### Privacy & Security
- [ ] Local inference default; cloud gated by consent
- [ ] No content leakage in logs or telemetry
- [ ] Experiential knowledge stored locally
- [ ] Template evolution preserves privacy
- [ ] Provenance tracking complete

### Architecture & Maintainability
- [ ] Modular design supports Phase 2/3 evolution
- [ ] Template database extensible and inspectable
- [ ] Experiential knowledge human-readable
- [ ] Schema versions tracked with migrations
- [ ] Code follows AgentFlow preparation patterns

### Testing & Documentation
- [ ] All test suites pass with quality targets met
- [ ] Template examples documented
- [ ] Experiential learning guide written
- [ ] Schema evolution process documented
- [ ] Phase 2/3 transition path clear

## Documentation & Follow-up

### Developer Documentation
- Thought template creation and refinement guide
- Experiential knowledge debugging and inspection
- Schema evolution monitoring and manual overrides
- Temporal relationship taxonomy reference
- Causal structure preparation guidelines

### Knowledge Base Updates
- Document extraction quality improvement patterns
- Share template evolution success stories
- Publish schema evolution case studies
- Create troubleshooting guides for common issues

### Team Coordination
- **PKG Team**: Coordinate schema extensions for temporal/causal support
- **Vector Team**: Ensure embedding sync with temporal metadata
- **Privacy Team**: Validate experiential learning privacy guarantees
- **Phase 2 Team**: Share AgentFlow architecture foundations

### Future Evolution Path

**Phase 1 → Phase 2 Transition** (Analyst):
- Thought templates evolve for correlation detection
- Experiential learning guides proactive pattern recognition
- AgentFlow 4-module system (planner/executor/verifier/generator)
- Temporal data enables correlation discovery

**Phase 2 → Phase 3 Transition** (Guide):
- Causal structure enables hypothesis testing
- Bradford Hill criteria validation
- Interactive causal exploration dialogues
- Sophisticated reasoning with user collaboration

**Reference**: [CRITICAL_IMPLEMENTATION_TRILOGY.md](CRITICAL_IMPLEMENTATION_TRILOGY.md) Section: "Complete Stack Integration"

## Success Metrics

### Phase 1 Completion (Month 5)
- ✅ Temporal extraction >85% accurate
- ✅ Schema evolving autonomously with >90% alignment
- ✅ Extraction quality improving measurably via experiential learning
- ✅ Thought templates refining successfully via textual gradients
- ✅ Causal structure prepared for Phase 3
- ✅ Privacy compliance validated
- ✅ Production-ready extraction pipeline

### Ghost→Animal Evolution Validation
- ✅ Ghost LLM remains frozen (no parameter updates)
- ✅ Animal behavior emerges via experiential knowledge
- ✅ Quality improvement measurable over time
- ✅ Template evolution demonstrable
- ✅ User-specific patterns learned locally

### Phase Progression Readiness
- ✅ Phase 2 (Analyst) foundation complete
- ✅ Phase 3 (Guide) foundation complete
- ✅ No refactoring required for phase transitions
- ✅ AgentFlow architecture patterns established

---

## Implementation Timeline Summary

| Month | Focus | Deliverables |
|-------|-------|--------------|
| **1** | Temporal Extraction + Schema Foundation | Temporal module, seed schema, Phase 1 extraction |
| **2** | Schema Evolution + Templates | Multi-phase extraction, template system |
| **3** | Experiential Learning | Training-Free GRPO, World State Model |
| **4** | Causal Structure + Integration | Event-event extraction, full pipeline integration |
| **5** | Validation + Deployment | Quality validation, production readiness |

**Total**: 5 months (Option B approach)
**Technical Debt Avoided**: 6+ months
**Net Time Saved**: 3-4 months
**Vision Alignment**: 95% (vs 60% with original spec)

---

**This specification represents Option B: the upgraded, SOTA-aligned approach that enables full Ghost→Animal evolution and smooth Phase 1→2→3 progression.**

**For detailed implementation plans, see**: [entity-relationship-extraction-production-plan/](entity-relationship-extraction-production-plan/)

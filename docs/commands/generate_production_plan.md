# Generate Production Plan

Use this template to create production plans for Phase 1 features following the established pattern.

## Template Structure

Every production plan should have:
1. **README.md** - Overview, modules, timeline, success metrics
2. **01-XX.md through 0N-XX.md** - Detailed implementation steps

## Standard Modules Pattern

Based on feature complexity, typical modules include:

### For Pipeline/Service Features (5-6 modules)
1. **Schema/Architecture Design** - Data structures, component design
2. **Core Implementation** - Main service logic
3. **Integration** - Connect with upstream/downstream
4. **Testing** - Unit, integration, performance tests
5. **Production Readiness** - Deployment, monitoring, validation

### For Connector Features (10+ modules)
See: `local-files-production-plan/`, `obsidian-connector-production-plan/`

### For Infrastructure Features (10+ modules)
See: `orchestrator-production-plan/`, `normalization-production-plan/`

## README.md Template

```markdown
# [Feature Name] Production Plan

**Status**: Ready for Implementation
**Timeline**: X weeks/months
**Dependencies**: [List dependencies]

## Overview

[Brief description of feature and Option B alignment]

## Critical for Option B

[Why this matters for Option B - temporal, causal, learning, etc.]

## Implementation Modules

### [01 · Module Name](01-module-name.md)
**Timeline**: Week X
**Deliverables**:
- [Deliverable 1]
- [Deliverable 2]

[Repeat for each module]

## Success Metrics

- ✅ [Metric 1]
- ✅ [Metric 2]

## Dependencies

- [Dependency 1]
- [Dependency 2]

## Next Steps

1. [Step 1]
2. [Step 2]
```

## Implementation File Template (01-XX.md)

```markdown
Summary: [One-line summary of this module]

# 01 · [Module Name]

## Purpose
[Why this module exists and what problem it solves]

**Criticality**: [CRITICAL/HIGH/MEDIUM/LOW] - [Why]

## Scope
- [Scope item 1]
- [Scope item 2]
- [Scope item 3]

## Requirements Alignment
- **Option B Requirement**: "[Requirement]"
- **Feature Requirement**: "[Requirement]"
- **System Architecture**: "[Alignment]"

## Component Design

[Code examples with Python/Cypher/etc.]

```python
class ExampleComponent:
    """Example component."""
    pass
```

## Implementation Details

### Week X: [Deliverable Name]

**Deliverable**: [What gets built]

[Detailed implementation guidance]

### Week Y: [Next Deliverable]

[Details...]

## Testing Strategy

```python
class TestModule:
    def test_something(self):
        """Test description."""
        pass
```

## Success Metrics

- ✅ [Metric 1]
- ✅ [Metric 2]

## Dependencies

- [Dependency 1]
- [Dependency 2]

## Next Steps

After [module] complete:
1. [Next step 1]
2. [Next step 2]

**[Impact statement]**
```

## Quick Start: Generate New Production Plan

```bash
# 1. Create directory
mkdir -p docs/phase-1/[feature-name]-production-plan

# 2. Create README.md
cp docs/commands/production_plan_readme_template.md \\
   docs/phase-1/[feature-name]-production-plan/README.md

# 3. Create implementation files (01-05 typically)
for i in {1..5}; do
  cp docs/commands/production_plan_module_template.md \\
     docs/phase-1/[feature-name]-production-plan/0${i}-module-name.md
done

# 4. Customize each file with feature-specific content
```

## Examples to Reference

**Best Examples by Type:**

### Extraction/Processing Pipeline
- ✅ `entity-relationship-extraction-production-plan/` - Option B aligned, Critical Trilogy
- ✅ `normalization-production-plan/` - 12 modules, comprehensive

### Connectors
- ✅ `obsidian-connector-production-plan/` - Well-structured
- ✅ `local-files-production-plan/` - Good hardening examples

### Infrastructure
- ✅ `orchestrator-production-plan/` - 10 modules, resilience focus

### Storage/Service
- ✅ `pkg-graph-storage-production-plan/` - Temporal/causal support, Option B aligned

## Checklist for New Production Plan

Before creating production plan:
- [ ] Feature spec exists in `feature-[name].md`
- [ ] Option B alignment analyzed
- [ ] Dependencies identified
- [ ] Timeline estimated (weeks/months)
- [ ] Module breakdown determined (5-12 modules typical)

While creating:
- [ ] README.md with overview and modules
- [ ] Each module has clear purpose and scope
- [ ] Code examples provided where helpful
- [ ] Testing strategy included
- [ ] Success metrics defined
- [ ] Dependencies documented

After creating:
- [ ] Review against existing patterns
- [ ] Validate Option B alignment
- [ ] Cross-reference with roadmap
- [ ] Update overview.md if needed

## Pattern Guidelines

### Module Naming Convention
- `01-[core-component].md` - Main component/schema
- `02-[implementation].md` - Core implementation
- `03-[integration].md` - Integration with pipeline
- `04-[specific-feature].md` - Feature-specific (e.g., temporal-query-support)
- `05-[testing].md` - Integration testing
- `0N-[extras].md` - Additional modules as needed

### File Size Guidelines
- README.md: 100-200 lines
- Module files: 200-500 lines (comprehensive examples)
- Total production plan: 2,000-6,000 lines typical

### Code Example Guidelines
- Include Python code with type hints
- Include Cypher for graph queries if applicable
- Include test examples
- Keep examples runnable/realistic

---

**This template ensures consistency across all Phase 1 production plans and maintains Option B alignment.**

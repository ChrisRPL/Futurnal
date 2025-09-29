# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Futurnal is a privacy-first personal knowledge and causal insight engine currently in **Phase 1 (Archivist)**. It implements a sophisticated data ingestion and knowledge graph pipeline that processes personal data sources (starting with Obsidian vaults) into semantic knowledge graphs for personal introspection and causal discovery.

**Core Philosophy**: Move from "What did I know?" to "Why do I think this?" through proactive, AI-assisted analysis and synthesis of personal data.

**Key Innovation**: Unlike standard GraphRAG systems that excel at information retrieval, Futurnal implements a **proactive causal inference engine** that detects temporal correlations in personal data and guides users through structured hypothesis exploration to understand the "why" behind patterns in their lives.

## Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.11+ required)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test module
pytest tests/ingestion/obsidian/

# Run with performance benchmarks
pytest -m performance

# Run integration tests
pytest tests/integration/

# Run single test function
pytest tests/ingestion/obsidian/test_connector.py::test_scan_vault_basic
```

### CLI Usage
```bash
# Main CLI entry point
python -m futurnal.cli --help

# Obsidian vault management
futurnal sources obsidian vault add /path/to/vault
futurnal sources obsidian vault scan vault_name

# Health diagnostics
futurnal health check
```

## Architecture Overview

### Core Pipeline Flow
```
Obsidian Vault → MarkdownNormalizer → Unstructured.io → SemanticTripleExtractor → PKG Storage
```

### Key Components

**Ingestion Layer** (`src/futurnal/ingestion/`):
- `ObsidianVaultConnector`: Processes Markdown files with frontmatter and wikilinks
- `LocalFilesConnector`: Generic file system scanner with state management
- Privacy integration with consent checks and audit logging throughout

**Pipeline Layer** (`src/futurnal/pipeline/`):
- `MarkdownNormalizer`: Prepares Obsidian content for Unstructured.io processing
- `SemanticTripleExtractor`: Converts document metadata into graph relationships
- `NormalizationSink`: Pluggable storage backend abstraction

**Orchestrator** (`src/futurnal/orchestrator/`):
- `IngestionOrchestrator`: Async job scheduling with APScheduler
- `JobQueue`: Persistent SQLite-backed queue with retry mechanisms
- Quarantine system for failed processing with retry policies

**Privacy Framework** (`src/futurnal/privacy/`):
- `ConsentRegistry`: Explicit permission management per data source
- `AuditLogger`: Activity logging without content exposure
- `RedactionPolicy`: Path anonymization for logs and telemetry

### Technology Stack
- **Python 3.11+** with Typer for CLI
- **Unstructured.io** for document parsing
- **Neo4j (embedded)** for knowledge graph storage
- **ChromaDB** for vector embeddings
- **APScheduler** for async job orchestration
- **190+ ML/AI dependencies** including PyTorch, Transformers

## Development Patterns

### Testing Infrastructure
- **120 test functions** across 17 modules with comprehensive external service mocking
- All major dependencies (Neo4j, ChromaDB, Unstructured.io) are stubbed in `conftest.py`
- Integration tests validate full pipeline functionality
- Use `pytest -m performance` for benchmark tests

### Privacy-First Engineering
- **All data access requires explicit consent** via ConsentRegistry
- **Comprehensive audit logging** without exposing content
- **Local-first processing** with optional cloud escalation
- Path redaction in logs and telemetry

### Error Handling & Resilience
- Structured exception handling with quarantine workflows
- Failed files moved to quarantine with retry policies
- Resource cleanup with proper context managers
- Extensive logging with privacy-aware contexts

## Configuration & Workspace

### Workspace Structure
```
~/.futurnal/
├── config.yaml           # Main configuration
├── consent/              # User consent records
├── audit/               # Audit logs
└── quarantine/          # Failed processing files
```

### Key Configuration Files
- `pyproject.toml`: Build configuration and project metadata
- `requirements.txt`: All dependencies (heavy ML/AI stack)
- `conftest.py`: Comprehensive test fixtures and mocks

## Current Development Status

**Active Branch**: `feat/p1-archivist`

**Phase 1 Implementation Status**:
- ✅ Obsidian connector with markdown normalization (40/40 tests passing)
- ✅ Unstructured.io integration bridge
- ✅ Semantic triple extraction from metadata
- ✅ Orchestrator integration with job queue
- ✅ Privacy framework with consent/audit/redaction
- ✅ Full pipeline integration from vault → PKG storage

## Development Workflow & Commands

### Feature Development Process
The project follows a structured feature development process documented in `docs/commands/`:

1. **Feature Planning**: Use `docs/commands/plan_feature.md` template to create technical plans
2. **Code Review**: Use `docs/commands/code_review.md` checklist for thorough reviews
3. **Documentation**: Follow `docs/commands/write_docs.md` for feature documentation

### Documentation Structure
Follow the onboarding sequence in `DEVELOPMENT_GUIDE.md`:
1. **Product Vision** (`docs/product-vision.md`, `docs/key-difference.md`)
2. **Architecture** (`architecture/system-architecture.md`)
3. **Requirements** (`requirements/system-requirements.md`, `requirements/roadmap.md`)
4. **Phase Execution** (`docs/phase-1/`, `prompts/phase-*-*.md`)

## Important Development Notes

### Dependency Management
- Heavy ML stack (190+ dependencies) requires careful environment management
- External services extensively mocked in tests to avoid installation complexity
- Apple Silicon optimization targeted

### CLI Architecture
Main entry: `futurnal.cli:cli` with structured subcommands:
- `sources`: Data source management with Obsidian specialization
- `config`: Configuration and settings management
- `health`: System diagnostics and health checks

### Performance Considerations
- Designed for consumer hardware (≥12GB VRAM recommended)
- Non-blocking ingestion with queued async processing
- Resource limits appropriate for on-device processing

### Testing Best Practices
- Always run full test suite before commits: `pytest`
- Use integration tests for pipeline changes: `pytest tests/integration/`
- Performance-critical changes should include benchmark tests: `pytest -m performance`
- Mock external dependencies following patterns in `conftest.py`

### Code Quality Standards
- Follow existing patterns for privacy-aware logging
- Maintain comprehensive error handling with quarantine workflows
- Preserve API compatibility when extending components
- Use type hints extensively following existing codebase patterns

## Technical Innovation Context

### Beyond Standard GraphRAG
While Futurnal uses GraphRAG as its foundation, the core innovation is the **causal inference engine** that:

1. **Proactive Correlation Detection**: Analyzes temporal patterns in timestamped personal data
2. **LLM-Powered Hypothesis Generation**: Uses LLM strengths for pattern matching, not causal reasoning
3. **Guided Causal Exploration**: Interactive dialogue to investigate hypotheses with PKG evidence
4. **Structured Framework**: Moves beyond simple retrieval to understanding complex life systems

This transforms personal data from static storage into an active intelligence engine for self-discovery.

### Production Architecture Integration
The recent architecture integration (documented in `ARCHITECTURE_INTEGRATION_SUMMARY.md`) completed the full pipeline:
- ✅ Unstructured.io bridge preserving metadata
- ✅ Semantic triple generation for graph construction
- ✅ Production Obsidian connector with privacy controls
- ✅ Orchestrator integration with job scheduling

## Documentation Resources

- `FUTURNAL_CONCEPT.md`: Comprehensive product vision (34k+ words)
- `DEVELOPMENT_GUIDE.md`: Structured onboarding sequence
- `architecture/system-architecture.md`: Technical architecture blueprint
- `docs/phase-1/`: Phase-specific feature documentation
- `requirements/`: Functional requirements and roadmap
- `prompts/`: Ready-to-use development prompts for each phase
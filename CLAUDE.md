# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Futurnal** is a privacy-first AI companion designed for personal knowledge discovery and introspective analysis. Unlike productivity tools that merely organize information, Futurnal employs sophisticated causal inference engines to help users uncover hidden patterns and root causes within their own thinking and behavior.

### Vision
Transform personal digital footprints into dynamic, explorable knowledge graphs that move from simple information retrieval to deep causal understanding.

### Target Audience
- Developers and AI/ML researchers
- PhD students and academics
- Prolific knowledge workers (authors, analysts, strategists)
- Privacy-conscious individuals seeking self-improvement

## Architecture

### Core Design Principles
1. **Privacy-First**: Local-first architecture with user data sovereignty
2. **Hybrid Intelligence**: On-device processing with optional cloud escalation
3. **Dynamic Knowledge Representation**: Personal Knowledge Graph (PKG) that evolves over time
4. **Causal Reasoning**: Move beyond correlation to understand underlying causes

### Architectural Layers

#### 1. On-Device Foundation
- **LLM Serving**: Ollama / llama.cpp for local model deployment
- **Models**: Llama-3.1-8B, Mistral-7B (4-bit quantized)
- **Hardware Optimization**: Apple Silicon, NVIDIA GPUs with 12GB+ VRAM

#### 2. Data Processing Pipeline
- **Ingestion**: Unstructured.io for parsing 64+ file types
- **Entity Extraction**: LLM-powered entity and relationship extraction
- **Knowledge Storage**: Neo4j (embedded) for graph database
- **Vector Search**: ChromaDB / Weaviate for semantic similarity

#### 3. Personal Knowledge Graph (PKG)
- **Dynamic Construction**: Automatically updates as user data changes
- **Temporal Awareness**: All data includes timestamps for causal analysis
- **Multi-hop Reasoning**: Graph traversal for complex queries
- **Hybrid Search**: Combines vector similarity with structured graph queries

#### 4. Causal & Proactive Layer
- **Emergent Insights**: Proactive pattern detection and correlation analysis
- **Causal Inference**: Hypothesis generation and guided exploration
- **Conversational Interface**: Natural language interaction with knowledge graph
- **Goal-Oriented Analysis**: "Aspirational Self" feature for personal growth tracking

## Technology Stack

### Core Technologies
| Component | Technology | Purpose |
|-----------|------------|---------|
| On-device LLM | Ollama / llama.cpp | Local model inference |
| Models | Llama-3.1-8B, Mistral-7B | Quantized models for consumer hardware |
| Data Processing | Unstructured.io | Multi-format document parsing |
| Graph Database | Neo4j (embedded) | Knowledge graph storage and querying |
| Vector Database | ChromaDB / Weaviate | Semantic search capabilities |
| Orchestration | LangChain / LangGraph | Agent workflows and prompt management |
| Frontend | Electron / Tauri | Cross-platform desktop application |

### Development Environment
- **Language**: Python (backend), JavaScript/TypeScript (frontend)
- **Package Management**: npm, pip
- **Testing**: pytest, Jest
- **Build Tools**: Webpack, Vite
- **Containerization**: Docker (optional)

## Development Commands

### Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Start local LLM service
ollama serve

# Pull required models
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### Development Workflow
```bash
# Start development server
npm run dev

# Run tests
npm test
pytest

# Build application
npm run build

# Run linting
npm run lint
```

### Database Management
```bash
# Start Neo4j instance
docker run -p 7474:7474 -p 7687:7687 neo4j:latest

# Start ChromaDB
chroma run --path ./chroma_db

# Run database migrations
python manage.py migrate
```

## Development Patterns

### Privacy-First Design
- All sensitive data processing occurs locally
- Cloud escalation is opt-in and anonymized
- No raw user data leaves the device without explicit permission
- Federated learning for model improvement (future)

### Agent-Based Development
This project uses Claude Code's agent system for specialized tasks:

- **AI Engineer** (`ai-engineer`): LLM integrations, RAG systems, prompt engineering
- **Frontend Developer** (`frontend-developer`): UI components, React applications
- **Python Pro** (`python-pro`): Advanced Python features and optimization
- **Debugger** (`debugger`): Error analysis and system troubleshooting
- **Code Reviewer** (`code-reviewer`): Quality assurance and best practices

### MCP Server Configuration
The project is configured with MCP servers for enhanced capabilities:
- **context7**: Library documentation and code examples
- **memory**: Knowledge graph management
- **github**: Repository integration and management
- **browser-server**: Web automation capabilities
- **fetch**: External data retrieval

### Context Monitoring
The project includes a real-time context monitoring system (`.claude/scripts/context-monitor.py`) that tracks:
- Context usage and warnings
- Session metrics (cost, duration, lines changed)
- Directory context and model information

## Key Files and Directories

### Configuration
- `.claude/settings.local.json`: Claude Code configuration with hooks and MCP servers
- `.mcp.json`: MCP server definitions
- `.claude/agents/`: Specialized agent configurations
- `.claude/commands/`: Custom command definitions

### Documentation
- `FUTURNAL_CONCEPT.md`: Comprehensive product vision and technical specification
- `CLAUDE.md`: This file - development guidance for Claude Code

### Project Structure (Planned)
```
src/
├── backend/           # Python backend services
│   ├── ingestion/     # Data processing pipeline
│   ├── graph/         # Knowledge graph management
│   ├── inference/     # LLM inference layer
│   └── analysis/     # Causal inference engine
├── frontend/          # Electron/Tauri desktop app
│   ├── components/   # UI components
│   ├── services/     # API services
│   └── store/        # State management
├── models/           # Local model configurations
├── tests/            # Test suites
└── docs/             # Technical documentation
```

## Development Workflow

### 1. Knowledge Graph Development
- Focus on entity extraction quality
- Ensure temporal data preservation
- Implement efficient graph traversal algorithms
- Design scalable storage patterns

### 2. Causal Inference Implementation
- Start with correlation detection
- Implement hypothesis generation prompts
- Create guided exploration interfaces
- Add confounder analysis capabilities

### 3. Privacy Architecture
- Implement local-first data processing
- Design secure cloud escalation protocols
- Add federated learning capabilities
- Ensure compliance with privacy regulations

### 4. User Experience
- Design intuitive knowledge graph visualization
- Create conversational exploration interfaces
- Implement proactive insight delivery
- Add goal tracking and feedback systems

## Important Considerations

### Hardware Requirements
- Minimum: 8GB RAM, 4-core CPU
- Recommended: 16GB RAM, Apple Silicon or NVIDIA GPU with 12GB+ VRAM
- Storage: SSD with 50GB+ free space for knowledge graph

### Performance Optimization
- Model quantization for on-device inference
- Graph indexing strategies
- Caching mechanisms for frequent queries
- Background processing for graph updates

### Security
- Never hardcode API keys or credentials
- Implement proper authentication for cloud features
- Use encrypted storage for sensitive user data
- Regular security audits and dependency updates

This project represents a paradigm shift from passive information management to active personal intelligence, with privacy and user sovereignty as foundational principles.
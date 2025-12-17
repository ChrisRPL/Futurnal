# Futurnal

**Privacy-first personal knowledge engine for understanding your digital life.**

[![Phase](https://img.shields.io/badge/Phase-1%20Archivist-blue)](docs/phase-1/overview.md)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## What is Futurnal?

Futurnal transforms your scattered digital knowledge into a connected, searchable intelligence engine. Unlike simple search tools, Futurnal understands relationships between your data, tracks temporal patterns, and helps you discover the "why" behind your thoughts.

**Core Philosophy**: Move from "What did I know?" to "Why do I think this?"

### Key Features

- **Intelligent Search**: Semantic + graph-based retrieval with temporal awareness
- **Conversational AI**: Chat with your knowledge base (grounded, no hallucination)
- **Privacy-First**: All processing happens locally on your device
- **Multiple Sources**: Obsidian, Email, GitHub, Local Files
- **Knowledge Graph**: Visualize connections in your data
- **Experiential Learning**: Improves with use (without cloud training)

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** for local LLM inference:
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

3. **Pull a model**:
   ```bash
   ollama pull llama3.2:3b
   ```

### Installation

```bash
# Clone repository
git clone https://github.com/futurnal/futurnal.git
cd futurnal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install
pip install -e .

# Verify
futurnal health check
```

### Add Your First Data Source

```bash
# Add an Obsidian vault
futurnal sources obsidian vault add /path/to/your/vault --name my-notes

# Wait for ingestion
futurnal sources obsidian vault status my-notes
```

### Search Your Knowledge

```bash
# Search
futurnal search "machine learning"

# Temporal search
futurnal search "what I worked on last week"

# Start chat
futurnal chat
```

## Desktop Application

For a graphical interface, use the desktop app:

- **macOS**: Download `.dmg` from [Releases](https://github.com/futurnal/futurnal/releases)
- **Windows**: Download `.msi` from [Releases](https://github.com/futurnal/futurnal/releases)
- **Linux**: Download `.AppImage` from [Releases](https://github.com/futurnal/futurnal/releases)

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide/README.md) | Complete user documentation |
| [Installation](docs/user-guide/installation.md) | Detailed installation guide |
| [Quickstart](docs/user-guide/quickstart.md) | Get started in 5 minutes |
| [API Reference](docs/api-reference/README.md) | Developer documentation |
| [Privacy Policy](PRIVACY_POLICY.md) | How we handle your data |
| [Changelog](CHANGELOG.md) | Version history |

## Privacy

Futurnal is built on privacy-first principles:

- **Local-First**: All processing on your device
- **No Cloud Default**: Internet not required
- **Explicit Consent**: Permission required for each data source
- **Full Control**: Delete your data anytime

Read our [Privacy Policy](PRIVACY_POLICY.md) for details.

## Architecture

```
+------------------+     +------------------+     +---------------+
|   Data Sources   | --> |   Ingestion      | --> |  Knowledge    |
| (Obsidian, IMAP, |     |   Pipeline       |     |  Graph (PKG)  |
|  GitHub, Files)  |     +------------------+     +---------------+
+------------------+              |                      |
                                  v                      v
                          +------------------+    +-----------+
                          |   Embeddings     |    |  Search   |
                          |   (ChromaDB)     |    |   API     |
                          +------------------+    +-----------+
                                                       |
                                                       v
                                              +---------------+
                                              |   Chat/UI     |
                                              +---------------+
```

## Research Foundation

Futurnal is built on 39 SOTA research papers (2024-2025):

| Feature | Research |
|---------|----------|
| Search | GFM-RAG, Personalized Graph-Based Retrieval |
| Chat | ProPerSim, Causal-Copilot |
| Temporal | Time-R1 |
| Schema | AutoSchemaKG |
| Learning | SEAgent, Training-Free GRPO |
| Privacy | Federated Prompt Learning (ICLR 2025) |

See [SOTA Research Summary](docs/phase-1/SOTA_RESEARCH_SUMMARY.md) for details.

## Development

```bash
# Run tests
pytest

# Run specific tests
pytest tests/search/

# Run with coverage
pytest --cov=futurnal

# Type checking
mypy src/futurnal
```

See [CLAUDE.md](CLAUDE.md) for development guidelines.

## Roadmap

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Archivist | Released | Search, chat, knowledge graph |
| 2 | Analyst | Planned | Proactive insights, patterns |
| 3 | Guide | Planned | Causal inference, recommendations |

## Contributing

Contributions welcome! Please read:
1. [Development Guide](DEVELOPMENT_GUIDE.md)
2. [Code of Conduct](CODE_OF_CONDUCT.md)
3. Open an issue before large changes

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/futurnal/futurnal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/futurnal/futurnal/discussions)

---

**Futurnal**: Know yourself more.

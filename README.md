<p align="center">
  <img src="assets/logo_text_horizontal.png" alt="Futurnal logo" width="440">
</p>

<p align="center"><strong>Privacy-first personal knowledge engine for understanding your digital life.</strong></p>

<p align="center">
  <a href="https://github.com/ChrisRPL/Futurnal/actions/workflows/quality-gates.yml"><img src="https://img.shields.io/github/actions/workflow/status/ChrisRPL/Futurnal/quality-gates.yml?branch=main&label=quality%20gates" alt="Quality Gates"></a>
  <a href="https://github.com/ChrisRPL/Futurnal/actions/workflows/production-release.yml"><img src="https://img.shields.io/github/actions/workflow/status/ChrisRPL/Futurnal/production-release.yml?branch=main&label=release%20pipeline" alt="Release Pipeline"></a>
  <a href="https://github.com/ChrisRPL/Futurnal/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-0b0b0b.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/desktop-Tauri%202-24C8DB?logo=tauri&logoColor=white" alt="Tauri 2">
</p>

## Why Futurnal

Futurnal turns scattered personal data into a local, queryable intelligence layer.

- Hybrid retrieval: semantic + graph + temporal
- Local-first privacy defaults
- Chat grounded in your own data
- Desktop shell for search, graph, insights, and privacy controls
- Multi-source ingestion: Obsidian, IMAP email, GitHub, local files

Core shift: from "What did I know?" to "Why do I think this?"

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a local model
ollama pull llama3.2:3b
```

### 2. Install Futurnal CLI

```bash
git clone https://github.com/ChrisRPL/Futurnal.git
cd Futurnal

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -e .
futurnal health check
```

### 3. Add data and query

```bash
# Add a source
futurnal sources obsidian vault add /path/to/vault --name my-notes

# Ingestion status
futurnal sources obsidian vault status my-notes

# Query
futurnal search "what did I work on last week"

# Chat
futurnal chat
```

## Desktop App

Tauri desktop shell lives in `desktop/`.

```bash
cd desktop
npm install
npm run tauri dev
```

Release artifacts are built in GitHub Actions (`production-release.yml`).

## Architecture

```text
Sources -> Ingestion -> PKG + Embeddings -> Hybrid Search API -> Chat / UI
```

- System architecture: `docs/architecture/system-architecture.md`
- Causal reasoning boundary: `docs/architecture/causal-reasoning.md`
- Research notes: `docs/research/README.md`

## Documentation

- Docs index: `docs/README.md`
- User guide: `docs/user-guide/README.md`
- API reference: `docs/api-reference/README.md`
- Privacy: `PRIVACY.md`
- Changelog: `CHANGELOG.md`

## Research Assets Decision

Bundled paper PDFs and converted paper dumps were removed from this repository during OSS handoff to keep clone size lean and avoid shipping large third-party artifacts. Research foundations remain documented in `docs/research/` with references.

## Project Status

| Phase | Name | Status | Focus |
| --- | --- | --- | --- |
| 1 | Archivist | In progress | Search, chat, PKG, connectors |
| 2 | Analyst | Planned | Insights, community detection, pattern analysis |
| 3 | Guide | Planned | Causal exploration and recommendations |

## Contributing

- Read `CONTRIBUTING.md`
- Follow `CODE_OF_CONDUCT.md`
- Open issues for bugs and feature requests

## GitHub Pages

A branded project site is included under `site/` and deployable via GitHub Pages workflow:

- Source: `site/index.html`
- Workflow: `.github/workflows/pages.yml`

## License

MIT. See `LICENSE`.

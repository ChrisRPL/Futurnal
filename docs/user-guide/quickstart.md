# Quickstart Guide

Get started with Futurnal in 5 minutes.

## Prerequisites

- Futurnal installed ([Installation Guide](installation.md))
- Ollama running with a model available
- An Obsidian vault (or other data source)

## Step 1: Launch Futurnal

**Desktop App**: Open Futurnal from your Applications

**CLI**:
```bash
futurnal
```

## Step 2: Complete Onboarding

On first launch, you'll see the onboarding wizard:

1. **Welcome**: Overview of Futurnal's capabilities
2. **Privacy Consent**: Review and accept the privacy policy
3. **LLM Check**: Verify Ollama connection
4. **Ready**: Setup complete

## Step 3: Add Your First Data Source

### Adding an Obsidian Vault

**Desktop App**:
1. Click "Add Data Source" in the sidebar
2. Select "Obsidian Vault"
3. Browse to your vault folder
4. Click "Add Vault"

**CLI**:
```bash
# Add vault
futurnal sources obsidian vault add /path/to/your/vault

# Verify it was added
futurnal sources obsidian vault list
```

## Step 4: Wait for Ingestion

Futurnal will scan and process your vault:

1. **Scanning**: Finding all markdown files
2. **Processing**: Extracting content and metadata
3. **Indexing**: Building the knowledge graph
4. **Complete**: Ready for search

You can monitor progress in the Activity panel (desktop) or:
```bash
futurnal sources obsidian vault status my-vault
```

**Typical times**:
- 100 notes: ~2 minutes
- 1,000 notes: ~10 minutes
- 10,000 notes: ~1 hour

## Step 5: Search Your Knowledge

**Desktop App**: Use the search bar at the top

**CLI**:
```bash
futurnal search "your query here"
```

### Example Queries

**Simple search**:
```
machine learning
```

**Temporal query**:
```
What was I researching last week?
```

**Relationship query**:
```
How are Python and data science connected in my notes?
```

## Step 6: Chat with Your Knowledge

**Desktop App**: Click the "Chat" tab

**CLI**:
```bash
futurnal chat
```

### Example Conversation

```
You: What are the main topics I write about?

Futurnal: Based on your knowledge graph, your main topics are:
1. Machine Learning [Source: ml-fundamentals.md]
2. Python Programming [Source: python-tips.md]
3. Personal Productivity [Source: gtd-workflow.md]

These topics are interconnected through your project notes.

You: Tell me more about my Python notes

Futurnal: Your Python notes focus on:
- Data analysis with Pandas [Source: pandas-tutorial.md]
- Web development with FastAPI [Source: fastapi-guide.md]
- Machine learning with scikit-learn [Source: sklearn-intro.md]

Your most recent Python note was about async/await patterns
created on December 10th [Source: async-python.md].
```

## Step 7: Explore the Graph

**Desktop App**: Click the "Graph" tab to visualize connections

The graph shows:
- **Nodes**: Entities from your knowledge (people, projects, concepts)
- **Edges**: Relationships between entities
- **Clusters**: Groups of related knowledge

**Interactions**:
- Click a node to see details
- Drag to rearrange
- Scroll to zoom
- Double-click to focus

## What's Next?

### Add More Data Sources

- [Email (IMAP)](data-sources.md#email-imap)
- [GitHub Repositories](data-sources.md#github)

### Configure Privacy

- [Privacy Settings](privacy-settings.md)

### Advanced Search

- [Search & Chat Guide](search-chat.md)

## Quick Reference

### CLI Commands

| Command | Description |
|---------|-------------|
| `futurnal health check` | Verify system health |
| `futurnal sources obsidian vault add <path>` | Add Obsidian vault |
| `futurnal sources obsidian vault list` | List vaults |
| `futurnal search "<query>"` | Search knowledge |
| `futurnal chat` | Start chat session |

### Keyboard Shortcuts (Desktop)

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + K` | Focus search |
| `Cmd/Ctrl + Enter` | Send chat message |
| `Cmd/Ctrl + N` | New chat session |
| `Cmd/Ctrl + ,` | Open settings |
| `Escape` | Clear search |

## Troubleshooting

### "No results found"
- Ensure ingestion is complete
- Check vault status: `futurnal sources obsidian vault status`
- Try a simpler query first

### "Cannot connect to Ollama"
```bash
# Ensure Ollama is running
ollama serve

# Check connection
curl http://localhost:11434/api/tags
```

### "Ingestion stuck"
```bash
# Check for quarantined files
futurnal health check

# View quarantine
futurnal sources obsidian vault quarantine list
```

---

Congratulations! You're now ready to explore your personal knowledge with Futurnal.

Next: [Data Sources Guide](data-sources.md)

# Search & Chat Guide

Master Futurnal's intelligent search and conversational AI.

## Search Overview

Futurnal uses hybrid search combining:
- **Semantic Search**: Understanding meaning, not just keywords
- **Graph Search**: Following relationships in your knowledge
- **Temporal Search**: Understanding time-based queries

## Basic Search

### Simple Queries

```
machine learning
python tips
project ideas
```

### Phrase Search

Use quotes for exact phrases:
```
"machine learning fundamentals"
"meeting notes"
```

### Boolean Operators

Combine terms:
```
python AND pandas
machine learning OR deep learning
python NOT javascript
```

## Intelligent Query Types

Futurnal automatically detects query intent and routes to the appropriate search mode.

### Temporal Queries

Questions about time trigger temporal search:

```
What was I working on last week?
Show me notes from December
What did I write about in Q4?
Recent Python projects
```

**Supported time expressions**:
- Relative: "yesterday", "last week", "3 days ago"
- Absolute: "December 2024", "2024-12-01"
- Ranges: "between October and December"

### Relationship Queries

Questions about connections trigger graph search:

```
How is X related to Y?
What connects Python and my project?
Show me everything linked to machine learning
```

### Exploratory Queries

Open-ended questions explore your knowledge:

```
What are my main research topics?
Summarize my notes about productivity
What patterns exist in my project notes?
```

## Advanced Search

### Filters

Narrow results by source or type:

**By source**:
```
python site:vault
meeting source:email
bug source:github
```

**By date**:
```
python after:2024-12-01
meeting before:2024-12-15
notes date:2024-12-10
```

**By type**:
```
python type:code
research type:note
discussion type:email
```

### CLI Search

```bash
# Basic search
futurnal search "machine learning"

# With filters
futurnal search "python" --source vault --after 2024-12-01

# Limit results
futurnal search "project" --limit 10

# Output format
futurnal search "notes" --format json
```

## Search Results

### Result Components

Each result includes:
- **Title**: Document or entity name
- **Snippet**: Relevant excerpt with highlights
- **Score**: Relevance score (0.0 - 1.0)
- **Source**: Where this came from
- **Timestamp**: When it was created/modified
- **Relationships**: Connected entities

### Understanding Scores

| Score | Interpretation |
|-------|----------------|
| 0.9+ | Highly relevant, strong match |
| 0.7-0.9 | Good relevance |
| 0.5-0.7 | Moderate relevance |
| < 0.5 | Tangentially related |

---

## Chat Interface

### Starting a Conversation

**Desktop App**: Click "Chat" tab or press `Cmd/Ctrl + Shift + C`

**CLI**:
```bash
futurnal chat
```

### How Chat Works

1. You ask a question
2. Futurnal searches your knowledge for relevant context
3. The LLM generates a response grounded in your data
4. Sources are cited inline

**Critical**: Futurnal ONLY uses your data. It will not make up information.

### Example Conversation

```
You: What are my main Python projects?

Futurnal: Based on your knowledge graph, your main Python projects are:

1. **Data Pipeline** - A data processing pipeline for ETL tasks
   [Source: data-pipeline/README.md]

2. **API Server** - FastAPI-based REST API server
   [Source: api-server/overview.md]

3. **ML Experiments** - Machine learning experiments with scikit-learn
   [Source: ml-experiments/notes.md]

The Data Pipeline project appears most active, with 15 related notes
created in the last month.

You: Tell me more about the Data Pipeline

Futurnal: Your Data Pipeline project focuses on:

**Architecture** [Source: data-pipeline/architecture.md]:
- Apache Airflow for orchestration
- PostgreSQL for storage
- Redis for caching

**Recent Work** [Source: data-pipeline/changelog.md]:
- Added new data source integration (Dec 15)
- Fixed memory leak in transformer (Dec 10)
- Improved logging (Dec 5)

**Related Concepts**:
- Connected to your ETL notes
- Links to your Database Design documentation
```

### Multi-Turn Context

Chat maintains context across turns:

```
You: What did I write about Python?
Futurnal: You have 47 notes about Python, covering...

You: What about the web framework ones?
Futurnal: Within your Python notes, 12 relate to web frameworks...

You: Which one mentions authentication?
Futurnal: The FastAPI notes mention authentication. Specifically...
```

### When Chat Can't Answer

If your knowledge doesn't contain the answer:

```
You: What is the weather today?

Futurnal: I couldn't find information about weather in your knowledge.
My responses are based only on your personal data (notes, emails,
repositories). For current weather, please check a weather service.

Would you like me to search for weather-related notes you may have
written instead?
```

### Chat Commands

Within chat, use these commands:

| Command | Action |
|---------|--------|
| `/clear` | Clear conversation history |
| `/new` | Start new session |
| `/sources` | Show sources used in last response |
| `/export` | Export conversation |
| `/help` | Show available commands |

### Session Management

Conversations are saved as sessions:

```bash
# List sessions
futurnal chat sessions list

# Resume session
futurnal chat --session <session-id>

# Export session
futurnal chat export <session-id> --format markdown
```

---

## Tips for Better Results

### Search Tips

1. **Be specific**: "Python pandas groupby" > "Python data"
2. **Use your terminology**: Futurnal learns your vocabulary
3. **Try different angles**: Rephrase if results aren't right
4. **Leverage time**: Temporal queries narrow effectively

### Chat Tips

1. **Ask follow-ups**: The context helps refine answers
2. **Request sources**: Ask "What notes say this?"
3. **Be conversational**: Natural language works best
4. **Verify important info**: Check the cited sources

### Understanding Limitations

- **Only your data**: No external knowledge
- **Knowledge cutoff**: Only synced content is available
- **Relationship depth**: Some connections take time to build
- **LLM variability**: Responses may vary slightly

---

## Performance

### Search Performance Targets

| Operation | Target |
|-----------|--------|
| First search | < 1s |
| Cached search | < 200ms |
| Graph expansion | < 500ms |

### Chat Performance Targets

| Operation | Target |
|-----------|--------|
| Context retrieval | < 1s |
| Full response | < 3s |
| Streaming first token | < 500ms |

### Optimizing Performance

1. **Enable caching**: Settings > Performance > Cache
2. **Limit scope**: Use filters to narrow search
3. **Use better model**: Faster models for quick queries

---

## Keyboard Shortcuts

### Search

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + K` | Focus search |
| `Enter` | Execute search |
| `Escape` | Clear/close |
| `Up/Down` | Navigate results |
| `Enter` (on result) | Open result |

### Chat

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + Enter` | Send message |
| `Shift + Enter` | New line |
| `Cmd/Ctrl + N` | New session |
| `Cmd/Ctrl + L` | Clear chat |
| `Up` (empty input) | Edit last message |

---

## Troubleshooting

### "No results found"
1. Check if ingestion is complete
2. Try simpler keywords
3. Verify the source contains the content
4. Check spelling

### "Response doesn't cite sources"
1. The LLM may summarize across sources
2. Use `/sources` command to see all
3. Rephrase to be more specific

### "Chat is slow"
1. Check Ollama status: `ollama ps`
2. Try a smaller model
3. Reduce context window in settings
4. Clear old sessions

### "Irrelevant results"
1. Use more specific terms
2. Apply filters (source, date, type)
3. Wait for full ingestion
4. Report via feedback

---

Next: [Privacy Settings](privacy-settings.md)

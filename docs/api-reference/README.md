# API Reference

Futurnal's programmatic interfaces for developers.

## Overview

Futurnal provides APIs at multiple levels:

| API | Description | Use Case |
|-----|-------------|----------|
| [Search API](search-api.md) | HybridSearchAPI for intelligent search | Building search integrations |
| [Chat API](chat-api.md) | ChatService for conversational AI | Custom chat interfaces |
| [CLI Reference](cli-reference.md) | Command-line interface | Scripting and automation |

## Quick Start

### Python SDK

```python
from futurnal.search.api import HybridSearchAPI, create_hybrid_search_api
from futurnal.chat.service import ChatService

# Create search API
search_api = await create_hybrid_search_api()

# Execute search
results = await search_api.search("machine learning", top_k=10)
for result in results:
    print(f"{result.id}: {result.content[:100]}...")

# Create chat service
chat = ChatService(search_api=search_api)
await chat.initialize()

# Chat with knowledge
response = await chat.chat("session-1", "What are my main topics?")
print(response.content)

# Cleanup
await chat.close()
await search_api.close()
```

### CLI

```bash
# Search
futurnal search "machine learning" --limit 10 --format json

# Chat
futurnal chat --session "my-session"

# Sources
futurnal sources list

# Health
futurnal health check
```

## Architecture

```
+-------------------+     +------------------+     +---------------+
|   User/Client     | --> |   HybridSearch   | --> |   PKG Store   |
|                   |     |       API        |     |   (Neo4j)     |
+-------------------+     +------------------+     +---------------+
                               |
                               v
                          +----------+
                          | Embeddings|
                          | (ChromaDB)|
                          +----------+
```

## Option B Compliance

All APIs follow Option B principles:

1. **Ghost Model Frozen**: Ollama inference only, no fine-tuning
2. **Local-First**: All processing on localhost:11434
3. **Consent Required**: Privacy checks before data access
4. **Audit Logged**: All operations logged (metadata only)

## Authentication

No authentication required for local APIs. All APIs run locally.

For programmatic access from external tools:
```python
# Set in environment
export FUTURNAL_API_TOKEN="your-local-token"
```

## Error Handling

All APIs raise specific exceptions:

```python
from futurnal.errors import (
    SearchError,
    ConsentRequiredError,
    SourceNotFoundError,
    OllamaConnectionError,
)

try:
    results = await search_api.search(query)
except ConsentRequiredError as e:
    print(f"Consent needed for: {e.source}")
except OllamaConnectionError:
    print("Cannot connect to Ollama")
except SearchError as e:
    print(f"Search failed: {e}")
```

## Rate Limits

Local processing has no external rate limits. Performance depends on:
- Ollama model inference speed
- PKG database size
- Available system resources

Recommended limits:
- Max concurrent searches: 10
- Max context for chat: 10,000 tokens
- Batch ingestion: 100 docs/minute

## API Sections

- [Search API](search-api.md) - Full search capabilities
- [Chat API](chat-api.md) - Conversational interface
- [CLI Reference](cli-reference.md) - Command-line tools

## Support

- GitHub Issues: Bug reports and feature requests
- Documentation: This guide
- Examples: `examples/` directory in repository

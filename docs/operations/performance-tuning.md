# Performance Tuning Guide

Optimize Futurnal for your hardware and usage patterns.

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| First search | < 1s | 200-800ms |
| Cached search | < 200ms | 50-150ms |
| Chat response | < 3s | 1-2s |
| Ingestion | > 5 docs/sec | 5-20 docs/sec |
| Memory | < 2GB | 500MB-1.5GB |
| Cold start | < 10s | 3-8s |

## Quick Wins

### 1. Enable Caching

```yaml
# ~/.futurnal/config.yaml
performance:
  cache_enabled: true
  cache_ttl_seconds: 300
  cache_max_entries: 1000
```

### 2. Use Appropriate Model

```bash
# For speed (recommended for most)
ollama pull llama3.2:3b

# For quality (needs more VRAM)
ollama pull llama3.1:8b

# Configure
futurnal config set llm.model llama3.2:3b
```

### 3. Limit Results

```bash
# Search with limited results
futurnal search "query" --limit 10

# Limit chat context
futurnal config set chat.context_window_size 10
```

## Hardware Recommendations

### Minimum (Functional)

- CPU: 4 cores
- RAM: 8GB
- Storage: SSD with 10GB free
- VRAM: 4GB (for 3B models)

### Recommended (Good Performance)

- CPU: 8 cores
- RAM: 16GB
- Storage: NVMe SSD with 20GB free
- VRAM: 8GB (for 8B models)

### Optimal (Best Performance)

- CPU: 12+ cores
- RAM: 32GB
- Storage: NVMe SSD with 50GB free
- VRAM: 12GB+ (for larger models)

## Configuration Tuning

### Memory Management

```yaml
performance:
  max_memory_gb: 2          # Hard limit
  batch_size: 10            # Documents per batch
  embedding_batch_size: 32  # Embeddings per batch
  gc_interval_docs: 100     # Force GC every N docs
```

### Search Optimization

```yaml
search:
  vector_weight: 0.6        # Balance vector/graph
  graph_weight: 0.4
  top_k_vector: 20          # Vector candidates
  top_k_graph: 10           # Graph candidates
  timeout_ms: 5000          # Search timeout
```

### Chat Optimization

```yaml
chat:
  context_window_size: 10   # Message history
  max_context_tokens: 4000  # Context sent to LLM
  stream_enabled: true      # Faster perceived response
```

### Ingestion Optimization

```yaml
ingestion:
  batch_size: 20            # Docs per batch
  concurrent_workers: 4     # Parallel processing
  embedding_batch_size: 32  # Embeddings per batch
```

## Model Selection

### Speed vs Quality Trade-offs

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| llama3.2:1b | Fastest | Good | 2GB |
| llama3.2:3b | Fast | Better | 4GB |
| llama3.1:8b | Medium | Best | 8GB |
| qwen2.5:7b | Medium | Best | 8GB |

### Switching Models

```bash
# Pull new model
ollama pull llama3.1:8b

# Configure Futurnal
futurnal config set llm.model llama3.1:8b

# Verify
futurnal health check
```

## Caching Strategy

### Multi-Layer Cache

Futurnal uses a multi-layer cache:

1. **L1 (Memory)**: Hot queries, 100 entries, 50ms
2. **L2 (Disk)**: Recent queries, 1000 entries, 200ms
3. **L3 (Embedding)**: Embedding cache, persistent

### Cache Configuration

```yaml
performance:
  cache:
    l1_size: 100
    l1_ttl_seconds: 60
    l2_size: 1000
    l2_ttl_seconds: 300
    embedding_cache_enabled: true
```

### Cache Warmup

```bash
# Warmup common queries
futurnal health warmup --queries "recent,important,project"
```

## Monitoring Performance

### Built-in Metrics

```bash
# View performance stats
futurnal health stats

# Output:
# Search latency (p50): 250ms
# Search latency (p95): 800ms
# Cache hit rate: 75%
# Memory usage: 1.2GB
# Embedding throughput: 15/sec
```

### Benchmarking

```bash
# Run performance tests
pytest tests/performance/ -v -m performance

# Run specific benchmark
pytest tests/performance/test_search_latency.py -v
```

## Troubleshooting Performance

### Slow Search

1. **Check cache**: `futurnal health stats` - low hit rate?
2. **Check model**: Using a model too large for your hardware?
3. **Check index size**: Very large knowledge graph?
4. **Check concurrent load**: Other processes using resources?

**Solutions**:
```bash
# Enable caching
futurnal config set performance.cache_enabled true

# Use smaller model
futurnal config set llm.model llama3.2:3b

# Reduce result count
futurnal search "query" --limit 5
```

### Slow Chat

1. **Check context size**: Too much conversation history?
2. **Check model**: Model too large?
3. **Check Ollama**: Is Ollama overloaded?

**Solutions**:
```bash
# Reduce context
futurnal config set chat.context_window_size 5

# Clear session
futurnal chat --session new

# Restart Ollama
pkill ollama && ollama serve
```

### Slow Ingestion

1. **Check batch size**: Too large or too small?
2. **Check disk I/O**: SSD or HDD?
3. **Check memory**: Swapping to disk?

**Solutions**:
```bash
# Adjust batch size
futurnal config set ingestion.batch_size 10

# Reduce concurrent workers
futurnal config set ingestion.concurrent_workers 2

# Process in smaller chunks
# (Split large vaults into multiple sources)
```

### High Memory Usage

1. **Check knowledge graph size**: Very large PKG?
2. **Check for leaks**: Memory growing over time?
3. **Check embedding cache**: Too large?

**Solutions**:
```bash
# Lower memory limit
futurnal config set performance.max_memory_gb 1.5

# Clear caches
futurnal health cache clear

# Force garbage collection
futurnal health gc
```

## Advanced Tuning

### Ollama Optimization

```bash
# Increase context size (slower but more context)
OLLAMA_NUM_CTX=4096 ollama serve

# GPU memory management
OLLAMA_MAX_LOADED_MODELS=1 ollama serve

# Parallel requests
OLLAMA_MAX_PARALLEL=2 ollama serve
```

### Database Tuning

For very large knowledge graphs:

```yaml
pkg:
  cache_size_mb: 256
  index_cache_mb: 128
  query_timeout_ms: 10000
```

### Network Tuning

If using multiple machines:

```yaml
network:
  connection_pool_size: 10
  request_timeout_ms: 30000
  keepalive_ms: 60000
```

## Performance Profiles

### Profile: Low Memory (8GB RAM)

```yaml
performance:
  max_memory_gb: 1.5
  batch_size: 5
  cache:
    l1_size: 50
    l2_size: 500
llm:
  model: llama3.2:1b
```

### Profile: Balanced (16GB RAM)

```yaml
performance:
  max_memory_gb: 2
  batch_size: 10
  cache:
    l1_size: 100
    l2_size: 1000
llm:
  model: llama3.2:3b
```

### Profile: High Performance (32GB RAM)

```yaml
performance:
  max_memory_gb: 4
  batch_size: 20
  cache:
    l1_size: 200
    l2_size: 2000
llm:
  model: llama3.1:8b
```

## Benchmarking Commands

```bash
# Full benchmark suite
pytest tests/performance/ -v --benchmark-json=benchmark.json

# Search benchmark
pytest tests/performance/test_search_latency.py -v

# Memory benchmark
pytest tests/performance/test_memory_usage.py -v

# Cold start benchmark
pytest tests/performance/test_cold_start.py -v
```

## Related Documentation

- [Recovery Procedures](recovery-procedures.md)
- [Monitoring Guide](monitoring-guide.md)
- [User Guide](../user-guide/README.md)

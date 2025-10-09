Summary: Optimize normalization pipeline for throughput, memory usage, and offline operation.

# 11 · Performance Optimization

## Purpose
Optimize normalization pipeline for production performance targets: ≥5 MB/s throughput, <2 GB peak memory, offline operation with cached models, and efficient streaming for large files.

## Scope
- Throughput optimization (≥5 MB/s target)
- Memory profiling and optimization
- Model caching for offline operation
- Streaming vs batch processing decisions
- Concurrent processing where safe
- Performance benchmarking

## Requirements Alignment
- **Performance Target**: "Process ≥5 MB/s of mixed document types"
- **Memory Constraint**: "Peak usage <2 GB for large document processing"
- **Offline Operation**: "100% functionality without network dependency"

## Component Design

### Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor normalization pipeline performance."""
    
    def __init__(self):
        self.start_time = None
        self.bytes_processed = 0
        self.documents_processed = 0
    
    def record_document(self, size_bytes: int) -> None:
        """Record document processing."""
        self.bytes_processed += size_bytes
        self.documents_processed += 1
    
    def get_throughput_mbps(self) -> float:
        """Calculate throughput in MB/s."""
        if not self.start_time:
            return 0.0
        
        duration_sec = (datetime.utcnow() - self.start_time).total_seconds()
        if duration_sec == 0:
            return 0.0
        
        mb_processed = self.bytes_processed / (1024 * 1024)
        return mb_processed / duration_sec
```

### Model Caching

```python
# Cache fasttext model for offline operation
CACHED_MODELS_DIR = Path.home() / ".futurnal" / "models"

def cache_language_detection_model():
    """Cache fasttext model for offline operation."""
    CACHED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download and cache model
    import fasttext
    model_path = CACHED_MODELS_DIR / "lid.176.bin"
    
    if not model_path.exists():
        # Download on first run
        logger.info("Downloading language detection model...")
        # Implementation
    
    return model_path
```

## Acceptance Criteria

- ✅ Throughput meets ≥5 MB/s target
- ✅ Peak memory <2 GB for large files
- ✅ 100% offline operation
- ✅ Models cached locally
- ✅ Benchmarks passing

## Test Plan

### Performance Benchmarks
- Throughput measurement per format
- Memory profiling for large documents
- Offline operation verification
- Concurrent processing scalability

## Dependencies

- NormalizationService (Task 02)
- StreamingProcessor (Task 06)
- Performance telemetry

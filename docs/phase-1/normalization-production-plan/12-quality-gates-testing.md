Summary: Comprehensive testing strategy for normalization pipeline with determinism, format coverage, and integration tests.

# 12 · Quality Gates & Testing

## Purpose
Define and implement comprehensive testing strategy for normalization pipeline covering determinism, format coverage, performance benchmarks, integration tests, and production readiness criteria.

## Scope
- Determinism tests (byte-identical outputs)
- Format coverage tests (60+ formats)
- Performance benchmarks (throughput, memory)
- Integration tests (full pipeline)
- Edge case coverage
- Production readiness checklist

## Requirements Alignment
- **Feature Requirement**: "Deterministic outputs for identical inputs (idempotency)"
- **Testing Strategy**: "Determinism Tests: Re-run normalization to confirm identical outputs"
- **Production Quality**: Comprehensive test coverage before GA

## Component Design

### Determinism Test Suite

```python
import pytest

class TestDeterminism:
    """Test suite for normalization determinism."""
    
    async def test_identical_content_produces_identical_hash(self):
        """Verify content hashing is deterministic."""
        content = "Test document content"
        
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        
        assert hash1 == hash2
    
    async def test_normalize_same_document_twice(self):
        """Verify normalizing same document twice produces identical results."""
        file_path = Path("fixtures/sample.md")
        
        result1 = await service.normalize_document(
            file_path=file_path,
            source_id="test_1",
            source_type="test"
        )
        
        result2 = await service.normalize_document(
            file_path=file_path,
            source_id="test_2",
            source_type="test"
        )
        
        assert result1.sha256 == result2.sha256
        assert result1.content == result2.content
```

### Format Coverage Tests

```python
# Test fixture generation for all 60+ formats
TEST_FORMATS = [
    DocumentFormat.MARKDOWN,
    DocumentFormat.PDF,
    DocumentFormat.HTML,
    DocumentFormat.EMAIL,
    DocumentFormat.DOCX,
    DocumentFormat.PPTX,
    DocumentFormat.XLSX,
    DocumentFormat.CSV,
    DocumentFormat.JSON,
    DocumentFormat.YAML,
    DocumentFormat.CODE,
    # ... all formats
]

@pytest.mark.parametrize("format", TEST_FORMATS)
async def test_format_processing(format: DocumentFormat):
    """Test processing for each supported format."""
    fixture_path = get_fixture_for_format(format)
    
    normalized = await service.normalize_document(
        file_path=fixture_path,
        source_id=f"test_{format.value}",
        source_type="test"
    )
    
    assert normalized.metadata.format == format
    assert normalized.sha256 is not None
    assert len(normalized.content or normalized.chunks) > 0
```

### Performance Benchmarks

```python
@pytest.mark.performance
async def test_throughput_benchmark():
    """Benchmark normalization throughput."""
    test_files = generate_test_corpus(total_size_mb=100)
    
    start_time = datetime.utcnow()
    
    for file_path in test_files:
        await service.normalize_document(
            file_path=file_path,
            source_id=str(file_path),
            source_type="benchmark"
        )
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    throughput = 100 / duration  # MB/s
    
    assert throughput >= 5.0, f"Throughput {throughput:.2f} MB/s below target"
```

## Production Readiness Checklist

- ✅ All 60+ formats parse successfully with sample documents
- ✅ Determinism tests pass 100% (byte-identical outputs)
- ✅ Performance benchmarks meet ≥5 MB/s target
- ✅ Memory usage <2 GB for largest test documents
- ✅ Integration tests pass for all connector types
- ✅ Quarantine workflow handles all failure modes
- ✅ Privacy audit shows no content leakage in logs
- ✅ Streaming processor handles 1GB+ documents without OOM
- ✅ Offline operation verified (no network calls)
- ✅ Metrics exported to telemetry correctly

## Test Plan

### Unit Tests
- Schema validation
- Adapter selection
- Chunking strategies
- Metadata enrichment
- Error classification

### Integration Tests
- Full pipeline (connector → normalization → PKG)
- Multi-format batches
- Error recovery
- Concurrent processing

### Performance Tests
- Throughput benchmarks
- Memory profiling
- Large file handling
- Concurrent processing scalability

### Edge Case Tests
- Empty documents
- Very large documents (>1GB)
- Corrupted files
- Malformed content
- Unicode and special characters

## Dependencies

- All normalization pipeline components
- Test fixtures for all formats
- Performance benchmarking utilities
- CI/CD integration

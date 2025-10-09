Summary: Integrate normalization service with IngestionOrchestrator, NormalizationSink, and audit logging.

# 10 · Orchestrator Integration

## Purpose
Integrate the NormalizationService with the existing IngestionOrchestrator, enabling automated normalization of documents from all connectors, delivery to NormalizationSink, state checkpointing, and comprehensive audit logging.

## Scope
- Registration with IngestionOrchestrator
- NormalizationSink delivery pipeline
- StateStore checkpoint integration
- AuditLogger integration for all normalization events
- Job scheduling and retry logic
- Metrics collection and telemetry

## Requirements Alignment
- **Feature Requirement**: "Normalization service interface invoked by ingestion orchestrator"
- **Architecture**: "Orchestrator Integration: Register with IngestionOrchestrator, feed to NormalizationSink"
- **Observability**: "Logging captures format, duration, and success status"

## Component Design

### Orchestrator Registration

```python
from ...orchestrator.scheduler import IngestionOrchestrator
from ..pipeline.stubs import NormalizationSink

async def register_normalization_service(
    orchestrator: IngestionOrchestrator,
    normalization_service: NormalizationService,
    sink: NormalizationSink
) -> None:
    """Register normalization service with orchestrator."""
    
    # Register as document processor
    orchestrator.register_processor(
        processor_id="normalization_service",
        processor=normalization_service,
        sink=sink
    )
    
    logger.info("Normalization service registered with orchestrator")
```

### Integration with Connectors

```python
# Connectors deliver raw documents to normalization service
# via orchestrator job queue

async def process_connector_output(
    *,
    file_path: Path,
    source_id: str,
    source_type: str,
    metadata: dict
) -> None:
    """Process connector output through normalization pipeline."""
    
    normalized = await normalization_service.normalize_document(
        file_path=file_path,
        source_id=source_id,
        source_type=source_type,
        source_metadata=metadata
    )
    
    # Deliver to sink
    sink_payload = normalized.to_sink_format()
    sink.handle(sink_payload)
```

## Implementation Details

### NormalizationProcessor Component

The orchestrator integration is implemented via `NormalizationProcessor` in `src/futurnal/pipeline/normalization/orchestrator_integration.py`. This component:

- Wraps `NormalizationService` for orchestrator use
- Manages state checkpointing via `StateStore` for idempotency
- Records comprehensive audit events for all normalization activities
- Collects metrics (processing time, success rates, cache hits)
- Handles errors and integrates with quarantine system

### Factory Functions

Two factory functions are provided in `src/futurnal/pipeline/normalization/factory.py`:

1. **`create_normalization_processor()`**: Create processor with custom components
2. **`create_normalization_processor_with_workspace()`**: Create processor from workspace path

### Usage Example

```python
from pathlib import Path
from futurnal.ingestion.local.state import StateStore
from futurnal.pipeline.normalization import create_normalization_processor_with_workspace
from futurnal.pipeline.stubs import NormalizationSink

# Setup workspace
workspace = Path("~/.futurnal").expanduser()
state_store = StateStore(workspace / "state" / "state.db")

# Setup storage
pkg_writer = MyPKGWriter()
vector_writer = MyVectorWriter()
sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

# Create processor
processor = create_normalization_processor_with_workspace(
    workspace_path=workspace,
    state_store=state_store,
    sink=sink,
)

# Process files
result = await processor.process_file(
    file_path=Path("/path/to/document.md"),
    source_id="doc-123",
    source_type="local_files",
)

# Check metrics
metrics = processor.get_metrics()
print(f"Processed: {metrics['files_processed']}")
print(f"Success rate: {metrics['success_rate']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']}")
```

### State Checkpointing

The processor uses `StateStore` to track processed files:
- Files are identified by path, size, and mtime
- Changed files are automatically reprocessed
- Unchanged files are skipped (cached) for efficiency
- Supports force reprocessing when needed

### Audit Logging

Comprehensive audit events are recorded:
- `normalization_processing` - Processing started
- `normalization_completed` - Processing succeeded
- `normalization_failed` - Processing failed
- `normalization_skipped` - File cached, not reprocessed

All events are privacy-aware (no content exposure).

### Metrics Collection

The processor tracks:
- `files_processed` - Successfully processed files
- `files_failed` - Failed processing attempts
- `files_skipped_cached` - Files skipped due to cache
- `success_rate` - Success percentage
- `cache_hit_rate` - Cache efficiency
- `average_processing_time_ms` - Average processing duration

## Acceptance Criteria

- ✅ Registered with IngestionOrchestrator (via NormalizationProcessor)
- ✅ Receives documents from all connectors
- ✅ Delivers to NormalizationSink successfully (backward compatible format)
- ✅ State checkpointing working (via StateStore integration)
- ✅ Audit logging comprehensive (privacy-aware events)
- ✅ Metrics exported to telemetry (success rates, timing, cache hits)

## Test Plan

### Integration Tests
- ✅ End-to-end from file → normalization → PKG (`test_orchestrator_integration.py`)
- ✅ State checkpointing prevents duplicate processing
- ✅ Force reprocessing bypasses cache
- ✅ Audit logging completeness without content exposure
- ✅ Error handling and metrics tracking
- ✅ Batch processing support
- ✅ Multiple file format handling

### End-to-End System Tests
- ✅ Complete pipeline with real workspace (`test_normalization_orchestration.py`)
- ✅ Multi-file processing with state persistence
- ✅ Audit trail verification
- ✅ Error recovery and quarantine integration
- ✅ File update detection and reprocessing
- ✅ Performance metrics accuracy

## Dependencies

- ✅ IngestionOrchestrator (existing)
- ✅ NormalizationSink (existing, enhanced for backward compatibility)
- ✅ NormalizationService (Task 02)
- ✅ StateStore (existing)
- ✅ AuditLogger (existing)
- ✅ QuarantineStore (existing)

## Files Created/Modified

### New Files
- `src/futurnal/pipeline/normalization/orchestrator_integration.py` - NormalizationProcessor component
- `tests/pipeline/normalization/test_orchestrator_integration.py` - Comprehensive integration tests (16 tests)
- `tests/integration/test_normalization_orchestration.py` - End-to-end system tests (8 tests)

### Modified Files
- `src/futurnal/pipeline/normalization/factory.py` - Added processor factory functions
- `src/futurnal/pipeline/normalization/__init__.py` - Exported new components
- `src/futurnal/pipeline/stubs.py` - Enhanced NormalizationSink for backward compatibility

## Test Results

All 24 tests passing:
- 16/16 integration tests passing (`test_orchestrator_integration.py`)
- 8/8 end-to-end tests passing (`test_normalization_orchestration.py`)

**Status**: ✅ **COMPLETE** - Production-ready orchestrator integration implemented and tested.

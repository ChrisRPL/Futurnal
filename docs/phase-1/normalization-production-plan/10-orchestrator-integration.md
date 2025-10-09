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

## Acceptance Criteria

- ✅ Registered with IngestionOrchestrator
- ✅ Receives documents from all connectors
- ✅ Delivers to NormalizationSink successfully
- ✅ State checkpointing working
- ✅ Audit logging comprehensive
- ✅ Metrics exported to telemetry

## Test Plan

### Integration Tests
- End-to-end from connector → normalization → PKG
- Job scheduling and retry
- Sink delivery verification

## Dependencies

- IngestionOrchestrator (existing)
- NormalizationSink (existing)
- NormalizationService (Task 02)

Summary: Integrate normalization failures with quarantine system for operator review and reprocessing.

# 08 · Quarantine & Error Routing

## Purpose
Integrate normalization pipeline failures with the existing quarantine system, providing detailed diagnostics, operator CLI for reprocessing, and retry policies per failure classification. Ensures failed documents are tracked, diagnosable, and recoverable.

## Scope
- Integration with existing QuarantineManager
- Normalization-specific error classification
- Detailed diagnostic capture for failures
- Operator CLI for quarantine inspection and reprocessing
- Retry policies per failure type
- Privacy-aware error logging

## Requirements Alignment
- **Feature Requirement**: "Error routing to quarantine workflows with detailed diagnostics"
- **Implementation Guide**: "Quarantine Workflow: Persist failed items with reason codes; provide operator CLI to reprocess"
- **Reliability**: "Gracefully handle missing or malformed documents with retry and quarantine workflows"

## Component Design

### Normalization Error Classification

```python
from enum import Enum

class NormalizationErrorType(str, Enum):
    """Classification of normalization errors."""
    
    # Format errors (unlikely to succeed on retry)
    UNSUPPORTED_FORMAT = "unsupported_format"
    MALFORMED_CONTENT = "malformed_content"
    CORRUPTED_FILE = "corrupted_file"
    
    # Processing errors (may succeed on retry)
    UNSTRUCTURED_PARSE_ERROR = "unstructured_parse_error"
    CHUNKING_FAILURE = "chunking_failure"
    ENRICHMENT_FAILURE = "enrichment_failure"
    
    # Resource errors (transient)
    MEMORY_EXHAUSTED = "memory_exhausted"
    DISK_FULL = "disk_full"
    FILE_ACCESS_DENIED = "file_access_denied"
    
    # Privacy errors (never retry)
    ENCRYPTION_DETECTED = "encryption_detected"
    PERMISSION_DENIED = "permission_denied"


class NormalizationErrorHandler:
    """Handler for normalization errors with quarantine integration."""
    
    def __init__(self, quarantine_manager: QuarantineManager):
        self.quarantine_manager = quarantine_manager
        
    async def handle_error(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        error: Exception,
        metadata: Optional[dict] = None
    ) -> None:
        """Handle normalization error and route to quarantine."""
        error_type = self._classify_error(error)
        retry_policy = self._get_retry_policy(error_type)
        
        await self.quarantine_manager.quarantine_item(
            item_path=file_path,
            source_id=source_id,
            source_type=source_type,
            error_type=error_type.value,
            error_message=str(error),
            metadata={
                "normalization_failure": True,
                "retry_policy": retry_policy,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
        )
    
    def _classify_error(self, error: Exception) -> NormalizationErrorType:
        """Classify error for retry policy selection."""
        error_name = type(error).__name__
        error_msg = str(error).lower()
        
        if "unsupported" in error_msg or "format" in error_msg:
            return NormalizationErrorType.UNSUPPORTED_FORMAT
        elif "memory" in error_msg:
            return NormalizationErrorType.MEMORY_EXHAUSTED
        elif "permission" in error_msg or "access denied" in error_msg:
            return NormalizationErrorType.FILE_ACCESS_DENIED
        elif "unstructured" in error_msg:
            return NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR
        else:
            return NormalizationErrorType.MALFORMED_CONTENT
    
    def _get_retry_policy(self, error_type: NormalizationErrorType) -> str:
        """Get retry policy for error type."""
        if error_type in [
            NormalizationErrorType.MEMORY_EXHAUSTED,
            NormalizationErrorType.DISK_FULL,
            NormalizationErrorType.FILE_ACCESS_DENIED
        ]:
            return "retry_with_backoff"
        elif error_type in [
            NormalizationErrorType.ENCRYPTION_DETECTED,
            NormalizationErrorType.PERMISSION_DENIED,
            NormalizationErrorType.UNSUPPORTED_FORMAT
        ]:
            return "never_retry"
        else:
            return "retry_once"
```

## Acceptance Criteria

- ✅ Integration with existing QuarantineManager
- ✅ Error classification per failure type
- ✅ Detailed diagnostics captured
- ✅ Retry policies per error type
- ✅ Privacy-aware error logging
- ✅ Operator CLI for inspection

## Test Plan

### Unit Tests
- Error classification accuracy
- Retry policy selection
- Diagnostic capture

### Integration Tests
- Quarantine workflow end-to-end
- Reprocessing after fixes

## Dependencies

- QuarantineManager (existing)
- NormalizationService (Task 02)
- AuditLogger (existing)

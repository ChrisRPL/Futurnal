Summary: Implement content-hash based versioning and provenance tracking for PKG diffs.

# 09 · Versioning & Provenance Tracking

## Purpose
Implement content-hash based change detection, temporal metadata tracking, and document revision support for PKG versioning. Enables the Ghost to understand document evolution over time and detect meaningful changes for experiential learning.

## Scope
- SHA-256 content hashing for change detection
- Parent hash tracking for version chains
- Temporal metadata (created/modified/ingested)
- Change detection logic
- Version history support
- PKG diff preparation

## Requirements Alignment
- **Feature Requirement**: "Maintains change history to support PKG versioning"
- **Implementation Guide**: "Versioning Strategy: Track document revisions using content hashes and timestamps for PKG diffs"
- **Idempotency**: "Deterministic outputs for identical inputs"

## Component Design

### ProvenanceTracker

```python
import hashlib
from datetime import datetime
from typing import Optional

class ProvenanceTracker:
    """Track document provenance and version history."""
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
    
    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def detect_change(
        self,
        *,
        source_path: str,
        content_hash: str,
    ) -> tuple[bool, Optional[str]]:
        """Detect if document has changed since last processing.
        
        Returns:
            Tuple of (has_changed, previous_hash)
        """
        previous_hash = await self.state_store.get_document_hash(source_path)
        
        if previous_hash is None:
            # New document
            return True, None
        
        has_changed = previous_hash != content_hash
        return has_changed, previous_hash
    
    async def record_version(
        self,
        *,
        source_path: str,
        content_hash: str,
        parent_hash: Optional[str],
        timestamp: datetime
    ) -> None:
        """Record document version in state store."""
        await self.state_store.record_document_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=parent_hash,
            timestamp=timestamp
        )
```

## Acceptance Criteria

- ✅ Content hashing working (SHA-256)
- ✅ Change detection accurate
- ✅ Version chains tracked via parent hash
- ✅ Temporal metadata preserved
- ✅ Idempotent processing (same input = same hash)

## Test Plan

### Unit Tests
- Hash computation stability
- Change detection logic
- Version chain tracking

### Integration Tests
- Document modification detection
- Version history accuracy

## Dependencies

- NormalizedDocument schema (Task 01)
- StateStore (existing)

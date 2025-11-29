Summary: Implement repository pattern and query builders for PKG access with batching, pagination, and streaming support.

# 03 · Data Access Layer

## Purpose
Implement the data access layer following repository pattern with query builders for common operations, batching support, pagination, and streaming for large result sets.

## Scope
- Repository pattern implementation
- Query builders for entities, events, relationships
- Transaction management
- Batching and pagination
- Streaming support for large queries

## Component Design

```python
class PKGRepository:
    """Repository for PKG operations."""

    def __init__(self, driver):
        self.driver = driver

    # Entity operations
    def create_entity(self, entity_type: str, properties: Dict) -> str:
        """Create entity and return ID."""
        pass

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        pass

    def find_entities(
        self,
        entity_type: str,
        filters: Dict,
        limit: int = 100,
        offset: int = 0
    ) -> List[Entity]:
        """Find entities with pagination."""
        pass

    # Relationship operations
    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Dict
    ) -> str:
        """Create relationship and return ID."""
        pass

    # Temporal queries (delegated to TemporalQueryService)
    def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime
    ) -> List[Event]:
        """Find events within time range."""
        pass
```

## Testing
- Unit tests for CRUD operations
- Integration tests with schema
- Performance tests for batching

See module 01 for schema and module 04 for temporal queries.

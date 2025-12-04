"""PKG Repository Exception Hierarchy.

Custom exceptions for PKG repository operations providing clear error classification
for CRUD operations, batch processing, and temporal validation failures.

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/03-data-access-layer.md

Option B Compliance:
- TemporalValidationError for temporal ordering enforcement
- Production-ready error handling with detailed context
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple

from futurnal.pkg.database.exceptions import PKGDatabaseError


class PKGRepositoryError(PKGDatabaseError):
    """Base exception for all PKG repository operations.

    All repository-specific exceptions inherit from this class,
    allowing callers to catch all repository errors with a single handler.

    Example:
        try:
            repo.create_entity(entity)
        except PKGRepositoryError as e:
            logger.error(f"Repository operation failed: {e}")
    """

    pass


class EntityNotFoundError(PKGRepositoryError):
    """Raised when an entity with the given ID does not exist.

    Attributes:
        entity_id: The ID that was searched for
        entity_type: The entity type that was searched (if known)
    """

    def __init__(
        self,
        entity_id: str,
        entity_type: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.entity_id = entity_id
        self.entity_type = entity_type

        if message is None:
            if entity_type:
                message = f"{entity_type} with id '{entity_id}' not found"
            else:
                message = f"Entity with id '{entity_id}' not found"

        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.entity_type:
            return f"{base} (type={self.entity_type})"
        return base


class RelationshipNotFoundError(PKGRepositoryError):
    """Raised when a relationship with the given ID does not exist.

    Attributes:
        relationship_id: The relationship ID that was searched for
        relationship_type: The relationship type (if known)
    """

    def __init__(
        self,
        relationship_id: str,
        relationship_type: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.relationship_id = relationship_id
        self.relationship_type = relationship_type

        if message is None:
            if relationship_type:
                message = f"{relationship_type} relationship with id '{relationship_id}' not found"
            else:
                message = f"Relationship with id '{relationship_id}' not found"

        super().__init__(message)


class DuplicateEntityError(PKGRepositoryError):
    """Raised when attempting to create an entity that already exists.

    Attributes:
        entity_id: The ID that already exists
        entity_type: The entity type (if known)
    """

    def __init__(
        self,
        entity_id: str,
        entity_type: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.entity_id = entity_id
        self.entity_type = entity_type

        if message is None:
            if entity_type:
                message = f"{entity_type} with id '{entity_id}' already exists"
            else:
                message = f"Entity with id '{entity_id}' already exists"

        super().__init__(message)


class TemporalValidationError(PKGRepositoryError):
    """Raised when temporal ordering constraint is violated.

    Option B Critical: This enforces temporal-first design for relationships.
    For BEFORE relationships, source event must have timestamp < target timestamp.
    For CAUSES relationships, cause must precede effect.

    Attributes:
        source_id: The source event ID
        target_id: The target event ID
        relationship_type: The relationship type being created
        source_timestamp: Timestamp of the source event
        target_timestamp: Timestamp of the target event
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        source_timestamp: Optional[datetime] = None,
        target_timestamp: Optional[datetime] = None,
        message: Optional[str] = None,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.source_timestamp = source_timestamp
        self.target_timestamp = target_timestamp

        if message is None:
            if source_timestamp and target_timestamp:
                message = (
                    f"Temporal ordering violation for {relationship_type}: "
                    f"source ({source_id}) at {source_timestamp.isoformat()} "
                    f"must precede target ({target_id}) at {target_timestamp.isoformat()}"
                )
            else:
                message = (
                    f"Temporal ordering violation for {relationship_type}: "
                    f"source ({source_id}) must precede target ({target_id})"
                )

        super().__init__(message)

    def __str__(self) -> str:
        return super().__str__()


class InvalidEntityTypeError(PKGRepositoryError):
    """Raised when an invalid entity type is specified.

    Attributes:
        entity_type: The invalid entity type
        valid_types: List of valid entity types
    """

    def __init__(
        self,
        entity_type: str,
        valid_types: Optional[List[str]] = None,
        message: Optional[str] = None,
    ):
        self.entity_type = entity_type
        self.valid_types = valid_types or []

        if message is None:
            if valid_types:
                message = (
                    f"Invalid entity type '{entity_type}'. "
                    f"Valid types: {', '.join(valid_types)}"
                )
            else:
                message = f"Invalid entity type '{entity_type}'"

        super().__init__(message)


class InvalidRelationshipTypeError(PKGRepositoryError):
    """Raised when an invalid relationship type is specified.

    Attributes:
        relationship_type: The invalid relationship type
        valid_types: List of valid relationship types
    """

    def __init__(
        self,
        relationship_type: str,
        valid_types: Optional[List[str]] = None,
        message: Optional[str] = None,
    ):
        self.relationship_type = relationship_type
        self.valid_types = valid_types or []

        if message is None:
            if valid_types:
                message = (
                    f"Invalid relationship type '{relationship_type}'. "
                    f"Valid types: {', '.join(valid_types)}"
                )
            else:
                message = f"Invalid relationship type '{relationship_type}'"

        super().__init__(message)


class BatchOperationError(PKGRepositoryError):
    """Raised when a batch operation partially or fully fails.

    Provides detailed information about succeeded and failed items
    for recovery and reporting.

    Attributes:
        succeeded_count: Number of items that succeeded
        failed_count: Number of items that failed
        failed_items: List of (item_id, error_message) tuples
        partial: True if some items succeeded
    """

    def __init__(
        self,
        message: str,
        succeeded_count: int = 0,
        failed_count: int = 0,
        failed_items: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__(message)
        self.succeeded_count = succeeded_count
        self.failed_count = failed_count
        self.failed_items = failed_items or []
        self.partial = succeeded_count > 0 and failed_count > 0

    def __str__(self) -> str:
        base = super().__str__()
        stats = f"succeeded={self.succeeded_count}, failed={self.failed_count}"
        if self.partial:
            stats += " (partial failure)"
        return f"{base} ({stats})"


class QueryBuildError(PKGRepositoryError):
    """Raised when query builder encounters an invalid configuration.

    Attributes:
        query_fragment: The problematic query fragment
        reason: Explanation of what's wrong
    """

    def __init__(
        self,
        reason: str,
        query_fragment: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.reason = reason
        self.query_fragment = query_fragment

        if message is None:
            if query_fragment:
                message = f"Query build error: {reason} (fragment: {query_fragment})"
            else:
                message = f"Query build error: {reason}"

        super().__init__(message)


class StreamingError(PKGRepositoryError):
    """Raised when streaming operation fails.

    Attributes:
        items_streamed: Number of items successfully streamed before failure
        cursor_position: Last cursor position for resume
    """

    def __init__(
        self,
        message: str,
        items_streamed: int = 0,
        cursor_position: Optional[Any] = None,
    ):
        super().__init__(message)
        self.items_streamed = items_streamed
        self.cursor_position = cursor_position

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (streamed={self.items_streamed})"

"""PKG Repository Base Class.

Provides common patterns for all PKG repositories including:
- Transaction management via PKGDatabaseManager.session()
- Result mapping from Neo4j records to Pydantic models
- Audit logging integration
- Privacy-aware logging

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/03-data-access-layer.md

Option B Compliance:
- Integrates with existing PKGDatabaseManager
- Production-ready with proper error handling
- Privacy-first logging patterns
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, Optional, Type, TYPE_CHECKING

from futurnal.pkg.database.manager import PKGDatabaseManager
from futurnal.pkg.database.exceptions import PKGConnectionError
from futurnal.pkg.schema.models import (
    BaseNode,
    PersonNode,
    OrganizationNode,
    ConceptNode,
    DocumentNode,
    EventNode,
    ChunkNode,
    SchemaVersionNode,
)

if TYPE_CHECKING:
    from neo4j import Record, Session
    from futurnal.privacy.audit import AuditLogger


# ---------------------------------------------------------------------------
# Node Type Mapping
# ---------------------------------------------------------------------------

# Maps Neo4j labels to Pydantic model classes
NODE_TYPE_MAP: Dict[str, Type[BaseNode]] = {
    "Person": PersonNode,
    "Organization": OrganizationNode,
    "Concept": ConceptNode,
    "Document": DocumentNode,
    "Event": EventNode,
    "Chunk": ChunkNode,
    "SchemaVersion": SchemaVersionNode,
}

# All valid node type names
VALID_NODE_TYPES = list(NODE_TYPE_MAP.keys())


def get_node_class(node_type: str) -> Type[BaseNode]:
    """Get the Pydantic model class for a node type.

    Args:
        node_type: Node type name (e.g., "Person", "Event")

    Returns:
        The corresponding Pydantic model class

    Raises:
        ValueError: If node_type is not valid
    """
    if node_type not in NODE_TYPE_MAP:
        raise ValueError(
            f"Unknown node type '{node_type}'. Valid types: {VALID_NODE_TYPES}"
        )
    return NODE_TYPE_MAP[node_type]


def get_label_for_node(node: BaseNode) -> str:
    """Get the Neo4j label for a Pydantic node model.

    Args:
        node: A BaseNode instance

    Returns:
        The Neo4j label string (class name without 'Node' suffix)
    """
    class_name = type(node).__name__
    if class_name.endswith("Node"):
        return class_name[:-4]  # Remove "Node" suffix
    return class_name


# ---------------------------------------------------------------------------
# Base Repository
# ---------------------------------------------------------------------------


class BaseRepository:
    """Abstract base class for PKG repositories.

    Provides common patterns:
    - Transaction management via PKGDatabaseManager.session()
    - Logging with privacy-aware context
    - Audit integration
    - Result mapping utilities

    Example:
        >>> class EntityRepository(BaseRepository):
        ...     def create(self, entity: BaseNode) -> str:
        ...         with self._transaction() as session:
        ...             result = session.run("CREATE (n:Person $props) RETURN n.id", props=entity.to_cypher_properties())
        ...             return result.single()["n.id"]
    """

    def __init__(
        self,
        db_manager: PKGDatabaseManager,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize the repository.

        Args:
            db_manager: The PKGDatabaseManager for database access.
                       Must be connected before calling repository methods.
            audit_logger: Optional audit logger for recording operations.
        """
        self._db = db_manager
        self._audit = audit_logger
        self._logger = logging.getLogger(self.__class__.__name__)

    # ---------------------------------------------------------------------------
    # Transaction Management
    # ---------------------------------------------------------------------------

    @contextmanager
    def _transaction(
        self, database: Optional[str] = None
    ) -> Generator["Session", None, None]:
        """Provide a transactional session context.

        Wraps PKGDatabaseManager.session() with additional error handling
        and logging.

        Args:
            database: Optional database name. If None, uses default.

        Yields:
            A Neo4j Session for executing queries.

        Raises:
            PKGConnectionError: If not connected to database.

        Example:
            >>> with self._transaction() as session:
            ...     session.run("CREATE (n:Person) RETURN n")
        """
        if not self._db.is_connected:
            raise PKGConnectionError(
                "Database not connected. Call db_manager.connect() first."
            )

        with self._db.session(database=database) as session:
            yield session

    def _execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> list:
        """Execute a read query and return all records.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name

        Returns:
            List of Neo4j Record objects
        """
        with self._transaction(database) as session:
            result = session.run(query, parameters or {})
            return list(result)

    def _execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Any:
        """Execute a write query and return the result.

        Uses session.execute_write() for proper transaction handling.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name

        Returns:
            Query result
        """
        def _work(tx):
            result = tx.run(query, parameters or {})
            return result.single()

        with self._transaction(database) as session:
            return session.execute_write(_work)

    # ---------------------------------------------------------------------------
    # Result Mapping
    # ---------------------------------------------------------------------------

    def _map_record_to_node(
        self,
        record: "Record",
        node_key: str = "n",
        labels_key: Optional[str] = "labels",
        node_class: Optional[Type[BaseNode]] = None,
    ) -> BaseNode:
        """Convert a Neo4j record to a Pydantic node model.

        Args:
            record: The Neo4j record containing node data
            node_key: The key in the record for the node data
            labels_key: The key in the record for labels (if any)
            node_class: Optional explicit node class. If None, auto-detects from labels.

        Returns:
            A Pydantic model instance

        Raises:
            ValueError: If node type cannot be determined
        """
        node_data = dict(record[node_key])

        # Determine the node class
        if node_class is None:
            # Try to get labels from record
            if labels_key and labels_key in record.keys():
                labels = record[labels_key]
                for label in labels:
                    if label in NODE_TYPE_MAP:
                        node_class = NODE_TYPE_MAP[label]
                        break

            if node_class is None:
                raise ValueError(
                    f"Could not determine node type from record. Labels: {labels}"
                )

        # Convert Neo4j types to Python types
        converted_data = self._convert_neo4j_types(node_data)

        return node_class.model_validate(converted_data)

    def _convert_neo4j_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Neo4j types to Python types for Pydantic.

        Handles:
        - neo4j.time.DateTime -> datetime
        - neo4j.time.Duration -> timedelta
        - ISO strings -> datetime (fallback)

        Args:
            data: Dictionary with Neo4j values

        Returns:
            Dictionary with Python types
        """
        converted = {}
        for key, value in data.items():
            if value is None:
                converted[key] = None
            elif hasattr(value, "to_native"):
                # neo4j.time types have to_native() method
                converted[key] = value.to_native()
            elif isinstance(value, str) and self._looks_like_iso_datetime(value):
                # Try parsing ISO datetime strings
                try:
                    converted[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    converted[key] = value
            elif isinstance(value, (int, float)) and key.endswith("_seconds"):
                # Convert duration fields from seconds to timedelta
                converted[key.replace("_seconds", "")] = timedelta(seconds=value)
            else:
                converted[key] = value
        return converted

    def _looks_like_iso_datetime(self, value: str) -> bool:
        """Check if a string looks like an ISO datetime."""
        # Quick check for common datetime patterns
        return (
            len(value) >= 10
            and value[4:5] == "-"
            and value[7:8] == "-"
            and ("T" in value or len(value) == 10)
        )

    def _map_records_to_nodes(
        self,
        records: list,
        node_key: str = "n",
        labels_key: Optional[str] = "labels",
        node_class: Optional[Type[BaseNode]] = None,
    ) -> list:
        """Convert multiple Neo4j records to Pydantic models.

        Args:
            records: List of Neo4j records
            node_key: The key in each record for the node data
            labels_key: The key for labels
            node_class: Optional explicit node class

        Returns:
            List of Pydantic model instances
        """
        return [
            self._map_record_to_node(record, node_key, labels_key, node_class)
            for record in records
        ]

    # ---------------------------------------------------------------------------
    # Audit Logging
    # ---------------------------------------------------------------------------

    def _audit_operation(
        self,
        operation: str,
        entity_type: str,
        entity_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an audit event if audit logger is configured.

        Privacy-aware: Does not log entity content, only operation metadata.

        Args:
            operation: Operation name (create, read, update, delete)
            entity_type: Entity type (Person, Event, etc.)
            entity_id: Entity identifier
            success: Whether operation succeeded
            metadata: Additional metadata for audit
        """
        if self._audit is None:
            return

        try:
            self._audit.record(
                job_id=f"pkg_repo_{operation}_{datetime.utcnow().isoformat()}",
                source=f"pkg_repository.{self.__class__.__name__}",
                action=operation,
                status="succeeded" if success else "failed",
                timestamp=datetime.utcnow(),
                metadata={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    **(metadata or {}),
                },
            )
        except Exception as e:
            # Don't fail operations due to audit logging issues
            self._logger.debug(f"Failed to record audit event: {e}")

    def _audit_batch_operation(
        self,
        operation: str,
        entity_type: str,
        count: int,
        success: bool,
        failed_count: int = 0,
    ) -> None:
        """Record audit event for batch operation.

        Args:
            operation: Operation name
            entity_type: Entity type
            count: Total items processed
            success: Whether operation fully succeeded
            failed_count: Number of failed items
        """
        if self._audit is None:
            return

        try:
            self._audit.record(
                job_id=f"pkg_repo_batch_{operation}_{datetime.utcnow().isoformat()}",
                source=f"pkg_repository.{self.__class__.__name__}",
                action=f"batch_{operation}",
                status="succeeded" if success else "partial" if failed_count < count else "failed",
                timestamp=datetime.utcnow(),
                metadata={
                    "entity_type": entity_type,
                    "total_count": count,
                    "failed_count": failed_count,
                },
            )
        except Exception as e:
            self._logger.debug(f"Failed to record audit event: {e}")

    # ---------------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------------

    def _chunk_list(self, items: list, chunk_size: int) -> Generator[list, None, None]:
        """Yield successive chunks from a list.

        Args:
            items: List to chunk
            chunk_size: Size of each chunk

        Yields:
            Lists of chunk_size items (last chunk may be smaller)
        """
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def _sanitize_property_name(self, name: str) -> str:
        """Sanitize a property name for use in Cypher.

        Ensures the property name is safe to use in dynamic queries.

        Args:
            name: Property name

        Returns:
            Sanitized property name

        Raises:
            ValueError: If property name contains invalid characters
        """
        # Only allow alphanumeric and underscores
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid property name: {name}")
        return name

    def _validate_node_type(self, node_type: str) -> str:
        """Validate and return a node type.

        Args:
            node_type: Node type to validate

        Returns:
            The validated node type

        Raises:
            ValueError: If node type is invalid
        """
        if node_type not in NODE_TYPE_MAP:
            raise ValueError(
                f"Invalid node type '{node_type}'. Valid types: {VALID_NODE_TYPES}"
            )
        return node_type

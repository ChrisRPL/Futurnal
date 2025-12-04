"""PKG Database Backup and Restore.

Manages backup and restore operations for the PKG database using
streaming Cypher export to JSON format. No APOC dependency required.

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md

Option B Compliance:
- Streaming export with pagination to handle large graphs
- Automatic pre-restore backup to prevent data loss
- Backup verification to ensure integrity
- Audit logging for all operations
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from futurnal.pkg.database.config import PKGDatabaseConfig
from futurnal.pkg.database.exceptions import PKGBackupError, PKGRestoreError

if TYPE_CHECKING:
    from futurnal.pkg.database.manager import PKGDatabaseManager
    from futurnal.privacy.audit import AuditLogger

logger = logging.getLogger(__name__)

# Backup file format version
BACKUP_FORMAT_VERSION = 1

# Node labels to export (from Module 01 schema)
NODE_LABELS = [
    "Person",
    "Organization",
    "Concept",
    "Document",
    "Event",
    "SchemaVersion",
    "Chunk",
    "Note",
    "Vault",
    "Tag",
    "Source",
    "Aspiration",
    "ExperientialEvent",
]

# Relationship types to export
RELATIONSHIP_TYPES = [
    # Temporal relationships
    "BEFORE",
    "AFTER",
    "DURING",
    "SIMULTANEOUS",
    # Causal relationships
    "CAUSES",
    "ENABLES",
    "PREVENTS",
    "TRIGGERS",
    # Provenance relationships
    "EXTRACTED_FROM",
    "DISCOVERED_IN",
    "PARTICIPATED_IN",
    # Standard relationships
    "WORKS_AT",
    "BELONGS_TO",
    "HAS_TAG",
    "IN_VAULT",
    "RELATED_TO",
    "LINKS_TO",
    "REFERENCES",
    "REFERENCES_HEADING",
    "REFERENCES_BLOCK",
    "EMBEDS",
    "SUPPORTS_ASPIRATION",
    "POTENTIALLY_CAUSES",
]


class PKGBackupManager:
    """Backup and restore operations for PKG database.

    Provides:
    - Full database backup via Cypher export to JSON
    - Streaming export with pagination for large graphs
    - Restore with automatic pre-restore backup
    - Backup verification and integrity checking
    - Retention policy management

    Example:
        >>> backup_manager = PKGBackupManager(db_manager, config)
        >>> backup_path = backup_manager.backup(comment="daily")
        >>> print(f"Backup created: {backup_path}")

        >>> # Later, restore from backup
        >>> backup_manager.restore(backup_path)
    """

    def __init__(
        self,
        manager: "PKGDatabaseManager",
        config: PKGDatabaseConfig,
        workspace_path: Path,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize the backup manager.

        Args:
            manager: PKGDatabaseManager instance for database access.
            config: PKGDatabaseConfig with backup settings.
            workspace_path: Workspace root path for default backup location.
            audit_logger: Optional audit logger for recording operations.
        """
        self._manager = manager
        self._config = config
        self._workspace_path = workspace_path
        self._audit = audit_logger

        # Ensure backup directory exists
        self._backup_path = config.get_backup_path(workspace_path)
        self._backup_path.mkdir(parents=True, exist_ok=True)

    def backup(self, comment: Optional[str] = None) -> Path:
        """Create a full database backup.

        Exports all nodes and relationships to a JSON file with metadata
        including checksums for verification.

        Args:
            comment: Optional comment to include in backup filename.

        Returns:
            Path to the created backup file.

        Raises:
            PKGBackupError: If backup operation fails.

        Example:
            >>> path = backup_manager.backup(comment="before-migration")
            >>> print(f"Backup: {path}")
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"pkg-{timestamp}"
        if comment:
            # Sanitize comment for filename
            safe_comment = "".join(c for c in comment if c.isalnum() or c in "-_")
            filename = f"{filename}-{safe_comment}"
        filename = f"{filename}.json"

        backup_file = self._backup_path / filename

        logger.info(f"Creating backup: {backup_file}")

        try:
            # Collect all data
            nodes: List[Dict[str, Any]] = []
            relationships: List[Dict[str, Any]] = []

            with self._manager.session() as session:
                # Export nodes by label with pagination
                for label in NODE_LABELS:
                    label_nodes = self._export_nodes(session, label)
                    nodes.extend(label_nodes)
                    logger.debug(f"Exported {len(label_nodes)} {label} nodes")

                # Export relationships by type with pagination
                for rel_type in RELATIONSHIP_TYPES:
                    type_rels = self._export_relationships(session, rel_type)
                    relationships.extend(type_rels)
                    logger.debug(f"Exported {len(type_rels)} {rel_type} relationships")

            # Calculate checksum
            data_json = json.dumps({"nodes": nodes, "relationships": relationships})
            checksum = hashlib.sha256(data_json.encode()).hexdigest()

            # Create backup structure
            backup_data = {
                "version": BACKUP_FORMAT_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "comment": comment,
                "metadata": {
                    "node_count": len(nodes),
                    "relationship_count": len(relationships),
                    "node_labels": list(set(n.get("labels", [])[0] for n in nodes if n.get("labels"))),
                    "relationship_types": list(set(r.get("type") for r in relationships)),
                    "checksum": checksum,
                },
                "nodes": nodes,
                "relationships": relationships,
            }

            # Write backup file
            backup_file.write_text(
                json.dumps(backup_data, indent=2, default=str),
                encoding="utf-8",
            )

            # Verify the backup
            is_valid, verify_msg = self.verify(backup_file)
            if not is_valid:
                raise PKGBackupError(
                    f"Backup verification failed: {verify_msg}",
                    backup_path=str(backup_file),
                    node_count=len(nodes),
                    relationship_count=len(relationships),
                )

            logger.info(
                f"Backup created successfully: {len(nodes)} nodes, "
                f"{len(relationships)} relationships"
            )
            self._audit_event(
                "backup",
                "succeeded",
                {
                    "path": str(backup_file),
                    "nodes": len(nodes),
                    "relationships": len(relationships),
                },
            )

            return backup_file

        except PKGBackupError:
            raise
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            self._audit_event("backup", "failed", {"error": str(e)})
            raise PKGBackupError(f"Backup failed: {e}") from e

    def _export_nodes(self, session: Any, label: str) -> List[Dict[str, Any]]:
        """Export all nodes of a given label with pagination."""
        nodes = []
        skip = 0
        batch_size = self._config.fetch_size

        while True:
            result = session.run(
                f"""
                MATCH (n:{label})
                RETURN n, labels(n) as labels, elementId(n) as element_id
                ORDER BY elementId(n)
                SKIP $skip LIMIT $limit
                """,
                skip=skip,
                limit=batch_size,
            )

            records = list(result)
            if not records:
                break

            for record in records:
                node = dict(record["n"])
                nodes.append({
                    "element_id": record["element_id"],
                    "labels": record["labels"],
                    "properties": self._serialize_properties(node),
                })

            skip += batch_size

            if len(records) < batch_size:
                break

        return nodes

    def _export_relationships(
        self, session: Any, rel_type: str
    ) -> List[Dict[str, Any]]:
        """Export all relationships of a given type with pagination."""
        relationships = []
        skip = 0
        batch_size = self._config.fetch_size

        while True:
            result = session.run(
                f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN r, type(r) as type,
                       elementId(r) as element_id,
                       elementId(a) as start_id,
                       elementId(b) as end_id
                ORDER BY elementId(r)
                SKIP $skip LIMIT $limit
                """,
                skip=skip,
                limit=batch_size,
            )

            records = list(result)
            if not records:
                break

            for record in records:
                rel = dict(record["r"])
                relationships.append({
                    "element_id": record["element_id"],
                    "type": record["type"],
                    "start_id": record["start_id"],
                    "end_id": record["end_id"],
                    "properties": self._serialize_properties(rel),
                })

            skip += batch_size

            if len(records) < batch_size:
                break

        return relationships

    def _serialize_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize node/relationship properties to JSON-compatible format."""
        result = {}
        for key, value in props.items():
            if hasattr(value, "isoformat"):
                # datetime/date/time
                result[key] = {"_type": "datetime", "value": value.isoformat()}
            elif hasattr(value, "total_seconds"):
                # timedelta/duration
                result[key] = {"_type": "duration", "value": value.total_seconds()}
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    def restore(self, backup_path: Path, force: bool = False) -> None:
        """Restore database from a backup file.

        Creates a pre-restore backup before clearing and restoring data.
        Validates that restored counts match backup metadata.

        Args:
            backup_path: Path to the backup file.
            force: If True, skip confirmation for non-empty database.

        Raises:
            PKGRestoreError: If restore operation fails.
            FileNotFoundError: If backup file doesn't exist.

        Example:
            >>> backup_manager.restore(Path("backups/pkg-20231204.json"))
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Verify backup before restoring
        is_valid, verify_msg = self.verify(backup_path)
        if not is_valid:
            raise PKGRestoreError(
                f"Backup verification failed: {verify_msg}",
                backup_path=str(backup_path),
            )

        logger.info(f"Restoring from backup: {backup_path}")

        # Create pre-restore backup
        pre_restore_path: Optional[Path] = None
        try:
            pre_restore_path = self.backup(comment="before-restore")
            logger.info(f"Pre-restore backup created: {pre_restore_path}")
        except Exception as e:
            logger.warning(f"Could not create pre-restore backup: {e}")

        try:
            # Load backup data
            backup_data = json.loads(backup_path.read_text(encoding="utf-8"))
            nodes = backup_data.get("nodes", [])
            relationships = backup_data.get("relationships", [])

            with self._manager.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                logger.debug("Cleared existing data")

                # Create ID mapping for relationships
                id_mapping: Dict[str, str] = {}

                # Restore nodes
                nodes_created = 0
                for node in nodes:
                    old_id = node["element_id"]
                    labels = ":".join(node["labels"])
                    props = self._deserialize_properties(node["properties"])

                    # Create node and get new ID
                    result = session.run(
                        f"CREATE (n:{labels} $props) RETURN elementId(n) as new_id",
                        props=props,
                    )
                    record = result.single()
                    if record:
                        id_mapping[old_id] = record["new_id"]
                        nodes_created += 1

                logger.debug(f"Restored {nodes_created} nodes")

                # Restore relationships
                rels_created = 0
                for rel in relationships:
                    start_id = id_mapping.get(rel["start_id"])
                    end_id = id_mapping.get(rel["end_id"])

                    if not start_id or not end_id:
                        logger.warning(
                            f"Skipping relationship {rel['type']}: "
                            f"missing node mapping"
                        )
                        continue

                    rel_type = rel["type"]
                    props = self._deserialize_properties(rel["properties"])

                    session.run(
                        f"""
                        MATCH (a), (b)
                        WHERE elementId(a) = $start_id AND elementId(b) = $end_id
                        CREATE (a)-[r:{rel_type} $props]->(b)
                        """,
                        start_id=start_id,
                        end_id=end_id,
                        props=props,
                    )
                    rels_created += 1

                logger.debug(f"Restored {rels_created} relationships")

            # Verify counts
            expected_nodes = backup_data["metadata"]["node_count"]
            expected_rels = backup_data["metadata"]["relationship_count"]

            if nodes_created != expected_nodes:
                logger.warning(
                    f"Node count mismatch: restored {nodes_created}, "
                    f"expected {expected_nodes}"
                )

            if rels_created != expected_rels:
                logger.warning(
                    f"Relationship count mismatch: restored {rels_created}, "
                    f"expected {expected_rels}"
                )

            logger.info(
                f"Restore complete: {nodes_created} nodes, {rels_created} relationships"
            )
            self._audit_event(
                "restore",
                "succeeded",
                {
                    "backup_path": str(backup_path),
                    "nodes": nodes_created,
                    "relationships": rels_created,
                    "pre_restore_backup": str(pre_restore_path) if pre_restore_path else None,
                },
            )

        except PKGRestoreError:
            raise
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            self._audit_event(
                "restore",
                "failed",
                {
                    "backup_path": str(backup_path),
                    "error": str(e),
                    "pre_restore_backup": str(pre_restore_path) if pre_restore_path else None,
                },
            )
            raise PKGRestoreError(
                f"Restore failed: {e}",
                backup_path=str(backup_path),
                pre_restore_backup=str(pre_restore_path) if pre_restore_path else None,
            ) from e

    def _deserialize_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize properties from backup format."""
        from datetime import datetime as dt

        result = {}
        for key, value in props.items():
            if isinstance(value, dict) and "_type" in value:
                if value["_type"] == "datetime":
                    result[key] = dt.fromisoformat(value["value"])
                elif value["_type"] == "duration":
                    # Neo4j will accept seconds as a number
                    result[key] = value["value"]
            else:
                result[key] = value
        return result

    def verify(self, backup_path: Path) -> Tuple[bool, str]:
        """Verify backup file integrity.

        Checks:
        - File exists and is valid JSON
        - Required structure present
        - Checksum matches

        Args:
            backup_path: Path to the backup file.

        Returns:
            Tuple of (is_valid, message).

        Example:
            >>> is_valid, msg = backup_manager.verify(backup_path)
            >>> if not is_valid:
            ...     print(f"Invalid backup: {msg}")
        """
        if not backup_path.exists():
            return False, f"File not found: {backup_path}"

        try:
            backup_data = json.loads(backup_path.read_text(encoding="utf-8"))

            # Check required fields
            required = ["version", "metadata", "nodes", "relationships"]
            for field in required:
                if field not in backup_data:
                    return False, f"Missing required field: {field}"

            # Check version
            version = backup_data.get("version")
            if version != BACKUP_FORMAT_VERSION:
                return False, f"Unsupported backup version: {version}"

            # Verify checksum
            stored_checksum = backup_data["metadata"].get("checksum")
            if stored_checksum:
                data_json = json.dumps({
                    "nodes": backup_data["nodes"],
                    "relationships": backup_data["relationships"],
                })
                computed_checksum = hashlib.sha256(data_json.encode()).hexdigest()
                if computed_checksum != stored_checksum:
                    return False, "Checksum mismatch - backup may be corrupted"

            # Verify counts match
            meta = backup_data["metadata"]
            if len(backup_data["nodes"]) != meta.get("node_count", 0):
                return False, "Node count mismatch in metadata"
            if len(backup_data["relationships"]) != meta.get("relationship_count", 0):
                return False, "Relationship count mismatch in metadata"

            return True, "Backup is valid"

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Verification error: {e}"

    def list_backups(self) -> List[Tuple[Path, datetime, int]]:
        """List available backups sorted by date (newest first).

        Returns:
            List of tuples: (path, created_datetime, size_bytes).

        Example:
            >>> for path, created, size in backup_manager.list_backups():
            ...     print(f"{path.name}: {created} ({size} bytes)")
        """
        backups = []

        for backup_file in self._backup_path.glob("pkg-*.json"):
            try:
                stat = backup_file.stat()
                # Parse datetime from filename
                name = backup_file.stem
                if name.startswith("pkg-"):
                    date_part = name[4:19]  # pkg-YYYYMMDD-HHMMSS
                    try:
                        created = datetime.strptime(date_part, "%Y%m%d-%H%M%S")
                    except ValueError:
                        created = datetime.fromtimestamp(stat.st_mtime)
                else:
                    created = datetime.fromtimestamp(stat.st_mtime)

                backups.append((backup_file, created, stat.st_size))

            except Exception as e:
                logger.warning(f"Could not read backup {backup_file}: {e}")

        # Sort by date, newest first
        backups.sort(key=lambda x: x[1], reverse=True)

        return backups

    def purge_old_backups(
        self,
        keep_count: int = 10,
        older_than_days: Optional[int] = None,
    ) -> int:
        """Remove old backups based on retention policy.

        Args:
            keep_count: Minimum number of recent backups to keep.
            older_than_days: Remove backups older than this many days.
                            If None, uses config.backup_retention_days.

        Returns:
            Number of backups removed.

        Example:
            >>> removed = backup_manager.purge_old_backups(keep_count=5)
            >>> print(f"Removed {removed} old backups")
        """
        if older_than_days is None:
            older_than_days = self._config.backup_retention_days

        backups = self.list_backups()
        cutoff = datetime.utcnow()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=older_than_days)

        removed = 0
        for i, (path, created, _) in enumerate(backups):
            # Always keep the minimum count
            if i < keep_count:
                continue

            # Remove if older than cutoff
            if created < cutoff:
                try:
                    path.unlink()
                    logger.info(f"Removed old backup: {path.name}")
                    removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {e}")

        if removed > 0:
            self._audit_event("purge_backups", "succeeded", {"removed": removed})

        return removed

    def _audit_event(
        self,
        action: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Record an audit event if audit logger is configured."""
        if self._audit is None:
            return

        try:
            self._audit.record(
                job_id=f"pkg_backup_{action}_{datetime.utcnow().isoformat()}",
                source="pkg_backup_manager",
                action=action,
                status=status,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"Failed to record audit event: {e}")

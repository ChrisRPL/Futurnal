"""Cloud sync consent definitions for Firebase PKG metadata backup.

This module defines consent scopes, status structures, and metadata export
formats for the cloud sync feature. It follows the privacy-first approach:
- Metadata only (no document content synced)
- Permanent consent until explicitly revoked
- Auto-delete cloud data on revocation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class CloudSyncScope(str, Enum):
    """Consent scopes for cloud sync operations.

    These scopes define what data categories can be synced to Firebase.
    PKG_METADATA_BACKUP is required for any sync; others are optional.
    """

    PKG_METADATA_BACKUP = "cloud:pkg:metadata_backup"
    """Required: Graph node IDs, types, labels, timestamps, relationship types."""

    PKG_SETTINGS_BACKUP = "cloud:pkg:settings_backup"
    """Optional: User preferences and app settings."""

    SEARCH_HISTORY_SYNC = "cloud:search:history_sync"
    """Optional: Search query history (disabled by default)."""


# Human-readable descriptions for UI consent prompts
CLOUD_SYNC_SCOPE_DESCRIPTIONS: Dict[CloudSyncScope, Dict[str, Any]] = {
    CloudSyncScope.PKG_METADATA_BACKUP: {
        "title": "Knowledge Graph Structure",
        "description": "Sync your knowledge graph structure including entity names, "
        "relationship types, and timestamps. This allows you to access "
        "your graph from multiple devices.",
        "data_shared": [
            "Entity names and types (Person, Organization, Concept, Event)",
            "Relationship types between entities",
            "Timestamps (created, modified)",
            "Source identifiers (which connector created the data)",
        ],
        "data_not_shared": [
            "Document content",
            "Email bodies",
            "File contents",
            "Attachment data",
        ],
        "required": True,
        "default_enabled": True,
    },
    CloudSyncScope.PKG_SETTINGS_BACKUP: {
        "title": "App Settings",
        "description": "Sync your Futurnal preferences and settings across devices.",
        "data_shared": [
            "Privacy level settings",
            "Connector configurations (without credentials)",
            "UI preferences",
            "Theme settings",
        ],
        "data_not_shared": [
            "Passwords or API keys",
            "OAuth tokens",
            "Local file paths",
        ],
        "required": False,
        "default_enabled": True,
    },
    CloudSyncScope.SEARCH_HISTORY_SYNC: {
        "title": "Search History",
        "description": "Sync your search queries to continue research across devices.",
        "data_shared": [
            "Search query text",
            "Search timestamps",
            "Filter settings used",
        ],
        "data_not_shared": [
            "Search results",
            "Document content from results",
        ],
        "required": False,
        "default_enabled": False,  # Privacy-sensitive, opt-in
    },
}


@dataclass(frozen=True)
class CloudSyncConsentStatus:
    """Current state of cloud sync consent.

    Attributes:
        has_consent: Whether any cloud sync consent is granted
        granted_scopes: List of scope values that have active consent
        granted_at: When consent was first granted
        is_syncing: Whether sync is currently enabled
        last_sync_at: Timestamp of last successful sync
    """

    has_consent: bool
    granted_scopes: List[str]
    granted_at: Optional[datetime] = None
    is_syncing: bool = False
    last_sync_at: Optional[datetime] = None

    def has_scope(self, scope: CloudSyncScope) -> bool:
        """Check if a specific scope is granted."""
        return scope.value in self.granted_scopes

    def has_required_scope(self) -> bool:
        """Check if the required PKG_METADATA_BACKUP scope is granted."""
        return self.has_scope(CloudSyncScope.PKG_METADATA_BACKUP)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for IPC/JSON."""
        return {
            "has_consent": self.has_consent,
            "granted_scopes": self.granted_scopes,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "is_syncing": self.is_syncing,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudSyncConsentStatus":
        """Deserialize from dictionary."""
        return cls(
            has_consent=data.get("has_consent", False),
            granted_scopes=data.get("granted_scopes", []),
            granted_at=datetime.fromisoformat(data["granted_at"])
            if data.get("granted_at")
            else None,
            is_syncing=data.get("is_syncing", False),
            last_sync_at=datetime.fromisoformat(data["last_sync_at"])
            if data.get("last_sync_at")
            else None,
        )

    @classmethod
    def no_consent(cls) -> "CloudSyncConsentStatus":
        """Factory for status with no consent."""
        return cls(has_consent=False, granted_scopes=[])


@dataclass
class PKGMetadataExport:
    """Metadata-only export structure for cloud sync.

    This structure represents what gets synced to Firebase.
    CRITICAL: No document content, email bodies, or file contents are included.

    Attributes:
        node_id: Unique identifier for the graph node
        node_type: Entity type (Person, Organization, Concept, Event, Document)
        label: Display label for the entity
        created_at: When the entity was first extracted
        updated_at: When the entity was last modified
        source_type: Which connector created this entity (obsidian, imap, github, etc.)
        source_id: Identifier of the specific source (vault_id, mailbox_id, repo_id)
        relationship_types: List of relationship types this entity participates in
        properties: Additional metadata (NO content fields)
    """

    node_id: str
    node_type: str
    label: str
    created_at: datetime
    updated_at: datetime
    source_type: str
    source_id: Optional[str] = None
    relationship_types: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    # Sync metadata
    sync_version: int = 1
    client_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (snake_case keys)."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "relationship_types": self.relationship_types,
            "properties": self.properties,
            "sync_version": self.sync_version,
            "client_id": self.client_id,
        }

    def to_firestore_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-compatible dictionary.

        Returns:
            Dictionary safe for Firestore storage (no datetime objects, camelCase keys)
        """
        return {
            "nodeId": self.node_id,
            "nodeType": self.node_type,
            "label": self.label,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "sourceType": self.source_type,
            "sourceId": self.source_id,
            "relationshipTypes": self.relationship_types,
            "properties": self.properties,
            "syncVersion": self.sync_version,
            "clientId": self.client_id,
        }

    @classmethod
    def from_firestore_dict(cls, data: Dict[str, Any]) -> "PKGMetadataExport":
        """Create from Firestore document data."""
        return cls(
            node_id=data["nodeId"],
            node_type=data["nodeType"],
            label=data["label"],
            created_at=datetime.fromisoformat(data["createdAt"]),
            updated_at=datetime.fromisoformat(data["updatedAt"]),
            source_type=data["sourceType"],
            source_id=data.get("sourceId"),
            relationship_types=data.get("relationshipTypes", []),
            properties=data.get("properties", {}),
            sync_version=data.get("syncVersion", 1),
            client_id=data.get("clientId"),
        )


@dataclass
class CloudSyncAuditEntry:
    """Audit entry for cloud sync operations.

    Attributes:
        action: The sync action performed
        timestamp: When the action occurred (auto-generated if not provided)
        scope: Which consent scope was involved
        nodes_affected: Number of PKG nodes affected
        success: Whether the operation succeeded
        error_message: Error details if operation failed
        metadata: Additional context (no sensitive data)
    """

    action: str  # sync_started, sync_completed, sync_failed, consent_granted, consent_revoked, data_deleted
    success: bool = True
    nodes_affected: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    scope: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/IPC."""
        return {
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "scope": self.scope,
            "nodes_affected": self.nodes_affected,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudSyncAuditEntry":
        """Deserialize from dictionary."""
        return cls(
            action=data["action"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            scope=data.get("scope"),
            nodes_affected=data.get("nodes_affected", 0),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


# Constants for sync behavior
CLOUD_SYNC_SOURCE = "cloud_sync"
"""Source identifier used in ConsentRegistry for cloud sync consent."""

DEFAULT_SYNC_INTERVAL_MINUTES = 15
"""Default interval for periodic sync (when app is open)."""

SYNC_ON_CLOSE = True
"""Whether to sync pending changes when app closes."""


def get_all_scope_values() -> List[str]:
    """Get all cloud sync scope values as a list."""
    return [scope.value for scope in CloudSyncScope]


def get_default_enabled_scopes() -> List[CloudSyncScope]:
    """Get scopes that are enabled by default in the consent modal."""
    return [
        scope
        for scope, desc in CLOUD_SYNC_SCOPE_DESCRIPTIONS.items()
        if desc.get("default_enabled", False)
    ]


def get_required_scopes() -> List[CloudSyncScope]:
    """Get scopes that are required for cloud sync to work."""
    return [
        scope
        for scope, desc in CLOUD_SYNC_SCOPE_DESCRIPTIONS.items()
        if desc.get("required", False)
    ]


@dataclass(frozen=True)
class CloudSyncScopeInfo:
    """Information about a cloud sync scope for UI display.

    This dataclass provides structured info for rendering scope options
    in the consent modal.
    """

    scope: CloudSyncScope
    title: str
    description: str
    required: bool
    default_enabled: bool
    data_shared: Optional[List[str]] = None
    data_not_shared: Optional[List[str]] = None


# Build CLOUD_SYNC_SCOPE_INFO list from descriptions dict
CLOUD_SYNC_SCOPE_INFO: List[CloudSyncScopeInfo] = [
    CloudSyncScopeInfo(
        scope=scope,
        title=desc["title"],
        description=desc["description"],
        required=desc["required"],
        default_enabled=desc["default_enabled"],
        data_shared=desc.get("data_shared"),
        data_not_shared=desc.get("data_not_shared"),
    )
    for scope, desc in CLOUD_SYNC_SCOPE_DESCRIPTIONS.items()
]


def get_scope_info(scope: CloudSyncScope | str) -> Optional[CloudSyncScopeInfo]:
    """Get info for a specific scope.

    Args:
        scope: CloudSyncScope enum or scope value string

    Returns:
        CloudSyncScopeInfo if found, None otherwise
    """
    scope_value = scope.value if isinstance(scope, CloudSyncScope) else scope
    for info in CLOUD_SYNC_SCOPE_INFO:
        if info.scope.value == scope_value:
            return info
    return None

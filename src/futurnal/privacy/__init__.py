"""Privacy utilities including redaction, consent, and audit logging."""

from .audit import AuditEvent, AuditLogger
from .consent import ConsentRecord, ConsentRegistry, ConsentRequiredError
from .redaction import RedactedPath, RedactionPolicy, build_policy, redact_path
from .cloud_consent import (
    CloudSyncScope,
    CloudSyncConsentStatus,
    PKGMetadataExport,
    CloudSyncAuditEntry,
    CLOUD_SYNC_SCOPE_DESCRIPTIONS,
    CLOUD_SYNC_SOURCE,
    DEFAULT_SYNC_INTERVAL_MINUTES,
    get_all_scope_values,
    get_default_enabled_scopes,
    get_required_scopes,
)
from .cloud_sync_manager import CloudSyncConsentManager, create_cloud_sync_manager

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "ConsentRecord",
    "ConsentRegistry",
    "ConsentRequiredError",
    "RedactedPath",
    "RedactionPolicy",
    "build_policy",
    "redact_path",
    # Cloud sync consent
    "CloudSyncScope",
    "CloudSyncConsentStatus",
    "PKGMetadataExport",
    "CloudSyncAuditEntry",
    "CLOUD_SYNC_SCOPE_DESCRIPTIONS",
    "CLOUD_SYNC_SOURCE",
    "DEFAULT_SYNC_INTERVAL_MINUTES",
    "get_all_scope_values",
    "get_default_enabled_scopes",
    "get_required_scopes",
    "CloudSyncConsentManager",
    "create_cloud_sync_manager",
]

"""Obsidian-specific ingestion utilities.

This module provides comprehensive Obsidian vault processing capabilities:
- ObsidianVaultConnector: Production-ready vault connector
- ObsidianDocumentProcessor: Document processing with Unstructured.io bridge
- ObsidianVaultDescriptor: Vault metadata and configuration
- VaultRegistry: Vault discovery and management
"""

from .descriptor import (
    ObsidianVaultDescriptor,
    VaultRegistry,
    DEFAULT_OBSIDIAN_IGNORE_RULES,
    VaultPrivacySettings,
    PrivacyLevel,
    ConsentScope,
)
from .processor import (
    ObsidianDocumentProcessor,
    ProcessedElement,
)
from .connector import (
    ObsidianVaultConnector,
    ObsidianVaultSource,
    ScanResult,
)
from .privacy_policy import (
    ObsidianPrivacyPolicy,
    VaultConsentManager,
)

__all__ = [
    # Descriptor and Registry
    "ObsidianVaultDescriptor",
    "VaultRegistry",
    "DEFAULT_OBSIDIAN_IGNORE_RULES",
    "VaultPrivacySettings",
    "PrivacyLevel",
    "ConsentScope",
    # Processor
    "ObsidianDocumentProcessor",
    "ProcessedElement",
    # Connector
    "ObsidianVaultConnector",
    "ObsidianVaultSource",
    "ScanResult",
    # Privacy
    "ObsidianPrivacyPolicy",
    "VaultConsentManager",
]



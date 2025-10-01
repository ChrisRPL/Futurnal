"""Obsidian-specific ingestion utilities."""

from .descriptor import (
    ObsidianVaultDescriptor,
    VaultRegistry,
    VaultPrivacySettings,
    PrivacyLevel,
    ConsentScope,
    DEFAULT_OBSIDIAN_IGNORE_RULES,
)
from .normalizer import (
    MarkdownNormalizer,
    NormalizedDocument,
    DocumentMetadata,
    ProvenanceInfo,
    ObsidianCallout,
    ObsidianLink,
    ObsidianTag,
    normalize_obsidian_document,
    CalloutType,
    FoldState,
)
from .security import (
    SecurityError,
    PathTraversalValidator,
    ResourceLimiter,
    validate_yaml_safety,
)
from .performance import (
    ContentCache,
    MemoryMonitor,
    ChunkedProcessor,
    PerformanceProfiler,
    get_content_cache,
    get_performance_profiler,
)
from .processor import ObsidianDocumentProcessor
from .connector import ObsidianVaultConnector, ObsidianVaultSource
from .privacy_policy import (
    ObsidianPrivacyPolicy,
    VaultConsentManager,
)

__all__ = [
    # Descriptor and registry
    "ObsidianVaultDescriptor",
    "VaultRegistry",
    "VaultPrivacySettings",
    "PrivacyLevel",
    "ConsentScope",
    "DEFAULT_OBSIDIAN_IGNORE_RULES",
    # Normalizer core
    "MarkdownNormalizer",
    "NormalizedDocument",
    "DocumentMetadata",
    "ProvenanceInfo",
    "ObsidianCallout",
    "ObsidianLink", 
    "ObsidianTag",
    "normalize_obsidian_document",
    "CalloutType",
    "FoldState",
    # Security
    "SecurityError",
    "PathTraversalValidator",
    "ResourceLimiter", 
    "validate_yaml_safety",
    # Performance
    "ContentCache",
    "MemoryMonitor",
    "ChunkedProcessor",
    "PerformanceProfiler",
    "get_content_cache",
    "get_performance_profiler",
    # Integration
    "ObsidianDocumentProcessor",
    "ObsidianVaultConnector",
    "ObsidianVaultSource",
    # Privacy
    "ObsidianPrivacyPolicy",
    "VaultConsentManager",
]



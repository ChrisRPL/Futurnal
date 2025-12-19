"""Obsidian Vault Connector - Production-ready connector for Obsidian vaults.

This module provides the main connector interface for processing Obsidian vaults,
integrating with the MarkdownNormalizer, privacy framework, and PKG storage.

Research Foundation:
- ProPerSim: Personal knowledge preservation
- GFM-RAG: Document processing with metadata preservation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Set, TYPE_CHECKING

from .normalizer import MarkdownNormalizer, normalize_obsidian_document
from .processor import ObsidianDocumentProcessor
from .descriptor import ObsidianVaultDescriptor, VaultRegistry, DEFAULT_OBSIDIAN_IGNORE_RULES
from .privacy_policy import ObsidianPrivacyPolicy, VaultConsentManager

if TYPE_CHECKING:
    from futurnal.ingestion.local.state import StateStore, FileRecord
    from futurnal.privacy.audit import AuditLogger
    from .sync_metrics import SyncMetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ObsidianVaultSource:
    """Configuration for an Obsidian vault source."""

    vault_path: Path
    vault_id: str
    vault_name: Optional[str] = None
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.md"])
    exclude_patterns: List[str] = field(default_factory=list)
    enable_link_graph: bool = True
    enable_asset_processing: bool = False
    privacy_policy: Optional[ObsidianPrivacyPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        self.vault_path = Path(self.vault_path)
        if not self.vault_name:
            self.vault_name = self.vault_path.name

        # Apply default ignore rules if no excludes specified
        if not self.exclude_patterns:
            self.exclude_patterns = list(DEFAULT_OBSIDIAN_IGNORE_RULES)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vault_path": str(self.vault_path),
            "vault_id": self.vault_id,
            "vault_name": self.vault_name,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "enable_link_graph": self.enable_link_graph,
            "enable_asset_processing": self.enable_asset_processing,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ScanResult:
    """Result of scanning a single file."""

    path: Path
    sha256: str
    size: int
    mtime: float
    is_new: bool = False
    is_modified: bool = False
    is_deleted: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
            "mtime": self.mtime,
            "is_new": self.is_new,
            "is_modified": self.is_modified,
            "is_deleted": self.is_deleted,
            "error": self.error,
        }


class ObsidianVaultConnector:
    """Connector for processing Obsidian vaults with full privacy and audit support.

    This connector handles:
    - Vault scanning with change detection
    - Document processing through MarkdownNormalizer
    - Link graph construction
    - Privacy policy enforcement
    - Audit logging

    Example:
        >>> connector = ObsidianVaultConnector(
        ...     workspace_dir=Path("/workspace"),
        ...     state_store=state_store,
        ... )
        >>> source = ObsidianVaultSource(
        ...     vault_path=Path("/vault"),
        ...     vault_id="my-vault",
        ... )
        >>> async for result in connector.scan(source):
        ...     print(result.path)
    """

    def __init__(
        self,
        workspace_dir: Path,
        state_store: "StateStore",
        vault_registry: Optional[VaultRegistry] = None,
        enable_link_graph: bool = True,
        audit_logger: Optional["AuditLogger"] = None,
        metrics_collector: Optional["SyncMetricsCollector"] = None,
    ):
        """Initialize the Obsidian vault connector.

        Args:
            workspace_dir: Directory for workspace files
            state_store: State store for change detection
            vault_registry: Optional vault registry
            enable_link_graph: Whether to enable link graph construction
            audit_logger: Optional audit logger for privacy compliance
            metrics_collector: Optional metrics collector
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.state_store = state_store
        self.vault_registry = vault_registry or VaultRegistry()
        self.enable_link_graph = enable_link_graph
        self.audit_logger = audit_logger
        self.metrics_collector = metrics_collector

        # Processors cache (created per vault)
        self._processors: Dict[str, ObsidianDocumentProcessor] = {}

        # Statistics
        self.files_scanned = 0
        self.files_new = 0
        self.files_modified = 0
        self.files_deleted = 0
        self.errors = 0

    async def scan(
        self,
        source: ObsidianVaultSource,
    ) -> AsyncIterator[ScanResult]:
        """Scan an Obsidian vault for files to process.

        Args:
            source: Vault source configuration

        Yields:
            ScanResult for each file found
        """
        vault_path = source.vault_path

        # Validate vault exists
        if not vault_path.exists():
            logger.error(f"Vault path does not exist: {vault_path}")
            return

        if not vault_path.is_dir():
            logger.error(f"Vault path is not a directory: {vault_path}")
            return

        # Check privacy consent if policy exists
        if source.privacy_policy:
            consent_manager = VaultConsentManager(source.privacy_policy)
            if consent_manager.require_consent_for_scan():
                if not consent_manager.has_consent("scan"):
                    logger.warning(f"Scan consent not granted for vault: {source.vault_id}")
                    return

        # Log audit event
        if self.audit_logger:
            await self._log_audit_event(
                "vault_scan_started",
                source.vault_id,
                {"vault_path": str(vault_path)},
            )

        # Get existing file records for change detection
        existing_paths: Set[Path] = set()

        # Scan for markdown files
        for pattern in source.include_patterns:
            for file_path in vault_path.glob(pattern):
                # Skip excluded patterns
                if self._should_exclude(file_path, source.exclude_patterns, vault_path):
                    continue

                existing_paths.add(file_path)
                result = await self._process_file(file_path, source)
                yield result
                self.files_scanned += 1

        # Detect deleted files
        async for deleted in self._detect_deleted_files(existing_paths, source):
            yield deleted

        # Log completion
        if self.audit_logger:
            await self._log_audit_event(
                "vault_scan_completed",
                source.vault_id,
                {
                    "files_scanned": self.files_scanned,
                    "files_new": self.files_new,
                    "files_modified": self.files_modified,
                    "files_deleted": self.files_deleted,
                },
            )

    def scan_sync(
        self,
        source: ObsidianVaultSource,
    ) -> Iterator[ScanResult]:
        """Synchronous version of scan for non-async contexts.

        Args:
            source: Vault source configuration

        Yields:
            ScanResult for each file found
        """
        vault_path = source.vault_path

        if not vault_path.exists() or not vault_path.is_dir():
            logger.error(f"Invalid vault path: {vault_path}")
            return

        existing_paths: Set[Path] = set()

        for pattern in source.include_patterns:
            for file_path in vault_path.glob(pattern):
                if self._should_exclude(file_path, source.exclude_patterns, vault_path):
                    continue

                existing_paths.add(file_path)
                result = self._process_file_sync(file_path, source)
                yield result
                self.files_scanned += 1

    async def _process_file(
        self,
        file_path: Path,
        source: ObsidianVaultSource,
    ) -> ScanResult:
        """Process a single file asynchronously."""
        return self._process_file_sync(file_path, source)

    def _process_file_sync(
        self,
        file_path: Path,
        source: ObsidianVaultSource,
    ) -> ScanResult:
        """Process a single file synchronously."""
        from futurnal.ingestion.local.state import compute_sha256

        try:
            stats = file_path.stat()
            sha256 = compute_sha256(file_path)

            # Check for existing record
            existing = self.state_store.fetch(file_path)

            is_new = existing is None
            is_modified = existing is not None and existing.sha256 != sha256

            if is_new:
                self.files_new += 1
            elif is_modified:
                self.files_modified += 1

            return ScanResult(
                path=file_path,
                sha256=sha256,
                size=stats.st_size,
                mtime=stats.st_mtime,
                is_new=is_new,
                is_modified=is_modified,
            )

        except Exception as e:
            self.errors += 1
            logger.error(f"Error processing file {file_path}: {e}")
            return ScanResult(
                path=file_path,
                sha256="",
                size=0,
                mtime=0,
                error=str(e),
            )

    async def _detect_deleted_files(
        self,
        existing_paths: Set[Path],
        source: ObsidianVaultSource,
    ) -> AsyncIterator[ScanResult]:
        """Detect files that were deleted from the vault."""
        # Get all tracked files for this vault
        for record in self.state_store.iter_all():
            record_path = Path(record.path)

            # Check if this file belongs to the current vault
            try:
                record_path.relative_to(source.vault_path)
            except ValueError:
                continue  # Not in this vault

            # Check if file still exists
            if record_path not in existing_paths and not record_path.exists():
                self.files_deleted += 1
                yield ScanResult(
                    path=record_path,
                    sha256=record.sha256,
                    size=record.size,
                    mtime=record.mtime,
                    is_deleted=True,
                )

    def _should_exclude(
        self,
        file_path: Path,
        exclude_patterns: List[str],
        vault_root: Path,
    ) -> bool:
        """Check if a file should be excluded based on patterns."""
        try:
            relative_path = file_path.relative_to(vault_root)
        except ValueError:
            return True  # Outside vault root

        path_str = str(relative_path)

        for pattern in exclude_patterns:
            # Handle glob patterns
            if "*" in pattern:
                from fnmatch import fnmatch
                if fnmatch(path_str, pattern):
                    return True
            # Handle directory patterns
            elif pattern.endswith("/"):
                if path_str.startswith(pattern[:-1]):
                    return True
            # Handle exact matches
            elif pattern in path_str:
                return True

        return False

    def get_processor(self, source: ObsidianVaultSource) -> ObsidianDocumentProcessor:
        """Get or create a document processor for a vault.

        Args:
            source: Vault source configuration

        Returns:
            ObsidianDocumentProcessor instance
        """
        if source.vault_id not in self._processors:
            processor = ObsidianDocumentProcessor(
                workspace_dir=self.workspace_dir / source.vault_id,
                vault_root=source.vault_path,
                vault_id=source.vault_id,
                enable_link_graph=source.enable_link_graph and self.enable_link_graph,
                asset_processing_config={
                    "enable_asset_text_extraction": source.enable_asset_processing,
                },
                metrics_collector=self.metrics_collector,
            )
            self._processors[source.vault_id] = processor

        return self._processors[source.vault_id]

    def process_document(
        self,
        file_record: "FileRecord",
        source: ObsidianVaultSource,
    ) -> Iterator[Dict[str, Any]]:
        """Process a document and yield elements.

        Args:
            file_record: File record to process
            source: Vault source configuration

        Yields:
            Element dictionaries
        """
        processor = self.get_processor(source)
        yield from processor.process_document(file_record, source.vault_id)

    async def _log_audit_event(
        self,
        event_type: str,
        vault_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Log an audit event if audit logger is configured."""
        if self.audit_logger:
            try:
                await self.audit_logger.log(
                    event_type=event_type,
                    source_id=vault_id,
                    details=details,
                )
            except Exception as e:
                logger.warning(f"Failed to log audit event: {e}")

    def get_statistics(self) -> Dict[str, int]:
        """Get connector statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "files_scanned": self.files_scanned,
            "files_new": self.files_new,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "errors": self.errors,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.files_scanned = 0
        self.files_new = 0
        self.files_modified = 0
        self.files_deleted = 0
        self.errors = 0

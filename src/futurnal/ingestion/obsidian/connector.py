"""Obsidian Vault Connector implementation with sync engine integration."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from pydantic import Field

from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.redaction import RedactionPolicy, build_policy, redact_path

from .privacy_policy import ObsidianPrivacyPolicy, VaultConsentManager

from ..local.config import LocalIngestionSource
from ..local.connector import ElementSink, _log_extra
from ..local.scanner import FileSnapshot, detect_deletions, walk_directory
from ..local.state import FileRecord, StateStore

from .descriptor import ObsidianVaultDescriptor, VaultRegistry
from .processor import ObsidianDocumentProcessor
from .path_tracker import ObsidianPathTracker
from .sync_metrics import SyncMetricsCollector

logger = logging.getLogger(__name__)


class ObsidianVaultSource(LocalIngestionSource):
    """Extended ingestion source for Obsidian vaults."""

    vault_id: Optional[str] = None
    vault_name: Optional[str] = None

    # Asset processing configuration
    enable_asset_processing: bool = Field(
        default=True,
        description="Enable processing of embedded assets (images, PDFs)"
    )
    enable_asset_text_extraction: bool = Field(
        default=True,
        description="Enable text extraction from processable assets using Unstructured.io"
    )
    asset_ocr_languages: str = Field(
        default="eng",
        description="Language codes for OCR processing (e.g., 'eng', 'fra', 'eng+fra')"
    )
    asset_max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size in MB for asset processing",
        ge=1,
        le=500
    )
    asset_processing_timeout_seconds: int = Field(
        default=60,
        description="Timeout in seconds for asset processing operations",
        ge=5,
        le=600
    )
    supported_asset_extensions: Optional[List[str]] = Field(
        default=None,
        description="Custom list of supported asset file extensions (defaults to standard set)"
    )
    
    @classmethod
    def from_vault_descriptor(
        cls, 
        descriptor: ObsidianVaultDescriptor,
        **kwargs
    ) -> 'ObsidianVaultSource':
        """Create source from vault descriptor."""
        return cls(
            name=kwargs.get('name', f"obsidian-{descriptor.name}"),
            root_path=descriptor.base_path,
            vault_id=descriptor.id,
            vault_name=descriptor.name,
            include=kwargs.get('include', ["**/*.md"]),
            exclude=kwargs.get('exclude', [
                "**/.obsidian/**",
                "**/node_modules/**", 
                "**/.git/**",
                "**/.DS_Store",
                "**/Thumbs.db"
            ]),
            **{k: v for k, v in kwargs.items() if k not in ['name', 'include', 'exclude']}
        )


class ObsidianVaultConnector:
    """Connector for Obsidian vaults with specialized markdown processing and sync engine integration."""

    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        state_store: StateStore,
        vault_registry: Optional[VaultRegistry] = None,
        element_sink: Optional[ElementSink] = None,
        audit_logger: Optional[AuditLogger] = None,
        consent_registry: Optional[ConsentRegistry] = None,
        asset_processing_config: Optional[Dict[str, any]] = None,
        enable_sync_engine: bool = True,
        sync_engine_config: Optional[Dict[str, any]] = None,
        metrics_collector: Optional[SyncMetricsCollector] = None,
    ) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir = self._workspace_dir / "quarantine"
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)

        self._state_store = state_store
        self._vault_registry = vault_registry or VaultRegistry()
        self._element_sink = element_sink
        self._audit_logger = audit_logger
        self._consent_registry = consent_registry
        self._asset_processing_config = asset_processing_config or {}
        self._metrics_collector = metrics_collector

        # Initialize processor
        self._processor = ObsidianDocumentProcessor(
            workspace_dir=self._workspace_dir,
            asset_processing_config=self._asset_processing_config,
            metrics_collector=self._metrics_collector
        )

        # Path tracker for rename/move detection (initialized per vault)
        self._path_trackers: Dict[str, ObsidianPathTracker] = {}

        # Privacy policies per vault
        self._privacy_policies: Dict[str, ObsidianPrivacyPolicy] = {}
        self._consent_managers: Dict[str, VaultConsentManager] = {}

        # Sync engine integration
        self._enable_sync_engine = enable_sync_engine
        self._sync_engine_config = sync_engine_config or {}
        self._sync_engine = None
        self._change_detectors: Dict[str, any] = {}  # Will be imported lazily

        # File watching integration
        self._file_watcher = None
        self._watching_vaults: Set[str] = set()

    def _get_path_tracker(self, source: ObsidianVaultSource) -> Optional[ObsidianPathTracker]:
        """Get or create a path tracker for the given vault source."""
        if not source.vault_id:
            return None

        if source.vault_id not in self._path_trackers:
            self._path_trackers[source.vault_id] = ObsidianPathTracker(
                vault_id=source.vault_id,
                vault_root=source.root_path,
                state_store=self._state_store
            )

        return self._path_trackers[source.vault_id]

    def _get_privacy_policy(self, source: ObsidianVaultSource) -> Optional[ObsidianPrivacyPolicy]:
        """Get or create a privacy policy for the given vault source."""
        if not source.vault_id:
            return None

        if source.vault_id not in self._privacy_policies:
            # Get vault descriptor to build privacy policy
            vault_descriptor = None
            if self._vault_registry:
                vault_descriptor = self._vault_registry.get(source.vault_id)

            if vault_descriptor:
                self._privacy_policies[source.vault_id] = ObsidianPrivacyPolicy.from_vault_descriptor(
                    vault_descriptor,
                    consent_registry=self._consent_registry
                )
            else:
                # Fallback to basic privacy policy
                from .descriptor import VaultPrivacySettings
                default_settings = VaultPrivacySettings()
                self._privacy_policies[source.vault_id] = ObsidianPrivacyPolicy(
                    vault_id=source.vault_id,
                    privacy_settings=default_settings,
                    consent_registry=self._consent_registry
                )

        return self._privacy_policies[source.vault_id]

    def _get_consent_manager(self, source: ObsidianVaultSource) -> Optional[VaultConsentManager]:
        """Get or create a consent manager for the given vault source."""
        if not source.vault_id:
            return None

        if source.vault_id not in self._consent_managers:
            privacy_policy = self._get_privacy_policy(source)
            if privacy_policy:
                self._consent_managers[source.vault_id] = VaultConsentManager(privacy_policy)

        return self._consent_managers[source.vault_id]

    def _handle_path_changes(self, path_changes: List, source: ObsidianVaultSource, job_id: str) -> None:
        """Handle path changes by updating graph relationships.

        Args:
            path_changes: List of PathChange objects
            source: Vault source
            job_id: Current job ID for logging
        """
        from .path_tracker import PathChange

        # Use vault-specific privacy policy for redaction
        privacy_policy = self._get_privacy_policy(source)
        if privacy_policy:
            policy = privacy_policy.build_redaction_policy()
        else:
            policy = build_policy()

        for change in path_changes:
            try:
                # Log the path change for audit
                self._log_path_change_event(change, source, job_id, policy)

                # If element sink is available, notify it of the path change
                if self._element_sink and hasattr(self._element_sink, 'handle_path_change'):
                    self._element_sink.handle_path_change(change.to_dict())

                logger.debug(
                    f"Processed path change: {change.old_path} -> {change.new_path}",
                    extra=_log_extra(
                        job_id=job_id,
                        source=source.name,
                        old_path=change.old_path,
                        new_path=change.new_path,
                        event="path_change_processed",
                    ),
                )

            except Exception as e:
                logger.error(
                    f"Failed to handle path change {change.old_path} -> {change.new_path}: {e}",
                    extra=_log_extra(
                        job_id=job_id,
                        source=source.name,
                        old_path=change.old_path,
                        new_path=change.new_path,
                        event="path_change_handling_failed",
                        error=str(e),
                    ),
                )

    def _log_path_change_event(self, change, source: ObsidianVaultSource, job_id: str, policy) -> None:
        """Log path change event to audit logger."""
        if self._audit_logger is None:
            return

        from futurnal.orchestrator.audit import AuditEvent

        event_metadata = {
            "vault_id": change.vault_id,
            "old_note_id": change.old_note_id,
            "new_note_id": change.new_note_id,
            "change_type": change.change_type,
            "is_content_change": change.is_content_change(),
            "is_rename_only": change.is_rename_only(),
            "is_move_only": change.is_move_only(),
        }

        event = AuditEvent(
            job_id=job_id,
            source=source.name,
            action="path_change",
            status="detected",
            timestamp=change.detected_at,
            metadata=event_metadata,
        )

        try:
            self._audit_logger.record(event)
        except Exception as e:
            logger.error(f"Failed to log path change audit event: {e}")
    
    def crawl_source(self, source: ObsidianVaultSource, *, job_id: str | None = None) -> List[FileRecord]:
        """Perform a crawl of the Obsidian vault returning updated records."""
        
        pathspec = source.build_pathspec()
        snapshots: List[FileRecord] = []
        current_paths: List[Path] = []
        batch_limit = source.max_files_per_batch or 0
        active_job = job_id or uuid.uuid4().hex

        # Get vault descriptor for enhanced processing
        vault_descriptor = None
        if source.vault_id:
            vault_descriptor = self._vault_registry.get(source.vault_id)

        limit_hit = False
        for snapshot in walk_directory(
            source.root_path,
            include_spec=pathspec,
            follow_symlinks=source.follow_symlinks,
        ):
            if batch_limit > 0 and len(snapshots) >= batch_limit:
                limit_hit = True
                break

            current_paths.append(snapshot.path)
            record = self._process_snapshot(source, snapshot)
            if record is not None:
                snapshots.append(record)

        # Process deletions
        if not limit_hit:
            tracked_paths = {record.path.resolve() for record in self._state_store.iter_all()}
            resolved_current = {path.resolve() for path in current_paths}
            deletions = list(detect_deletions(tracked_paths, resolved_current))
            for deleted_path in deletions:
                logger.info(
                    "Detected file deletion",
                    extra=_log_extra(
                        job_id=active_job,
                        source=source.name,
                        path=deleted_path,
                        event="file_deleted",
                    ),
                )

        # Detect path changes for rename/move tracking
        path_tracker = self._get_path_tracker(source)
        if path_tracker:
            try:
                path_changes = path_tracker.detect_path_changes(snapshots)
                if path_changes:
                    logger.info(
                        f"Detected {len(path_changes)} path changes in vault {source.vault_id}",
                        extra=_log_extra(
                            job_id=active_job,
                            source=source.name,
                            event="path_changes_detected",
                            count=len(path_changes),
                        ),
                    )

                    # Handle graph updates for path changes
                    self._handle_path_changes(path_changes, source, active_job)

            except Exception as e:
                logger.error(
                    f"Failed to detect path changes: {e}",
                    extra=_log_extra(
                        job_id=active_job,
                        source=source.name,
                        event="path_change_detection_failed",
                        error=str(e),
                    ),
                )

        return snapshots
    
    def ingest(self, source: ObsidianVaultSource, *, job_id: str | None = None) -> Iterable[dict]:
        """Ingest documents from Obsidian vault through specialized processing pipeline."""

        active_job_id = job_id or uuid.uuid4().hex

        # Use vault-specific privacy policy for redaction
        privacy_policy = self._get_privacy_policy(source)
        if privacy_policy:
            policy = privacy_policy.build_redaction_policy()
        else:
            policy = build_policy()

        # Get vault descriptor
        vault_descriptor = None
        if source.vault_id:
            vault_descriptor = self._vault_registry.get(source.vault_id)

        # Update processor with vault information and asset configuration
        if vault_descriptor:
            self._processor.vault_root = vault_descriptor.base_path
            self._processor.vault_id = vault_descriptor.id
        else:
            self._processor.vault_root = source.root_path
            self._processor.vault_id = source.vault_id

        # Update asset processing configuration from source
        self._processor.update_asset_config(self._extract_asset_config(source))
        
        records = self.crawl_source(source, job_id=active_job_id)
        
        for record in records:
            # Skip non-markdown files
            if not record.path.suffix.lower() in {'.md', '.markdown'}:
                continue
                
            # Check consent using vault-specific privacy policy
            consent_manager = self._get_consent_manager(source)
            if consent_manager:
                # Check basic vault scan consent
                try:
                    if not consent_manager.require_consent_for_scan():
                        logger.info(
                            "Skipping file due to missing vault scan consent",
                            extra=_log_extra(
                                job_id=active_job_id,
                                source=source.name,
                                record=record,
                                policy=policy,
                                event="vault_scan_consent_denied",
                            ),
                        )
                        continue
                except ConsentRequiredError:
                    logger.info(
                        "Consent required for vault scanning",
                        extra=_log_extra(
                            job_id=active_job_id,
                            source=source.name,
                            record=record,
                            policy=policy,
                            event="vault_scan_consent_required",
                        ),
                    )
                    continue

                # Check content analysis consent for markdown processing
                try:
                    if not consent_manager.require_consent_for_content_analysis():
                        logger.info(
                            "Skipping file due to missing content analysis consent",
                            extra=_log_extra(
                                job_id=active_job_id,
                                source=source.name,
                                record=record,
                                policy=policy,
                                event="content_analysis_consent_denied",
                            ),
                        )
                        continue
                except ConsentRequiredError:
                    logger.info(
                        "Consent required for content analysis",
                        extra=_log_extra(
                            job_id=active_job_id,
                            source=source.name,
                            record=record,
                            policy=policy,
                            event="content_analysis_consent_required",
                        ),
                    )
                    continue
            
            # Process document through Obsidian pipeline
            try:
                for element_data in self._processor.process_document(record, source.name):
                    # Log link graph information if present
                    self._log_processed_document_links(
                        job_id=active_job_id,
                        source=source,
                        record=record,
                        element_data=element_data,
                        policy=policy,
                    )

                    # Send to sink if available
                    if self._element_sink is not None:
                        try:
                            self._element_sink.handle(element_data)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception(
                                "Element sink reported a failure",
                                extra=_log_extra(
                                    job_id=active_job_id,
                                    source=source.name,
                                    record=record,
                                    policy=policy,
                                    event="sink_failed",
                                ),
                            )
                            self._quarantine(record.path, "sink_error", str(exc), source.name, policy)
                            self._log_file_event(
                                job_id=active_job_id,
                                source=source,
                                record=record,
                                action="sink",
                                status="failed",
                                policy=policy,
                                metadata={"detail": str(exc)},
                            )
                            continue
                    
                    yield element_data
                
                # Log successful processing
                self._log_file_event(
                    job_id=active_job_id,
                    source=source,
                    record=record,
                    action="ingest",
                    status="succeeded",
                    policy=policy,
                )
                logger.debug(
                    "Obsidian document ingested successfully",
                    extra=_log_extra(
                        job_id=active_job_id,
                        source=source.name,
                        record=record,
                        policy=policy,
                        event="ingest_succeeded",
                    ),
                )
                
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to process Obsidian document",
                    extra=_log_extra(
                        job_id=active_job_id,
                        source=source.name,
                        record=record,
                        policy=policy,
                        event="process_failed",
                    ),
                )
                self._quarantine(record.path, "process_error", str(exc), source.name, policy)
                self._log_file_event(
                    job_id=active_job_id,
                    source=source,
                    record=record,
                    action="process",
                    status="failed",
                    policy=policy,
                    metadata={"detail": str(exc)},
                )
                continue

    def _extract_asset_config(self, source: ObsidianVaultSource) -> Dict[str, any]:
        """Extract asset processing configuration from vault source."""
        config = {
            "enable_asset_processing": source.enable_asset_processing,
            "enable_asset_text_extraction": source.enable_asset_text_extraction,
            "asset_ocr_languages": source.asset_ocr_languages,
            "asset_max_file_size_mb": source.asset_max_file_size_mb,
            "asset_processing_timeout_seconds": source.asset_processing_timeout_seconds,
        }

        if source.supported_asset_extensions is not None:
            config["supported_asset_extensions"] = set(source.supported_asset_extensions)

        return config

    def _process_snapshot(self, source: ObsidianVaultSource, snapshot: FileSnapshot) -> FileRecord | None:
        """Process a file snapshot into a record, checking for changes."""
        existing = self._state_store.fetch(snapshot.path)
        if existing is not None:
            if existing.mtime == snapshot.mtime and existing.size == snapshot.size:
                return existing
        
        from ..local.state import compute_sha256
        
        try:
            current_hash = compute_sha256(snapshot.path)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.warning(f"Failed to compute hash for {snapshot.path}: {exc}")
            return None
        
        record = FileRecord(
            path=snapshot.path,
            size=snapshot.size,
            mtime=snapshot.mtime,
            sha256=current_hash,
        )
        self._state_store.upsert(record)
        return record
    
    def _quarantine(
        self, 
        path: Path, 
        reason: str, 
        detail: str, 
        source_name: str, 
        policy: RedactionPolicy
    ) -> None:
        """Quarantine a file with detailed error information."""
        try:
            redacted = redact_path(path, policy=policy)
            quarantine_file = self._quarantine_dir / f"{redacted.path_hash}.txt"
            quarantine_info = {
                "original_path": redacted.redacted,
                "path_hash": redacted.path_hash,
                "reason": reason,
                "detail": detail,
                "source": source_name,
                "timestamp": datetime.utcnow().isoformat(),
            }
            quarantine_file.write_text(str(quarantine_info))
            logger.info(
                f"File quarantined due to {reason}",
                extra=_log_extra(
                    source=source_name,
                    path=path,
                    policy=policy,
                    event="quarantined",
                    reason=reason,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to quarantine file {path}: {e}")
    
    def _log_file_event(
        self,
        *,
        job_id: str,
        source: ObsidianVaultSource,
        record: FileRecord,
        action: str,
        status: str,
        policy: RedactionPolicy,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log file processing event to audit logger."""
        if self._audit_logger is None:
            return
            
        from futurnal.orchestrator.audit import AuditEvent
        
        event_metadata = {
            "action": action,
            "status": status,
            "size_bytes": record.size,
            "mtime": record.mtime,
            "sha256": record.sha256[:16],  # Truncated for privacy
        }
        
        if metadata:
            event_metadata.update(metadata)
            
        if source.vault_id:
            event_metadata["vault_id"] = source.vault_id
        if source.vault_name:
            event_metadata["vault_name"] = source.vault_name
        
        redacted = redact_path(record.path, policy=policy)
        event = AuditEvent(
            job_id=job_id,
            source=source.name,
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=event_metadata,
        )
        
        try:
            self._audit_logger.record(event)
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    def _log_link_graph_event(
        self,
        *,
        job_id: str,
        source: ObsidianVaultSource,
        action: str,
        status: str,
        policy: RedactionPolicy,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log link graph events in a privacy-safe manner."""
        if self._audit_logger is None:
            return

        privacy_policy = self._get_privacy_policy(source)
        if privacy_policy and not privacy_policy.privacy_settings.audit_link_changes:
            return  # Link change auditing disabled for this vault

        from futurnal.orchestrator.audit import AuditEvent

        event_metadata = {
            "action": action,
            "status": status,
            "vault_id": source.vault_id,
            "vault_name": source.vault_name,
        }

        if metadata:
            # Filter out any sensitive metadata before logging
            safe_metadata = self._sanitize_link_metadata(metadata, policy)
            event_metadata.update(safe_metadata)

        event = AuditEvent(
            job_id=job_id,
            source=source.name,
            action=f"link_graph_{action}",
            status=status,
            timestamp=datetime.utcnow(),
            metadata=event_metadata,
        )

        try:
            self._audit_logger.record(event)
        except Exception as e:
            logger.error(f"Failed to log link graph audit event: {e}")

    def _sanitize_link_metadata(self, metadata: dict, policy: RedactionPolicy) -> dict:
        """Sanitize link metadata to remove sensitive information."""
        safe_metadata = {}

        for key, value in metadata.items():
            if key in ["link_count", "edge_count", "node_count", "graph_size"]:
                # Numerical statistics are safe
                safe_metadata[key] = value
            elif key in ["link_types", "relationship_types"]:
                # Types are generally safe
                safe_metadata[key] = value
            elif key in ["source_path", "target_path", "file_path"]:
                # Redact file paths
                if isinstance(value, (str, Path)):
                    redacted = policy.apply(Path(value))
                    safe_metadata[f"{key}_hash"] = redacted.path_hash
                    safe_metadata[f"{key}_redacted"] = redacted.redacted
                else:
                    safe_metadata[key] = "redacted"
            elif key in ["note_titles", "link_text", "content"]:
                # Never log content or titles
                if isinstance(value, list):
                    safe_metadata[f"{key}_count"] = len(value)
                else:
                    safe_metadata[f"{key}_present"] = bool(value)
            elif key in ["checksums", "hashes"]:
                # Hashes are safe to log
                safe_metadata[key] = value
            else:
                # For unknown keys, err on the side of caution
                if isinstance(value, (int, float, bool)):
                    safe_metadata[key] = value
                elif isinstance(value, list):
                    safe_metadata[f"{key}_count"] = len(value)
                else:
                    safe_metadata[f"{key}_present"] = bool(value)

        return safe_metadata

    def _log_relationship_change(
        self,
        *,
        job_id: str,
        source: ObsidianVaultSource,
        change_type: str,
        source_file: Path,
        target_file: Optional[Path] = None,
        relationship_type: str = "wikilink",
        policy: RedactionPolicy,
    ) -> None:
        """Log a specific relationship change."""
        metadata = {
            "change_type": change_type,
            "relationship_type": relationship_type,
            "source_path": source_file,
        }

        if target_file:
            metadata["target_path"] = target_file

        self._log_link_graph_event(
            job_id=job_id,
            source=source,
            action="relationship_changed",
            status="detected",
            policy=policy,
            metadata=metadata,
        )

    def _log_processed_document_links(
        self,
        *,
        job_id: str,
        source: ObsidianVaultSource,
        record: 'FileRecord',  # Import type
        element_data: dict,
        policy: RedactionPolicy,
    ) -> None:
        """Log link information from processed documents."""
        try:
            # Extract link information from element data
            metadata = {}

            # Check for wikilinks in metadata
            if 'metadata' in element_data:
                doc_metadata = element_data['metadata']

                if 'obsidian_links' in doc_metadata:
                    links = doc_metadata['obsidian_links']
                    if links:
                        metadata['wikilink_count'] = len(links)
                        metadata['link_types'] = list(set(link.get('type', 'wikilink') for link in links))

                        # Log each significant link (without content)
                        for link in links[:5]:  # Limit to first 5 for auditing
                            if link.get('target'):
                                self._log_relationship_change(
                                    job_id=job_id,
                                    source=source,
                                    change_type="link_detected",
                                    source_file=record.path,
                                    target_file=Path(source.root_path) / link['target'] if link['target'] else None,
                                    relationship_type=link.get('type', 'wikilink'),
                                    policy=policy,
                                )

                if 'obsidian_tags' in doc_metadata:
                    tags = doc_metadata['obsidian_tags']
                    if tags:
                        metadata['tag_count'] = len(tags)

                if 'obsidian_callouts' in doc_metadata:
                    callouts = doc_metadata['obsidian_callouts']
                    if callouts:
                        metadata['callout_count'] = len(callouts)
                        metadata['callout_types'] = list(set(c.get('type', 'unknown') for c in callouts))

            # Check for asset references
            if 'text' in element_data or 'content' in element_data:
                content = element_data.get('text', element_data.get('content', ''))
                if content and isinstance(content, str):
                    # Look for image references (basic detection)
                    import re
                    image_refs = re.findall(r'!\[\[([^\]]+)\]\]|!\[([^\]]*)\]\(([^)]+)\)', content)
                    if image_refs:
                        metadata['image_reference_count'] = len(image_refs)

            if metadata:
                self._log_link_graph_event(
                    job_id=job_id,
                    source=source,
                    action="document_processed",
                    status="success",
                    policy=policy,
                    metadata={
                        **metadata,
                        "source_path": record.path,
                        "file_size": record.size,
                        "checksum": record.sha256[:16],  # Truncated for privacy
                    },
                )

        except Exception as e:
            logger.debug(f"Failed to log document links: {e}")

    def _log_asset_processing_event(
        self,
        *,
        job_id: str,
        source: ObsidianVaultSource,
        asset_path: Path,
        action: str,
        status: str,
        policy: RedactionPolicy,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log asset processing events with privacy protection."""
        if self._audit_logger is None:
            return

        # Check if asset processing should be audited
        consent_manager = self._get_consent_manager(source)
        if consent_manager:
            try:
                if not consent_manager.require_consent_for_asset_extraction():
                    return  # Asset processing not consented
            except ConsentRequiredError:
                return  # No consent for asset processing

        from futurnal.orchestrator.audit import AuditEvent

        # Build safe metadata
        safe_metadata = {
            "action": action,
            "status": status,
            "vault_id": source.vault_id,
            "vault_name": source.vault_name,
        }

        if metadata:
            # Sanitize asset metadata
            for key, value in metadata.items():
                if key in ["file_size", "processing_time_ms", "extracted_text_length"]:
                    safe_metadata[key] = value
                elif key in ["file_type", "mime_type", "processor_used"]:
                    safe_metadata[key] = value
                elif key in ["ocr_language", "extraction_method"]:
                    safe_metadata[key] = value
                elif key == "asset_path":
                    redacted = policy.apply(Path(value))
                    safe_metadata["asset_path_hash"] = redacted.path_hash
                    safe_metadata["asset_path_redacted"] = redacted.redacted
                elif key == "error":
                    safe_metadata["error_message"] = str(value)[:200]  # Truncate errors
                else:
                    # Skip unknown metadata for privacy
                    continue

        # Redact the asset path
        redacted_path = policy.apply(asset_path)

        event = AuditEvent(
            job_id=job_id,
            source=source.name,
            action=f"asset_{action}",
            status=status,
            timestamp=datetime.utcnow(),
            redacted_path=redacted_path.redacted,
            path_hash=redacted_path.path_hash,
            metadata=safe_metadata,
        )

        try:
            self._audit_logger.record(event)
        except Exception as e:
            logger.error(f"Failed to log asset processing audit event: {e}")

    # Sync Engine Integration Methods

    async def initialize_sync_engine(self, job_queue) -> None:
        """Initialize the sync engine for advanced synchronization capabilities.

        Args:
            job_queue: Job queue instance for sync coordination
        """
        if not self._enable_sync_engine or self._sync_engine is not None:
            return

        try:
            # Lazy import to avoid circular dependencies
            from .sync_engine import create_sync_engine

            self._sync_engine = create_sync_engine(
                vault_connector=self,
                job_queue=job_queue,
                state_store=self._state_store,
                audit_logger=self._audit_logger,
                **self._sync_engine_config
            )

            await self._sync_engine.start()
            logger.info("Obsidian sync engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize sync engine: {e}", exc_info=True)
            self._sync_engine = None

    async def shutdown_sync_engine(self) -> None:
        """Shutdown the sync engine and stop file watching."""
        if self._sync_engine:
            await self._sync_engine.stop()
            self._sync_engine = None

        if self._file_watcher:
            await self._file_watcher.stop_watching()
            self._file_watcher = None

        self._watching_vaults.clear()
        logger.info("Obsidian sync engine shutdown completed")

    async def enable_vault_sync(self, vault_source: ObsidianVaultSource) -> bool:
        """Enable sync for a specific vault.

        Args:
            vault_source: Vault source configuration

        Returns:
            True if sync was enabled successfully
        """
        if not self._enable_sync_engine or not vault_source.vault_id:
            return False

        try:
            # Initialize sync engine if needed
            if self._sync_engine is None:
                logger.warning("Sync engine not initialized, cannot enable vault sync")
                return False

            # Initialize change detector for this vault
            change_detector = await self._get_or_create_change_detector(vault_source)
            if not change_detector:
                return False

            # Start file watching for this vault
            await self._start_vault_file_watching(vault_source)

            self._watching_vaults.add(vault_source.vault_id)

            logger.info(f"Enabled sync for vault {vault_source.vault_id}")

            if self._audit_logger:
                await self._log_sync_event(
                    "vault_sync_enabled",
                    "success",
                    vault_id=vault_source.vault_id
                )

            return True

        except Exception as e:
            logger.error(f"Failed to enable sync for vault {vault_source.vault_id}: {e}", exc_info=True)

            if self._audit_logger:
                await self._log_sync_event(
                    "vault_sync_enable_failed",
                    "error",
                    vault_id=vault_source.vault_id,
                    error=str(e)
                )

            return False

    async def disable_vault_sync(self, vault_id: str) -> bool:
        """Disable sync for a specific vault.

        Args:
            vault_id: ID of the vault to stop syncing

        Returns:
            True if sync was disabled successfully
        """
        try:
            if vault_id in self._watching_vaults:
                self._watching_vaults.remove(vault_id)

            # Remove change detector
            self._change_detectors.pop(vault_id, None)

            # Note: File watcher is shared, so we don't stop it here
            # It will filter events based on active vaults

            logger.info(f"Disabled sync for vault {vault_id}")

            if self._audit_logger:
                await self._log_sync_event(
                    "vault_sync_disabled",
                    "success",
                    vault_id=vault_id
                )

            return True

        except Exception as e:
            logger.error(f"Failed to disable sync for vault {vault_id}: {e}", exc_info=True)
            return False

    async def trigger_incremental_sync(self, vault_source: ObsidianVaultSource) -> Optional[str]:
        """Trigger an incremental sync for a vault.

        Args:
            vault_source: Vault source configuration

        Returns:
            Job ID if sync was triggered successfully
        """
        if not self._sync_engine or not vault_source.vault_id:
            logger.warning("Sync engine not available for incremental sync")
            return None

        try:
            # Perform change detection
            change_detector = await self._get_or_create_change_detector(vault_source)
            if not change_detector:
                return None

            # Get current records for change detection
            records = self.crawl_source(vault_source)
            path_changes, content_changes = change_detector.detect_changes(records)

            # Handle path changes through sync engine
            if path_changes:
                await self._sync_engine.handle_path_changes(path_changes, vault_source.vault_id)

            # Handle content changes as file events
            if content_changes:
                for change in content_changes:
                    await self._sync_engine.handle_file_event(
                        event_type=self._map_content_change_to_event_type(change),
                        file_path=change.file_path,
                        vault_id=vault_source.vault_id,
                        metadata={
                            "change_type": change.change_type,
                            "has_content_change": change.has_content_change(),
                            "has_metadata_change": change.has_metadata_change(),
                        }
                    )

            logger.info(
                f"Triggered incremental sync for vault {vault_source.vault_id}: "
                f"{len(path_changes)} path changes, {len(content_changes)} content changes"
            )

            if self._audit_logger:
                await self._log_sync_event(
                    "incremental_sync_triggered",
                    "success",
                    vault_id=vault_source.vault_id,
                    path_changes_count=len(path_changes),
                    content_changes_count=len(content_changes)
                )

            return f"incremental_sync_{uuid.uuid4().hex[:8]}"

        except Exception as e:
            logger.error(f"Failed to trigger incremental sync for vault {vault_source.vault_id}: {e}", exc_info=True)

            if self._audit_logger:
                await self._log_sync_event(
                    "incremental_sync_failed",
                    "error",
                    vault_id=vault_source.vault_id,
                    error=str(e)
                )

            return None

    async def get_sync_status(self, vault_id: str) -> Dict[str, any]:
        """Get sync status for a vault.

        Args:
            vault_id: ID of the vault

        Returns:
            Dictionary containing sync status information
        """
        status = {
            "vault_id": vault_id,
            "sync_enabled": vault_id in self._watching_vaults,
            "sync_engine_available": self._sync_engine is not None,
            "file_watching_active": self._file_watcher is not None,
        }

        if self._sync_engine:
            sync_status = await self._sync_engine.get_sync_status(vault_id)
            status.update(sync_status)

        if vault_id in self._change_detectors:
            change_detector = self._change_detectors[vault_id]
            if hasattr(change_detector, 'get_cache_stats'):
                status["change_detector_stats"] = change_detector.get_cache_stats()

        return status

    async def _get_or_create_change_detector(self, vault_source: ObsidianVaultSource):
        """Get or create a change detector for a vault."""
        if not vault_source.vault_id:
            return None

        if vault_source.vault_id not in self._change_detectors:
            try:
                # Lazy import to avoid circular dependencies
                from .change_detector import create_change_detector

                path_tracker = self._get_path_tracker(vault_source)
                change_detector = create_change_detector(
                    vault_id=vault_source.vault_id,
                    vault_root=vault_source.root_path,
                    state_store=self._state_store,
                    path_tracker=path_tracker
                )

                self._change_detectors[vault_source.vault_id] = change_detector

            except Exception as e:
                logger.error(f"Failed to create change detector for vault {vault_source.vault_id}: {e}")
                return None

        return self._change_detectors[vault_source.vault_id]

    async def _start_vault_file_watching(self, vault_source: ObsidianVaultSource) -> None:
        """Start file watching for a vault."""
        if self._file_watcher is None:
            try:
                # Lazy import to avoid circular dependencies
                from ..orchestrator.file_watcher import create_optimized_watcher, WatcherConfig

                # Create watcher config optimized for Obsidian
                config = WatcherConfig(
                    include_patterns=["**/*.md", "**/*.markdown", "**/*.png", "**/*.jpg", "**/*.pdf"],
                    exclude_patterns=[
                        "**/.obsidian/**",
                        "**/.trash/**",
                        "**/.git/**",
                        "**/node_modules/**",
                    ],
                    enable_large_vault_mode=True,
                    large_vault_threshold=1000,
                )

                self._file_watcher = create_optimized_watcher(
                    config=config,
                    event_callback=self._handle_file_events
                )

            except Exception as e:
                logger.error(f"Failed to create file watcher: {e}")
                return

        # Start watching the vault path
        if not self._file_watcher.is_watching():
            await self._file_watcher.start_watching(vault_source.root_path)

    def _handle_file_events(self, events) -> None:
        """Handle file events from the file watcher."""
        if not self._sync_engine:
            return

        for event in events:
            # Determine which vault this event belongs to
            vault_id = self._find_vault_for_path(event.path)
            if not vault_id or vault_id not in self._watching_vaults:
                continue

            # Convert file watcher event to sync engine event
            try:
                from .sync_engine import SyncEventType

                event_type_map = {
                    'created': SyncEventType.FILE_CREATED,
                    'modified': SyncEventType.FILE_MODIFIED,
                    'deleted': SyncEventType.FILE_DELETED,
                    'moved': SyncEventType.FILE_MOVED,
                }

                sync_event_type = event_type_map.get(event.event_type.value)
                if sync_event_type:
                    # Create async task to handle the event
                    asyncio.create_task(self._sync_engine.handle_file_event(
                        event_type=sync_event_type,
                        file_path=event.path,
                        vault_id=vault_id,
                        metadata=event.metadata
                    ))

            except Exception as e:
                logger.error(f"Failed to handle file event: {e}")

    def _find_vault_for_path(self, file_path: Path) -> Optional[str]:
        """Find which vault a file path belongs to."""
        # Simple implementation - could be optimized with a lookup table
        for vault_descriptor in self._vault_registry.list_vaults():
            try:
                file_path.relative_to(vault_descriptor.base_path)
                return vault_descriptor.id
            except ValueError:
                continue
        return None

    def _map_content_change_to_event_type(self, content_change):
        """Map content change to sync event type."""
        from .sync_engine import SyncEventType

        if content_change.has_content_change():
            return SyncEventType.FILE_MODIFIED
        else:
            return SyncEventType.FILE_MODIFIED  # Metadata change is still a modification

    async def _log_sync_event(self, action: str, status: str, **metadata) -> None:
        """Log sync-related audit event."""
        if not self._audit_logger:
            return

        from futurnal.orchestrator.audit import AuditEvent

        event = AuditEvent(
            job_id="sync_connector",
            source="obsidian_vault_connector",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )

        try:
            self._audit_logger.record(event)
        except Exception as e:
            logger.error(f"Failed to log sync audit event: {e}")

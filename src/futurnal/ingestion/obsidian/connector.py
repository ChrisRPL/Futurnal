"""Obsidian Vault Connector implementation."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.redaction import RedactionPolicy, build_policy, redact_path

from ..local.config import LocalIngestionSource
from ..local.connector import ElementSink, _log_extra
from ..local.scanner import FileSnapshot, detect_deletions, walk_directory
from ..local.state import FileRecord, StateStore

from .descriptor import ObsidianVaultDescriptor, VaultRegistry
from .processor import ObsidianDocumentProcessor

logger = logging.getLogger(__name__)


class ObsidianVaultSource(LocalIngestionSource):
    """Extended ingestion source for Obsidian vaults."""
    
    vault_id: Optional[str] = None
    vault_name: Optional[str] = None
    
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
    """Connector for Obsidian vaults with specialized markdown processing."""
    
    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        state_store: StateStore,
        vault_registry: Optional[VaultRegistry] = None,
        element_sink: Optional[ElementSink] = None,
        audit_logger: Optional[AuditLogger] = None,
        consent_registry: Optional[ConsentRegistry] = None,
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
        
        # Initialize processor
        self._processor = ObsidianDocumentProcessor(workspace_dir=self._workspace_dir)
    
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

        return snapshots
    
    def ingest(self, source: ObsidianVaultSource, *, job_id: str | None = None) -> Iterable[dict]:
        """Ingest documents from Obsidian vault through specialized processing pipeline."""
        
        active_job_id = job_id or uuid.uuid4().hex
        policy = build_policy()
        
        # Get vault descriptor
        vault_descriptor = None
        if source.vault_id:
            vault_descriptor = self._vault_registry.get(source.vault_id)
        
        # Update processor with vault information
        if vault_descriptor:
            self._processor.vault_root = vault_descriptor.base_path
            self._processor.vault_id = vault_descriptor.id
        else:
            self._processor.vault_root = source.root_path
            self._processor.vault_id = source.vault_id
        
        records = self.crawl_source(source, job_id=active_job_id)
        
        for record in records:
            # Skip non-markdown files
            if not record.path.suffix.lower() in {'.md', '.markdown'}:
                continue
                
            # Check consent for processing if required
            if source.require_external_processing_consent:
                if self._consent_registry is None:
                    logger.warning(
                        "Consent required but no registry available",
                        extra=_log_extra(
                            job_id=active_job_id,
                            source=source.name,
                            record=record,
                            policy=policy,
                            event="consent_unavailable",
                        ),
                    )
                    continue
                    
                try:
                    consent_granted = self._consent_registry.check_consent(
                        scope=source.external_processing_scope,
                        resource=str(record.path),
                    )
                    if not consent_granted:
                        logger.info(
                            "Skipping file due to missing consent",
                            extra=_log_extra(
                                job_id=active_job_id,
                                source=source.name,
                                record=record,
                                policy=policy,
                                event="consent_denied",
                            ),
                        )
                        continue
                except ConsentRequiredError:
                    logger.info(
                        "Consent required for external processing",
                        extra=_log_extra(
                            job_id=active_job_id,
                            source=source.name,
                            record=record,
                            policy=policy,
                            event="consent_required",
                        ),
                    )
                    continue
            
            # Process document through Obsidian pipeline
            try:
                for element_data in self._processor.process_document(record, source.name):
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

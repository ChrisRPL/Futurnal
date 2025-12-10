"""Local Files Connector - Grounds Ghost in user's file-based experiential data.

This connector enables the Ghost to learn from the user's local filesystem,
which represents their document universe and file organization patterns. By
processing diverse file types with temporal awareness, the Ghost develops
understanding of:
- How the user organizes documents (via directory structure, naming patterns)
- What content types matter to the user (via file type distributions)
- How the document landscape evolves (via file creation/modification patterns)

The connector transforms local directories into structured experiential memory
within the PKG, providing the Ghost with high-fidelity recall of the user's
file-based knowledge. This forms the foundational layer of Phase 1 (Archivist)
experiential memory construction.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Protocol, runtime_checkable

from unstructured.partition.auto import partition

from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.redaction import RedactionPolicy, build_policy, redact_path

from .config import LocalIngestionSource
from .scanner import FileSnapshot, detect_deletions, walk_directory
from .state import FileRecord, StateStore, compute_sha256


def _log_extra(
    *,
    job_id: str | None = None,
    source: str | None = None,
    record: FileRecord | None = None,
    path: Path | None = None,
    policy: RedactionPolicy | None = None,
    **metadata: object,
) -> dict[str, object]:
    """Build structured logging context with privacy-aware path handling."""

    extra: dict[str, object] = {}
    if job_id:
        extra["ingestion_job_id"] = job_id
    if source:
        extra["ingestion_source"] = source
    if record:
        extra["ingestion_sha256"] = record.sha256
        extra["ingestion_size_bytes"] = record.size
        extra["ingestion_mtime"] = record.mtime
        if path is None:
            path = record.path
    if path is not None:
        active_policy = policy or build_policy()
        redacted = redact_path(path, policy=active_policy)
        extra["ingestion_path"] = redacted.redacted
        extra["ingestion_path_hash"] = redacted.path_hash
    for key, value in metadata.items():
        if value is not None:
            extra[f"ingestion_{key}"] = value
    return extra


@runtime_checkable
class ElementSink(Protocol):
    """Sink interface for handling parsed elements."""

    def handle(self, element: dict) -> None:
        ...

    def handle_deletion(self, element: dict) -> None:  # pragma: no cover - optional
        ...


logger = logging.getLogger(__name__)


class LocalFilesConnector:
    """Connector responsible for ingesting local files into Futurnal pipelines."""

    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        state_store: StateStore,
        element_sink: ElementSink | None = None,
        audit_logger: AuditLogger | None = None,
        consent_registry: ConsentRegistry | None = None,
    ) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._parsed_dir = self._workspace_dir / "parsed"
        self._parsed_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir = self._workspace_dir / "quarantine"
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._state_store = state_store
        self._element_sink = element_sink
        self._audit_logger = audit_logger
        self._consent_registry = consent_registry

    def crawl_source(self, source: LocalIngestionSource, *, job_id: str | None = None) -> List[FileRecord]:
        """Perform a crawl of the provided source returning updated records.

        The crawl honours ignore patterns, batches work per `max_files_per_batch`,
        and tracks which paths were seen so deletions can be processed after the
        scan completes.
        """

        pathspec = source.build_pathspec()
        snapshots: List[FileRecord] = []
        current_paths: List[Path] = []
        batch_limit = source.max_files_per_batch or 0
        active_job = job_id or uuid.uuid4().hex

        limit_hit = False
        for snapshot in walk_directory(
            source.root_path,
            include_spec=pathspec,
            follow_symlinks=source.follow_symlinks,
        ):
            current_paths.append(snapshot.path)
            if limit_hit:
                continue
            record = self._process_snapshot(source, snapshot)
            if record:
                snapshots.append(record)
                if batch_limit and len(snapshots) >= batch_limit:
                    limit_hit = True
        if not limit_hit:
            self._handle_deletions(source, current_paths, job_id=active_job)
        else:
            logger.debug(
                "Batch limit reached for source; remaining files will run later",
                extra=_log_extra(
                    job_id=active_job,
                    source=source.name,
                    batch_limit=batch_limit,
                    event="batch_limit_reached",
                ),
            )
        return snapshots

    def ingest(self, source: LocalIngestionSource, *, job_id: str | None = None) -> Iterable[dict]:
        """Yield parsed document elements for the given source."""

        active_job_id = job_id or uuid.uuid4().hex
        policy = build_policy(allow_plaintext=source.allow_plaintext_paths)

        for record in self.crawl_source(source, job_id=active_job_id):
            logger.debug(
                "Parsing file for ingestion",
                extra=_log_extra(
                    job_id=active_job_id,
                    source=source.name,
                    record=record,
                    policy=policy,
                    event="parse_start",
                ),
            )
            if source.require_external_processing_consent and self._consent_registry:
                try:
                    self._consent_registry.require(
                        source=source.name, scope=source.external_processing_scope
                    )
                except ConsentRequiredError as exc:
                    logger.error(
                        "Consent missing before external processing",
                        extra=_log_extra(
                            job_id=active_job_id,
                            source=source.name,
                            record=record,
                            policy=policy,
                            event="consent_missing",
                            scope=source.external_processing_scope,
                        ),
                        exc_info=exc,
                    )
                    self._log_file_event(
                        job_id=active_job_id,
                        source=source,
                        record=record,
                        action="external_processing",
                        status="blocked",
                        policy=policy,
                        metadata={"reason": "consent_required"},
                    )
                    continue
            try:
                elements = partition(
                    filename=str(record.path), strategy="fast", include_metadata=True
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Partition failed during ingestion",
                    extra=_log_extra(
                        job_id=active_job_id,
                        source=source.name,
                        record=record,
                        policy=policy,
                        event="partition_failed",
                    ),
                )
                self._quarantine(record.path, "partition_error", str(exc), source.name, policy)
                self._log_file_event(
                    job_id=active_job_id,
                    source=source,
                    record=record,
                    action="partition",
                    status="failed",
                    policy=policy,
                    metadata={"detail": str(exc)},
                )
                continue
            had_failure = False
            for element in elements:
                try:
                    parsed = self._persist_element(source, record, element)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Persist failed during ingestion",
                        extra=_log_extra(
                            job_id=active_job_id,
                            source=source.name,
                            record=record,
                            policy=policy,
                            event="persist_failed",
                        ),
                    )
                    self._quarantine(record.path, "persist_error", str(exc), source.name, policy)
                    self._log_file_event(
                        job_id=active_job_id,
                        source=source,
                        record=record,
                        action="persist",
                        status="failed",
                        policy=policy,
                        metadata={"detail": str(exc)},
                    )
                    had_failure = True
                    continue
                if self._element_sink is not None:
                    try:
                        self._element_sink.handle(parsed)
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
                        had_failure = True
                        continue
                yield parsed
            if not had_failure:
                self._log_file_event(
                    job_id=active_job_id,
                    source=source,
                    record=record,
                    action="ingest",
                    status="succeeded",
                    policy=policy,
                )
                logger.debug(
                    "File ingested successfully",
                    extra=_log_extra(
                        job_id=active_job_id,
                        source=source.name,
                        record=record,
                        policy=policy,
                        event="ingest_succeeded",
                    ),
                )

    def _persist_element(self, source: LocalIngestionSource, record: FileRecord, element) -> dict:
        storage_path = self._parsed_dir / f"{record.sha256}.json"
        if hasattr(element, "to_dict"):
            payload = element.to_dict()
        elif isinstance(element, dict):
            payload = element
        else:
            payload = {"text": str(element)}

        payload.setdefault("metadata", {})
        payload["metadata"].update(
            {
                "source": source.name,
                "path": str(record.path),
                "sha256": record.sha256,
                "size_bytes": record.size,
                "modified_at": datetime.utcfromtimestamp(record.mtime).isoformat(),
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )

        storage_path.write_text(json.dumps(payload, ensure_ascii=False))
        return {
            "source": source.name,
            "path": str(record.path),
            "sha256": record.sha256,
            "element_path": str(storage_path),
            "size_bytes": record.size,
        }

    def _process_snapshot(self, source: LocalIngestionSource, snapshot: FileSnapshot) -> FileRecord | None:
        policy = build_policy(allow_plaintext=source.allow_plaintext_paths)
        try:
            current_hash = compute_sha256(snapshot.path)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.warning(
                "Skipping snapshot due to hash error",
                extra=_log_extra(
                    source=source.name,
                    path=snapshot.path,
                    policy=policy,
                    event="hash_error",
                    error=str(exc),
                ),
            )
            self._quarantine(snapshot.path, "hash_error", str(exc), source.name, policy)
            return None
        existing = self._state_store.fetch(snapshot.path)
        if existing and existing.sha256 == current_hash and existing.mtime == snapshot.mtime:
            logger.debug(
                "Skipping unchanged file",
                extra=_log_extra(
                    source=source.name,
                    path=snapshot.path,
                    policy=policy,
                    event="unchanged",
                ),
            )
            return None

        record = FileRecord(
            path=snapshot.path,
            size=snapshot.size,
            mtime=snapshot.mtime,
            sha256=current_hash,
        )
        self._state_store.upsert(record)
        return record

    def _handle_deletions(
        self, source: LocalIngestionSource, current_paths: Iterable[Path], *, job_id: str
    ) -> None:
        # Only track files under this source's root to avoid cross-source deletion
        source_root = Path(source.root_path).resolve()
        tracked = {
            record.path.resolve(): record
            for record in self._state_store.iter_all()
            if record.path.resolve().is_relative_to(source_root)
        }
        removed = detect_deletions(tracked.keys(), current_paths)
        policy = build_policy(allow_plaintext=source.allow_plaintext_paths)
        for path in removed:
            record = tracked[path]
            logger.debug(
                "Removing deleted file",
                extra=_log_extra(
                    job_id=job_id,
                    source=source.name,
                    record=record,
                    policy=policy,
                    event="delete",
                ),
            )
            self._state_store.remove(record.path)
            self._notify_sink_of_deletion(source, record)
            self._cleanup_parsed(record.sha256)
            self._log_file_event(
                job_id=job_id,
                source=source,
                record=record,
                action="delete",
                status="succeeded",
                policy=policy,
            )

    def _notify_sink_of_deletion(self, source: LocalIngestionSource, record: FileRecord) -> None:
        if self._element_sink and hasattr(self._element_sink, "handle_deletion"):
            try:
                self._element_sink.handle_deletion(
                    {
                        "source": source.name,
                        "path": str(record.path),
                        "sha256": record.sha256,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Element sink deletion callback failed",
                    extra=_log_extra(
                        source=source.name,
                        record=record,
                        policy=build_policy(allow_plaintext=source.allow_plaintext_paths),
                        event="sink_deletion_failed",
                    ),
                )
                policy = build_policy(allow_plaintext=source.allow_plaintext_paths)
                self._quarantine(record.path, "sink_deletion_error", str(exc), source.name, policy)

    def _cleanup_parsed(self, sha256: str) -> None:
        storage_path = self._parsed_dir / f"{sha256}.json"
        if storage_path.exists():
            storage_path.unlink()

    def _quarantine(
        self,
        path: Path,
        reason: str,
        detail: str,
        source_name: str | None = None,
        policy: RedactionPolicy | None = None,
    ) -> None:
        payload = {
            "path": str(path),
            "reason": reason,
            "detail": detail,
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": 0,
            "last_retry_at": None,
            "notes": [],
            "source": source_name,
        }
        redacted = redact_path(path, policy=policy)
        payload["redacted_path"] = redacted.redacted
        payload["path_hash"] = redacted.path_hash
        identifier = uuid.uuid4().hex
        if path.exists():
            try:
                identifier = compute_sha256(path)
            except Exception:  # pragma: no cover - hashing failure
                logger.debug(
                    "Falling back to UUID for quarantine file",
                    extra=_log_extra(path=path, policy=policy, event="quarantine_uuid_fallback"),
                )
        quarantine_file = self._quarantine_dir / f"{identifier}.json"
        quarantine_file.write_text(json.dumps(payload, ensure_ascii=False))
        logger.warning(
            "File written to quarantine",
            extra=_log_extra(
                source=source_name,
                path=path,
                policy=policy,
                event="quarantine",
                reason=reason,
            ),
        )

    def _log_file_event(
        self,
        *,
        job_id: str,
        source: LocalIngestionSource,
        record: FileRecord,
        action: str,
        status: str,
        policy: RedactionPolicy,
        metadata: dict | None = None,
    ) -> None:
        if not self._audit_logger:
            return
        self._audit_logger.record_file_event(
            job_id=job_id,
            source=source.name,
            action=action,
            status=status,
            path=record.path,
            sha256=record.sha256,
            metadata=metadata,
            policy=policy,
        )


